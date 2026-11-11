import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _make_norm(channels: int) -> nn.GroupNorm:
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            _make_norm(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_norm(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class LightConvNormAct(nn.Module):
    """Pointwise bottleneck and multiscale depthwise mixing; not EMCAD weights."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        hidden = max(8, out_ch // 4)
        self.reduce = ConvNormAct(in_ch, hidden, 1, 1, 0)
        self.dw3 = nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False)
        # A dilated 3x3 covers a 5x5 receptive field without a large DW5 kernel.
        self.dw5 = nn.Conv2d(hidden, hidden, 3, stride, 2, dilation=2, groups=hidden, bias=False)
        self.mix_norm = _make_norm(hidden)
        self.expand = nn.Sequential(nn.GELU(), nn.Conv2d(hidden, out_ch, 1, bias=False),
                                    _make_norm(out_ch), nn.GELU())

    def forward(self, x):
        x = self.reduce(x)
        return self.expand(self.mix_norm((self.dw3(x) + self.dw5(x)) * 0.5))


class LightResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = LightConvNormAct(channels, channels)

    def forward(self, x):
        return F.gelu(x + self.block(x))


class SimpleChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class GCN(nn.Module):
    def __init__(self, node_num: int, node_fea: int):
        super().__init__()
        self.conv_adj = nn.Conv1d(node_num, node_num, kernel_size=1, bias=False)
        self.bn_adj = nn.BatchNorm1d(node_num)
        self.conv_wg = nn.Conv1d(node_fea, node_fea, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(node_fea)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.relu(self.bn_adj(self.conv_adj(x)))
        z = z + x
        z = z.transpose(1, 2).contiguous()
        z = self.relu(self.bn_wg(self.conv_wg(z)))
        return z.transpose(1, 2).contiguous()


class ContourPointGCN(nn.Module):
    def __init__(self, in_channels: int, num_points: int = 96, threshold: float = 0.6):
        super().__init__()
        self.num_points = int(num_points)
        self.threshold = float(threshold)
        self.gcn = GCN(self.num_points, in_channels)

    def _get_point_indices(self, edge_prob: torch.Tensor):
        b, _, h, w = edge_prob.shape
        num_points = min(h * w, self.num_points)
        flat = edge_prob.view(b, -1)
        _, point_indices = torch.topk(flat, k=num_points, dim=1)
        if num_points < self.num_points:
            raise ValueError('Feature map has fewer pixels than configured GCN nodes.')
        return point_indices

    def _point_sample(self, feature: torch.Tensor, point_indices: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = feature.shape
        point_indices = point_indices.unsqueeze(1).expand(-1, c, -1)
        return feature.flatten(start_dim=2).gather(dim=2, index=point_indices)

    def forward(self, feature: torch.Tensor, edge_prob: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feature.shape
        if edge_prob.shape[-2:] != (h, w):
            edge_prob = F.interpolate(edge_prob, size=(h, w), mode='bilinear', align_corners=False)
        edge_prob = torch.where(edge_prob >= self.threshold, edge_prob, 0.)
        point_indices = self._get_point_indices(edge_prob)
        sampled = self._point_sample(feature, point_indices)
        gcn_feat = self.gcn(sampled.permute(0, 2, 1)).permute(0, 2, 1)
        confidence = edge_prob.flatten(2).gather(2, point_indices[:, None]).to(feature.dtype)
        gcn_feat = sampled + confidence * (gcn_feat.to(feature.dtype) - sampled)
        scatter_index = point_indices.unsqueeze(1).expand(-1, c, -1)
        return feature.reshape(b, c, h * w).scatter(2, scatter_index, gcn_feat.to(feature.dtype)).view(b, c, h, w)


class BoundaryReflectionRefiner(nn.Module):
    """
    EBLNet 思路：边界显式增强 + 点集 GCN 细化
    TransCues 思路：补一条 reflection-aware 分支，增强透明物体反射/高光 cue
    """
    def __init__(
        self,
        channels: int,
        rgb_hidden: int = 64,
        num_points: int = 96,
        threshold: float = 0.6,
        use_reflection: bool = True,
        use_point_gcn: bool = True,
        residual_scale: float = 1.0,
        variant: str = 'dense',
        checkpoint_blocks: bool = False,
    ):
        super().__init__()
        if variant not in ('dense', 'light'):
            raise ValueError('refiner_variant must be dense or light')
        self.variant = variant
        self.checkpoint_blocks = bool(checkpoint_blocks)
        # Dense defaults preserve the existing module keys and checkpoint shapes.
        self.use_reflection = bool(use_reflection)
        self.use_point_gcn = bool(use_point_gcn)
        self.residual_scale = float(residual_scale)
        if not 0 <= self.residual_scale <= 1:
            raise ValueError('residual_scale must be in [0,1]')
        self.edge_encoder = nn.Sequential(
            ConvNormAct(4, rgb_hidden, 3, 1, 1),
            ConvNormAct(rgb_hidden, rgb_hidden, 3, 1, 1),
        )
        self.reflection_encoder = nn.Sequential(
            ConvNormAct(6, rgb_hidden, 3, 1, 1),
            ConvNormAct(rgb_hidden, rgb_hidden, 3, 1, 1),
        ) if self.use_reflection else None
        self.boundary_prior_proj = nn.Conv2d(1, rgb_hidden, kernel_size=1, bias=False)
        if variant == 'dense':
            self.boundary_fuse = nn.Sequential(
                ConvNormAct(channels * 2 + rgb_hidden * 3, channels, 3, 1, 1),
                ResidualBlock(channels),
            )
            self.boundary_head = nn.Sequential(
                ConvNormAct(channels, channels, 3, 1, 1),
                nn.Conv2d(channels, 1, kernel_size=1),
            )
            self.seg_fuse = nn.Sequential(
                ConvNormAct(channels + rgb_hidden * 2 + 1, channels, 3, 1, 1),
                ResidualBlock(channels),
                SimpleChannelGate(channels),
            )
            self.depth_fuse = nn.Sequential(
                ConvNormAct(channels + rgb_hidden + 1, channels, 3, 1, 1),
                ResidualBlock(channels),
                SimpleChannelGate(channels),
            )
        else:
            # Keep spatial processing at cue width; expand only the final task
            # deltas. Expanding every intermediate block defeats activation savings.
            hidden = max(8, channels // 4)
            self.boundary_fuse = nn.Sequential(
                ConvNormAct(channels * 2 + rgb_hidden * 3, hidden, 1, 1, 0),
                LightResidualBlock(hidden),
            )
            self.boundary_head = nn.Sequential(
                LightConvNormAct(hidden, hidden), nn.Conv2d(hidden, 1, 1),
            )
            self.seg_fuse = nn.Sequential(
                ConvNormAct(channels + rgb_hidden * 2 + 1, hidden, 1, 1, 0),
                LightResidualBlock(hidden), SimpleChannelGate(hidden),
                nn.Conv2d(hidden, channels, 1, bias=False),
            )
            self.depth_fuse = nn.Sequential(
                ConvNormAct(channels + rgb_hidden + 1, hidden, 1, 1, 0),
                LightResidualBlock(hidden), SimpleChannelGate(hidden),
                nn.Conv2d(hidden, channels, 1, bias=False),
            )
        self.seg_point_refine = ContourPointGCN(channels, num_points=num_points, threshold=threshold) if self.use_point_gcn else None
        self.depth_point_refine = ContourPointGCN(channels, num_points=max(1, num_points // 2), threshold=threshold) if self.use_point_gcn else None

    @staticmethod
    def _rgb_edge_map(rgb: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        dy = F.pad(dy, (0, 0, 0, 1))
        mag = torch.sqrt(dx * dx + dy * dy + 1e-6) - 1e-3
        mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return mag

    @staticmethod
    def _reflection_map(rgb: torch.Tensor) -> torch.Tensor:
        blur = F.avg_pool2d(rgb, kernel_size=5, stride=1, padding=2, count_include_pad=False)
        high_pass = torch.abs(rgb - blur)
        spec = high_pass.max(dim=1, keepdim=True)[0]
        spec = spec / (spec.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return spec, high_pass

    def encode_rgb_cues(self, rgb_image: torch.Tensor, target_size):
        """Iteration-invariant cues; callers may reuse within one forward only.

        Encoders use GroupNorm (no running state) and no randomness. Features
        stay attached so all decoder uses contribute to encoder/input gradients.
        Boundary priors and point GCNs are deliberately excluded.
        """
        if rgb_image.shape[-2:] != target_size:
            rgb = F.interpolate(rgb_image, size=target_size, mode='bilinear', align_corners=False)
        else:
            rgb = rgb_image

        edge_map = self._rgb_edge_map(rgb)
        edge_feat = self.edge_encoder(torch.cat([rgb, edge_map], dim=1))
        if self.use_reflection:
            _, high_pass = self._reflection_map(rgb)
            reflection_feat = self.reflection_encoder(torch.cat([rgb, high_pass], dim=1))
        else:
            reflection_feat = torch.zeros_like(edge_feat)
        return edge_feat, reflection_feat

    def forward(
        self,
        seg_feature: torch.Tensor,
        depth_feature: torch.Tensor,
        rgb_image: torch.Tensor,
        boundary_prior: torch.Tensor = None,
        encoded_cues=None,
    ):
        target_size = seg_feature.shape[-2:]
        edge_feat, reflection_feat = (
            self.encode_rgb_cues(rgb_image, target_size)
            if encoded_cues is None else encoded_cues
        )

        if boundary_prior is None:
            boundary_prior_feat = torch.zeros_like(edge_feat)
        else:
            if boundary_prior.shape[-2:] != target_size:
                boundary_prior = F.interpolate(boundary_prior, size=target_size, mode='bilinear', align_corners=False)
            boundary_prior_feat = self.boundary_prior_proj(torch.sigmoid(boundary_prior))

        recompute = self.checkpoint_blocks and self.training and torch.is_grad_enabled()
        def run(module, tensor):
            return checkpoint(module, tensor, use_reentrant=False) if recompute else module(tensor)
        def boundary_predict(tensor):
            return self.boundary_head(self.boundary_fuse(tensor))
        boundary_logit = run(boundary_predict, torch.cat(
            [seg_feature, depth_feature, edge_feat, reflection_feat, boundary_prior_feat], dim=1))
        boundary_prob = torch.sigmoid(boundary_logit)

        seg_delta = run(self.seg_fuse, torch.cat([seg_feature, edge_feat, reflection_feat, boundary_prob], dim=1))
        depth_delta = run(self.depth_fuse, torch.cat([depth_feature, edge_feat, boundary_prob], dim=1))

        seg_feature = seg_feature + self.residual_scale * seg_delta
        depth_feature = depth_feature + self.residual_scale * depth_delta

        if self.use_point_gcn:
            seg_feature = self.seg_point_refine(seg_feature, boundary_prob)
            depth_feature = self.depth_point_refine(depth_feature, boundary_prob)

        return seg_feature, depth_feature, boundary_logit

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class Fusion(nn.Module):
    """
    更稳的双任务融合：
    1) 当前尺度 reassemble feature 作为主干输入
    2) 上一尺度 depth/seg feature 作为自顶向下先验
    3) 上一轮迭代的 depth/seg logits 作为显式语义引导
    4) 可选 boundary prior 进一步约束透明物体边界区域
    """
    def __init__(self, resample_dim, nclasses, use_boundary_guidance=True, use_cross_task=True):
        super().__init__()
        self.use_boundary_guidance = bool(use_boundary_guidance)
        self.use_cross_task = bool(use_cross_task)
        self.stem = ResidualConvUnit(resample_dim)

        self.depth_in = nn.Sequential(
            nn.Conv2d(resample_dim * 2, resample_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(resample_dim),
            nn.GELU(),
            ResidualConvUnit(resample_dim),
        )
        self.seg_in = nn.Sequential(
            nn.Conv2d(resample_dim * 2, resample_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(resample_dim),
            nn.GELU(),
            ResidualConvUnit(resample_dim),
        )

        self.depth_from_logit = nn.Sequential(
            nn.Conv2d(1, resample_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(resample_dim),
            nn.GELU(),
        )
        self.seg_from_logit = nn.Sequential(
            nn.Conv2d(nclasses, resample_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(resample_dim),
            nn.GELU(),
        )
        self.boundary_from_logit = nn.Sequential(
            nn.Conv2d(1, resample_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(resample_dim),
            nn.GELU(),
        )

        if not self.use_boundary_guidance:
            self.boundary_from_logit = None
        self.cross_depth = nn.Sequential(
            nn.Conv2d(resample_dim * 2, resample_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(resample_dim),
            nn.Sigmoid(),
        )
        self.cross_seg = nn.Sequential(
            nn.Conv2d(resample_dim * 2, resample_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(resample_dim),
            nn.Sigmoid(),
        )
        self.depth_out = ResidualConvUnit(resample_dim)
        self.seg_out = ResidualConvUnit(resample_dim)
        if not self.use_cross_task:
            self.cross_depth = self.cross_seg = None

    def forward(self, reassemble, index, previous_depth=None, previous_seg=None,
                out_depths=None, out_segs=None, boundary_prior=None):
        base = self.stem(reassemble)
        if previous_depth is None:
            previous_depth = torch.zeros_like(base)
        if previous_seg is None:
            previous_seg = torch.zeros_like(base)

        depth_feature = self.depth_in(torch.cat([base, previous_depth], dim=1))
        seg_feature = self.seg_in(torch.cat([base, previous_seg], dim=1))

        if out_depths is not None and len(out_depths) != 0 and out_segs is not None and len(out_segs) != 0:
            prev_depth_logit = out_depths[-1][3 - index]
            prev_seg_logit = out_segs[-1][3 - index]
            prev_depth_logit = F.interpolate(prev_depth_logit, size=base.shape[-2:], mode='bilinear', align_corners=False)
            prev_seg_logit = F.interpolate(prev_seg_logit, size=base.shape[-2:], mode='bilinear', align_corners=False)
            # Depth head already returns normalized depth; avoid a second sigmoid.
            depth_feature = depth_feature + self.depth_from_logit(prev_depth_logit)
            seg_feature = seg_feature + self.seg_from_logit(torch.softmax(prev_seg_logit, dim=1))

        if boundary_prior is not None and self.use_boundary_guidance:
            boundary_prior = F.interpolate(boundary_prior, size=base.shape[-2:], mode='bilinear', align_corners=False)
            boundary_feat = self.boundary_from_logit(torch.sigmoid(boundary_prior))
            depth_feature = depth_feature + boundary_feat
            seg_feature = seg_feature + boundary_feat

        if self.use_cross_task:
            # Compute both updates from the same inputs rather than an already updated branch.
            depth_gate = self.cross_depth(torch.cat([depth_feature, seg_feature], dim=1))
            seg_gate = self.cross_seg(torch.cat([seg_feature, depth_feature], dim=1))
            next_depth = self.depth_out(depth_feature + seg_gate * seg_feature)
            next_seg = self.seg_out(seg_feature + depth_gate * depth_feature)
            depth_feature, seg_feature = next_depth, next_seg
        else:
            depth_feature = self.depth_out(depth_feature)
            seg_feature = self.seg_out(seg_feature)

        depth_feature = F.interpolate(depth_feature, scale_factor=2, mode="bilinear", align_corners=False)
        seg_feature = F.interpolate(seg_feature, scale_factor=2, mode="bilinear", align_corners=False)
        return depth_feature, seg_feature

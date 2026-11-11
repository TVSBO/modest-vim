import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )


def _norm(channels: int):
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class PredictionStem(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _norm(mid_channels),
            nn.GELU(),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _norm(mid_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class HeadSeg(nn.Module):
    def __init__(self, features, nclasses=3):
        super().__init__()
        mid = max(features // 2, 64)
        self.boundary_proj = nn.Sequential(
            nn.Conv2d(1, mid, kernel_size=1, bias=False),
            _norm(mid),
            nn.GELU(),
        )
        self.stem = PredictionStem(features, mid)
        self.out = nn.Conv2d(mid, nclasses, kernel_size=1)

    def forward(self, x, boundary_logits=None):
        feat = self.stem(x)
        if boundary_logits is not None:
            boundary_logits = F.interpolate(boundary_logits, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            feat = feat + self.boundary_proj(torch.sigmoid(boundary_logits))
        return self.out(feat)


class HeadDepth(nn.Module):
    def __init__(self, features, use_sigmoid=True):
        super().__init__()
        mid = max(features // 2, 64)
        self.stem = PredictionStem(features, mid)
        self.out = nn.Conv2d(mid, 1, kernel_size=1)
        self.use_sigmoid = bool(use_sigmoid)

    def forward(self, x):
        depth = self.out(self.stem(x))
        return torch.sigmoid(depth) if self.use_sigmoid else depth


class HeadBoundary(nn.Module):
    def __init__(self, features):
        super().__init__()
        mid = max(features // 2, 64)
        self.stem = PredictionStem(features, mid)
        self.out = nn.Conv2d(mid, 1, kernel_size=1)

    def forward(self, x):
        return self.out(self.stem(x))


class MultiscaleHead(nn.Module):
    def __init__(self, channels, nclasses, use_sigmoid_depth=True):
        super().__init__()
        self.head_depth = HeadDepth(channels, use_sigmoid=use_sigmoid_depth)
        self.head_seg = HeadSeg(channels, nclasses=nclasses)

    def forward(self, depth_feature, seg_feature, boundary_logits=None):
        depth = self.head_depth(depth_feature)
        seg = self.head_seg(seg_feature, boundary_logits=boundary_logits)
        return depth, seg

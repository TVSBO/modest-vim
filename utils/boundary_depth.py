"""Optional metric-independent supervision near GT semantic boundaries."""
import torch
import torch.nn.functional as F
from utils.loss import build_boundary_target


def boundary_depth_loss(pred, target, segmentation, valid):
    if pred.ndim == 4:
        pred = pred[:, 0]
    valid = valid.bool() & torch.isfinite(target)
    mask = build_boundary_target(segmentation, dilation=1)[:, 0].bool() & valid
    # Invalid targets must not contaminate multiplication through NaN*0.
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    error = F.smooth_l1_loss(pred.float(), safe_target.float(), reduction='none')
    counts = mask.sum((1,2))
    active = counts > 0
    per_image = torch.where(mask, error, 0.).sum((1,2)) / counts.clamp_min(1)
    return per_image[active].mean() if active.any() else pred.sum() * 0.

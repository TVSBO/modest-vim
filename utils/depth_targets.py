"""Training-only depth target resampling with explicit validity support."""
import torch
import torch.nn.functional as F


def resize_valid_depth_target(depth, valid, loss_mask, size):
    """Normalize bilinear targets over valid contributors only.

    Keep nearest-neighbor center eligibility, and reject centers with no
    bilinear support. Never expand the task mask or alter full-scale targets.
    Invalid values are removed with where (rather than multiplication), so
    even NaN sentinels outside validity cannot enter coarse supervision.
    """
    valid = valid.bool()
    if tuple(depth.shape[-2:]) == tuple(size):
        return depth, valid, None if loss_mask is None else loss_mask.bool() & valid
    weights = F.interpolate(valid[:, None].float(), size=size, mode='bilinear', align_corners=False)[:, 0]
    numerator = F.interpolate(torch.where(valid, depth, torch.zeros_like(depth))[:, None],
                              size=size, mode='bilinear', align_corners=False)[:, 0]
    target = numerator / weights.clamp_min(torch.finfo(weights.dtype).eps)
    center_valid = F.interpolate(valid[:, None].float(), size=size, mode='nearest')[:, 0].bool()
    eligible = center_valid & (weights > 0)
    main = None if loss_mask is None else F.interpolate(loss_mask[:, None].float(), size=size, mode='nearest')[:, 0].bool() & eligible
    return target, eligible, main

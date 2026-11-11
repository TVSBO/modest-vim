"""Validity support of depth derivative operators; no model parameters."""
import torch
import torch.nn.functional as F


def depth_auxiliary_masks(valid, loss_mask, operator):
    """Return support-valid zero/main masks, preserving existing image padding.

    Sobel uses all in-image 3x3 pixels. Normals use center/right/bottom;
    the final row/column derivatives are explicitly zero in gradient().
    A task-specific loss mask restricts centers, not valid scene neighbors.
    """
    valid = valid.bool()
    if operator == 'sobel':
        padded = F.pad(valid[:, None].float(), (1, 1, 1, 1), mode='replicate')
        support = F.avg_pool2d(padded, 3, 1)[:, 0] == 1
    elif operator == 'normal':
        support = valid.clone()
        support[:, :, :-1] &= valid[:, :, 1:]
        support[:, :-1, :] &= valid[:, 1:, :]
    else:
        raise ValueError(f'Unknown depth derivative operator: {operator}')
    main = support if loss_mask is None else support & loss_mask.bool()
    return support, main

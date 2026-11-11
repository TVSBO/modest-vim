import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F


def depth_to_xyz(depthImage, dataset_name):
    if dataset_name in ('syntodd','syntodd_raw'):
        fx = 613.9624633789062
        fy = 613.9624633789062
        du = 320.0
        dv = 240.0
    elif dataset_name == 'clearpose':
        fx = 601.46000163
        fy = 601.5933431
        du = 334.89998372
        dv = 248.15334066
    elif dataset_name == 'my_new_dataset':
        fx = 601.46000163
        fy = 601.5933431
        du = 334.89998372
        dv = 248.15334066
    else:
        raise ValueError(f"Dataset name '{dataset_name}' not correct ! Please add camera intrinsics in utils/loss.py")

    B, C, H, W = depthImage.shape
    # Default calibration is for 640x480; resize the intrinsics with each output scale.
    fx *= W / 640.0
    fy *= H / 480.0
    du = (du + .5) * W / 640.0 - .5
    dv = (dv + .5) * H / 480.0 - .5
    device = depthImage.device

    xyz = torch.zeros([B, H, W, 3], device=device)
    imageIndexX = torch.arange(0, W, 1, device=device) - du
    imageIndexY = torch.arange(0, H, 1, device=device) - dv
    depthImage = depthImage.squeeze()
    if B == 1:
        depthImage = depthImage.unsqueeze(0)

    xyz[:, :, :, 0] = depthImage / fx * imageIndexX
    xyz[:, :, :, 1] = (depthImage.transpose(1, 2) / fy * imageIndexY).transpose(1, 2)
    xyz[:, :, :, 2] = depthImage
    xyz = xyz.permute(0, 3, 1, 2).to(device)
    return xyz


def gradient(x):
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = right - left, bottom - top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


def get_surface_normal(x, dataset_name):
    xyz = depth_to_xyz(x, dataset_name)
    dx, dy = gradient(xyz)
    surface_normal = torch.cross(dx, dy, dim=1)
    surface_normal = surface_normal / (torch.norm(surface_normal, dim=1, keepdim=True) + 1e-8)
    return surface_normal, dx, dy


def visualize_surface_normals(normals):
    normals = (normals + 1) / 2
    normal_image = normals[0].permute(1, 2, 0).cpu().numpy()
    plt.imshow(normal_image)
    plt.title("Surface Normals")
    plt.axis('off')
    plt.show()


def _safe_masked_mean(loss_tensor, mask=None):
    if mask is None:
        return loss_tensor.mean()
    mask = mask.bool()
    if mask.sum() == 0:
        return loss_tensor.mean() * 0.0
    return loss_tensor[mask].mean()


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        edge_ky = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k, requires_grad=False)

    def forward(self, x):
        if self.edge_conv.weight.device != x.device:
            self.edge_conv = self.edge_conv.to(x.device)
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        return out


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = Sobel()

    def forward(self, input, target, zero_mask=None, loss_mask=None):
        input_grad = self.sobel(input)
        target_grad = self.sobel(target)

        input_grad_dx = input_grad[:, 0, :, :].contiguous().view_as(input)
        input_grad_dy = input_grad[:, 1, :, :].contiguous().view_as(input)
        target_grad_dx = target_grad[:, 0, :, :].contiguous().view_as(target)
        target_grad_dy = target_grad[:, 1, :, :].contiguous().view_as(target)

        loss_dx = torch.abs(input_grad_dx - target_grad_dx)
        loss_dy = torch.abs(input_grad_dy - target_grad_dy)

        if zero_mask is None:
            zero_mask = torch.ones_like(loss_dx[:, 0], dtype=torch.bool, device=loss_dx.device)
        if loss_mask is None:
            main_mask = zero_mask
        else:
            main_mask = loss_mask.bool()

        main_mask = main_mask.unsqueeze(1).expand_as(loss_dx)
        zero_mask = zero_mask.bool().unsqueeze(1).expand_as(loss_dx)

        main_loss = _safe_masked_mean(loss_dx, main_mask) + _safe_masked_mean(loss_dy, main_mask)
        aux_loss = _safe_masked_mean(loss_dx, zero_mask) + _safe_masked_mean(loss_dy, zero_mask)
        return main_loss + 0.01 * aux_loss


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, normal_pred, normal_gt, zero_mask=None, loss_mask=None):
        loss = self.loss(normal_pred, normal_gt)
        if zero_mask is None:
            zero_mask = torch.ones_like(loss[:, 0], dtype=torch.bool, device=loss.device)
        if loss_mask is None:
            main_mask = zero_mask
        else:
            main_mask = loss_mask.bool()

        main_mask = main_mask.unsqueeze(1).expand_as(normal_gt)
        zero_mask = zero_mask.bool().unsqueeze(1).expand_as(normal_gt)

        return _safe_masked_mean(loss, main_mask) + 0.01 * _safe_masked_mean(loss, zero_mask)


class SegIouLoss(nn.Module):
    def __init__(self, num_classes):
        super(SegIouLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, seg_pred, seg_gt):
        seg_pred_mask = F.softmax(seg_pred, dim=1)
        assert seg_pred_mask.shape[2] == seg_gt.shape[1] and seg_pred_mask.shape[3] == seg_gt.shape[2]
        IoUs = []
        for cls in range(self.num_classes):
            pred_cls = seg_pred_mask[:, cls, :, :]
            target_cls = (seg_gt == cls).float()

            intersection = (pred_cls * target_cls).sum(dim=(1, 2))
            union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2)) - intersection

            iou = (intersection + 1e-6) / (union + 1e-6)
            IoUs.append(iou)

        mean_iou = torch.mean(torch.stack(IoUs, dim=0), dim=0)
        loss = 1 - mean_iou.mean()
        return loss


class MaskedL2Loss(nn.Module):
    def __init__(self):
        super(MaskedL2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, depth_pred, depth_gt, zero_mask=None, loss_mask=None):
        loss = self.loss(depth_pred, depth_gt)
        if zero_mask is None:
            zero_mask = torch.ones_like(loss, dtype=torch.bool, device=loss.device)
        if loss_mask is None:
            main_mask = zero_mask
        else:
            main_mask = loss_mask.bool()

        return _safe_masked_mean(loss, main_mask) + 0.01 * _safe_masked_mean(loss, zero_mask.bool())


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, depth_pred, depth_gt, zero_mask=None, loss_mask=None):
        loss = self.loss(depth_pred, depth_gt)
        if zero_mask is None:
            zero_mask = torch.ones_like(loss, dtype=torch.bool, device=loss.device)
        if loss_mask is None:
            main_mask = zero_mask
        else:
            main_mask = loss_mask.bool()

        return _safe_masked_mean(loss, main_mask) + 0.01 * _safe_masked_mean(loss, zero_mask.bool())


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self):
        super(MaskedSmoothL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, depth_pred, depth_gt, zero_mask=None, loss_mask=None):
        loss = self.loss(depth_pred, depth_gt)
        if zero_mask is None:
            zero_mask = torch.ones_like(loss, dtype=torch.bool, device=loss.device)
        if loss_mask is None:
            main_mask = zero_mask
        else:
            main_mask = loss_mask.bool()

        return _safe_masked_mean(loss, main_mask) + 0.01 * _safe_masked_mean(loss, zero_mask.bool())


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, net_output, gt):
        net_output = F.softmax(net_output, dim=1)
        with torch.no_grad():
            y_onehot = torch.zeros_like(net_output)
            y_onehot.scatter_(1, gt.unsqueeze(1), 1)

        axes = tuple(range(2, net_output.ndim))
        intersect = (net_output * y_onehot).sum(dim=axes)
        sum_pred = net_output.sum(dim=axes)
        sum_gt = y_onehot.sum(dim=axes)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))
        return 1 - dc.mean()


class DC_and_CE_loss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, net_output, target):
        loss_ce = self.ce(net_output, target)
        loss_dice = self.dice(net_output, target)
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice


from .cbdice_loss import SoftcbDiceLoss


class TransparentSOTALoss(nn.Module):
    def __init__(self, ce_weight=1.0, cbdice_weight=1.0):
        super(TransparentSOTALoss, self).__init__()
        self.ce_weight = ce_weight
        self.cbdice_weight = cbdice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.cbdice_loss = SoftcbDiceLoss(iter_=10, smooth=1e-5)

    def forward(self, pred, target):
        loss_ce = self.ce_loss(pred, target.long())
        target_expanded = target.unsqueeze(1).float()
        loss_cbdice = self.cbdice_loss(pred, target_expanded)
        total_loss = (self.ce_weight * loss_ce) + (self.cbdice_weight * loss_cbdice)
        return total_loss


class TransparentHybridLoss(nn.Module):
    """推荐版本：用 CE+Dice 稳住主体，再用小权重 cbDice 补边界/拓扑。"""

    def __init__(self, ce_weight=1.0, dice_weight=1.0, cbdice_weight=0.05):
        super(TransparentHybridLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cbdice_weight = cbdice_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = SoftDiceLoss()
        self.cbdice_loss = SoftcbDiceLoss(iter_=10, smooth=1e-5)

    def forward(self, pred, target):
        target = target.long()
        target_expanded = target.unsqueeze(1).float()

        loss_ce = self.ce_loss(pred, target)
        loss_dice = self.dice_loss(pred, target)
        loss_cbdice = self.cbdice_loss(pred, target_expanded)

        total_loss = (
            self.ce_weight * loss_ce
            + self.dice_weight * loss_dice
            + self.cbdice_weight * loss_cbdice
        )
        return total_loss


def build_boundary_target(seg_gt: torch.Tensor, dilation: int = 1) -> torch.Tensor:
    """
    从分割标签构造 1 通道边界监督图。
    只要邻域标签发生变化，就判为边界像素；随后做轻微膨胀，提升监督稳定性。
    """
    if seg_gt.dim() == 4:
        seg_gt = seg_gt.squeeze(1)
    seg_gt = seg_gt.long()
    b, h, w = seg_gt.shape
    boundary = torch.zeros((b, 1, h, w), device=seg_gt.device, dtype=torch.float32)

    boundary[:, :, :, 1:] = torch.maximum(boundary[:, :, :, 1:], (seg_gt[:, :, 1:] != seg_gt[:, :, :-1]).unsqueeze(1).float())
    boundary[:, :, :, :-1] = torch.maximum(boundary[:, :, :, :-1], (seg_gt[:, :, 1:] != seg_gt[:, :, :-1]).unsqueeze(1).float())
    boundary[:, :, 1:, :] = torch.maximum(boundary[:, :, 1:, :], (seg_gt[:, 1:, :] != seg_gt[:, :-1, :]).unsqueeze(1).float())
    boundary[:, :, :-1, :] = torch.maximum(boundary[:, :, :-1, :], (seg_gt[:, 1:, :] != seg_gt[:, :-1, :]).unsqueeze(1).float())

    if dilation > 0:
        kernel = 2 * dilation + 1
        boundary = F.max_pool2d(boundary, kernel_size=kernel, stride=1, padding=dilation)
    return boundary.clamp_(0.0, 1.0)


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        target = target.float()
        dims = (1, 2, 3)
        intersection = (prob * target).sum(dim=dims)
        denom = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BoundaryBCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 3.0, dice_weight: float = 1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.dice_weight = float(dice_weight)
        self.dice = BinaryDiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        pos_weight = torch.tensor([self.pos_weight], dtype=pred.dtype, device=pred.device)
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), pos_weight=pos_weight)
        dice = self.dice(pred, target.float())
        return bce + self.dice_weight * dice


class BoundaryAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = Sobel()

    def forward(self, seg_logits: torch.Tensor, boundary_logits: torch.Tensor) -> torch.Tensor:
        if seg_logits.size(1) == 1:
            object_prob = torch.sigmoid(seg_logits)
        else:
            object_prob = torch.softmax(seg_logits, dim=1)[:, -1:, :, :]
        seg_grad = self.sobel(object_prob)
        seg_mag = torch.sqrt(seg_grad[:, 0:1] ** 2 + seg_grad[:, 1:2] ** 2 + 1e-6)
        seg_mag = seg_mag / (seg_mag.amax(dim=(2, 3), keepdim=True) + 1e-6)
        boundary_prob = torch.sigmoid(boundary_logits)
        return F.l1_loss(boundary_prob, seg_mag)

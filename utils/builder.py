import os
import errno
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from utils.loss import *
from datasets.clearpose import MyNewDataset


def multiview_collate(batch):
    targets = []
    for ii in range(len(batch[0])):
        targets.append([batch_element[ii] for batch_element in batch])

    stacked_images = torch.stack(targets[0])
    stacked_camera_poses = torch.stack(targets[1])
    stacked_intrinsics = torch.stack(targets[2])

    return stacked_images, stacked_camera_poses, targets[2], targets[3], targets[4], targets[5], targets[6], targets[9]


def get_dataloader(config, mode):
    dataset_name = config["Dataset"]["dataset_name"]
    dataset_config = config["Dataset"][dataset_name]

    if dataset_name != "my_new_dataset":
        raise ValueError("This release uses the ClearPose image-list loader.")
    dataset = MyNewDataset(config, mode)

    batch_size = dataset_config["batch_size"]
    num_workers = dataset_config["num_workers"]

    if mode == 'train':
        shuffle = True
        drop_last = True
    else:
        shuffle_key = f"{mode}_shuffle"
        shuffle = dataset_config[shuffle_key] if shuffle_key in dataset_config else False
        drop_last = False

    from torch.utils.data import DataLoader
    from utils.reproducibility import seed_worker
    generator = torch.Generator().manual_seed(int(config['Trainer'].get('seed', 0)))
    worker_options = {}
    if num_workers > 0:
        prefetch = int(dataset_config.get('prefetch_factor', 2))
        if prefetch < 1:
            raise ValueError('prefetch_factor must be positive')
        worker_options['prefetch_factor'] = prefetch
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(dataset_config.get('pin_memory', True)),
        shuffle=shuffle,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
        **worker_options,
    )


def get_losses(config):
    loss_depth = None
    loss_segmentation = None
    loss_depth_grad = None
    loss_depth_normal = None
    loss_seg_iou = None

    loss_depth_type = config['Trainer']['loss_depth']
    loss_seg_type = config['Trainer']['loss_segmentation']
    model_type = config['Model']['type']
    dataset_name = config["Dataset"]["dataset_name"]

    if 'classes' in config['Dataset'][dataset_name]:
        num_classes = len(config['Dataset'][dataset_name]['classes'])
    else:
        num_classes = 2

    if 'depth' in model_type or 'full' in model_type:
        if 'mse' in loss_depth_type:
            loss_depth = MaskedL2Loss()
        elif 'l1' in loss_depth_type:
            loss_depth = MaskedL1Loss()
        elif 'smooth' in loss_depth_type:
            loss_depth = MaskedSmoothL1Loss()
        else:
            loss_depth = MaskedL2Loss()

        if 'grad' in loss_depth_type:
            loss_depth_grad = GradientLoss()
        if 'normal' in loss_depth_type:
            loss_depth_normal = NormalLoss()

    if 'seg' in model_type or 'full' in model_type:
        if 'ce' in loss_seg_type:
            loss_segmentation = DC_and_CE_loss()
            print(">>> 已启用基础分割损失 (Dice + CrossEntropy) <<<")
        if 'iou' in loss_seg_type:
            loss_seg_iou = SegIouLoss(num_classes)

    return loss_depth, loss_depth_grad, loss_depth_normal, loss_segmentation, loss_seg_iou


def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_optimizer(config, net):
    real_net = net
    params_backbone = list(real_net.transformer_encoders.parameters())
    backbone_ids = {id(p) for p in params_backbone}
    params_scratch = [p for p in real_net.parameters() if id(p) not in backbone_ids]
    weight_decay = float(config['Trainer'].get('weight_decay', 3e-5))

    if config['Trainer']['optim'] == 'adam':
        optimizer_backbone = optim.AdamW(
            params_backbone,
            lr=config['Trainer']['lr_backbone'],
            weight_decay=weight_decay
        )
        optimizer_scratch = optim.AdamW(
            params_scratch,
            lr=config['Trainer']['lr_scratch'],
            weight_decay=weight_decay
        )
        print(f">>> 已启用 AdamW 优化器，Weight Decay = {weight_decay} <<<")
    elif config['Trainer']['optim'] == 'sgd':
        optimizer_backbone = optim.SGD(
            params_backbone,
            lr=config['Trainer']['lr_backbone'],
            momentum=config['Trainer']['momentum'],
            weight_decay=weight_decay,
            nesterov=True
        )
        optimizer_scratch = optim.SGD(
            params_scratch,
            lr=config['Trainer']['lr_scratch'],
            momentum=config['Trainer']['momentum'],
            weight_decay=weight_decay,
            nesterov=True
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer: {config['Trainer']['optim']}")

    return optimizer_backbone, optimizer_scratch


class PolyLRSchedulerAdapter(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, last_epoch=-1)

    def step(self, val_loss_or_step=None):
        current_step = self.ctr
        self.ctr += 1

        if current_step < self.max_steps:
            new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        else:
            new_lr = self.initial_lr * 1e-3

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


def get_schedulers(config, optimizers):
    max_steps = int(config['Trainer']['epochs'])
    print(f">>> 已启用 PolyLR 平滑学习率调度器 | max_steps={max_steps} <<<")
    return [PolyLRSchedulerAdapter(opt, opt.param_groups[0]['lr'], max_steps=max_steps) for opt in optimizers]

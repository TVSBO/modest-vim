import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms  # <--- [新增] 导入 transforms 库

try:
    from torchvision.transforms import InterpolationMode
except ImportError:
    InterpolationMode = None

import random

class MyNewDataset(Dataset):
    def __init__(self, config, mode='train'):
        super(MyNewDataset, self).__init__()
        self.mode = mode

        # 1. 从配置中获取参数
        dataset_config = config['Dataset'][config['Dataset']['dataset_name']]
        self.root = dataset_config['path']
        self.im_resize = dataset_config['transforms']['im_resize']
        self.depth_min = dataset_config['depth_min']
        self.depth_max = dataset_config['depth_max']

        # 获取类别数量，用于修正标签
        if 'classes' in dataset_config:
            self.num_classes = len(dataset_config['classes'])
        else:
            self.num_classes = 2 # 默认值

        # 2. 读取之前生成的 txt 分割文件
        split_file = dataset_config['splits'][f'{mode}_list']
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines() if line.strip()]

        print(f"[{mode.upper()}] Loaded {len(self.file_list)} samples from {split_file}")

        # --- [新增] 3. 定义颜色抖动增强器 (仅在训练模式下使用) ---
        if self.mode == 'train':
            # brightness(亮度), contrast(对比度), saturation(饱和度), hue(色相)
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        # --------------------------------------------------------

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # 1. 拼接完整路径
        rel_path = self.file_list[index]
        rgb_path = os.path.join(self.root, rel_path)

        depth_path = rgb_path.replace("-color.png", "-depth_true.png")
        seg_path = rgb_path.replace("-color.png", "-label.png")

        # 2. 加载图像
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path)
        seg = Image.open(seg_path)

        return self._make_sample(rgb, depth, seg, rel_path)

    def _make_sample(self, rgb, depth, seg, sample_id):

        # 3. 预处理 (Resize, 翻转 和 颜色增强)
        rgb, depth, seg = self._transform(rgb, depth, seg)

        # 4. 转换为 Tensor 并处理数据类型

        # --- RGB 处理 ---
        rgb = TF.to_tensor(rgb)
        rgb = TF.normalize(rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # --- Depth 处理 ---
        depth = np.array(depth).astype(np.float32)
        depth = depth / 1000.0 # mm 转 m

        valid_mask = (np.isfinite(depth) & (depth > 0) & (depth >= self.depth_min) & (depth <= self.depth_max)).astype(np.float32)
        valid_mask = torch.from_numpy(valid_mask).float()

        depth = np.clip(depth, self.depth_min, self.depth_max)
        depth = np.nan_to_num(depth, nan=self.depth_min, posinf=self.depth_max, neginf=self.depth_min)
        depth = (depth - self.depth_min) / (self.depth_max - self.depth_min)
        depth = torch.from_numpy(depth).float()

        # --- Segmentation 处理 (关键修复位置) ---
        seg_np = np.array(seg)

        # 修复逻辑：
        # 如果是二分类 (0:背景, 1:物体)，将所有大于0的ID都视为1
        if self.num_classes == 2:
            seg_np[seg_np > 0] = 1
        else:
            # 如果是多分类，将超过范围的标签设为0 (背景)，防止越界报错
            # 注意：如果你的数据中有 255 (ignore)，通常也需要在这里处理，比如设为 0 或者 -100
            seg_np[seg_np >= self.num_classes] = 0

        seg = torch.from_numpy(seg_np).long()

        # 5. 返回字典
        sample = {
            'rgb': rgb,
            'depth_gt': depth,
            'seg_gt': seg,
            'zero_mask': valid_mask,
            'loss_mask': valid_mask,
            'depth_min': torch.tensor([self.depth_min]),
            'depth_max': torch.tensor([self.depth_max])
        }
        sample['sample_id'] = sample_id
        return sample

    def _transform(self, rgb, depth, seg):
        target_size = [self.im_resize, self.im_resize]

        if InterpolationMode is not None:
            interp_rgb = InterpolationMode.BILINEAR
            interp_nearest = InterpolationMode.NEAREST
        else:
            interp_rgb = Image.BILINEAR
            interp_nearest = Image.NEAREST

        rgb = TF.resize(rgb, size=target_size, interpolation=interp_rgb)
        depth = TF.resize(depth, size=target_size, interpolation=interp_nearest)
        seg = TF.resize(seg, size=target_size, interpolation=interp_nearest)

        if self.mode == 'train':
            # --- [新增] 数据增强逻辑 ---

            # 1. 50% 的概率水平翻转 (保证 rgb, depth, seg 同步翻转)
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
                seg = TF.hflip(seg)

            # 2. 80% 的概率进行颜色抖动 (仅作用于 RGB！绝不能动 depth 和 seg)
            # 留 20% 原图的概率，有助于稳定模型
            if random.random() > 0.2:
                rgb = self.color_jitter(rgb)
            # ---------------------------

        return rgb, depth, seg

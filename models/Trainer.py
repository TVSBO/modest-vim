import os
import math
import json
import time
import random
import numpy as np

import torch
try:
    import wandb
except ImportError:
    wandb = None
import torch.nn.functional as F

from tqdm import tqdm
from models.Model import ISGNet
from utils.builder import get_losses, get_optimizer, get_schedulers, create_dir
from utils.loss import (
    get_surface_normal,
    TransparentHybridLoss,
    DC_and_CE_loss,
    BoundaryBCEDiceLoss,
    BoundaryAlignmentLoss,
    build_boundary_target,
)
from utils.visualize import vis_rgb
from utils.protocol_metrics import EvaluationAccumulator
from utils.boundary_depth import boundary_depth_loss
from utils.progress import TrainingProgress
from utils.task_weighting import TaskWeighting
from utils.depth_support import depth_auxiliary_masks
from utils.depth_targets import resize_valid_depth_target


class Trainer(object):
    def __init__(self, config, model_factory=None):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f">>> Trainer 初始化在单卡: {self.device} <<<")

        self.type = config['Model']['type']
        self.dataset_name = config['Dataset']['dataset_name']
        self.num_classes = len(config['Dataset'][self.dataset_name].get('classes', {'0': {}, '1': {}}))

        trainer_cfg = config['Trainer']
        model_cfg = config['Model']
        self.task_balancer = TaskWeighting(
            trainer_cfg.get('task_weighting', 'fixed'),
            float(trainer_cfg.get('dwa_temperature', 2.0)),
            float(trainer_cfg.get('dwa_min_weight', 0.5)),
        )
        self.seg_multi = float(trainer_cfg.get('seg_multi', 1.0))
        self.depth_multi = float(trainer_cfg.get('depth_multi', 1.0))
        self.boundary_multi = float(trainer_cfg.get('boundary_multi', 0.0))
        self.boundary_align_multi = float(trainer_cfg.get('boundary_align_multi', 0.0))
        self.boundary_depth_multi = float(trainer_cfg.get('boundary_depth_multi', 0.0))
        self.depth_scale_multi = float(trainer_cfg.get('depth_scale_multi', 1.0))
        self.depth_grad_multi = float(trainer_cfg.get('depth_grad_multi', 1.0))
        self.depth_normal_multi = float(trainer_cfg.get('depth_normal_multi', 1.0))
        self.depth_aux_valid_support = bool(trainer_cfg.get('depth_aux_valid_support', False))
        self.depth_valid_target_resize = bool(trainer_cfg.get('depth_valid_target_resize', False))
        self.depth_error_multi = float(trainer_cfg.get('depth_error_multi', 1.0))
        self.gamma = float(trainer_cfg.get('gamma', 0.33))
        input_size = config['Dataset'][self.dataset_name]['transforms']['im_resize']
        self.resolutions = [input_size // 8, input_size // 4, input_size // 2, input_size]

        self.use_amp = bool(trainer_cfg.get('use_amp', True)) and self.device.type == 'cuda'
        amp_name = trainer_cfg.get('amp_dtype', 'bfloat16')
        if amp_name not in ('bfloat16', 'float16'):
            raise ValueError('amp_dtype must be bfloat16 or float16')
        self.amp_dtype = torch.bfloat16 if amp_name == 'bfloat16' else torch.float16
        self.max_grad_norm = float(trainer_cfg.get('max_grad_norm', 1.0))
        self.skip_nonfinite_batch = bool(trainer_cfg.get('skip_nonfinite_batch', True))
        self.freeze_backbone_epochs = max(0, int(trainer_cfg.get('freeze_backbone_epochs', 0)))
        self.freeze_bn_stats = bool(trainer_cfg.get('freeze_bn_stats', False))

        self.eval_every = int(trainer_cfg.get('eval_every', 1))
        self.eval_on_last_epoch = bool(trainer_cfg.get('eval_on_last_epoch', True))
        self.save_best = bool(trainer_cfg.get('save_best', True))
        self.save_last = bool(trainer_cfg.get('save_last', True))
        self.save_periodic = bool(trainer_cfg.get('save_periodic', True))
        self.save_periodic_every = int(trainer_cfg.get('save_periodic_every', 2))

        self.train_bar_update_interval = max(1, int(trainer_cfg.get('train_bar_update_interval', 20)))
        self.val_bar_update_interval = max(1, int(trainer_cfg.get('val_bar_update_interval', 5)))
        self.disable_tqdm = bool(trainer_cfg.get('disable_tqdm', False))
        self.bar_leave = bool(trainer_cfg.get('bar_leave', False))

        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.start_epoch = 0
        self.current_epoch_id = 0
        self.resume_rng = None

        im_resize = config['Dataset'][self.dataset_name]['transforms']['im_resize']
        self.model = (model_factory or ISGNet)(
            image_size=(3, im_resize, im_resize),
            emb_dim=model_cfg['emb_dim'],
            resample_dim=model_cfg['resample_dim'],
            read=model_cfg['read'],
            nclasses=self.num_classes,
            hooks=model_cfg['hooks'],
            model_timm=model_cfg['model_timm'],
            type=self.type,
            patch_size=model_cfg['patch_size'],
            pretrain=model_cfg['pretrain'],
            iterations=model_cfg.get('iterations', 2),
            num_layers_encoder=model_cfg.get('num_layers_encoder', 24),
            bimamba_type=model_cfg.get('bimamba_type', 'v2'),
            use_reflection=model_cfg.get('use_reflection', True),
            use_point_gcn=model_cfg.get('use_point_gcn', True),
            use_boundary_guidance=model_cfg.get('use_boundary_guidance', True),
            use_cross_task=model_cfg.get('use_cross_task', True),
            rgb_mean=model_cfg.get('rgb_mean', [0.485,0.456,0.406] if self.dataset_name=='my_new_dataset' else [0.5]*3),
            rgb_std=model_cfg.get('rgb_std', [0.229,0.224,0.225] if self.dataset_name=='my_new_dataset' else [0.5]*3),
            refine_residual_scale=model_cfg.get('refine_residual_scale', 1.),
            refiner_variant=model_cfg.get('refiner_variant', 'dense'),
            refiner_checkpoint=model_cfg.get('refiner_checkpoint', False),
            reuse_invariant_features=model_cfg.get('reuse_invariant_features', False),
            in_chans=config['Dataset'][self.dataset_name]['in_chans'],
            use_boundary_head=model_cfg.get('use_boundary_head', True),
            refine_levels=tuple(model_cfg.get('refine_levels', [0, 1])),
            refine_hidden_dim=model_cfg.get('refine_hidden_dim', 64),
            boundary_gcn_points=model_cfg.get('boundary_gcn_points', 96),
            boundary_gcn_threshold=model_cfg.get('boundary_gcn_threshold', 0.6),
            final_depth_use_sigmoid=model_cfg.get('final_depth_use_sigmoid', True),
        ).to(self.device)

        self.loss_depth, self.loss_depth_grad, self.loss_depth_normal, _, self.loss_seg_iou = get_losses(config)
        seg_loss_cfg = trainer_cfg.get('seg_loss', {})
        self.seg_ce_weight = float(seg_loss_cfg.get('ce_weight', 1.0))
        self.seg_dice_weight = float(seg_loss_cfg.get('dice_weight', 1.0))
        self.seg_cbdice_weight = float(seg_loss_cfg.get('cbdice_weight', 0.05))
        self.seg_use_cbdice = bool(seg_loss_cfg.get('use_cbdice', True))
        self.seg_highres_only = bool(seg_loss_cfg.get('highres_only', True))
        self.seg_cbdice_start_epoch = max(1, int(seg_loss_cfg.get('cbdice_start_epoch', 8)))
        self.seg_cbdice_warmup_epochs = max(0, int(seg_loss_cfg.get('cbdice_warmup_epochs', 4)))
        self.val_use_cbdice = bool(seg_loss_cfg.get('val_use_cbdice', False))
        self.fp32_topology_loss = bool(seg_loss_cfg.get('fp32_topology_loss', True))

        self.loss_seg_base = DC_and_CE_loss(
            weight_ce=self.seg_ce_weight,
            weight_dice=self.seg_dice_weight,
        ).to(self.device)
        self.loss_seg_topo = TransparentHybridLoss(
            ce_weight=self.seg_ce_weight,
            dice_weight=self.seg_dice_weight,
            cbdice_weight=self.seg_cbdice_weight,
        ).to(self.device)
        self.loss_boundary = BoundaryBCEDiceLoss(
            pos_weight=float(trainer_cfg.get('boundary_pos_weight', 3.0)),
            dice_weight=float(trainer_cfg.get('boundary_dice_weight', 1.0)),
        ).to(self.device)
        self.loss_boundary_align = BoundaryAlignmentLoss().to(self.device)

        self.optimizer_backbone, self.optimizer_scratch = get_optimizer(config, self.model)
        self.schedulers = get_schedulers(config, [self.optimizer_backbone, self.optimizer_scratch])

        self.ckpt_path = trainer_cfg['ckpt_path']
        self.path_model_root = model_cfg['path_model']
        self.path_model = os.path.join(self.path_model_root, 'ISGNet')
        create_dir(self.path_model_root)

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        if trainer_cfg.get('resume_path'):
            self._load_training_resume(trainer_cfg['resume_path'])
        else:
            self._load_weights(self.ckpt_path)

        print(
            f">>> 训练配置 | seg={self.seg_multi:.3f} | depth={self.depth_multi:.3f} | "
            f"boundary={self.boundary_multi:.3f} | boundary_align={self.boundary_align_multi:.3f} | "
            f"freeze_backbone_epochs={self.freeze_backbone_epochs} <<<"
        )

    def _load_weights(self, ckpt_path):
        if not ckpt_path:
            if self.config['Model'].get('pretrain', False):
                raise ValueError('Pretrained initialization requested but no checkpoint supplied')
            print('Explicit random initialization (pretrain=False).')
            return
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        print(f">>> 🚀 正在加载权重: {ckpt_path} <<<")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self._load_full_checkpoint(checkpoint['model_state_dict'])
            return
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint and self._looks_like_full_model_state_dict(checkpoint['state_dict']):
            self._load_full_checkpoint(checkpoint['state_dict'])
            return
        if isinstance(checkpoint, dict) and 'model' in checkpoint and self._looks_like_full_model_state_dict(checkpoint['model']):
            self._load_full_checkpoint(checkpoint['model'])
            return
        self._load_pretrained_backbone(checkpoint)

    def _looks_like_full_model_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            return False
        clean_keys = [k.replace('module.', '') for k in state_dict.keys()]
        prefixes = (
            'transformer_encoders.', 'reassembles.', 'fusions.', 'stage_refiners.',
            'boundary_heads.', 'head_depth', 'head_seg', 'head_multiscale',
        )
        return any(k.startswith(prefixes) for k in clean_keys)

    def _load_training_resume(self, path):
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if checkpoint.get('resume_version')!=1:
            raise ValueError('This checkpoint lacks epoch-resume state; use it only for initialization')
        previous=checkpoint['config']
        for section in ('Model','Dataset','Trainer'):
            left=dict(previous[section]);right=dict(self.config[section])
            for key in ('path_model','resume_path','ckpt_path'):
                left.pop(key,None);right.pop(key,None)
            if left!=right:
                raise ValueError(f'Resume {section} differs from the saved experiment protocol')
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.optimizer_backbone.load_state_dict(checkpoint['optimizer_backbone_state_dict'])
        self.optimizer_scratch.load_state_dict(checkpoint['optimizer_scratch_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        for scheduler,state in zip(self.schedulers,checkpoint['scheduler_states']):
            scheduler.load_state_dict(state)
            if checkpoint['scheduler_step_pending']:
                scheduler.step()
        self.start_epoch=int(checkpoint['completed_epoch'])
        self.best_epoch=int(checkpoint['best_epoch'])
        self.best_val_loss=float(checkpoint['best_val_loss'])
        self.resume_rng=checkpoint['rng_state']
        weighting_state=checkpoint.get('task_weighting_state')
        if weighting_state is not None:
            self.task_balancer.load_state_dict(weighting_state)
        elif self.task_balancer.mode == 'dwa':
            raise ValueError('DWA resume requires saved task weighting history')
        if self.task_balancer.mode == 'dwa' and len(self.task_balancer.history) != self.start_epoch:
            raise ValueError('DWA history does not match the completed epoch')
        print(f'Resuming after epoch {self.start_epoch}; next epoch {self.start_epoch+1}')

    @staticmethod
    def _strip_module_prefix(state_dict):
        return {k.removeprefix('module.'): v for k, v in state_dict.items()}

    def _load_full_checkpoint(self, state_dict):
        clean_state_dict = self._strip_module_prefix(state_dict)
        strict = not self.config['Trainer'].get('allow_partial_checkpoint', False)
        load_msg = self.model.load_state_dict(clean_state_dict, strict=strict)
        with open(os.path.join(self.path_model_root,'weight_load_report.json'),'w') as f:
            json.dump({'kind':'full_model','strict':strict,'missing':load_msg.missing_keys,'unexpected':load_msg.unexpected_keys},f,indent=2)
        print(f">>> 已按完整模型 checkpoint 加载 | missing={len(load_msg.missing_keys)} | unexpected={len(load_msg.unexpected_keys)} <<<")

    def _load_pretrained_backbone(self, checkpoint):
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        state_dict = self._strip_module_prefix(state_dict)
        current_state = self.model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = 'transformer_encoders.' + k if not k.startswith('transformer_encoders.') else k
            if new_k not in current_state:
                continue
            if new_k == 'transformer_encoders.pos_embed' and v.shape != current_state[new_k].shape:
                v = self._interpolate_vim_pos_embed(v, current_state[new_k].shape)
            if current_state[new_k].shape == v.shape:
                new_state_dict[new_k] = v

        backbone_state = {k:v for k,v in current_state.items() if k.startswith('transformer_encoders.')}
        eligible = sum(v.numel() for v in backbone_state.values())
        loaded = sum(v.numel() for v in new_state_dict.values())
        coverage = loaded / max(1,eligible)
        required = float(self.config['Trainer'].get('min_backbone_coverage',.9))
        report = {'kind':'backbone','coverage_by_elements':coverage,'required_coverage':required,
                  'loaded_keys':list(new_state_dict),'missing_backbone_keys':sorted(set(backbone_state)-set(new_state_dict))}
        with open(os.path.join(self.path_model_root,'weight_load_report.json'),'w') as f:
            json.dump(report,f,indent=2)
        if coverage < required:
            raise ValueError(f'Backbone pretrained coverage {coverage:.3f} below required {required:.3f}; see weight_load_report.json')
        load_msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(f">>> 已按 backbone 预训练加载 | 注入参数={len(new_state_dict)} | missing={len(load_msg.missing_keys)} | unexpected={len(load_msg.unexpected_keys)} <<<")

    def _interpolate_vim_pos_embed(self, pos_embed, target_shape):
        orig_seq_len = pos_embed.shape[1]
        target_seq_len = target_shape[1]
        dim = pos_embed.shape[-1]
        orig_cls_pos = orig_seq_len // 2
        target_cls_pos = target_seq_len // 2
        orig_size = int(math.sqrt(orig_seq_len - 1))
        target_size = int(math.sqrt(target_seq_len - 1))
        cls_tok = pos_embed[:, orig_cls_pos:orig_cls_pos + 1, :]
        patch_toks = torch.cat([pos_embed[:, :orig_cls_pos, :], pos_embed[:, orig_cls_pos + 1:, :]], dim=1)
        patch_toks = patch_toks.permute(0, 2, 1).reshape(1, dim, orig_size, orig_size)
        patch_toks = F.interpolate(patch_toks, size=(target_size, target_size), mode='bicubic', align_corners=False)
        patch_toks = patch_toks.flatten(2).permute(0, 2, 1)
        return torch.cat([patch_toks[:, :target_cls_pos, :], cls_tok, patch_toks[:, target_cls_pos:, :]], dim=1)

    def _current_scaler_value(self):
        if not self.use_amp:
            return None
        try:
            return float(self.scaler.get_scale())
        except Exception:
            return None

    def _reduce_scaler(self):
        if not self.use_amp:
            return
        current_scale = self._current_scaler_value()
        if current_scale is None:
            return
        self.scaler.update(new_scale=max(current_scale / 2.0, 1.0))

    def _set_training_mode(self):
        self.model.train()
        if self.freeze_bn_stats:
            # Fix only running statistics. Affine parameters and ViM gradients
            # remain trainable; GroupNorm and stochastic depth are unchanged.
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            print('>>> BatchNorm running statistics fixed; affine parameters and ViM remain trainable <<<')

    def _set_backbone_trainable(self, epoch_id: int):
        trainable = epoch_id > self.freeze_backbone_epochs
        for p in self.model.transformer_encoders.parameters():
            p.requires_grad = trainable
        status = '可训练' if trainable else '冻结'
        print(f">>> backbone状态={status} | freeze_backbone_epochs={self.freeze_backbone_epochs} <<<")

    def _should_run_eval(self, epoch, epochs):
        epoch_id = epoch + 1
        return (epoch_id % self.eval_every == 0) or (self.eval_on_last_epoch and epoch_id == epochs)

    def _resize_depth_target(self, depth_gt, resolution):
        if resolution == self.resolutions[-1]:
            return depth_gt
        return F.interpolate(depth_gt.unsqueeze(1), size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze(1)

    def _resize_mask(self, mask, resolution):
        if mask is None:
            return None
        if resolution == self.resolutions[-1]:
            return mask.bool()
        return F.interpolate(mask.unsqueeze(1).float(), size=(resolution, resolution), mode='nearest').squeeze(1).bool()

    def _resize_seg_target(self, seg_gt, resolution):
        if resolution == self.resolutions[-1]:
            return seg_gt.long()
        return F.interpolate(seg_gt.unsqueeze(1).float(), size=(resolution, resolution), mode='nearest').squeeze(1).long()

    def _get_cbdice_weight(self, epoch_id):
        if (not self.seg_use_cbdice) or epoch_id < self.seg_cbdice_start_epoch:
            return 0.0
        if self.seg_cbdice_warmup_epochs <= 0:
            return self.seg_cbdice_weight
        progress = (epoch_id - self.seg_cbdice_start_epoch + 1) / float(self.seg_cbdice_warmup_epochs)
        return self.seg_cbdice_weight * min(max(progress, 0.0), 1.0)

    def _depth_supervision_at_scale(self, depth_gt, zero_mask, loss_mask, resolution):
        if self.depth_valid_target_resize:
            return resize_valid_depth_target(depth_gt, zero_mask, loss_mask, (resolution, resolution))
        return (self._resize_depth_target(depth_gt, resolution),
                self._resize_mask(zero_mask, resolution), self._resize_mask(loss_mask, resolution))

    def _compute_depth_multiscale_loss(self, output_depths, depth_gt, zero_mask, loss_mask):
        weighted_terms = []
        for iter_idx, output_depths_iter in enumerate(output_depths, start=1):
            for output_depth, resolution in zip(output_depths_iter, self.resolutions):
                depth_gt_ms, zero_mask_ms, loss_mask_ms = self._depth_supervision_at_scale(
                    depth_gt, zero_mask, loss_mask, resolution)
                iter_weight = self.gamma * iter_idx
                depth_main = self.loss_depth(output_depth.squeeze(1), depth_gt_ms, zero_mask_ms, loss_mask_ms)
                weighted_terms.append(iter_weight * self.depth_scale_multi * depth_main)
                if self.loss_depth_grad is not None and self.depth_grad_multi > 0:
                    grad_masks = (zero_mask_ms, loss_mask_ms)
                    if self.depth_aux_valid_support:
                        grad_masks = depth_auxiliary_masks(zero_mask_ms, loss_mask_ms, 'sobel')
                    depth_grad = self.loss_depth_grad(output_depth, depth_gt_ms.unsqueeze(1), *grad_masks)
                    weighted_terms.append(iter_weight * self.depth_grad_multi * depth_grad)
                if self.loss_depth_normal is not None and self.depth_normal_multi > 0:
                    settings=self.config['Dataset'][self.dataset_name]
                    depth_min=float(settings.get('depth_min',0.))
                    depth_range=float(settings.get('depth_max',3.))-depth_min
                    output_normal, _, _ = get_surface_normal(output_depth*depth_range+depth_min, self.dataset_name)
                    target_normal, _, _ = get_surface_normal(depth_gt_ms.unsqueeze(1)*depth_range+depth_min, self.dataset_name)
                    normal_masks = (zero_mask_ms, loss_mask_ms)
                    if self.depth_aux_valid_support:
                        normal_masks = depth_auxiliary_masks(zero_mask_ms, loss_mask_ms, 'normal')
                    depth_normal = self.loss_depth_normal(output_normal, target_normal, *normal_masks)
                    weighted_terms.append(iter_weight * self.depth_normal_multi * depth_normal)
        return sum(weighted_terms) / max(1, len(weighted_terms))

    def _compute_seg_multiscale_loss(self, output_segs, seg_gt, cbdice_active):
        weighted_terms = []
        for iter_idx, output_segs_iter in enumerate(output_segs, start=1):
            for output_seg, resolution in zip(output_segs_iter, self.resolutions):
                seg_gt_ms = self._resize_seg_target(seg_gt, resolution)
                use_topology_loss = cbdice_active and (not self.seg_highres_only or resolution == self.resolutions[-1])
                iter_weight = self.gamma * iter_idx
                if use_topology_loss and self.fp32_topology_loss and self.use_amp:
                    with torch.amp.autocast('cuda', enabled=False):
                        seg_term = self.loss_seg_topo(output_seg.float(), seg_gt_ms.long())
                else:
                    seg_loss_fn = self.loss_seg_topo if use_topology_loss else self.loss_seg_base
                    seg_term = seg_loss_fn(output_seg, seg_gt_ms)
                weighted_terms.append(iter_weight * seg_term)
        return sum(weighted_terms) / max(1, len(weighted_terms))

    def _compute_boundary_multiscale_loss(self, output_boundaries, output_segs, seg_gt):
        weighted_terms = []
        if self.boundary_multi == 0 and self.boundary_align_multi == 0:
            return torch.tensor(0., device=self.device)
        for iter_idx, boundary_iter in enumerate(output_boundaries, start=1):
            seg_iter = output_segs[iter_idx - 1]
            for boundary_logit, output_seg, resolution in zip(boundary_iter, seg_iter, self.resolutions):
                if boundary_logit is None:
                    continue
                seg_gt_ms = self._resize_seg_target(seg_gt, resolution)
                boundary_gt = build_boundary_target(seg_gt_ms, dilation=1)
                iter_weight = self.gamma * iter_idx
                boundary_term = self.boundary_multi * self.loss_boundary(boundary_logit, boundary_gt)
                if self.boundary_align_multi > 0:
                    align_term = self.loss_boundary_align(output_seg, boundary_logit)
                    boundary_term = boundary_term + self.boundary_align_multi * align_term
                weighted_terms.append(iter_weight * boundary_term)
        if not weighted_terms:
            return torch.tensor(0.0, device=self.device)
        return sum(weighted_terms) / len(weighted_terms)

    def _compute_boundary_depth_loss(self, output_depths, depth_gt, seg_gt, valid):
        if self.boundary_depth_multi == 0:
            return torch.tensor(0., device=self.device)
        # Finest outputs are supervised with GT boundaries, independent of boundary heads.
        terms = [boundary_depth_loss(outputs[-1], depth_gt, seg_gt, valid) for outputs in output_depths]
        return self.boundary_depth_multi * sum(terms) / len(terms)

    def train(self, train_dataloader, val_dataloader):
        synthetic = bool(self.config.get('Provenance', {}).get('synthetic_test',False))
        self.progress = TrainingProgress(self.path_model_root, self.config['Trainer']['epochs'], self.dataset_name, synthetic)
        try:
            return self._train_impl(train_dataloader, val_dataloader)
        except BaseException as error:
            self.progress.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
            raise

    def _train_impl(self, train_dataloader, val_dataloader):
        self.train_loader_ref=train_dataloader
        self.val_loader_ref=val_dataloader
        if self.resume_rng is not None:
            state=self.resume_rng
            random.setstate(state['python']);np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch'].cpu())
            if state['cuda'] is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
            for loader,key in [(train_dataloader,'train_generator'),(val_dataloader,'val_generator')]:
                generator=getattr(loader,'generator',None)
                if generator is not None and state[key] is not None:
                    generator.set_state(state[key].cpu())
        epochs = self.config['Trainer']['epochs']
        if self.config['Trainer'].get('evaluate_initial', False) and self.resume_rng is None:
            self.progress.update(state='validating_initial', batch=0, batches_total=len(val_dataloader))
            initial_loss, initial_metrics, _ = self.run_eval(val_dataloader, 0)
            self.best_val_loss = initial_loss
            self.best_epoch = 0
            self.save_model('best')
            self.progress.update(state='initial_validation_complete', metrics=initial_metrics,
                                 best_epoch=0, best_val_loss=initial_loss)
            print(f'>>> Initial checkpoint validation | loss={initial_loss:.6f} | '
                  f"mIoU={initial_metrics['val_iou']:.6f} | RMSE={initial_metrics['val_rmse']:.6f} <<<")
        if self.config['wandb']['enable']:
            wandb.init(
                project="transparent-object-seg",
                entity=self.config['wandb']['username'],
                name=self.config['Model']['path_model'],
            )

        for epoch in range(self.start_epoch,epochs):
            epoch_id = epoch + 1
            self.current_epoch_id=epoch_id
            epoch_started=time.monotonic()
            self.progress.update(state='training',epoch=epoch_id,batch=0,batches_total=len(train_dataloader))
            print(f"\nEpoch {epoch_id}")
            self._set_backbone_trainable(epoch_id)
            cbdice_weight_now = self._get_cbdice_weight(epoch_id)
            cbdice_active = cbdice_weight_now > 0.0
            self.loss_seg_topo.cbdice_weight = cbdice_weight_now
            task_weights = self.task_balancer.weights()
            print(f'>>> task weighting={self.task_balancer.mode} | weights={task_weights} <<<')
            self.progress.update(task_weights=task_weights)

            train_loss = 0.0
            train_depth_all_loss = 0.0
            train_seg_all_loss = 0.0
            train_boundary_all_loss = 0.0
            depth_monitor_sum = 0.0
            seg_monitor_sum = 0.0
            valid_batches = 0
            skipped_loss_batches = 0
            skipped_grad_batches = 0

            self._set_training_mode()
            pbar = tqdm(train_dataloader, desc=f"Train e{epoch_id:02d}", leave=self.bar_leave,
                        dynamic_ncols=True, mininterval=1.0, disable=self.disable_tqdm)

            for index, data in enumerate(pbar):
                self.optimizer_backbone.zero_grad(set_to_none=True)
                self.optimizer_scratch.zero_grad(set_to_none=True)

                rgb = data['rgb'].to(self.device, non_blocking=True)
                depth_gt = data['depth_gt'].to(self.device, non_blocking=True)
                seg_gt = data['seg_gt'].to(self.device, non_blocking=True)
                zero_mask = data['zero_mask'].to(self.device, non_blocking=True).bool()
                loss_mask = data['loss_mask'].to(self.device, non_blocking=True).bool()

                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    output_depths, output_segs, output_boundaries = self.model(rgb)
                    depth_loss = self._compute_depth_multiscale_loss(output_depths, depth_gt, zero_mask, loss_mask)
                    seg_loss = self._compute_seg_multiscale_loss(output_segs, seg_gt, cbdice_active)
                    # Keep DWA monitors nonnegative and independent of topology warmup.
                    with torch.no_grad():
                        seg_monitor = (self._compute_seg_multiscale_loss(output_segs, seg_gt, False)
                                       if cbdice_active and self.task_balancer.mode == 'dwa'
                                       else seg_loss.detach())
                    boundary_loss = self._compute_boundary_multiscale_loss(output_boundaries, output_segs, seg_gt)
                    boundary_loss = boundary_loss + self._compute_boundary_depth_loss(output_depths, depth_gt, seg_gt, zero_mask)
                    losses = (
                        task_weights['depth'] * self.depth_multi * depth_loss +
                        task_weights['seg'] * self.seg_multi * seg_loss +
                        boundary_loss
                    )

                loss_value = float(losses.detach().item())
                if not math.isfinite(loss_value):
                    skipped_loss_batches += 1
                    self.optimizer_backbone.zero_grad(set_to_none=True)
                    self.optimizer_scratch.zero_grad(set_to_none=True)
                    self._reduce_scaler()
                    if self.skip_nonfinite_batch:
                        continue
                    raise FloatingPointError(f"Non-finite loss detected: {loss_value}")

                self.scaler.scale(losses).backward()
                active_optimizers = [optimizer for optimizer in (self.optimizer_scratch,self.optimizer_backbone)
                                     if any(p.grad is not None for group in optimizer.param_groups for p in group['params'])]
                for optimizer in active_optimizers:
                    self.scaler.unscale_(optimizer)
                total_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm))
                if not math.isfinite(total_grad_norm):
                    skipped_grad_batches += 1
                    self.optimizer_backbone.zero_grad(set_to_none=True)
                    self.optimizer_scratch.zero_grad(set_to_none=True)
                    self._reduce_scaler()
                    if self.skip_nonfinite_batch:
                        continue
                    raise FloatingPointError(f"Non-finite grad norm detected: {total_grad_norm}")

                for optimizer in active_optimizers:
                    self.scaler.step(optimizer)
                self.scaler.update()

                train_depth_all_loss += float((self.depth_multi * depth_loss).detach().item())
                train_seg_all_loss += float((self.seg_multi * seg_loss).detach().item())
                train_boundary_all_loss += float(boundary_loss.detach().item())
                train_loss += loss_value
                depth_monitor_sum += float(depth_loss.detach().item())
                seg_monitor_sum += float(seg_monitor.detach().item())
                valid_batches += 1

                denom = max(1, valid_batches)
                if ((index + 1) % self.train_bar_update_interval == 0) or (index + 1 == len(train_dataloader)):
                    batch_time=(time.monotonic()-epoch_started)/(index+1)
                    self.progress.update(state='training',batch=index+1,loss=train_loss/denom,
                        skipped_loss_batches=skipped_loss_batches,skipped_grad_batches=skipped_grad_batches,
                        epoch_eta_seconds=round(batch_time*(len(train_dataloader)-index-1),1),
                        gpu_allocated_bytes=torch.cuda.memory_allocated(self.device) if self.device.type=='cuda' else 0)
                    pbar.set_postfix({
                        'loss': f"{train_loss / denom:.4f}",
                        'seg': f"{train_seg_all_loss / denom:.4f}",
                        'bnd': f"{train_boundary_all_loss / denom:.4f}",
                        'skipL': skipped_loss_batches,
                        'skipG': skipped_grad_batches,
                    })

            train_denom = max(1, valid_batches)
            if valid_batches == 0:
                raise RuntimeError('No successful training batches in this epoch')
            self.task_balancer.record(depth_monitor_sum/train_denom, seg_monitor_sum/train_denom)
            print(
                f">>> epoch {epoch_id} 训练摘要 | train_loss={train_loss / train_denom:.6f} | "
                f"train_depth={train_depth_all_loss / train_denom:.6f} | "
                f"train_seg={train_seg_all_loss / train_denom:.6f} | "
                f"train_boundary={train_boundary_all_loss / train_denom:.6f} | "
                f"有效batch={valid_batches} | skip_loss={skipped_loss_batches} | skip_grad={skipped_grad_batches} <<<"
            )

            should_eval = self._should_run_eval(epoch, epochs)
            current_val_loss = None
            eval_metrics = {'val_rmse': None, 'val_mae': None, 'val_foreground_iou': None, 'val_iou': None}
            if should_eval:
                self.progress.update(state='validating',batch=0,batches_total=len(val_dataloader))
                current_val_loss, eval_metrics, _ = self.run_eval(val_dataloader, epoch_id)
                print(
                    f">>> epoch {epoch_id} 验证摘要 | val_loss={current_val_loss:.6f} | "
                    f"RMSE={eval_metrics['val_rmse']:.4f} | MAE={eval_metrics['val_mae']:.4f} | "
                    f"foreground IoU={eval_metrics['val_foreground_iou']:.4f} | IoU={eval_metrics['val_iou']:.4f} <<<"
                )
                if self.save_best and math.isfinite(current_val_loss) and current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.best_epoch = epoch_id
                    self.save_model('best')
                # Every validated epoch has a latest checkpoint for interruption recovery.
                if self.save_last:
                    self.save_model('last')
                if self.save_periodic and (epoch_id % self.save_periodic_every == 0):
                    self.save_model(f'epoch{epoch_id}')

            metric_for_scheduler = current_val_loss if current_val_loss is not None and math.isfinite(current_val_loss) else (self.best_val_loss if math.isfinite(self.best_val_loss) else 1.0)
            self.schedulers[0].step(metric_for_scheduler)
            self.schedulers[1].step(metric_for_scheduler)
            record={'epoch':epoch_id,'train_loss':train_loss/train_denom,'val_loss':current_val_loss,
                    'task_weights':task_weights,
                    'task_monitors':{'depth':depth_monitor_sum/train_denom,'seg':seg_monitor_sum/train_denom},
                    'metrics':eval_metrics,'epoch_seconds':time.monotonic()-epoch_started,
                    'skipped_loss_batches':skipped_loss_batches,'skipped_grad_batches':skipped_grad_batches}
            self.progress.epoch_summary(record)
            self.progress.update(state='epoch_complete',metrics=eval_metrics,best_epoch=self.best_epoch,
                                 best_val_loss=self.best_val_loss)

        if self.best_epoch != -1:
            print(f">>> 训练结束 | 最佳模型来自 epoch {self.best_epoch} | best val_loss={self.best_val_loss:.6f} <<<")
        print('Finished Training')
        self.progress.update(state='completed')

    def run_eval(self, val_dataloader, epoch_id=None):
        val_loss = 0.0
        valid_val_batches = 0
        accumulator = EvaluationAccumulator(self.num_classes)
        val_samples = 0
        self.model.eval()
        rgb_visual = depth_gt_visual = depth_pred_visual = seg_gt_visual = seg_pred_visual = zero_mask_visual = None

        with torch.no_grad():
            val_desc = 'Validation' if epoch_id is None else f'Val   e{epoch_id:02d}'
            pbar = tqdm(val_dataloader, desc=val_desc, leave=self.bar_leave, dynamic_ncols=True,
                        mininterval=1.0, disable=self.disable_tqdm)
            for index, data in enumerate(pbar):
                rgb = data['rgb'].to(self.device, non_blocking=True)
                depth_gt = data['depth_gt'].to(self.device, non_blocking=True)
                seg_gt = data['seg_gt'].to(self.device, non_blocking=True)
                zero_mask = data['zero_mask'].to(self.device, non_blocking=True).bool()
                loss_mask = data['loss_mask'].to(self.device, non_blocking=True).bool()
                depth_min = data['depth_min'][0].to(self.device)
                depth_max = data['depth_max'][0].to(self.device)

                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    output_depths, output_segs, output_boundaries = self.model(rgb)
                    depth_loss = self.loss_depth(output_depths[-1][-1].squeeze(1), depth_gt, zero_mask, loss_mask)
                    val_seg_loss_fn = self.loss_seg_topo if self.val_use_cbdice else self.loss_seg_base
                    if self.val_use_cbdice and self.fp32_topology_loss and self.use_amp:
                        with torch.amp.autocast('cuda', enabled=False):
                            seg_loss = val_seg_loss_fn(output_segs[-1][-1].float(), seg_gt.long())
                    else:
                        seg_loss = val_seg_loss_fn(output_segs[-1][-1], seg_gt)
                    boundary_loss = torch.tensor(0.0, device=self.device)
                    if output_boundaries[-1][-1] is not None and (self.boundary_multi > 0 or self.boundary_align_multi > 0):
                        boundary_gt = build_boundary_target(seg_gt, dilation=1)
                        boundary_loss = self.boundary_multi * self.loss_boundary(output_boundaries[-1][-1], boundary_gt)
                        boundary_loss = boundary_loss + self.boundary_align_multi * self.loss_boundary_align(output_segs[-1][-1], output_boundaries[-1][-1])
                    boundary_loss = boundary_loss + self._compute_boundary_depth_loss([output_depths[-1]], depth_gt, seg_gt, zero_mask)
                    batch_val_loss = self.depth_multi * depth_loss + self.seg_multi * seg_loss + boundary_loss

                batch_val_loss_value = float(batch_val_loss.detach().item())
                if not math.isfinite(batch_val_loss_value):
                    raise ValueError("Non-finite validation loss; do not drop difficult samples silently.")
                batch_size = rgb.shape[0]
                val_loss += batch_val_loss_value * batch_size
                val_samples += batch_size
                valid_val_batches += 1

                depth_pred = output_depths[-1][-1].squeeze(1).float()
                if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
                    depth_pred = F.interpolate(depth_pred[:, None], size=depth_gt.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
                depth_range = depth_max - depth_min
                pred_m = depth_pred * depth_range + depth_min
                gt_m = depth_gt * depth_range + depth_min
                to_numpy = lambda t: t.detach().cpu().numpy()
                accumulator.update_depth(to_numpy(pred_m), to_numpy(gt_m), to_numpy(zero_mask), to_numpy(seg_gt))
                seg_logits = output_segs[-1][-1].float()
                if seg_logits.shape[-2:] != seg_gt.shape[-2:]:
                    seg_logits = F.interpolate(seg_logits, size=seg_gt.shape[-2:], mode='bilinear', align_corners=False)
                accumulator.update_seg(to_numpy(seg_logits.argmax(1)), to_numpy(seg_gt))

                if ((index + 1) % self.val_bar_update_interval == 0) or (index + 1 == len(val_dataloader)):
                    if hasattr(self,'progress'):
                        self.progress.update(state='validating',batch=index+1,batches_total=len(val_dataloader))
                    pbar.set_postfix({'val_loss': f"{val_loss / max(1, val_samples):.4f}"})
                if rgb_visual is None:
                    rgb_visual = rgb
                    depth_gt_visual = depth_gt
                    depth_pred_visual = output_depths
                    seg_gt_visual = seg_gt
                    seg_pred_visual = output_segs
                    zero_mask_visual = zero_mask

        if self.config['wandb']['enable'] and valid_val_batches > 0 and rgb_visual is not None:
            wandb.log({"val_loss": val_loss / max(1, val_samples)})
            self.img_logger(rgb_visual, depth_gt_visual, seg_gt_visual, depth_pred_visual, seg_pred_visual, zero_mask_visual)

        if not val_samples:
            raise ValueError('Validation set contains no evaluated samples.')
        report = accumulator.compute()
        report['epoch'] = epoch_id
        report['validation_samples'] = val_samples
        report['config_provenance'] = self.config.get('Provenance', {})
        def json_safe(value):
            if isinstance(value, dict):
                return {k: json_safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [json_safe(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        report_name = 'validation.json' if epoch_id is None else f'validation_epoch{epoch_id}.json'
        with open(os.path.join(self.path_model_root, report_name), 'w', encoding='utf-8') as f:
            json.dump(json_safe(report), f, indent=2, allow_nan=False)
        metrics = {
            'val_rmse': report['all_depth']['image_rmse'],
            'val_mae': report['all_depth']['image_mae'],
            'val_iou': report['miou'],
            'val_foreground_iou': report['foreground_iou'],
            'protocol_report': report,
        }
        return val_loss / val_samples, metrics, None

    def save_model(self, name=None):
        def generator_state(loader):
            generator=getattr(loader,'generator',None)
            return generator.get_state() if generator is not None else None
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict(),
            'metric_protocol': 'dataset_global_semantic_and_image_macro_metric_depth_v1',
            'resume_version':1,
            'task_weighting_state':self.task_balancer.state_dict(),
            'completed_epoch':self.current_epoch_id,
            'best_epoch':self.best_epoch,
            'best_val_loss':self.best_val_loss,
            'scheduler_states':[s.state_dict() for s in self.schedulers],
            'scheduler_step_pending':self.current_epoch_id > 0,
            'rng_state':{'python':random.getstate(),'numpy':np.random.get_state(),
                'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'train_generator':generator_state(self.train_loader_ref),
                'val_generator':generator_state(self.val_loader_ref)},
            'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
            'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict(),
        }
        save_path = self.path_model + ('.p' if name is None else f'_{name}.p')
        temporary=save_path+'.tmp'
        torch.save(save_dict, temporary)
        os.replace(temporary,save_path)
        print(f'Model saved at : {save_path}')

    def img_logger(self, rgb, depth_gt, seg_gt, depth_pred, seg_pred, zero_mask):
        try:
            nb_to_show = min(self.config['wandb']['images_to_show'], len(rgb))
            tmp = rgb[:nb_to_show].detach()
            _ = [vis_rgb(img, self.dataset_name) for img in tmp]
        except Exception as e:
            print(f"Visualization failed: {e}")

    def isnan(self, x):
        return x != x

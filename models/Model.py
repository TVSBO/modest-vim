import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.Reassemble import Reassemble
from models.Fusion import Fusion
from models.Head import HeadBoundary, HeadDepth, HeadSeg, MultiscaleHead
from models.TransparentModules import BoundaryReflectionRefiner


HAS_OFFICIAL_VIM = False
VisionMamba = None
try:
    from models.vim.models_mamba import VisionMamba
    HAS_OFFICIAL_VIM = True
except Exception:
    try:
        from vim.models_mamba import VisionMamba
        HAS_OFFICIAL_VIM = True
    except Exception:
        HAS_OFFICIAL_VIM = False


class ISGNet(nn.Module):
    def __init__(
        self,
        image_size=(3, 384, 384),
        patch_size=16,
        emb_dim=1024,
        resample_dim=256,
        read='projection',
        num_layers_encoder=24,
        hooks=(5, 11, 17, 23),
        reassemble_s=(4, 8, 16, 32),
        transformer_dropout=0,
        nclasses=3,
        type="full",
        model_timm="vit_large_patch16_384",
        pretrain=True,
        iterations=2,
        in_chans=3,
        use_boundary_head=True,
        refine_levels=(0, 1),
        refine_hidden_dim=64,
        boundary_gcn_points=96,
        boundary_gcn_threshold=0.6,
        final_depth_use_sigmoid=True,
        bimamba_type='v2',
        use_reflection=True,
        use_point_gcn=True,
        use_boundary_guidance=True,
        use_cross_task=True,
        rgb_mean=None,
        rgb_std=None,
        refine_residual_scale=1.0,
        refiner_variant='dense',
        refiner_checkpoint=False,
        reuse_invariant_features=False,
    ):
        super().__init__()
        self.type_ = type
        self.iterations = int(iterations)
        self.reuse_invariant_features = bool(reuse_invariant_features)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hooks = list(hooks)
        self.activation = {}
        self.use_mamba = self._is_vim_model_name(model_timm)
        self.model_timm = model_timm
        self.read = read
        self.use_boundary_head = bool(use_boundary_head)
        self.use_boundary_guidance = bool(use_boundary_guidance)
        if (rgb_mean is None) != (rgb_std is None):
            raise ValueError('rgb_mean and rgb_std must be provided together')
        mean = (0., 0., 0.) if rgb_mean is None else rgb_mean
        std = (1., 1., 1.) if rgb_std is None else rgb_std
        if len(mean) != 3 or len(std) != 3 or min(std) <= 0:
            raise ValueError('RGB normalization requires three means and positive standard deviations')
        self.register_buffer('rgb_mean', torch.tensor(mean).view(1,3,1,1))
        self.register_buffer('rgb_std', torch.tensor(std).view(1,3,1,1))
        self.bimamba_type = bimamba_type
        if bimamba_type not in ('none', 'v1', 'v2'):
            raise ValueError('bimamba_type must be none, v1 or v2')
        if len(hooks) != len(reassemble_s) or len(hooks) != 4:
            raise ValueError('This decoder requires four feature hooks and four scales.')
        if self.iterations < 1:
            raise ValueError('iterations must be positive')
        if image_size[1] != image_size[2] or image_size[1] % 32:
            raise ValueError('Input must be square with a side divisible by 32.')

        self.transformer_encoders = self._build_backbone(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_layers_encoder=num_layers_encoder,
            transformer_dropout=transformer_dropout,
            model_timm=model_timm,
            pretrain=pretrain,
            in_chans=in_chans,
        )
        self.emb_dim = self._infer_backbone_dim(self.transformer_encoders, emb_dim)

        if not self.use_mamba:
            self._get_layers_from_hooks(self.hooks)

        self.reassembles = nn.ModuleList([
            Reassemble(image_size, read, patch_size, s, self.emb_dim, resample_dim)
            for s in reassemble_s
        ])
        self.fusions = nn.ModuleList([Fusion(resample_dim, nclasses, use_boundary_guidance, use_cross_task) for _ in reassemble_s])

        self.refine_levels = self._sanitize_levels(refine_levels, len(reassemble_s))
        self.stage_refiners = nn.ModuleDict({
            str(level): BoundaryReflectionRefiner(
                channels=resample_dim,
                rgb_hidden=refine_hidden_dim,
                num_points=boundary_gcn_points,
                threshold=boundary_gcn_threshold,
                use_reflection=use_reflection,
                use_point_gcn=use_point_gcn,
                residual_scale=refine_residual_scale,
                variant=refiner_variant,
                checkpoint_blocks=refiner_checkpoint,
            )
            for level in self.refine_levels
        })
        self.boundary_heads = nn.ModuleList([
            HeadBoundary(resample_dim) if level not in self.refine_levels else nn.Identity()
            for level in range(len(reassemble_s))
        ]) if self.use_boundary_head else None

        if type == "full":
            self.head_multiscale = MultiscaleHead(
                resample_dim,
                nclasses=nclasses,
                use_sigmoid_depth=final_depth_use_sigmoid,
            )
        elif type == "depth":
            self.head_depth = HeadDepth(resample_dim, use_sigmoid=final_depth_use_sigmoid)
            self.head_segmentation = None
        elif type == "seg":
            self.head_depth = None
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        else:
            self.head_depth = HeadDepth(resample_dim, use_sigmoid=final_depth_use_sigmoid)
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)

        if not self.use_boundary_guidance:
            if type == 'full':
                self.head_multiscale.head_seg.boundary_proj = None
            elif self.head_segmentation is not None:
                self.head_segmentation.boundary_proj = None

        print(
            f">>> 新版透明物体网络已启用 | refine_levels={self.refine_levels} | "
            f"refiner_variant={refiner_variant} | boundary_head={self.use_boundary_head} | iterations={self.iterations} <<<"
        )

    @staticmethod
    def _is_vim_model_name(model_name: str) -> bool:
        name = model_name.lower()
        return ('vim' in name) or ('mamba' in name)

    @staticmethod
    def _infer_backbone_dim(backbone: nn.Module, default_dim: int) -> int:
        for attr in ['embed_dim', 'd_model', 'num_features']:
            if hasattr(backbone, attr):
                value = getattr(backbone, attr)
                if isinstance(value, int):
                    return value
        return default_dim

    @staticmethod
    def _sanitize_levels(levels, total_stages: int):
        if levels is None:
            return [0, 1]
        if any(int(level) < 0 or int(level) >= total_stages for level in levels):
            raise ValueError('refine_levels contains an invalid decoder level')
        valid = sorted({int(level) for level in levels})
        return valid

    def _build_backbone(self, image_size, patch_size, emb_dim, num_layers_encoder,
                        transformer_dropout, model_timm, pretrain, in_chans):
        if self.use_mamba and not HAS_OFFICIAL_VIM:
            raise RuntimeError('Unable to import the compatible ViM backend. Check mamba_ssm and timm.')
        if self.use_mamba and HAS_OFFICIAL_VIM:
            _, h, _ = image_size
            return VisionMamba(
                img_size=h,
                patch_size=patch_size,
                stride=patch_size,
                depth=num_layers_encoder,
                embed_dim=emb_dim,
                channels=in_chans,
                num_classes=0,
                drop_rate=float(transformer_dropout),
                drop_path_rate=0.1,
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                final_pool_type='mean',
                if_abs_pos_embed=True,
                if_cls_token=True,
                if_divide_out=True,
                use_middle_cls_token=True,
                bimamba_type=self.bimamba_type,
            )
        return timm.create_model(model_timm, pretrained=pretrain, in_chans=in_chans)

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(_model, _input, output):
                self.activation[name] = output[0] if isinstance(output, tuple) else output
            return hook

        target_layers = self.transformer_encoders.blocks if hasattr(self.transformer_encoders, 'blocks') \
            else self.transformer_encoders.layers
        for h in hooks:
            target_layers[h].register_forward_hook(get_activation('t' + str(h)))

    def _vim_extract_multilevel_features(self, x: torch.Tensor):
        model = self.transformer_encoders
        feats = {}
        hidden_states = model.patch_embed(x)
        b, m, _ = hidden_states.shape
        residual = None

        cls_token = model.cls_token.expand(b, -1, -1)
        cls_pos = m // 2
        hidden_states = torch.cat(
            (hidden_states[:, :cls_pos, :], cls_token, hidden_states[:, cls_pos:, :]),
            dim=1,
        )

        if getattr(model, 'if_abs_pos_embed', False):
            hidden_states = hidden_states + model.pos_embed
            hidden_states = model.pos_drop(hidden_states)

        for idx, layer in enumerate(model.layers):
            hidden_states, residual = layer(hidden_states, residual)
            if idx in self.hooks:
                seq = hidden_states
                cls_tok = seq[:, cls_pos:cls_pos + 1, :]
                patch_tok = torch.cat((seq[:, :cls_pos, :], seq[:, cls_pos + 1:, :]), dim=1)
                feats['t' + str(idx)] = torch.cat((cls_tok, patch_tok), dim=1)
        return feats

    def _predict_outputs(self, depth_feature, seg_feature, boundary_logit=None):
        if not self.use_boundary_guidance:
            boundary_logit = None
        if self.type_ == 'full':
            return self.head_multiscale(depth_feature, seg_feature, boundary_logits=boundary_logit)
        output_depth = self.head_depth(depth_feature) if self.head_depth else None
        output_seg = self.head_segmentation(seg_feature, boundary_logits=boundary_logit) if self.head_segmentation else None
        return output_depth, output_seg

    def forward(self, img):
        if tuple(img.shape[1:]) != tuple(self.image_size):
            raise ValueError(f'Expected input shape B,{self.image_size}, received {tuple(img.shape)}')
        # Backbone uses normalized RGB; optical cues use the original RGB domain.
        cue_rgb = (img.float() * self.rgb_std + self.rgb_mean).clamp(0, 1)
        backbone_features = (
            self._vim_extract_multilevel_features(img)
            if self.use_mamba
            else (self.transformer_encoders(img), self.activation)[1]
        )

        guide_depth, guide_seg = [], []
        out_depths, out_segs, out_boundaries = [], [], []
        # Local to this forward: retain autograd, never reuse across images or
        # optimizer steps. Reassemble and cue encoders contain no BN/dropout.
        reassembled = {}
        encoded_cues = {}

        for _ in range(self.iterations):
            depth_feature, seg_feature = None, None
            multiscale_depth, multiscale_seg, multiscale_boundary = [], [], []
            boundary_prior = None

            for i in np.arange(len(self.fusions) - 1, -1, -1):
                hook_to_take = 't' + str(self.hooks[i])
                activation_result = backbone_features[hook_to_take]

                if not self.reuse_invariant_features or i not in reassembled:
                    reassemble_result = self.reassembles[i](activation_result)
                    if self.reuse_invariant_features:
                        reassembled[i] = reassemble_result
                else:
                    reassemble_result = reassembled[i]
                depth_feature, seg_feature = self.fusions[i](
                    reassemble_result,
                    i,
                    depth_feature,
                    seg_feature,
                    guide_depth,
                    guide_seg,
                    boundary_prior=boundary_prior,
                )

                if i in self.refine_levels:
                    refiner = self.stage_refiners[str(i)]
                    if self.reuse_invariant_features and i not in encoded_cues:
                        encoded_cues[i] = refiner.encode_rgb_cues(cue_rgb, seg_feature.shape[-2:])
                    seg_feature, depth_feature, boundary_logit = self.stage_refiners[str(i)](
                        seg_feature,
                        depth_feature,
                        cue_rgb,
                        boundary_prior=boundary_prior if self.use_boundary_guidance else None,
                        encoded_cues=encoded_cues.get(i),
                    )
                else:
                    boundary_logit = self.boundary_heads[i](seg_feature) if self.boundary_heads is not None else None

                output_depth, output_seg = self._predict_outputs(depth_feature, seg_feature, boundary_logit=boundary_logit)
                if boundary_logit is not None and output_seg is not None and boundary_logit.shape[-2:] != output_seg.shape[-2:]:
                    boundary_logit = F.interpolate(boundary_logit, size=output_seg.shape[-2:], mode='bilinear', align_corners=False)
                multiscale_depth.append(output_depth)
                multiscale_seg.append(output_seg)
                multiscale_boundary.append(boundary_logit)
                boundary_prior = boundary_logit if self.use_boundary_guidance else None

            # Fusion consumes predictions, not resample_dim-channel features.
            guide_depth.append(multiscale_depth)
            guide_seg.append(multiscale_seg)
            out_depths.append(multiscale_depth)
            out_segs.append(multiscale_seg)
            out_boundaries.append(multiscale_boundary)

        return out_depths, out_segs, out_boundaries

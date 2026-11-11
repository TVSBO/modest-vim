"""Dataset-global semantic metrics and explicitly scoped metric depth.

There is no instance AP implementation here: semantic masks do not establish an
instance matching protocol. Inputs to depth updates must already be in meters.
"""
import numpy as np


class EvaluationAccumulator:
    def __init__(self, num_classes, foreground_class=None):
        self.num_classes = int(num_classes)
        self.foreground_class = self.num_classes - 1 if foreground_class is None else foreground_class
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.depth = {scope: dict(count=0, abs_sum=0., sq_sum=0., rel_sum=0.,
                                 images=0, image_mae_sum=0., image_rmse_sum=0.,
                                 delta105=0, delta110=0, delta125=0)
                      for scope in ('all', 'foreground')}

    def update_seg(self, labels, gt):
        labels, gt = np.asarray(labels), np.asarray(gt)
        if labels.shape != gt.shape:
            raise ValueError('Segmentation prediction and GT shapes must match.')
        if np.any((gt < 0) | (gt >= self.num_classes)):
            raise ValueError('GT class outside declared label map; map ignore labels explicitly.')
        if np.any((labels < 0) | (labels >= self.num_classes)):
            raise ValueError('Prediction outside declared class range.')
        ids = self.num_classes * gt.astype(np.int64).ravel() + labels.astype(np.int64).ravel()
        self.confusion += np.bincount(ids, minlength=self.num_classes**2).reshape(self.confusion.shape)

    def update_depth(self, pred_m, gt_m, valid, seg_gt):
        pred_m, gt_m = np.asarray(pred_m), np.asarray(gt_m)
        valid, seg_gt = np.asarray(valid, dtype=bool), np.asarray(seg_gt)
        if not (pred_m.shape == gt_m.shape == valid.shape == seg_gt.shape):
            raise ValueError('Depth, valid mask and segmentation shapes must match.')
        if pred_m.ndim != 3:
            raise ValueError('Depth updates require B,H,W arrays.')
        valid = valid & np.isfinite(gt_m) & (gt_m > 0)
        if not np.isfinite(pred_m[valid]).all():
            raise ValueError('Non-finite prediction on valid GT pixels.')
        for scope, mask in [('all', valid), ('foreground', valid & (seg_gt == self.foreground_class))]:
            state = self.depth[scope]
            for pred, gt, m in zip(pred_m, gt_m, mask):
                count = int(m.sum())
                if not count:
                    continue
                error = pred[m].astype(np.float64) - gt[m].astype(np.float64)
                absolute, squared = np.abs(error).sum(), np.square(error).sum()
                state['count'] += count
                state['abs_sum'] += absolute
                state['sq_sum'] += squared
                state['rel_sum'] += (np.abs(error) / gt[m]).sum()
                ratio = np.full(count, np.inf)
                positive = pred[m] > 0
                ratio[positive] = np.maximum(pred[m][positive]/gt[m][positive], gt[m][positive]/pred[m][positive])
                for key, threshold in [('delta105',1.05),('delta110',1.10),('delta125',1.25)]:
                    state[key] += int((ratio < threshold).sum())
                state['images'] += 1
                state['image_mae_sum'] += absolute / count
                state['image_rmse_sum'] += np.sqrt(squared / count)

    def compute(self):
        diagonal = np.diag(self.confusion).astype(np.float64)
        support = self.confusion.sum(axis=0) + self.confusion.sum(axis=1)
        union = support - diagonal
        iou = np.divide(diagonal, union, out=np.full(self.num_classes, np.nan), where=union > 0)
        dice = np.divide(2*diagonal, support, out=np.full(self.num_classes, np.nan), where=support > 0)
        finite = iou[np.isfinite(iou)]
        result = {'miou': float(finite.mean()) if finite.size else float('nan'),
                  'foreground_iou': float(iou[self.foreground_class]),
                  'foreground_dice': float(dice[self.foreground_class]),
                  'iou_per_class': iou.tolist(), 'confusion': self.confusion.tolist(),
                  'depth_unit': 'meter', 'seg_aggregation': 'dataset_global_confusion'}
        for scope, s in self.depth.items():
            count, images = s['count'], s['images']
            result[scope+'_depth'] = {
                'valid_pixels': count, 'valid_images': images,
                'pixel_mae': s['abs_sum']/count if count else float('nan'),
                'pixel_rmse': float(np.sqrt(s['sq_sum']/count)) if count else float('nan'),
                'pixel_absrel': s['rel_sum']/count if count else float('nan'),
                'image_mae': s['image_mae_sum']/images if images else float('nan'),
                'image_rmse': s['image_rmse_sum']/images if images else float('nan'),
                **{key: s[key]/count if count else float('nan') for key in ('delta105','delta110','delta125')}}
        return result

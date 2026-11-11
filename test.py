"""Evaluate the image list specified in config/config.json."""
import argparse
import json
from pathlib import Path
import torch
from model import load_model, read_config, DEFAULT_CONFIG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--output', default='outputs/metrics.json')
    args = parser.parse_args()
    from utils.builder import get_dataloader
    from utils.protocol_metrics import EvaluationAccumulator
    config = read_config(args.config)
    model = load_model(args.checkpoint, config=config)
    accumulator = EvaluationAccumulator(2)
    for batch in get_dataloader(config, args.split):
        with torch.inference_mode():
            depth, segmentation, _ = model(batch['rgb'].cuda())
        depth = depth[-1][-1][:, 0].float().cpu().numpy()
        settings = config['Dataset']['my_new_dataset']
        scale = settings['depth_max'] - settings['depth_min']
        offset = settings['depth_min']
        labels = segmentation[-1][-1].argmax(1).cpu().numpy()
        gt = batch['seg_gt'].numpy()
        accumulator.update_depth(depth * scale + offset, batch['depth_gt'].numpy() * scale + offset,
                                 batch['zero_mask'].numpy(), gt)
        accumulator.update_seg(labels, gt)
    result = accumulator.compute()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

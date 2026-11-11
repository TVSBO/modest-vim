"""Predict metric depth and a binary mask from one RGB image."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF, InterpolationMode
from model import load_model, read_config, DEFAULT_CONFIG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='outputs/inference')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = read_config(args.config)
    image = Image.open(args.image).convert('RGB')
    settings = config['Dataset']['my_new_dataset']
    size = settings['transforms']['im_resize']
    rgb = TF.to_tensor(TF.resize(image, [size, size], interpolation=InterpolationMode.BILINEAR))
    rgb = TF.normalize(rgb, [.485, .456, .406], [.229, .224, .225])[None].to(args.device)
    model = load_model(args.checkpoint, args.device, config)
    with torch.inference_mode():
        depth, segmentation, _ = model(rgb)
        depth = depth[-1][-1][0, 0].float().cpu().numpy()
        depth = depth * (settings['depth_max'] - settings['depth_min']) + settings['depth_min']
        mask = segmentation[-1][-1][0].argmax(0).byte().cpu().numpy()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'depth_m.npy', depth)
    Image.fromarray(mask * 255).save(output / 'mask.png')
    print(f'Saved depth_m.npy and mask.png to {output}')

if __name__ == '__main__':
    main()

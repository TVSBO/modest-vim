# MODEST-ViM

Code for **Vision Mamba for Joint Transparent Object Segmentation and Depth Estimation in Robotic Grasping**.

## Installation

Use Linux, Python 3.10, an NVIDIA GPU, and the CUDA 12.8 toolkit (including `nvcc`).

```bash
git clone https://github.com/TVSBO/modest-vim.git
cd modest-vim
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
MAX_JOBS=2 pip install ./third_party/causal_conv1d --no-build-isolation
MAX_JOBS=2 pip install ./third_party/mamba --no-build-isolation
```

The bundled Mamba source provides the bidirectional interface used by this model; do not replace it with the standard pip package.

## Data and configuration

Download [ClearPose](https://github.com/opipari/ClearPose) from its original provider. This package includes a ClearPose image-list loader; [Syn-TODD](https://github.com/ac-rad/MVTrans) is available separately from its original provider.

In `config/config.json`, set the dataset root, train/validation/test image lists, checkpoint path, and output directory. Each list contains one RGB path relative to the dataset root per line, such as `set1/scene1/000000-color.png`. The loader reads the corresponding `-depth_true.png` depth in millimeters and `-label.png` object masks.

The default configuration uses 384 × 384 inputs, batch size 2, and 30 training epochs.

## Weights

Model weights and datasets are not included in this source package.

- Backbone initialization: ViM-Small/16, BiMamba-v2, ImageNet-1K checkpoint `vim_s_midclstok_80p5acc.pth` from the [official Vim repository](https://github.com/hustvl/Vim). Set `Trainer.ckpt_path` to its local path for pretrained training. An empty path uses random initialization.
- Inference and evaluation require a full MODEST-ViM checkpoint with the same architecture as `config/config.json`; the ImageNet backbone checkpoint alone is insufficient. Pass its local path using `--checkpoint`.
- `--resume` requires a full training checkpoint containing optimizer and scheduler states.

## Training

```bash
python train.py --config config/config.json
```

## Evaluation

```bash
python test.py --checkpoint /path/to/modest_vim_clearpose.pth --split test
```

Depth errors are evaluated in meters without scale alignment. `image_rmse` and `image_mae` average image-level errors; `pixel_rmse` and `pixel_mae` pool valid pixels. The main region contains all valid ground-truth pixels; `foreground_depth` restricts evaluation to the ground-truth object mask.

## Inference

```bash
python inference.py --image /path/to/image.png --checkpoint /path/to/modest_vim_clearpose.pth
```

Outputs are saved to `outputs/inference/`: `depth_m.npy` contains metric depth, and `mask.png` is the binary object mask.

## Acknowledgements

Built on [MODEST](https://github.com/L-J-Yuan/MODEST), [Vim](https://github.com/hustvl/Vim), and [Mamba](https://github.com/state-spaces/mamba). Upstream licenses and attribution are retained in `LICENSE`, `NOTICE`, and the corresponding source directories.

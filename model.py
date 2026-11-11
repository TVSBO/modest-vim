"""MODEST-ViM model construction and checkpoint loading."""
import json
from pathlib import Path
import torch
from models.Model import ISGNet

DEFAULT_CONFIG = Path(__file__).parent / 'config' / 'config.json'

def read_config(path=DEFAULT_CONFIG):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)

def create_model(config=None):
    config = read_config() if config is None else config
    kwargs = dict(config['Model'])
    kwargs.pop('path_model', None)
    kwargs['pretrain'] = False
    size = config['Dataset']['my_new_dataset']['transforms']['im_resize']
    return ISGNet(image_size=(3, size, size), nclasses=2, in_chans=3, **kwargs)

def load_model(checkpoint, device='cuda', config=None):
    model = create_model(config)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = saved.get('model_state_dict', saved)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()

"""Train with the configuration in config/config.json."""
import argparse
from pathlib import Path
from model import read_config, DEFAULT_CONFIG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--resume', help='Full training checkpoint, not inference weights')
    args = parser.parse_args()
    from models.Trainer import Trainer
    from utils.builder import get_dataloader
    from utils.reproducibility import seed_everything
    config = read_config(args.config)
    if args.resume:
        config['Trainer']['resume_path'] = str(Path(args.resume).resolve())
    seed_everything(config['Trainer']['seed'])
    train = get_dataloader(config, 'train')
    val = get_dataloader(config, 'val')
    trainer = Trainer(config)
    seed_everything(config['Trainer']['seed'])
    trainer.train(train, val)

if __name__ == '__main__':
    main()

"""Cache frozen ResNet-18 features from the official CIFAR-10 train/test sets.

Pass dataset and model locations as arguments. The model can be a Hugging Face
ID or a local save_pretrained directory. Cached/local weights are required by
default; --allow-download permits a model download when running.
CIFAR-10 is always read locally, with no download or
training. Labels support source anchoring and offline controlled evaluation.
"""
from pathlib import Path
import argparse
import json
import numpy as np

MODEL_NAME = 'microsoft/resnet-18'
OUT = Path(__file__).resolve().parents[3]/'experiments/results/visual_features'
BATCH_SIZE = 128
DEVICE = 'cuda'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Local CIFAR-10 root')
    parser.add_argument('--model', default=MODEL_NAME, help='HF model ID or local model directory')
    parser.add_argument('--cache-dir', type=Path, help='Hugging Face model cache directory')
    parser.add_argument('--allow-download', action='store_true', help='Allow fetching model files')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    if args.output.exists():
        raise FileExistsError(f'Choose a new cache directory: {args.output}')
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from transformers import AutoImageProcessor, ResNetModel
    # Both official sets must be readable before any output is written.
    sets = {name: datasets.CIFAR10(str(args.data_root), train=train, download=False)
            for name, train in [('train', True), ('test', False)]}
    load_args = dict(local_files_only=not args.allow_download,
                     cache_dir=str(args.cache_dir) if args.cache_dir is not None else None)
    processor = AutoImageProcessor.from_pretrained(args.model, use_fast=False, **load_args)
    model = ResNetModel.from_pretrained(args.model, **load_args)
    model.requires_grad_(False).eval().to(args.device)

    def collate(batch):
        images, labels = zip(*batch)
        inputs = processor(images=list(images), return_tensors='pt')
        return inputs['pixel_values'], torch.tensor(labels)

    values = {}
    with torch.inference_mode():
        for name, dataset in sets.items():
            features, labels = [], []
            for images, target in DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=0, collate_fn=collate):
                pooled = model(pixel_values=images.to(args.device), return_dict=True).pooler_output
                features.append(pooled.flatten(1).cpu().numpy())
                labels.append(target.numpy())
            x, y = np.concatenate(features), np.concatenate(labels)
            if x.shape != (len(dataset),512) or not np.isfinite(x).all():
                raise ValueError(f'Invalid {name} feature tensor')
            if len(np.unique(y)) != 10:
                raise ValueError(f'Expected all ten CIFAR-10 classes in {name}')
            values[name+'_features'], values[name+'_labels'] = x, y
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'features.npz', **values)
    protocol = dict(dataset='CIFAR-10', splits=['official train', 'official test'],
                    architecture='ResNet-18', checkpoint=args.model, data_root=str(args.data_root),
                    cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
                    device=args.device, batch_size=args.batch_size,
                    preprocessing=processor.to_dict(),
                    feature='Frozen 512-dimensional average pool; eval mode',
                    classes=sets['train'].classes, no_training=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

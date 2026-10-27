"""Cache frozen ResNet-18 features from the official CIFAR-10 train/test sets.

Run on the server after setting DATA, MODEL_NAME and OUT. MODEL_NAME can be a
Hugging Face ID or a local save_pretrained directory. Cached/local weights are
required by default; set LOCAL_FILES_ONLY=False explicitly to allow a model
download when running. CIFAR-10 is always read locally, with no download or
training. Labels support source anchoring and offline controlled evaluation.
"""
from pathlib import Path
import json
import numpy as np

DATA = Path('/data/cifar10')
MODEL_NAME = 'microsoft/resnet-18'
LOCAL_FILES_ONLY = True
OUT = Path(__file__).resolve().parents[3]/'experiments/results/visual_features'
BATCH_SIZE = 128
DEVICE = 'cuda'


def main():
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from transformers import AutoImageProcessor, ResNetModel
    if OUT.exists():
        raise FileExistsError(f'Choose a new cache directory: {OUT}')
    # Both official sets must be readable before any output is written.
    sets = {name: datasets.CIFAR10(str(DATA), train=train, download=False)
            for name, train in [('train', True), ('test', False)]}
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, local_files_only=LOCAL_FILES_ONLY,
                                                   use_fast=False)
    model = ResNetModel.from_pretrained(MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
    model.requires_grad_(False).eval().to(DEVICE)

    def collate(batch):
        images, labels = zip(*batch)
        inputs = processor(images=list(images), return_tensors='pt')
        return inputs['pixel_values'], torch.tensor(labels)

    values = {}
    with torch.inference_mode():
        for name, dataset in sets.items():
            features, labels = [], []
            for images, target in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=0, collate_fn=collate):
                pooled = model(pixel_values=images.to(DEVICE), return_dict=True).pooler_output
                features.append(pooled.flatten(1).cpu().numpy())
                labels.append(target.numpy())
            x, y = np.concatenate(features), np.concatenate(labels)
            if x.shape != (len(dataset),512) or not np.isfinite(x).all():
                raise ValueError(f'Invalid {name} feature tensor')
            if len(np.unique(y)) != 10:
                raise ValueError(f'Expected all ten CIFAR-10 classes in {name}')
            values[name+'_features'], values[name+'_labels'] = x, y
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'features.npz', **values)
    protocol = dict(dataset='CIFAR-10', splits=['official train', 'official test'],
                    architecture='ResNet-18', checkpoint=str(MODEL_NAME),
                    preprocessing=processor.to_dict(),
                    feature='Frozen 512-dimensional average pool; eval mode',
                    classes=sets['train'].classes, no_training=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

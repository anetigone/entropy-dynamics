"""Extract frozen features from locally unpacked official CIFAR-10-C arrays."""
import argparse
import json
from pathlib import Path

import numpy as np

CORRUPTIONS = ['gaussian_noise', 'defocus_blur', 'contrast']


def check_preprocessing(source, target):
    for key in ['do_resize', 'size', 'resample', 'crop_pct', 'do_center_crop',
                'crop_size', 'do_rescale', 'rescale_factor', 'do_normalize',
                'image_mean', 'image_std']:
        left, right = source.get(key), target.get(key)
        if key in ['crop_pct', 'rescale_factor', 'image_mean', 'image_std'] and left is not None and right is not None:
            left_array, right_array = np.asarray(left), np.asarray(right)
            equal = left_array.shape == right_array.shape and np.allclose(
                left_array, right_array, rtol=1e-7, atol=1e-8)
        else:
            equal = left == right
        if not equal:
            raise ValueError(f'Clean/corrupted preprocessing differs: {key}: clean={left!r}, corrupt={right!r}')


def load_images(root, corruption, severity):
    if not 1 <= severity <= 5:
        raise ValueError('Severity must be in 1..5')
    images = np.load(root / (corruption + '.npy'), mmap_mode='r', allow_pickle=False)
    labels = np.load(root / 'labels.npy', allow_pickle=False)
    if images.shape != (50000, 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError(f'Expected official 50000x32x32x3 uint8 images: {corruption}')
    if labels.shape not in [(10000,), (50000,)]:
        raise ValueError('Expected 10000 or 50000 official labels')
    start = (severity - 1) * 10000
    # Some official distributions repeat the labels for all five severity blocks.
    y = labels if len(labels) == 10000 else labels[start:start + 10000]
    if not np.issubdtype(y.dtype, np.integer) or set(np.unique(y)) != set(range(10)):
        raise ValueError('Expected integer CIFAR-10 labels 0..9')
    if not np.all(np.bincount(y, minlength=10) == 1000):
        raise ValueError('Expected the full official test block, 1000 images per class')
    return images[start:start + 10000], y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model', default='microsoft/resnet-18')
    p.add_argument('--clean-protocol', type=Path,
                   default=Path('experiments/results/visual_features/protocol.json'))
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--allow-download', action='store_true', help='Permit model files only')
    p.add_argument('--output', type=Path, default=Path('experiments/results/corrupt_features'))
    p.add_argument('--corruptions', nargs='+', default=CORRUPTIONS)
    p.add_argument('--severities', nargs='+', type=int, default=[1, 3])
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a new feature directory: {args.output}')
    if args.batch_size <= 0 or any(s not in range(1, 6) for s in args.severities):
        p.error('Require positive batch size and severities in 1..5')
    if (len(set(args.corruptions)) != len(args.corruptions)
            or len(set(args.severities)) != len(args.severities)):
        p.error('Do not repeat corruption/severity entries')
    for corruption in args.corruptions:
        if Path(corruption).name != corruption:
            p.error('Use corruption names, not paths')
        for severity in args.severities:
            load_images(args.data_root, corruption, severity)

    import torch
    from PIL import Image
    import transformers
    from transformers import ResNetModel
    options = dict(local_files_only=not args.allow_download,
                   cache_dir=str(args.cache_dir) if args.cache_dir else None)
    clean_protocol = json.loads(args.clean_protocol.read_text(encoding='utf-8'))
    # The clean cache defines the preprocessing, not checkpoint defaults, which
    # may differ across local model directories or Transformers versions.
    config = clean_protocol['preprocessing']
    processor_class = getattr(transformers, config['image_processor_type'])
    processor = processor_class.from_dict(config)
    check_preprocessing(clean_protocol['preprocessing'], processor.to_dict())
    model = ResNetModel.from_pretrained(args.model, **options)
    model.requires_grad_(False).eval().to(args.device)
    args.output.mkdir(parents=True)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(dataset='CIFAR-10-C official test corruption blocks', complete=False,
                    preprocessing=processor.to_dict(), feature_dim=512, no_training=True,
                    clean_checkpoint=clean_protocol['checkpoint'],
                    feature='Frozen 512-dimensional average pool; eval mode', environments=[])

    def save():
        (args.output / 'protocol.json').write_text(
            json.dumps(protocol, indent=2, allow_nan=False), encoding='utf-8')

    save()
    with torch.inference_mode():
        for corruption in args.corruptions:
            for severity in args.severities:
                images, labels = load_images(args.data_root, corruption, severity)
                features = []
                for start in range(0, len(images), args.batch_size):
                    batch = [Image.fromarray(x) for x in images[start:start + args.batch_size]]
                    inputs = processor(images=batch, return_tensors='pt')
                    pooled = model(pixel_values=inputs['pixel_values'].to(args.device),
                                   return_dict=True).pooler_output
                    features.append(pooled.flatten(1).cpu().numpy())
                features = np.concatenate(features)
                if features.shape != (10000, 512) or not np.isfinite(features).all():
                    raise ValueError('Invalid frozen feature tensor')
                key = f'{corruption}_s{severity}'
                np.savez_compressed(args.output / (key + '.npz'),
                                    test_features=features, test_labels=labels)
                protocol['environments'].append(dict(key=key, corruption=corruption,
                                                     severity=severity, images=len(labels)))
                save()
                print(f'{key}: saved {features.shape}', flush=True)
    protocol['complete'] = True
    save()


if __name__ == '__main__':
    main()

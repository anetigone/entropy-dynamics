"""Full-batch Tent updates and fixed-pool risk measurements for a BN pilot."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--eval-batch-size', type=int, default=200)
    p.add_argument('--horizon', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--epsilon', type=float, default=0.02)
    p.add_argument('--severity', type=int, default=3)
    p.add_argument('--corruptions', nargs='+', default=['gaussian_noise', 'defocus_blur', 'contrast'])
    p.add_argument('--mean', nargs=3, type=float, default=[0.4914, 0.4822, 0.4465])
    p.add_argument('--std', nargs=3, type=float, default=[0.2023, 0.1994, 0.2010])
    return p


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


class Dynamics:
    def __init__(self, args):
        self.args = args
        if args.batch_size < 2 or args.horizon < 1 or args.lr <= 0:
            raise ValueError('Require batch >=2, positive horizon and learning rate')
        if not 1 <= args.severity <= 5 or args.eval_batch_size < 2 or 10000 % args.eval_batch_size:
            raise ValueError('Severity 1..5; evaluation batch must divide 10000')
        if args.epsilon <= 0 or min(args.std) <= 0:
            raise ValueError('Require positive epsilon and normalization scales')
        torch.manual_seed(args.seed)
        self.model = resnet18(weights=None, num_classes=10)
        self.model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        self.model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        self.model.to(args.device).train().requires_grad_(False)
        self.params, self.names = [], []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = module.running_var = None
                for key in ('weight', 'bias'):
                    self.params.append(getattr(module, key))
                    self.names.append(name + '.' + key)
        self.origin = self.vector()
        self.mean = torch.tensor(args.mean, device=args.device)[None, :, None, None]
        self.std = torch.tensor(args.std, device=args.device)[None, :, None, None]

    def vector(self):
        return torch.cat([p.detach().flatten() for p in self.params]).cpu().numpy().copy()

    def set_vector(self, vector):
        offset = 0
        with torch.no_grad():
            for p in self.params:
                p.copy_(torch.as_tensor(vector[offset:offset+p.numel()], device=p.device).reshape(p.shape))
                offset += p.numel()
        if offset != len(vector) or not np.isfinite(vector).all():
            raise ValueError('Invalid BN vector')

    def load(self, corruption):
        a = self.args
        if Path(corruption).name != corruption:
            raise ValueError('Use corruption names, not paths')
        images = np.load(a.data_root / (corruption + '.npy'), mmap_mode='r')
        labels = np.load(a.data_root / 'labels.npy')
        if images.shape != (50000, 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError('Require official CIFAR-10-C image arrays')
        start = (a.severity - 1) * 10000
        if labels.shape == (50000,):
            labels = labels[start:start+10000]
        if labels.shape != (10000,) or set(np.unique(labels)) != set(range(10)):
            raise ValueError('Require official ten-class labels')
        return images[start:start+10000], labels

    def inputs(self, images):
        x = torch.from_numpy(np.array(images, copy=True)).to(self.args.device)
        return (x.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std

    def update(self, images, rng, batch_size=None):
        """One complete batch gradient: never treat examples as independent gradients."""
        size = batch_size or self.args.batch_size
        index = rng.integers(len(images), size=size)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(self.inputs(images[index]))
        loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        delta = np.concatenate([-self.args.lr * p.grad.detach().cpu().numpy().ravel() for p in self.params])
        if not np.isfinite(delta).all():
            raise ValueError('Nonfinite update')
        return delta

    @torch.no_grad()
    def risk(self, images, labels, order):
        wrong = 0
        for index in order.reshape(-1, self.args.eval_batch_size):
            logits = self.model(self.inputs(images[index]))
            wrong += (logits.argmax(1).cpu().numpy() != labels[index]).sum()
        return float(wrong / len(order))

    def trajectory(self, images, labels, rng):
        self.set_vector(self.origin)
        # Fresh independent partition per run, held fixed across all H+1 risks.
        order = rng.permutation(len(images))
        states, risks = [self.origin.copy()], [self.risk(images, labels, order)]
        for _ in range(self.args.horizon):
            self.set_vector(states[-1])
            states.append(states[-1] + self.update(images, rng))
            self.set_vector(states[-1])
            risks.append(self.risk(images, labels, order))
        return np.asarray(states), np.asarray(risks)


def scalar_forecast(offset, increments, risk_coef, horizon, epsilon, simulations, rng):
    """Local affine drift, pooled batch noise; development-only quadratic risk map."""
    means = increments.mean(axis=1)
    intercept = means[1]
    slope = (means[2] - means[0]) / (2 * offset)
    noise = (increments - means[:, None]).ravel()
    x = np.zeros(simulations)
    hit = np.zeros(simulations, dtype=bool)
    for _ in range(horizon):
        x += intercept + slope * x + rng.choice(noise, size=simulations)
        hit |= risk_coef[0] * x + risk_coef[1] * x*x > epsilon
    return float(hit.mean()), float(intercept), float(slope), float(noise.var(ddof=1))

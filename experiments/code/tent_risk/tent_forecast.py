"""Unlabelled batch probes and small regularized event probability models."""
import sys
from pathlib import Path

import numpy as np

# Reuse the existing, uniquely named full-network implementation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bn_probe'))

SIGNALS = ['entropy', 'confidence', 'margin', 'class_concentration']
STATIC = [f'{s}_{v}' for s in SIGNALS for v in ('mean', 'sd')]
CONTEXT = ['learning_rate', 'start_step', 'horizon', 'epsilon']
FEATURES = CONTEXT + STATIC + [f'{s}_delta_{v}' for s in SIGNALS for v in ('mean', 'sd')]
STATIC_COLUMNS = list(range(len(CONTEXT)+len(STATIC)))
DRIFT_COLUMNS = STATIC_COLUMNS + list(range(len(STATIC_COLUMNS), len(FEATURES), 2))


def signals(dyn, batches):
    import torch
    values = []
    with torch.no_grad():
        for images in batches:
            p = dyn.model(dyn.inputs(images)).softmax(1)
            top = p.topk(2, dim=1).values
            values.append([float(-(p*p.clamp_min(1e-12).log()).sum(1).mean()),
                           float(top[:, 0].mean()), float((top[:, 0]-top[:, 1]).mean()),
                           float(p.mean(0).square().sum())])
    return np.asarray(values)


def summarize(a):
    return np.stack([a.mean(0), a.std(0, ddof=1)], axis=1).ravel()


def probe(dyn, images, state, rng, count, panel_batches):
    """No labels. Reset state for each independent complete-batch update."""
    dyn.set_vector(state)
    batches = [images[rng.integers(len(images), size=dyn.args.batch_size)]
               for _ in range(max(count, panel_batches))]
    base = signals(dyn, batches)
    panel = batches[:panel_batches]
    before = base[:panel_batches].mean(0)
    deltas = []
    for _ in range(count):
        dyn.set_vector(state)
        delta = dyn.update(images, rng)
        dyn.set_vector(state+delta)
        deltas.append(signals(dyn, panel).mean(0)-before)
    dyn.set_vector(state)
    result = np.concatenate([summarize(base), summarize(np.asarray(deltas))])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite probe features')
    return result


def fit_logistic(x, y, penalty=1.):
    mean, scale = x.mean(0), x.std(0)
    scale = np.maximum(scale, 1e-8)
    a = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    beta = np.zeros(a.shape[1])
    # Small strictly convex penalty also keeps single-class folds finite.
    regularizer = np.full(len(beta), penalty)
    regularizer[0] = 1e-4
    def objective(b):
        z = a@b
        return np.mean(np.logaddexp(0, z)-y*z)+.5*np.sum(regularizer*b*b)
    for _ in range(100):
        z = a@beta
        p = 1/(1+np.exp(-np.clip(z, -40, 40)))
        gradient = a.T@(p-y)/len(y)+regularizer*beta
        hessian = (a.T*(p*(1-p)))@a/len(y)+np.diag(regularizer)
        step = np.linalg.solve(hessian, gradient)
        rate = 1.
        old = objective(beta)
        while rate > 1e-8 and objective(beta-rate*step) > old:
            rate *= .5
        beta -= rate*step
        if np.linalg.norm(rate*step) < 1e-8:
            break
    return dict(mean=mean.tolist(), scale=scale.tolist(), beta=beta.tolist())


def predict(model, x):
    z = np.column_stack([np.ones(len(x)), (x-np.array(model['mean']))/model['scale']])@model['beta']
    return 1/(1+np.exp(-np.clip(z, -40, 40)))


def load_tasks(folder):
    import json
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    if not protocol['complete']:
        raise ValueError('Task collection is incomplete')
    paths = sorted(folder.glob('task_*.npz'))
    features, outcomes, metadata = [], [], []
    for path in paths:
        with np.load(path) as d:
            features.append(d['features'])
            outcomes.append(d['events'])
            metadata.append(json.loads(str(d['metadata'])))
    if not paths:
        raise ValueError('No tasks')
    x, y = np.asarray(features), np.asarray(outcomes)
    if x.shape != (len(paths), len(FEATURES)) or y.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Invalid task arrays')
    return x, y, metadata, protocol

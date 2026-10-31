"""Score a frozen predictor on independent Tent task collections."""
import argparse
import json
from pathlib import Path
import numpy as np
from tent_forecast import FEATURES, load_tasks, predict


def evaluate(predictions, events, meta, bootstrap, seed):
    rng = np.random.default_rng(seed)
    losses = {k:np.mean((v[:, None]-events)**2, axis=1) for k,v in predictions.items()}
    out = dict(tasks=len(events), future_repeats=events.shape[1], event_rate=float(events.mean()),
               brier={k:float(v.mean()) for k,v in losses.items()}, improvements={}, reliability={})
    # Paired resampling of whole task clusters. No N_tasks*N_futures pseudo-N.
    indices = rng.integers(len(events), size=(bootstrap,len(events)))
    for name in predictions:
        difference = losses[name]-losses['dynamics']
        ci = np.quantile(difference[indices].mean(1), [.025,.975])
        out['improvements'][name] = dict(brier_reduction=float(difference.mean()),
                                         descriptive_95_interval=ci.tolist())
        bins = []
        edges = np.linspace(0,1,6)
        for i, lower in enumerate(edges[:-1]):
            upper = edges[i+1]
            mask = (predictions[name]>=lower)&((predictions[name]<=upper) if i==4 else (predictions[name]<upper))
            if mask.any():
                bins.append(dict(lower=float(lower), tasks=int(mask.sum()),
                    predicted=float(predictions[name][mask].mean()),
                    observed=float(events[mask].mean())))
        out['reliability'][name] = bins
    out['by_condition'] = []
    for group in sorted(set((m['corruption'],m['severity'],m['lr'],m['start_step']) for m in meta)):
        mask = np.array([(m['corruption'],m['severity'],m['lr'],m['start_step'])==group for m in meta])
        out['by_condition'].append(dict(corruption=group[0], severity=group[1], lr=group[2],
            start_step=group[3], tasks=int(mask.sum()), event_rate=float(events[mask].mean()),
            brier={k:float(v[mask].mean()) for k,v in losses.items()}))
    return out


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--predictor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=93)
    args = p.parse_args()
    if args.bootstrap < 2:
        p.error('Bootstrap count must be >=2')
    if args.output.exists():
        raise FileExistsError(args.output)
    model = json.loads(args.predictor.read_text(encoding='utf-8'))
    x, events, meta, protocol = load_tasks(args.tasks)
    dev = model['development_protocol']
    if protocol['seed'] == dev['seed']:
        raise ValueError('Confirmation requires a new task seed')
    for key in ['features','parameter_names','mean','std','batch_size','eval_batch_size',
                'horizon','epsilon','probe_batches','panel_batches']:
        if protocol[key] != dev[key]:
            raise ValueError(f'Frozen interface mismatch: {key}')
    if model['features'] != FEATURES:
        raise ValueError('Feature interface mismatch')
    predictions = dict(constant=np.full(len(x), model['constant']))
    for name, fitted in model['models'].items():
        predictions[name] = predict(fitted, x[:, fitted['columns']])
    report = evaluate(predictions, events, meta, args.bootstrap, args.seed)
    report['unseen_corruptions'] = sorted(set(m['corruption'] for m in meta)-set(model['development_corruptions']))
    if all('probe_seconds' in m for m in meta):
        report['mean_probe_seconds'] = float(np.mean([m['probe_seconds'] for m in meta]))
        report['mean_future_evaluation_seconds'] = float(np.mean([m['future_seconds'] for m in meta]))
    report['interpretation'] = 'Task-cluster intervals conditional on chosen environments; no environment-population guarantee'
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'predictions.npz', events=events, **predictions)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report['brier'],indent=2))


if __name__ == '__main__':
    main()

"""Forecast from a saved BN start state without loading any labels or futures."""
import json
from pathlib import Path
import numpy as np
from tent_forecast import FEATURES, probe, predict
from bn_dynamics import Dynamics, parser, save_json


def main():
    p = parser(__doc__)
    p.set_defaults(horizon=100)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--predictor', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = json.loads(args.predictor.read_text(encoding='utf-8'))
    dev = model['development_protocol']
    for key in ['mean','std','batch_size','eval_batch_size','horizon','epsilon']:
        if getattr(args,key) != dev[key]:
            p.error(f'Frozen interface mismatch: {key}')
    if model['features'] != FEATURES:
        raise ValueError('Feature interface mismatch')
    with np.load(args.state) as d:
        state = d['start_state']
        meta = json.loads(str(d['metadata']))
    args.lr, args.severity = meta['lr'], meta['severity']
    corruption = meta['corruption']
    if Path(corruption).name != corruption:
        raise ValueError('Invalid corruption name')
    dyn = Dynamics(args)
    if dyn.names != dev['parameter_names']:
        raise ValueError('BN parameter interface mismatch')
    images = np.load(args.data_root/(corruption+'.npy'),mmap_mode='r')
    if images.shape != (50000,32,32,3) or images.dtype != np.uint8:
        raise ValueError('Require official CIFAR-10-C images')
    begin = (args.severity-1)*10000
    images = images[begin:begin+10000]
    features = np.concatenate([[args.lr,meta['start_step'],args.horizon,args.epsilon],
        probe(dyn,images,state,np.random.default_rng(args.seed),dev['probe_batches'],dev['panel_batches'])])
    probabilities = dict(constant=model['constant'])
    for name,fitted in model['models'].items():
        probabilities[name] = float(predict(fitted,features[None,fitted['columns']])[0])
    args.output.mkdir(parents=True)
    save_json(args.output/'forecast.json',dict(probabilities=probabilities,
        features=features.tolist(),feature_names=FEATURES,seed=args.seed,
        corruption=corruption,severity=args.severity,lr=args.lr,start_step=meta['start_step'],
        horizon=args.horizon,epsilon=args.epsilon,
        information='Saved BN state, task settings, image-only probes. No labels or future arrays loaded.'))
    print(json.dumps(probabilities,indent=2))


if __name__ == '__main__':
    main()

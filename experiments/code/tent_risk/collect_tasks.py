"""Collect fixed-start Tent probes and independent finite-horizon futures."""
import json
import time
import numpy as np
from tent_forecast import FEATURES, probe
from bn_dynamics import Dynamics, parser, save_json


def main():
    p = parser(__doc__)
    p.set_defaults(horizon=100, severity=None, lr=None)
    p.add_argument('--severities', nargs='+', type=int)
    p.add_argument('--learning-rates', nargs='+', type=float)
    p.add_argument('--start-steps', nargs='+', type=int, default=[0, 100])
    p.add_argument('--starts', type=int, default=2)
    p.add_argument('--futures', type=int, default=4)
    p.add_argument('--probe-batches', type=int, default=16)
    p.add_argument('--panel-batches', type=int, default=4)
    args = p.parse_args()
    if (args.lr is not None and args.learning_rates is not None) or (args.severity is not None and args.severities is not None):
        p.error('Use either scalar or list parameters for learning rate and severity')
    args.learning_rates = args.learning_rates if args.learning_rates is not None else ([args.lr] if args.lr is not None else [.001,.01])
    args.severities = args.severities if args.severities is not None else ([args.severity] if args.severity is not None else [5])
    args.lr, args.severity = args.learning_rates[0], args.severities[0]
    if min(args.starts, args.futures, args.probe_batches, args.panel_batches) < 2:
        p.error('Repetition counts must be >=2')
    if min(args.start_steps) < 0 or min(args.learning_rates) <= 0 or any(s not in range(1,6) for s in args.severities):
        p.error('Invalid starts, learning rates or severities')
    if len(set(args.start_steps)) != len(args.start_steps):
        p.error('Start steps must be unique')
    if args.output.exists():
        raise FileExistsError(args.output)
    dyn = Dynamics(args)
    args.output.mkdir(parents=True)
    protocol = {k: str(v) if hasattr(v, '__fspath__') else v for k,v in vars(args).items()}
    protocol.update(complete=False, features=FEATURES, task_count=0,
        parameter_names=dyn.names,
        event='max over future steps 1..H of panel error minus panel error at fixed start > epsilon',
        information='Unlabelled state signals and independent complete-batch updates; no risk labels in features',
        risk='Full official 10000-image block; fresh fixed evaluation partition per future',
        streams='SeedSequence keyed by task and purpose; warmup, probes, futures independent')
    save_json(args.output/'protocol.json', protocol)
    task = 0
    for corruption in args.corruptions:
        for severity in args.severities:
            args.severity = severity
            images, labels = dyn.load(corruption)
            for lr in args.learning_rates:
                args.lr = lr
                for start_step in args.start_steps:
                    for rep in range(args.starts):
                        warm, cal, *future_streams = np.random.SeedSequence([args.seed, task]).spawn(2+args.futures)
                        warm_rng, cal_rng = np.random.default_rng(warm), np.random.default_rng(cal)
                        state = dyn.origin.copy()
                        for _ in range(start_step):
                            dyn.set_vector(state)
                            state = state+dyn.update(images, warm_rng)
                        # Features frozen before ANY labelled future evaluation.
                        begin = time.perf_counter()
                        features = np.concatenate([[lr, start_step, args.horizon, args.epsilon],
                            probe(dyn, images, state, cal_rng, args.probe_batches, args.panel_batches)])
                        probe_seconds = time.perf_counter()-begin
                        begin = time.perf_counter()
                        curves, initial_risks = [], []
                        for stream in future_streams:
                            rng = np.random.default_rng(stream)
                            dyn.set_vector(state)
                            order = rng.permutation(len(images))
                            risk = [dyn.risk(images, labels, order)]
                            initial_risks.append(risk[0])
                            current = state.copy()
                            for _ in range(args.horizon):
                                dyn.set_vector(current)
                                current = current+dyn.update(images, rng)
                                dyn.set_vector(current)
                                risk.append(dyn.risk(images, labels, order))
                            curves.append(np.asarray(risk)-risk[0])
                        curves = np.asarray(curves)
                        events = np.max(curves[:, 1:], axis=1)>args.epsilon
                        meta = dict(corruption=corruption, severity=severity, lr=lr,
                                    start_step=start_step, repetition=rep, task=task,
                                    probe_seconds=probe_seconds,
                                    future_seconds=time.perf_counter()-begin)
                        np.savez_compressed(args.output/f'task_{task:04d}.npz', features=features,
                            events=events, risk_changes=curves, initial_risks=initial_risks, start_state=state,
                            metadata=json.dumps(meta))
                        print(meta, 'events', int(events.sum()), '/', len(events), flush=True)
                        task += 1
    protocol.update(complete=True, task_count=task)
    save_json(args.output/'protocol.json', protocol)


if __name__ == '__main__':
    main()

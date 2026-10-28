"""Independent-path stopping experiment on the fixed visual empirical pool.

Forecasts are asymptotic plug-ins, NOT finite-sample safety certificates.
Target risk endpoints are explicitly oracle; source endpoints are a transfer
ablation. All rejected forecasts freeze at initialization and remain in scores.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from entropy_visual import CACHE, BASE, OUT, Marginal, project_pair, class_pair
from visual_calibration import UnlabeledField
from field_geometry import conjugate_prediction
from entropy_prediction import joint_prediction


def safe_horizon(cdf, alpha):
    """Last safe prefix; an unavailable or initially excessive forecast gives 0."""
    bad = np.flatnonzero(np.asarray(cdf) > alpha)
    return max(0, int(bad[0])-1) if len(bad) else len(cdf)-1


def trajectories(target, x0, n, paths, steps, eta, w, rng):
    states = np.empty((steps+1, paths))
    states[0] = x0
    for t in range(steps):
        pos = target.pos[rng.integers(len(target.pos), size=(paths, n))]
        neg = target.neg[rng.integers(len(target.neg), size=(paths, n))]
        z = np.where(rng.random((paths, n)) < target.p, pos, neg)
        u = w*(z-states[t, :, None])
        states[t+1] = states[t]-eta*np.mean(w*u*expit(u)*expit(-u), axis=1)
    if not np.isfinite(states).all():
        raise ValueError('Nonfinite SGD states')
    return states


def disagreement(target, states, x0):
    def cdf(x):
        return (target.p*np.searchsorted(target.pos, x, side='left')/len(target.pos)
                +(1-target.p)*np.searchsorted(target.neg, x, side='left')/len(target.neg))
    return abs(cdf(states)-cdf(x0))


def evaluate(risks, stops, epsilon):
    stops = np.broadcast_to(np.asarray(stops, dtype=int), (risks.shape[1],))
    ids = np.arange(risks.shape[1])
    final = risks[stops, ids]
    baseline = risks[0]
    unsafe = risks > baseline[None, :]+epsilon
    ever = np.maximum.accumulate(unsafe, axis=0)[stops, ids]
    best = risks.min(axis=0)
    return dict(path_violation=float(ever.mean()),
                final_violation=float(np.mean(final > baseline+epsilon)),
                gain=float(np.mean(baseline-final)),
                hindsight_gain_gap=float(np.mean(final-best)),
                mean_steps=float(stops.mean()), frozen_fraction=float(np.mean(stops == 0)))


def forecast_row(risks, geometry, n, alpha, epsilon, steps):
    if geometry is None:
        return dict(horizon=0, forecast_available=False,
                    **evaluate(risks, 0, epsilon))
    left, right = joint_prediction(np.arange(steps+1), n, geometry)
    h = safe_horizon(left+right, alpha)
    return dict(horizon=h, forecast_available=True, predicted_exit=float(left[h]+right[h]),
                **evaluate(risks, h, epsilon))


def run(args):
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    if min(args.paths, args.steps, args.repeats, *args.budgets) <= 0:
        raise ValueError('Positive paths, steps, repeats and budgets required')
    if any(not 0 < a < 1 for a in args.alphas):
        raise ValueError('Alpha must be between zero and one')
    original = json.loads(args.results.read_text(encoding='utf-8'))
    diagnosis = json.loads(args.diagnosis.read_text(encoding='utf-8'))
    protocol = original['protocol']
    eta, w, epsilon = [protocol[k] for k in ['eta', 'w', 'epsilon']]
    # Existing shared field routines use the visual protocol constants.
    if eta != .1 or w != 2.:
        raise ValueError('Shared field routines require eta=.1 and w=2')
    selected = [r for r in original['cases'] if r['mode'] == 'threshold'
                and tuple(r['pair']) in args.pairs and r['n'] in args.batches
                and r['prior'] in args.priors and r['offset'] in args.offsets]
    if not selected:
        raise ValueError('No matching cases')
    with np.load(args.features, allow_pickle=False) as cache:
        train, ty, test, vy = [cache[k] for k in
            ['train_features', 'train_labels', 'test_features', 'test_labels']]
    args.output.mkdir(parents=True)
    output, fields, arrays = [], {}, {}
    for row in selected:
        key, pair, prior, n, x0 = row['key'], tuple(row['pair']), row['prior'], row['n'], -row['delta']
        group = (pair, prior)
        if group not in fields:
            pp, nn, ep, en, _ = project_pair(train, ty, test, vy, pair)
            target = Marginal(ep, en, prior)
            pool = np.r_[target.pos, target.neg]
            weights = np.r_[np.full(len(target.pos), prior/len(target.pos)),
                            np.full(len(target.neg), (1-prior)/len(target.neg))]
            calibrated = []
            for repeat in range(args.repeats):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [args.seed, 1, *pair, int(round(prior*1000)), repeat]))
                counts, previous = np.zeros(len(pool), dtype=int), 0
                for budget in sorted(set(args.budgets)):
                    counts += rng.multinomial(budget-previous, weights)
                    previous = budget
                    field = UnlabeledField(pool, counts.copy())
                    try:
                        calibration, error = field.calibrate(), None
                    except ValueError as exc:
                        calibration, error = None, str(exc)
                    calibrated.append((budget, repeat, field, calibration, error))
            fields[group] = target, calibrated
        target, calibrated = fields[group]
        old = next(r for r in diagnosis['cases'] if r['key'] == key)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0, int(key.split('_')[1])]))
        states = trajectories(target, x0, n, args.paths, args.steps, eta, w, rng)
        risks = target.risk(states)
        arrays[key+'_states'], arrays[key+'_risks'] = states, risks
        records = []
        def add(method, parameter, stops, **extra):
            records.append(dict(method=method, parameter=parameter,
                                **evaluate(risks, stops, epsilon), **extra))
        add('frozen', 0, 0)
        for h in sorted(set([min(args.steps, h) for h in args.fixed_steps]+[args.steps])):
            add('fixed', h, h)
        # Hindsight reference can have crossed the boundary before its minimum.
        add('oracle_checkpoint', 0, risks.argmin(axis=0))
        # Exact unlabeled pool gate: inspect candidate before deploying it;
        # reject the first candidate exceeding epsilon and freeze previous state.
        bad = disagreement(target, states, x0) > epsilon
        first = np.argmax(bad, axis=0)
        gate = np.where(bad.any(axis=0), np.maximum(0, first-1), args.steps)
        add('disagreement_gate', epsilon, gate,
            candidate_rejection_rate=float(np.mean(bad.any(axis=0))))
        oracle = old.get('predictions', {}).get('target_conjugate', {}).get('geometry')
        for alpha in args.alphas:
            records.append(dict(method='oracle_forecast', parameter=alpha,
                **forecast_row(risks, oracle, n, alpha, epsilon, args.steps)))
        for budget, repeat, field, calibration, error in calibrated:
            for endpoint in ['source', 'oracle']:
                boundary = row['diagnostic'] if endpoint == 'source' else old.get('target_risk', {})
                geometry, reason = None, error
                if calibration is not None:
                    try:
                        lo, hi = boundary['left_boundary'], boundary['right_boundary']
                        predictions, _ = conjugate_prediction(field, calibration, n, x0, lo, hi, eta)
                        geometry = predictions['conjugate']
                    except (ValueError, OverflowError, KeyError) as exc:
                        reason = str(exc)
                for alpha in args.alphas:
                    records.append(dict(method='calibrated_'+endpoint, parameter=alpha,
                        budget=budget, repeat=repeat, unavailable=reason,
                        **forecast_row(risks, geometry, n, alpha, epsilon, args.steps)))
        output.append(dict(key=key, pair=list(pair), prior=prior, n=n,
                           offset=row['offset'], x0=x0, frozen_risk=float(risks[0, 0]),
                           unstopped_exit_rate=float(np.mean((risks > risks[0]+epsilon).any(axis=0))),
                           records=records))
        print(f'{key} pair={pair} p={prior} n={n} offset={row["offset"]}: done', flush=True)
        # Save every completed case; interruptions leave a reviewable partial run.
        result = dict(protocol=dict(features=str(args.features), results=str(args.results),
            diagnosis=str(args.diagnosis), paths=args.paths, steps=args.steps, seed=args.seed,
            eta=eta, w=w, epsilon=epsilon, alphas=args.alphas, budgets=args.budgets,
            repeats=args.repeats, fixed_steps=args.fixed_steps, selected_cases=[r['key'] for r in selected],
            complete=len(output) == len(selected),
            interpretation='Conditional finite-pool pilot, no population or finite-sample forecast guarantee',
            independence='New SGD seed; calibration independent conditional on pool; shared across batch/offset',
            endpoints='oracle uses target labels; source transfers original labeled train endpoints without a guarantee',
            fallback='Unavailable forecast freezes at step zero; all selected cases retained',
            gate='Exact unlabeled marginal of target pool; candidate checked before deployment; not a sample-based estimator',
            oracle_checkpoint='Hindsight minimum over full budget, not path-safe',
            monitoring='Published risk-monitoring baseline not implemented; disagreement gate is not that method'), cases=output)
        (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        np.savez_compressed(args.output/'paths.npz', **arrays)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=CACHE)
    parser.add_argument('--results', type=Path, default=OUT/'results.json')
    parser.add_argument('--diagnosis', type=Path, default=OUT/'diagnosis/results.json')
    parser.add_argument('--output', type=Path, default=BASE/'stopping_pilot')
    parser.add_argument('--pairs', nargs='+', type=class_pair, default=[(3,5),(1,9),(4,7),(0,8)])
    parser.add_argument('--batches', nargs='+', type=int, default=[128])
    parser.add_argument('--priors', nargs='+', type=float, default=[.5])
    parser.add_argument('--offsets', nargs='+', type=float, default=[-2,0,2])
    parser.add_argument('--paths', type=int, default=128)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--alphas', nargs='+', type=float, default=[.01,.05,.1,.2])
    parser.add_argument('--budgets', nargs='+', type=int, default=[2048,131072])
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--fixed-steps', nargs='+', type=int, default=[10,25,50,100,200,400,800])
    parser.add_argument('--seed', type=int, default=20261029)
    run(parser.parse_args())

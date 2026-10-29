"""Target-centered calibration-scale test; conditional fixed-pool mechanism study.

Tests the empirical-root LOCAL direction predictor, asymptotically equivalent
to the conjugacy predictor under E.7. Does not test the full time-CDF law or
establish an information-theoretic lower bound. Target centering and risk
endpoints are oracle experimental controls, not an unlabeled deployment rule.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, ndtr
from scipy.optimize import brentq
from entropy_visual import CACHE, BASE, Marginal, project_pair, class_pair
from field_geometry import calibrate_peak
from visual_calibration import UnlabeledField


def empirical_roots(pool, probabilities, budgets, repeats, seed):
    """Nested iid calibration prefixes; scan all roots on the usual 513 grid."""
    mesh = np.linspace(pool.min()-2, pool.max()+2, 513)
    u = 2*(pool[:, None]-mesh[None, :])
    gradient = -2*u*expit(u)*expit(-u)
    rng = np.random.default_rng(seed)
    counts = np.zeros((repeats, len(pool)), dtype=np.int64)
    previous, output = 0, {}
    for budget in sorted(set(budgets)):
        counts += rng.multinomial(budget-previous, probabilities, size=repeats)
        previous = budget
        velocities = (counts/budget)@gradient
        rows = []
        for repeat, velocity in enumerate(velocities):
            intervals = np.flatnonzero((velocity[:-1] <= 0) & (velocity[1:] > 0))
            row = dict(repeat=repeat, budget=budget, roots_found=len(intervals))
            if len(intervals) != 1:
                row['unavailable'] = 'Expected exactly one scanned unstable root'
            else:
                field = UnlabeledField(pool, counts[repeat])
                i = intervals[0]
                q = brentq(lambda x:field.moments(x)[0], mesh[i], mesh[i+1], xtol=1e-12)
                _, lam, variance = field.moments(q)
                if lam <= 0 or variance <= 0:
                    row['unavailable'] = 'Nonpositive curvature or root noise'
                else:
                    row.update(q=float(q), lambda_=lam, variance=variance)
            rows.append(row)
        output[budget] = rows
    return output


def centered_sgd(target, q, lo, hi, n, paths, steps, eta, epsilon, seed):
    """Actual entropy SGD; separately track component and actual risk exits."""
    rng = np.random.default_rng(seed)
    x = np.full(paths, q)
    times = np.full(paths, -1, dtype=int)
    sides = np.zeros(paths, dtype=int)
    risk_times, risk_sides = times.copy(), sides.copy()
    risk0 = target.risk(q)
    for t in range(1, steps+1):
        ids = np.flatnonzero((times < 0) | (risk_times < 0))
        if not len(ids):
            break
        shape = (len(ids), n)
        pos = target.pos[rng.integers(len(target.pos), size=shape)]
        neg = target.neg[rng.integers(len(target.neg), size=shape)]
        z = np.where(rng.random(shape) < target.p, pos, neg)
        u = 2*(z-x[ids, None])
        x[ids] -= eta*np.mean(2*u*expit(u)*expit(-u), axis=1)
        if not np.isfinite(x[ids]).all():
            raise ValueError('Nonfinite centered SGD states')
        # Empirical risk plateaus use (lo, hi], matching score tie convention.
        crossed = ids[(times[ids] < 0) & ((x[ids] <= lo) | (x[ids] > hi))]
        times[crossed], sides[crossed] = t, np.where(x[crossed] > q, 1, -1)
        crossed = ids[(risk_times[ids] < 0) & (target.risk(x[ids]) > risk0+epsilon)]
        risk_times[crossed], risk_sides[crossed] = t, np.where(x[crossed] > q, 1, -1)
    return dict(times=times, sides=sides, risk_times=risk_times, risk_sides=risk_sides)


def run(args):
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    if min(args.repeats, args.paths, args.steps, *args.batches, *args.ratios) <= 0:
        raise ValueError('Positive experiment budgets required')
    if not 0 < args.prior < 1 or args.epsilon <= 0 or not 0 < args.eta < 1:
        raise ValueError('Require interior prior, positive tolerance and eta in (0,1)')
    with np.load(args.features, allow_pickle=False) as cache:
        train, ty, test, vy = [cache[k] for k in
            ['train_features','train_labels','test_features','test_labels']]
    args.output.mkdir(parents=True)
    groups, samples = [], {}
    protocol = dict(features=str(args.features), pairs=[list(p) for p in args.pairs],
        batches=args.batches, ratios=args.ratios, eta=args.eta, prior=args.prior,
        epsilon=args.epsilon, repeats=args.repeats, paths=args.paths, steps=args.steps,
        seed=args.seed, initialization='x0=q from exact target empirical field (oracle)',
        predictor='Local empirical-root Phi(sqrt(n)*(q-qhat)/shat), not full conjugacy',
        calibration='Nested multinomial prefixes, independent of SGD conditional on fixed official test pool',
        limits='Fixed eta, fixed pool; no new images, population certificate or universal lower bound',
        root_search='513-node full-range scan plus Brent; numerical diagnostic, not interval certification',
        families='fixed_R: M proportional to n; vanishing_R: M proportional to n squared, R=1 at minimum n')
    for pair in args.pairs:
        _, _, ep, en, _ = project_pair(train, ty, test, vy, pair)
        target = Marginal(ep, en, args.prior)
        true = calibrate_peak(target, min(ep.min(),en.min()), max(ep.max(),en.max()), args.eta)
        q, lam, sd = true['q'], true['lambda_'], true['seed_sd']
        factor = (2+args.eta*lam)/(args.eta*lam)
        settings = []
        for n in args.batches:
            for r in args.ratios:
                settings.append(dict(family='fixed_R', target_ratio=r, n=n,
                                     budget=max(2, int(round(n*factor/r**2)))))
            settings.append(dict(family='vanishing_R', target_ratio=float(np.sqrt(min(args.batches)/n)),
                n=n, budget=max(2, int(round(n*factor*n/min(args.batches))))))
        pool = np.r_[target.pos, target.neg]
        probabilities = np.r_[np.full(len(target.pos),target.p/len(target.pos)),
                              np.full(len(target.neg),(1-target.p)/len(target.neg))]
        calibration = empirical_roots(pool, probabilities, [s['budget'] for s in settings],
            args.repeats, np.random.SeedSequence([args.seed, 1, *pair]))
        group = dict(pair=list(pair), true=true, settings=settings,
                     calibrations={str(k):v for k,v in calibration.items()}, sgd=[])
        for setting in settings:
            n, budget = setting['n'], setting['budget']
            setting['actual_ratio'] = float(np.sqrt(n/budget*factor))
            values, root_errors = [], []
            for row in calibration[budget]:
                if 'q' not in row:
                    continue
                A = 1+args.eta*row['lambda_']
                shat = np.sqrt(args.eta**2*row['variance']/(A*A-1))
                values.append(float(ndtr(np.sqrt(n)*(q-row['q'])/shat)))
                root_errors.append(row['q']-q)
            setting['predictions'] = values
            setting['root_errors'] = root_errors
            setting['success_rate'] = len(values)/args.repeats
            setting['theory_mae'] = float(np.arctan(setting['actual_ratio'])/np.pi)
        try:
            lo, hi, components = target.safe_component(q, args.epsilon)
            group['risk_interval'] = dict(lo=lo, hi=hi, components=components)
        except ValueError as exc:
            group['sgd_unavailable'] = str(exc)
        else:
            for n in args.batches:
                sample = centered_sgd(target, q, lo, hi, n, args.paths, args.steps,
                    args.eta, args.epsilon, np.random.SeedSequence([args.seed, 0, *pair, n]))
                key = f'pair_{pair[0]}_{pair[1]}_n{n}'
                samples.update({key+'_'+k:v for k,v in sample.items()})
                record = dict(n=n, key=key)
                for prefix, times, sides in [('component',sample['times'],sample['sides']),
                                             ('risk',sample['risk_times'],sample['risk_sides'])]:
                    right = int(np.sum((times >= 0)&(sides > 0)))
                    censored = int(np.sum(times < 0))
                    record[prefix] = dict(right=right, censored=censored, paths=args.paths,
                        right_probability_bounds=[right/args.paths,(right+censored)/args.paths])
                record['different_event_paths'] = int(np.sum(sample['times'] != sample['risk_times']))
                group['sgd'].append(record)
                print(f'pair={pair}, n={n}: SGD complete', flush=True)
        groups.append(group)
        result = dict(protocol=dict(protocol, complete=len(groups)==len(args.pairs)), groups=groups)
        (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        np.savez_compressed(args.output/'samples.npz', **samples)
        print(f'pair={pair}: calibration and scores saved', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=CACHE)
    parser.add_argument('--output', type=Path, default=BASE/'centered_calibration')
    parser.add_argument('--pairs', nargs='+', type=class_pair, default=[(3,5),(1,9),(4,7),(0,8)])
    parser.add_argument('--batches', nargs='+', type=int, default=[32,128,512])
    parser.add_argument('--ratios', nargs='+', type=float, default=[.5,1.,2.])
    parser.add_argument('--eta', type=float, default=.1)
    parser.add_argument('--prior', type=float, default=.5)
    parser.add_argument('--epsilon', type=float, default=.02)
    parser.add_argument('--repeats', type=int, default=128)
    parser.add_argument('--paths', type=int, default=256)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=20261031)
    run(parser.parse_args())

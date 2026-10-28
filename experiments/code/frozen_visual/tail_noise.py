"""Controlled noise ablations for finite-pool long-time exits.

Only real_sgd is the original algorithm. Other modes keep population drift but
alter centered noise, and are mechanism diagnostics, not proposed TTA methods.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import numpy as np
from scipy.special import expit
from scipy.interpolate import PchipInterpolator
from entropy_visual import BASE, CACHE, ETA, W, Marginal, project_pair, class_pair
from visual_calibration import UnlabeledField
from exit_probe import component


def population_diagnostic(target, x0, n, lo, hi, q, radius, steps):
    trace = []
    x, variance = x0, 0.
    for t in range(steps+1):
        v, derivative, noise_var = target.moments(x)
        trace.append([t, x, v, derivative, noise_var, variance])
        if not lo <= x <= hi:
            break
        variance = (1+ETA*derivative)**2*variance+ETA**2*noise_var/n
        x += ETA*v
    gain, total, far = 1., 0., 0.
    for _, x, _, derivative, noise_var, _ in trace[-2::-1]:
        contribution = ETA**2*noise_var/n*gain**2
        total += contribution
        if abs(x-q) > radius:
            far += contribution
        gain *= 1+ETA*derivative
    return dict(population_steps=len(trace)-1, population_exited=bool(not lo <= trace[-1][1] <= hi),
        linearized_time_sd=float(np.sqrt(variance)/max(abs(ETA*trace[-1][2]), 1e-15)),
        far_variance_fraction=float(far/total) if total else None,
        scope='First-order population-orbit diagnostic; invalid as a full exit law near branch switching'), np.asarray(trace)


def calculate(task):
    pos, neg, row, paths, steps, radius, seed, points = task
    target = Marginal(pos, neg, row['prior'])
    q = target.calibrate()['q']
    x0, n, epsilon = -row['delta'], row['n'], row['epsilon']
    knots, first, last = component(target, x0, epsilon)
    lo, hi = knots[first-1], knots[last]
    pool = np.r_[pos, neg]
    weights = np.r_[np.full(len(pos), target.p/len(pos)), np.full(len(neg), (1-target.p)/len(neg))]
    field = UnlabeledField(pool, weights)
    mesh = np.unique(np.r_[np.linspace(min(lo, pool.min())-2, max(hi, pool.max())+2, points), q])
    drift = PchipInterpolator(mesh, field.velocities(mesh), extrapolate=False)
    probes = (mesh[:-1]+mesh[1:])/2
    error = float(np.max(abs(drift(probes)-field.velocities(probes))))
    diagnostic, trace = population_diagnostic(target, x0, n, lo, hi, q, radius, steps)
    summary = dict(key=row['key'], n=n, offset=row['offset'], pair=row['pair'], prior=row['prior'],
        q=q, x0=x0, radius=radius, drift_midpoint_max_error=error, diagnostic=diagnostic, modes={})
    arrays = {'population_trace': trace}
    for mode in ['real_sgd', 'frozen_root', 'near_noise', 'far_noise']:
        rng = np.random.default_rng(seed)
        x = np.full(paths, x0)
        times = np.full(paths, -1, dtype=np.int32)
        directions = np.zeros(paths, dtype=np.int8)
        component_times = times.copy()
        skipped = np.zeros(paths, dtype=bool)
        limit = target.risk(x0)+epsilon
        for t in range(1, steps+1):
            # Full arrays preserve common random batches across all four modes.
            zp = pos[rng.integers(len(pos), size=(paths, n))]
            zn = neg[rng.integers(len(neg), size=(paths, n))]
            z = np.where(rng.random((paths, n)) < target.p, zp, zn)
            active = times < 0
            if not active.any():
                break
            xx, zz = x[active], z[active]
            if mode == 'frozen_root':
                u = W*(zz-q)
                noise = np.mean(-W*u*expit(u)*expit(-u), axis=1)-target.moments(q)[0]
                speed = drift(xx)+noise
            else:
                u = W*(zz-xx[:, None])
                sampled = np.mean(-W*u*expit(u)*expit(-u), axis=1)
                if mode == 'real_sgd':
                    speed = sampled
                else:
                    gate = abs(xx-q) <= radius
                    if mode == 'far_noise':
                        gate = ~gate
                    speed = drift(xx)+gate*(sampled-drift(xx))
            x[active] += ETA*speed
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite state or drift interpolation outside component')
            risk_safe = target.risk(x) <= limit
            index = np.searchsorted(knots, x, side='left')
            escaped = (component_times < 0) & ((index < first) | (index > last))
            component_times[escaped] = t
            skipped[escaped] = risk_safe[escaped]
            unsafe = (times < 0) & ~risk_safe
            times[unsafe], directions[unsafe] = t, np.where(x[unsafe] > q, 1, -1)
        ordered = np.sort(np.where(times < 0, np.inf, times))
        quantiles = [ordered[int(np.ceil(p*paths))-1] for p in [.1, .5, .9]]
        summary['modes'][mode] = dict(censored=int(np.sum(times < 0)), skip_count=int(skipped.sum()),
            quantiles=[int(v) if np.isfinite(v) else None for v in quantiles],
            right_by_horizon=float(np.mean(directions == 1)))
        for name, value in dict(times=times, directions=directions, component_times=component_times, skipped=skipped).items():
            arrays[mode+'_'+name] = value
    return summary, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, default=CACHE)
    p.add_argument('--results', type=Path, default=BASE/'visual_shift/results.json')
    p.add_argument('--output', type=Path, default=BASE/'tail_noise')
    p.add_argument('--pairs', type=class_pair, nargs='+', default=[(1, 9)])
    p.add_argument('--priors', type=float, nargs='+', default=[.5])
    p.add_argument('--batches', type=int, nargs='+', default=[32, 128, 512])
    p.add_argument('--offsets', type=float, nargs='+', default=[-2., 0., 2.])
    p.add_argument('--paths', type=int, default=600)
    p.add_argument('--max-steps', type=int, default=4000)
    p.add_argument('--radius', type=float, default=.5)
    p.add_argument('--grid-points', type=int, default=8193)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261030)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if min(a.paths, a.max_steps, a.workers, *a.batches) <= 0 or a.radius <= 0 or a.grid_points < 65:
        raise ValueError('Positive budgets/radius and at least 65 grid points required')
    if a.output.exists() and not a.resume:
        raise FileExistsError('Choose a new directory or use --resume')
    old = json.loads(a.results.read_text(encoding='utf-8'))
    if old['protocol']['modes'] != ['threshold'] or old['protocol']['eta'] != ETA or old['protocol']['w'] != W:
        raise ValueError('Use matching fixed-slope threshold results')
    selected = [dict(r, epsilon=old['protocol']['epsilon']) for r in old['cases']
        if tuple(r['pair']) in {tuple(v) for v in a.pairs} and r['prior'] in a.priors
        and r['n'] in a.batches and r['offset'] in a.offsets]
    if not selected:
        raise ValueError('No selected settings')
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()
                if k not in ['resume', 'workers', 'output']}
    if a.resume:
        if json.loads((a.output/'settings.json').read_text()) != json.loads(json.dumps(settings)):
            raise ValueError('Resume requires the same settings except workers')
    else:
        a.output.mkdir(parents=True)
        (a.output/'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    with np.load(a.features, allow_pickle=False) as f:
        features = [f[k] for k in ['train_features','train_labels','test_features','test_labels']]
    tasks, cache = [], {}
    for row in selected:
        pair = tuple(row['pair'])
        if pair not in cache:
            _, _, pos, neg, _ = project_pair(*features, pair)
            cache[pair] = pos, neg
        if not (a.output/(row['key']+'.json')).exists():
            tasks.append((*cache[pair], row, a.paths, a.max_steps, a.radius,
                np.random.SeedSequence([a.seed, int(row['key'].split('_')[-1])]), a.grid_points))
    executor = ProcessPoolExecutor(max_workers=a.workers) if a.workers > 1 and tasks else None
    try:
        completed = executor.map(calculate, tasks) if executor else map(calculate, tasks)
        for summary, arrays in completed:
            key = summary['key']
            temporary = a.output/(key+'.tmp.npz')
            np.savez_compressed(temporary, **arrays)
            temporary.replace(a.output/(key+'.npz'))
            temporary = a.output/(key+'.tmp.json')
            temporary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
            temporary.replace(a.output/(key+'.json'))
            print('saved', key, flush=True)
    finally:
        if executor:
            executor.shutdown(cancel_futures=True)
    cases = [json.loads((a.output/(r['key']+'.json')).read_text()) for r in selected]
    (a.output/'results.json').write_text(json.dumps(dict(protocol=settings, cases=cases), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

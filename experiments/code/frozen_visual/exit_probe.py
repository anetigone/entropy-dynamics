"""Finite-pool interval geometry and fresh threshold-SGD component/risk exits.

Labels define evaluation only. Grid checks are numerical diagnostics, not
certificates. Independent paths are necessary: old exit times cannot recover
component skips. This does not fit any prediction to the new exits.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.special import expit
from entropy_visual import CACHE, BASE, Marginal, project_pair, ETA, W, class_pair


def component(target, x0, epsilon):
    knots = np.unique(np.r_[target.pos, target.neg, x0])
    probes = np.r_[np.nextafter(knots[0], -np.inf), np.nextafter(knots, np.inf)]
    safe = target.risk(probes) <= target.risk(x0)+epsilon
    first = last = int(np.searchsorted(knots, x0, side='left'))
    while first > 0 and safe[first-1]:
        first -= 1
    while last+1 < len(safe) and safe[last+1]:
        last += 1
    if first == 0 or last == len(safe)-1:
        raise ValueError('Initial safe component is unbounded')
    return knots, first, last


def probe(target, x0, n, paths, steps, epsilon, seed, points, geometry_only=False):
    knots, first, last = component(target, x0, epsilon)
    lo, hi = float(knots[first-1]), float(knots[last])
    cal = target.calibrate()
    q = cal['q']
    jump = ETA*W/4
    mesh = np.unique(np.r_[np.linspace(lo-2*jump, hi+2*jump, points), q])
    moments = np.array([target.moments(x) for x in mesh])
    away = abs(mesh-q) > 1e-8
    slopes = moments[away, 0]/(mesh[away]-q)
    diagnostic = dict(left=lo, right=hi, q=q, lambda_=cal['lambda_'],
        root_variance=target.moments(q)[2], eta_L=ETA*W*W/4,
        sampled_outward=bool(np.all(slopes > 0)),
        minimum_sampled_secant=float(slopes.min()),
        minimum_sampled_map_derivative=float((1+ETA*moments[:, 1]).min()),
        peak_inside=bool(lo < q < hi), root_boundary_distance=float(min(q-lo, hi-q)),
        domain=[float(mesh[0]), float(mesh[-1])], overshoot_margin=jump,
        normalized_initialization=float(np.sqrt(n)*(x0-q)/cal['seed_sd']),
        grid_points=len(mesh), endpoint_membership='Plateau indices from searchsorted(side=left)',
        scope='Finite grid diagnostic only; not interval certification or proof of an asymptotic initialization sequence')
    if geometry_only:
        return dict(geometry=diagnostic), dict(geometry_x=mesh, geometry_velocity=moments[:, 0])
    rng = np.random.default_rng(seed)
    x = np.full(paths, x0, dtype=float)
    ct = np.full(paths, -1, dtype=np.int32)
    rt = ct.copy()
    cd = np.zeros(paths, dtype=np.int8)
    rd = cd.copy()
    skipped = np.zeros(paths, dtype=bool)
    first_component_state = np.full(paths, np.nan)
    first_risk_state = first_component_state.copy()
    risk_limit = target.risk(x0)+epsilon
    for t in range(1, steps+1):
        # Keep every path alive after component exit to measure true risk exit.
        zp = target.pos[rng.integers(len(target.pos), size=(paths, n))]
        zn = target.neg[rng.integers(len(target.neg), size=(paths, n))]
        z = np.where(rng.random((paths, n)) < target.p, zp, zn)
        u = W*(z-x[:, None])
        x -= ETA*np.mean(W*u*expit(u)*expit(-u), axis=1)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite threshold in probe')
        index = np.searchsorted(knots, x, side='left')
        safe = target.risk(x) <= risk_limit
        out = (ct < 0) & ((index < first) | (index > last))
        ct[out], cd[out] = t, np.where(x[out] > q, 1, -1)
        skipped[out] = safe[out]
        first_component_state[out] = x[out]
        unsafe = (rt < 0) & ~safe
        rt[unsafe], rd[unsafe] = t, np.where(x[unsafe] > q, 1, -1)
        first_risk_state[unsafe] = x[unsafe]
    grid = np.arange(steps+1)
    def cdf(times, direction, side):
        return np.cumsum(np.bincount(times[(times >= 0) & (direction == side)], minlength=steps+1))/paths
    differences = [np.max(abs(cdf(ct, cd, side)-cdf(rt, rd, side))) for side in [-1, 1]]
    skip_fraction = float(skipped.mean())
    total_c = np.cumsum(np.bincount(ct[ct >= 0], minlength=steps+1))/paths
    total_r = np.cumsum(np.bincount(rt[rt >= 0], minlength=steps+1))/paths
    error = float(np.max(abs(total_c-total_r)))
    if max(error, *differences) > skip_fraction+1e-12:
        raise ValueError('Component/risk coupling check failed')
    summary = dict(geometry=diagnostic, skip_count=int(skipped.sum()), skip_fraction=skip_fraction,
        total_cdf_discrepancy=error, joint_cdf_discrepancy=float(max(differences)),
        component_censored=int(np.sum(ct < 0)), risk_censored=int(np.sum(rt < 0)),
        skip_mc_se=float(np.sqrt(skip_fraction*(1-skip_fraction)/paths)))
    arrays = dict(component_times=ct, risk_times=rt, component_directions=cd, risk_directions=rd,
        skipped=skipped, component_state=first_component_state, risk_state=first_risk_state,
        grid=grid, component_cdf=total_c, risk_cdf=total_r,
        geometry_x=mesh, geometry_velocity=moments[:, 0])
    return summary, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, default=CACHE)
    p.add_argument('--results', type=Path, default=BASE/'visual_shift/results.json')
    p.add_argument('--output', type=Path, default=BASE/'exit_probe')
    p.add_argument('--pairs', type=class_pair, nargs='+', default=[[1, 9]])
    p.add_argument('--priors', type=float, nargs='+', default=[.5])
    p.add_argument('--batches', type=int, nargs='+', default=[32, 128, 512])
    p.add_argument('--offsets', type=float, nargs='+', default=[-2., 0., 2.])
    p.add_argument('--paths', type=int, default=600)
    p.add_argument('--max-steps', type=int, default=4000)
    p.add_argument('--grid-points', type=int, default=1025)
    p.add_argument('--seed', type=int, default=20261029)
    p.add_argument('--geometry-only', action='store_true', help='Check domains without sampling SGD paths')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f'Choose a new output directory: {a.output}')
    if min(a.paths, a.max_steps, *a.batches) <= 0 or a.grid_points < 17:
        raise ValueError('Positive budgets and at least 17 grid points required')
    old = json.loads(a.results.read_text(encoding='utf-8'))
    if old['protocol']['modes'] != ['threshold'] or old['protocol']['eta'] != ETA or old['protocol']['w'] != W:
        raise ValueError('Use matching fixed-slope threshold results')
    rows = [r for r in old['cases'] if tuple(r['pair']) in {tuple(pair) for pair in a.pairs} and r['prior'] in a.priors
            and r['n'] in a.batches and r['offset'] in a.offsets]
    if not rows:
        raise ValueError('No saved settings selected')
    with np.load(a.features, allow_pickle=False) as f:
        train, ty, test, vy = [f[k] for k in ['train_features','train_labels','test_features','test_labels']]
    cases, arrays, cache = [], {}, {}
    for row in rows:
        pair = tuple(row['pair'])
        if pair not in cache:
            _, _, pos, neg, _ = project_pair(train, ty, test, vy, pair)
            cache[pair] = pos, neg
        target = Marginal(*cache[pair], row['prior'])
        key = row['key']
        record = dict(key=key, pair=row['pair'], prior=row['prior'], n=row['n'], offset=row['offset'])
        try:
            result, sample = probe(target, -row['delta'], row['n'], a.paths, a.max_steps,
                old['protocol']['epsilon'], np.random.SeedSequence([a.seed, int(key.split('_')[-1])]), a.grid_points,
                geometry_only=a.geometry_only)
            record.update(result)
            arrays.update({key+'_'+k: v for k, v in sample.items()})
        except ValueError as exc:
            record['unavailable'] = str(exc)
        cases.append(record)
        print(key, record.get('skip_fraction', record.get('unavailable')), flush=True)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    protocol.update(eta=ETA, w=W, epsilon=old['protocol']['epsilon'],
        event='Component and risk directions relative to target peak; all paths retained in CDF denominators',
        scope='Fresh independent finite-pool SGD, not new training; geometry grids are diagnostics only')
    a.output.mkdir(parents=True)
    (a.output/'results.json').write_text(json.dumps(dict(protocol=protocol, cases=cases), indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(a.output/'samples.npz', **arrays)


if __name__ == '__main__':
    main()

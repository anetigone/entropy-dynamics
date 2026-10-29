"""Unknown local visual environments, fixed baseline initialization, fresh SGD.

Conditional official-test-pool experiment. Reweighting images changes label risk;
this is not the exact common-risk construction or a population minimax test.
"""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, ndtr

from entropy_visual import CACHE, BASE, PAIRS, Marginal, project_pair, class_pair
from field_geometry import calibrate_peak
from visual_calibration import UnlabeledField

NAMES = ['constant', 'root', 'moment', 'clipped', 'optimized']
RULES = Path(__file__).resolve().parents[3]/'independent-bundles/gaussian_decision.zip'


def velocity(x, z):
    u = 2*(z-x)
    return -2*u*expit(u)*expit(-u)


def load_rules(path):
    if path.suffix.lower() == '.zip':
        with ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith('certificates.json')]
            if len(names) != 1:
                raise ValueError('Expected one certificates.json in rule archive')
            records = json.loads(archive.read(names[0]))
    else:
        records = json.loads(path.read_text(encoding='utf-8-sig'))
    for row in records:
        support = np.asarray(row['support'], float)
        thresholds = np.asarray(row['thresholds'], float)
        if (len(thresholds) != len(support)-1 or not np.isfinite(support).all()
                or not np.isfinite(thresholds).all() or np.any(np.diff(thresholds) <= 0)
                or np.any(np.diff(support) <= 0)):
            raise ValueError('Invalid fixed Gaussian rule')
    return records


def select_rule(rules, window, ratio):
    for row in rules:
        if np.isclose(float(row['K']), window) and np.isclose(float(row['r']), ratio):
            return row
    raise ValueError(f'No frozen rule for K={window:g}, r={ratio:g}; supply --rules')


class WeightedRisk:
    """Exact weighted finite-pool risk, with classifier positive for z >= x."""
    def __init__(self, scores, labels, weights):
        order = np.argsort(scores)
        self.scores = np.asarray(scores)[order]
        labels, weights = np.asarray(labels)[order], np.asarray(weights)[order]
        self.prefix = np.r_[0., np.cumsum(np.where(labels, weights, -weights))]
        self.negative = float(weights[~labels].sum())

    def risk(self, x):
        i = np.searchsorted(self.scores, x, side='left')
        return self.negative+self.prefix[i]

    def component(self, x0, epsilon):
        knots = np.unique(self.scores)
        probes = np.r_[np.nextafter(knots[0], -np.inf), np.nextafter(knots, np.inf)]
        safe = self.risk(probes) <= self.risk(x0)+epsilon
        i = int(np.searchsorted(knots, x0, side='left'))
        left = right = i
        while left >= 0 and safe[left]:
            left -= 1
        while right < len(safe) and safe[right]:
            right += 1
        return dict(lo=None if left < 0 else float(knots[left]),
                    hi=None if right == len(safe) else float(knots[right-1]),
                    components=int(safe[0])+int(np.sum(safe[1:] & ~safe[:-1])))


def simulate(scores, weights, risk, x0, component, n, paths, steps, eta, seed, block=256):
    """Continue after component exit until actual risk exit or fixed cap."""
    rng = np.random.default_rng(seed)
    arrays = {k: np.full(paths, -1, np.int32) for k in ['risk_times', 'component_times']}
    arrays.update({k: np.zeros(paths, np.int8) for k in ['risk_sides', 'component_sides']})
    cdf = np.cumsum(weights)
    cdf[-1] = 1.
    for start in range(0, paths, block):
        stop = min(paths, start+block)
        x = np.full(stop-start, x0)
        rt = arrays['risk_times'][start:stop]
        ct = arrays['component_times'][start:stop]
        rs = arrays['risk_sides'][start:stop]
        cs = arrays['component_sides'][start:stop]
        for step in range(1, steps+1):
            ids = np.flatnonzero(rt < 0)
            if not len(ids):
                break
            index = np.searchsorted(cdf, rng.random((len(ids), n)))
            x[ids] += eta*velocity(x[ids, None], scores[index]).mean(axis=1)
            if not np.isfinite(x[ids]).all():
                raise ValueError('Nonfinite SGD states')
            low = np.zeros(len(ids), bool) if component['lo'] is None else x[ids] <= component['lo']
            high = np.zeros(len(ids), bool) if component['hi'] is None else x[ids] > component['hi']
            crossed = ids[(ct[ids] < 0) & (low | high)]
            ct[crossed] = step
            cs[crossed] = np.where(x[crossed] > x0, 1, -1)
            crossed = ids[risk.risk(x[ids]) > risk.threshold]
            rt[crossed] = step
            rs[crossed] = np.where(x[crossed] > x0, 1, -1)
    return arrays


def predict_counts(scores, counts, baseline, n, eta, window, rule, mesh, gradients):
    """Estimator inputs contain no labels, target root, theta, t or reference data."""
    budget = int(counts.sum())
    weights = counts/budget
    coordinate = np.sqrt(n)*float(weights@baseline['psi'])
    out = [.5, .5, float(ndtr(coordinate)),
           float(ndtr(np.clip(coordinate, -window, window))),
           float(ndtr(float(rule['support'][np.searchsorted(
               np.asarray(rule['thresholds'], float), coordinate)])))]
    v = weights@gradients
    intervals = np.flatnonzero((v[:-1] <= 0) & (v[1:] > 0))
    detail = dict(roots_found=int(len(intervals)), fallback=True)
    if len(intervals) != 1:
        detail['reason'] = 'Expected exactly one scanned unstable root'
        return out, detail
    field = UnlabeledField(scores, counts)
    i = intervals[0]
    try:
        q = brentq(lambda x: field.moments(x)[0], mesh[i], mesh[i+1])
        _, lam, variance = field.moments(q)
        if lam <= 0 or variance <= 0:
            raise ValueError('Nonpositive root curvature or variance')
        A = 1+eta*lam
        sd = np.sqrt(eta**2*variance/(A*A-1))
        out[1] = float(ndtr(np.sqrt(n)*(baseline['x0']-q)/sd))
        detail.update(q=float(q), lambda_=lam, seed_sd=float(sd), fallback=False)
    except (ValueError, RuntimeError) as exc:
        detail['reason'] = str(exc)
    return out, detail


def settings(args):
    rows = []
    for n in args.batches:
        rows.extend(dict(family='fixed', n=n, ratio=r) for r in args.ratios)
        if args.decreasing:
            rows.append(dict(family='decreasing', n=n,
                             ratio=args.decreasing_start*np.sqrt(min(args.batches)/n)))
    return rows


def run(args):
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    numbers = [args.eta, args.epsilon, args.prior, args.window, args.decreasing_start,
               *args.ratios, *args.seeds]
    if not np.isfinite(numbers).all() or not 0 < args.eta < 1 or not 0 < args.prior < 1:
        raise ValueError('Require finite parameters, eta in (0,1), interior prior')
    if min(args.epsilon, args.window, args.repeats, args.paths, args.steps,
           args.block, args.decreasing_start, *args.batches, *args.ratios) <= 0:
        raise ValueError('Positive budgets, ratios, window and tolerance required')
    if max(abs(t) for t in args.seeds) > args.window:
        raise ValueError('Seed grid must lie in the declared window')
    rules = load_rules(args.rules)
    design = settings(args)
    for setting in design:
        select_rule(rules, args.window, setting['ratio'])
    with np.load(args.features, allow_pickle=False) as f:
        train, ty, test, vy = [f[k] for k in
            ['train_features', 'train_labels', 'test_features', 'test_labels']]
    if (train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]
            or len(train) != len(ty) or len(test) != len(vy)
            or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError('Invalid aligned finite feature arrays')
    args.output.mkdir(parents=True)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(complete=False, predictors=NAMES,
        information='Known official-test baseline; fixed x0=baseline root for every environment',
        sampling='Independent iid draws with replacement, conditional on the fixed official test pool',
        calibration='Nested budget prefixes per environment; all rules share each repetition',
        risk='Environment-specific weighted empirical label risk, not shared full risk',
        limits='No new independent images, population minimax guarantee or Gaussian finite-n certificate',
        root_failure='Forecast .5; failures retained in every risk denominator')
    result = dict(protocol=protocol, groups=[], cells=[])

    def save():
        (args.output/'results.json').write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

    for pi, pair in enumerate(args.pairs):
        pp, nn, ep, en, anchor = project_pair(train, ty, test, vy, pair)
        scores = np.r_[ep, en]
        labels = np.r_[np.ones(len(ep), bool), np.zeros(len(en), bool)]
        p0 = np.r_[np.full(len(ep), args.prior/len(ep)),
                   np.full(len(en), (1-args.prior)/len(en))]
        group = dict(pair=list(pair), anchor=anchor)
        result['groups'].append(group)
        try:
            base = calibrate_peak(Marginal(ep, en, args.prior), scores.min(), scores.max(), args.eta)
        except ValueError as exc:
            group['unavailable'] = str(exc)
            save()
            continue
        q0, lam, s0 = base['q'], base['lambda_'], base['seed_sd']
        h = velocity(q0, scores)
        h -= p0@h  # Numerical centering makes the finite-pool tilt normalize exactly.
        sigma = np.sqrt(p0@(h*h))
        if sigma <= 0:
            raise ValueError('Degenerate baseline noise')
        h /= sigma
        c = sigma/(lam*s0)
        baseline = dict(x0=q0, psi=c*h)
        group.update(baseline=base, c=float(c), h_bound=float(abs(h).max()),
                     source_score_range=[float(min(pp.min(), nn.min())), float(max(pp.max(), nn.max()))])
        mesh = np.linspace(scores.min()-2, scores.max()+2, 513)
        gradients = velocity(mesh[None, :], scores[:, None])
        for ni, n in enumerate(args.batches):
            chosen = [s for s in design if s['n'] == n]
            budgets = sorted(set(max(2, int(round(c*c*n/s['ratio']**2))) for s in chosen))
            for ti, t in enumerate(args.seeds):
                theta = t/(c*np.sqrt(n))
                weights = p0*(1+theta*h)
                if weights.min() <= 0:
                    raise ValueError('Tilt exceeds positivity range; increase n or reduce seed window')
                weights /= weights.sum()
                risk = WeightedRisk(scores, labels, weights)
                risk.threshold = float(risk.risk(q0)+args.epsilon)
                component = risk.component(q0, args.epsilon)
                cell = dict(pair=list(pair), n=n, t=t, theta=float(theta), component=component,
                            risk0=float(risk.risk(q0)), threshold=risk.threshold,
                            target_prior=float(weights[labels].sum()), settings=[])
                # Exact target field is diagnostics only; it never enters predict_counts.
                try:
                    truth = calibrate_peak(UnlabeledField(scores, weights), scores.min(), scores.max(), args.eta)
                    cell['target_geometry'] = truth
                    cell['actual_seed'] = float(np.sqrt(n)*(q0-truth['q'])/truth['seed_sd'])
                    lo, hi = component['lo'], component['hi']
                    cell['root_inside_component'] = ((lo is None or lo < truth['q'])
                                                     and (hi is None or truth['q'] <= hi))
                    cell['bounded_component'] = lo is not None and hi is not None
                    if cell['bounded_component']:
                        probe = np.linspace(lo-args.eta*.5, hi+args.eta*.5, 513)
                        drift = UnlabeledField(scores, weights).velocities(probe)
                        distance = probe-truth['q']
                        keep = abs(distance) > 1e-8
                        cell['outward_ratio_grid_min'] = float(np.min(drift[keep]/distance[keep]))
                except ValueError as exc:
                    cell['target_geometry_unavailable'] = str(exc)
                key = f'cell_{pi}_{ni}_{ti}'
                sample = simulate(scores, weights, risk, q0, component, n, args.paths,
                                  args.steps, args.eta,
                                  np.random.SeedSequence([args.seed, pi, ni, ti, 0]), args.block)
                arrays = dict(sample)
                rt, rs = sample['risk_times'], sample['risk_sides']
                cell.update(key=key, paths=args.paths, right=int(np.sum(rs > 0)),
                            left=int(np.sum(rs < 0)), censored=int(np.sum(rt < 0)),
                            event_disagreements=int(np.sum(sample['component_times'] != rt)),
                            direction_disagreements=int(np.sum(
                                (sample['component_sides'] != rs) & (rt >= 0))),
                            component_right=int(np.sum(sample['component_sides'] > 0)),
                            component_left=int(np.sum(sample['component_sides'] < 0)),
                            component_censored=int(np.sum(sample['component_times'] < 0)))
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, pi, ni, ti, 1]))
                counts = np.zeros((args.repeats, len(scores)), dtype=np.int64)
                previous = 0
                for budget in budgets:
                    counts += rng.multinomial(budget-previous, weights, size=args.repeats)
                    previous = budget
                    equivalent = [s for s in chosen if max(2, int(round(c*c*n/s['ratio']**2))) == budget]
                    for setting in equivalent:
                        rule = select_rule(rules, args.window, setting['ratio'])
                        forecasts, details = [], []
                        for row in counts:
                            pred, detail = predict_counts(scores, row, baseline, n, args.eta,
                                                          args.window, rule, mesh, gradients)
                            forecasts.append(pred)
                            details.append(detail)
                        name = f"{setting['family']}_r{setting['ratio']:g}"
                        arrays[name] = np.asarray(forecasts)
                        cell['settings'].append(dict(**setting, budget=budget, key=name,
                            actual_ratio=float(c*np.sqrt(n/budget)), root_fits=details,
                            rule=dict(K=rule['K'], r=rule['r'], support=rule['support'],
                                      thresholds=rule['thresholds'])))
                np.savez_compressed(args.output/(key+'.npz'), **arrays)
                result['cells'].append(cell)
                save()
                print(f'{pair}, n={n}, t={t:g}: right={cell["right"]}, censored={cell["censored"]}', flush=True)
    protocol['complete'] = True
    save()
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, default=CACHE)
    p.add_argument('--rules', type=Path, default=RULES)
    p.add_argument('--output', type=Path, default=BASE/'local_visual')
    p.add_argument('--pairs', nargs='+', type=class_pair, default=PAIRS)
    p.add_argument('--batches', nargs='+', type=int, default=[32, 128, 512])
    p.add_argument('--seeds', nargs='+', type=float, default=[-1, -.5, 0, .5, 1])
    p.add_argument('--ratios', nargs='+', type=float, default=[.5, 1, 2])
    p.add_argument('--decreasing', action='store_true')
    p.add_argument('--decreasing-start', type=float, default=2.)
    p.add_argument('--window', type=float, default=1.)
    p.add_argument('--prior', type=float, default=.5)
    p.add_argument('--eta', type=float, default=.1)
    p.add_argument('--epsilon', type=float, default=.02)
    p.add_argument('--repeats', type=int, default=256)
    p.add_argument('--paths', type=int, default=1024)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--block', type=int, default=256)
    p.add_argument('--seed', type=int, default=20261029)
    return p


if __name__ == '__main__':
    run(parser().parse_args())

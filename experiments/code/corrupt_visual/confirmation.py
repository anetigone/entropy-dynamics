"""Independent corruption confirmation; freeze risk geometry before sampling."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from corrupt_prediction import (parser, read_features, geometry, compute_cell,
    project_pair, UnlabeledField, WeightedRisk, select_rule, load_rules,
    check_preprocessing, NAMES)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def prepare(args):
    """No reference trajectories or calibration draws are used here."""
    train, labels = read_features(args.clean_features, 'train')
    clean, clean_labels = read_features(args.clean_features)
    cp = json.loads(args.clean_features.with_name('protocol.json').read_text())
    tp = json.loads((args.corrupt_features/'protocol.json').read_text())
    check_preprocessing(cp['preprocessing'], tp['preprocessing'])
    rule = select_rule(load_rules(args.rules), args.window, args.rule_ratio)
    environments = [dict(key='clean', corruption='clean', severity=0,
                         features=str(args.clean_features))]
    for name in args.corruptions:
        for severity in args.severities:
            key = f'{name}_s{severity}'
            environments.append(dict(key=key, corruption=name, severity=severity,
                features=str(args.corrupt_features/(key+'.npz'))))
    groups, cells = [], []
    for pair in args.pairs:
        pp, nn, _, _, anchor = project_pair(train, labels, clean, clean_labels, pair)
        z = np.r_[pp, nn]
        weights = np.full(len(z), 1/len(z))
        source, failure = geometry(z, weights, args.eta)
        if source is None:
            raise ValueError(f'Confirmation requires declared source geometry: {pair}, {failure}')
        x0 = source['q']
        mean, _, variance = UnlabeledField(z, weights).moments(x0)
        group = dict(pair=list(pair), anchor=anchor, source_geometry=source,
            source_geometry_failure=None, x0=x0, source_mean_velocity=mean,
            c=float(np.sqrt(variance)/(source['lambda_']*source['seed_sd'])),
            scan_interval=[float(z.min()-args.scan_margin), float(z.max()+args.scan_margin)],
            rule=rule)
        groups.append(group)
        for index, env in enumerate(environments):
            x, y = read_features(env['features'])
            keep = (y == pair[0]) | (y == pair[1])
            scores = ((x[keep]-np.asarray(anchor['center'])) @
                      np.asarray(anchor['direction'])/anchor['projection_scale'])
            weights = np.full(len(scores), 1/len(scores))
            risk = WeightedRisk(scores, y[keep] == pair[0], weights)
            threshold = float(risk.risk(x0)+args.epsilon)
            knots = np.unique(scores)
            probes = np.r_[np.nextafter(knots[0], -np.inf), np.nextafter(knots, np.inf)]
            maximum = float(np.max(risk.risk(probes)))
            truth, error = geometry(scores, weights, args.eta)
            t = rscale = None
            if truth is not None:
                t = float(np.sqrt(args.batch)*(x0-truth['q'])/truth['seed_sd'])
                # Delta-method variance of the empirical-root coordinate at target Q.
                _, _, var = UnlabeledField(scores, weights).moments(truth['q'])
                rscale = float(np.sqrt(var)/(truth['lambda_']*truth['seed_sd']))
            cells.append(dict(key=f'cell_{pair[0]}_{pair[1]}_{index}', pair=list(pair),
                environment=env, index=index, eligible=maximum > threshold,
                maximum_risk=maximum, threshold=threshold, target_t=t,
                target_noise_scale=rscale, target_geometry_failure=error,
                stratum=None if t is None else int(np.searchsorted(args.strata, abs(t), side='left'))))
    return groups, cells


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if (min(args.paths, args.repeats, args.steps, args.batch, args.workers, args.block) < 1
            or args.repeats < 2 or len(args.budgets) < 2
            or args.budgets != sorted(set(args.budgets)) or min(args.budgets) < 1
            or not np.isfinite([args.eta, args.epsilon, args.scan_margin, *args.strata]).all()
            or not 0 < args.eta < 1 or min(args.epsilon, args.scan_margin) <= 0
            or not 0 < args.alpha < 1 or args.seed < 0
            or args.strata != sorted(set(args.strata)) or min(args.strata) <= 0
            or len(set(args.pairs)) != len(args.pairs)
            or len(set(args.corruptions)) != len(args.corruptions)
            or len(set(args.severities)) != len(args.severities)
            or any(s not in range(1, 6) for s in args.severities)
            or any(a == b for a, b in args.pairs)):
        raise ValueError('Invalid confirmation parameters')
    groups, cells = prepare(args)
    if not any(c['eligible'] for c in cells):
        raise ValueError('No geometrically eligible environments')
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(predictors=NAMES, primary='target_root',
        target='P(T finite and x_T > x0), not conditional on observed exits',
        comparison='equal-environment mean MAE(low M) minus MAE(high M)',
        sampling='fixed official pools, replacement; independent reference and calibration',
        uncertainty='alpha/2 simultaneous reference; alpha/2 one paired aggregate Bernstein',
        strata_role='offline descriptive only; no significance claims',
        target_noise='delta-method target-root coordinate, not a finite-sample guarantee')
    args.output.mkdir(parents=True)
    dump(args.output/'design.json', dict(protocol=protocol, groups=groups, cells=cells))
    result = dict(protocol=dict(protocol, complete=False), groups=groups, cells=[])
    dump(args.output/'results.json', result)
    by_pair = {tuple(g['pair']): g for g in groups}
    tasks = [(args, by_pair[tuple(c['pair'])], c['environment'], c['index'])
             for c in cells if c['eligible']]
    # Independent sampling starts only after the complete design is on disk.
    if args.workers == 1:
        for task in tasks:
            result['cells'].append(compute_cell(task))
            dump(args.output/'results.json', result)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            for cell in pool.map(compute_cell, tasks):
                result['cells'].append(cell)
                dump(args.output/'results.json', result)
    result['protocol']['complete'] = True
    dump(args.output/'results.json', result)


if __name__ == '__main__':
    p = parser()
    p.description = __doc__
    p.set_defaults(output=Path('experiments/results/corrupt_confirmation'), seed=73191)
    p.add_argument('--strata', type=float, nargs='+', default=[1., 3.])
    p.add_argument('--alpha', type=float, default=.05)
    # Precision budgets must be chosen explicitly rather than inherited from exploration.
    for action in p._actions:
        if action.dest in ('paths', 'repeats'):
            action.required = True
    run(p.parse_args())

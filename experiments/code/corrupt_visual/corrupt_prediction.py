"""Prospective direction forecasts on fixed CIFAR-10-C feature distributions.

Source initialization stays fixed across corruptions. References and unlabeled
calibration streams are independent; labels are only for generators/evaluation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr
from corrupt_features import check_preprocessing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frozen_visual'))
from entropy_visual import PAIRS, class_pair, project_pair
from field_geometry import calibrate_peak
from visual_calibration import UnlabeledField
from local_calibration import (WeightedRisk, simulate, velocity, load_rules,
                               select_rule, predict_counts, RULES)

NAMES = ['constant', 'source_root', 'target_root', 'source_moment',
         'source_clipped', 'gaussian_transfer', 'oracle_root']


def read_features(path, prefix='test'):
    with np.load(path, allow_pickle=False) as f:
        x, y = f[prefix + '_features'], f[prefix + '_labels']
    if (x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all()
            or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y > 9))):
        raise ValueError(f'Invalid aligned feature arrays: {path}, {prefix}')
    return x, y


def geometry(scores, weights, eta):
    try:
        value = calibrate_peak(UnlabeledField(scores, weights),
                               float(scores.min()), float(scores.max()), eta)
        return value, None
    except (ValueError, RuntimeError) as exc:
        return None, str(exc)


def compute_cell(task):
    args, group, environment, index = task
    features, labels = read_features(environment['features'])
    a, b = group['pair']
    anchor = group['anchor']
    direction, center = np.asarray(anchor['direction']), np.asarray(anchor['center'])
    if features.shape[1] != len(direction):
        raise ValueError('Clean and corrupted feature dimensions differ')
    keep = (labels == a) | (labels == b)
    scores = ((features[keep] - center) @ direction / anchor['projection_scale']).astype(float)
    labels = labels[keep] == a
    if min(int(labels.sum()), int((~labels).sum())) < 2:
        raise ValueError('Missing target classes')
    # Official CIFAR-10-C has equal class counts. No additional target reweighting.
    weights = np.full(len(scores), 1 / len(scores))
    x0 = group['x0']
    risk = WeightedRisk(scores, labels, weights)
    risk.threshold = float(risk.risk(x0) + args.epsilon)
    component = risk.component(x0, args.epsilon)
    truth, truth_failure = geometry(scores, weights, args.eta)
    cell = dict(key=f'cell_{a}_{b}_{index}', pair=group['pair'], n=args.batch,
                environment=environment['key'], corruption=environment['corruption'],
                severity=environment['severity'], x0=x0, risk0=float(risk.risk(x0)),
                threshold=risk.threshold, component=component, paths=args.paths,
                target_geometry=truth, target_geometry_failure=truth_failure,
                source_geometry_failure=group['source_geometry_failure'], settings=[])
    if truth is not None:
        actual_seed = np.sqrt(args.batch) * (x0 - truth['q']) / truth['seed_sd']
        cell.update(actual_seed=float(actual_seed), outside_local_window=bool(abs(actual_seed) > args.window),
                    root_inside_component=bool((component['lo'] is None or component['lo'] < truth['q'])
                                              and (component['hi'] is None or truth['q'] <= component['hi'])))
    sample = simulate(scores, weights, risk, x0, component, args.batch, args.paths,
                      args.steps, args.eta, np.random.SeedSequence([args.seed, a, b, index, 0]), args.block)
    rt, rs = sample['risk_times'], sample['risk_sides']
    cell.update(right=int((rs > 0).sum()), left=int((rs < 0).sum()), censored=int((rt < 0).sum()),
                event_disagreements=int((sample['component_times'] != rt).sum()),
                direction_disagreements=int(((sample['component_sides'] != rs) & (rt >= 0)).sum()))

    # The root scan interval is declared from source scores, never target extrema.
    mesh = np.linspace(*group['scan_interval'], 513)
    gradients = velocity(mesh[None, :], scores[:, None])
    source = group['source_geometry']
    source_forecast = .5 if source is None else float(ndtr(
        np.sqrt(args.batch) * (x0 - source['q']) / source['seed_sd']))
    oracle_forecast = .5 if truth is None else float(ndtr(cell['actual_seed']))
    psi = np.zeros_like(scores) if source is None else (
        velocity(x0, scores) - group['source_mean_velocity']) / (source['lambda_'] * source['seed_sd'])
    baseline = dict(x0=x0, psi=psi)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, a, b, index, 1]))
    counts = np.zeros((args.repeats, len(scores)), dtype=np.int64)
    previous = 0
    arrays = dict(sample)
    for budget in args.budgets:
        counts += rng.multinomial(budget - previous, weights, size=args.repeats)
        previous = budget
        forecasts, fits = [], []
        for count in counts:
            pred, fit = predict_counts(scores, count, baseline, args.batch, args.eta,
                                       args.window, group['rule'], mesh, gradients)
            coordinate = float(np.sqrt(args.batch) * ((count/budget) @ psi))
            fit.update(source_coordinate=coordinate,
                       outside_source_window=bool(abs(coordinate) > args.window))
            # Target-root estimation is available even if source geometry failed.
            if source is None:
                pred[2:] = [.5, .5, .5]
            forecasts.append([.5, source_forecast, pred[1], *pred[2:], oracle_forecast])
            fits.append(fit)
        key = f'm{budget}'
        arrays[key] = np.asarray(forecasts)
        cell['settings'].append(dict(key=key, budget=budget, root_fits=fits,
            source_noise_ratio=None if source is None else float(group['c'] * np.sqrt(args.batch / budget)),
            gaussian_rule_ratio=args.rule_ratio,
            source_rule_ratio_matches=bool(source is not None and np.isclose(
                group['c'] * np.sqrt(args.batch / budget), args.rule_ratio, rtol=.01))))
    np.savez_compressed(args.output / (cell['key'] + '.npz'), **arrays)
    return cell


def run(args):
    if args.output.exists():
        raise FileExistsError(f'Choose a new result directory: {args.output}')
    if (not np.isfinite([args.eta, args.epsilon, args.window, args.rule_ratio, args.scan_margin]).all()
            or not 0 < args.eta < 1 or min(args.epsilon, args.window, args.rule_ratio, args.scan_margin,
                args.batch, args.paths, args.repeats, args.steps, args.block, args.workers, *args.budgets) <= 0):
        raise ValueError('Require finite positive parameters and eta in (0,1)')
    if args.seed < 0 or args.budgets != sorted(set(args.budgets)):
        raise ValueError('Require nonnegative random seed and strictly increasing budgets')
    if len(set(args.pairs)) != len(args.pairs) or len(set(args.corruptions)) != len(args.corruptions):
        raise ValueError('Do not duplicate pairs or corruptions')
    if len(set(args.severities)) != len(args.severities) or any(s not in range(1, 6) for s in args.severities):
        raise ValueError('Require distinct severities in 1..5')
    train, train_y = read_features(args.clean_features, 'train')
    clean, clean_y = read_features(args.clean_features)
    clean_protocol = json.loads(args.clean_features.with_name('protocol.json').read_text(encoding='utf-8'))
    corrupt_protocol = json.loads((args.corrupt_features / 'protocol.json').read_text(encoding='utf-8'))
    check_preprocessing(clean_protocol['preprocessing'], corrupt_protocol['preprocessing'])
    if train.shape[1] != clean.shape[1]:
        raise ValueError('Train/test feature dimensions differ')
    rule = select_rule(load_rules(args.rules), args.window, args.rule_ratio)
    environments = [dict(key='clean', corruption='clean', severity=0, features=str(args.clean_features))]
    for corruption in args.corruptions:
        for severity in args.severities:
            key = f'{corruption}_s{severity}'
            path = args.corrupt_features / (key + '.npz')
            if not path.is_file():
                raise FileNotFoundError(path)
            environments.append(dict(key=key, corruption=corruption, severity=severity, features=str(path)))
    groups = []
    for pair in args.pairs:
        if pair[0] == pair[1]:
            raise ValueError('A class pair must contain two distinct classes')
        pp, nn, _, _, anchor = project_pair(train, train_y, clean, clean_y, pair)
        scores = np.r_[pp, nn]
        weights = np.full(len(scores), 1 / len(scores))
        source, failure = geometry(scores, weights, args.eta)
        x0 = 0. if source is None else source['q']
        group = dict(pair=list(pair), anchor=anchor, source_geometry=source,
                     source_geometry_failure=failure, x0=float(x0),
                     initialization_fallback=source is None, rule=rule,
                     scan_interval=[float(scores.min()-args.scan_margin), float(scores.max()+args.scan_margin)])
        if source is not None:
            field = UnlabeledField(scores, weights)
            mean, _, variance = field.moments(x0)
            group.update(source_mean_velocity=mean,
                         c=float(np.sqrt(variance)/(source['lambda_']*source['seed_sd'])))
        groups.append(group)
    args.output.mkdir(parents=True)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(complete=False, predictors=NAMES, environments=environments,
        clean_checkpoint=clean_protocol.get('checkpoint'),
        corrupt_checkpoint=corrupt_protocol.get('model'),
        preprocessing=clean_protocol['preprocessing'],
        information='Clean training baseline and fixed initialization; unlabeled target calibration only',
        risk='Actual empirical label risk of each unweighted environment; not common risk',
        sampling='Independent iid reference and calibration draws with replacement from a fixed test pool',
        failures='All failures scored with .5 fallback; no environment filtering',
        gaussian='One frozen rule across budgets; transfer ablation, not corruption minimax optimality',
        source_root='Equals .5 when initialization is the source root; same information as constant',
        oracle='Exact target root/scale is an offline diagnostic, with .5 fallback on failure')
    result = dict(protocol=protocol, groups=groups, cells=[])

    def save():
        result['cells'].sort(key=lambda c: (c['pair'], c['environment']))
        (args.output / 'results.json').write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

    def accept(cell):
        result['cells'].append(cell)
        save()
        print(f"{cell['pair']} {cell['environment']}: right={cell['right']}, censored={cell['censored']}", flush=True)

    save()
    tasks = [(args, group, env, i) for group in groups for i, env in enumerate(environments)]
    if args.workers == 1:
        for task in tasks:
            accept(compute_cell(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(compute_cell, task) for task in tasks]
            for future in as_completed(futures):
                accept(future.result())
    protocol['complete'] = True
    save()
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean-features', type=Path, default=Path('experiments/results/visual_features/features.npz'))
    p.add_argument('--corrupt-features', type=Path, default=Path('experiments/results/corrupt_features'))
    p.add_argument('--output', type=Path, default=Path('experiments/results/corrupt_prediction'))
    p.add_argument('--pairs', nargs='+', type=class_pair, default=PAIRS)
    p.add_argument('--corruptions', nargs='+', default=['gaussian_noise', 'defocus_blur', 'contrast'])
    p.add_argument('--severities', nargs='+', type=int, default=[1, 3])
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--budgets', nargs='+', type=int, default=[2048, 16384, 131072])
    p.add_argument('--paths', type=int, default=1024)
    p.add_argument('--repeats', type=int, default=256)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--eta', type=float, default=.1)
    p.add_argument('--epsilon', type=float, default=.02)
    p.add_argument('--window', type=float, default=1.)
    p.add_argument('--rule-ratio', type=float, default=1.)
    p.add_argument('--rules', type=Path, default=RULES)
    p.add_argument('--scan-margin', type=float, default=2.)
    p.add_argument('--block', type=int, default=256)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261034)
    return p


if __name__ == '__main__':
    run(parser().parse_args())

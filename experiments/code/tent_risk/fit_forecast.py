"""Fit and freeze Tent event predictors on labelled development tasks only."""
import argparse
import json
from pathlib import Path
import numpy as np
from tent_forecast import (FEATURES, STATIC_COLUMNS, DRIFT_COLUMNS,
                           load_tasks, fit_logistic, predict)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--development', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--penalty', type=float, default=.1)
    args = p.parse_args()
    if args.penalty <= 0:
        p.error('Penalty must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    x, events, meta, protocol = load_tasks(args.development)
    args.output.mkdir(parents=True)
    coverage = []
    groups = sorted(set((m['corruption'], m['severity'], m['lr'], m['start_step']) for m in meta))
    for group in groups:
        mask = np.array([(m['corruption'],m['severity'],m['lr'],m['start_step'])==group for m in meta])
        y = events[mask]
        coverage.append(dict(corruption=group[0], severity=group[1], lr=group[2],
            start_step=group[3], starts=int(mask.sum()), futures=int(y.size), events=int(y.sum()),
            event_rate=float(y.mean())))
    (args.output/'coverage.json').write_text(json.dumps(coverage, indent=2)+'\n', encoding='utf-8')
    if events.min() == events.max():
        raise ValueError('No event diversity: coverage saved; no predictive method fitted. Preserve all tasks.')
    y = events.mean(1)
    columns = dict(static=STATIC_COLUMNS, drift=DRIFT_COLUMNS, dynamics=list(range(len(FEATURES))))
    models = {name:dict(columns=c, **fit_logistic(x[:, c], y, args.penalty)) for name,c in columns.items()}
    # Environment-grouped development checks; not confirmation or a tuning loop.
    cv = []
    environments = np.array([m['corruption'] for m in meta])
    if len(np.unique(environments)) >= 2:
        for env in np.unique(environments):
            train, test = environments!=env, environments==env
            entry = dict(environment=str(env), tasks=int(test.sum()),
                         constant_brier=float(np.mean((events[test]-events[train].mean())**2)))
            for name,c in columns.items():
                fitted = fit_logistic(x[train][:, c], y[train], args.penalty)
                pred = predict(fitted, x[test][:, c])
                entry[name+'_brier'] = float(np.mean((events[test]-pred[:, None])**2))
            cv.append(entry)
    result = dict(features=FEATURES, penalty=args.penalty, models=models,
        constant=float(events.mean()), development_protocol=protocol,
        development_corruptions=sorted(set(environments)), grouped_development=cv,
        interpretation='Supervised event forecast from unlabelled probes; not a proved scalar exit rule')
    (args.output/'predictor.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(event_rate=float(events.mean()), tasks=len(x), grouped_development=cv),indent=2))


if __name__ == '__main__':
    main()

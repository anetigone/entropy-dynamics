"""Summarize saved corruption forecasts, including all failures and censoring."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from corrupt_prediction import NAMES
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frozen_visual'))
from local_plot import reference_interval, bernstein

PAIRS = [('source_root', 'target_root'), ('source_moment', 'gaussian_transfer'),
         ('source_clipped', 'gaussian_transfer')]


def deterministic_interval(value, lo, hi):
    return [0. if lo <= value <= hi else min(abs(value-lo), abs(value-hi)),
            max(abs(value-lo), abs(value-hi))]


def run(folder, alpha=.05):
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    saved = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
    if not saved['protocol']['complete'] or not saved['cells']:
        raise ValueError('Require a complete, nonempty computation')
    cells = saved['cells']
    count = sum(len(c['settings']) for c in cells)
    tail, delta = alpha/(4*len(cells)), alpha/(2*(len(NAMES)+len(PAIRS))*count)
    rows, references = [], []
    for cell in cells:
        lo, hi = reference_interval(cell['right'], cell['censored'], cell['paths'], tail)
        phat = cell['right']/cell['paths']
        error = max(phat-lo, hi-phat)
        references.append(dict(pair=cell['pair'], environment=cell['environment'],
            estimate=phat, interval=[lo, hi], censored=cell['censored'],
            actual_seed=cell.get('actual_seed'), outside_local_window=cell.get('outside_local_window'),
            root_inside_component=cell.get('root_inside_component'),
            component=cell['component'], event_disagreements=cell['event_disagreements'],
            direction_disagreements=cell['direction_disagreements'],
            target_geometry_failure=cell['target_geometry_failure'],
            source_geometry_failure=cell['source_geometry_failure']))
        with np.load(folder / (cell['key'] + '.npz'), allow_pickle=False) as arrays:
            if (int((arrays['risk_sides'] > 0).sum()) != cell['right']
                    or int((arrays['risk_times'] < 0).sum()) != cell['censored']):
                raise ValueError('Saved reference arrays and exit counts disagree')
            for setting in cell['settings']:
                forecasts = arrays[setting['key']]
                if (forecasts.shape != (saved['protocol']['repeats'], len(NAMES))
                        or not np.isfinite(forecasts).all() or np.any((forecasts < 0) | (forecasts > 1))):
                    raise ValueError('Invalid forecast arrays')
                losses = abs(forecasts-phat)
                scores = {}
                for i, name in enumerate(NAMES):
                    mean = float(losses[:, i].mean())
                    radius = bernstein(losses[:, i], 1, delta)
                    interval = [max(0., mean-radius-error), min(1., mean+radius+error)]
                    if name in ['constant', 'source_root', 'oracle_root']:
                        interval = deterministic_interval(float(forecasts[0, i]), lo, hi)
                    scores[name] = dict(mae=mean, interval=interval,
                        mean_forecast=float(forecasts[:, i].mean()),
                        forecast_quantiles=np.quantile(forecasts[:, i], [.1, .5, .9]).tolist())
                paired = {}
                for baseline, candidate in PAIRS:
                    values = losses[:, NAMES.index(baseline)]-losses[:, NAMES.index(candidate)]
                    mean = float(values.mean())
                    radius = bernstein(values, 2, delta)+2*error
                    paired[baseline+'-'+candidate] = dict(estimate=mean,
                        interval=[max(-1., mean-radius), min(1., mean+radius)])
                fits = setting['root_fits']
                rows.append(dict(pair=cell['pair'], environment=cell['environment'],
                    budget=setting['budget'], scores=scores, paired=paired,
                    reference_interval=[lo, hi], root_failure_rate=sum(f['fallback'] for f in fits)/len(fits),
                    source_window_exceedance_rate=float(np.mean([f['outside_source_window'] for f in fits])),
                    source_noise_ratio=setting['source_noise_ratio'],
                    source_rule_ratio_matches=setting['source_rule_ratio_matches']))
    aggregates = []
    for budget in saved['protocol']['budgets']:
        chosen = [r for r in rows if r['budget'] == budget]
        scores = {name: dict(mae=float(np.mean([r['scores'][name]['mae'] for r in chosen])),
            interval=np.mean([r['scores'][name]['interval'] for r in chosen], axis=0).tolist()) for name in NAMES}
        aggregates.append(dict(budget=budget, scores=scores, environments=len(chosen),
            root_failure_rate=float(np.mean([r['root_failure_rate'] for r in chosen]))))
    output = dict(alpha=alpha, complete=True, cells=rows, references=references, aggregates=aggregates,
        unavailable_source_groups=[g['pair'] for g in saved['groups'] if g['source_geometry'] is None],
        totals=dict(environments=len(cells), paths=sum(c['paths'] for c in cells),
                    censored=sum(c['censored'] for c in cells),
                    target_geometry_failures=sum(c['target_geometry'] is None for c in cells),
                    outside_local_window=sum(c.get('outside_local_window', False) for c in cells),
                    seed_diagnostic_unavailable=sum('actual_seed' not in c for c in cells),
                    event_disagreements=sum(c['event_disagreements'] for c in cells),
                    direction_disagreements=sum(c['direction_disagreements'] for c in cells)),
        scope='Simultaneous two-layer intervals conditional on fixed pools; failures retained; no minimax transfer claim')
    (folder / 'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Corruption prediction: independent calibration', '',
        'Conditional on fixed official test pools; no target reweighting or recentering. All failures fall back to .5 and remain in scores.', '',
        'Source-root forecasts equal .5 by initialization; they are the same baseline as constant. Oracle-root is offline diagnosis, not a deployable method.', '',
        f'At least {100*(1-alpha):g}% simultaneous coverage over reference probabilities, rule risks and the three declared paired comparisons, including censoring and calibration uncertainty.', '',
        '| M | Source MAE | Target-root MAE | Oracle MAE | Gaussian-transfer MAE | Root failure rate |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in aggregates:
        s = row['scores']
        lines.append(f"|{row['budget']}|{s['source_root']['mae']:.4f}|{s['target_root']['mae']:.4f}|{s['oracle_root']['mae']:.4f}|{s['gaussian_transfer']['mae']:.4f}|{row['root_failure_rate']:.4f}|")
    lines += ['', 'Aggregate MAE averages environments equally, including clean controls and failures. Per-cell intervals and paired gains are in summary.json; means are not claims of per-environment dominance.', '',
        '## Applicability and censoring', '', '```json', json.dumps(output['totals'], indent=2), '```', '',
        'Gaussian-transfer uses one frozen (K,r) rule across all budgets. Noise-ratio mismatch and local-window violations are diagnostics, not exclusion criteria. Exact target seed diagnostics require a unique target root; missing diagnostics remain missing.', '',
        '## Figures', '', 'budget.png: per-pair mean MAE versus calibration budget. direction.png: reference simultaneous intervals and mean forecasts at the largest budget. Forecast means are descriptive; MAE is always scored per calibration repetition.', '',
        'Root scan limits are fixed from source scores. Root failures report scan nonuniqueness or numerical/geometry failure; the scan is not a proof of global root uniqueness.', '',
        'Current research state and next steps: ../../../STATUS.md.']
    (folder / 'analysis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    pairs = [g['pair'] for g in saved['groups']]
    fig, axes = plt.subplots(1, len(pairs), squeeze=False, figsize=(4.5*len(pairs), 3.8))
    for ax, pair in zip(axes[0], pairs):
        for name in NAMES:
            means = [np.mean([r['scores'][name]['mae'] for r in rows if r['pair'] == pair and r['budget'] == m])
                     for m in saved['protocol']['budgets']]
            ax.plot(saved['protocol']['budgets'], means, marker='.', label=name)
        ax.set(xscale='log', xlabel='Calibration M', ylabel='Mean probability MAE', title=str(pair))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(folder/'budget.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, len(pairs), squeeze=False, figsize=(5*len(pairs), 4.4))
    environments = [e['key'] for e in saved['protocol']['environments']]
    for ax, pair in zip(axes[0], pairs):
        ref = [next(r for r in references if r['pair'] == pair and r['environment'] == e) for e in environments]
        y = np.array([r['estimate'] for r in ref])
        ax.errorbar(range(len(y)), y, yerr=[y-[r['interval'][0] for r in ref],
                     [r['interval'][1] for r in ref]-y], fmt='ko', label='SGD simultaneous interval')
        for name in ['source_root', 'target_root', 'oracle_root', 'gaussian_transfer']:
            values = [next(r for r in rows if r['pair'] == pair and r['environment'] == e
                          and r['budget'] == saved['protocol']['budgets'][-1])['scores'][name]['mean_forecast'] for e in environments]
            ax.plot(range(len(values)), values, marker='.', label=name)
        ax.set_xticks(range(len(y)), environments, rotation=55, ha='right', fontsize=7)
        ax.set(title=str(pair), ylabel='Right risk-exit probability', ylim=(-.04, 1.04)); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(folder/'direction.png', dpi=160); plt.close(fig)
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=.05)
    args = p.parse_args()
    run(args.input, args.alpha)

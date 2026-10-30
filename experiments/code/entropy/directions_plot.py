"""Summarize unknown-direction samples with censor-compatible intervals."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'frozen_visual'))
from local_plot import reference_interval, bernstein


def run(folder, alpha):
    saved = json.loads((folder/'results.json').read_text())
    p, cells = saved['protocol'], saved['cells']
    if not p['complete'] or not cells or not 0 < alpha < 1:
        raise ValueError('Require completed results and alpha in (0,1)')
    count = sum(len(c['settings']) for c in cells)
    rows = []
    for c in cells:
        with np.load(folder/(c['key']+'.npz'), allow_pickle=False) as a:
            rt, rs = a['risk_times'], a['risk_sides']
            if (rt.shape != (p['paths'],) or rs.shape != rt.shape
                    or not np.isin(rs, [-1, 0, 1]).all()
                    or not np.array_equal(rt < 0, rs == 0)
                    or int((rs > 0).sum()) != c['right']
                    or int((rt < 0).sum()) != c['censored']):
                raise ValueError('Invalid saved reference')
            lo, hi = reference_interval(c['right'], c['censored'], p['paths'], alpha/(4*len(cells)))
            phat = c['right']/p['paths']
            error = max(phat-lo, hi-phat)
            for setting in c['settings']:
                f = a[f"m{setting['budget']}"]
                if f.shape != (p['repeats'],) or not np.isfinite(f).all() or np.any((f < 0) | (f > 1)):
                    raise ValueError('Invalid saved predictions')
                loss = abs(f-phat)
                radius = bernstein(loss, 1, alpha/(2*count))+error
                rows.append(dict(n=c['n'], t=c['t'], amplitude=c['amplitude'],
                    **setting, reference=phat, reference_interval=[lo, hi], censored=c['censored'],
                    direction_approximation=float(ndtr(c['t'])),
                    mae=float(loss.mean()), interval=[max(0., float(loss.mean())-radius),
                                                     min(1., float(loss.mean())+radius)]))
    (folder/'summary.json').write_text(json.dumps(dict(alpha=alpha, cells=rows,
        scope='Simultaneous finite-cell probability and MAE intervals, not minimax certification'),
        indent=2, allow_nan=False), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(6, 4))
    for amplitude in p['amplitudes']:
        selected = [r for r in rows if r['amplitude'] == amplitude]
        ax.scatter([r['r'] for r in selected], [r['mae'] for r in selected], label=f'a={amplitude}')
    ax.set(xlabel='Relative calibration noise r', ylabel='Probability MAE', xscale='log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder/'risk.png', dpi=160)
    plt.close(fig)
    lines = ['# Unknown local directions', '',
        'Continuous rejection sampling; nonlinear entropy SGD; common complete risk.',
        f'Simultaneous coverage at least {100*(1-alpha):g}% for the reported finite cells.',
        'Quadrature geometry is diagnostic. These samples do not certify minimax optimality.', '',
        '|n|t|nuisance amplitude|M|r|right estimate|reference interval|censored|MAE|MAE interval|',
        '|---:|---:|---:|---:|---:|---:|---|---:|---:|---|']
    for r in rows:
        lines.append(f"|{r['n']}|{r['t']}|{r['amplitude']}|{r['budget']}|{r['r']:.3f}|"
                     f"{r['reference']:.4f}|{r['reference_interval']}|{r['censored']}|"
                     f"{r['mae']:.4f}|{r['interval']}|")
    lines += ['', 'Current state and follow-up: [STATUS](../../../STATUS.md).']
    (folder/'analysis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=.05)
    args = p.parse_args()
    run(args.input, args.alpha)

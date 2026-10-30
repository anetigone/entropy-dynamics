"""Score the frozen low/high-budget comparison; never generate samples."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from corrupt_prediction import NAMES
from local_plot import bernstein, reference_interval
from confirmation import dump


def run(folder):
    design = json.loads((folder/'design.json').read_text())
    saved = json.loads((folder/'results.json').read_text())
    p = design['protocol']
    if not saved['protocol']['complete']:
        raise ValueError('Confirmation incomplete')
    for key, value in p.items():
        if saved['protocol'][key] != value:
            raise ValueError(f'Design mismatch: {key}')
    eligible = {c['key']: c for c in design['cells'] if c['eligible']}
    if len(saved['cells']) != len(eligible) or {c['key'] for c in saved['cells']} != set(eligible):
        raise ValueError('Results must cover exactly the frozen geometry-selected set')
    rows, differences, errors = [], [], []
    count = len(eligible)
    for cell in saved['cells']:
        dc = eligible[cell['key']]
        if cell['pair'] != dc['pair'] or cell['environment'] != dc['environment']['key']:
            raise ValueError('Cell identity mismatch')
        with np.load(folder/(cell['key']+'.npz'), allow_pickle=False) as a:
            rt, rs = a['risk_times'], a['risk_sides']
            if (rt.shape != (p['paths'],) or rs.shape != rt.shape
                    or not np.isin(rs, [-1, 0, 1]).all()
                    or not np.array_equal(rt < 0, rs == 0)
                    or int((rs > 0).sum()) != cell['right']
                    or int((rt < 0).sum()) != cell['censored']):
                raise ValueError('Invalid reference arrays or counts')
            phat = float(np.mean(rs > 0))
            lo, hi = reference_interval(cell['right'], cell['censored'], p['paths'],
                                        p['alpha']/(4*count))
            error = max(phat-lo, hi-phat)
            forecasts = []
            for m in p['budgets']:
                f = a[f'm{m}']
                if (f.shape != (p['repeats'], len(NAMES)) or not np.isfinite(f).all()
                        or np.any((f < 0) | (f > 1))):
                    raise ValueError('Invalid forecast arrays')
                forecasts.append(f.copy())
        loss = np.abs(np.asarray(forecasts)-phat)
        j = NAMES.index(p['primary'])
        differences.append(loss[0, :, j]-loss[-1, :, j])
        errors.append(error)
        scale = dc['target_noise_scale']
        r = None if scale is None else [float(scale*np.sqrt(p['batch']/m)) for m in p['budgets']]
        rows.append(dict(pair=cell['pair'], environment=cell['environment'],
            reference=phat, reference_interval=[lo, hi], censored=cell['censored'],
            t=dc['target_t'], stratum=dc['stratum'], target_r=r,
            signal_to_noise=None if r is None else [abs(dc['target_t'])/v for v in r],
            mae={name: loss[:, :, k].mean(axis=1).tolist() for k, name in enumerate(NAMES)},
            root_failure_rates=[sum(f['fallback'] for f in s['root_fits'])/p['repeats']
                                for s in cell['settings']]))
    # Repetition j aggregates independent streams across environments. Conditional
    # on all references, these aggregate paired differences are iid in [-1,1].
    paired = np.mean(differences, axis=0)
    estimate = float(paired.mean())
    cal_radius = bernstein(paired, 2, p['alpha']/2)
    ref_radius = 2*float(np.mean(errors))
    lower, upper = max(-1., estimate-cal_radius-ref_radius), min(1., estimate+cal_radius+ref_radius)
    outcome = 'confirmed' if lower > 0 else 'opposite' if upper < 0 else 'unresolved'
    strata = []
    for label in [*range(len(p['strata'])+1), None]:
        chosen = [r for r in rows if r['stratum'] == label]
        strata.append(dict(stratum=label, environments=len(chosen),
            primary_mae=None if not chosen else np.mean([r['mae'][p['primary']] for r in chosen], axis=0).tolist()))
    output = dict(primary=dict(estimate=estimate, interval=[lower, upper], outcome=outcome,
        calibration_radius=cal_radius, reference_radius=ref_radius, coverage=1-p['alpha']),
        cells=rows, strata=strata,
        excluded_geometry=[c for c in design['cells'] if not c['eligible']],
        auxiliary='All per-rule, per-budget and stratum MAEs are descriptive point estimates')
    dump(folder/'summary.json', output)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ['target_root', 'source_moment', 'gaussian_transfer']:
        ax.plot(p['budgets'], np.mean([r['mae'][name] for r in rows], axis=0), 'o-', label=name)
    ax.set(xscale='log', xlabel='Calibration draws M (fixed pool)', ylabel='Mean probability MAE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder/'budget.png', dpi=160)
    plt.close(fig)
    text = ['# Corruption independent confirmation', '',
        f"Primary: {p['primary']}, MAE({p['budgets'][0]}) minus MAE({p['budgets'][-1]}).",
        f'Geometry-selected environments: {count}; no-exit environments retained separately: {len(output["excluded_geometry"])}.',
        f'Paired gain: {estimate:.6f}; {100*(1-p["alpha"]):g}% interval [{lower:.6f}, {upper:.6f}]; {outcome}.',
        'The interval includes simultaneous reference uncertainty, censoring and paired calibration uncertainty.',
        'Other MAEs and strata are descriptive; no per-environment or monotonicity significance claim.', '',
        '| Stratum | Environments | MAE by ascending M |', '|---|---:|---|']
    text += [f"|{s['stratum']}|{s['environments']}|{s['primary_mae']}|" for s in strata]
    text += ['', 'Target: unconditional eventual right risk exit; never renormalized over observed exits.',
             'Current state and follow-up: [STATUS](../../../STATUS.md).']
    (folder/'analysis.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(output['primary'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    run(p.parse_args().input)

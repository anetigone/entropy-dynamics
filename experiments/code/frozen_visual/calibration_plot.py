"""Plot independent calibration errors from saved results; no recalculation."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visual_calibration import OUT


def main(folder=OUT):
    folder = Path(folder)
    result = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    cases = result['cases']
    conditions = sorted({(tuple(c['pair']),c['prior']) for c in cases})
    budgets = result['protocol']['budgets']
    repeats = result['protocol']['repeats']
    fig,axes = plt.subplots(len(conditions),3,figsize=(14,3*len(conditions)),
                            squeeze=False,constrained_layout=True)
    for i,(pair,prior) in enumerate(conditions):
        cc = [c for c in result['calibrations'] if tuple(c['pair'])==pair and c['prior']==prior]
        rms,se = [],[]
        for budget in budgets:
            rr = [c for c in cc if c['budget']==budget and 'root_error' in c]
            rms.append(np.sqrt(np.mean([c['root_error']**2 for c in rr])) if rr else np.nan)
            se.append(np.mean([c['root_se'] for c in rr]) if rr else np.nan)
        axes[i,0].loglog(budgets,rms,'o-',label='Root RMSE across calibrations')
        axes[i,0].loglog(budgets,se,'--',label='Mean plug-in root SE')
        axes[i,0].set(xlabel='Unlabeled calibration draws',ylabel='Threshold units',
                      title=f'Classes {pair}, prior={prior}')
        rows = [c for c in cases if tuple(c['pair'])==pair and c['prior']==prior]
        ids = {c['key'] for c in rows}
        for n in sorted({c['n'] for c in rows}):
            n_ids = {c['key'] for c in rows if c['n']==n}
            means = []
            for budget in budgets:
                rr = [f for f in result['fits'] if f['case'] in n_ids and f['budget']==budget]
                means.append(np.mean([f['root_noise_ratio'] for f in rr]) if rr else np.nan)
            axes[i,1].loglog(budgets,means,'o-',label=f'n={n}')
        axes[i,1].axhline(1,color='black',ls=':',label='Calibration SE = SGD seed SD')
        axes[i,1].set(xlabel='Unlabeled calibration draws',ylabel='Root SE / SGD threshold noise')
        # Pick the middle saved batch and the closest saved offset to zero.
        ns = sorted({c['n'] for c in rows})
        n = ns[len(ns)//2]
        row = min([c for c in rows if c['n']==n],key=lambda c:abs(c['offset']))
        for model,color in [('source_risk','#b45d32'),('oracle_risk','#4775a4')]:
            values = []
            for budget in budgets:
                rr = next((a for a in result['aggregate'] if a['case']==row['key']
                           and a['budget']==budget and a['model']==model),None)
                values.append(rr['mean_cdf_error'] if rr else np.nan)
                axes[i,2].annotate(f'{rr["count"] if rr else 0}/{repeats}',
                    (budget,rr['mean_cdf_error'] if rr else .98),fontsize=7,color=color)
            axes[i,2].semilogx(budgets,values,'o-',label=model,color=color)
        for name,color in [('source_conjugate','#b45d32'),('target_conjugate','#4775a4')]:
            if name in row['baseline']:
                axes[i,2].axhline(row['baseline'][name]['cdf_error'],color=color,ls='--',
                                  label=f'{name} baseline')
        axes[i,2].set(xlabel='Unlabeled calibration draws',ylabel='Mean per-calibration CDF error',
                      ylim=(0,1.02),title=f'n={n}, offset={row["offset"]}; labels: valid/repeats')
        for ax in axes[i]:
            ax.legend(fontsize=7,frameon=False)
    fig.suptitle('Independent calibration; oracle_risk still uses target-label endpoints')
    fig.savefig(folder/'calibration.png',dpi=160)
    plt.close(fig)
    # Each point summarizes forecasts over calibration draws for one saved case.
    # These repeats are not additional independent SGD paths.
    groups = []
    for row in cases:
        for budget in budgets:
            fits = [f for f in result['fits'] if f['case']==row['key'] and f['budget']==budget
                    and 'theoretical_ratio' in f]
            if fits:
                groups.append((row, budget, fits))
    if groups:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for row, budget, fits in groups:
            ratio = fits[0]['theoretical_ratio']
            for name, color in [('source_risk', '#b45d32'), ('oracle_risk', '#4775a4')]:
                errors = [f['predictions'][name]['cdf_error'] for f in fits if name in f['predictions']]
                if errors:
                    ax[0].scatter(ratio, np.mean(errors), c=color, s=14, alpha=.6)
            # Compare the local root diagnostic with its matching local target
            # reference. The arctan law only applies at target-critical centering.
            error = np.mean([abs(f['local_direction']-f['target_local_direction']) for f in fits])
            ax[1].scatter(ratio, error, s=14, alpha=.6)
        xx = np.geomspace(.01, 100, 200)
        ax[1].plot(xx, np.arctan(xx)/np.pi, 'k--', label='Critical-centered limit only')
        ax[0].set(xscale='log', xlabel='Theoretical calibration / SGD scale', ylabel='Mean CDF error',
                  title='Orange: training endpoints; blue: oracle endpoints')
        ax[1].set(xscale='log', xlabel='Theoretical calibration / SGD scale', ylabel='Local direction probability MAE',
                  title='All initializations; curve is not a fit')
        ax[1].legend(fontsize=8)
        fig.savefig(folder/'calibration_scale.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=OUT)
    main(parser.parse_args().input)

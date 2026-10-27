"""Plot independent calibration predictions and scale/LA probes from saved data."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from entropy_visual import OUT


def main(out=OUT):
    out = Path(out)
    result = json.loads((out/'results.json').read_text(encoding='utf-8'))
    with np.load(out/'samples.npz',allow_pickle=False) as saved:
        arrays = dict(saved)
    # One figure for every preselected pair/prior/batch, not a best-looking subset.
    groups = sorted({(tuple(r['pair']),r['prior'],r['n']) for r in result['cases']})
    for pair, p, n in groups:
        rows = [r for r in result['cases'] if tuple(r['pair'])==pair and r['prior']==p and r['n']==n]
        reference_mode = 'threshold' if any(r['mode']=='threshold' for r in rows) else rows[0]['mode']
        fixed = sorted([r for r in rows if r['mode']==reference_mode],key=lambda r:r['offset'])
        if not fixed:
            continue
        fig, axes = plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        for row in fixed:
            key, grid = row['key'], arrays['grid']
            line, = axes[0].step(grid,arrays[key+'_observed_left']+arrays[key+'_observed_right'],
                                where='post',label=f"offset={row['offset']}")
            if 'conjugate' in row['predictions']:
                axes[0].plot(grid,arrays[key+'_conjugate_left']+arrays[key+'_conjugate_right'],
                             '--',color=line.get_color())
        for mode in result['protocol']['modes']:
            rr = sorted([r for r in rows if r['mode']==mode],key=lambda r:r['offset'])
            # Censoring bounds shown for all modes, without a normal approximation.
            bounds = []
            for r in rr:
                key=r['key']
                right=float(arrays[key+'_observed_right'][-1])
                censored=float(np.mean(arrays[key+'_times']<0))
                bounds.append([right,right+censored])
            if bounds:
                b=np.array(bounds); x=[r['offset'] for r in rr]
                line,=axes[1].plot(x,b[:,0],'o-',label=mode)
                axes[1].fill_between(x,b[:,0],b[:,1],color=line.get_color(),alpha=.1)
            center = min(rr,key=lambda r:abs(r['offset'])) if rr else None
            if center:
                trace=arrays[center['key']+'_trace']
                axes[2].plot(trace[:,0],trace[:,1]-center['evaluation']['frozen_risk'],label=mode)
        axes[2].axhline(result['protocol']['epsilon'],color='k',ls=':',label='tolerance')
        axes[0].set(xlabel='Update',ylabel='First exceedance CDF',ylim=(0,1),title=f'{reference_mode}: SGD and fixed-slope prediction')
        axes[1].set(xlabel=f"Translation offset ({fixed[0]['offset_units']})",ylabel='Right exit probability bounds',ylim=(0,1))
        axes[2].set(xlabel='Update',ylabel='Mean excess empirical risk',title='Closest offset to cancellation')
        for ax in axes:
            ax.legend(fontsize=7,frameon=False)
        fig.suptitle(f'Classes {pair}, prior={p}, batch={n}')
        fig.savefig(out/f'pair_{pair[0]}_{pair[1]}_p{p:g}_n{n}.png',dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=OUT, help='Saved experiment directory')
    args = parser.parse_args()
    main(args.input)

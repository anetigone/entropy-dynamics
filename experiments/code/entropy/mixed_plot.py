"""Plot saved mixed-shift computations in a separate process."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from entropy_mixed import OUT


def main():
    result = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    with np.load(OUT/'samples.npz',allow_pickle=False) as saved:
        arrays = dict(saved)
    fig, axes = plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for n in result['protocol']['batches']:
        rows = [r for r in result['cases'] if r['n']==n and r['kappa']==max(result['protocol']['kappas'])]
        offsets = [r['offset'] for r in rows]
        summaries = [r['predictions']['conjugate'] for r in rows]
        bounds = np.array([s['right_probability_bounds'] for s in summaries])
        line, = axes[0].plot(offsets,bounds[:,0],'o-',label=f'SGD n={n}')
        axes[0].fill_between(offsets,bounds[:,0],bounds[:,1],alpha=.15,color=line.get_color())
        axes[0].plot(offsets,[s['eventual_right'] for s in summaries],'--',color=line.get_color())
        med = [s['observed_quantiles'][1] if s['observed_quantiles'][1] is not None else np.nan for s in summaries]
        axes[1].plot(offsets,med,'o-',color=line.get_color(),label=f'n={n}')
        axes[1].plot(offsets,[s['predicted_quantiles'][1] if s['predicted_quantiles'][1] is not None else np.nan for s in summaries],
                     '--',color=line.get_color())
    selected = next(r for r in result['cases'] if r['offset']==min(result['protocol']['offsets']))
    key, grid = selected['key'], arrays['grid']
    axes[2].step(grid,arrays[key+'_observed_left']+arrays[key+'_observed_right'],where='post',label='SGD')
    for name in ['balanced_limit','actual_linear','conjugate']:
        if 'unavailable' not in selected['predictions'][name]:
            axes[2].plot(grid,arrays[key+'_'+name+'_left']+arrays[key+'_'+name+'_right'],label=name)
    axes[0].set(xlabel='Translation offset (seed SD)',ylabel='Right exit probability',ylim=(0,1))
    axes[1].set(xlabel='Translation offset (seed SD)',ylabel='Median first risk exceedance')
    axes[2].set(xlabel='Update',ylabel='Exit CDF',ylim=(0,1),
                title=f"n={selected['n']}, kappa={selected['kappa']}, offset={selected['offset']}")
    for ax in axes:
        ax.legend(fontsize=8,frameon=False)
    fig.savefig(OUT/'mixed_shift.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()

"""Plot saved empirical-geometry diagnosis without recalculating features or SGD."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visual_diagnosis import OUT


def main(folder=OUT):
    folder = Path(folder)
    result = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    with np.load(folder/'curves.npz',allow_pickle=False) as saved:
        curves = dict(saved)
    pairs = result['protocol']['original_protocol']['pairs']
    priors = result['protocol']['original_protocol']['priors']
    grid = curves['grid']
    names = ['source_conjugate','target_linear','target_conjugate']
    colors = ['#b45d32','#6a8d52','#4775a4']
    for prior in priors:
        fig, axes = plt.subplots(len(pairs),3,figsize=(14,3*len(pairs)),squeeze=False,
                                  constrained_layout=True)
        for i, pair in enumerate(pairs):
            geom = next(g for g in result['geometry'] if g['pair']==pair and g['prior']==prior)
            gkey = geom['key']
            ax = axes[i,0]
            ax.plot(curves[gkey+'_x'],curves[gkey+'_source_drift'],label='Calibration drift')
            ax.plot(curves[gkey+'_x'],curves[gkey+'_target_drift'],label='Target drift',ls='--')
            ax.axhline(0,color='gray',lw=.5)
            ax.axvline(geom['source']['q'],color='#b45d32',ls=':',label='Calibration peak')
            ax.axvline(geom['target']['q'],color='#4775a4',ls=':',label='Target peak')
            q0,q1 = geom['source']['q'],geom['target']['q']
            ax.set(xlim=(min(q0,q1)-.15,max(q0,q1)+.15),xlabel='Centered threshold',
                   ylabel='Population drift',title=f'Classes {pair}')
            rows = [r for r in result['cases'] if r['pair']==pair and r['prior']==prior and r['offset']==0]
            for n, color in zip(sorted({r['n'] for r in rows}), ['#9daec5','#4775a4','#163f6b']):
                rr = next(r for r in rows if r['n']==n)
                keys = [k for k in names if k in rr['predictions']]
                axes[i,1].plot([names.index(k) for k in keys],
                    [rr['predictions'][k]['joint_cdf_error'] for k in keys],
                    'o-',color=color,label=f'n={n}')
            axes[i,1].set(xticks=range(3),xticklabels=['Calibrated H','Oracle linear','Oracle H'],
                          ylabel='Joint exit CDF error',ylim=(0,1.02),title='offset=0')
            rr = max(rows,key=lambda r:r['n'])
            key=rr['key']
            axes[i,2].step(grid,curves[key+'_observed_left']+curves[key+'_observed_right'],
                           where='post',color='black',label='Saved SGD')
            for name,color in zip(names,colors):
                if name in rr['predictions']:
                    axes[i,2].plot(grid,curves[key+'_'+name+'_left']+curves[key+'_'+name+'_right'],
                                   color=color,label=name)
            axes[i,2].set(xlabel='Update',ylabel='Total exit CDF',ylim=(0,1.02),
                           title=f'n={rr["n"]}, offset=0')
            for ax in axes[i]:
                ax.legend(fontsize=7,frameon=False)
        fig.suptitle(f'Prior={prior}: target geometry is a post-hoc oracle, not independent prediction')
        fig.savefig(folder/f'prior_{prior:g}.png',dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=OUT)
    main(parser.parse_args().input)

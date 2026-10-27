"""Plot saved geometry results; run this file directly."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from entropy_geometry import OUT, Model, trajectory


def main():
    results = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    with np.load(OUT / 'sgd_crossing_times.npz', allow_pickle=False) as saved:
        times = saved['times']
    model = Model()
    epsilon = .005
    rhos = np.array([row['rho'] for row in results['log_law']])
    log_rows = results['log_law']
    slope = results['log_slope']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','figure.facecolor':'white'})
    fig, ax = plt.subplots(2,2,figsize=(11,7.8),constrained_layout=True)
    xs, rr, hh = trajectory(model,steps=180)
    ax[0,0].plot(rr,color='#1c4e80',lw=2,label='Classification risk')
    ax[0,0].axhline(rr[0]+epsilon,color='#b95c32',ls='--',label='Frozen target risk + 0.005')
    ax[0,0].axvline(94,color='#b95c32',alpha=.5)
    ax[0,0].set(title='A  Population rebound',xlabel='Update',ylabel='Risk')
    ax[0,0].legend(frameon=False,fontsize=8)
    xx=np.log(1/rhos)
    yy=np.array([r['T_fixed_epsilon'] for r in log_rows])
    ax[0,1].plot(xx,yy,'o-',color='#1c4e80',label='Numerical crossing')
    intercept=np.mean(yy[-4:]-slope*xx[-4:])
    asymptotic_mask = rhos <= .01
    ax[0,1].plot(xx[asymptotic_mask],slope*xx[asymptotic_mask]+intercept,'--',
                 color='#b95c32',label=f'Theory slope {slope:.2f}; fitted offset')
    ax[0,1].set(title='B  Fixed tolerance: logarithmic delay',xlabel='log(1 / prior imbalance)',ylabel='First crossing update')
    ax[0,1].legend(frameon=False,fontsize=8)
    ax[1,0].semilogx(rhos,[r['T_zero'] for r in log_rows],'o-',color='#1c4e80')
    ax[1,0].axhline(79,color='#b95c32',ls='--',label='Small-imbalance prediction: 79')
    ax[1,0].set(title='C  Zero tolerance: bounded update count',xlabel='Prior imbalance',ylabel='First crossing update')
    ax[1,0].legend(frameon=False,fontsize=8)
    ax[1,1].hist(times,bins=np.arange(times.min()-.5,times.max()+1.5),
                 color='#1c4e80',alpha=.85)
    ax[1,1].axvline(50,color='#b95c32',ls='--')
    ax[1,1].axvline(210,color='#b95c32',ls='--',label='Drift bound: (50, 210]')
    ax[1,1].set(title='D  Fresh-batch SGD: n = 64',xlabel='First crossing update',ylabel='Runs')
    ax[1,1].legend(frameon=False,fontsize=8)
    fig.suptitle('Entropy adaptation: proved model regimes and numerical checks',fontsize=14,fontweight='bold')
    fig.savefig(OUT/'theory_checks.png',dpi=180)
    fig.savefig(OUT/'theory_checks.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()

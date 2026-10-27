"""Plot saved critical results; run this file directly."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from entropy_critical import ROOT, Geometry, cdf_prediction, finite_cdf
from scipy.special import ndtr


def main():
    results = json.loads((ROOT / 'critical_results.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'critical_samples.npz', allow_pickle=False) as saved:
        arrays = dict(saved)
    g=Geometry()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    colors={32:'#9daec5',128:'#4775a4',512:'#163f6b'}
    for n in [32,128,512]:
        rr=[r for r in results['macro'] if r['n']==n]
        ax[0,0].errorbar([r['kappa'] for r in rr],[r['right_fraction'] for r in rr],
                         yerr=[1.96*np.sqrt(r['right_fraction']*(1-r['right_fraction'])/r['paths']) for r in rr],
                         fmt='o',color=colors[n],capsize=3,label=f'Actual SGD, n={n}')
    k=np.linspace(-.1,2.1,200)
    ax[0,0].plot(k,ndtr(k),'--',color='#b45d32',label='Gaussian seed prediction')
    ax[0,0].set(title='A  Escape direction',xlabel='Effective seed signal / noise',ylabel='Probability of right exit',ylim=(.4,1.03))
    ax[0,0].legend(frameon=False,fontsize=8)
    for n in [32,128,512]:
        tt=arrays[f'macro_n{n}_k1_times']
        grid=np.arange(50,850)
        empirical=np.searchsorted(np.sort(tt),grid,side='right')/len(tt)
        ax[0,1].step(grid,empirical,where='post',color=colors[n],label=f'SGD, n={n}')
        ax[0,1].plot(grid,cdf_prediction(grid,n,1,results['boundaries']['0.005']['H_boundary'],g),'--',color=colors[n])
    ax[0,1].plot([],[],'--',color='#555555',label='Balanced-limit prediction')
    ax[0,1].set(title=r'B  Fixed tolerance, $\kappa=1$',xlabel='Update',ylabel='Cumulative probability',ylim=(0,1),xlim=(70,750))
    ax[0,1].legend(frameon=False,fontsize=8)
    for n in [32,128,512]:
        tt=arrays[f'micro_n{n}_times']
        grid=np.arange(0,200)
        ax[1,0].step(grid,np.searchsorted(np.sort(tt),grid,side='right')/len(tt),where='post',color=colors[n],label=f'SGD, n={n}')
    tt=arrays['gaussian_ar_reference_times']
    ax[1,0].plot(grid,np.searchsorted(np.sort(tt),grid,side='right')/len(tt),'--',color='#b45d32',label='Gaussian AR(1) limit')
    ax[1,0].set(title='C  Tolerance = 0.005 / n: finite update scale',xlabel='Update',ylabel='Cumulative probability',ylim=(0,1),xlim=(0,180))
    ax[1,0].legend(frameon=False,fontsize=8)
    tt=arrays['macro_n32_k2_times'];grid=np.arange(0,650)
    ax[1,1].step(grid,np.searchsorted(np.sort(tt),grid,side='right')/len(tt),where='post',color='#163f6b',label='Actual nonlinear SGD')
    b=results['boundaries']['0.005']
    ax[1,1].plot(grid,cdf_prediction(grid,32,2,b['H_boundary'],g),':',color='#6d7d63',label='Balanced-limit approximation')
    row=next(r for r in results['macro'] if r['n']==32 and r['kappa']==2)
    if 'finite_geometry' in row:
        ax[1,1].plot(grid,finite_cdf(grid,32,row['finite_geometry']),'--',color='#b45d32',label='Actual peak and risk boundaries')
    ax[1,1].set(title=r'D  Geometry correction, $n=32$, $\kappa=2$',xlabel='Update',ylabel='Cumulative probability',ylim=(0,1),xlim=(0,450))
    ax[1,1].legend(frameon=False,fontsize=8)
    fig.savefig(ROOT/'critical_checks.png',dpi=180,bbox_inches='tight',pad_inches=.15)
    fig.savefig(ROOT/'critical_checks.svg',bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


if __name__ == '__main__':
    main()

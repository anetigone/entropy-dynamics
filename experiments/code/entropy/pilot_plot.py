"""Plot saved pilot results; run this file directly."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from entropy_pilot import OUT


def main():
    report = json.loads((OUT / 'pilot_summary.json').read_text(encoding='utf-8'))
    with np.load(OUT / 'pilot_paths.npz', allow_pickle=False) as saved:
        b, r, entropy = saved['b'], saved['risk'], saved['entropy']
        rr = saved['risk_sgd']
    delta, pi = report['selected']['delta'], report['selected']['pi']
    fig,ax=plt.subplots(1,3,figsize=(12,3.4))
    ax[0].plot(b);ax[0].set(title='Threshold trajectory',xlabel='Update',ylabel='b')
    ax[1].plot(rr.T,color='gray',alpha=.12);ax[1].plot(r,color='navy',label='Population');ax[1].axhline(r[0],color='red',ls='--',label='Source risk');ax[1].legend();ax[1].set(title='Fresh-sample classification risk',xlabel='Update')
    ax[2].plot(entropy);ax[2].set(title='Population prediction entropy',xlabel='Update')
    fig.suptitle(f'Gaussian pilot: delta={delta}, pi={pi}, m=sigma=1, w=2, eta=.1')
    fig.tight_layout();fig.savefig(OUT/'pilot.png',dpi=170);plt.close(fig)


if __name__ == '__main__':
    main()

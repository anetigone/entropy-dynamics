"""Plot saved BN mean/residual diagnostics; no model or data access."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--result', type=Path, required=True)
    args = p.parse_args()
    data = json.loads((args.result/'variation.json').read_text(encoding='utf-8'))
    names = ['gaussian_noise', 'defocus_blur', 'contrast']
    labels = ['Gaussian noise', 'Defocus blur', 'Contrast']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for name, label in zip(names, labels):
        d = data['environments'][name]
        times = np.array([int(t) for t in d['checkpoints']])
        rows = list(d['checkpoints'].values())
        axes[0, 0].plot(times, [100*r['mean_energy_fraction'] for r in rows], 'o-', label=label)
        variance = np.array([r['residual_rms_norm']**2 for r in rows])
        axes[0, 1].plot(times, variance/variance[list(times).index(50)], 'o-', label=label)
        axes[1, 1].plot(times, [100*r['global_axis_residual_fraction'] for r in rows], 'o-', label=label)
    axes[0, 0].set(title='Mean path dominates displacement', ylabel='Empirical mean energy (%)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(times, times/50, 'k--', alpha=.6, label='Linear growth from step 50')
    axes[0, 1].set(title='Residual energy accumulation', ylabel='Residual energy / energy at step 50')
    axes[0, 1].legend(fontsize=8)
    x = np.arange(3)
    axes[1, 0].bar(x-.18, [100*data['environments'][n]['residual_first_fraction_sampled'] for n in names], .36, label='Pooled residual axis (in sample)')
    axes[1, 0].bar(x+.18, [100*np.mean(data['environments'][n]['residual_axis_heldout_fractions']) for n in names], .36, label='Split-fit axis (held-out runs)')
    axes[1, 0].set(xticks=x, xticklabels=['Gaussian', 'Defocus', 'Contrast'], title='Residual axis does not transfer', ylabel='Residual energy explained (%)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(title='Original global axis misses residuals', ylabel='Residual energy explained (%)')
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_xlabel('Tent update')
        ax.grid(alpha=.2)
    fig.savefig(args.result/'variation.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()

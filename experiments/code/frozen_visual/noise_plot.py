"""Plot stored tail-noise ablations without simulation or parameter fitting."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(folder):
    result = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(len(result['cases']), 2, figsize=(12, 3*len(result['cases'])),
                             squeeze=False, constrained_layout=True)
    for row, ax in zip(result['cases'], axes):
        with np.load(folder/(row['key']+'.npz'), allow_pickle=False) as saved:
            for mode in row['modes']:
                times = saved[mode+'_times']
                grid = np.arange(result['protocol']['max_steps']+1)
                cdf = np.cumsum(np.bincount(times[times >= 0], minlength=len(grid)))/len(times)
                ax[0].plot(grid, cdf, label=mode)
            trace = saved['population_trace']
            ax[1].plot(trace[:, 1], trace[:, 2], label='population velocity')
        ax[0].set(title=f"n={row['n']}, offset={row['offset']}, radius={row['radius']}",
                  xlabel='Updates', ylabel='Risk-exit CDF', ylim=(0, 1.02))
        ax[0].legend(fontsize=8)
        ax[1].set(xlabel='Threshold on population orbit', ylabel='Velocity',
                  title=f"Far contribution to linearized variance: {row['diagnostic']['far_variance_fraction']}")
        for x in [row['q']-row['radius'], row['q']+row['radius']]:
            ax[1].axvline(x, color='grey', ls=':')
    fig.savefig(folder/'noise_ablation.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    main(p.parse_args().input)

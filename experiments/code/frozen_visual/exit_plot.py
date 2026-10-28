"""Plot saved geometry and component/risk exits without new simulation."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(folder):
    result = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    rows = [r for r in result['cases'] if 'risk_censored' in r]
    if not rows:
        raise ValueError('No applicable component probes; inspect results.json')
    with np.load(folder/'samples.npz', allow_pickle=False) as data:
        fig, axes = plt.subplots(len(rows), 2, figsize=(11, 2.8*len(rows)), squeeze=False, constrained_layout=True)
        for row, ax in zip(rows, axes):
            key, g = row['key'], row['geometry']
            ax[0].plot(data[key+'_geometry_x'], data[key+'_geometry_velocity'])
            ax[0].axhline(0, color='black', lw=.6)
            for x in [g['left'], g['q'], g['right']]:
                ax[0].axvline(x, color='grey', ls=':')
            ax[0].set(title=f"{row['pair']} p={row['prior']} n={row['n']} offset={row['offset']}", xlabel='Threshold', ylabel='Target empirical drift')
            for name in ['component', 'risk']:
                ax[1].plot(data[key+'_grid'], data[key+'_'+name+'_cdf'], label=name)
            ax[1].set(title=f"Skip={row['skip_fraction']:.4f}; censored risk={row['risk_censored']}", xlabel='Updates', ylabel='CDF', ylim=(0, 1.02))
            ax[1].legend()
        fig.savefig(folder/'exit_probe.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    main(p.parse_args().input)

"""Plot independent forecast scores and reliability from saved summaries."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--result', type=Path, required=True)
    args = p.parse_args()
    d = json.loads((args.result/'summary.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    names = list(d['brier'])
    axes[0].bar(names,[d['brier'][n] for n in names])
    axes[0].set(ylabel='Brier score',title='All predefined tasks')
    for name in names:
        rows = d['reliability'][name]
        axes[1].plot([r['predicted'] for r in rows],[r['observed'] for r in rows],'o-',label=name)
    axes[1].plot([0,1],[0,1],'k--',alpha=.4)
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='Predicted probability',ylabel='Observed event fraction',title='Descriptive reliability')
    axes[1].legend()
    fig.savefig(args.result/'forecast.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()

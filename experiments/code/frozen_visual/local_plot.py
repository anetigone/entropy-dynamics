"""Score and plot saved unknown-environment visual forecasts; no simulation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr
from scipy.stats import beta

from local_calibration import NAMES


def bernstein(values, width, delta):
    m = len(values)
    if m < 2:
        return width
    log = np.log(4/delta)
    return float(np.sqrt(2*np.var(values, ddof=1)*log/m)+7*width*log/(3*(m-1)))


def reference_interval(right, censored, total, tail):
    # Right by cap <= eventual right <= not-left by cap.
    lo = 0. if right == 0 else float(beta.ppf(tail, right, total-right+1))
    k = right+censored
    hi = 1. if k == total else float(beta.ppf(1-tail, k+1, total-k))
    return lo, hi


def run(folder, alpha=.05):
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    saved = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    if not saved['protocol']['complete']:
        raise ValueError('Computation incomplete; keep partial outputs for inspection')
    cells = saved['cells']
    if not cells:
        raise ValueError('No generated environments; inspect baseline applicability failures')
    count = sum(len(c['settings']) for c in cells)
    delta = alpha/(2*9*count)  # Five rule risks and four paired gains per setting.
    tail = alpha/(4*len(cells))
    rows, reference = [], []
    for cell in cells:
        lo, hi = reference_interval(cell['right'], cell['censored'], cell['paths'], tail)
        phat = cell['right']/cell['paths']
        e = max(phat-lo, hi-phat)
        reference.append(dict(pair=cell['pair'], n=cell['n'], t=cell['t'], estimate=phat,
                              interval=[lo, hi], censored=cell['censored'], paths=cell['paths'],
                              event_disagreements=cell['event_disagreements']))
        with np.load(folder/(cell['key']+'.npz'), allow_pickle=False) as arrays:
            for setting in cell['settings']:
                forecasts = arrays[setting['key']]
                if (forecasts.shape != (saved['protocol']['repeats'], len(NAMES))
                        or not np.isfinite(forecasts).all() or np.any((forecasts < 0) | (forecasts > 1))):
                    raise ValueError('Invalid saved forecasts')
                losses = abs(forecasts-phat)
                scores = {}
                for i, name in enumerate(NAMES):
                    mean = float(losses[:, i].mean())
                    rad = bernstein(losses[:, i], 1, delta)
                    interval = [max(0., mean-rad-e), min(1., mean+rad+e)]
                    if name == 'constant':
                        interval = [0. if lo <= .5 <= hi else min(abs(lo-.5), abs(hi-.5)),
                                    max(abs(lo-.5), abs(hi-.5))]
                    scores[name] = dict(mae=mean, interval=interval,
                        calibration_radius=rad, mean_forecast=float(forecasts[:, i].mean()))
                paired = {}
                for i, name in enumerate(NAMES[:-1]):
                    gain = losses[:, i]-losses[:, -1]
                    center = float(gain.mean())
                    rad = bernstein(gain, 2, delta)+2*e
                    paired[name] = dict(estimate=center,
                                       interval=[max(-1., center-rad), min(1., center+rad)])
                fits = setting['root_fits']
                qtrue = cell.get('target_geometry', {}).get('q')
                errors = [f['q']-qtrue for f in fits if not f['fallback']] if qtrue is not None else []
                rows.append(dict(pair=cell['pair'], n=cell['n'], t=cell['t'],
                    family=setting['family'], ratio=setting['ratio'],
                    actual_ratio=setting['actual_ratio'], budget=setting['budget'],
                    scores=scores, paired=paired, reference_interval=[lo, hi],
                    root_failure_rate=sum(f['fallback'] for f in fits)/len(fits),
                    root_rmse=float(np.sqrt(np.mean(np.square(errors)))) if errors else None))
    groups = {}
    for row in rows:
        key = (tuple(row['pair']), row['n'], row['family'], row['ratio'])
        groups.setdefault(key, []).append(row)
    maxima = []
    for (pair, n, family, ratio), items in groups.items():
        risk = {name: dict(estimate=max(r['scores'][name]['mae'] for r in items),
                interval=[max(r['scores'][name]['interval'][0] for r in items),
                          max(r['scores'][name]['interval'][1] for r in items)]) for name in NAMES}
        gains = {name: dict(estimate=risk[name]['estimate']-risk['optimized']['estimate'],
                 interval=[risk[name]['interval'][0]-risk['optimized']['interval'][1],
                           risk[name]['interval'][1]-risk['optimized']['interval'][0]]) for name in NAMES[:-1]}
        maxima.append(dict(pair=list(pair), n=n, family=family, ratio=ratio,
                           budget=items[0]['budget'], actual_ratio=items[0]['actual_ratio'],
                           risks=risk, grid_gains=gains))
    output = dict(alpha=alpha, cells=rows, references=reference, grid_maxima=maxima,
                  unavailable_baselines=[g for g in saved['groups'] if 'unavailable' in g],
                  coverage='Simultaneous reference and conditional empirical-Bernstein bounds; conditional fixed pool')
    (folder/'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')

    pairs = sorted({tuple(c['pair']) for c in cells})
    batches = sorted({c['n'] for c in cells})
    fig, axes = plt.subplots(len(batches), len(pairs), squeeze=False,
                             figsize=(4*len(pairs), 3.3*len(batches)))
    for i, n in enumerate(batches):
        for j, pair in enumerate(pairs):
            ax = axes[i, j]
            chosen = sorted([c for c in reference if tuple(c['pair']) == pair and c['n'] == n], key=lambda c:c['t'])
            x = [c['t'] for c in chosen]
            y = np.array([c['estimate'] for c in chosen])
            ax.errorbar(x, y, yerr=[y-[c['interval'][0] for c in chosen],
                                   [c['interval'][1] for c in chosen]-y], fmt='o', label='SGD, simultaneous interval')
            ax.plot(x, ndtr(x), '--', label='Gaussian Phi(t)')
            ax.set(title=f'{pair[0]}/{pair[1]}, n={n}', xlabel='Hidden seed coordinate t', ylabel='Right risk-exit probability', ylim=(-.04, 1.04))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder/'direction.png', dpi=160); plt.close(fig)
    for pair in pairs:
        columns = sorted({(r['family'], r['ratio']) for r in rows if tuple(r['pair']) == pair}, key=str)
        # Decreasing budgets occupy one column even though their r varies with n.
        columns = [('fixed', r) for r in sorted({r for f, r in columns if f == 'fixed'})]+([('decreasing', None)] if any(f == 'decreasing' for f, _ in columns) else [])
        fig, axes = plt.subplots(len(batches), len(columns), squeeze=False,
                                 figsize=(4*len(columns), 3*len(batches)))
        for i, n in enumerate(batches):
            for j, (family, ratio) in enumerate(columns):
                ax = axes[i, j]
                chosen = sorted([r for r in rows if tuple(r['pair']) == pair and r['n'] == n and r['family'] == family
                                 and (ratio is None or r['ratio'] == ratio)], key=lambda r:r['t'])
                for name in NAMES:
                    ax.plot([r['t'] for r in chosen], [r['scores'][name]['mae'] for r in chosen], marker='.', label=name)
                ax.set(title=f'n={n}, {family}, r={chosen[0]["actual_ratio"]:.3g}' if chosen else f'n={n}: unavailable', xlabel='Hidden t', ylabel='Probability MAE')
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7); fig.tight_layout()
        fig.savefig(folder/f'risk_{pair[0]}_{pair[1]}.png', dpi=160); plt.close(fig)
        fig, axes = plt.subplots(1, len(columns), squeeze=False, figsize=(4*len(columns), 3.6))
        for j, (family, ratio) in enumerate(columns):
            ax = axes[0, j]
            chosen = sorted([r for r in maxima if tuple(r['pair']) == pair and r['family'] == family and (ratio is None or r['ratio'] == ratio)], key=lambda r:r['n'])
            for name in NAMES:
                ax.plot([r['n'] for r in chosen], [r['risks'][name]['estimate'] for r in chosen], marker='o', label=name)
            ax.set(xscale='log', title=f'{family}, r={ratio}' if ratio else family, xlabel='Batch n', ylabel='Local-grid maximum MAE')
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7); fig.tight_layout()
        fig.savefig(folder/f'scaling_{pair[0]}_{pair[1]}.png', dpi=160); plt.close(fig)
    lines = ['# 未知局部视觉环境：独立事前预测', '',
             '固定官方测试池上的条件实验。视觉重加权改变实际标签风险，不是严格共同风险构造。', '',
             '所有经验根失败均以 .5 回退并计入 MAE。区间同时计入参考路径、删失与校准重复误差；网格最大值不是连续窗口或总体 minimax 证书。', '',
             '|类对|n|序列|实际 r|M|root 失败率最大值|矩最大 MAE|优化最大 MAE|矩减优化区间|',
             '|---|---:|---|---:|---:|---:|---:|---:|---|']
    for row in maxima:
        failure = max(r['root_failure_rate'] for r in rows if r['pair'] == row['pair'] and r['n'] == row['n'] and r['family'] == row['family'] and r['ratio'] == row['ratio'])
        lo, hi = row['grid_gains']['moment']['interval']
        lines.append(f"|{row['pair']}|{row['n']}|{row['family']}|{row['actual_ratio']:.3g}|{row['budget']}|{failure:.3f}|{row['risks']['moment']['estimate']:.4f}|{row['risks']['optimized']['estimate']:.4f}|[{lo:.4f}, {hi:.4f}]|")
    lines += ['', f"基准不适用组数：{len(output['unavailable_baselines'])}；不从可用组外推全部类对。", '',
              '完整单元区间、配对改善、删失、事件差异、经验根误差与失败率见 summary.json。',
              '固定 r 序列检验误差平台；decreasing 序列只在有限预算网格上检验下降趋势，不把冻结有限阶梯规则称为一致估计器。']
    (folder/'analysis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=.05)
    args = p.parse_args()
    run(args.input, args.alpha)

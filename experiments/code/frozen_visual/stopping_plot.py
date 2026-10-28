"""Summarize saved stopping decisions; never choose policy parameters from risks."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(folder):
    result = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    if not result['protocol']['complete']:
        raise ValueError('Run is incomplete; finish computation before summarizing')
    groups = defaultdict(list)
    metrics = ['path_violation', 'final_violation', 'gain', 'hindsight_gain_gap',
               'mean_steps', 'frozen_fraction']
    for case in result['cases']:
        within = defaultdict(list)
        for row in case['records']:
            key = (row['method'], row['parameter'], row.get('budget', 0))
            within[key].append(row)
        for key, rows in within.items():
            groups[key].append({**{m: float(np.mean([r[m] for r in rows])) for m in metrics},
                'availability': float(np.mean([r.get('forecast_available', True) for r in rows]))})
    summary = []
    for (method, parameter, budget), rows in sorted(groups.items()):
        summary.append(dict(method=method, parameter=parameter, budget=budget,
            cases=len(rows), **{m: float(np.mean([r[m] for r in rows]))
                               for m in metrics+['availability']}))
    (folder/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    curves = defaultdict(list)
    for r in summary:
        label = r['method']+(f' M={r["budget"]}' if r['budget'] else '')
        curves[label].append(r)
    for label, rows in curves.items():
        rows.sort(key=lambda r:r['parameter'])
        axes[0].plot([r['path_violation'] for r in rows], [100*r['gain'] for r in rows],
                     marker='o', label=label, markersize=4)
        axes[1].plot([r['path_violation'] for r in rows], [100*r['gain'] for r in rows],
                     marker='o', label=label, markersize=4)
        if 'forecast' in label or 'calibrated' in label:
            axes[2].plot([r['parameter'] for r in rows], [r['path_violation'] for r in rows],
                         marker='o', label=label, markersize=4)
    axes[0].set(xlabel='Empirical path violation rate', ylabel='Gain over frozen (percentage points)')
    axes[1].set(xlabel='Empirical path violation rate', ylabel='Gain (percentage points)',
                xlim=(-.004,.12), ylim=(-.6,3.1), title='Low-risk region')
    axes[2].plot([0,.2], [0,.2], 'k--', linewidth=1)
    axes[2].set(xlabel='Nominal exit probability budget', ylabel='Empirical path violation rate')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.suptitle('Independent-path finite-pool pilot; plug-in forecasts are not certificates')
    fig.tight_layout()
    fig.savefig(folder/'tradeoff.png', dpi=170)
    plt.close(fig)
    p = result['protocol']
    lines = ['# 停止闭环首轮结果', '',
        f"{len(result['cases'])} 个预定设置，每组 {p['paths']} 条新独立 SGD 路径，预算 {p['steps']} 步，风险容忍度 {p['epsilon']}。",
        '先在每个 case 内平均校准重复，再对 case 等权汇总。全部不适用预测以冻结回退计入；不筛选成功子集。', '',
        '## 信息与解释', '',
        '- oracle_forecast 使用精确目标场和有标签目标风险端点，仅作理想条件诊断。',
        '- calibrated_oracle 使用独立无标签场估计和目标 oracle 风险端点；calibrated_source 使用相同场估计和原训练端点。二者均不是完整无标签风险证书。',
        '- disagreement_gate 使用精确目标无标签经验边际，先检查候选再部署；其零违规仅针对固定经验池。没有估计有限样本分歧。',
        '- oracle_checkpoint 是全预算逐路径风险最低点，可能此前已越界；仅为 hindsight 收益参考。',
        '- 名义 alpha 是近似退出 CDF 的分位选择，不是已证明的违规率上界。未拟合安全余量。',
        '- M 为固定池重复观测次数；校准重复在 batch/offset 间共享，不增加独立图像或 SGD 路径。',
        '- 图中参数连线仅展示预定扫描，不是经过独立验证的参数选择器。没有实现或声称复现已发表的风险监测基线。',
        '- 本轮仅为 p=.5、n=128、三个 offset 的小规模筛选，不支持一般 TTA 或总体风险结论。', '',
        '## 全部设置汇总', '',
        '| 方法 | 参数 | M | 可用率 | 路径违规率 | 最终违规率 | 收益(pp) | 相对 hindsight 损失(pp) | 平均步数 | 冻结率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['method']} | {r['parameter']:g} | {r['budget']} | {r['availability']:.3f} | {r['path_violation']:.4f} | {r['final_violation']:.4f} | {100*r['gain']:.3f} | {100*r['hindsight_gain_gap']:.3f} | {r['mean_steps']:.1f} | {r['frozen_fraction']:.3f} |")
    lines += ['', '逐 case 与逐重复决策见 results.json，逐步阈值及风险见 paths.npz。此表不把共享校准和共享路径当成独立重复，不给出伪独立的总体置信区间。', '']
    (folder/'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    run(parser.parse_args().input)

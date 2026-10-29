"""Plot target-centered calibration scaling from saved results only."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def wilson(k, n):
    z = norm.ppf(.975)
    p = k/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return max(0., center-half), min(1., center+half)


def run(folder):
    result = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    if not result['protocol']['complete']:
        raise ValueError('Computation is incomplete')
    rows = []
    for group in result['groups']:
        for setting in group['settings']:
            predictions = np.asarray(setting['predictions'])
            errors = np.asarray(setting['root_errors'])
            row = {k:setting[k] for k in ['family','target_ratio','actual_ratio','n','budget','success_rate','theory_mae']}
            row['pair'] = group['pair']
            if len(predictions):
                absolute = abs(predictions-.5)
                row.update(root_rmse=float(np.sqrt(np.mean(errors**2))),
                    root_mae=float(np.mean(abs(errors))),
                    centered_mae=float(absolute.mean()),
                    centered_mae_se=float(absolute.std(ddof=1)/np.sqrt(len(absolute))) if len(absolute)>1 else None,
                    mean_prediction=float(predictions.mean()),
                    prediction_quantiles=np.quantile(predictions,[.1,.5,.9]).tolist(),
                    failure_inclusive_mae_bounds=[float(setting['success_rate']*absolute.mean()),
                        float(setting['success_rate']*absolute.mean()+(1-setting['success_rate'])*.5)])
            record = next((r for r in group['sgd'] if r['n']==setting['n']), None)
            if record is not None:
                sample = record['component']
                lo = wilson(sample['right'],sample['paths'])[0]
                hi = wilson(sample['right']+sample['censored'],sample['paths'])[1]
                row['component_pi_interval_95'] = [lo,hi]
                if len(predictions):
                    row['mae_over_pi_interval'] = [float(np.mean(np.maximum(np.maximum(lo-predictions,predictions-hi),0))),
                        float(max(np.mean(abs(predictions-lo)),np.mean(abs(predictions-hi))))]
            rows.append(row)
    (folder/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
    fig, axes = plt.subplots(1,3,figsize=(15,4.6))
    for group in result['groups']:
        pair = group['pair']
        for family, ratio in [('fixed_R',r) for r in result['protocol']['ratios']]+[('vanishing_R',None)]:
            chosen = [r for r in rows if r['pair']==pair and r['family']==family
                      and (ratio is None or r['target_ratio']==ratio) and 'centered_mae' in r]
            chosen.sort(key=lambda r:r['n'])
            label = f'{pair[0]}/{pair[1]} '+(f'R={ratio:g}' if ratio is not None else 'R decreasing')
            style = '--' if family=='vanishing_R' else '-'
            axes[0].plot([r['n'] for r in chosen],[r['root_rmse'] for r in chosen],style,marker='.',label=label)
            axes[1].plot([r['n'] for r in chosen],[r['centered_mae'] for r in chosen],style,marker='.',label=label)
            axes[2].scatter([r['actual_ratio'] for r in chosen],[r['centered_mae'] for r in chosen],s=20)
    rgrid = np.linspace(0, max(result['protocol']['ratios'])*1.05, 200)
    axes[2].plot(rgrid,np.arctan(rgrid)/np.pi,'k--',label='arctan(R)/pi')
    axes[0].set(xscale='log',yscale='log',xlabel='Batch size n',ylabel='Absolute root RMSE')
    axes[1].set(xscale='log',xlabel='Batch size n',ylabel='Mean |prediction - 0.5|')
    axes[2].set(xlabel='Relative calibration scale R',ylabel='Mean |prediction - 0.5|')
    axes[1].legend(fontsize=6,ncol=2)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Target-centered empirical-root predictor: conditional finite-pool study')
    fig.tight_layout()
    fig.savefig(folder/'scaling.png',dpi=170)
    plt.close(fig)
    p = result['protocol']
    lines = ['# 目标中心化校准尺度实验', '',
        f"固定eta={p['eta']}、p={p['prior']}；{len(result['groups'])}类对、batch={p['batches']}；每预算{p['repeats']}次独立校准、每batch {p['paths']}条新SGD、最多{p['steps']}步。", '',
        '## 预定检验及信息边界', '',
        '- fixed_R设置按真实目标曲率选择M，使R约为.5/1/2；vanishing_R对照令M正比n平方。参数只依赖几何，不依赖退出观测。',
        '- 初始化x0=q和风险边界来自精确目标经验池，是受控oracle条件。估计器只接收无标签分数与计数。',
        '- 当前检验局部经验根方向预测器Phi(sqrt(n)(q-qhat)/shat)，不是完整共轭时间CDF，也不是所有预测器的普适下界。',
        '- 中心化MAE相对理论极限.5评分；有限n真实方向概率另由独立SGD检查，不把.5当实测真值。',
        '- 校准失败不强行补概率：给出成功率及失败预测任取[0,1]时的全重复MAE界。成功样本的MAE及SE单列；共享嵌套预算不可当独立重复。',
        '- Wilson区间是逐组95%模拟区间，不是跨组同时覆盖；删失加入右方向上界。相对有限n方向概率的MAE范围另存summary.json，未把SGD估计当无误差真值。',
        '- 四组均为同一固定特征池的条件分析。M增长不增加独立图像；根扫描非严格区间认证。', '',
        '## 独立真实SGD方向检查', '',
        '| 类对 | n | 分量右退出 / 总数 | 分量删失 | 分量方向概率区间95% | 风险右退出 / 总数 | 风险删失 | 两事件时间不同路径数 |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for group in result['groups']:
        for record in group['sgd']:
            c,r = record['component'],record['risk']
            interval=[wilson(c['right'],c['paths'])[0],wilson(c['right']+c['censored'],c['paths'])[1]]
            lines.append(f"| {group['pair']} | {record['n']} | {c['right']}/{c['paths']} | {c['censored']} | [{interval[0]:.3f}, {interval[1]:.3f}] | {r['right']}/{r['paths']} | {r['censored']} | {record['different_event_paths']} |")
        if 'sgd_unavailable' in group:
            lines.append(f"\n{group['pair']}不适用风险区间：{group['sgd_unavailable']}。校准记录仍保留。\n")
    lines += ['', '## 逐设置校准结果', '',
        '| 类对 | 序列 | n | M | R | 可用率 | 根RMSE | 中心化MAE | 校准SE | 理论MAE |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values = [f"{r[k]:.4f}" if r.get(k) is not None else 'NA'
                  for k in ['root_rmse','centered_mae','centered_mae_se']]
        lines.append(f"| {r['pair']} | {r['family']} | {r['n']} | {r['budget']} | {r['actual_ratio']:.3f} | {r['success_rate']:.3f} | {' | '.join(values)} | {r['theory_mae']:.4f} |")
    lines += ['', '原始逐重复根/曲率/噪声见results.json；首次分量/风险退出时间与方向见samples.npz；汇总及有限n方向误差范围见summary.json。不能只凭均值接近曲线断言定理已被完整验证。', '']
    (folder/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    for family in ['fixed_R','vanishing_R']:
        for n in p['batches']:
            subset=[r for r in rows if r['family']==family and r['n']==n and 'centered_mae' in r]
            print(family,n,[(r['pair'],round(r['actual_ratio'],3),round(r['centered_mae'],4),round(r['root_rmse'],5)) for r in subset])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    run(parser.parse_args().input)

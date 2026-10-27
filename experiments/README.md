# 熵阈值实验

代码按研究用途统一放在 [code/entropy/](code/entropy/)，使用 NumPy、SciPy、Matplotlib。[STATUS.md](../STATUS.md) 是当前结果与下一步的唯一入口。

| 入口 | 用途 | phase |
|---|---|---|
| [entropy_pilot.py](code/entropy/entropy_pilot.py) | 基础网格与轨迹 | compute、plot |
| [entropy_geometry.py](code/entropy/entropy_geometry.py) | 分布几何、确定性时长、正漂移区间 SGD | compute、plot |
| [entropy_critical.py](code/entropy/entropy_critical.py) | 临界非线性 SGD 与 AR 对照 | compute、refine、plot |
| [entropy_check.py](code/entropy/entropy_check.py) | 独立积分、梯度与 CDF 数值检查 | 直接运行 |

从仓库根目录运行。计算和绘图分进程执行，未指定 phase 时只计算：

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_pilot.py --phase compute
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_pilot.py --phase plot

& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_geometry.py --phase compute
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_geometry.py --phase plot

& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_critical.py --phase compute
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_critical.py --phase plot
```

临界 SGD 的 compute 包括有限参数几何计算；对已有轨迹仅更新几何修正时使用 --phase refine，然后另行 --phase plot，不重新模拟。数值检查可独立运行：

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_critical.py --phase refine
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_check.py
```

历史输出保持原位：pilot 在本目录，几何结果在 [theory_progress_reproducibility/](../theory_progress_reproducibility/)，临界结果在 [critical_sgd_reproducibility/critical_sgd/](../critical_sgd_reproducibility/critical_sgd/)。旧目录仅保存结果、图、日志和分析，代码不按交付轮次组织。

所有入口支持 --output，默认使用各自历史目录并会重写同名输出。新实验参数使用独立目录，例如：

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_geometry.py --phase compute --output experiments/results/geometry
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_geometry.py --phase plot --output experiments/results/geometry
```

同一结果的 compute、refine、plot 和 check 使用同一个 --output。plot 读取保存结果，不重新抽样；geometry 绘图仅重新计算便宜的确定性示例曲线。诊断不是实验运行或汇总的门槛。

结果分析分别保存在 [pilot 记录](RESULTS.md)、[几何分析](../theory_progress_reproducibility/analysis.md) 和 [临界分析](../critical_sgd_reproducibility/critical_sgd/analysis.md)。

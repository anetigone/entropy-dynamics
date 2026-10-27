# 理论与基础合成复现

本目录保留既有理论复现代码。当前结果与待办见 [STATUS.md](../../../STATUS.md)，其它实验见 [实验导航](../../README.md)。依赖 NumPy、SciPy；绘图另需 Matplotlib。

| 计算文件 | 用途 | 绘图文件 |
|---|---|---|
| [entropy_pilot.py](entropy_pilot.py) | 基础扫描与合成轨迹 | [pilot_plot.py](pilot_plot.py) |
| [entropy_geometry.py](entropy_geometry.py) | 几何、安全时长和正漂移 SGD | [geometry_plot.py](geometry_plot.py) |
| [entropy_critical.py](entropy_critical.py) | 临界 SGD、AR 对照及有限参数修正 | [critical_plot.py](critical_plot.py) |

[entropy_check.py](entropy_check.py) 独立进行数值检查。计算和绘图使用不同文件，不带 `--phase`。

从仓库根目录运行，例如：

```shell
python experiments/code/entropy/entropy_pilot.py
python experiments/code/entropy/pilot_plot.py
```

这些既有入口目前使用模块顶部的实验参数与输出常量，不提供命令行参数。修改预算或输出位置时直接修改对应常量，绘图复用计算模块的路径。默认 pilot 输出在 `experiments/`，几何结果在 `experiments/results/geometry/`，临界结果在 `experiments/results/critical/`。这些入口可能重写同名文件，新参数应使用独立输出位置。

临界 SGD 计算末尾会进行有限参数几何修正。已有轨迹可通过模块中的 `refine_saved_results()` 单独重算修正，再另行绘图，无需重新模拟。混合偏移实验复用本目录的高斯几何和真实 SGD 实现，不复制模型代码。

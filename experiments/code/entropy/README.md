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

## 未知局部方向检查

[unknown_directions.py](unknown_directions.py)从连续标准正态基准的有界倾斜密度拒绝采样，运行真实非线性熵 SGD，不使用离散化池代替未来样本。方向族为 `k=t*psi/c²+a*(tanh(z)²-E0 tanh(Z)²)`：a=0 是最不利方向，非零 a 添加与奇影响函数正交的偶干扰。预测器只用基准影响函数及独立校准样本，不接收方向、t 或真实根。不能由一个有限方向列表宣称验证整个无限维 minimax 定理。

所有环境共用 `g=.5*tanh(z)*f0`，风险安全集严格为给定区间；代码从边界积分得到相同风险容忍度。开始采样前保存 design.json，检查共同有界包络、联合密度非负条件、根位置及局部斜率网格。根、信息和场几何用 Gaussian quadrature 数值诊断，斜率网格不是理论条件的严格证书。拒绝采样、预测和 SGD 使用连续分数；校准与参考随机流独立。

```shell
python experiments/code/entropy/unknown_directions.py --output experiments/results/unknown_directions --batches 128 512 --targets -1 0 1 --amplitudes 0 -.5 .5 --ratios .5 1 2 --paths 1024 --repeats 256
python experiments/code/entropy/directions_plot.py --input experiments/results/unknown_directions
```

参数另有 `--envelope`、`--eta`、`--boundary`、`--steps`、`--block`、`--seed`；默认包络 2、eta=.1、边界 .25、1500 步、随机种子 73192。M 按 round(c²n/r²) 确定；不同 M 独立采样，不用于跨预算配对检验。默认配置是可修改的数值检查配置，正式使用前按 STATUS.md 固定设计。已有输出拒绝覆盖，部分结果保留，不支持续跑。

[directions_plot.py](directions_plot.py)在独立进程汇总参考方向概率、Phi(t)、删失及原始矩预测 MAE，生成 summary.json、analysis.md 与 risk.png。默认 alpha=.05，一半分配同时参考界，一半分配所有有限单元的 MAE Bernstein 界；参考误差加入 MAE 半径。不提供连续窗口、有限 n 最优性或方法间显著性声明。

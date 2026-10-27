# 混合偏移机制实验

[entropy_mixed.py](entropy_mixed.py) 扫描平移和先验偏移的抵消关系，计算真实 fresh-batch SGD 的方向与首次风险越界时间。[mixed_plot.py](mixed_plot.py) 单独读取结果绘图。当前结果与待办见 [STATUS.md](../../../STATUS.md)，其它实验见 [实验导航](../../README.md)。依赖 NumPy、SciPy、Matplotlib。

从仓库根目录运行：

```shell
python experiments/code/mixed_shift/entropy_mixed.py --output experiments/results/mixed_shift
python experiments/code/mixed_shift/mixed_plot.py --input experiments/results/mixed_shift
```

计算参数：

| 参数 | 含义与默认值 |
|---|---|
| `--output` | 结果目录；默认 `experiments/results/mixed_shift` |
| `--batches` | batch 列表；默认 `32 128 512` |
| `--kappas` | 先验强度列表；默认 `0 1 2` |
| `--offsets` | 抵消点两侧的平移偏移；默认 `-2 -1 0 1 2` |
| `--paths` | 每组随机轨迹数；默认 1000 |
| `--max-steps` | 最大更新次数；默认 2500 |
| `--seed` | 随机种子；默认 20261027 |

绘图的 `--input` 指向保存 `results.json` 与 `samples.npz` 的目录，图片写入同目录。计算目录已存在时会提示更换目录，不覆盖结果。所有参数可用 `--help` 查看；不使用 `--phase`。

比较平衡极限、真实参数局部线性、完整共轭，以及分别去掉峰值、曲率、噪声和风险端点修正的预测；参数由模型独立计算，不从退出曲线拟合。复用 `../entropy/entropy_critical.py` 的高斯模型及 `../common/entropy_prediction.py` 的退出分布工具。

输出保留左右联合退出 CDF、方向、退出时间及删失轨迹。首次越界相对冻结模型风险加容忍度定义，不等于最佳停止时间；`times=-1` 表示预算内未观测到越界，CDF 分母保留这些路径。

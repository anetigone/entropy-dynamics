# 几何与安全时长：历史结果

本目录保存交付的 results.json、sgd_crossing_times.npz、theory_checks.png 和 theory_checks.svg；分析见 [analysis.md](analysis.md)。当前证据与下一步见根目录 [STATUS.md](../STATUS.md)。

代码已统一为 [entropy_geometry.py](../experiments/code/entropy/entropy_geometry.py)，用途和完整运行说明见 [实验 README](../experiments/README.md)。从仓库根目录运行：

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_geometry.py --phase compute
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_geometry.py --phase plot
```

默认输出到本目录。只重绘时仅运行 plot；新输出可用 --output 指定独立目录。计算与绘图分进程运行，均可能重写同名文件。

活跃证明见 [几何笔记](../paper/notes/entropy_geometry.md)；交付包附带的原始理论副本保存在 [archive](../paper/notes/archive/theory_package.md)。

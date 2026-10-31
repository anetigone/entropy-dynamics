# 实验导航

结果与原处分析见 [结果导航](results/README.md)。当前研究状态和下一步见 [STATUS.md](../STATUS.md)，工作纪律与本机环境约定见 [AGENTS.md](../AGENTS.md)。

实验代码按功能分组，各目录的 README 说明用途、依赖、参数和运行方式：

- [理论与基础合成复现](code/entropy/README.md)：基础扫描、几何与安全时长、临界 SGD 和独立数值检查。
- [混合偏移机制实验](code/mixed_shift/README.md)：平移与先验抵消、方向和退出时间预测消融。
- [冻结视觉特征实验](code/frozen_visual/README.md)：特征提取、独立校准、尺度和 LA 对照。
- [CIFAR-10-C 独立预测](code/corrupt_visual/README.md)：腐蚀特征、固定源端初始化、独立无标签校准与风险退出方向评价。
- [BN/Tent 投影小试](code/bn_probe/README.md)：完整多分类 BN 更新、冻结投影、整批噪声和有限时域风险预测。
- [Tent 失效概率预测开发](code/tent_risk/README.md)：固定启动状态、多条独立未来、无标签完整批次探针和冻结概率模型比较。

共用的退出分布与含删失 CDF 数值模块在 [code/common/](code/common/)。计算与绘图分别运行，不使用 `--phase`；数据、模型、缓存和输出位置使用各脚本提供的参数。不同参数的计算结果使用独立目录，已有结果保持原位。

高斯证书与既有确认实验的材料通过[结果导航](results/README.md)查阅；完整理论通过[稿件导航](../paper/README.md)查阅。

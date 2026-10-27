# 实验导航

当前研究状态、采用的证据和下一步见 [STATUS.md](../STATUS.md)，工作纪律与本机环境约定见 [AGENTS.md](../AGENTS.md)。

实验代码按功能分组，各目录的 README 说明用途、依赖、参数和运行方式：

- [理论与基础合成复现](code/entropy/README.md)：基础扫描、几何与安全时长、临界 SGD 和独立数值检查。
- [混合偏移机制实验](code/mixed_shift/README.md)：平移与先验抵消、方向和退出时间预测消融。
- [冻结视觉特征实验](code/frozen_visual/README.md)：特征提取、独立校准、尺度和 LA 对照。

共用的退出分布与含删失 CDF 数值模块在 [code/common/](code/common/)。计算与绘图分别运行，不使用 `--phase`；数据、模型、缓存和输出位置使用各脚本提供的参数。不同参数的计算结果使用独立目录，已有结果保持原位。

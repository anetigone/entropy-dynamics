# 实验结果导航

本文件只索引结果用途与原处分析。当前研究主线及下一步见 [STATUS.md](../../STATUS.md)；代码和运行方式见 [实验导航](../README.md)。结果目录、内部文件及服务器来源记录保持原位。

| 结果目录 | 用途 | 分析入口 |
|---|---|---|
| [bn_extended](bn_extended/) | BN/Tent平均路径、随机残差与风险适用性检查 | [分析](bn_extended/analysis.md) |
| [bn_development](bn_development/) | 完整BN/Tent投影、动力学与有限时域风险开发检查 | [分析](bn_development/analysis.md) |
| [corrupt_confirmation](corrupt_confirmation/) | 自然腐蚀方向校准的独立预算比较 | [分析](corrupt_confirmation/analysis.md) |
| [unknown_directions](unknown_directions/) | 共同风险下非对齐局部方向数值检查 | [分析](unknown_directions/analysis.md) |
| [precision confirmation](precision_confirmation) | 冻结规则、独立双层精度确认 | [原处分析](precision_confirmation/analysis.md) |
| [gaussian_decision](gaussian_decision) | 有限信息高斯 minimax 证书与真实熵 SGD pilot | [原处分析](../../paper/gaussian_decision.md) |
| [local_scaling](local_scaling/) | 固定初始化、未知局部视觉环境的独立预测与预算尺度 | [分析](local_scaling/analysis.md) |
| [local_confirmation](local_confirmation/) | 视觉局部环境的独立精度确认 | [固定设计与分析](local_confirmation/analysis.md) |
| [corrupt_prediction](corrupt_prediction/) | CIFAR-10-C 自然偏移下的迁移、校准与风险退出诊断 | [分析](corrupt_prediction/analysis.md) |
| [centered_calibration](centered_calibration/) | 目标中心化校准尺度、独立SGD与汽车/卡车删失定位 | [分析](centered_calibration/analysis.md) |
| [stopping_pilot](stopping_pilot/) | 停止闭环初次筛选 | [分析](stopping_pilot/analysis.md) |
| [stopping_repeat](stopping_repeat/) | 停止闭环独立随机种子重复 | [分析](stopping_repeat/analysis.md) |
| [mixed_shift](mixed_shift/) | 混合偏移抵消与退出方向 | [分析](mixed_shift/analysis.md) |
| [visual_shift](visual_shift/) | 冻结视觉阈值实验及几何替换诊断 | [分析](visual_shift/analysis.md) |
| [calibration_pilot](calibration_pilot/) | 独立无标签场校准小规模实验 | [分析](calibration_pilot/analysis.md) |
| [calibration_high](calibration_high/) | 高预算校准核对 | [分析](calibration_high/analysis.md) |
| [calibration_grid](calibration_grid/) | 多设置独立校准网格 | [分析](calibration_grid/analysis.md) |
| [exit_geometry](exit_geometry/) | 安全分量和退出域几何检查 | [分析](exit_geometry/analysis.md) |
| [exit_tail](exit_tail/) | 新独立路径长尾、分量跳跃及沿程方差 | [分析](exit_tail/analysis.md) |
| [tail_noise](tail_noise/) | 近峰/远端持续噪声消融 | [分析](tail_noise/analysis.md) |
| [tail_inner](tail_inner/) | 较小分区半径的噪声消融 | [分析](tail_inner/../tail_noise/analysis.md) |
| [geometry](geometry/) | 分布几何与安全时长 | [分析](geometry/analysis.md) |
| [critical](critical/) | 临界SGD、容忍度与AR对照 | [分析](critical/analysis.md) |
| [visual_features](visual_features/) | 官方 train/test 冻结特征及提取协议 | [protocol.json](visual_features/protocol.json) |

基础探索结果保持原位置：[pilot 分析](../RESULTS.md)、[汇总](../pilot_summary.json)、[网格](../population_grid.csv)。

视觉几何替换的原始输出位于 [visual_shift/diagnosis](visual_shift/diagnosis/)，解释集中在视觉分析中。数值结论、适用率、删失与信息限制以各分析为准；本导航不复制结果摘要。

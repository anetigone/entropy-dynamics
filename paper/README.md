# 论文文件导航

## 主文与连续阅读

[主文 DRAFT.md](DRAFT.md)按共同风险下的独立事前预测问题组织：信息与损失 → 共同风险构造 → 统一真实 SGD 方向退出 → 样本门槛与有限信息精度 → 未知方向与 nuisance → 高斯证书和固定规则确认 → 文献与限制。

可编译的英文双栏主文与完整补充材料见 [LaTeX 入口](latex/README.md)。工作稿暂不限制篇幅。当前研究安排统一见 [STATUS.md](../STATUS.md)。

主文四个核心定理及其展开证明通过 L/G/N 编号连接源证明。正文展开统一退出、双向实验归约、未知方向上下界、证书与双层区间；补充材料保留完整源章节与支撑结果，不替代原证明编号：

| 文件 | 内容 |
|---|---|
| [local_prediction.md](local_prediction.md) | L.1–L.8：共同风险构造、统一方向退出、信息门槛、低信息 minimax 常数和指定矩预测器风险 |
| [Gaussian decision extension](../gaussian_decision/gaussian_decision_extension.md) | L.9–L.10、G 方程：完整统计实验等价、minimax 风险转移、高斯数值证书及早期 pilot |
| [Unknown-direction theory](../entropy_prediction_revision/contribution_precision/unknown_direction_theory.md) | N.1–N.4：已知基准下未知有界局部方向、最不利熵倾斜、统一退出与 nuisance 投影 |
| [Fixed-design confirmation](../entropy_prediction_revision/contribution_precision/precision_report.md) | 冻结规则的真实 SGD 确认、参考和校准双层同时区间；完整分析保留原位 |
| [Contribution and literature](../entropy_prediction_revision/contribution_precision/contribution_and_literature.md) | 问题定位、信息设定和主要文献的精确比较 |

建议先读 DRAFT.md，再顺次阅读上表的完整证明和分析。所有理论定义与论证均在稿件部分中给出，不依赖状态日志。包中的 source_manuscript 与旧副本作为用户提供的原始材料保留，不另维护成主文。

## 支撑理论与既有实验

| 文件 | 内容 |
|---|---|
| [gaussian_model.md](gaussian_model.md) | 原高斯混合模型、命题1–12陈述、rebound、容忍度和整数退出时间；保留原章节编号供证明引用 |
| [gaussian_proofs.md](gaussian_proofs.md) | 原高斯命题1–12的完整证明 |
| [scalar_exit.md](scalar_exit.md) | G1–G4、E.1–E.3：一般标量真实 SGD、共轭与整数时间退出近似 |
| [calibration.md](calibration.md) | E.4–E.8：共轭稳定性、独立校准、经验根与随机预测极限 |
| [risk_boundaries.md](risk_boundaries.md) | E.9–E.13：偏差、端点、跨分量跳跃、经验池与校准初始化 |
| [experiments.md](experiments.md) | 既有实验及原处分析入口；§10.14链接固定规则确认 |
| [RELATED_WORK.md](RELATED_WORK.md) | 原高斯风险与时间口径的详细比较；新增统计文献链接到上表原处对照 |

支撑部分保留原数字与 E 编号，不与主文四个汇总定理混用。原高斯推导中的局部更新幅度记号 M 不代表校准样本量；统计主文统一使用 B_U 表示幅度界，M 表示校准预算。

[notes/entropy_geometry.md](notes/entropy_geometry.md)、[notes/critical_sgd.md](notes/critical_sgd.md)保留理论背景。[notes/archive/](notes/archive/)保存历史记录，[references/](references/)保存参考文献 PDF。

## 实验入口

[全部结果与原处分析](../experiments/results/README.md)及[代码运行导航](../experiments/README.md)。分析、证书与既有结果保持原位；本文件只负责导航。当前状态与待办只在 [STATUS.md](../STATUS.md)维护。

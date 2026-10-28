# 论文文件导航

## 稿件阅读顺序

本稿由多个文件共同组成，不要求 DRAFT.md 单独包含全部证明和实验。原高斯理论保留原命题编号，一般标量与校准扩展使用 E 编号；两套编号互不替代。

| 文件 | 内容 |
|---|---|
| [DRAFT.md](DRAFT.md) | 摘要、引言、模型、高斯理论陈述、§8A 扩展主线、讨论与参考文献 |
| [scalar_exit.md](scalar_exit.md) | 扩展 §1–2；G1–G4、E.1–E.3，一般标量真实SGD及统一整数时间退出近似，含证明 |
| [calibration.md](calibration.md) | 扩展 §3–5；E.4–E.8，共轭稳定性、独立场估计、预测误差及随机预测极限，含证明 |
| [risk_boundaries.md](risk_boundaries.md) | 扩展 §6–9、§11；E.9–E.13，固定偏差、端点扰动、跨分量跳跃、经验池及校准初始化，含证明 |
| [gaussian_proofs.md](gaussian_proofs.md) | 原命题1–12的证明，模型和陈述见 DRAFT.md |
| [experiments.md](experiments.md) | §10 实验协议、表格、消融、删失及数值局限 |

扩展各部分保留连续章节号和公式 E 编号，各自附必要参考文献。正文先给出研究主线，再按链接查阅证明或实验。原高斯部分的更新幅度界用局部记号 M；扩展改用 B_U，把 M 留给校准样本量，跨部分使用时须区分。

[RELATED_WORK.md](RELATED_WORK.md) 是中文文献对照，独立定义比较问题和贡献边界；不代替稿件证明。

## 推导与原始材料

[notes/entropy_geometry.md](notes/entropy_geometry.md) 和 [notes/critical_sgd.md](notes/critical_sgd.md) 保留前两轮推导及其局部编号；涉及尚未完成扩展的文字按原轮次阅读，当前扩展见上述稿件部分。[本轮扩展原件](notes/archive/entropy_exit_theory_extension.md) 原样归档，含作者合并说明，不作为另一份活跃稿件维护。

[notes/archive/](notes/archive/) 保存历史推导及检索记录；[references/](references/) 保存参考文献 PDF。

## 结果材料

逐项结果分析与原始输出保留在实验目录，稿件的实验部分汇总必要协议和数值：

- [基础 pilot](../experiments/RESULTS.md)、[几何](../experiments/results/geometry/analysis.md)、[临界 SGD](../experiments/results/critical/analysis.md)。
- [混合偏移](../experiments/results/mixed_shift/analysis.md)、[冻结视觉与几何诊断](../experiments/results/visual_shift/analysis.md)。
- [独立校准](../experiments/results/calibration_pilot/analysis.md)、[高预算核对](../experiments/results/calibration_high/analysis.md)。

当前研究状态、已完成范围和下一步统一见 [STATUS.md](../STATUS.md)。运行方式见 [实验导航](../experiments/README.md)，工作约定见 [AGENTS.md](../AGENTS.md)。

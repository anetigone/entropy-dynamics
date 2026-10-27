# 当前研究状态

更新：2026-10-27（Asia/Shanghai）。本文件是当前研究状态、已完成工作和下一步安排的唯一入口。导航和命令见 [README.md](README.md)，工作约定见 [AGENTS.md](AGENTS.md)。

## 当前判断

已将两轮外部推导接入 [论文初稿](paper/DRAFT.md)：分布结构决定熵阈值稳定性，随后分析确定性安全时长、正漂移区间首次退出，以及临界区真实 fresh-batch SGD 的方向和越界时间分布。模型内已有带证明的陈述；已补齐几何与安全时长复现包，并把实验代码按用途统一到 experiments/code/entropy/；计算、修正和绘图由同一入口的 phase 参数选择，分进程运行。未进行独立完整复证、完整模拟重跑或新的文献检索。新颖性、视觉有效性和可执行无标签停止仍未完成。

论文文档已改为自包含：初稿纳入完整模型、主要证明、实验设置与必要数值，相关工作独立定义比较问题和文献边界；两份正文不再依赖其它仓库文件。文件之间的关系集中在 paper/README.md。此次仅改写文档，没有新增理论结果、文献检索或实验运行。

## WACV 文献核对

2026-10-27 已阅读 WACV 正文（9 页），并渲染核对理论页；[正文](paper/references/wacv_logit_adjusted.pdf) 和用户补充提供的 [补充材料](paper/references/Weerasinghe_Logit-Adjusted_Test-Time_Adaptation_WACV_2026_supplemental.pdf) 均归入参考文献目录。补充材料尚未阅读；按用户要求，直接接受其全部理论结论成立，不继续审查证明，不把证明疑点用作本项目贡献，也不将该审查作为推进门槛。详细对照集中在 [RELATED_WORK.md §1](paper/RELATED_WORK.md)。先改善后退化、BN shift 对齐、logit 偏置及 LA 缓解漂移均视为已有结果。候选贡献必须在此基础上体现风险首次越界的实质参数律、真实 mini-batch 的联合分布或容忍度时间尺度区别；若可由该文加标准工具直接推出，则不足以作为主贡献。AllAcc/PeAR 与本项目的比较分布及指标不同，LA 应纳入后续直接基线。新颖性仍未确认。

## 理论完成范围

| 模块 | 当前可用结论 | 条件与边界 |
|---|---|---|
| 弱分离几何与反弹 | 唯一不稳定熵峰；显式非空混合偏移反弹区域 | 0<m≤σ；熵下降需步长条件；不是任意分离度的完整相图 |
| 强分离安全收敛 | 平衡时正曲率的显式充分条件及非平衡开放延拓 | population 局部结论；不保证固定步长 SGD 永不退出 |
| 确定性越界时间 | 固定正容忍度下 log(1/ρ)/log(1+ηλ)+O(1)；零及 O(ρ²) 容忍度可为有限步 | 固定 η；初始化与不平衡按指定尺度变化 |
| 正漂移区间 SGD | 直接控制左右首次退出和过晚退出，无全程轨迹指数误差 | 整个安全区间 ν>0；不覆盖包含熵峰的临界安全区 |
| 临界真实 SGD | 共轭坐标中冻结噪声误差；方向和时间联合 CDF 近似 | 固定正 η、ε；距熵峰 O(n^-1/2)；误差常数尚未数值化 |
| 临界尺度与离散相位 | Tε=log n/[2log(1+ηλ)]+O_P(1)，方向由有效种子信噪比决定 | 须保留整数格点相位；不能套用于固定非零先验的所有 batch |
| 微小容忍度 | nε→e 时收敛至有限步 Gaussian AR 首次退出 | 与固定正 ε 的对数时间窗不同 |
| 一步风险限制 | 完全平衡零容忍下 T0=1 a.s.；正步长缩小不能消除走错方向概率 | 仅限制无条件执行当前 batch 熵更新，不是所有无标签方法的不可能性 |

证明分工：[几何笔记](paper/notes/entropy_geometry.md) §2–6；[临界 SGD 笔记](paper/notes/critical_sgd.md) §2–5。初稿统一记号与命题编号，详细笔记保留各自局部编号。第一轮 §8 的“尚未证明”是当时记录，现由第二轮处理；η→0 联合极限和完整相边界仍未解决。

## 采用的实验与数值证据

- [基础 pilot 分析](experiments/RESULTS.md)、[汇总](experiments/pilot_summary.json)、[全网格](experiments/population_grid.csv)、[轨迹](experiments/pilot_paths.npz)、[图](experiments/pilot.png)：30 格探索扫描，8 格出现指定反弹；展示例 δ=0、p=.35 的最低风险在第 47 步，T0.005=94。保留其历史结果含义。
- [临界 SGD 分析](critical_sgd_reproducibility/critical_sgd/analysis.md)、[完整汇总](critical_sgd_reproducibility/critical_sgd/critical_results.json)、[轨迹](critical_sgd_reproducibility/critical_sgd/critical_samples.npz)、[图](critical_sgd_reproducibility/critical_sgd/critical_checks.png)：固定容忍度九组各 600 路径；微小容忍度三组各 2000 路径，AR 参考 150000 路径；一步检查各 12000 batch。保留平衡预测失准、修正后仍超单组 DKW 带的组及宽区间无明显收益的检查。
- [几何诊断结果](critical_sgd_reproducibility/critical_sgd/geometry_audit.json)：现有浮点诊断，不是区间认证或训练/汇总门槛。
- [几何与安全时长分析](theory_progress_reproducibility/analysis.md)、[汇总](theory_progress_reproducibility/results.json)、[2000 条越界时间](theory_progress_reproducibility/sgd_crossing_times.npz)、[图](theory_progress_reproducibility/theory_checks.png)：原缺失附件现已补齐。非零 δ 的 T0.005 为 67 和 124；n=64 的越界时间范围 80–112、中位数 94，与保存轨迹和汇总一致。漂移包围和 .99763428 的下界仍按原报告的浮点精度范围解释，模拟不代替证明。

代码统一放在 [experiments/code/entropy/](experiments/code/entropy/)，入口与命令见 [实验 README](experiments/README.md)。旧复现包目录只保留历史结果、图、日志和分析，结果内容与位置保持不变；新的实验参数应显式使用独立 --output 目录。导入的重复理论副本及历史检索记录放在 [archive](paper/notes/archive/)。

本次验证：四个入口的内存语法检查与 CLI help、迁移后的直接 import、历史默认路径、100 步 population 轨迹及两组各 8 条 fresh-batch 合成 smoke 均通过；三个 plot 入口和可选数值检查在独立进程中读取临时结果副本并通过，生成图片可解码。文档本地链接已检查。未重新运行完整实验，未覆盖历史结果。

## 下一步安排

1. 逐式复核整合后的证明，优先看共轭 C² 延拓、停止噪声尾部、精确退出事件和格点子序列；若要报告可用有限 n 覆盖率，再处理分布误差常数。第一轮数值漂移包围若作为严格计算机辅助证明使用，需要相应数值认证；这不是普通实验运行要求。
2. 按 [相关工作](paper/RELATED_WORK.md)，以 WACV 全部结论正确为前提比较新增结果，继续对照 PeTTA、CED、GALA/STAG、风险监测与经典随机退出。WACV 补充证明审查不再作为任务；其它文献的历史阅读范围未变。经典对数逃逸、高斯种子、卷积保对数凹不计为新发现。
3. 在已知锚的模型中研究可识别的逐步拒绝/停止证据，纳入参数不确定性和锚误差；比较一次校准与直接 plug-in Bayes 阈值。条件 supremum 证书仍只是通用基线。
4. 再开展冻结视觉表征验证，分别操纵先验与位置偏移；使用官方数据划分，标签只作评估，正式训练仅在服务器运行。本机仅小规模合成验证与结果分析。

若结果最终只剩已知不稳定点机制加通用拒绝规则，或视觉表征不支持预测的相区差异，应重新判断论文主线。

# 相关工作与新颖性边界

核查日期：2026-10-24。本文是研究初稿，不将“未检出”写成“确认开放”。B-notes 的历史判断仅作为入口；以下区分本次读到的内容和仍未完成的核查。

## 1. 实际熵梯度与类别偏置：首要对照

[Weerasinghe et al., Logit-Adjusted Test-Time Adaptation under Partial Class Imbalance, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Weerasinghe_Logit-Adjusted_Test-Time_Adaptation_under_Partial_Class_Imbalance_WACV_2026_paper.pdf)，§3.1 / Lemmas 1–3；[官方补充材料](https://openaccess.thecvf.com/content/WACV2026/supplemental/Weerasinghe_Logit-Adjusted_Test-Time_Adaptation_WACV_2026_supplemental.pdf)，Theoretical Analysis。

已核官方搜索索引提供的正文与补充推导段落，直接打开官方 PDF 本次返回 403，不能说完成整篇附件审计。其理论给熵梯度、简化 BN 位移更新的方向对齐和 logit 偏置。补充 Lemma 2 的迭代展开使用固定方向与常量系数；这为当前研究提出“从真实状态依赖梯度推出长期轨迹”的候选差别。但本项目仍须以逐页原文复核为闸门。还有一个审查细节：Cauchy–Schwarz 给不同类别 logit 增量比的范数比上界，若要推出主类增量严格最大，还要检查各 pullback 范数关系；不能直接照抄这一结论作为无条件事实。

**禁止的主张：** 首次发现不平衡使熵适应发生偏置、首次解析 BN shift 漂移。可能保留的区别仅为状态依赖的风险首次越界时间，且须有实质参数律。

## 2. 崩溃与错误反馈

[Hoang, Vo and Do, Persistent Test-time Adaptation in Recurring Testing Scenarios, v4, 2024-11-02](https://arxiv.org/html/2311.18193v4)，§3、Assumption 1、Theorem 1 / Corollary 1、Appendix B。本次核模型和证明入口，全文文本已缓存。PeTTA 已以扰动高斯混合分类器研究持续错误累积，并用其分析指导稳定适应。要逐项对照其伪标签/教师更新与本项目熵阈值递推；模型形式不同不自动提供新颖性。当前未逐行复证附录。

新增近邻：[CED: CLIP-guided entropy dynamics for robust test-time adaptation in harsh visual conditions](https://www.sciencedirect.com/science/article/pii/S0031320326002621)。本次读出版社摘要和结论，尚未审完整证明。已明确涉及熵动态、过度自信和轨迹约束，列为后续全文阻断，不把它当作仅经验方法略过。

## 3. 无标签早停、选层与梯度锚

[Sahoo et al., arXiv:2404.03784v2, 2025-02-23](https://arxiv.org/html/2404.03784v2) 的最新版题为 **A Layer Selection Approach to Test Time Adaptation**，方法为 GALA。旧版题为 **Layerwise Early Stopping for Test Time Adaptation**，LEAST。按同一 arXiv ID 去重。已读 §2 的梯度方向准则及 §5 分析；其在线选层、跳过和重置机制是必要方法对照，不能把“无需标签决定何时更新”作为新贡献。完整版本差异与相应正式发表文本仍待核。

[Shin et al., STAG, arXiv:2402.09004v2, 2026-02-06](https://arxiv.org/html/2402.09004v2)，§3.2、Appendix A.2–A.4。已核结构锚和熵梯度推导的范围；它使用分类器几何提供更新信号。本项目必须说明源锚实际存储了什么，不能把结构锚方向首次引入 TTA 当卖点。此次 HTML 阅读成功，本地抓取超时，未缓存成功。

[Nguyen Phuong et al., Dual Strategies for Test-Time Adaptation, v1](https://arxiv.org/html/2604.17542v1)，§3.3–3.4，双增强可靠性判据与分组熵最小/最大化；arXiv 页面标注 Findings of Computer Vision and Pattern Recognition 2026。已核方法范围，完整证明没有认证。其可靠样本选择与本项目风险证书需区别；网页未检出 Theorem 字样不构成缺乏理论的证据。

## 4. 抽象恢复保证与信息边界

[Zhou et al., On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective, v2, ICML 2026](https://arxiv.org/html/2605.28057v2)，§2–4 / Appendices D–E。本次再次核版本与结构；B-notes 已详细记录局部对齐、曲率、保持在邻域及动态扩展的假设边界。该工作已有恢复复杂度理论。本文拟从混合分布和真实熵递推推导对齐何时失效；不能把有限 batch、依赖流或非平稳本身称为现有理论空白。未完成所有证明复证。

[Jhawar and Wang, When is Test-Time Adaptation Identifiable From Unlabeled Evidence?, v1, 2026-09-10](https://arxiv.org/html/2609.11235v1)，§2–3。已核观测通道不可辨识性及 Gaussian KEEP/RECENTER 边界。本项目不再声称首次证明无标签安全选择不可能，也不把改成迭代动作当充分区别。在已知 m、σ、类别朝向的受限模型内，锚消除了部分歧义；保证应明确依赖这些信息，讨论未知锚时再改变可辨识性结论。

## 5. 在线概率建模与保护机制

[Corrales and Ríos Insua, A probabilistic framework for online test-time adaptation, v1, 2026-06-24](https://arxiv.org/html/2606.26457v1)，§3.4–3.5、§4、§6。已读状态空间建模、转移及近似推断结构；这里的随机参数转移是模型设计，不能直接等同真实熵 SGD 的首次风险越界保证。现有阅读尚未发现与本项目目标相同的结论，但未做逐条完整证明审计。

[Protected Test-Time Adaptation via Online Entropy Matching: A Betting Approach, v2](https://arxiv.org/html/2408.07511v2)：保留 B-notes 对检验误报控制和监督风险保护的区分；本次没有重读全文。可作为序贯检验工具入口，不可将其保证改述为分类风险不增。

## 6. 尚需完成的决定性工作

1. 取得并逐页核 WACV 正文、全部补充；全文核 CED，比较其轨迹约束理论。
2. 对 PeTTA、GALA/STAG 的关键推导建立“参数、损失、更新、比较风险、时间范围”逐式对照。
3. OpenAlex 已成功返回参考文献 ID，但本轮尚未系统遍历引用图；不能声称完成前向引用普查。对新版本和同作者后续继续检索。
4. 审核随机逼近的退出时间及序贯置信集合文献；初稿的 Hoeffding 追踪和 supremum 证书属于标准基线。
5. 若只能得到高斯单阈值的常规分析，或无标签证书基本总拒绝，则停止扩展论文。

## 可直接用于论文的 related-work 初段

Entropy-based adaptation is already known to suffer from class imbalance and long-horizon degradation. Logit-adjusted TTA analyzes a simplified BN-shift mechanism, while PeTTA studies accumulated errors in a perturbed mixture classifier. Layer-selection and structural-gradient methods, including GALA and STAG, provide practical controls on adaptation. These works rule out treating entropy collapse or label-free update selection as new phenomena. Our intended focus is narrower: deriving the first crossing of frozen-source classification risk from the state-dependent entropy update itself. Recovery-complexity guarantees and observation-channel identifiability results further require us to state both the local optimization conditions and the information supplied by the source anchor. Establishing that this dynamical characterization goes beyond these neighbors remains a prerequisite for a novelty claim.

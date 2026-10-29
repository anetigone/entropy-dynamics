# 熵适应风险越界的相关工作与贡献边界

当前统计主线的贡献与新增文献对照见[原处修订稿](../../RELATED_WORK.md)。本文保留原高斯混合风险与时间口径的详细比较。

本文独立定义所研究的问题，并逐项比较相关工作的机制、保证对象和评价口径。讨论以已取得的文献内容为限，不将“未见同一结论”当作“确认没有先行工作”。初始文献核查于 2026-10-24 进行，WACV 正文于 2026-10-27 完成本地阅读；本次重写没有进行新一轮检索。

对 Weerasinghe 等人的 WACV 工作，本文直接接受其全部理论结论成立。比较的目标是明确在这些结论已经成立的前提下还增加了什么，不审查其证明是否有错，也不把补充证明的复核作为研究门槛。正文已阅读，补充材料已取得但尚未阅读；接受结论是比较时的前提，并不代表已独立核验全部证明。

## 1. 问题定义：比较的是哪一种风险与哪一种时间？

考虑二分类目标特征
\[
Z=\delta+Ym+\sigma\xi,\qquad
\Pr(Y=1)=p,\qquad \xi\sim N(0,1),
\]
其中 \(m,\sigma,w>0\)。源锚提供类别朝向、分离度 \(m\) 和尺度 \(\sigma\)；目标位置偏移 \(\delta\) 和先验 \(p\) 则改变部署分布。固定特征与置信度斜率 \(w\)，只更新预测 \(\operatorname{sign}(Z-b)\) 的阈值 \(b\)，初始化为 \(b_0=0\)。

令 \(s(u)=(1+e^{-u})^{-1}\)，\(a(u)=s(u)(1-s(u))\)，\(h(u)\) 为 \(s(u)\) 的二元熵。实际更新是每步使用独立新 batch 的熵梯度：
\[
b_{t+1}=b_t-\frac{\eta}{n}\sum_{i=1}^n
w^2(Z_{t,i}-b_t)a(w(Z_{t,i}-b_t)).
\]
更新不读取标签。标签仅用于定义分布和评估风险。

用 \(x=b-\delta\)、\(z=Z-\delta\) 中心化，初始位置为 \(x_0=-\delta\)。目标分类风险为
\[
r_p(x)=p\Phi((x-m)/\sigma)+(1-p)\Phi((-x-m)/\sigma),
\]
其中 \(\Phi\) 是标准正态 CDF。我们比较的是冻结初始化在同一目标分布上的风险，定义
\[
T_\epsilon=\inf\{t\ge0:r_p(x_t)>r_p(x_0)+\epsilon\}.
\]
它是首次不安全迭代，不是平均准确率、在线累计遗憾、熵达到某个阈值的时刻，也不是相对轨迹最优点的性能下降。

本项目试图连接四个对象：目标边际分布的熵几何、实际状态依赖更新、分类风险安全集合，以及首次退出的方向和时间。以下比较均围绕这四者展开。

| 比较维度 | 本项目所指对象 | 不能直接替代它的对象 |
|---|---|---|
| 算法 | 固定斜率、固定特征的真实阈值熵梯度 | 手工规定的常量漂移或直接均值重置 |
| 监督指标 | 同一固定目标分布上的分类风险 | 熵、置信度、适应 batch 准确率 |
| 比较器 | 不更新的初始化模型 | 源训练风险、轨迹最好风险、另一测试分布 |
| 时间事件 | 首次超过冻结风险加容忍度 | 最终退化、时间平均下降、累计遗憾 |
| 噪声 | 给定当前状态的新 batch 估计噪声 | 未控制误差就直接替换成固定高斯噪声 |
| 信息 | 已知类别朝向、分离度与尺度的锚 | 权重之外没有额外信息的一般部署模型 |

## 2. WACV：不平衡、BN 漂移与 Logit Adjustment

### 2.1. 已有结果应完整接受

Weerasinghe、Tennakoon、Chuah 和 Bab-Hadiashar 的 *Logit-Adjusted Test-Time Adaptation under Partial Class Imbalance*，WACV 2026，直接研究部分源类别出现在目标适应流中的情形。[官方正文](https://openaccess.thecvf.com/content/WACV2026/papers/Weerasinghe_Logit-Adjusted_Test-Time_Adaptation_under_Partial_Class_Imbalance_WACV_2026_paper.pdf)，[官方补充材料](https://openaccess.thecvf.com/content/WACV2026/supplemental/Weerasinghe_Logit-Adjusted_Test-Time_Adaptation_WACV_2026_supplemental.pdf)。

正文 Fig. 1 已展示准确率先提高、随后持续下降的轨迹。§3.1 的 Lemmas 1–3 给出 softmax 熵梯度、BN shift 的方向对齐及其 logit 偏置分析。§4 提出 logit adjustment 缓解该漂移。本文将这些机制和理论结论按其适用范围作为成立的先行结果。

其 BN 表示可写成
\[
h^\ell=\gamma^\ell\odot\widetilde x^\ell+\beta^\ell,\qquad
z=W\psi(h^\ell)+b.
\]
记下游映射的 Jacobian 为 \(J_\psi\)，则类别方向回传到 BN shift 空间形成
\[
v_i=J_\psi^\top\omega_{i:}.
\]
其分析说明熵更新如何在不平衡下使 BN shift 偏向占优类别，并导致预测偏置。这里的向量 \(v_i\) 是该文的下游回传方向，不等于前述标量阈值模型中的人口漂移函数。

因此，下述现象与机制均不能作为本项目的新贡献：

- 熵适应受类别不平衡影响。
- 适应时间增加后性能可能下降。
- BN 参数漂移和 logit 偏置会影响预测。
- 熵持续降低不保证分类性能持续改善。
- 利用目标先验或修改熵目标可以缓解退化。

换成高斯一维模型、延长轨迹、把“漂移”改名为“风险反弹”，也不足以形成实质区别。

### 2.2. 方法必须纳入直接基线

该文采用
\[
z_i'=z_i+\tau\log\pi_i
\]
调整 logits，再计算熵，其中 \(\pi\) 为估计的目标先验，\(\tau>0\)。先验由未调整 logits 的 argmax 预测类别计数 \(n_i\) 得到：
\[
\pi_i^{\mathrm{batch}}=
\frac{\sqrt{n_i}}{\sum_k\sqrt{n_k}},
\qquad
\pi^{(t)}=\alpha\pi^{(t-1)}+(1-\alpha)\pi^{\mathrm{batch}}.
\]
初始化采用均匀先验，并在更新后归一化。正文默认 \(\tau=1\)、\(\alpha=.1\)；ViT 实验另使用 100 步先验 warm-up。

这是一种直接修改熵更新的可执行控制方法，不能在后续实验中仅以未约束熵更新作为对手。合适的比较至少需要冻结模型、原始熵更新、LA 熵更新，以及利用模型结构的一次校准或 plug-in 阈值。若一个停止方法只优于最容易退化的原始更新，而没有与已有稳定化方法比较，证据是不充分的。

### 2.3. 相同的退化曲线不代表相同的评价问题

WACV 的适应流只包含选中的类别，选中类别之间均匀采样，并将流长度扩展到原数据规模的三倍。正文在 CIFAR-100-C、ImageNet-C 和 DomainNet-126 上实验。

其 All-Class Accuracy（AllAcc）在独立留出集上评价，留出集包含适应中可见与不可见的全部类别。可见类的评价样本不用于适应。AllAcc 汇总适应过程中的平均准确率。Peak-to-Average Reduction 定义为
\[
\mathrm{PeAR}=
\left(\frac{\text{峰值测试准确率}}{\mathrm{AllAcc}}-1\right)100\%.
\]
它描述峰值与平均值之间的差距，能反映长期稳定性，但不等于本项目的 \(T_\epsilon\)。

这个区别不是贡献本身，而是比较时必须保持的口径：我们的适应分布与监督风险分布相同，冻结模型是同一分布上的比较器。若复用其“仅部分可见类适应、全部类别测试”的实验，必须重新说明风险对应的分布。不能把其自建留出划分、重采样流或 PeAR 静默当作本项目的官方默认评估规则。

### 2.4. 在接受该文全部结论后，还需要证明什么？

候选区别只能是更具体的定量问题：

1. 从目标分布参数推出何处会逃逸、何处会安全收敛，而不是只对给定漂移方向做推演。
2. 用分类风险安全集合定义边界，计算首次超过冻结风险的时间。
3. 对实际状态依赖的 mini-batch 熵更新控制线性化或噪声替换误差。
4. 给出可区分的方向与时间联合预测，保留有限参数风险边界和离散更新次数。
5. 解释固定容忍度与缩小容忍度为什么出现不同时间尺度。

这五项只是候选新增对象，不自动构成新颖性。如果它们能由已有漂移结论加标准退出定理直接得到，而没有实质模型推导或新的可检验风险预测，贡献仍然不足。

## 3. PeTTA：混合模型中的错误积累

Hoang、Vo 和 Do 的 *Persistent Test-time Adaptation in Recurring Testing Scenarios*，arXiv:2311.18193v4，2024-11-02，提出 PeTTA。[论文](https://arxiv.org/html/2311.18193v4)。

已核查的范围包括 §3、Assumption 1、Theorem 1、Corollary 1 与 Appendix B 的证明入口；没有逐行复证全部附录。其工作已在扰动高斯混合分类器中研究持续错误积累，并利用分析指导稳定适应。

高斯混合、持续适应和错误反馈因此都不是本项目独占的建模对象。有效比较需同时说明以下项目：

| 项目 | 必须区分的内容 |
|---|---|
| 学习信号 | 伪标签、教师更新与直接预测熵梯度是否相同 |
| 适应状态 | 教师、学生或分类器参数与单阈值状态的差别 |
| 数据情景 | 反复出现的场景与固定平稳混合分布 |
| 被保证量 | 错误积累或稳定性与冻结风险首次越界 |
| 时间尺度 | 给定更新时长与参数极限下的首次退出分布 |

模型形式不同不能代替贡献判断。尤其需要排除一种弱结果：把已有混合分类器的错误积累重新写成阈值移动，却没有获得新的风险时间结论。

## 4. 在线选层、梯度锚和可靠性控制

### 4.1. GALA 与早期 LEAST 版本

Sahoo 等人的 arXiv:2404.03784v2，2025-02-23，题为 *A Layer Selection Approach to Test Time Adaptation*，方法名为 GALA。早期版本题为 *Layerwise Early Stopping for Test Time Adaptation*，方法名为 LEAST。[论文](https://arxiv.org/html/2404.03784v2)。

已读范围包括 §2 的梯度方向准则及 §5 分析。该工作涉及在线选层、跳过更新和重置机制。应按同一个 arXiv 标识符识别版本演化，不能把两个标题当成互不相关的两篇论文；完整版本差异与正式发表文本尚未全面核对。

这条工作线意味着“无需标签决定是否继续更新”不是空白。一个标量阈值的拒绝规则，需要在风险比较器、信息条件和保证对象上说明新增内容，而不能把“能早停”当作贡献。

### 4.2. STAG 与结构梯度锚

Shin 等人的 STAG，arXiv:2402.09004v2，2026-02-06，使用分类器结构提供梯度信号。[论文](https://arxiv.org/html/2402.09004v2)。

已核 §3.2 与 Appendices A.2–A.4 的结构锚和熵梯度推导范围，没有完成整套证明的独立复证。本项目同样依赖源端结构：类别朝向、分离度、尺度或源校准统计。这些信息必须列入问题定义。把它们隐藏在一个“已知高斯模型”里，再声称严格权重可用的 source-free 保证，会改变比较条件。

因此，不能声称首次向 TTA 引入结构锚、首次利用分类器几何识别有利更新，也不能把锚误差留在保证之外。

### 4.3. Dual Strategies 的可靠性判据

Nguyen Phuong 等人的 *Dual Strategies for Test-Time Adaptation*，arXiv:2604.17542v1，网页标注 Findings of Computer Vision and Pattern Recognition 2026。[论文](https://arxiv.org/html/2604.17542v1)。

已核 §3.3–3.4 的双增强可靠性判据以及分组熵最小化、最大化机制，未认证完整证明。其样本可靠性选择与风险安全证书的对象不同，但不能因此忽略它作为实用控制方法的对照。网页上没有检索到某个定理关键词，也不构成它缺少理论的证据。

## 5. 恢复复杂度、可辨识性与信息边界

### 5.1. 恢复复杂度并非尚未研究

Zhou 等人的 *On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective*，arXiv:2605.28057v2，ICML 2026，研究恢复复杂度。[论文](https://arxiv.org/html/2605.28057v2)。

已核 §2–4 与 Appendices D–E 的模型和证明结构，未完成所有证明的复证。比较时要保留其局部梯度对齐、曲率与保持在邻域等假设，以及其动态扩展的范围。

我们的目标如果成立，应当是从具体分布和实际熵递推推出对齐何时成立、何时失效，以及风险第一次越界的时间。不能把“已有恢复理论中需要假设局部条件”简化成“现有理论不能处理长期适应”，也不能把有限 batch、依赖流或非平稳本身一概称为文献空白。

### 5.2. 无标签安全选择已有可辨识性限制

Jhawar 和 Wang 的 *When is Test-Time Adaptation Identifiable From Unlabeled Evidence?*，arXiv:2609.11235v1，2026-09-10，分析观察通道的不可辨识性以及 Gaussian KEEP/RECENTER 边界。[论文](https://arxiv.org/html/2609.11235v1)。

已核 §2–3 的问题定义与相关边界。该文排除了“首次证明无标签安全选择不可能”的贡献表述；把一次动作改成迭代动作，也不足以绕开信息限制。

本项目的正结果使用额外信息：已知有序类别朝向、\(m\) 与 \(\sigma\)，限制了混合分布的标签解释。不能把一般不可辨识性结论不加区分地套到这一受限模型；反过来也不能把受限模型内的可分析性宣传成一般无标签 TTA 可保证安全。

尤其要区分两个命题：

- 已知模型参数时，可以计算风险边界并分析随机轨迹。
- 参数未知时，仅凭在线无标签观测，可以在部署前可靠识别风险变化方向。

前者不蕴含后者。任何实现性的安全证书，都需要处理参数估计、锚误差和重复决策。

## 6. 概率建模、风险监测与在线保护

Corrales 和 Ríos Insua 的 *A probabilistic framework for online test-time adaptation*，arXiv:2606.26457v1，2026-06-24，采用状态空间建模、转移和近似推断。[论文](https://arxiv.org/html/2606.26457v1)。已读 §3.4–3.5、§4 与 §6。它的随机参数转移是方法中的概率模型，不能直接等同于真实熵 SGD 的首次风险越界分布；但若以“我们分析随机在线参数”作为区别，同样过于宽泛。

*Protected Test-Time Adaptation via Online Entropy Matching: A Betting Approach*，arXiv:2408.07511v2，提供在线熵匹配与 betting 保护机制。[论文](https://arxiv.org/html/2408.07511v2)。已有核查将检验误报控制与监督风险保护区分；本次没有重新阅读全文。对某个无标签统计量的序贯检验保证，不能改述为每一步目标分类风险不超过冻结模型。

*Monitoring Risks in Test-Time Adaptation*，NeurIPS 2025，是必须直接对照的风险监测工作。[官方论文页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/746960ad49ddb47248970a0e1404230c-Abstract-Conference.html)。现有核查不足以逐条确认其完整定理范围，不能把它排除为仅经验方法。应具体区分监督风险或其估计、运行平均风险、首次越界事件，以及部署前拒绝更新的保证。

上述工作提醒我们：一个方法“监测”“保护”或“稳定”了某个量，并不自动对应相同的风险比较器；相反，命名不同也不保证定理不同。比较必须落实到随机事件、可观测信息和概率保证。

## 7. 经典随机退出理论：不能主张的新机制

### 7.1. 对数凹卷积与分布几何

Saumard 和 Wellner 的 *Log-concavity and strong log-concavity: a review*，Statistics Surveys 8，2014，系统讨论对数凹性质。[论文](https://arxiv.org/abs/1404.5886)，[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4847755/)。

本项目使用“非负可积对数凹函数的卷积仍对数凹”这一标准结果。在弱分离条件 \(m\le\sigma\) 下，混合密度的对数二阶导数为
\[
-\sigma^{-2}+\frac{m^2}{\sigma^4}
\operatorname{sech}^2\!\left(
\frac{mz}{\sigma^2}+\frac12\log\frac p{1-p}\right)\le0.
\]
再核对特定熵核的对数凹性，可定位唯一不稳定熵峰。可讨论的具体内容是这个模型的核和密度满足条件，以及由此形成的风险区间分类；卷积闭包本身不属于新结果。

### 7.2. 不稳定点的对数逃逸

Kifer 的 *The exit problem for small random perturbations of dynamical systems with a hyperbolic fixed point*，Israel Journal of Mathematics 40，1981，74–96，是经典小随机扰动退出文献。[出版记录](https://cris.huji.ac.il/en/publications/the-exit-problem-for-small-random-perturbations-of-dynamical-syst/)。现有工作仅核对书目信息，没有审读完整证明。

Bakhtin 的 *Exit asymptotics for small diffusion about an unstable equilibrium*，arXiv:math/0701569，2007，期刊 DOI 为 10.1016/j.spa.2007.06.003，研究不稳定平衡附近的小扩散退出。[论文](https://arxiv.org/abs/math/0701569)。

Bakhtin 和 Gaál 的 *Exit time asymptotics for dynamical systems with fast random switching near an unstable equilibrium*，arXiv:1901.05513，2019，给出随机切换模型的联合退出方向与时间渐近。[论文](https://arxiv.org/abs/1901.05513)，[全文](https://arxiv.org/html/1901.05513v1)。已有阅读记录核对其 Theorem 1.1 和全文相关论述：连续时间中的确定性对数时间，加高斯种子绝对值的负对数修正，已有明确先例；有限时间 CLT 本身不足以控制对数时长，也不是我们首次指出的问题。

因此不能将以下内容单独计为新贡献：不稳定点附近发生指数增长、退出时间与初始扰动的对数成正比、高斯噪声决定退出方向、时间修正包含高斯绝对值的负对数，或用共轭坐标消除确定性非线性。

### 7.3. 具体需要超出哪些标准推论？

在阈值模型中，令 \(p_\rho=(1-\rho)/2\)，
\[
K(t)=\mathbb E h(w(t+\sigma\xi)),\quad
\beta=-K'(m)>0,\quad\lambda=-K''(m)>0,\quad A=1+\eta\lambda.
\]
平衡附近的熵峰满足 \(q_\rho=-(\beta/\lambda)\rho+O(\rho^2)\)。

固定容忍度的确定性结论是
\[
T_\epsilon=\frac{\log(1/\rho)}{\log A}+O(1)
\]
（初始化与 \(\rho\) 同阶，并且位于峰值的指定一侧）。临界 batch 极限则使用
\[
\sqrt n\rho_n\to c,\qquad \sqrt n x_{0,n}\to u,\qquad
\mu=u+\beta c/\lambda,
\]
以及
\[
s_0^2=\operatorname{Var}_{p=1/2}U(0,z),\qquad
s_\eta^2=\frac{\eta^2s_0^2}{A^2-1},
\]
其中 \(U(x,z)=-w^2(z-x)a(w(z-x))\) 是单样本向右速度。其风险退出方向满足
\[
\Pr(\text{右侧退出})\to\Phi(\mu/s_\eta),
\]
固定正容忍度下的中心时间为
\[
T_\epsilon=\frac{\log n}{2\log A}+O_{\mathbb P}(1).
\]
这些形式与经典不稳定退出高度相关，不能只凭公式看起来适用于 TTA 就主张新颖性。

真正需要核对的是它们是否落实了额外内容：

| 核对对象 | 必须具体体现的内容 |
|---|---|
| 分布结构 | 从熵核与混合参数推出峰值、曲率及反弹或收敛区域 |
| 真实更新 | 对状态依赖 batch 梯度与冻结噪声种子的误差给出控制 |
| 风险边界 | 使用冻结目标风险定义的两端边界，而非任意小球退出 |
| 有限参数 | 保留实际峰值、曲率及左右风险边界，产生不同时间预测 |
| 离散性 | 固定步长时保留整数取整和随 batch 序列变化的格点相位 |
| 容忍度 | 固定正 \(\epsilon\) 与 \(\epsilon=O(1/n)\) 分别出现对数时长和有限步极限 |
| 可检验差异 | 能提出超出笼统“晚期会退化”的、可被数据否定的预测 |

这些对象全部写出来仍不等于证明充分新颖。必须检查是否是对标准定理的直接实例化，以及模型限制是否使结论失去研究价值。

## 8. 停止证书为何仍是通用基线？

设未知参数为 \(\vartheta=(\delta,p)\)，源锚参数已知，并定义 \(R_\vartheta(b)=r_p(b-\delta)\)。若一个无标签序贯置信集合 \(\mathcal C_t\) 满足
\[
\Pr(\vartheta\in\mathcal C_t\text{ 对所有 }t)\ge1-\alpha,
\]
则在部署候选 \(b'\) 前要求
\[
\sup_{\vartheta\in\mathcal C_t}
\{R_\vartheta(b')-R_\vartheta(b_0)\}\le\epsilon
\]
即可在覆盖事件上保证所有接受的更新安全。

这个推论直接来自置信集合定义，不是独立的方法贡献。数学上接受更新安全，与统计上能构造有用的集合、实际中存在改善机会，是不同层次的问题。

一个有意义的方法还需说明：集合如何在重复观察和数据依赖提案下保持覆盖，是否需要独立校准数据，锚误差怎样进入界，是否因集合过宽而几乎总拒绝，以及相比一次校准后的固定估计究竟增加了什么。不能把 oracle 风险边界视为无标签可观测量，也不能把普通固定时间置信区间反复使用后仍称序贯保证。

因此，通用 supremum 过滤器、Hoeffding 轨迹追踪或一个简单拒绝规则，都不能自动补足机制论文的新颖性。

## 9. 可保留的贡献表述与应避免的表述

| 不成立或证据不足的说法 | 可审查的更窄问题 |
|---|---|
| 首次发现熵适应坍塌或先改善后退化 | 能否从混合参数推出冻结风险何时首次越界 |
| 首次分析类别不平衡导致 BN 漂移 | 能否建立与实际熵阈值递推一致的风险时间律 |
| 首次提出无标签早停或安全选择 | 在明确锚信息下，能否保证非平凡接受与改善 |
| 首次发现对数逃逸或高斯种子 | 能否严格连接状态依赖更新和具体风险边界 |
| batch 越大安全时长总按 log n 增长 | 平衡或临界缩放参数族内是否出现预测的中心时间移动 |
| 小正学习率可以避免第一步风险增加 | 对无条件熵方向，错误符号事件是否仍有正概率 |
| 单阈值定理保证完整神经网络安全 | 冻结视觉表征是否近似出现相同相区和时间尺度 |
| 没读到同样定理，所以问题开放 | 已读范围内尚未识别相同结果，仍需直接对照 |

在接受 WACV 全部结论和经典退出机制以后，比较稳妥的候选表述是：

> 在固定表征的熵适应阈值模型中，由目标分布结构和分类风险安全集合共同确定稳定性与首次风险越界；对真实状态依赖 mini-batch 更新给出方向、时间及其容忍度依赖的定量结论。

是否足以形成新贡献，还取决于这些结论相对近邻的不可直接推出性和视觉意义。文献比较应围绕最终结论，而不是围绕另一个方法可能存在的证明缺口。

## 10. 可独立使用的英文定位段落

Entropy-based test-time adaptation is already known to suffer from class imbalance and long-horizon degradation. Logit-adjusted TTA studies normalization-parameter drift and mitigates its effects, while PeTTA analyzes accumulated errors in a mixture-based classifier. Layer-selection and structural-gradient methods, including GALA and STAG, provide practical controls on whether and where to adapt. We accept these results as established and do not claim entropy collapse or label-free update selection as new phenomena. Our narrower object is the first crossing of frozen-model classification risk under the actual state-dependent entropy update. A meaningful distinction must connect distribution geometry, batch noise, and risk-defined boundaries, rather than restating the classical logarithmic escape of an unstable equilibrium. Recovery-complexity and observation-channel identifiability results additionally require the source-anchor information and the risk comparator to be declared explicitly. Whether this model-specific characterization provides sufficient novelty remains a separate question from the validity of its proofs.

## 11. 阅读范围与结论强度

WACV 正文已完整阅读并核对理论页；其补充材料尚未阅读，但全部结论被接受为比较前提。PeTTA、GALA、STAG、Dual Strategies、恢复复杂度、可辨识性和概率框架的阅读范围已在各自条目中具体说明，没有把阅读入口或结构识别写成完整证明复证。CED（*CLIP-guided entropy dynamics for robust test-time adaptation in harsh visual conditions*）目前只读出版社摘要与结论，涉及熵动态、过度自信与轨迹约束，仍是需要全文对照的近邻。[出版社页面](https://www.sciencedirect.com/science/article/pii/S0031320326002621)。

风险监测和部分经典文献同样保留了阅读深度限制。当前没有完成系统的前向引用遍历或全领域综述，不能把这一文献集合称为完整新颖性认证。后续判断即使改变，也应基于结果范围的直接比较，而不是降低对先行工作的接受标准。

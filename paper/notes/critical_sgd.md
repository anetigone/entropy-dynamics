# 理论续篇：真实熵 SGD 的临界逃逸、离散时间分布与容忍度标度

研究笔记，2026-10-26。接续 2026-10-24 的 [分布几何笔记](entropy_geometry.md)，处理其中 §8 当时尚未证明的非线性、状态依赖 SGD 临界区。

本篇给出模型内可逐步核查的证明、有限参数预测式和可复现的数值检查。“定理”指下述明确假设下的数学陈述；证明尚未经外部同行审查。高斯随机种子、对数逃逸时间和不稳定点的线性化属于经典机制，不能把这些机制本身宣称为新贡献。

论文中的统一陈述与编号见 [DRAFT.md](../DRAFT.md)；当前研究状态和后续安排以 [STATUS.md](../../STATUS.md) 为准。下述章节记录交付推导的内容与数学适用范围。

## 0. 本次实际推进

1. 对真实 fresh-batch 熵 SGD，构造精确的确定性共轭坐标，证明其停止后的随机种子与冻结噪声级数之间的均方误差为 $O(\log n/n)$。这补上了上一轮仅在线性 AR 模型中成立的环节。
2. 给出退出方向和首次风险越界时间的联合分布近似，包括真实熵峰、左右风险边界，以及一个不声称最优的分布误差率。
3. 导出弱偏置与噪声竞争时的临界标度，保留离散时间的格点相位；同时证明容忍度为 $O(1/n)$ 时是不同的有限步极限。
4. 证明一个有针对性的限制：临界先验偏移下，只把无条件熵更新的正学习率调小，不能消除第一步走错方向的概率。

本篇始终限定于标量阈值、固定斜率、平稳目标分布与独立新 batch。不包含一般神经网络、重复使用固定测试集、动量、随时间变化的先验、联合学习率极限或完整的无标签停止算法。

## 1. 模型、方向与适用条件

令

\[
Z=Ym+\sigma\xi,\qquad \Pr(Y=1)=p=\frac{1-\rho}{2},
\qquad \xi\sim N(0,1),\qquad 0<m\le\sigma,\quad w>0.
\]

原问题中的位置偏移已吸收到阈值 $x=b-\delta$ 中，因此初始化为 $x_0=-\delta$。记

\[
s(u)=\frac1{1+e^{-u}},\quad a(u)=s(u)(1-s(u)),\quad
h(u)=-s(u)\log s(u)-(1-s(u))\log(1-s(u)).
\]

期望熵、向右速度和单样本向右更新为

\[
F_\rho(x)=\mathbb E_\rho h(w(Z-x)),\qquad
v_\rho(x)=-F_\rho'(x),\qquad
U(x,z)=-w^2(z-x)a(w(z-x)).
\tag{1}
\]

实际 SGD 为

\[
x_{t+1}=x_t+\eta\frac1n\sum_{i=1}^n U(x_t,Z_{t,i}),
\tag{2}
\]

所有 $Z_{t,i}$ 独立且来自同一目标混合分布。真实标签只用于分布定义及离线评估；更新式不读取标签。

以下固定

\[
M=w/4,\qquad L=w^2/4,\qquad 0<\eta<1/L.
\tag{3}
\]

直接计算可得 $|U|\le M$、$|\partial_x U|\le L$，且高阶导数有界。充分条件 (3) 比单纯保证期望熵下降的步长条件更强，其作用是使确定性更新映射全局严格递增。

目标分类风险为

\[
r_\rho(x)=\frac{1-\rho}{2}\Phi\!\left(\frac{x-m}{\sigma}\right)
+\frac{1+\rho}{2}\Phi\!\left(\frac{-x-m}{\sigma}\right).
\tag{4}
\]

它在

\[
x_*(\rho)=\frac{\sigma^2}{2m}\log\frac{1+\rho}{1-\rho}
\]

处唯一最小。本文的“越界”始终比较冻结初始化在同一目标分布上的风险：

\[
T_\epsilon=\inf\{t\ge0:r_\rho(x_t)>r_\rho(x_0)+\epsilon\}.
\tag{5}
\]

### 1.1 熵峰与局部常数

上一轮已证明：当 $m\le\sigma$，$F_\rho$ 只有一个极大点 $q_\rho$，且

\[
v_\rho(x)<0\ (x<q_\rho),\qquad
v_\rho(x)>0\ (x>q_\rho).
\]

简述所用结构：高斯混合密度的对数二阶导数不超过 $-1/\sigma^2+m^2/\sigma^4\le0$；$h(w\cdot)$ 对数凹；卷积保对数凹。解析性、正性和两端趋零排除平台，从而得到唯一峰值。

令 $K(t)=\mathbb E h(w(t+\sigma\xi))$，则

\[
\beta=-K'(m)>0,\qquad \lambda=-K''(m)>0,
\qquad A=1+\eta\lambda>1.
\tag{6}
\]

这里曲率严格为正也可独立检查。平衡混合密度 $f_0$ 满足 $f_0'(z)<0$ 对所有 $z>0$ 成立，因为

\[
(\log f_0)'(z)=\frac{-z+m\tanh(mz/\sigma^2)}{\sigma^2}<0.
\]

函数 $k(z)=h(wz)$ 也在 $z>0$ 严格下降，故分部积分给出

\[
F_0''(0)=-\int k'(z)f_0'(z)\,dz<0.
\]

同样，对 $t>0$，$K'(t)=\int_0^\infty k'(z)[\varphi_\sigma(z-t)-\varphi_\sigma(z+t)]\,dz<0$，所以 $\beta>0$。

由隐函数定理，在 $\rho=0$ 的邻域内

\[
q_\rho=-\frac\beta\lambda\rho+O(\rho^2),\qquad
\lambda_\rho=v_\rho'(q_\rho)=\lambda+O(\rho),\qquad
A_\rho=1+\eta\lambda_\rho>1.
\tag{7}
\]

以下把 $|\rho|\le\rho_0$ 的邻域取得足够小，使上述性质和后续紧区间界一致成立。所有大 $O$ 常数允许依赖 $m,\sigma,w,\eta,\epsilon,\rho_0$ 及初始条件界，不随 $n,t,\rho,x_0$ 改变。

## 2. 先精确处理确定性非线性

定义 $G_\rho(x)=x+\eta v_\rho(x)$。由 (3)，$G_\rho'(x)\ge1-\eta L>0$；又因更新有界，它是实线上的光滑双射。

**引理 1（归一化共轭）。** 存在严格递增的 $C^2$ 函数 $H_\rho:\mathbb R\to\mathbb R$，满足

\[
H_\rho(q_\rho)=0,\qquad H_\rho'(q_\rho)=1,\qquad
H_\rho(G_\rho(x))=A_\rho H_\rho(x).
\tag{8}
\]

在任意固定紧区间上，$H_\rho$ 及其前两阶导数对小 $\rho$ 一致有界；$H_\rho'$ 有一致正下界，且这些函数随 $\rho$ 连续变化。平衡时 $H_0$ 是奇函数。

**证明。** 在峰值附近写 $y=x-q_\rho$，逆映射为

\[
g_\rho(y)=G_\rho^{-1}(q_\rho+y)-q_\rho
=A_\rho^{-1}y+O(y^2).
\]

把参数邻域和空间邻域适当缩小，可选一致的 $b<1$，使 $|g_\rho'|\le b$ 且 $(\sup_\rho A_\rho)b^2<1$。令

\[
H_\rho(q_\rho+y)=\lim_{j\to\infty}A_\rho^j g_\rho^{\,j}(y).
\tag{9}
\]

相邻项之差是 $A_\rho^{j+1}R_\rho(g_\rho^{\,j}(y))$，其中 $R_\rho(z)=g_\rho(z)-A_\rho^{-1}z=O(z^2)$。迭代满足

\[
|g_\rho^{\,j}(y)|+|(g_\rho^{\,j})'(y)|+|(g_\rho^{\,j})''(y)|\le C b^j.
\]

由 $R_\rho(0)=R_\rho'(0)=0$，相邻项之差连同前两阶导数均由 $C(A_\rho b^2)^j$ 控制。因此 (9) 在 $C^2$ 意义下一致收敛，导数在原点为 1，并在小邻域内保持为正。移去一个逆迭代即得 (8)。

对任意 $x>q_\rho$，逆迭代严格递减且下界为 $q_\rho$，其极限只能是固定点 $q_\rho$；左侧同理。故有限次逆迭代把任意紧区间送入上述局部邻域，借助 (8) 将 $H_\rho$ 延拓到实线。紧性给出参数一致界，正导数由链式法则保持。平衡时映射为奇函数，(9) 给出奇对称。证毕。

此处使用的是经典不稳定固定点线性化思想。关键用途是：在固定风险区间上精确消去确定性非线性，避免对持续 $O(\log n)$ 步的原坐标误差直接作不稳定递推。

## 3. 真实 SGD 的随机种子：带误差的主定理

固定正容忍度

\[
0<\epsilon<\frac12-\Phi(-m/\sigma).
\tag{10}
\]

设 $|\rho|\le\rho_0$，初始点满足

\[
|x_0-q_\rho|\le K/\sqrt n
\tag{11}
\]

其中 $K<\infty$ 固定。对足够大的 $n$，风险安全集合是有限区间

\[
[a_n,c_n]=\{x:r_\rho(x)\le r_\rho(x_0)+\epsilon\},
\]

且其内部包含 $q_\rho$，峰值到两端的距离有一致正下界。记 $\tau=T_\epsilon$，右侧退出时 $J=+1$，左侧退出时 $J=-1$。

定义可由目标模型计算的量

\[
\mu_n=\sqrt n\,H_\rho(x_0),\qquad
s_n^2=\frac{\eta^2\operatorname{Var}_\rho[U(q_\rho,Z)]}{A_\rho^2-1},
\tag{12}
\]

\[
B_n^+=H_\rho(c_n)>0,\qquad B_n^-=-H_\rho(a_n)>0.
\tag{13}
\]

这里 $s_n$ 一致有正下界；下标 $n$ 允许 $\rho,x_0$ 随 $n$ 变化。

**定理 1（状态依赖 SGD 的分布近似）。** 在 (1)–(13) 下，$\tau<\infty$ 几乎处处成立。存在常数 $C<\infty$，使对所有整数 $t\ge0$，

\[
\left|\Pr(\tau\le t,J=+1)
-\Phi\!\left(\frac{\mu_n-\sqrt n A_\rho^{-t}B_n^+}{s_n}\right)\right|
\le C\left[n^{-1/2}+\left(\frac{1+\log n}{n}+A_\rho^{-2t}\right)^{1/3}\right],
\tag{14}
\]

\[
\left|\Pr(\tau\le t,J=-1)
-\Phi\!\left(\frac{-\mu_n-\sqrt n A_\rho^{-t}B_n^-}{s_n}\right)\right|
\le C\left[n^{-1/2}+\left(\frac{1+\log n}{n}+A_\rho^{-2t}\right)^{1/3}\right].
\tag{15}
\]

退出方向满足

\[
\left|\Pr(J=+1)-\Phi(\mu_n/s_n)\right|
\le C\left[n^{-1/2}+\left(\frac{1+\log n}{n}\right)^{1/3}\right].
\tag{16}
\]

在 $nA_\rho^{-2t}=O(1)$ 的对数时间窗内，分布误差为 $O((\log n/n)^{1/3})$。这不是最优率的宣称。当前证明没有把常数 $C$ 数值化，因此不能将 (14)–(16) 当作已校准的有限 batch 置信保证。

### 3.1 证明：停止与共轭展开

令

\[
\xi_j(x)=\frac1n\sum_{i=1}^n U(x,Z_{j,i})-v_\rho(x).
\]

相对于第 $j$ 个新 batch 到来前的信息，它条件均值为零，并满足

\[
\mathbb E[\xi_j(x)^2\mid\mathcal F_j]\le M^2/n,
\qquad \mathbb E[\xi_j(x)^4\mid\mathcal F_j]\le C/n^2,
\tag{17}
\]

\[
\mathbb E[(\xi_j(x)-\xi_j(q_\rho))^2\mid\mathcal F_j]
\le L^2|x-q_\rho|^2/n.
\tag{18}
\]

先说明停止时间有限。所有安全区间位于一个固定紧区间内。选一个完全位于其左侧的有界样本区间 $I_Z$，则 $U(x,z)\ge u_*>0$ 对安全区内的 $x$ 和 $z\in I_Z$ 一致成立。如果连续若干个 batch 的全部样本均落在 $I_Z$，迭代必定从右端退出。该事件对于每个固定 $n$ 有严格正概率，独立时间块反复提供此机会，因此 $\tau<\infty$ 几乎处处。

在退出前及单步越界可达的紧区间上，$H_\rho''$ 有界。由 (8) 和 Taylor 公式，

\[
H_\rho(x_{j+1})=A_\rho H_\rho(x_j)
+\eta H_\rho'(G_\rho(x_j))\xi_j(x_j)+R_j,
\qquad |R_j|\le C\xi_j(x_j)^2.
\tag{19}
\]

固定学习率已吸收在常数中。定义停止后的归一化坐标

\[
Y_{n,t}=\sqrt n A_\rho^{-(t\wedge\tau)}H_\rho(x_{t\wedge\tau}).
\tag{20}
\]

把 (19) 乘以相应几何权重并累加，得到初值、一个停止鞅和一个加权余项。由 (17) 及几何级数，鞅的二阶矩一致有界；余项满足

\[
\left\|\sqrt n\sum_{j<t\wedge\tau}A_\rho^{-j-1}R_j\right\|_2
\le C\sqrt n\sum_{j\ge0}A_\rho^{-j-1}/n\le C/\sqrt n.
\tag{21}
\]

初值由 (11) 一致有界，所以

\[
\sup_t\mathbb E|Y_{n,t}|^2\le C,\qquad
\mathbb E|Y_{n,\tau}|^2\le C.
\tag{22}
\]

退出时 $|H_\rho(x_\tau)|\ge\min(B_n^+,B_n^-)\ge b>0$，故

\[
\mathbb E A_\rho^{-2\tau}\le C/n.
\tag{23}
\]

### 3.2 证明：冻结噪声，控制状态依赖误差

在同一个无限新样本数组上定义

\[
W_n=\mu_n+\eta\sum_{j=0}^\infty A_\rho^{-j-1}\sqrt n\,\xi_j(q_\rho).
\tag{24}
\]

注意 $v_\rho(q_\rho)=0$。在退出前，

\[
D_j=H_\rho'(G_\rho(x_j))\xi_j(x_j)-\xi_j(q_\rho)
\]

是条件均值为零的量。由 (18)、$H_\rho'(q_\rho)=1$ 和导数的局部 Lipschitz 性，

\[
\mathbb E[D_j^2\mid\mathcal F_j]\le C|x_j-q_\rho|^2/n.
\tag{25}
\]

又由 $H_\rho'$ 的一致正下界、(20)–(22)，以及安全区的紧性，

\[
\mathbb E[\mathbf1_{\{j<\tau\}}|x_j-q_\rho|^2]
\le C\min\{A_\rho^{2j}/n,1\}.
\tag{26}
\]

鞅差分正交，故加权状态依赖误差满足

\[
\mathbb E\left|\sqrt n\eta\sum_{j<t\wedge\tau}A_\rho^{-j-1}D_j\right|^2
\le C\sum_{j\ge0}A_\rho^{-2j}\min\{A_\rho^{2j}/n,1\}
\le C\frac{1+\log n}{n}.
\tag{27}
\]

最后，$W_n$ 在 $t\wedge\tau$ 之后的冻结噪声尾部，条件方差为 $s_n^2 A_\rho^{-2(t\wedge\tau)}$。由 (23)，其期望不超过 $C(A_\rho^{-2t}+n^{-1})$。结合 (21)、(27) 得到关键估计

\[
\boxed{\ \mathbb E|Y_{n,t}-W_n|^2
\le C\left[\frac{1+\log n}{n}+A_\rho^{-2t}\right].\ }
\tag{28}
\]

若直接停在 $\tau$，则

\[
\mathbb E|Y_{n,\tau}-W_n|^2\le C(1+\log n)/n.
\tag{29}
\]

这一步是从线性 toy recursion 推进到真实状态依赖 SGD 所需的核心控制；并未把所有状态的梯度噪声强行假定为同一个高斯变量。

### 3.3 证明：Berry–Esseen 与精确越界事件

将 (24) 按 batch 中的样本位置重排：

\[
W_n-\mu_n=\frac1{\sqrt n}\sum_{i=1}^n V_{n,i},\qquad
V_{n,i}=\eta\sum_{j\ge0}A_\rho^{-j-1}U(q_\rho,Z_{j,i}).
\tag{30}
\]

对每个 $n$，这些列变量独立同分布，均值为零，方差为 $s_n^2$，且

\[
|V_{n,i}|\le\eta M/(A_\rho-1).
\]

因此标准 Berry–Esseen 不等式给出一致界

\[
\sup_z\left|\Pr(W_n\le z)-\Phi((z-\mu_n)/s_n)\right|\le C/\sqrt n.
\tag{31}
\]

本问题有一个避免额外“越界后误差”估计的精确恒等式：

\[
\{\tau\le t,J=+1\}=\{Y_{n,t}>\sqrt n A_\rho^{-t}B_n^+\},
\tag{32}
\]

\[
\{\tau\le t,J=-1\}=\{Y_{n,t}<-\sqrt n A_\rho^{-t}B_n^-\}.
\tag{33}
\]

若尚未退出，$H_\rho(x_t)$ 位于两端之间；若在 $s\le t$ 右侧退出，则 $H_\rho(x_s)>B_n^+$ 且 $A_\rho^{-s}\ge A_\rho^{-t}$；左侧同理。这证明了两个恒等式。

对任意阈值，利用 (28) 的二阶矩界和宽度为 $h$ 的高斯反集中区间，可把分布误差界为

\[
C\{n^{-1/2}+h+\Delta/h^2\},\qquad
\Delta=(1+\log n)/n+A_\rho^{-2t}.
\]

取 $h=\Delta^{1/3}$ 得到 (14)、(15)。使用 (29) 及 $\{J=+1\}=\{Y_{n,\tau}>0\}$ 得到 (16)。定理证毕。

## 4. 临界弱偏置：方向、时间与离散格点

进一步设

\[
\rho_n\to0,\qquad \sqrt n\,\rho_n\to c\in\mathbb R,\qquad
\sqrt n\,x_{0,n}\to u\in\mathbb R.
\tag{34}
\]

由 $x_0=-\delta$，位置偏移与先验偏移在这里同时出现。定义

\[
\mu=u+\frac{\beta c}{\lambda},\qquad
s_0^2=\operatorname{Var}_{\rho=0}[U(0,Z)]>0,
\]

\[
s_\eta^2=\frac{\eta^2s_0^2}{A^2-1}
=\frac{\eta s_0^2}{\lambda(2+\eta\lambda)}.
\tag{35}
\]

令 $c_\epsilon>0$ 是平衡风险边界：

\[
r_0(c_\epsilon)=r_0(0)+\epsilon,\qquad C_\epsilon=H_0(c_\epsilon)>0.
\tag{36}
\]

**定理 2（临界联合极限）。** 在定理 1 和 (34) 的条件下，

\[
\Pr(J=+1)\longrightarrow\Phi(\mu/s_\eta).
\tag{37}
\]

对任何整数时间序列 $t_n$，若 $\sqrt n A^{-t_n}\to r\in(0,\infty)$，则

\[
\Pr(T_\epsilon\le t_n,J=+1)\longrightarrow
\Phi\!\left(\frac{\mu-C_\epsilon r}{s_\eta}\right),
\tag{38}
\]

\[
\Pr(T_\epsilon\le t_n,J=-1)\longrightarrow
\Phi\!\left(\frac{-\mu-C_\epsilon r}{s_\eta}\right).
\tag{39}
\]

因此

\[
\boxed{\ T_\epsilon=\frac{\log n}{2\log(1+\eta\lambda)}+O_{\mathbb P}(1).\ }
\tag{40}
\]

**证明。** 由 (7) 和 $H_\rho'(q_\rho)=1$，

\[
\sqrt n H_{\rho_n}(x_{0,n})
=\sqrt n(x_{0,n}-q_{\rho_n})+o(1)\to\mu.
\]

由引理 1 的参数连续性、风险边界的连续性，
$s_n\to s_\eta$、$B_n^\pm\to C_\epsilon$。又因
$\log A_{\rho_n}-\log A=O(\rho_n)$ 且 $t_n=O(\log n)$，
$A_{\rho_n}^{-t_n}/A^{-t_n}\to1$。将这些极限代入定理 1 即得 (37)–(39)。
在中心时间两侧分别加减足够大的固定整数，(38)–(39) 的两尾可任意小，因此得到紧性 (40)。证毕。

### 4.1 不能漏掉离散时间相位

令

\[
b_n=\frac{\log\sqrt n}{\log A}.
\]

沿着小数部分 $\{b_n\}\to\theta\in[0,1]$ 的子序列，对任意整数 $k$，

\[
\Pr(T_\epsilon-\lfloor b_n\rfloor\le k)
\longrightarrow
\Pr\bigl(|N(\mu,s_\eta^2)|>C_\epsilon A^{\theta-k}\bigr).
\tag{41}
\]

其等价的整数值极限是

\[
\left\lceil\,\theta+\frac{\log C_\epsilon-\log|N(\mu,s_\eta^2)|}{\log A}\,\right\rceil;
\]

高斯变量恰好落在任一阈值上的概率为零，所以严格不等号和取整不产生歧义。这里的随机时间修正是“高斯种子绝对值的负对数，再作离散取整”；不是高斯时间波动，也不自动等于 Gumbel 分布。固定 $\eta$ 时一般不能省略格点相位，声称一个无子序列限制的连续中心化极限。

### 4.2 哪个量决定先验与噪声谁占主导

在峰值附近，有效种子信噪比是

\[
\kappa_n\simeq
\frac{\sqrt n(x_0-q_\rho)}{s_\eta}
\simeq
\frac{\sqrt n(\beta\rho/\lambda-\delta)}{s_\eta}.
\tag{42}
\]

因此，只按先验 $\rho$ 的大小讨论噪声是否可以忽略还不够；位置偏移可以抵消或放大初始点相对熵峰的位移。

- 若 $|\kappa_n|=O(1)$，退出方向具有非退化随机性，适用临界定理。
- 若 $\rho_n\downarrow0$、$\sqrt n\rho_n\to\infty$、$x_0=s\rho_n$，且 $s+\beta/\lambda\ne0$，噪声相对种子可以忽略。退出方向趋于 $\operatorname{sign}(s+\beta/\lambda)$，并有
  \[
  T_\epsilon=\frac{\log(1/\rho_n)}{\log A}+O_{\mathbb P}(1).
  \tag{43}
  \]
  证明使用同一共轭展开，但用 $\rho_n$ 归一化：加权鞅为 $O_{\mathbb P}(n^{-1/2})=o_{\mathbb P}(\rho_n)$，Taylor 余项为 $O_{\mathbb P}(n^{-1})$，种子为 $(s+\beta/\lambda)\rho_n+o(\rho_n)$。再用 (32)–(33) 的阈值比较即可。
- 若领先种子被抵消，(43) 不适用；必须看抵消后的位移是否落回 $n^{-1/2}$ 尺度。

代数上 $s_\eta\sim s_0\sqrt{\eta/(2\lambda)}$ 当 $\eta\downarrow0$，所以临界位移尺度常被写成 $\sqrt{\eta/n}$。本篇所有概率极限固定 $\eta$；尚未证明 $\eta=\eta_n\to0$ 的联合极限，不能把上述代数展开当作联合极限定理。

对于纯先验偏移、$x_0=0$，

\[
\rho_{\mathrm{cross}}\simeq
\frac{\lambda s_\eta}{\beta\sqrt n}.
\tag{44}
\]

同样，batch 翻倍带来约 $\log2/(2\log A)$ 步的中心时间移动，成立于平衡或保持临界 $\kappa$ 的参数族；对固定非零先验，不应声称 batch 越大安全时长始终按 $\log n$ 增长。

## 5. 当容忍度缩到 \(1/n\)：有限步的另一套定理

现在改用 $\epsilon=\epsilon_n\ge0$，满足

\[
n\epsilon_n\to e\in[0,\infty),
\tag{45}
\]

并保留 (34)。此时不能使用定理 1 的固定正风险边界。令

\[
k_0=\frac{\sigma^2}{m},\qquad
B=\frac{m\varphi(m/\sigma)}{\sigma^3}>0.
\tag{46}
\]

**定理 3（微小容忍度的首次越界极限）。** 设高斯 AR 递推为

\[
X_0=u,\qquad
X_{t+1}=A X_t+\eta\beta c+\eta s_0 Z_t,\qquad Z_t\overset{\mathrm{iid}}{\sim}N(0,1).
\tag{47}
\]

定义有限区间

\[
I_{c,u,e}=
\left[k_0c-\sqrt{(u-k_0c)^2+2e/B},\
k_0c+\sqrt{(u-k_0c)^2+2e/B}\right]
\tag{48}
\]

以及 $\tau_{\mathrm{AR}}=\inf\{t\ge1:X_t\notin I_{c,u,e}\}$。则

\[
\boxed{\ T_{\epsilon_n}\ \Rightarrow\ \tau_{\mathrm{AR}},\qquad
T_{\epsilon_n}=O_{\mathbb P}(1).\ }
\tag{49}
\]

**证明。** 对有界的缩放坐标 $y$，

\[
v_{\rho_n}(y/\sqrt n)
=\frac{\lambda y+\beta c}{\sqrt n}+o(n^{-1/2}).
\]

每个固定步数上，新 batch 中有界随机变量的中心极限定理以及状态连续性给出

\[
(\sqrt n x_0,\ldots,\sqrt n x_t)\Rightarrow (X_0,\ldots,X_t).
\]

风险的二阶 Taylor 展开在有界 $y$ 上一致：

\[
n[r_{\rho_n}(y/\sqrt n)-r_{\rho_n}(x_{0,n})]
\longrightarrow Q_c(y)-Q_c(u),\qquad
Q_c(y)=\frac B2(y-k_0c)^2.
\tag{50}
\]

验证一次项即可检查常数：$r_\rho'(0)=-\rho\varphi(m/\sigma)/\sigma$，
$r_0''(0)=B$，而 $Bk_0=\varphi(m/\sigma)/\sigma$。
不等式 $Q_c(y)-Q_c(u)\le e$ 恰好给出 (48)。

每个 $t\ge1$ 的 $X_t$ 都有连续密度，因此落在任一风险边界上的概率为零，有限步的首次越界 CDF 随之收敛。对有界区间 (48)，一步高斯噪声有一致正概率把任意区间内的点送出区间，故 $\tau_{\mathrm{AR}}<\infty$ 几乎处处。有限步 CDF 收敛至一个总质量为 1 的分布，也给出尾部紧性及 (49)。证毕。

### 5.1 完全平衡、零容忍的精确结论

如果 $\rho=0$、$x_0=0$ 且 $\epsilon=0$，则对每个有限 $n$ 和任意 $\eta>0$，

\[
\boxed{\ T_0=1\quad\text{几乎处处}.\ }
\tag{51}
\]

原因是 $x=0$ 为风险唯一最小点，而第一步的样本平均更新非原子、几乎不等于零。更具体地，$U(0,Z)$ 是连续密度输入的非恒定实解析函数；任意给定值的原像为零测集，故其分布无原子，独立和也无原子。

与之对应，完全平衡的 population 梯度在 $x=0$ 恰好为零，永远保持原风险。这说明零容忍下，population 行为并不能直接代表有限 batch 的第一步行为。

### 5.2 降低正学习率不能消除走错方向的概率

考虑纯先验偏移 $x_0=0$、$\sqrt n\rho_n\to c>0$。对固定 $\eta$，由定理 3 的第一步，

\[
\Pr(r_{\rho_n}(x_1)<r_{\rho_n}(0))
\longrightarrow
\Phi\!\left(\frac{(2k_0-\eta\beta)c}{\eta s_0}\right)
-\Phi\!\left(-\frac{\beta c}{s_0}\right).
\tag{52}
\]

更强的是，对于任意确定性的正学习率序列 $\eta_n>0$，

\[
\liminf_{n\to\infty}
\Pr(r_{\rho_n}(x_1)>r_{\rho_n}(0))
\ge \Phi(-\beta c/s_0)>0.
\tag{53}
\]

**证明。** 当 $\rho_n>0$，Bayes 阈值严格位于 0 的右侧；若第一步的经验向右速度为负，则无论正步长多小，$x_1<0$ 必定增大风险。独立样本 CLT 给出

\[
\sqrt n\,\frac1n\sum_i U(0,Z_i)\Rightarrow N(\beta c,s_0^2),
\]

从而得到 (53)。这里的负速度事件只取决于样本，不取决于正学习率。证毕。

(53) 限制的是“无条件执行当前 batch 的熵下降方向”，不是所有无标签方法的不可能性结论。设置为零的拒绝更新、利用额外结构或控制置信度的门控不在这个下界的对象之内。它为后续停止/拒绝规则提供明确动机，但本篇没有据此声称已构造出安全算法。

## 6. 数值核查与复现

分析、完整表格、误差说明与四面板图解释保存在结果原处的 [analysis.md](../../critical_sgd_reproducibility/critical_sgd/analysis.md)。运行命令见 [复现 README](../../critical_sgd_reproducibility/critical_sgd/README.md)。本节原分析已收拢至该文件，不在理论笔记重复保存。

## 7. 应如何写成论文贡献

比较稳妥的理论主线是：

> 在可解析的熵测试时适应模型中，先用目标分布结构识别不稳定熵峰，再把真实状态依赖的 mini-batch 更新与分类风险安全集合结合，得到偏置与噪声竞争时的退出方向、首次风险越界分布及容忍度依赖的时间尺度。

主文可保留定理 2、定理 3、推论 (53)；定理 1 的定量版本及共轭证明放附录。定理 1 是连接实际 SGD 与极限结论的关键，不能只用 AR 直觉替代它。定理 3 则说明“安全时长”必须连同容忍度一起定义。

目前可以陈述的是模型内的风险结论。不能声称已得到以下结果：

- 一般神经网络或完整测试时适应系统的理论保证。
- 未知混合参数下、仅凭无标签在线样本即可实现的可靠停止规则。
- 关于所有无标签适应方法的无改进或不可能性定理。
- 小学习率、弱曲率 $\lambda\to0$、大 batch 同时变化的统一极限。
- 本篇分布误差率最优，或其常数已经给出实用有限样本覆盖率。
- 文献中首次发现不稳定点的对数逃逸与高斯种子规律。

未知参数下的可识别证据、拒绝更新的覆盖率及改善机会属于停止算法问题；oracle 风险边界本身不是可执行的无标签规则。当前安排见 [STATUS.md](../../STATUS.md)。

## 8. 相关概率文献与定位

以下只核对本轮直接相关的机制，不构成完整新颖性审查。

1. Yuri Bakhtin, *Exit asymptotics for small diffusion about an unstable equilibrium*, arXiv:math/0701569，2007 年预印本；论文期刊 DOI: 10.1016/j.spa.2007.06.003。研究小白噪声扰动下不稳定点附近的退出渐近，是本项目应承认的经典背景。[论文页面](https://arxiv.org/abs/math/0701569)
2. Yuri Bakhtin and Alexisz Gaál, *Exit time asymptotics for dynamical systems with fast random switching near an unstable equilibrium*, arXiv:1901.05513，2019 年预印本。其定理 1.1 已给出连续时间随机切换模型中的联合退出方向与“对数确定性时间 + 高斯绝对值的负对数修正”，正文也明确讨论为何有限时间 CLT 不足以直接控制对数时间尺度。[论文页面](https://arxiv.org/abs/1901.05513)；[已核对的全文](https://arxiv.org/html/1901.05513v1)

本篇对照这些经典结果，落实的是固定步长 mini-batch 熵更新、弱先验与位置偏移、分类风险边界、离散格点，以及 $\epsilon$ 从固定量变为 $1/n$ 时的区别。这些模型化组合是否形成足够的新贡献，仍需与 TTA、SGD 退出和在线安全适应的近邻工作逐项核对；单凭本篇不能判定 CVPR 录用潜力。



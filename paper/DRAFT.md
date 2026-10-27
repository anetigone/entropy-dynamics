# When Does Entropy Adaptation Stop Helping?

## Risk Rebound and Safe Horizons under Mixed Feature and Prior Shift

Working manuscript. All model definitions, result statements, proofs, numerical protocols, and limitations needed to read this manuscript are included below. The analysis concerns a scalar classifier with fixed features and fixed confidence slope. It does not establish a general guarantee for neural-network adaptation or a completed unlabeled stopping algorithm.

## Abstract

Entropy minimization can initially improve a deployed classifier and later make it worse than its frozen initialization. We quantify this behavior for actual entropy-gradient updates of a decision threshold in a Gaussian mixture under simultaneous feature translation and class-prior shift. In the weak-separation regime, the marginal distribution implies a unique unstable entropy maximum and an explicit open parameter region of risk rebound. Sufficiently separated mixtures instead admit locally safe population convergence. Near class balance, fixed positive risk tolerance yields a discrete logarithmic population crossing time, whereas tolerance comparable to the available risk improvement yields bounded crossing times. For fresh-batch stochastic gradient descent, a direct drift argument bounds first risk exit outside the critical region. Near the unstable entropy maximum, a deterministic conjugacy controls the nonlinear, state-dependent updates and yields a joint approximation for exit direction and crossing time. The resulting law retains the discrete-time lattice phase. When tolerance shrinks as the inverse batch size, the limit changes to finite-step exit of a Gaussian autoregression. Numerical experiments distinguish these regimes and show why finite-parameter risk geometry matters for time prediction. Classical instability mechanisms and known class-imbalance failures are treated as prior knowledge; the candidate contribution is their precise connection to actual entropy updates and risk-defined safety.

## 1. Introduction

Test-time adaptation changes a model using observations from its deployment distribution. Entropy minimization is attractive because it uses predictions rather than labels. Its optimization objective, however, is not the classification risk. A low-entropy prediction can be confidently wrong, and continuing adaptation can remove an improvement that was present earlier.

The deterioration itself is established prior work. Weerasinghe et al. study partial class imbalance, show accuracy trajectories that improve and then decline, analyze normalization-parameter drift, and propose logit-adjusted entropy minimization [1]. We accept those results as valid. An additional example of collapse, a scalar restatement of drift, or the observation that entropy and accuracy can disagree would therefore be insufficient.

Our question is quantitative: under what distributional conditions does actual entropy adaptation cross the risk of the frozen classifier, and how does the crossing time depend on initialization, prior imbalance, batch size, step size, and tolerated risk increase? A useful answer must distinguish the geometry that produces instability from a favorable drift sign assumed along a trajectory. It must also distinguish fixed positive tolerance from a tolerance that vanishes with the perturbation or noise.

We use a controlled binary model. The feature extractor, class orientation, separation, variance, and confidence slope are fixed. Only a scalar threshold adapts. This restriction makes the entropy objective, supervised risk, and stochastic update simultaneously explicit. The same model contains both escaping and locally convergent population trajectories. It also exposes a singular distinction: at exact class balance the population threshold can remain optimal forever, while any finite batch makes its first nonzero update unsafe under zero tolerance.

The results develop three connected statements.

1. Distribution structure determines threshold stability. Weak separation gives a unique entropy maximum and explicit rebound conditions; sufficiently strong separation gives a safe population basin.
2. Safety duration depends on tolerance. Near balance, fixed positive tolerance gives logarithmic escape, while tolerance on the scale of the initial risk improvement can be crossed in finitely many steps.
3. Fresh-batch noise changes both direction and time. Outside the critical region, positive drift gives direct first-exit bounds. In the critical region, the true nonlinear updates admit a Gaussian-seed approximation with discrete-time and risk-boundary corrections.

These are mathematical statements about the declared model. Their novelty relative to entropy-drift analyses and classical random-exit theory requires direct comparison; no claim rests on finding an error in prior work.

## 2. Prior work and the precise comparison

**Imbalance, drift, and stabilization.** Logit-adjusted TTA [1] already studies degradation during long adaptation under partial class imbalance and connects it to BN shift and logit bias. Its adjustment uses target-prior estimates in the entropy objective. Persistent TTA, including PeTTA [2], studies accumulated errors in mixture-based classifiers. GALA [3] and STAG [4] provide practical controls using layer selection and structural gradients. These works preclude treating imbalance-induced degradation, iterative failure, or label-free update selection as new phenomena.

The comparison with [1] also requires matching the evaluation distribution. Its adaptation stream contains a subset of classes, while its All-Class Accuracy evaluates an independent holdout containing both observed and unobserved classes. Its Peak-to-Average Reduction measures a peak-to-time-average accuracy gap. Here every risk comparison uses the same fixed target distribution, and the event is the first iterate whose risk exceeds that of the frozen initialization by a specified amount. These different metrics cannot be substituted for one another.

**Recovery and information.** Recovery-complexity theory [5] makes local optimization and information conditions explicit. Identifiability analyses [6] show why unlabeled observations do not always determine a safe adaptation action. Our ordered Gaussian components and known anchor parameters supply additional structure; our theorems cannot be transferred to arbitrary observation channels or unknown anchors without further analysis.

**Classical mathematical mechanisms.** Convolution preserves log-concavity [7]. Small random perturbations near unstable equilibria have logarithmic exit times and random-seed corrections [8–10]. In particular, a Gaussian seed and a negative logarithm of its magnitude are not new mechanisms. The technical question here is whether the actual state-dependent mini-batch entropy update can be connected to such a seed while retaining its classification-risk boundary, finite-parameter geometry, and integer update times.

This manuscript establishes that connection within the stated scalar family. It does not assert that the underlying probability tools are new or that combining them automatically establishes sufficient novelty for a vision contribution.

## 3. Model, information, and risk safety

### 3.1. Anchored mixture and the deployed classifier

Let the target feature and label satisfy
\[
Z=\delta+Ym+\sigma\xi,\qquad
\Pr(Y=1)=p,\qquad \xi\sim N(0,1),
\]
where \(Y\in\{-1,+1\}\), \(0<p<1\), and \(m,\sigma,w>0\). The positive class orientation and the parameters \(m,\sigma\) are supplied by a source anchor. The translation \(\delta\) and target prior \(p\) characterize the shift. An oracle analysis may use them to describe the dynamics; an unlabeled deployment rule would have to estimate or otherwise handle their uncertainty.

The frozen classifier has threshold \(b_0=0\). Its prediction is \(\operatorname{sign}(Z-b)\), and its positive-class confidence is
\[
s(w(Z-b)),\qquad s(u)=(1+e^{-u})^{-1}.
\]
Only \(b\) adapts. The feature representation and positive slope \(w\) remain fixed.

Center the feature and threshold:
\[
z=Z-\delta,\qquad x=b-\delta,\qquad x_0=-\delta.
\]
The centered feature density is
\[
f_p(z)=p\varphi_\sigma(z-m)+(1-p)\varphi_\sigma(z+m),
\]
where \(\varphi_\sigma(z)=\sigma^{-1}\varphi(z/\sigma)\), and \(\varphi,\Phi\) are the standard normal density and CDF. Translation changes the initialization and the frozen-risk comparator, but not the centered entropy vector field.

### 3.2. Actual entropy updates

Let
\[
a(u)=s(u)(1-s(u)),\qquad
h(u)=-s(u)\log s(u)-(1-s(u))\log(1-s(u)).
\]
Define expected entropy and rightward velocity by
\[
F_p(x)=\mathbb E_p h(w(z-x)),\qquad v_p(x)=-F_p'(x).
\]
The rightward update from one sample is
\[
U(x,z)=-w^2(z-x)a(w(z-x)),\qquad \mathbb E_pU(x,z)=v_p(x).
\]
Thus population gradient descent is
\[
\bar x_{t+1}=\bar x_t+\eta v_p(\bar x_t),
\]
and fresh-batch SGD is
\[
x_{t+1}=x_t+\frac{\eta}{n}\sum_{i=1}^n U(x_t,z_{t,i}).
\]
The batch size is \(n\). Every batch is independent of the past, and all samples have the same target distribution. Labels are not read by the update. Reusing a fixed test set, dependent streams, changing priors, momentum, and feature learning are outside this recursion.

### 3.3. The comparator and the first unsafe iterate

Classification risk on an independent target draw is
\[
r_p(x)=p\Phi((x-m)/\sigma)+(1-p)\Phi((-x-m)/\sigma).
\]
The corresponding uncentered risk is \(R(b)=r_p(b-\delta)\). For tolerance \(\epsilon\ge0\), define
\[
T_\epsilon=\inf\{t\ge0:r_p(x_t)>r_p(x_0)+\epsilon\},
\qquad \inf\varnothing=\infty.
\]
The comparator is the frozen classifier evaluated on this same target distribution, not its source training risk. The strict inequality specifies the first unsafe iterate. The risk-safe set is
\[
\mathcal S_\epsilon=\{x:r_p(x)\le r_p(x_0)+\epsilon\}.
\]

A deployment rule intended to guarantee safety must reject an unsafe proposal before it is used. Low average risk over time does not imply that every deployed iterate is safe. Nor does an exit to the more frequent class represent safety: either boundary of \(\mathcal S_\epsilon\) is a violation.

## 4. Exact identities and entropy-kernel geometry

### Proposition 1. Risk geometry and uniform update bounds

The risk has a unique finite minimizer
\[
x_*(p)=\frac{\sigma^2}{2m}\log\frac{1-p}{p}.
\]
It decreases strictly left of \(x_*\), increases strictly right of \(x_*\), and satisfies
\[
r_p(-\infty)=1-p,\qquad r_p(+\infty)=p.
\]
Moreover,
\[
|U(x,z)|\le M:=w/4,\qquad
|\partial_xU(x,z)|\le L:=w^2/4.
\]
If \(0<\eta<2/L\), every nonstationary population update strictly decreases \(F_p\).

**Proof.** Differentiation gives
\[
r_p'(x)=\sigma^{-1}
\{p\varphi((x-m)/\sigma)-(1-p)\varphi((x+m)/\sigma)\}.
\]
The ratio of the positive and negative terms is
\[
\frac{p}{1-p}\exp(2mx/\sigma^2),
\]
which increases from zero to infinity and equals one only at \(x_*\). This proves the risk claims.

Write \(t(u)=\tanh(u/2)\). Then
\[
h'(u)=-ua(u),\qquad h''(u)=a(u)[ut(u)-1].
\]
For \(u\ge0\),
\(\cosh^2(u/2)\ge1+u^2/4\ge u\), so \(ua(u)\le1/4\).
If \(ut(u)\le1\), then \(|h''(u)|\le a(u)\le1/4\); otherwise
\(|h''(u)|\le ua(u)\le1/4\). Symmetry handles negative \(u\).
Scaling by \(w\) gives \(M,L\). Finally,
\[
F_p(x+\eta v_p(x))
\le F_p(x)-\eta(1-\eta L/2)|F_p'(x)|^2.
\]
The right-hand decrease is strict away from stationary points. \(\square\)

### Lemma 2. The entropy kernel is log-concave

The function \(h\) is positive, even, integrable, and log-concave. If
\[
K(t)=\mathbb E h(w(t+\sigma\xi)),
\]
then \(K\) is positive and even, and \(K'(t)<0\) for \(t>0\). Consequently,
\[
F_p(x)=pK(x-m)+(1-p)K(x+m).
\]

**Proof.** For \(u>0\), put \(q(u)=ua(u)\). Since \(h(\infty)=0\),
\[
h(u)=\int_u^\infty q(t)\,dt,\qquad
(\log q)''(u)=-u^{-2}-2a(u)<0.
\]
For every \(z>0\), the ratio \(q(u+z)/q(u)\) decreases with \(u\). Hence
\[
h(u)/q(u)=\int_0^\infty q(u+z)/q(u)\,dz
\]
decreases. Thus \(q/h\) increases and
\((\log h)'=-q/h\) decreases. Evenness and \(h'(0)=0\) extend log-concavity across zero. The tail is \(O((1+|u|)e^{-|u|})\), proving integrability.

For \(h_w(u)=h(wu)\), differentiation of the Gaussian convolution yields
\[
K'(t)=\int_0^\infty h_w'(u)
[\varphi_\sigma(t-u)-\varphi_\sigma(t+u)]\,du<0
\]
when \(t>0\): the derivative is negative and the bracket positive. \(\square\)

We use the standard convolution closure property: if two nonnegative integrable functions are log-concave, their convolution is log-concave [7]. The model-specific task is verifying the kernel and the mixture density, rather than claiming this closure result as a contribution.

## 5. Distribution structure determines population behavior

### Theorem 3. Weak separation gives a unique unstable entropy peak

Assume \(0<m\le\sigma\). There is a unique stationary point \(q_p\) of \(F_p\), and it is a strict global maximum. Its drift signs are
\[
v_p(x)<0\quad(x<q_p),\qquad v_p(x)>0\quad(x>q_p).
\]
The sign of \(q_p\) is the sign of \(p-1/2\). For every \(\eta>0\), population iterates initialized right of \(q_p\) increase to \(+\infty\); iterates initialized left of it decrease to \(-\infty\). Initialization at \(q_p\) stays fixed.

**Proof.** Direct differentiation gives
\[
(\log f_p)''(z)=
-\sigma^{-2}+\frac{m^2}{\sigma^4}
\operatorname{sech}^2\!\left(
\frac{mz}{\sigma^2}+\frac12\log\frac p{1-p}\right)\le0.
\]
By Lemma 2 and convolution closure, \(F_p\) is log-concave. It is positive, integrable, and vanishes at both infinities, so it has a maximum. Gaussian smoothing makes it real analytic. Two maximizers would force an interval of maxima by log-concavity, then a globally constant function by analyticity, contradicting the tails. Any stationary point of a differentiable positive log-concave function is a maximizer. The stationary point is therefore unique, with the asserted derivative signs.

Since
\[
F_p'(0)=(1-2p)K'(m),
\]
the sign of its maximizer follows from \(K'(m)<0\). An iterate to the right remains to the right and increases strictly. If it had a finite limit \(\ell>q_p\), the recursion would imply \(v_p(\ell)=0\), a contradiction. The left side is identical. \(\square\)

Escape monotonicity holds for every positive step size. Entropy descent additionally requires a condition such as \(0<\eta<2/L\).

### Corollary 4. An explicit mixed-shift rebound region

Fix \(\epsilon\ge0\). Suppose
\[
0<m\le\sigma,\qquad
\Phi(-m/\sigma)+\epsilon<p<1/2,\qquad -x_*(p)<\delta\le0,
\]
and
\[
0<\eta<\min\left\{\frac8{w^2},
\frac{4[x_*(p)+\delta]}{w}\right\}.
\]
Then the first population update strictly reduces classification risk, every population update strictly reduces entropy, and \(T_\epsilon<\infty\). The iterates tend to \(+\infty\) and risk tends to \(p\). Taking \(m<\sigma,\delta<0\) with strict inequalities gives a nonempty open set with genuine translation and prior shift. Reflection \((x,\delta,p)\mapsto(-x,-\delta,1-p)\) gives the other orientation.

**Proof.** Here \(x_0=-\delta\in[0,x_*)\) and \(q_p<0\). Theorem 3 gives rightward escape. The first step is at most \(\eta M<x_*-x_0\), so it stays in the strict risk-decrease region. Proposition 1 gives entropy descent. Finally,
\[
r_p(x_0)\le r_p(0)=\Phi(-m/\sigma)<p-\epsilon.
\]
Convergence of risk to \(p\) therefore forces a finite strict crossing. \(\square\)

For \(p<1/2\), weak separation also yields the following classification.

| Initialization | Threshold motion | Risk behavior |
|---|---|---|
| \(x_0<q_p\) | To \(-\infty\) | Immediately worsens; limit \(1-p\) |
| \(x_0=q_p\) | Fixed | No population adaptation |
| \(q_p<x_0<x_*\) | To \(+\infty\) | Initially improves for a sufficiently small step; then rises |
| \(x_0\ge x_*\) | To \(+\infty\) | Strictly worsens; limit \(p\) |

On a rightward trajectory, \(T_\epsilon<\infty\) exactly when \(r_p(x_0)+\epsilon<p\). On a leftward trajectory the corresponding criterion uses \(1-p\). Thus collapse toward one predicted class does not automatically imply crossing the risk of an already poor initialization.

### Theorem 5. A sufficiently separated safe population basin

Set \(p=1/2\), \(a_m=m/\sigma>1\), and
\[
C_0=\int h(u)\,du=\pi^2/3,\qquad
C_2=\int u^2h(u)\,du=7\pi^4/45.
\]
If
\[
(w\sigma)^2>
\frac{3C_2}{2C_0}\frac{e^{a_m^2/2}}{a_m^2-1},
\]
then \(F_{1/2}''(0)>0\). There is an \(r>0\) such that, for
\(0<|\delta|<r\) and \(0<\eta\le1/L\), population iterates converge monotonically to zero and their risks strictly decrease to Bayes risk.

For any fixed nonzero initialization strictly inside this basin and \(\eta<1/L\), a sufficiently small open perturbation of the parameters permits \(p\ne1/2,\delta\ne0\): the iterates converge to a finite entropy minimum \(x_e\), and
\[
r_p(x_t)\le r_p(x_0)\quad\text{for every }t,\qquad
r_p(x_e)<r_p(x_0).
\]

**Proof.** For any smooth density \(f\), a change of variables gives
\[
F''(0)=\frac1w\int h(u)f''(u/w)\,du.
\]
The odd first-order Taylor term integrates to zero, so
\[
\left|F''(0)-\frac{C_0}{w}f''(0)\right|
\le\frac{C_2}{2w^3}\|f^{(4)}\|_\infty.
\]
For the balanced Gaussian mixture,
\[
f''(0)=\frac{a_m^2-1}{\sigma^3}\varphi(a_m),\qquad
\|f^{(4)}\|_\infty\le\frac{3\varphi(0)}{\sigma^5}.
\]
The latter follows by integrating the absolute Fourier transform of the fourth Gaussian derivative; a mixture preserves the bound. The displayed sufficient condition makes the curvature positive.

For an explicit radius, define
\[
\kappa_0=\frac{C_0}{w}f''(0)
-\frac{C_2}{2w^3}\frac{3\varphi(0)}{\sigma^5}>0,
\]
\[
B_3=\min\{3w^3/4,\;2C_0/(\pi w\sigma^4)\},\qquad r=\kappa_0/(2B_3).
\]
The first bound on \(\|F'''\|_\infty\) follows from
\[
h'''(u)=a(u)\{2\tanh(u/2)+u[2a(u)-\tanh^2(u/2)]\},
\quad |h'''|\le3/4.
\]
The second follows from
\(\|F'''\|_\infty\le(C_0/w)\|f'''\|_\infty\) and
\(\|f'''\|_\infty\le2/(\pi\sigma^4)\).
Thus \(F''\ge\kappa_0/2\) on \([-r,r]\). Evenness gives \(F'(0)=0\). For \(\eta\le1/L\), the update map is nondecreasing and does not cross zero; positive curvature contracts it toward zero. Balanced risk increases strictly with \(|x|\), proving strict risk decrease.

The implicit function theorem preserves the entropy minimum and a common contraction interval under small parameter perturbations. The iterates stay between \(x_0\) and \(x_e\). Strict endpoint improvement persists by continuity, and a function with a unique risk minimum has its maximum over this interval at an endpoint. This proves the perturbed safety statement. The constants \(C_0,C_2\) follow by integrating the expansion
\(h(u)=\sum_{j\ge1}(-1)^{j+1}e^{-ju}(j^{-1}+u)\) on the positive half-line and using symmetry. \(\square\)

This is a local sufficient condition, not the exact boundary between all stable and unstable regimes. It does not guarantee that fixed-step SGD remains in the basin forever.

## 6. Deterministic safety duration near class balance

Assume \(0<m\le\sigma\) and write \(p_\rho=(1-\rho)/2\). Subscripts \(\rho\) denote the corresponding prior. Define
\[
\beta=-K'(m)>0,\qquad \lambda=-K''(m)>0,\qquad
k_0=\sigma^2/m,\qquad A=1+\eta\lambda.
\]
Positivity of \(\lambda\) follows from the balanced density \(f_0\): for \(z>0\),
\[
(\log f_0)'(z)=\frac{-z+m\tanh(mz/\sigma^2)}{\sigma^2}<0.
\]
Since \(h_w'(z)<0\), integration by parts gives
\[
F_0''(0)=-\int h_w'(z)f_0'(z)\,dz<0.
\]
Near the origin,
\[
v_\rho(x)=\beta\rho+\lambda x+O(x^2+|\rho x|+\rho^2).
\]
The implicit function theorem then gives
\[
q_\rho=-\frac\beta\lambda\rho+O(\rho^2),\qquad
\lambda_\rho=v_\rho'(q_\rho)=\lambda+O(\rho).
\]

### Theorem 6. Fixed and shrinking tolerances have different time scales

Fix \(\eta>0\) and initialize \(x_0=\alpha_0\rho\), where
\[
-\beta/\lambda<\alpha_0<k_0.
\]
For fixed
\[
0<\epsilon<1/2-\Phi(-m/\sigma),
\]
the population crossing time satisfies
\[
T_\epsilon(\rho)=\frac{\log(1/\rho)}{\log(1+\eta\lambda)}+O(1),
\qquad \rho\downarrow0.
\]
If also
\[
\eta<\min\{2/L,\;(k_0-\alpha_0)/(\beta+\lambda\alpha_0)\},
\]
the first update improves risk and every update reduces entropy.

For \(\epsilon=c\rho^2\), \(c\ge0\), the crossing time is bounded as \(\rho\downarrow0\). Put
\[
B=\frac{m\varphi(m/\sigma)}{\sigma^3},\qquad
u_t=(\alpha_0+\beta/\lambda)A^t-\beta/\lambda,
\]
\[
u_c=k_0+\sqrt{(\alpha_0-k_0)^2+2c/B}.
\]
If no integer \(t\) satisfies \(u_t=u_c\), then, for sufficiently small \(\rho>0\),
\[
T_{c\rho^2}=
\left\lfloor
\frac{\log[(u_c+\beta/\lambda)/(\alpha_0+\beta/\lambda)]}
{\log A}\right\rfloor+1.
\]
At an integer equality \(u_j=u_c\), the first crossing is \(j\) or \(j+1\), depending on higher-order terms.

**Proof.** Center the recursion at its actual peak and set \(d_t=x_t-q_\rho\). The initialization satisfies
\[
d_0=(\alpha_0+\beta/\lambda)\rho+O(\rho^2)>0,
\]
and, in a fixed small neighborhood,
\[
d_{t+1}=(1+\eta\lambda_\rho)d_t+O(d_t^2).
\]
Until \(d_t\) first reaches a fixed small \(r>0\), it grows at a geometric rate bounded away from one, uniformly for small \(\rho\). The exit overshoot is bounded, and summing backward from exit gives \(\sum d_t=O(1)\). Taking logarithms and summing yields
\[
\log d_{\tau_r}-\log d_0
=\tau_r\log(1+\eta\lambda_\rho)+O(1).
\]
Since \(\rho\log(1/\rho)\to0\),
\[
\tau_r=\log(1/\rho)/\log A+O(1).
\]
The right risk boundary converges to the strictly positive finite solution \(c_\epsilon\) of
\(r_0(c_\epsilon)=r_0(0)+\epsilon\). From the small neighborhood to that boundary, drift has a uniform positive lower bound on a compact interval. Only \(O(1)\) additional steps are needed.

For shrinking tolerance, fixed-step recursion expansion gives \(x_t/\rho\to u_t\). The derivatives
\[
r_\rho'(0)=-\rho\varphi(m/\sigma)/\sigma,\qquad r_0''(0)=B
\]
give, for bounded \(u\),
\[
r_\rho(\rho u)-r_\rho(\rho\alpha_0)
=\frac{B\rho^2}{2}(u-\alpha_0)(u+\alpha_0-2k_0)+o(\rho^2).
\]
The upper equality boundary for tolerance \(c\rho^2\) is \(u_c\). Strict signs at the finitely many candidate crossing steps give the integer formula; equality requires the next-order terms. Finally,
\(x_*(p_\rho)=k_0\rho+O(\rho^3)\), and the step-size condition places the first iterate strictly between initialization and this minimizer. \(\square\)

At exact balance with \(x_0=0\), the population recursion remains fixed forever. This does not contradict either limiting regime. The available risk improvement under a small prior perturbation is itself only \(O(\rho^2)\). A longer fixed-tolerance horizon need not mean a larger useful improvement.

## 7. Direct first-exit bounds in a positive-drift corridor

Write SGD as
\[
x_{t+1}=x_t+\eta[v_p(x_t)+\xi_t],\qquad
\mathbb E[\xi_t\mid\mathcal F_t]=0,
\]
where \(\mathcal F_t\) contains the history before the new batch. Proposition 1 and Hoeffding's lemma imply
\[
\mathbb E[e^{s\xi_t}\mid\mathcal F_t]
\le\exp(s^2M^2/(2n)).
\]
The conditional formulation permits state-dependent noise.

### Theorem 7. Two-sided first-exit control

Suppose \(\mathcal S_\epsilon=[a,c]\) is finite,
\(a<x_0<c\), and
\[
0<\nu\le v_p(x)\le V\qquad(x\in[a,c]).
\]
Let \(d=x_0-a\), \(D=c-x_0\), and \(\tau=T_\epsilon\). Then
\[
\Pr(x_\tau<a,\tau<\infty)\le
\exp\left(-\frac{2n\nu d}{\eta M^2}\right).
\]
For integers \(t\ge1\),
\[
\Pr(\tau\le t,x_\tau>c)
\le\exp\left[-\frac{n(D-\eta Vt)^2}{2\eta^2M^2t}\right]
\quad\text{if }D>\eta Vt,
\]
and
\[
\Pr(\tau>t)
\le\exp\left[-\frac{n(\eta\nu t-D)^2}{2\eta^2M^2t}\right]
\quad\text{if }\eta\nu t>D.
\]
Calling the three bounds \(\alpha_L,\alpha_E(t),\alpha_D(t)\), respectively,
\[
\Pr(t_-<T_\epsilon\le t_+,\ x_{T_\epsilon}>c)
\ge1-\alpha_L-\alpha_E(t_-)-\alpha_D(t_+).
\]

**Proof.** Define the stopped noise martingale
\[
S_t=\eta\sum_{j<t\wedge\tau}\xi_j.
\]
Its increments have conditional variance proxy at most \(\eta^2M^2/n\). With \(\kappa=2n\nu/(\eta M^2)\),
\[
\mathbb E[e^{-\kappa(x_{t+1}-x_t)}\mid\mathcal F_t]
\le e^{-\kappa\eta\nu+\kappa^2\eta^2M^2/(2n)}=1
\]
before exit. Thus \(e^{-\kappa(x_{t\wedge\tau}-x_0)}\) is a nonnegative supermartingale. Left exit exceeds the level \(e^{\kappa d}\), giving the first bound by the maximal inequality.

If right exit occurs at \(s\le t\), then
\[
S_s=x_s-x_0-\eta\sum_{j<s}v_p(x_j)>D-\eta Vt.
\]
An exponential martingale maximal bound, optimized over its exponent on the fixed time window, gives the second inequality. If \(\tau>t\), then \(x_t\le c\) and all \(t\) drifts are at least \(\nu\), so
\(S_t\le D-\eta\nu t\). The conditional moment bound gives the third inequality. A union bound proves the interval statement. The late-exit bound also gives \(\tau<\infty\) almost surely. \(\square\)

The theorem allows backward stochastic steps and overshoot. Its price is a positive drift bound on the entire safe interval. It cannot cover a critical safe interval containing the entropy peak.

For comparison, a standard global tracking argument gives, for fixed \(T\ge1\) and failure level \(\alpha\),
\[
q=M\sqrt{2\log(2T/\alpha)/n},\qquad
e_t=q[(1+\eta L)^t-1]/L.
\]
With probability at least \(1-\alpha\), simultaneously through \(T\),
\[
|x_t-\bar x_t|\le e_t,\qquad
|r_p(x_t)-r_p(\bar x_t)|\le e_t/(\sigma\sqrt{2\pi}).
\]
Conditional Hoeffding bounds followed by
\(d_{t+1}\le(1+\eta L)d_t+\eta q\) prove this. Its exponential amplification can make it uninformative before rebound; it is a baseline, not a principal result.

## 8. Critical nonlinear SGD: a quantitative joint distribution

Throughout this section fix \(0<m\le\sigma\),
\(0<\eta<1/L\), and
\[
0<\epsilon<1/2-\Phi(-m/\sigma).
\]
Take \(|\rho|\le\rho_0\) for a sufficiently small fixed \(\rho_0\). Constants may depend on these fixed parameters and on the initialization bound, but not on \(n,t\) or the parameter sequence within this neighborhood.

### Lemma 8. A normalized deterministic conjugacy

Let \(G_\rho(x)=x+\eta v_\rho(x)\) and
\(A_\rho=G_\rho'(q_\rho)=1+\eta\lambda_\rho>1\).
There is a strictly increasing \(C^2\) function \(H_\rho\) with
\[
H_\rho(q_\rho)=0,\qquad H_\rho'(q_\rho)=1,\qquad
H_\rho(G_\rho(x))=A_\rho H_\rho(x).
\]
On fixed compact intervals, \(H_\rho,H_\rho',H_\rho''\) are uniformly bounded for small \(\rho\), \(H_\rho'\) has a uniform positive lower bound, and these functions depend continuously on \(\rho\). At balance, \(H_0\) is odd.

**Proof.** Since \(G_\rho'\ge1-\eta L>0\) and its displacement is bounded, it is a smooth increasing bijection of the line. Around its fixed point, its centered inverse satisfies
\[
g_\rho(y)=G_\rho^{-1}(q_\rho+y)-q_\rho
=A_\rho^{-1}y+O(y^2).
\]
Shrink parameter and spatial neighborhoods so that
\(|g_\rho'|\le b<1\) and \((\sup_\rho A_\rho)b^2<1\). Define locally
\[
H_\rho(q_\rho+y)=\lim_{j\to\infty}A_\rho^j g_\rho^{\,j}(y).
\]
For \(R_\rho(y)=g_\rho(y)-A_\rho^{-1}y\), both \(R_\rho(0)\) and \(R_\rho'(0)\) vanish. Iterated contraction bounds the value and first two derivatives of \(g_\rho^{\,j}\) by \(Cb^j\). The difference between consecutive approximants, together with its first two derivatives, is bounded by \(C(A_\rho b^2)^j\). The limit exists in \(C^2\), has derivative one at zero, and has positive derivative nearby. Shifting the inverse iteration proves the conjugacy identity.

Every inverse orbit converges monotonically to the unique fixed point: an orbit right of it decreases and can have no other finite fixed-point limit, and similarly on the left. Finitely many inverse steps therefore bring any fixed compact interval into the local domain, uniformly over small \(\rho\). The functional identity extends \(H_\rho\), preserving regularity and positive derivative. The construction also proves parameter continuity. At \(\rho=0\), the map and inverse are odd, hence so is \(H_0\). \(\square\)

### Theorem 9. State-dependent SGD admits a Gaussian-seed exit approximation

Suppose
\[
|x_0-q_\rho|\le K/\sqrt n
\]
for a fixed \(K<\infty\). For sufficiently large \(n\), the safe set is a finite interval \([a_n,c_n]\) containing \(q_\rho\) in its interior, with peak-to-boundary distances uniformly positive. Put
\[
\tau=T_\epsilon,\qquad
J=\begin{cases}+1,&x_\tau>c_n,\\-1,&x_\tau<a_n,\end{cases}
\]
\[
\mu_n=\sqrt n H_\rho(x_0),\qquad
s_n^2=\frac{\eta^2\operatorname{Var}_\rho U(q_\rho,z)}{A_\rho^2-1},
\]
\[
B_n^+=H_\rho(c_n)>0,\qquad B_n^-=-H_\rho(a_n)>0.
\]
Then \(\tau<\infty\) almost surely. There is a constant \(C\) such that, for every integer \(t\ge0\),
\[
\left|\Pr(\tau\le t,J=+1)
-\Phi\!\left(\frac{\mu_n-\sqrt n A_\rho^{-t}B_n^+}{s_n}\right)\right|
\le C\left[n^{-1/2}+
\left(\frac{1+\log n}{n}+A_\rho^{-2t}\right)^{1/3}\right],
\]
\[
\left|\Pr(\tau\le t,J=-1)
-\Phi\!\left(\frac{-\mu_n-\sqrt n A_\rho^{-t}B_n^-}{s_n}\right)\right|
\le C\left[n^{-1/2}+
\left(\frac{1+\log n}{n}+A_\rho^{-2t}\right)^{1/3}\right].
\]
Also,
\[
|\Pr(J=+1)-\Phi(\mu_n/s_n)|
\le C[n^{-1/2}+((1+\log n)/n)^{1/3}].
\]
In the logarithmic window \(nA_\rho^{-2t}=O(1)\), the error is \(O((\log n/n)^{1/3})\). This rate is not asserted optimal, and \(C\) is not numerically calibrated.

**Proof, step 1: stopping and moments.** Define the centered batch velocity
\[
\xi_j(x)=n^{-1}\sum_{i=1}^nU(x,z_{j,i})-v_\rho(x).
\]
For a state measurable before batch \(j\),
\[
\mathbb E[\xi_j(x)^2\mid\mathcal F_j]\le M^2/n,\qquad
\mathbb E[\xi_j(x)^4\mid\mathcal F_j]\le C/n^2,
\]
\[
\mathbb E[(\xi_j(x)-\xi_j(q_\rho))^2\mid\mathcal F_j]
\le L^2|x-q_\rho|^2/n.
\]
These follow from bounded independent samples and the sample Lipschitz bound.

All safe intervals lie in a common compact interval. Choose a bounded sample interval strictly to its left. On this interval \(U(x,z)\) is uniformly positive for safe \(x\). A sufficiently long block whose samples all belong to that interval forces right exit. For every fixed \(n\) this block has positive probability, independently across disjoint blocks, proving almost-sure finite exit.

On the safe interval and its one-step overshoot enlargement, Taylor expansion of the conjugacy gives
\[
H_\rho(x_{j+1})=
A_\rho H_\rho(x_j)+
\eta H_\rho'(G_\rho(x_j))\xi_j(x_j)+R_j,
\qquad |R_j|\le C\xi_j(x_j)^2.
\]
Define
\[
Y_{n,t}=\sqrt n A_\rho^{-(t\wedge\tau)}
H_\rho(x_{t\wedge\tau}).
\]
Summing the expansion with geometric weights gives a stopped martingale plus a remainder with \(L^2\) norm at most
\[
C\sqrt n\sum_{j\ge0}A_\rho^{-j-1}/n\le C/\sqrt n.
\]
Consequently,
\[
\sup_t\mathbb E|Y_{n,t}|^2\le C,\qquad
\mathbb E|Y_{n,\tau}|^2\le C.
\]
At exit \(|H_\rho(x_\tau)|\ge\min(B_n^+,B_n^-)\ge b>0\), so
\[
\mathbb E A_\rho^{-2\tau}\le C/n.
\]

**Step 2: freeze noise at the peak.** On the same infinite sample array define
\[
W_n=\mu_n+\eta\sum_{j\ge0}A_\rho^{-j-1}\sqrt n\,\xi_j(q_\rho).
\]
Before exit, the difference
\[
D_j=H_\rho'(G_\rho(x_j))\xi_j(x_j)-\xi_j(q_\rho)
\]
has conditional mean zero and conditional second moment at most
\(C|x_j-q_\rho|^2/n\). Positive lower bounds on \(H_\rho'\), the preceding moment estimate, and compactness give
\[
\mathbb E[\mathbf1_{\{j<\tau\}}|x_j-q_\rho|^2]
\le C\min\{A_\rho^{2j}/n,1\}.
\]
Orthogonality of martingale differences therefore implies
\[
\mathbb E\left|
\sqrt n\eta\sum_{j<t\wedge\tau}A_\rho^{-j-1}D_j
\right|^2
\le C\sum_{j\ge0}A_\rho^{-2j}\min\{A_\rho^{2j}/n,1\}
\le C(1+\log n)/n.
\]
The frozen-noise tail after \(t\wedge\tau\) has conditional variance
\(s_n^2A_\rho^{-2(t\wedge\tau)}\). Combining the tail estimate with the remainder yields
\[
\mathbb E|Y_{n,t}-W_n|^2
\le C[(1+\log n)/n+A_\rho^{-2t}],
\]
and, at exit,
\[
\mathbb E|Y_{n,\tau}-W_n|^2\le C(1+\log n)/n.
\]
This step controls the actual state dependence rather than assuming Gaussian state-independent noise.

**Step 3: Gaussian approximation and exit events.** Since \(v_\rho(q_\rho)=0\), rearrange by sample column:
\[
W_n-\mu_n=\frac1{\sqrt n}\sum_{i=1}^nV_{n,i},\qquad
V_{n,i}=\eta\sum_{j\ge0}A_\rho^{-j-1}U(q_\rho,z_{j,i}).
\]
The columns are independent and identically distributed, with mean zero, variance \(s_n^2\) uniformly bounded below, and
\[
|V_{n,i}|\le\eta M/(A_\rho-1).
\]
The Berry–Esseen bound for a normalized sum of independent variables with finite third absolute moment gives
\[
\sup_z|\Pr(W_n\le z)-\Phi((z-\mu_n)/s_n)|\le C/\sqrt n.
\]

The stopped coordinate has exact event identities:
\[
\{\tau\le t,J=+1\}=
\{Y_{n,t}>\sqrt n A_\rho^{-t}B_n^+\},
\]
\[
\{\tau\le t,J=-1\}=
\{Y_{n,t}<-\sqrt n A_\rho^{-t}B_n^-\}.
\]
Before exit the coordinate remains between its two boundaries. If right exit occurs at \(s\le t\), the coordinate is frozen with
\(A_\rho^{-s}H_\rho(x_s)>A_\rho^{-t}B_n^+\); the left case is analogous.

For any threshold and smoothing width \(h>0\), the mean-square comparison, Chebyshev's inequality, and the bounded Gaussian density bound the CDF error by
\[
C[n^{-1/2}+h+\Delta/h^2],\qquad
\Delta=(1+\log n)/n+A_\rho^{-2t}.
\]
Choose \(h=\Delta^{1/3}\). The stopped-at-exit estimate and
\(\{J=+1\}=\{Y_{n,\tau}>0\}\) give the direction bound. \(\square\)

## 9. Critical scaling, lattice phase, and microscopic tolerance

### Corollary 10. The critical joint limit

Suppose
\[
\sqrt n\rho_n\to c\in\mathbb R,\qquad
\sqrt n x_{0,n}\to u\in\mathbb R.
\]
Define
\[
\mu=u+\beta c/\lambda,\qquad
s_0^2=\operatorname{Var}_0U(0,z)>0,
\]
\[
s_\eta^2=\frac{\eta^2s_0^2}{A^2-1}
=\frac{\eta s_0^2}{\lambda(2+\eta\lambda)}.
\]
Let \(c_\epsilon>0\) solve \(r_0(c_\epsilon)=r_0(0)+\epsilon\), and set
\(C_\epsilon=H_0(c_\epsilon)>0\). Then
\[
\Pr(J=+1)\to\Phi(\mu/s_\eta).
\]
For integer time sequences with \(\sqrt n A^{-t_n}\to r\in(0,\infty)\),
\[
\Pr(T_\epsilon\le t_n,J=+1)\to
\Phi((\mu-C_\epsilon r)/s_\eta),
\]
\[
\Pr(T_\epsilon\le t_n,J=-1)\to
\Phi((-\mu-C_\epsilon r)/s_\eta),
\]
and
\[
T_\epsilon=\frac{\log n}{2\log A}+O_{\mathbb P}(1).
\]

**Proof.** Peak and conjugacy expansions give
\(\sqrt nH_{\rho_n}(x_{0,n})\to\mu\). Continuity gives
\(s_n\to s_\eta\) and \(B_n^\pm\to C_\epsilon\).
Since \(\log A_{\rho_n}-\log A=O(\rho_n)\) and \(t_n=O(\log n)\),
\(A_{\rho_n}^{-t_n}/A^{-t_n}\to1\). Substitute into Theorem 9. Evaluating the two tails a sufficiently large fixed number of steps on either side of the center proves tightness. \(\square\)

The lattice cannot generally be removed at fixed step size. Put
\[
b_n=\log\sqrt n/\log A.
\]
Along a subsequence whose fractional part satisfies \(\{b_n\}\to\theta\), for integer \(k\),
\[
\Pr(T_\epsilon-\lfloor b_n\rfloor\le k)
\to\Pr(|N(\mu,s_\eta^2)|>C_\epsilon A^{\theta-k}).
\]
Equivalently, the integer-valued limit is
\[
\left\lceil\theta+
\frac{\log C_\epsilon-\log|N(\mu,s_\eta^2)|}{\log A}\right\rceil.
\]
Gaussian boundary atoms have probability zero. The time fluctuation is a rounded negative logarithm of a Gaussian magnitude; it is not Gaussian and is not automatically Gumbel.

The effective seed signal-to-noise ratio is
\[
\kappa_n\simeq\frac{\sqrt n(x_0-q_\rho)}{s_\eta}
\simeq\frac{\sqrt n(\beta\rho/\lambda-\delta)}{s_\eta}.
\]
Translation can cancel or reinforce the leading prior seed. For pure prior shift,
\[
\rho_{\rm cross}\simeq\frac{\lambda s_\eta}{\beta\sqrt n}.
\]
If \(\rho_n\downarrow0\), \(\sqrt n\rho_n\to\infty\),
\(x_0=\alpha_0\rho_n\), and \(\alpha_0+\beta/\lambda\ne0\), the seed dominates noise and
\[
T_\epsilon=\log(1/\rho_n)/\log A+O_{\mathbb P}(1).
\]
To see this, use the same weighted conjugacy expansion normalized by \(\rho_n\): the noise is \(O_{\mathbb P}(n^{-1/2})=o_{\mathbb P}(\rho_n)\), the Taylor remainder is \(O_{\mathbb P}(n^{-1})\), and the initial seed is
\((\alpha_0+\beta/\lambda)\rho_n+o(\rho_n)\). The sign is consequently deterministic in the limit. Leading-seed cancellation requires another scale analysis.

Although \(s_\eta\sim s_0\sqrt{\eta/(2\lambda)}\) algebraically as \(\eta\downarrow0\), all probability limits above fix \(\eta>0\). They do not prove a joint limit with \(\eta=\eta_n\to0\). Likewise the logarithmic batch-size delay applies to balanced or critically rescaled families, not to every fixed nonzero prior.

### Theorem 11. Inverse-batch tolerance gives finite-step AR exit

Retain the critical scaling and let \(n\epsilon_n\to e\in[0,\infty)\). Define
\[
X_0=u,\qquad X_{t+1}=AX_t+\eta\beta c+\eta s_0Z_t,
\qquad Z_t\overset{\rm iid}{\sim}N(0,1),
\]
and
\[
I_{c,u,e}=
[k_0c-\sqrt{(u-k_0c)^2+2e/B},\
 k_0c+\sqrt{(u-k_0c)^2+2e/B}],
\]
where \(B=m\varphi(m/\sigma)/\sigma^3\).
For \(\tau_{\rm AR}=\inf\{t\ge1:X_t\notin I_{c,u,e}\}\),
\[
T_{\epsilon_n}\Rightarrow\tau_{\rm AR},\qquad
T_{\epsilon_n}=O_{\mathbb P}(1).
\]

**Proof.** Uniformly on bounded scaled states,
\[
v_{\rho_n}(y/\sqrt n)
=(\lambda y+\beta c)/\sqrt n+o(n^{-1/2}).
\]
A batch CLT and continuity give joint convergence of the scaled iterates at every fixed finite collection of times. Taylor expansion of risk yields
\[
n[r_{\rho_n}(y/\sqrt n)-r_{\rho_n}(x_{0,n})]
\to Q_c(y)-Q_c(u),\qquad Q_c(y)=\frac B2(y-k_0c)^2.
\]
Here \(Bk_0=\varphi(m/\sigma)/\sigma\), agreeing with the linear risk term. The limiting safe set is exactly \(I_{c,u,e}\).

Each \(X_t\), \(t\ge1\), has a continuous density, so boundary equality has zero probability and the finite-time first-exit CDFs converge. On a bounded interval, a Gaussian innovation has a uniformly positive chance of exiting in one step. Hence \(\tau_{\rm AR}<\infty\) almost surely. Finite-time CDF convergence to a proper exit law also gives tightness. \(\square\)

### Proposition 12. Zero tolerance and a limitation of shrinking the step

If \(\rho=0,x_0=0,\epsilon=0\), then for every finite \(n\) and every \(\eta>0\),
\[
T_0=1\quad\text{almost surely}.
\]
For pure prior shift \(x_0=0,\sqrt n\rho_n\to c>0\), every deterministic sequence \(\eta_n>0\) satisfies
\[
\liminf_n\Pr(r_{\rho_n}(x_1)>r_{\rho_n}(0))
\ge\Phi(-\beta c/s_0)>0.
\]
For fixed \(\eta>0\), the first-step improvement probability converges to
\[
\Phi\!\left(\frac{(2k_0-\eta\beta)c}{\eta s_0}\right)
-\Phi(-\beta c/s_0).
\]

**Proof.** At exact balance, zero is the unique risk minimum. The analytic nonconstant function \(U(0,z)\), applied to a continuously distributed feature, has no atoms: each level set has Lebesgue measure zero. An independent sample average is therefore nonzero almost surely, proving first-step violation.

For positive prior bias, the risk minimizer is right of zero. Any negative empirical velocity moves left and strictly increases risk for every positive step size. The CLT
\[
\sqrt n\,n^{-1}\sum_iU(0,z_i)\Rightarrow N(\beta c,s_0^2)
\]
proves the lower bound independently of \(\eta_n\). For fixed \(\eta\), improvement requires the limiting scaled first iterate to lie between \(0\) and \(2k_0c\), giving the displayed Gaussian interval probability. \(\square\)

The limitation concerns unconditional execution of the empirical entropy direction. It does not exclude rejecting an update, using extra information, or a different unlabeled adaptation method. The probability of a strict risk increase also says nothing by itself about the size of that increase.

## 10. Numerical evidence

All reported experiments are synthetic and use the actual nonlinear entropy update with fresh batches. Gaussian autoregressions are separately identified reference limits. Labels enter the synthetic generator and analytic evaluation risk, not the update. The saved experiments were run before this manuscript revision; no new simulation is implied by the presentation below.

### 10.1. Population rebound and safe convergence

For \(m=\sigma=1,w=2,\eta=.1,\epsilon=.005\), with \(p=.35\), Gaussian quadrature gives:

| Translation \(\delta\) | Frozen target risk | Minimum risk | Minimum-risk step | \(T_{.005}\) |
|---:|---:|---:|---:|---:|
| 0 | .158655254 | .147597606 | 47 | 94 |
| -.1 | .152603983 | .147597604 | 30 | 67 |
| .1 | .167122203 | .147597778 | 67 | 124 |

Entropy decreases throughout 2500 updates. The first two rows satisfy the explicit sufficient rebound region. The last row is an additional numerical example, not a claim that the same simple sufficient inequalities cover it.

An initial exploratory pilot considered 30 cells:
\(\delta\in\{-.6,-.3,0,.3,.6,1\}\) and
\(p\in\{.2,.35,.5,.65,.8\}\), using 160-point Gauss–Hermite quadrature. Eight cells improved risk by more than .001 and later exceeded frozen risk by .005 within 2500 steps. The illustrated zero-translation case was selected from this grid. This is exploratory evidence, not a preregistered estimate of how frequently rebound occurs.

For safe convergence, \(m=2,\sigma=1,w=5\) satisfies the curvature condition: its right-hand side is approximately 17.0163, below \((w\sigma)^2=25\). The explicit radius is about .0406252. Separate numerical trajectories with \(\delta=.2,\eta=.02\), beyond that conservative radius, reduce risk from .0249169 to .0227501 at \(p=.5\), and from .0251372 to .0227471 at \(p=.49\), over 6000 steps. These larger-initialization examples do not enlarge the proved basin.

### 10.2. Population tolerance contrast

For \(m=\sigma=1,w=2,\eta=.1,x_0=0\),
\[
\beta=.1835111816,\qquad\lambda=.0825771460,\qquad
1/\log A=121.598196.
\]

| \(\rho\) | \(T_{.005}\) | \(T_0\) |
|---:|---:|---:|
| .1 | 112 | 79 |
| .01 | 288 | 79 |
| .001 | 552 | 79 |
| .0001 | 830 | 79 |
| .00001 | 1110 | 79 |
| .000001 | 1390 | 79 |

For small \(\rho\), \(T_{.005}-121.598196\log(1/\rho)\) stays near \(-290\), with integer oscillations. The zero-tolerance limiting index before rounding is about 78.046262, predicting step 79. Near zero tolerance, risk differences are evaluated by a scaled integral rather than subtraction of nearly equal CDF values. A plotted asymptote uses the theoretical slope and a fitted intercept from the smallest four imbalance values; it is a slope check, not an independently predicted absolute time.

### 10.3. A nonvacuous positive-drift bound

For the \(p=.35,\delta=0\) example, the safe interval is approximately
\[
[a,c]=[-.062394603,\ .713054205].
\]
Conservative constants are
\[
d\ge.062,\quad .713\le D\le.714,\quad
\nu=.049,\quad V=.098,\quad M=.5.
\]
The drift calculation uses standard-normal truncation to \([-8,8]\), 8192 midpoint subintervals, and 801 threshold grid points over \([-.063,.714]\). Its discretization, tail, and interpolation bounds give the enclosure
\[
[.0492117163,\ .0977563734].
\]
For clarity, if \(Q\) is sample velocity times standard-normal density,
\[
\|Q''\|_\infty\le
(3w^3/4)\sigma^2\varphi(0)+
2L\sigma/\sqrt{2\pi e}+M\varphi(0).
\]
The midpoint error is at most \(16h^2\|Q''\|_\infty/24\), \(h=16/8192\); the tail error is at most \(2M\Phi(-8)\), and threshold interpolation contributes at most \(L\Delta x/2\). An additional \(10^{-8}\) floating-point guard is included. This is not formal interval certification of all rounding errors.

With \(n=64,t_-=50,t_+=210\), Theorem 7 gives failure terms approximately
\[
1.7569\times10^{-7},\quad2.9591\times10^{-6},\quad.00236258.
\]
Conditional on the stated drift enclosure, the coverage lower bound for
\(50<T_{.005}\le210\) with right exit is .99763428.

A separate 2000-path simulation with seed 20261024 gives minimum 80, maximum 112, median 94, and 2.5%/97.5% quantiles 85/104, with no observed left exits. This checks typical behavior; absence of rare failures in a simulation does not prove the probability bound.

### 10.4. Critical SGD: finite geometry and direction

Use \(m=\sigma=1,w=2,\eta=.1,\delta=0\), and set
\[
\rho=(\lambda s_\eta/\beta)\kappa/\sqrt n.
\]
Each fixed-tolerance cell uses 600 actual SGD paths, seed 20261026, and a 2500-step cap. All paths exit before the cap. The model gives
\[
s_0^2=.0935303760,\quad s_\eta=.2374852733,\quad
\lambda s_\eta/\beta=.1068646385.
\]
For \(\epsilon=.005\), \(c_\epsilon=.2039969278\) and
\(C_\epsilon=.2038878864\).

The balanced-limit prediction uses Corollary 10. The finite-geometry prediction uses the true \(q_\rho,A_\rho,\mu_n,s_n,B_n^\pm\) in Theorem 9. Neither prediction fits a time offset or empirical quantile.

| \(n\) | \(\kappa\) | Observed median | Balanced median | Finite-geometry median | Balanced CDF error | Finite-geometry CDF error |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0 | 233 | 241 | 241 | .0450 | .0450 |
| 32 | 1 | 209 | 187 | 210 | .1482 | .0254 |
| 32 | 2 | 162 | 108 | 167 | .4026 | .0445 |
| 128 | 0 | 322.5 | 325 | 325 | .0295 | .0295 |
| 128 | 1 | 298.5 | 271 | 284 | .1015 | .0595 |
| 128 | 2 | 224 | 193 | 225 | .2478 | .0155 |
| 512 | 0 | 412 | 409 | 409 | .0216 | .0216 |
| 512 | 1 | 361.5 | 355 | 362 | .0600 | .0238 |
| 512 | 2 | 293 | 277 | 294 | .1235 | .0212 |

CDF errors are empirical maximum discrepancies on an integer grid and include Monte Carlo noise. A single-cell 95% DKW band for 600 paths is approximately .0554; it is neither simultaneous across all cells nor a theorem-approximation guarantee. The corrected \(n=128,\kappa=1\) error remains above that single-cell value.

| \(\kappa\) | Predicted right-exit probability | Observed, \(n=32\) | \(n=128\) | \(n=512\) |
|---:|---:|---:|---:|---:|
| 0 | .5000 | .5150 | .4867 | .5117 |
| 1 | .8413 | .8533 | .8383 | .8483 |
| 2 | .9772 | .9750 | .9817 | .9750 |

Accurate direction prediction does not ensure accurate time prediction. Both exit directions are unsafe relative to the chosen comparator.

### 10.5. Microscopic tolerance and checks with limited benefit

With \(\rho=.2/\sqrt n\) and \(\epsilon=.005/n\), 2000 actual paths per batch size are compared with 150000 separately simulated AR reference paths.

| \(n\) | Actual median | AR reference median | Empirical CDF difference |
|---:|---:|---:|---:|
| 32 | 45 | 45 | .0127 |
| 128 | 47 | 45 | .0243 |
| 512 | 45 | 45 | .0228 |

The actual risk endpoints are solved from the original risk, not substituted from its quadratic limit. Single-sample-set 95% DKW bands are about .0304 for the actual paths and .00351 for the AR reference. The nearly unchanged medians contrast with the logarithmic fixed-tolerance shift.

For a wider interval, \(\epsilon=.15,n=128,\kappa=1\), 1000 actual paths have median 499. Replacing the balanced linear boundary with the conjugacy boundary changes the CDF discrepancy only from .02903 to .02893; both predicted medians are 493. This does not support a claim that the conjugacy boundary alone produces a substantial numerical improvement.

At \(n=512\), 12000 independent first batches per setting give improvement probabilities .5497, .7196, and .8838 for \(c=.2,1,2\), compared with .5478, .7258, and .8849 from Proposition 12. At \(c=0\), improvement is exactly impossible. These probability comparisons are distinct from bounds on the magnitude of the first-step risk change.

### 10.6. Numerical scope

The geometry and critical experiments record Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0. Independent adaptive-integration and finite-difference diagnostics support quadrature and gradient consistency. Conjugacy endpoint values change by at most about \(3.71\times10^{-12}\) when the inverse-iteration cutoff is tightened from \(10^{-3}\) to \(10^{-4}\); checked functional-equation residuals are at most about \(5.55\times10^{-17}\). These are floating-point diagnostics, not certified probability constants.

The experiments establish controlled-model checks. They do not measure visual benchmark performance, verify unknown-anchor robustness, or demonstrate a practical unlabeled stopping procedure.

## 11. What would make a stopping rule observable?

The risk boundaries used above depend on the target parameters and serve as analytical objects. They do not themselves define an implementable unlabeled rule.

A generic certificate clarifies the additional requirement. Let
\(\vartheta=(\delta,p)\), with known anchor parameters, and suppose label-free, nested confidence sets \(\mathcal C_t\) satisfy
\[
\Pr(\vartheta\in\mathcal C_t\text{ for all }t)\ge1-\alpha.
\]
For a proposed threshold \(b'\), accept only if
\[
\sup_{\vartheta\in\mathcal C_t}
\{R_\vartheta(b')-R_\vartheta(b_0)\}\le\epsilon.
\]
On the coverage event every accepted iterate is safe, including data-dependent proposals. This follows directly from the supremum and is a generic baseline.

To become a useful algorithm, this construction needs a sequential confidence-set method under the stated observation model, accounting for repeated inspection and proposal dependence. It also needs a nontrivial acceptance and improvement guarantee. Ordinary fixed-time intervals repeatedly reused do not suffice. Anchor error, unknown component scales, orientation mistakes, and model misspecification require explicit treatment.

Appropriate empirical comparators include frozen source, original entropy updates, logit-adjusted entropy updates, mean recentering, an oracle risk threshold, and one-time unlabeled calibration with a frozen plug-in estimate. Known anchors supply more information than strict weights-only source-free deployment. No universal claim about all unlabeled adaptation is implied.

## 12. Limitations and falsifiable scope

The main unresolved scientific questions are the exact separation phase boundary, numerical constants for the critical finite-batch approximation, robustness to unknown anchors, and whether the predicted regimes persist for useful visual features. None is resolved by the synthetic examples alone.

A frozen-encoder validation would fit a source direction once, keep its slope fixed, manipulate projected translation and class prior separately and jointly, and use target labels only for evaluation. It should test complete trajectories, risk crossing, classwise errors, and seed variation, while retaining the official dataset splits. Image-space corruptions should not be claimed to satisfy a Gaussian projection model without checking that approximation.

The model omits nonstationarity, batch reuse, temporal dependence, learning-rate schedules, momentum, joint feature learning, and multilayer normalization dynamics. A safe population basin is not permanent finite-batch safety. A high right-exit probability is not a safety probability. A fixed-tolerance logarithmic horizon is not an improvement guarantee, and a longer horizon under larger batch size is not universal at fixed prior.

The contribution would be insufficient if its statements followed immediately from accepted entropy-drift results plus a generic unstable-exit theorem without a meaningful new risk prediction. The standard mechanisms are acknowledged explicitly. The case for a contribution must instead rest on the actual recursion, risk-defined boundary, finite-parameter predictions, and tolerance distinction, followed by a direct comparison with existing results.

## References

[1] T. Weerasinghe, R. Tennakoon, W. Chuah, and A. Bab-Hadiashar. *Logit-Adjusted Test-Time Adaptation under Partial Class Imbalance*. WACV, 2026. [Official paper](https://openaccess.thecvf.com/content/WACV2026/papers/Weerasinghe_Logit-Adjusted_Test-Time_Adaptation_under_Partial_Class_Imbalance_WACV_2026_paper.pdf).

[2] Hoang, Vo, and Do. *Persistent Test-time Adaptation in Recurring Testing Scenarios*. arXiv:2311.18193, v4, 2024. [Paper](https://arxiv.org/abs/2311.18193).

[3] Sahoo et al. *A Layer Selection Approach to Test Time Adaptation*. arXiv:2404.03784, v2, 2025. Earlier version: *Layerwise Early Stopping for Test Time Adaptation*. [Paper](https://arxiv.org/abs/2404.03784).

[4] Shin et al. *STAG*. arXiv:2402.09004, v2, 2026. [Paper identifier](https://arxiv.org/abs/2402.09004).

[5] Zhou et al. *On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective*. arXiv:2605.28057, v2, 2026. [Paper](https://arxiv.org/abs/2605.28057).

[6] Jhawar and Wang. *When is Test-Time Adaptation Identifiable From Unlabeled Evidence?* arXiv:2609.11235, v1, 2026. [Paper](https://arxiv.org/abs/2609.11235).

[7] A. Saumard and J. A. Wellner. *Log-concavity and strong log-concavity: a review*. Statistics Surveys 8, 2014. [Paper](https://arxiv.org/abs/1404.5886).

[8] Y. Kifer. *The exit problem for small random perturbations of dynamical systems with a hyperbolic fixed point*. Israel Journal of Mathematics 40:74–96, 1981. [Publication record](https://cris.huji.ac.il/en/publications/the-exit-problem-for-small-random-perturbations-of-dynamical-syst/).

[9] Y. Bakhtin. *Exit asymptotics for small diffusion about an unstable equilibrium*. arXiv:math/0701569, 2007. DOI: 10.1016/j.spa.2007.06.003. [Paper](https://arxiv.org/abs/math/0701569).

[10] Y. Bakhtin and A. Gaál. *Exit time asymptotics for dynamical systems with fast random switching near an unstable equilibrium*. arXiv:1901.05513, 2019. [Paper](https://arxiv.org/abs/1901.05513).

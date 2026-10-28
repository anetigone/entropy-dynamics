# When Does Entropy Adaptation Stop Helping?

## Risk Rebound and Safe Horizons under Mixed Feature and Prior Shift

Working manuscript in multiple parts. This file gives the main narrative and Gaussian-model result statements. Read it with [general scalar exit theory](scalar_exit.md), [independent calibration theory](calibration.md), [risk boundaries and empirical pools](risk_boundaries.md), [Gaussian proofs](gaussian_proofs.md), and [numerical evidence](experiments.md). The analysis concerns a scalar classifier with fixed features and fixed confidence slope. It does not establish a general guarantee for neural-network adaptation or a completed unlabeled stopping algorithm.

## Abstract

Entropy minimization can initially improve a deployed classifier and later make it worse than its frozen initialization. We quantify this behavior for actual entropy-gradient updates of a decision threshold in a Gaussian mixture under simultaneous feature translation and class-prior shift. In the weak-separation regime, the marginal distribution implies a unique unstable entropy maximum and an explicit open parameter region of risk rebound. Sufficiently separated mixtures instead admit locally safe population convergence. Near class balance, fixed positive risk tolerance yields a discrete logarithmic population crossing time, whereas tolerance comparable to the available risk improvement yields bounded crossing times. For fresh-batch stochastic gradient descent, a direct drift argument bounds first risk exit outside the critical region. Near the unstable entropy maximum, a deterministic conjugacy controls the nonlinear, state-dependent updates and yields a joint approximation for exit direction and crossing time. The resulting law retains the discrete-time lattice phase. When tolerance shrinks as the inverse batch size, the limit changes to finite-step exit of a Gaussian autoregression. A mixed-shift scan supports cancellation and direction reversal. Frozen visual features expose a calibration gap: source-calibrated predictions fail overall, whereas accurate target empirical geometry substantially improves exit prediction. A general scalar extension under explicit exit-domain conditions gives uniform integer-time approximation and separates peak, growth-rate, noise, and boundary errors under independent calibration. Its critical limit predicts persistent random forecast error when root uncertainty matches SGD noise. A small independent unlabeled field-calibration study reduces the empirical gap as its budget grows, but unlabeled target risk-boundary estimation remains unresolved. These visual results concern a fixed empirical distribution, not a general neural-network guarantee or an unlabeled stopping algorithm. Classical instability mechanisms and known class-imbalance failures are treated as prior knowledge; the candidate contribution is their precise connection to actual entropy updates and risk-defined safety.

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

The empirical investigation separates three questions: whether mixed shifts produce the predicted cancellation, whether correct empirical geometry explains exits on frozen visual features, and whether that geometry can be calibrated independently. The first two receive positive evidence, with residual failures; the third remains incomplete. In particular, target-geometry diagnostics use target labels for risk boundaries and cannot establish an executable unlabeled prediction rule.

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
The batch size is \(n\). Every batch is independent of the past, and all samples have the same target distribution. Labels are not read by the update. Dependent reuse of observations, changing priors, momentum, and feature learning are outside this recursion. The visual experiments below instead draw independently with replacement from a fixed feature pool: conditional on that pool, the samples are iid from its empirical distribution. This preserves sampling independence but does not make that non-Gaussian empirical distribution satisfy the Gaussian model assumptions.

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

**Proof.** See [Gaussian proofs](gaussian_proofs.md), under the same result number.

### Lemma 2. The entropy kernel is log-concave

The function \(h\) is positive, even, integrable, and log-concave. If
\[
K(t)=\mathbb E h(w(t+\sigma\xi)),
\]
then \(K\) is positive and even, and \(K'(t)<0\) for \(t>0\). Consequently,
\[
F_p(x)=pK(x-m)+(1-p)K(x+m).
\]

**Proof.** See [Gaussian proofs](gaussian_proofs.md), under the same result number.

We use the standard convolution closure property: if two nonnegative integrable functions are log-concave, their convolution is log-concave [7]. The model-specific task is verifying the kernel and the mixture density, rather than claiming this closure result as a contribution.

## 5. Distribution structure determines population behavior

### Theorem 3. Weak separation gives a unique unstable entropy peak

Assume \(0<m\le\sigma\). There is a unique stationary point \(q_p\) of \(F_p\), and it is a strict global maximum. Its drift signs are
\[
v_p(x)<0\quad(x<q_p),\qquad v_p(x)>0\quad(x>q_p).
\]
The sign of \(q_p\) is the sign of \(p-1/2\). For every \(\eta>0\), population iterates initialized right of \(q_p\) increase to \(+\infty\); iterates initialized left of it decrease to \(-\infty\). Initialization at \(q_p\) stays fixed.

**Proof.** See [Gaussian proofs](gaussian_proofs.md), under the same result number.

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

**Proof.** See [Gaussian proofs](gaussian_proofs.md), under the same result number.

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

**Proof.** See [Gaussian proofs](gaussian_proofs.md), under the same result number.

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

**Proof.** See [Gaussian proofs](gaussian_proofs.md), under the same result number.

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

**Proof.** See [Gaussian proofs](gaussian_proofs.md), under the same result number.

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

**Proof.** See [Gaussian proofs](gaussian_proofs.md), under the same result number.

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

**Proof.** See [Gaussian proofs](gaussian_proofs.md), under the same result number.

## 8A. General scalar laws and estimated geometry

The Gaussian model supplies explicit distributional geometry. The extension in [general scalar exit theory](scalar_exit.md) instead fixes any score law \(Q\), including a finite weighted pool, and assumes one repelling root on the required compact domain, positive root noise variance, fixed positive step size, two finite interior exit boundaries with an overshoot margin, and \(O(n^{-1/2})\) initialization relative to the true root (G1–G4). These assumptions are not inferred from local positive curvature alone. Lemmas E.1–E.2 and Theorem E.3 give the actual entropy-SGD joint interval-exit approximation, including
\[
\sup_{t\in\mathbb Z_{\ge0}}\max_\pm
|\Pr(\tau_n\le t,J_n=\pm1)-\Psi_n^\pm(t)|\le C e_n,
\qquad e_n=n^{-1/2}+((1+\log n)/n)^{1/3}.
\]
The early-time stopped-martingale tail argument is essential for this uniform statement. The original Theorem 9 retains its Gaussian-family formulation and proof; the extension does not automatically imply uniformity over changing score laws without common geometric constants.

[Independent calibration theory](calibration.md) treats a separate iid sample of size \(M\) from the same \(Q\). It changes the predictor, not the actual initialization or adaptation recursion. Lemma E.4 controls normalized conjugacy under small \(C^3\) field perturbations; Proposition E.5 gives empirical field concentration and the root CLT. Theorem E.6 yields, on the declared good-calibration and endpoint event,
\[
\sup_t\max_\pm |\Pr(\tau_n\le t,J_n=\pm1)-\widehat\Psi_n^\pm(t)|
\le C[e_n+\sqrt n|d_q|+\sqrt n d_q^2
 +(1+\log n)d_v+d_\sigma+d_b],
\]
where \(d_q=\widehat q-q\), \(d_v=\|\widehat v-v\|_{C^3}\), \(d_\sigma=|\widehat\sigma_g^2-\sigma_g^2|\), and \(d_b\) is endpoint error. Thus peak error acts on the shrinking SGD seed scale, growth-rate error accumulates over logarithmic time, and endpoint error remains a separate input. With fixed \(Q,w,\eta\), consistent endpoints and \(M/n\to\infty\) suffice for consistency of this empirical-root predictor; no universal sample-complexity lower bound is asserted.

For \(s^2=\eta^2\sigma_g^2/((1+\eta\lambda)^2-1)\), define
\[
\mathcal R_{M,n}^2=\frac nM\frac{2+\eta\lambda}{\eta\lambda}.
\]
If \(\sqrt n(x_{0,n}-q)/s\to z\) and \(\mathcal R_{M,n}\to r<\infty\), Theorem E.7 gives
\[
\Pr(J_n=+1)\to\Phi(z),\qquad
\widehat\pi_{M,n}\Rightarrow\Phi(z-rZ),\quad Z\sim N(0,1).
\]
The randomness on the right is over calibration samples. At exact critical centering, Corollary E.8 gives limiting mean absolute direction-probability error \(\arctan(r)/\pi\), equal to \(1/4\) when \(r=1\). Joint time limits retain an integer-phase subsequence. This random forecast limit does not inflate the variance of the unchanged SGD process. The distinct calibration-driven initialization result is Proposition E.13 in [the boundary and pool part](risk_boundaries.md).

[Risk boundaries and empirical pools](risk_boundaries.md) supplies the remaining interfaces. Proposition E.9 distinguishes a fixed off-peak initialization, whose strict-margin exit converges to a finite deterministic count, from a genuinely critical parameter sequence. Proposition E.10 controls regular differentiable endpoints by risk-difference error divided by a positive transversality constant; it does not apply directly to step-function empirical risk. Proposition E.11 adds a component-skip probability when transferring interval exit to actual risk exit. Proposition E.12 separates conditional-pool inference from population-risk uncertainty. None estimates label-dependent risk endpoints from an arbitrary unlabeled score marginal.

These are supplied statements with proofs, integrated here without claiming a completed independent proof audit or numerical calibration of their constants. The visual experiments must still verify interval-wide conditions, initialization regime, endpoint conventions, and component-skip discrepancies before any setting is declared covered. The extension does not generalize the Gaussian inverse-batch-tolerance result to arbitrary score laws.

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

**Proof.** See [Gaussian proofs](gaussian_proofs.md), under the same result number.

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

**Proof.** See [Gaussian proofs](gaussian_proofs.md), under the same result number.

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

**Proof.** See [Gaussian proofs](gaussian_proofs.md), under the same result number.

The limitation concerns unconditional execution of the empirical entropy direction. It does not exclude rejecting an update, using extra information, or a different unlabeled adaptation method. The probability of a strict risk increase also says nothing by itself about the size of that increase.

## 10. Numerical evidence

The experimental protocols, tables, ablations, censoring conventions, and numerical limitations are in [the experimental part](experiments.md). Sections 10.6–10.9 distinguish mixed-shift evidence, failed source-calibrated visual prediction, target empirical geometry diagnostics, and independent unlabeled field calibration.

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

The integrated extension supplies proofs for independent-geometry prediction error and regular boundary perturbations under explicit assumptions. The immediate unresolved questions are independent verification of those arguments, alignment of their assumptions and events with each visual setting, unlabeled identification of risk boundaries, and the cause of long-time oracle tail errors. The calibration pilot has only four repetitions per setting and covers balanced priors and one initialization offset. Larger calibration repetition counts and nonbalanced, nonzero-offset checks are needed before claiming broad independent prediction. The exact separation phase boundary, numerical constants for the critical approximation, and unknown-anchor robustness also remain open.

The completed frozen-encoder study retains official train/test splits and controls projected translation and priors. It supplies useful conditional-oracle evidence, but the original independently calibrated prediction fails overall and the improved field-calibration study still uses labeled oracle risk endpoints for its strongest results. Neither full BN adaptation nor scale/LA comparisons have yet been validated by the available results. Image-space corruptions should not be claimed to satisfy a Gaussian projection model without checking that approximation. The general scalar theorem now states such geometric conditions; their validity on each visual exit domain remains to be checked.

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

# How Much Unlabeled Information Is Needed to Predict Entropy-SGD Risk Exit?

This manuscript studies prospective probability prediction under local scalar entropy dynamics. The [proof chapters](README.md) retain complete arguments and original theorem labels. Supporting Gaussian-mixture geometry and exit-time results are in [gaussian_model.md](gaussian_model.md); they are distinct from the shared-risk family used for the statistical results below.

## Abstract

Predicting the consequences of entropy adaptation requires knowledge of target-domain dynamics, even when classification risk is known. We study this distinction for a scalar threshold following actual fresh-batch entropy-gradient updates. Target environments share initialization and their complete classification-risk function, while unlabeled marginals move an unstable entropy root on the batch-noise scale. A uniform exit-direction approximation connects this future trajectory to an independent calibration experiment. Around a known baseline, uniformly consistent prediction is possible exactly when calibration size M grows faster than future batch size n. At positive finite \(M/n\), the minimax risk converges to a bounded-parameter Gaussian decision value, for which we compute lower and continuous-window upper certificates. The same value holds over unknown uniformly bounded local directions: the entropy-gradient tilt is already least favorable for root displacement relative to Fisher information. A frozen prediction rule improves five-cell maximum MAE over raw and clipped moment rules in real entropy-SGD confirmation, with simultaneous intervals accounting for calibration and reference-trajectory uncertainty. Independent confirmation on frozen CIFAR-10 features reduces five-point maximum MAE relative to clipping in all four class pairs by point estimate, with positive simultaneous improvement intervals in three; this visual experiment is conditional on known empirical baselines and has environment-specific risk. We further allow a baseline estimated from an independent sample: conditioning on the realized initialization yields a random-target Gaussian minimax problem, while the calibration consistency threshold remains M/n→∞ when L/n has a positive limit. A root-preserving subcase admits a closed-form pooling rule. A full-network BN/Tent applicability check separates stable mean motion from poorly transferring stochastic residual axes, limiting direct extrapolation of the scalar mechanism. These results concern regular local scalar dynamics at fixed step size and do not establish general neural-network safety.

## 1. Introduction

Knowing which classifier states are unsafe does not determine how likely stochastic adaptation is to reach them. A complete risk function specifies the first object; an unlabeled score distribution determines the entropy dynamics. We isolate the second uncertainty by supplying identical complete risk information across environments and asking for an independent forecast before adaptation begins.

Entropy deterioration itself is established prior knowledge. POEM includes a one-dimensional Gaussian example of entropy collapse [1]; logit-adjusted TTA analyzes degradation under class imbalance [2]. Risk monitoring observes an evolving adapted model and test stream [3], while action-identifiability work studies what unlabeled evidence determines about competing actions [4]. Here the action is fixed, risk is shared, and the target is a probability over a future independent stochastic path. The predictor is not asked to guess that path's realized branch.

A finite-baseline extension conditions on the sample-selected initialization and derives a new random-target minimax problem without requiring a superlinear baseline budget. Our contribution is the connection between this information experiment and a uniform approximation of actual entropy-SGD risk exit. It yields an exact consistency threshold, a finite-information minimax characterization, and a least-favorable direction that remains sufficient to describe worst-case difficulty in a larger bounded local class. Classical unstable-exit, experiment-comparison and efficiency tools are explicitly credited. The numerical contribution is a pair of valid decision bounds and a fixed-rule finite-batch comparison, rather than a claim that an optimizer found an exact least-favorable prior.

The paper proceeds from the observation experiment and common-risk construction to the exit approximation, statistical limits, unknown directions, numerical certificates and independent confirmation. Complete proofs remain in the linked chapters. Independent frozen-feature confirmation tests the prediction interface on unknown local visual environments. Earlier rebound, tolerance, source-transfer and stopping investigations remain supporting evidence.

The full-network BN/Tent check complements these controlled validations by testing the conditions needed for transfer: a reproducible stochastic projection, local unstable dynamics, and a link to classification-risk events. Stable mean motion is observed, but residual axes transfer poorly across runs and no risk-increase event occurs within the tested horizon. This is evidence about applicability, not a neural-network consequence of the scalar theorem.

## 2. Information, dynamics and prediction loss

Fix confidence slope \(w>0\) and positive step size \(\eta\). For \(s(u)=(1+e^{-u})^{-1}\) and \(a(u)=s(u)(1-s(u))\), define
\[
U(x,z)=-w^2(z-x)a(w(z-x)),\qquad v_Q(x)=E_Q U(x,Z).
\]
Actual fresh-batch adaptation starts at \(x_0=0\) and uses
\[
x_{j+1}=x_j+\frac\eta n\sum_{i=1}^n U(x_j,Z_{j,i}).
\]
All future scores are iid from the unknown score law Q. Labels do not enter this update. Uniformly in x,z, \(|U|\le B_U=w/4\) and \(|\partial_x U|\le L_U=w^2/4\); every fixed higher state derivative is bounded.

Let R_Q be classification risk and \(\epsilon>0\). Define
\[
T_\epsilon=\inf\{j\ge0:R_Q(x_j)>R_Q(0)+\epsilon\}.
\]
Our construction makes the entire risk-safe set exactly \([-b,b]\), and exit is almost surely finite. The prediction target is
\[
\pi_n(Q)=P_Q(x_{T_\epsilon}>b).
\]

The predictor receives M iid calibration scores from Q, independent of all future SGD samples. It knows the declared baseline family, initialization, fixed constants, and may receive the entire common risk function and identical source information. It does not receive the unknown environment, root, local seed displacement, future trajectory or exact target marginal. The risk of a forecast is expected absolute probability error over calibration randomness; \(\pi_n\) itself averages over future SGD.

For a local parameter u in \([-A_0,A_0]\), define
\[
\mathfrak R_{n,M}(A_0)=\inf_{\widehat\pi}\sup_{|u|\le A_0}
 E_{u/\sqrt n}|\widehat\pi-\pi_n(Q_{u/\sqrt n})|.
\]
The fixed width \(A_0>0\) is part of the declared problem. Known exact centering is a different experiment: the symmetric center admits the exact forecast .5. The root-calibration arctangent error of a particular plug-in is consequently not a universal lower bound.

## 3. A common-risk environment family

Let \(f_0\) be the centered Gaussian density with variance \(\nu^2>0\). Define
\[
\lambda=v_0'(0)=\frac{w^2}{\nu^2}E_0[Z^2a(wZ)]>0,
\quad \sigma_g^2=E_0U(0,Z)^2>0,
\quad h_0=U(0,\cdot)/\sigma_g.
\]
Symmetry gives \(E_0h_0=0\), \(E_0h_0^2=1\), and \(h_0\) is bounded. Choose \(0<\kappa<1\) and
\[
f_\theta=f_0(1+\theta h_0),\quad
 g(z)=\kappa\tanh(z)f_0(z),\quad
 p_\theta(z,\pm1)=\tfrac12(f_\theta(z)\pm g(z)).
\]
When \(|\theta|\|h_0\|_\infty\le\min(1/2,1-\kappa)\), these are legal joint densities with equal class priors. For the classifier predicting +1 when \(z>x\), every environment has the same complete risk
\[
R(x)=\tfrac12+\int_{-\infty}^x g(z)\,dz,
\quad R(x)-R(0)=\kappa\int_0^x\tanh(z)f_0(z)\,dz.
\]
This function is even and strictly increases with \(|x|\). Thus for any chosen b>0 and \(\epsilon=R(b)-R(0)\), its global safe set is exactly \([-b,b]\). Interval exit and actual risk exit coincide path by path, even for discrete overshoots. The labeled conditionals of this family are not assumed Gaussian mixtures.

To establish uniform exit geometry, choose small d>0 with \(v_0'\ge\lambda/2\) on \([-d,d]\), set b=d/2, and fix
\[
0<\eta<\min\{1/L_U,d/(4B_U)\}.
\]
For all sufficiently small |\(\theta\)|, \(v_\theta'\ge\lambda/4\) on this compact domain and there is a unique root with \(|q_\theta|\le b/2\). The overshoot interval stays strictly inside \([-d,d]\), the root noise variance is bounded below, and
\[
q_\theta=-\frac{\sigma_g}{\lambda}\theta+O(\theta^2).
\]
These are Proposition L.1 and its proof in [local_prediction.md](local_prediction.md). Complete risk knowledge therefore does not remove uncertainty about the location of the entropy repeller.

## 4. Uniform direction exit of actual SGD

Write \(\lambda_\theta=v_\theta'(q_\theta)\), \(A_\theta=1+\eta\lambda_\theta\), and
\[
s_\theta^2=\frac{\eta^2\operatorname{Var}_\theta U(q_\theta,Z)}{A_\theta^2-1}.
\]
At the baseline put \(s=s_0\), \(A=1+\eta\lambda\) and
\[
c=\frac{\sigma_g}{\lambda s}=\sqrt{\frac{2+\eta\lambda}{\eta\lambda}},
\quad e_n=n^{-1/2}+((1+\log n)/n)^{1/3}.
\]

**Theorem 1 (Uniform actual-SGD direction law; L.2).** For every fixed finite \(A_0\), a constant C independent of \(n,u\) satisfies
\[
\sup_{|u|\le A_0}|\pi_n(Q_{u/\sqrt n})-\Phi(cu)|\le Ce_n
\]
for all sufficiently large n. The step size and tolerance are fixed; C is not numerically calibrated.

The proof constructs increasing normalized \(C^2\) conjugacies \(H_\theta\) with \(H_\theta(q_\theta)=0\), \(H_\theta'(q_\theta)=1\) and \(H_\theta(G_\theta(x))=A_\theta H_\theta(x)\). Uniform inverse contraction gives common derivative bounds, including a positive lower bound. The latter follows from a convergent product of positive derivatives, not normalization alone.

At interval exit \(\tau\) define the stopped coordinate \(Y_\tau=\sqrt n A_\theta^{-\tau}H_\theta(x_\tau)\). Compare it to the frozen-root weighted seed
\[
W=\sqrt n H_\theta(0)+\eta\sum_{j\ge0}A_\theta^{-j-1}\sqrt n\,\xi_j(q_\theta),
\quad \xi_j(x)=n^{-1}\sum_i U(x,Z_{j,i})-v_\theta(x).
\]
The stopped martingale and Taylor expansion give uniformly
\[
E A_\theta^{-2\tau}\le C/n,\qquad
E|Y_\tau-W|^2\le C(1+\log n)/n.
\]
The first bound is obtained from a stopped second moment without assuming a typical exit-time estimate. Batches after \(\tau\) are fresh conditional on the stopping history, which bounds the frozen tail. Grouping W by sample column gives a uniform Gaussian approximation. Anti-concentration handles seeds close to zero before equating their signs to exit directions. Finally \(\sqrt n H_\theta(0)/s_\theta=cu+O(n^{-1/2})\). The following subsections give the proof, retaining the source labels from L.2; the broader time-distribution theory remains in [scalar_exit.md](scalar_exit.md).

### 4.1. Proof of the uniform direction law

Within this proof, \(K=[-d,d]\) denotes the compact state interval. We supply the common constants and the stopped-seed argument, so this is not an application of a pointwise exit limit to changing laws.

Construct normalized \(C^2\) conjugacies \(H_\theta\) with \(H_\theta\)(\(q_\theta\))=0, \(H_\theta\)'(\(q_\theta\))=1 and \(H_\theta\)(\(G_\theta\)(x))=\(A_\theta\) \(H_\theta\)(x) whenever both arguments are in K. The uniform inverse construction is as follows.

Let K=[-d,d] and \(G_\theta\)(x)=x+\(\eta\) \(v_\theta\)(x). The signs of \(v_\theta\) at the endpoints imply \(G_\theta\)(K) contains K. Thus \(F_\theta\)=\(G_\theta\)^{-1} is defined on K, maps K into itself, and fixes \(q_\theta\). Its derivative satisfies

\[
0<F_\theta'(x)\le \gamma:=(1+\eta\lambda/4)^{-1}<1.
\]

In centered coordinates write

\[
\widetilde F_\theta(y)=A_\theta^{-1}y+r_\theta(y),
\qquad r_\theta(0)=r_\theta'(0)=0.
\]

On a common small neighborhood, uniform smoothness gives bounded third derivatives and a bound \(\beta\) on the inverse derivative, with \(A_{\max}\) \(\beta\)^2<1. If y_j=F_tilde_theta^j(y), then, uniformly in \(\theta\),

\[
|y_j|\le C\beta^j|y|,\qquad
|y_j'|\le C\beta^j,\qquad |y_j''|\le C\beta^j.
\]

For the second inequality use the derivative product. The third follows from

\[
y_{j+1}''=\widetilde F_\theta''(y_j)(y_j')^2
             +\widetilde F_\theta'(y_j)y_j''.
\]

The successive differences in \(A_\theta\)^j y_j equal
\(A_\theta\)^{j+1} r_theta(y_j). Their functions, first derivatives and second derivatives have summable bounds C(\(A_{\max}\) \(\beta\)^2)^j. This proves local \(C^2\) convergence. To establish positivity of the limiting derivative, use its derivative product:

\[
\prod_{k=0}^{j-1} A_\theta\widetilde F_\theta'(y_k).
\]

Each factor is positive and differs from one by O(|y_k|). On a sufficiently small common neighborhood the sum of these deviations is uniformly bounded, so the limiting product is bounded away from zero. A fixed number of inverse iterations maps all of K into that neighborhood, since \(F_\theta\) contracts by \(\gamma\). Finite composition gives uniform \(C^2\) bounds and a positive derivative lower bound on K. This also verifies the functional equation wherever its two arguments are in K.

Thus, for constants independent of \(\theta\),
\[
\|H_\theta\|_{C^2(K)}\le C,\quad \inf_K H_\theta'\ge m>0,
\quad 1<A_{\min}\le A_\theta\le A_{\max},\quad
0<s_{\min}\le s_\theta\le s_{\max}.
\tag{L.3}
\]
The inverse derivative bounds use the uniform smoothness of U and the positive derivative of \(G_\theta\). The root-to-boundary distances are at least b/2. These facts also give uniform bounds above and below for the absolute conjugate boundary values.

Let \(\tau\) be exit from [-b,b] and \(\xi_j\)(x)=n^{-1}sum_i U(x,Z_{j,i})-\(v_\theta\)(x). Each \(\theta\) has finite \(\tau\) almost surely: choose two fixed bounded score intervals respectively left and right of K. Their masses are uniformly positive since f_theta>=\(f_0\)/2. Repeated batches entirely in the left interval drive any state at least \(q_\theta\) rightward by a positive minimum speed until exit; the right interval does the converse. Compactness supplies a fixed finite block length N. The conditional chance of an appropriate exiting block is at least p^{nN}>0; geometric blocking proves finiteness. This is not a bound on the typical exit time.

For |\(\theta\)|<=\(A_0\)/\(\sqrt n\), (L.1) implies |\(q_\theta\)|<=C/\(\sqrt n\) and \(\mu\)=sqrt(n)\(H_\theta\)(0) is bounded. Conditional batch moments give
\[
E[\xi_j(x)^2\mid\mathcal F_j]\le C/n,\quad
E[\xi_j(x)^4\mid\mathcal F_j]\le C/n^2,\quad
E[(\xi_j(x)-\xi_j(q_\theta))^2\mid\mathcal F_j]
\le C|x-q_\theta|^2/n.
\]
All constants are distribution independent, by bounded U and partial_x U. Taylor expansion on the established overshoot domain gives, for j<\(\tau\),
\[
H_\theta(x_{j+1})=A_\theta H_\theta(x_j)
 +\eta H_\theta'(G_\theta(x_j))\xi_j(x_j)+r_j,
\quad |r_j|\le C\xi_j(x_j)^2.
\]
Set \(Y_t\)=\(\sqrt n\) \(A_\theta\)^{-(t wedge \(\tau\))}\(H_\theta\)(x_{t wedge \(\tau\)}).

Let F_j denote the history before batch j and write I_j=1_{j<\(\tau\)}. With

\[
D_j=\sqrt n\,\eta A_\theta^{-j-1} I_j
 H_\theta'(G_\theta(x_j))\xi_j(x_j),
\]

and

\[
E_j=\sqrt n A_\theta^{-j-1}I_j r_j,
\]

the stopped recursion gives

\[
Y_t=\mu+\sum_{j<t}D_j+\sum_{j<t}E_j.
\]

The indicator and the coefficient are F_j-measurable. Thus D_j are martingale differences and

\[
\sum_j\mathbb E D_j^2\le C\sum_j A_\theta^{-2j}\le C.
\]

The conditional fourth-moment bound gives

\[
\|E_j\|_2\le C n^{-1/2}A_\theta^{-j},
\qquad \left\|\sum_{j<t}E_j\right\|_2\le Cn^{-1/2}.
\]

Therefore sup_t E|\(Y_t\)|^2<=C without assuming a logarithmic exit-time estimate. Finite exit and Fatou then imply E|\(Y_\tau\)|^2<=C. At exit, the uniform root-to-boundary separation and positive derivative of H imply |\(H_\theta\)(x_tau)|>=\(b_H\)>0. Consequently

\[
\mathbb E A_\theta^{-2\tau}\le (nb_H^2)^{-1}\mathbb E|Y_\tau|^2\le C/n.
\]

There is no circular inference from a desired exit approximation to its own moment estimate.

Before exit, \(H_\theta\)'>=m and bounded K also imply
\[
E[\mathbf1_{j<\tau}|x_j-q_\theta|^2]\le C\min(A_\theta^{2j}/n,1).
\tag{L.4}
\]

On the same infinite sample array introduce the frozen seed
\[
W=\mu+\eta\sum_{j\ge0}A_\theta^{-j-1}\sqrt n\,\xi_j(q_\theta).
\]
The difference \(H_\theta\)'(\(G_\theta\)(\(x_j\)))\(\xi_j\)(\(x_j\))-\(\xi_j\)(\(q_\theta\)) has conditional mean zero and variance at most C|\(x_j\)-\(q_\theta\)|²/n, using \(H_\theta\)'(\(q_\theta\))=1 and (L.3). Consequently its stopped weighted sum has second moment at most
\[
C\sum_{j\ge0}A_\theta^{-2j}\min(A_\theta^{2j}/n,1)
\le C(1+\log n)/n.
\]
Use the convention that batch j moves \(x_j\) to x_{j+1}. At \(\tau\), all batches with indices j>=\(\tau\) remain fresh. Conditional on F_tau, the frozen tail has mean zero and variance

\[
\eta^2\sigma_\theta^2\sum_{j\ge\tau}A_\theta^{-2j-2}
=s_\theta^2 A_\theta^{-2\tau}.
\]

This proves the O(1/n) tail second moment in (L.5) below. It is a stopped-tail independence argument, not a claim that the stopped trajectory is independent of the full frozen seed.

Together,
\[
E|Y_\tau-W|^2\le C(1+\log n)/n.
\tag{L.5}
\]
Grouping W-\(\mu\) by sample column gives n^{-1/2}sum_i V_i with iid centered
V_i=\(\eta\) sum_{j>=0} \(A_\theta\)^{-j-1}U(\(q_\theta\),Z_{j,i}). These variables have uniform absolute bounds \(\eta\) B_U/(\(A_{\min}\)-1), and variance \(s_\theta\)² bounded below. Berry–Esseen gives a uniform C/\(\sqrt n\) Kolmogorov error relative to N(\(\mu\),\(s_\theta\)²). In particular P(|W|<=\(\delta\))<=C \(\delta\)+C/\(\sqrt n\).

The sign of \(Y_\tau\) is exactly the exit direction. Hence (L.5), Chebyshev, and the preceding anti-concentration bound give
\[
|\pi_n(Q_\theta)-\Phi(\mu/s_\theta)|
\le C\{n^{-1/2}+\delta+(1+\log n)/(n\delta^2)\}\le C e_n,
\]
choosing \(\delta\)=((1+log n)/n)^{1/3}. This explicitly controls seeds near zero, rather than identifying their signs without an error bound. Finally normalization and the uniform \(C^2\) bound imply \(H_\theta\)(0)=-\(q_\theta\)+O(\(q_\theta\)²); smoothness gives \(s_\theta\)=s+O(|\(\theta\)|). Uniformly on |u|<=\(A_0\),
\[
\mu/s_\theta=cu+O(n^{-1/2}).
\]
The Lipschitz property of Phi proves (L.2). □


## 5. Information threshold and finite-information precision

**Theorem 2 (Calibration threshold and low information; L.3–L.7).** Uniform consistency on any fixed positive local window is possible if and only if \(M/n\) tends to infinity. If eventually \(M/n\le C_0\) and
\[
0<a\le\min\{A_0,1/(4\sqrt{\max(C_0,1)})\},
\]
then
\[
\liminf_n\mathfrak R_{n,M}(A_0)\ge[2\Phi(ca)-1]/4>0.
\]
If \(M/n\) tends to zero, the exact limiting minimax MAE is \(\Phi(cA_0)-1/2\), attained by the constant forecast .5.

For necessity compare \(\theta=\pm a/\sqrt n\). Product KL is at most \(8a^2M/n\) and total variation at most \(2a\sqrt{M/n}\). Directly integrating absolute loss over the density overlap gives maximum risk at least half the target separation times \(1-\mathrm{TV}\). This avoids the factor of two lost by reducing to a nearest-target testing error. Sufficiency uses \(E_\theta h_0=\theta\) and \(\operatorname{Var}_\theta h_0=1-\theta^2\): the raw moment forecast
\[
\widehat\pi^{\rm mom}=\Phi(c\sqrt n\,\overline h_0)
\]
has uniform error at most \(c\sqrt{n/M}/\sqrt{2\pi}+Ce_n\).

**Theorem 3 (Finite-ratio minimax limit; L.9–L.10).** If \(M/n\) tends to \(\rho\) in (0,infinity), the entire local calibration experiment on the known tilt subfamily is asymptotically equivalent to \(Y=u+Z/\sqrt\rho\), uniformly for \(|u|\le A_0\). Consequently
\[
\mathfrak R_{n,M}(A_0)\longrightarrow V_G(K,r),\qquad
K=cA_0,\quad r=c/\sqrt\rho,
\]
where
\[
V_G(K,r)=\inf_d\sup_{|t|\le K}E|d(t+rZ)-\Phi(t)|.
\]

The [complete experiment-level proof](gaussian_decision.md) constructs both Markov kernels. Under the baseline, transform the standardized sample moment \(S_M\) by \(T_M=\Phi^{-1}(F_M(S_M))\), with its known continuous baseline CDF \(F_M\). \(T_M\) is exactly standard normal. A uniform log-likelihood expansion and bounded exponential moments give uniform L1 approximation by \(\exp(\sqrt\rho uT_M-\rho u^2/2)\). A baseline conditional-distribution kernel reconstructs the auxiliary full sample from \(T_M\). The two deficiencies tend to zero. Transporting bounded absolute loss and replacing \(\pi_n\) by \(\Phi(cu)\) proves the minimax limit; a scalar CLT alone would not establish it.

For comparison, L.8 proves that the raw moment rule has worst-window limiting risk \(\arctan(r)/\pi\). The constant rule has risk \(\Phi(K)-1/2\). Neither formula equals the general finite-ratio minimax value. Window projection improves the raw rule's loss against \(\Phi(t)\), but does not itself establish optimality.

### 5.1. Proof of the information threshold

Write \(r_n\) for the uniform error in Theorem 1, so \(r_n\)<=Ce_n. We retain source equation labels to connect each step to the complete proof chapters.

 At \(\theta\)=+/-a/\(\sqrt n\), positivity gives
\[
D_{KL}(f_\theta\|f_{-\theta})\le\chi^2(f_\theta,f_{-\theta})
=4\theta^2 E_0\frac{h_0^2}{1-\theta h_0}\le8\theta^2.
\]
Product KL is at most 8a²\(M/n\). Pinsker gives total variation at most 2a sqrt(\(M/n\))<=1/2. Lemma L.2 implies the target separation \(\Delta_n\) tends to Delta=2Phi(ca)-1>0. Let p_+,p_- be the calibration densities and t_+,t_- their true exit probabilities. For any forecast T, the triangle inequality on their common density overlap gives
\[
E_+|T-t_+|+E_-|T-t_-|
\ge\int\min(p_+,p_-)(|T-t_+|+|T-t_-|)
\ge\Delta_n(1-\operatorname{TV}).
\]
The maximum risk is at least half this sum, hence at least \(\Delta_n\)/4 when TV<=1/2. This proves (L.6). Common risk and fixed source information have identical values and cannot change this comparison. Independent randomization cannot improve the testing bound. □

The matching consistency argument uses the raw moment estimator already defined above. Its exact variance is at most 1/M, so the Lipschitz bound gives risk at most c sqrt(n/M)/sqrt(2pi)+Ce_n. If the budget ratio does not diverge, a bounded-ratio subsequence and the two-point lower bound exclude consistency.

### 5.2. Exact low-information minimax limit

 Put \(B_0=\Phi(cA_0)-1/2\). For the upper bound, Theorem 1 and monotonicity of \(\Phi\) give

\[
\sup_{|u|\le A_0}|\tfrac12-\pi_n(Q_{u/\sqrt n})|
\le B_0+r_n.
\]

For the lower bound take a=\(A_0\) in the density-overlap bound. The calibration TV bound tends to zero and the target separation tends to \(2B_0\), so liminf risk>=\(B_0\). These two bounds agree. If M=0 is allowed, use the identical no-observation experiment, with TV exactly zero. □

The endpoint argument establishes the exact asymptotic constant in the regime \(M/n\to0\). The finite-ratio value is instead characterized by Theorem 3. The result strengthens the interpretation of the sample threshold: a best constant predictor may be optimal when information vanishes, while its nonzero worst-case error still prevents uniform consistency.

### 5.3. Full-experiment equivalence: likelihoods and both kernels

Within this likelihood proof put \(h=h_0\). Write \(P_{n,0}=Q_0^{\otimes M}\), \(\rho_n=M/n\), and

\[
S_n=M^{-1/2}\sum_{j=1}^M h(Z_j),\qquad
V_n=M^{-1}\sum_{j=1}^M h(Z_j)^2.
\]

Let \(F_M\) be the CDF of \(S_n\) under \(P_{n,0}\). In the present family h is a nonconstant real-analytic function, so the preimage of any value is a Lebesgue-null set. Since \(f_0\) has a density, h(Z) has no atoms; the sum also has no atoms. Therefore \(F_M\) is continuous. Define

\[
T_n=\Phi^{-1}(F_M(S_n)).
\]

This parameter-independent statistic has the exact N(0,1) law under \(P_{n,0}\); events where the argument is zero or one have probability zero. The baseline law and the transform are known because the model family is known.

The CLT and continuity of Phi give sup_x|\(F_M\)(x)-Phi(x)|->0. On a compact interval for \(S_n\) the inverse normal CDF is uniformly continuous after restricting to a slightly larger compact probability interval. Tightness of \(S_n\) then proves

\[
S_n-T_n\longrightarrow0\quad\text{in }P_{n,0}\text{-probability}.
\tag{G.2}
\]

This use of a CLT is only one part of the proof. The essential next step controls the full likelihood ratios.

Let \(\Lambda_n\)(u)=dP_{n,u}/dP_{n,0}. Boundedness of h and compactness of u allow a uniform Taylor expansion, valid for all sufficiently large n:

\[
\log\Lambda_n(u)
=u\sqrt{\rho_n}S_n-\tfrac12u^2\rho_n V_n+R_n(u),
\qquad
\sup_{|u|\le A_0}|R_n(u)|\le C M n^{-3/2}.
\tag{G.3}
\]

Under \(P_{n,0}\), \(V_n\)->1 in probability, and \(\rho_n\)->\(\rho\). Equations (G.2)–(G.3) imply

\[
\sup_{|u|\le A_0}
\left|\log\Lambda_n(u)-\left(\sqrt\rho\,uT_n-\tfrac12\rho u^2\right)\right|
\longrightarrow0
\tag{G.4}
\]

in probability. Define g_u(t)=exp(sqrt(\(\rho\))u t-\(\rho\) u^2/2). We now strengthen convergence of log likelihoods to uniform L1 convergence of likelihoods.

For any fixed p>1, \(\log(1+x)\le x\) gives

\[
\sup_{|u|\le A_0}\Lambda_n(u)
\le\exp\left(A_0\left|\sum_jh(Z_j)\right|/\sqrt n\right).
\]

The centered bounded summands have a sub-Gaussian exponential-moment bound. Since \(M/n\) is bounded, the pth moments of the last display are uniformly bounded. Also sup_u g_u(\(T_n\))<=exp(sqrt(\(\rho\))\(A_0\)|\(T_n\)|), whose pth moment is bounded because \(T_n\) is exactly standard normal. Equation (G.4), tightness, and these bounds give uniform integrability and hence

\[
\varepsilon_n:=\tfrac12\mathbb E_{n,0}
\sup_{|u|\le A_0}|\Lambda_n(u)-g_u(T_n)|\longrightarrow0.
\tag{G.5}
\]

Define an auxiliary experiment on the original sample space by

\[
d\widetilde P_{n,u}=g_u(T_n)\,dP_{n,0}.
\]

These are probability measures, because E_{n,0}g_u(\(T_n\))=1 exactly. Equation (G.5) bounds sup_u TV(\(P_{n,u}\),\(\widetilde P_{n,u}\)) by \(\varepsilon_n\).

Under \(\widetilde P_{n,u}\), \(T_n\) has law N(sqrt(\(\rho\))u,1). Conversely, use the regular conditional distribution \(P_{n,0}\)(d z|\(T_n\)=t) to turn such a normal observation back into a full calibration sample. This kernel is independent of u; tilting by a function of \(T_n\) leaves the conditional law unchanged. Regular conditional distributions exist on the Euclidean sample spaces, and arbitrary versions on baseline-null sets do not matter because all normal shifts are mutually absolutely continuous. Thus the auxiliary experiment is exactly equivalent to observing T~N(sqrt(\(\rho\))u,1), or Y=T/\(\sqrt\rho\).

Applying the statistic \(T_n\) in one direction and this baseline conditional kernel in the other gives both deficiencies bounded by \(\varepsilon_n\). This proves the full-experiment equivalence in Theorem 3. □

The proof supplies asymptotic equivalence on a fixed compact local parameter interval. It does not give a calibrated numerical value of \(\varepsilon_n\), nor global equivalence over an expanding parameter range. The conditional kernel is an existence construction; implementing it is not required to use a decision rule in the forward direction.

### 5.4. Minimax risk transfer

 Project all decision values to [0,1], which cannot increase loss. Absolute loss is then bounded by one, so transporting any randomized rule through either Markov kernel changes each parameterwise risk by at most \(\varepsilon_n\). Replacing the target \(\pi_n\) by \(\Phi(cu)\) changes each loss by at most \(r_n\). Taking supremum risks and then infima in both directions gives

\[
|\mathfrak R_{n,M}(A_0)-V(A_0,c,\rho)|\le\varepsilon_n+r_n.
\]

This reasoning does not require the minimax infimum to be attained: apply it to arbitrarily near-optimal rules. Under absolute loss, averaging any external decision randomization conditionally on the observation cannot worsen risk, by convexity. Thus restricting to deterministic rules leaves the values unchanged. □

Standardize with

\[
t=cu,\qquad K=cA_0,\qquad X=cY=t+rZ,\qquad r=c/\sqrt\rho.
\]

The limiting decision problem depends only on (K,r):

\[
V_G(K,r)=\inf_d\sup_{|t|\le K}\mathbb E_t|d(X)-\Phi(t)|.
\tag{G.7}
\]

Actions may be restricted further to [\(\Phi(-K)\),\(\Phi(K)\)]. Reflection and averaging permit a symmetric rule d(-x)=1-d(x) without increasing worst-case risk, although the certificates below do not require a proof that the candidate optimizer is minimax.

### 5.5. Uniform risk of the raw moment predictor

 By the exact calibration moment identities, the standardized calibration sum

\[
Z_{n,u}=\frac{\sqrt M(\widehat\theta-u/\sqrt n)}
 {\sqrt{1-u^2/n}}
\]

has a uniform Berry–Esseen bound C/\(\sqrt M\) on the compact u-window: h is bounded and its variance is uniformly bounded away from zero for large n. Hence

\[
c\sqrt n\widehat\theta
=cu+c\sqrt{n/M}\sqrt{1-u^2/n}\,Z_{n,u}.
\]

The coefficient of \(Z_{n,u}\) tends uniformly to r. This implies uniform convergence of the bounded loss expectations to \(D(cu,r)\). One direct justification is by subsequences: any alleged nonuniformity supplies u_n in the compact window; extract u_n->u_*, apply the triangular CLT and bounded continuous mapping along that subsequence, and use continuity of D. Replacing \(\Phi(cu)\) by the true \(\pi_n\) changes expected absolute loss by at most \(r_n\). This proves the uniform moment-risk convergence.

Pair positive and negative Z to obtain

\[
D(t,r)=\int_0^\infty
 [\Phi(t+rz)-\Phi(t-rz)]\varphi(z)\,dz.
\tag{L.18}
\]

Symmetry makes this even in t. For t>0 and z>0,
|t+rz|>|t-rz|, so differentiation under the integral gives a strictly negative derivative in t. Therefore the maximum over a window containing zero is \(D(0,r)\).

Finally,

\[
\frac{\partial}{\partial r}D(0,r)
=2\int_0^\infty z\varphi(rz)\varphi(z)\,dz
=\frac1{\pi(1+r^2)}.
\]

Since D(0,0)=0, integration yields \(D(0,r)\)=arctan(r)/pi. Uniform convergence in the uniform moment-risk convergence allows passage to the supremum, proving (L.17). □

The arctan expression remains a result for this specified predictor. It is now its worst-case local-window risk limit, rather than a universal lower bound. In this known family, the constant predictor has worst-case limiting risk \(B_0\); it can outperform the raw moment predictor for some combinations of window width and calibration ratio.


## 6. Unknown local directions and nuisance

The known-tilt subfamily already realizes the hardest first-order direction around its known baseline. Define the influence function
\[
\psi(z)=\frac{U(0,z)}{\lambda s},\qquad E_0\psi^2=c^2.
\]
For any bounded centered score h with \(I_h=E_0h^2>0\), differentiating the root equation gives normalized root sensitivity \(d_h=E_0[\psi h]\). Hence
\[
\frac{d_h^2}{I_h}\le c^2,
\]
with equality precisely for a nonzero multiple of the entropy gradient \(U(0,\cdot)\). The comparison fixes the seed window, not an arbitrary common parameter window with different attainable seeds.

Consider all unknown measurable k in
\[
\mathcal K_{B,K}=\{k:E_0k=0,\ \|k\|_\infty\le B,\ |E_0\psi k|\le K\},
\quad f_{n,k}=f_0(1+k/\sqrt n).
\]
The baseline, B and K are known; k is not. For large n, these densities preserve legality with the same signed density g and the same complete risk. State derivatives of U are bounded, so the whole class lies in a common shrinking smooth-field neighborhood. A repeated stopped-seed proof gives
\[
\sup_{k\in\mathcal K_{B,K}}|\pi_n(Q_{n,k})-\Phi(E_0\psi k)|\le Ce_n.
\]

**Theorem 4 (Unknown-direction minimax value; N.3).** Suppose \(B\ge K\|\psi\|_\infty/c^2\). For \(M/n\) tending to positive finite \(\rho\), the minimax prediction risk over this unknown-direction class converges to \(V_G(K,c/\sqrt\rho)\). Uniform consistency is possible exactly when \(M/n\) tends to infinity; for \(M/n\) tending to zero the value tends to \(\Phi(K)-1/2\).

The lower bound restricts to \(k_t=t\psi/c^2\), \(|t|\le K\), which the envelope condition guarantees is contained in the class. For the upper bound use the direction-free statistic
\[
\widehat t=\frac{\sqrt n}{M}\sum_{j=1}^M\psi(\widetilde Z_j),
\quad E\widehat t=E_0\psi k,
\quad \operatorname{Var}_{n,k}\psi=c^2+O(n^{-1/2}).
\]
Its uniform CLT transfers any bounded Lipschitz Gaussian rule's risk. Gaussian smoothing approximates an arbitrary near-optimal bounded rule with no asymptotic loss, completing the upper bound. Fixed finite step rules also transfer because the limiting Gaussian puts vanishing probability near their thresholds.

This is equality of minimax decision values, not scalar equivalence of the full infinite-dimensional experiment. Orthogonal directions may still contain additional information about nuisance. N.4 gives, for a finite score span with information matrix I and \(d=E_0[H\psi]\), the variance \(d^\top I^{-1}d/\rho=\|\Pi_{\operatorname{span}(H)}\psi\|_2^2/\rho\le c^2/\rho\). Root-preserving nuisance is orthogonal to \(\psi\); it causes no additional first-order penalty for the influence-aligned entropy tilt. These facts do not extend without a new argument to an unknown baseline or growing local envelope. Complete proofs and projection qualifications are in [N.1–N.4](unknown_direction.md).

### 6.1. Root sensitivity and the information bound

 Differentiate E_epsilon U(q_epsilon,Z)=0. At epsilon=0 the state derivative is \(\lambda\) and the distribution derivative is \(E_0\) U0 h, proving (N.2). Differentiating log f_epsilon at zero gives score h and information \(E_0\) h^2. Cauchy–Schwarz gives (\(E_0\) \(\psi\) h)^2<=\(E_0\) \(\psi\)^2 \(E_0\) h^2. Its equality condition proves the last assertion. □

Thus the original \(h_0\)=U0/\(\sigma_g\) is not merely a convenient tilt: it maximizes normalized seed displacement for a given local statistical distinguishability. The term “least favorable” here concerns estimation of this target, after fixing the target window; it is not an ordering of all nonlinear environmental shifts.

If \(\theta\)=u/\(\sqrt n\) and the score h is known, its limiting seed is \(d_h\) u, while the calibration information is \(\rho\) \(I_h\). At a fixed seed window t=\(d_h\) u in [-K,K], the resulting Gaussian noise standard deviation is

\[
r_h=\frac{|d_h|}{\sqrt{\rho I_h}}\le\frac c{\sqrt\rho}.
\tag{N.4}
\]

For \(d_h\)=0 the first-order seed is zero, so this reparameterization is not applicable; a second-order analysis would be a different local problem. Comparing different h at a fixed u-window would change the attainable target window and would not be the same least-favorability comparison.

### 6.2. Uniform exit over the unknown-direction class

For sufficiently large n, all these fields have one root q_{n,k} on the same compact exit domain, and uniformly over k,

\[
q_{n,k}=-\frac{E_0[U_0 k]}{\lambda\sqrt n}+O(n^{-1}),
\quad \lambda_{n,k}=\lambda+O(n^{-1/2}),
\quad s_{n,k}=s+O(n^{-1/2}).
\tag{N.6}
\]

Their true right risk-exit probabilities obey

\[
\boxed{\sup_{k\in\mathcal K_{B_{\rm loc},K}}
|\pi_n(Q_{n,k})-\Phi(t(k))|\le C e_n,}
\quad e_n=n^{-1/2}+((1+\log n)/n)^{1/3}.
\tag{N.7}
\]

**Proof.** For every fixed derivative order j needed by the stopped-seed proof,

\[
\|v_{n,k}-v_0\|_{C^j}
\le \frac{B_{\rm loc}}{\sqrt n}
\max_{0\le l\le j}\sup_{x,z}|\partial_x^lU(x,z)|.
\]

Thus the entire class lies in one shrinking smooth-field neighborhood. No derivative of k with respect to z is needed. The baseline derivative lower bound and endpoint drift signs persist uniformly. Since v_{n,k}(0)=\(E_0\) U0 k/\(\sqrt n\), Taylor expansion at zero and the common positive derivative give q=O(n^{-1/2}) and the root expansion (N.6). Bounded derivatives and bounded U^2 give the remaining expansions. The baseline root noise is nonzero, so its uniform lower bound persists.

For completeness, the probabilistic part can be repeated with exactly the uniform constants of L.2 rather than applying a fixed-law theorem. All maps have inverse contraction and \(C_2\) conjugacy bounds in the same smooth neighborhood; their root-to-boundary distances and conjugate exit magnitudes are bounded away from zero. Bounded U and its derivative give common batch second/fourth moments and the state-difference noise bound. For the stopped normalized conjugacy Y and frozen-root seed W, the same orthogonality and Taylor estimates yield

\[
\sup_{k,j}E|Y_j|^2\le C,\quad
\sup_k E A_{n,k}^{-2\tau}\le C/n,\quad
\sup_k E|Y_\tau-W|^2\le C(1+\log n)/n.
\]

Exit is finite for every member because f_{n,k}>=\(f_0\)/2 supplies a common positive mass in each of the fixed forcing intervals. Frozen seed columns have uniformly bounded third moments and variances bounded below, so Berry–Esseen and seed anti-concentration hold uniformly. Choosing the same anti-concentration cutoff as in L.2 bounds the error of identifying the final exit sign by Ce_n. Finally,

\[
\sqrt n H_{n,k}(0)/s_{n,k}=t(k)+O(n^{-1/2})
\]

uniformly by (N.6) and the normalized \(C^2\) bound. The global risk-safe interval is unchanged, so this is actual risk exit without a component-transfer correction. This proves (N.7). □

The proof uses a uniform bounded envelope and a finite number of smooth U derivatives. It does not require entropy or covering-number control of the function class of directions, since no uniform empirical process over k is estimated.

### 6.3. Upper and lower minimax arguments

Here \(B_{\mathrm{loc}}\) denotes the envelope B used in the main statement, and the class has the same target window K.

Define the scalar statistic from independent calibration scores

\[
\widehat t=\frac{\sqrt n}{M}\sum_{j=1}^M\psi(\widetilde Z_j).
\tag{N.8}
\]

It is available without knowing k. Under \(Q_{n,k}\),

\[
E\widehat t=t(k),\qquad
\operatorname{Var}(\psi(\widetilde Z))
=c^2+n^{-1/2}E_0[\psi^2 k]-t(k)^2/n
=c^2+O(n^{-1/2})
\tag{N.9}
\]

uniformly. Boundedness of \(\psi\) gives a uniform triangular-array CLT. For \(M/n\to\rho\) in (0,infinity), the distribution of \(\widehat t\) is uniformly approximated in Kolmogorov distance by N(\(t(k)\),\(c^2\)/\(\rho\)).

Let \(R^{\mathcal K}_{n,M}\) be the minimax expected absolute prediction loss over the declared unknown-direction class, using the exact \(\pi_n\) as the target. Let

\[
V_G(K,r)=\inf_d\sup_{|t|\le K}E|d(t+rZ)-\Phi(t)|.
\]

**Lower bound.** The unknown-direction class contains the subfamily

\[
k_t=(t/c^2)\psi=(t/c)h_0,\qquad |t|\le K,
\]

by the envelope condition of Theorem 4. Its seed coordinate is exactly t, and it is the original entropy-gradient tilt with \(\theta\)=t/(c \(\sqrt n\)). Restricting to this subfamily and even telling the predictor that this restriction holds can only reduce the minimax risk. The experiment-level Gaussian reduction already proved for that subfamily gives liminf R^K>=\(V_G\)(K,c/\(\sqrt\rho\)). This invokes a full-experiment lower bound, not a lower bound inferred from a moment CLT.

**Proof of the upper bound.** Any fixed bounded Lipschitz Gaussian rule d, applied to \(\widehat t\), has uniformly convergent risk by (N.9), its uniform CLT, and (N.7). This suffices to approach \(V_G\) arbitrarily closely. To see why arbitrary measurable Gaussian rules do not create a gap, start with an epsilon-optimal rule valued in [0,1]. Smooth it by d_a(x)=E[d(x+aZ')], with independent standard-normal Z'. The smoothed rule is bounded Lipschitz. Jensen gives

\[
E_t|d_a(X)-\Phi(t)|
\le E_t|d(X+aZ')-\Phi(t)|.
\]

The last expectation differs from that under X~N(t,\(r^2\)) by at most TV(N(0,\(r^2+a^2\)),N(0,\(r^2\))), uniformly in t. This tends to zero as a->0. Thus bounded Lipschitz rules are sufficient to approximate the Gaussian minimax value. Apply their uniform risk convergence, then let a and epsilon tend to zero. The upper and lower bounds agree.

For sufficiency when \(M/n\to\infty\), use \(\Phi(\widehat t)\). Its error is bounded uniformly by

\[
\frac1{\sqrt{2\pi}}\sqrt{\frac nM}
\sup_k\sqrt{\operatorname{Var}_{n,k}\psi}+Ce_n\to0.
\]

For necessity, a bounded-ratio subsequence contains the original nondegenerate least-favorable subfamily, to which the two-point lower bound applies. □

This theorem equates a particular decision problem's minimax value. It does **not** assert that the entire infinite-dimensional experiment is equivalent to a scalar Gaussian observation: the full sample may contain information about orthogonal nuisance directions. The lower bound comes from a subexperiment and the upper bound from a direction-free rule.

If \(M/n\to0\), the same subfamily lower bound and the constant forecast upper bound also give \(R^{\mathcal K}_{n,M}\)->\(\Phi(K)\)-1/2. This extends the earlier zero-information constant to the enlarged class, without characterizing the entire nuisance experiment.

The fixed finite step rules already computed for the Gaussian problem also transfer uniformly: their finitely many discontinuities have vanishing neighborhood probability under the limiting normal densities. Consequently the existing numerical minimax brackets are relevant to the enlarged local class without rerunning the optimizer or knowing k.

### 6.4. Tangent projection and target-preserving nuisance

For a finite collection of bounded centered scores H=(h1,...,hd) with nonsingular information I=\(E_0\) \(HH^\top\), let d=\(E_0\)[H \(\psi\)]. The usual local variance associated with estimating the seed functional in this score span is

\[
\frac1\rho d^\top I^{-1}d
=\frac1\rho\|\Pi_{\operatorname{span}(H)}\psi\|_2^2
\le\frac{c^2}{\rho}.
\tag{N.12}
\]

**Proof.** The projection has coefficients \(I^{-1}d\), so its squared L2 norm is \(d^\top I^{-1}d\). Projection contracts norm. Equivalently, an empirical mean of \(d^\top\) I^{-1}H estimates the derivative functional in this local model with that first-order variance. □

For a closed admissible tangent space T, the corresponding sensitivity supremum is

\[
\sup_{h\in T,\,\|h\|_2>0}
\frac{\langle\psi,h\rangle^2}{\|h\|_2^2}
=\|\Pi_T\psi\|_2^2.
\tag{N.13}
\]

Equality as an operational bounded density tilt requires that the maximizing direction be admissible and bounded, or be approximable by admissible bounded scores. Equation (N.13) alone does not supply the geometry and uniform exit proof for a new model class.

If an interest score h is accompanied by nuisance scores spanning N, and those nuisance directions preserve the seed functional to first order, then N is orthogonal to \(\psi\). The efficient interest score is \(h_{\rm eff}=h-\Pi_Nh\) and

\[
\frac{\langle\psi,h\rangle^2}{\|h_{\rm eff}\|_2^2}
=\frac{\langle\psi,h_{\rm eff}\rangle^2}{\|h_{\rm eff}\|_2^2}
\le c^2.
\tag{N.14}
\]

Thus nuisance can reduce information relative to a restricted one-direction model, but cannot exceed the full tangent variance \(c^2\) in this known-baseline first-order problem. In particular, for h=\(h_0\) proportional to \(\psi\) and root-preserving nuisance, \(h_{\rm eff}\)=\(h_0\): there is no additional first-order penalty.

If a proposed nuisance direction has \(E_0\) \(\psi\) k nonzero, it changes the target and is not a target-preserving nuisance. It must be included in \(t(k)\), as in Theorem N.3, rather than projected out while keeping the target fixed. Moreover, a bounded multidimensional parameter window can permit decision improvements beyond a scalar variance comparison; (N.12) is an information calculation, not a general finite-window minimax formula.

**Concrete nuisance example.** At the symmetric Gaussian baseline, let b(z)=cos(z)-\(E_0\) cos(Z). It is bounded, centered and even, whereas \(\psi\) and \(h_0\) are odd. Thus \(E_0\) \(\psi\) b=\(E_0\) \(h_0\) b=0. The two-direction perturbation k_{t,v}=(t/c)\(h_0\)+v b has seed t, with an unknown bounded v changing other aspects of the score distribution. It stays in the declared unknown-direction class whenever |t| H_*/c+|v| ||b||_infinity<=\(B_{\mathrm{loc}}\). This supplies a legal common-risk nuisance family with no additional first-order penalty. No new nuisance simulation is claimed in the present confirmation run.


### 6.5. Finite baseline information and conditional prediction

 We also allow the baseline marginal to be unknown in a fixed bounded density neighborhood, while preserving the complete common classification risk. An independent baseline sample of size \(L\) selects the initialization through an empirical entropy root. The prediction target is the future risk-exit probability conditional on this realized initialization. A uniform actual-SGD approximation and a two-sample expansion give the Gaussian interface
\[
S=t+\frac{c}{\sqrt\ell}Z_L,\qquad
D=S+\frac{c}{\sqrt\rho}Z_C,\qquad
\text{target}=\Phi(S),
\]
when \(L/n\to\ell\in(0,\infty)\) and \(M/n\to\rho\in(0,\infty)\). The shared initialization noise changes the decision problem: its minimax value is \(V_{\mathrm{FB}}\), rather than the original deterministic-target Gaussian value with an added variance. We prove matching upper and lower bounds over the unknown-baseline class. For \(K>0\) and \(L/n\to\ell\in(0,\infty]\), consistent conditional prediction remains possible exactly when \(M/n\to\infty\); superlinear baseline information is unnecessary. In the root-preserving case \(K=0\), the optimal limiting rule shrinks the raw displacement estimate by \(M/(L+M)\), and its MAE has the closed form (F.35) in [the complete integrated proofs](unknown_direction.md#6-finite-baseline-information-and-conditional-prediction). These claims concern a regular scalar experiment and do not establish a reduction of neural-network adaptation.

 The added model-specific contribution is the connection between sample-based initialization, conditional actual-SGD exit and the joint baseline/calibration decision problem. The conjugacy and unstable-exit mechanism, likelihood comparison, Gaussian conditioning, convex minimax and posterior-median facts are classical. The result should not be presented as a new general principle of statistical efficiency.


For finite positive ratios, writing $a_P=c_P/\sqrt{\ell}$ and $b_P=c_P/\sqrt{\rho}$, the asymptotic minimax value over the declared regular class is $\sup_P V_{\mathrm{FB}}(K;a_P,b_P)$, where
\[
V_{\mathrm{FB}}(K;a,b)=\inf_d\sup_{|t|\le K}E_t|d(S+bZ_C)-\Phi(S)|,\qquad S=t+aZ_L.
\]
In the root-preserving subcase the feasible asymptotically optimal forecast is
\[
\widehat\pi_{\mathrm{pool}}=\Phi\!\left(\frac{M}{L+M}\widehat D_n\right).
\]
The pooled observations affect only prediction; the future trajectory still starts at the baseline-sample initialization. If $K=0$ and $L/n\to\infty$, the constant forecast $1/2$ is consistent without target calibration. General-window numerical optimization and actual-SGD confirmation of this finite-baseline experiment remain open.

## 7. Gaussian certificates and real-SGD confirmation

### 7.1. Gaussian decision certificates

A finite prior supplies a Bayes lower bound, and its finite-decimal posterior-median step rule supplies an upper bound. The certificate encloses the true posterior crossings and normal probabilities with directed arithmetic. An interpolation remainder based on a risk-curvature bound controls the entire continuous parameter window between all loss kinks. It neither assumes the prior is exactly least favorable nor relies on optimizer convergence.

| K | r | Certified minimax interval | Raw moment risk |
|---:|---:|---:|---:|
| 1 | 2 | [0.226230, 0.226386] | 0.352416 |
| 1 | 1 | [0.164401, 0.164587] | 0.250000 |
| 1 | 0.5 | [0.107333, 0.107539] | 0.147584 |
| 2 | 2 | [0.244686, 0.244931] | 0.352416 |
| 2 | 1 | [0.169889, 0.170268] | 0.250000 |
| 2 | 0.5 | [0.108534, 0.109092] | 0.147584 |

At \(K=r=1\), the candidate rule's certified worst risk is at most .164587, at least 34.16% below the raw moment rule's exact .25. This is a Gaussian worst-risk improvement, not pointwise dominance or a finite-n SGD certificate. The interval implementation trusts the stated Decimal runtime; it is not a proof-assistant verification. [Full numerical analysis and specifications](gaussian_decision.md).

### 7.2. Independent fixed-rule confirmation

The experiment keeps the pilot's selected rule and ten cells fixed, excludes pilot data, and uses actual sampled nonlinear entropy updates. Fix \(w=2\), \(\nu=1\), \(\eta=.1\), \(x_0=0\), \(b=.25\), \(\kappa=.5\), with \(\epsilon\) approximately .0060745243 and c approximately 9.1436882. For \(n\in\{128,512\}\), set \(M=\operatorname{round}(c^2n)\) and generate \(\theta=t/(c\sqrt n)\) at \(t\in\{-1,-.5,0,.5,1\}\). Predictors receive calibration scores and common constants only. The rule is not optimized again after seeing these trajectories.

Each cell has 16,384 reference paths and 4,096 independent calibration repetitions. All four rules share calibration scores within a repetition; reference streams are independent. The fixed cap is 1,500 steps. All 163,840 paths exited, and the largest observed exit step was 720. Numerical geometry checks are diagnostic rather than interval-certified. No nuisance simulation is claimed.

| n | Raw moment maximum MAE | Clipped maximum MAE | Optimized maximum MAE | Moment minus optimized, simultaneous 95% interval | Clipped minus optimized, simultaneous 95% interval |
|---:|---:|---:|---:|---:|---:|
| 128 | .2481 | .2238 | .1663 | [.0299, .1338] | [.0076, .1075] |
| 512 | .2476 | .2228 | .1684 | [.0273, .1311] | [.0046, .1043] |

These maxima are over five fixed parameter cells. The comparison subtracts maxima of expected losses, not a maximum of paired gains. Joint reference/calibration intervals show positive five-cell maximum-risk improvement over both baselines at both batch sizes. Clipping remains better at some endpoints, so there is no pointwise dominance claim.

### 7.3. Two uncertainty layers

Allocate .025 error to reference paths and .025 to calibration losses. At each cap, right-exit probability by the cap is below eventual \(\pi_n\), while the probability of not having exited left is above it. One-sided exact binomial bounds bracket these probabilities with a union bound over cells; possible population censoring is thus covered even when no sampled paths are censored.

Conditional on the independent reference sample, losses lie in [0,1] and paired differences in [-1,1]. Two-sided empirical Bernstein bounds [8], with a union bound over 40 rule risks and 30 paired gains, control calibration error. Add the reference error to each rule-risk bound and twice that error to paired differences. Propagate cellwise risk intervals through maxima and differences. This provides simultaneous coverage at least 95% under the declared iid experiment, including both uncertainty layers. The beta quantiles use floating-point routines, so these statistical intervals are distinct from the directed Gaussian numerical certificate.

Complete design, uncertainty formulas, cellwise tables and original raw arrays are in the [confirmation report](../experiments/results/precision_confirmation/analysis.md). Results and code stay in their original directories. The earlier [pilot](../experiments/results/gaussian_decision/analysis.md#72-actual-entropy-sgd-pilot) is exploratory and is not pooled into these intervals.

### 7.4. Constructing a certified minimax bracket

Choose any finite prior with ordered support t_1<...<t_m in [-K,K] and positive weights w_i. Its posterior median under absolute loss is a Bayes rule taking the values Phi(t_i). The boundary between actions k and k+1 solves

\[
\sum_{i\le k}w_i e^{x t_i/r^2-t_i^2/(2r^2)}
=\sum_{i>k}w_i e^{x t_i/r^2-t_i^2/(2r^2)}.
\tag{G.8}
\]

The ratio of the left sum to the right sum is strictly decreasing: its log derivative is the difference of two tilted means divided by \(r^2\), and every left support point is smaller than every right support point. Its limits are infinity and zero. Thus every crossing is unique; the ordered crossings define the posterior-median rule.

For any step rule with thresholds b_j and action values a_j, its exact risk is

\[
R_d(t)=\sum_j |a_j-\Phi(t)|
\left[\Phi\left(\frac{b_{j+1}-t}{r}\right)
      -\Phi\left(\frac{b_j-t}{r}\right)\right],
\tag{G.9}
\]

including b_0=-infinity and the last boundary +infinity. Therefore the full unbounded observation domain is integrated through CDF differences, with no omitted integration tail.

The code optimizes a finite prior using floating-point quadrature only to find a candidate. It then rounds support points, prior numerators and rule thresholds to explicit finite decimals. These rounded quantities define the actual certified prior and rule.

For each true crossing, directed interval arithmetic verifies opposite signs of (G.8) at endpoints l_j and u_j, with the candidate threshold inside the bracket. The candidate rule and exact Bayes rule can differ only on the union of these strips. Gaussian mixture density is at most 1/(r sqrt(2 pi)); loss differences are at most one. Hence

\[
\text{Bayes risk}\ge
\sum_i w_i R_d(t_i)-\frac{\sum_j(u_j-l_j)}{r\sqrt{2\pi}}.
\tag{G.10}
\]

A lower enclosure for the right side is a valid minimax lower bound. This does not assume that the candidate prior is least favorable, that the search objective was computed exactly, or that the optimizer converged.

### 4.2. Continuous-window upper bound

The candidate step rule itself gives an upper bound once sup_{|t|<=K} R_d(t) is bounded. Its action values are exactly Phi(t_i), so the only absolute-loss kinks are at the known support points t_i. Between successive kinks, differentiation under the Gaussian integral gives

\[
|R_d''(t)|\le C_2:=1+2/r+2/r^2.
\tag{G.11}
\]

Indeed the target's first and second derivatives are bounded in absolute value by one; the sum of absolute bin-probability derivatives is bounded by E|Z|/r<=1/r, and that of second derivatives by E|Z^2-1|/\(r^2\)<=2/\(r^2\). All action losses are bounded by one. The bound is deliberately conservative.

For a partition interval [a,b] with no kink in its interior, linear interpolation and (G.11) imply

\[
\sup_{t\in[a,b]}R_d(t)
\le\max\{R_d(a),R_d(b)\}+C_2(b-a)^2/8.
\tag{G.12}
\]

The partition includes every support point and both endpoints. Adaptive refinement produces an upper bound over the entire continuous window, not just the evaluated grid.

### 4.3. Arithmetic and probability enclosures

`certify.py` uses Python Decimal with 90-digit precision and outward rounding for interval operations. Correctly rounded Decimal exponentials and square roots are expanded by a representable value on each side. Pi is enclosed by Machin's identity and alternating arctangent remainders. Normal CDF values are enclosed by integrating the Taylor expansion of exp(-x^2/2), including a rigorous next-term remainder once the terms decrease. For |x|>=12 a Mills-ratio tail bound is used instead. Both negative and positive tails are retained in their intervals.

The optimizer's normal CDF, integration grid, stopping tolerance and floating-point objective do not enter the validity of the final bounds. The certificate relies on the stated mathematical bounds, the correctness of the supplied implementation, and the documented Decimal arithmetic; it is not a proof-assistant verification of the runtime.

The search gap and the certified gap are different quantities. The final upper-minus-lower interval accounts for an imperfect prior/rule pair as well as root, probability-evaluation and continuous-window errors.

### 7.5. Using the fixed rule without the local parameter

For an arbitrary measurable Gaussian rule, the direct risk-transfer construction applies it to c \(T_n\)/\(\sqrt\rho\). This uses the known baseline CDF \(F_M\), not the unknown environment parameter.

For each of the fixed finite step rules produced here, a simpler implementation is also asymptotically justified: apply it to Xhat=c \(\sqrt n\) hat_theta, the raw empirical moment coordinate. Under the baseline, Xhat-c \(T_n\)/\(\sqrt\rho\) tends to zero in probability. The likelihood-ratio moment bounds used in (G.5) transfer this to the compact local family uniformly. The Gaussian statistic has a continuous density uniformly bounded on the local window, so the probability of lying near one of the finitely many thresholds tends to zero with the threshold neighborhood. Therefore the two implementations have uniformly vanishing risk difference.

This argument is for a fixed finite rule. Refining the rule with n so that the number of thresholds diverges requires additional control. No finite-n risk certificate for the SGD experiment follows merely from the Gaussian numerical certificate.

### 7.6. Simultaneous reference and calibration uncertainty

At the fixed cap, let \(p_R\) be the probability of a right exit by the cap and \(p_{NL}\) the probability of not having exited left by the cap. The eventual right-exit probability satisfies

\[
p_R\le\pi_n\le p_{NL}.
\]

For each cell use a one-sided exact Clopper–Pearson lower bound on \(p_R\) and upper bound on \(p_{NL}\), each with error \(\alpha_{\rm ref}\)/(2C). Their endpoints [l_i,u_i] jointly cover every cell's eventual \(\pi_n\) with probability at least 1-\(\alpha_{\rm ref}\). This remains valid if a cell has censored paths; lack of an observed censor is not treated as proof that its population censoring probability is zero.

Set \(\widehat p_i\)=\(N_R/N\) and \(e_i\)=max(\(\widehat p_i\)-l_i,u_i-\(\widehat p_i\)). On the reference event, |\(\widehat p_i\)-pi_i|<=\(e_i\).

#### 2.2. Calibration loss uncertainty conditional on the reference sample

For each rule and cell, score calibration repetition j by L_j=|forecast_j-\(\widehat p_i\)|. Conditional on the independent reference sample, these are iid in [0,1]. Paired baseline-minus-optimized loss differences are iid in [-1,1].

For m observations taking values in an interval of width B, with unbiased sample variance \(S_m^2\), use the two-sided empirical Bernstein radius

\[
b_m(B,\delta)=\sqrt{\frac{2S_m^2\log(4/\delta)}m}
+\frac{7B\log(4/\delta)}{3(m-1)}.
\]

This follows by scaling and applying both tails of Maurer–Pontil (2009), Theorem 4. Allocate \(\delta\)=\(\alpha_{\rm cal}\)/(7C) to each of 40 rule-risk estimates and 30 paired gains. Their conditional coverage is at least 1-\(\alpha_{\rm cal}\) by a union bound; integration over the reference sample preserves that bound. Independence between the different rules is not required.

For each rule risk, add \(e_i\) to its calibration radius. For a paired gain add 2\(e_i\), by the reverse triangle inequality. For the deterministic constant rule, propagate [l_i,u_i] through |.5-p| exactly instead of using an unnecessary calibration bound.

The resulting risk and gain intervals have **simultaneous coverage at least 95%** over all reported cells and rules, under the declared iid simulation experiment. They include both uncertainty layers. They are not just conditional standard errors or pointwise normal intervals. Evaluation of \(\beta\) quantiles uses SciPy floating-point routines; these are statistical confidence statements, distinct from the previous directed-arithmetic Gaussian decision certificates.

#### 2.3. Finite-grid maximum risks

For each n let \(R_{\rm rule}^{\rm grid}\) be the maximum of the five true expected calibration losses at the fixed t cells. If cellwise intervals are [L_i,U_i], then

\[
\max_iL_i\le R_{\rm rule}^{\rm grid}\le\max_iU_i.
\]

For the improvement in grid maximum risk, subtract intervals: its lower bound is max_i L_baseline,i minus max_i U_optimized,i. The same simultaneous event covers all these derived comparisons. This is a five-cell statement, not a continuous-window maximum or finite-n minimax theorem.

Reference: Andreas Maurer and Massimiliano Pontil, *Empirical Bernstein Bounds and Sample Variance Penalization*, COLT 2009, Theorem 4. https://www.cs.mcgill.ca/~colt2009/papers/012.pdf

### 7.7. Cellwise confirmation and interpretation

| n | t | Reference right probability | Simultaneous reference interval | Moment MAE | Clipped MAE | Optimized MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | -1 | 0.16321 | [0.1545, 0.1721] | 0.18038 | 0.12645 | 0.15757 |
| 128 | -0.5 | 0.30817 | [0.2973, 0.3192] | 0.23400 | 0.20093 | 0.16552 |
| 128 | 0 | 0.49817 | [0.4863, 0.5101] | 0.24813 | 0.22381 | 0.16306 |
| 128 | 0.5 | 0.69037 | [0.6793, 0.7013] | 0.23015 | 0.19765 | 0.16299 |
| 128 | 1 | 0.84259 | [0.8338, 0.8511] | 0.18340 | 0.13257 | 0.16626 |
| 512 | -1 | 0.16205 | [0.1534, 0.1710] | 0.18425 | 0.13184 | 0.16259 |
| 512 | -0.5 | 0.30695 | [0.2960, 0.3180] | 0.22602 | 0.19352 | 0.16187 |
| 512 | 0 | 0.50385 | [0.4920, 0.5157] | 0.24760 | 0.22282 | 0.16288 |
| 512 | 0.5 | 0.68506 | [0.6739, 0.6960] | 0.23076 | 0.19884 | 0.16429 |
| 512 | 1 | 0.84711 | [0.8384, 0.8556] | 0.18185 | 0.13448 | 0.16837 |

The interval construction supports positive paired improvements over the raw moment rule in 6/10 cells and over the clipped rule in 2/10 cells. At endpoint cells clipping is often better, so there is no claim of uniform pointwise improvement. At t=0 the target equals .5 exactly by symmetry in this family; for consistency of the primary analysis, that cell was scored with the independent reference estimate, without using the exact value to narrow its interval.

The largest observed exit step was 720. Raw exit arrays, forecast arrays, fixed-design metadata and the frozen rule are retained. The aggregate counts and MAEs agree with the retained NPZ files.


### 7.8. Unknown local visual environments: independent confirmation

**Setup and information.** We next test prospective probability prediction on real visual scores, using the frozen ImageNet-pretrained ResNet-18 features and four prespecified CIFAR-10 class pairs. Official training features determine a fixed projection. For each pair, the official test empirical baseline determines one root \(q_0\) and its constants; all environments retain \(x_0=q_0\). With centered, unit-variance baseline entropy velocity \(h_0(z)\), we draw images with weights proportional to \(1+\theta h_0(z)\), where \(\theta=t/(c\sqrt n)\) and \(c=\sigma_0/(\lambda_0s_0)\). The generator knows \(t\); predictors receive only independent unlabeled calibration scores and the declared baseline constants. They do not receive target roots, labels, generating coordinates or reference exits. Unlike the theoretical construction, reweighting labeled images changes the complete classification-risk function: risk is evaluated under each environment's actual weighted label law.

**Exploration and independent confirmation.** The exploratory study covered four pairs, \(n\in\{32,128,512\}\) and \(t\in\{-1,-.5,0,.5,1\}\), with 1,024 reference paths per environment and 256 calibration repetitions per budget. Fixed relative calibration noise \(r=c\sqrt{n/M}\in\{.5,1,2\}\) gave an error plateau across batch sizes; the sequence \(r=2,1,.5\) gave decreasing error. These are finite-grid trends conditional on fixed image pools, not new independent images or a consistency theorem for a finite collection of frozen rules. All 36 fixed-ratio settings favored the optimized rule in maximum-MAE point estimates, but their simultaneous improvement intervals crossed zero. We therefore fixed a new confirmation design before generating its samples, rather than pooling exploration into confirmation.

**Fixed design and scoring.** Confirmation retains all four pairs, fixes \(n=128\), \(K=r=1\), and uses the same five-point \(t\) grid. Each of the 20 environments has 16,384 fresh reference paths and 4,096 independent calibration repetitions, totaling 327,680 paths and 81,920 repetitions. We use \(\eta=.1\), slope \(w=2\), tolerance \(\epsilon=.02\), baseline prior \(.5\), \(M=\operatorname{round}(c^2n)\), and a fixed 4,000-update cap for every pair. Five predictors share each calibration repetition: constant \(.5\), empirical-root plug-in, raw influence-function moment, window-clipped moment, and the previously frozen Gaussian step rule. Root failures would fall back to \(.5\) and remain in the score. The primary comparison is clipped minus optimized five-point maximum expected MAE, separately for each pair.

**Uncertainty and results.** We allocate error \(.0125\) to each pair, giving at least 95% simultaneous coverage across all four pairs, five rules and recorded paired comparisons by a union bound. Censor-compatible exact one-sided binomial reference bounds and conditional empirical-Bernstein calibration bounds account for both uncertainty layers. Maximum-risk differences are obtained by subtracting maximum-risk intervals, not by maximizing cellwise paired gains. The optimized rule has a lower grid maximum MAE in every pair. Its improvement over clipping has a positive interval in three pairs; automobile/truck remains unresolved, with interval \([-.0030,.1032]\). Comparisons with constant, empirical-root and raw-moment baselines have positive improvement intervals in all four pairs. These are finite-grid comparisons, with no claim of pointwise dominance.

**Dynamics diagnostics and scope.** There are three censored reference paths and no empirical-root failures. The maximum point-estimate discrepancy between actual right-risk-exit probability and \(\Phi(t)\) is \(.01046\). Initial-safe-component exit and actual risk exit have different times on 18,378 paths, but no direction disagreement among paths with observed risk exits; censored paths remain in the bounds. The optimized rule trades endpoint accuracy for a smaller central peak. This confirms a reproducible prediction improvement on the declared finite visual grid. It does not validate shared risk on images, unknown-baseline robustness, corruption transfer, continuous-window finite-\(n\) minimax optimality or exit-time accuracy.

| Pair | M | Moment | Clipped | Optimized | Clipped minus optimized | Simultaneous interval |
|---|---:|---:|---:|---:|---:|---|
| Cat/dog | 17625 | 0.2530 | 0.2266 | 0.1682 | 0.0585 | [0.0052, 0.1117] |
| Automobile/truck | 48557 | 0.2552 | 0.2288 | 0.1787 | 0.0501 | [-0.0030, 0.1032] |
| Deer/horse | 38254 | 0.2536 | 0.2276 | 0.1676 | 0.0600 | [0.0068, 0.1129] |
| Airplane/ship | 20213 | 0.2502 | 0.2246 | 0.1680 | 0.0566 | [0.0036, 0.1095] |

![Independent visual confirmation: probability MAE across hidden environments](../experiments/results/local_confirmation/risk.png)

The table reports five-point maximum MAE; plotted curves are point estimates. Full protocols, per-cell intervals, censoring and retained arrays are documented in the [visual confirmation analysis](../experiments/results/local_confirmation/analysis.md); the independent exploration is documented in [local_scaling](../experiments/results/local_scaling/analysis.md).

### 7.9. Natural corruptions: independent direction-calibration confirmation

We retain the direction target, not the probability of any deterioration. On natural corruptions the target is the unconditional event P(T<infinity, x_T>x0); conditioning on finite exit requires a separate justification. Complete labeled risk geometry is used only for offline applicability and evaluation, not as an unlabeled deployment detector.

Using frozen ResNet-18 scores for four CIFAR-10 class pairs, we consider clean and six CIFAR-10-C corruption conditions, retaining the source projection and initialization. Before new sampling, complete empirical risk enumeration fixes 22 environments with an attainable unsafe state; six environments have no unsafe state and are retained separately. Geometric attainability does not prove almost-sure dynamical exit. Each selected environment has 1,024 fresh reference paths, a 4,000-update cap, and 256 independent calibration repetitions at M=2,048/16,384/131,072. Budgets share nested prefixes within repetitions. The fixed-pool sampling protocol counts replacement draws, not new independent images. This is a controlled binary-threshold study on standard corruption data, not a full multiclass Tent evaluation.

The primary predictor estimates the target entropy root and noise scale from unlabeled calibration scores. The sole primary comparison is equal-environment mean MAE at the lowest minus highest budget. Conditional empirical-Bernstein bounds for paired losses and simultaneous censor-compatible binomial reference bounds jointly give at least 95% coverage. Average MAE decreases from .108344 to .047292 and .016906. The gain is .091438, with interval [-.037798,.220674]: the independent confirmation remains unresolved. No paths are censored. The calibration and reference contributions to the gain radius are .101054 and .028182; the run used the illustrative 1,024/256 budgets, without a prior precision calculation.

Offline displacement strata contain 2, 3 and 17 environments for absolute t at most 1, between 1 and 3, and above 3. Their low/high-budget mean MAEs are .369887/.103069, .361697/.054515 and .032865/.000132. This descriptive pattern is compatible with greater difficulty near ambiguous directions, but does not establish a universal difficulty law. Target noise ratios are also reported offline. Exploration found almost identical errors for untrimmed source-influence moments and target-root estimates, so no unique benefit of nonlinear root estimation is claimed. A fixed K=1 Gaussian rule cannot output outside [.158655,.841345]; its transfer error on extreme probabilities does not refute the local theorem. Full protocols, failures and the six no-exit environments remain in the [confirmation analysis](../experiments/results/corrupt_confirmation/analysis.md).

### 7.10. Bounded nonaligned local directions

We check the direction-free prediction interface beyond the original tilt using k=t*psi/c^2+a*(tanh(z)^2-E0 tanh(Z)^2), with t in {-1,0,1}, a in {0,-.5,.5}, and n in {128,512}. The even centered perturbation is orthogonal to the odd influence function, so it preserves the first-order target t while changing the environment. All members retain the same signed label density and complete risk; continuous rejection sampling drives the actual nonlinear SGD. The predictor uses only the baseline influence and independent calibration scores, with no direction or generating coordinate input.

The 18 environments each have 1,024 reference paths and 256 calibration repetitions at each of three relative noise levels. All 18,432 paths exit within 1,500 updates. The maximum point-estimate discrepancy from Phi(t) is .023962. All 18 simultaneous reference intervals contain Phi(t), which is not an equivalence test. Across each batch and nuisance amplitude, averaging over the three targets gives MAE ranges .297-.321, .202-.212 and .111-.120 for r=2,1,.5. These are descriptive mechanism checks. They do not test an infinite direction class, compare optimized rules, or certify finite-n minimax optimality. Geometry quadrature and slope grids are diagnostics rather than rigorous bounds. See the [direction analysis](../experiments/results/unknown_directions/analysis.md).

### 7.11. Full-network BN: testing the bridge to scalar dynamics

A CIFAR-10 ResNet-18 with all 9,600 BN affine parameters adapted by current-batch entropy gradients was checked on Gaussian noise, defocus blur and contrast at severity 3. The 500-update development design uses 12 trajectories per environment, batch size 64, fixed SGD learning rate .001 and a two-percentage-point finite-horizon risk-increase event. Risk is measured on the complete official test block with a separately randomized evaluation grouping held fixed within each trajectory. This differs from the scalar eventual direction target.

None of the 36 trajectories triggered the event. At update 500, mean error changes were -4.942, -.520 and -1.150 percentage points, respectively; mean risks continued improving at the recorded checkpoints. Zero events among 12 runs per environment do not exclude rare deterioration. The global first axis explained 74.11% of squared displacements from initialization, with .99494 absolute cosine between axes estimated from alternating runs. To determine whether this represents the stochastic mode required by the scalar mechanism, we separate each environment's displacement at each update into its across-run mean and residual. The two components are orthogonal in the sum of squared norms over runs and updates.

| Environment | Mean-path energy / total | Global-axis residual energy / residual total | Residual-axis split cosine |
|---|---:|---:|---:|
| Gaussian noise | 92.39% | 0.56% | .034 |
| Defocus blur | 90.11% | 0.90% | .011 |
| Contrast | 96.56% | 2.18% | .090 |

Energy fractions use all 501 recorded states per run. Residual axes use updates 25,50,...,500, centering each update separately within the fitted run group. Their in-sample first-axis fractions are 9.21%, 9.55% and 11.90%; fitting on alternating runs and scoring on the other half yields only .12%–.60% residual energy across the six train/test directions. Thus the stable uncentered direction primarily describes common mean motion; it does not establish a reproducible dominant stochastic direction. These are descriptive checks with 12 runs in 9,600 dimensions, not a proof that all stochastic structure is absent. Residual energies increase by factors 8.76, 9.52 and 9.57 between updates 50 and 500, compared with a tenfold time increase. This does not display clear accelerated spread, but does not establish local stability or exclude weaker unstable modes.

The development quadratic risk map has a maximum increase of .0797 percentage points over all scalar coordinates, below the two-point event threshold. Its probability forecast is therefore structurally zero, regardless of the fitted scalar dynamics or calibration budget. This is a limitation of the fitted prediction interface, distinct from the absence of observed events; zero Brier loss would not validate the mechanism. Development labels were used to fit that map. Auxiliary risk regressions show some association with small risk fluctuations, but use a projection estimated from all development runs and do not validate prospective event prediction.

The theoretical bridge consequently has two complementary forms of evidence. Controlled scalar experiments test the exit approximation and prediction rules where the unstable-root structure is present. This full-network applicability check shows that a stable mean direction alone does not supply the stochastic reduction or risk-event link needed to transfer those conclusions to BN-affine Tent. Under this configuration and horizon, those links remain unestablished; this limits direct extrapolation rather than contradicting the conditional scalar theorems. This supplies an applicability boundary rather than a validation of full-network risk prediction. No finite-baseline optimality or general Tent safety claim follows. See the [500-step analysis and numerical decomposition](../experiments/results/bn_extended/analysis.md) and [initial development](../experiments/results/bn_development/analysis.md).

## 8. Related work and contribution boundary

POEM [1, §3.2] already exhibits scalar Gaussian entropy collapse; imbalance-driven deterioration and stabilization are established [2]. Sequential risk monitoring [3] and unlabeled action identifiability [4] have different observation timing and targets. Our full risk function is identical across environments, while score marginals differ locally and the target is an independent future path probability. Merely changing assumptions is not the contribution; the quantitative shared-risk prediction result is.

The random-seed amplification mechanism is classical unstable-exit theory [5]. Experiment comparison and bounded-loss transfer [6], influence-aligned least-favorable directions [7], and posterior medians under absolute loss are classical tools. The new model-specific objects are the legal common-risk family, the uniform real-SGD exit interface and the resulting information/precision characterization. The [focused literature table](RELATED_WORK.md) records the comparisons and the limits of its source audit; it is not an exhaustive priority claim.

## 9. Limitations and discussion

The scalar update adapts one threshold on a frozen visual projection using actual entropy gradients. It retains batch randomness, local instability and labeled risk exits, but is not an established reduction of BN-affine Tent updates. A right exit describes the direction of threshold bias, not inevitable eventual class collapse. The full-network check in Section 7.11 supplies an empirical boundary: a highly reproducible mean direction coexists with poorly transferring residual axes and improving risk. It therefore supports neither a dominant one-dimensional stochastic mode nor the declared risk-event forecast. This applicability result distinguishes evidence for the scalar mechanism under controlled conditions from evidence needed for its transfer; it does not refute a theorem whose unstable-root assumptions have not been established in the tested network.


Sections 2–6 first analyze a known baseline. Section 6.5 allows an unknown baseline in a fixed regular density neighborhood with sample-based initialization and estimated constants, for L/n tending to a positive finite limit or infinity. Neither experiment covers an arbitrary unknown target distribution. The envelope is uniformly bounded, perturbations are of order \(n^{-1/2}\), the seed window is fixed, and the least-favorable subfamily must fit in that envelope. Step size and risk tolerance stay fixed and positive. Momentum, schedules, temporal dependence, batch reuse, online trajectory information and feature learning require separate analysis.

The minimax transfer is asymptotic, with no numerically usable bound on experiment deficiency or the exit remainder. The Gaussian intervals certify a limiting decision value; the confirmation intervals certify finite-grid comparisons in the stated simulation experiment. Neither certifies continuous-window finite-n minimax optimality. The optimized-rule confirmation does not test nuisance directions; the separate finite-direction check tests the raw moment interface, and a known risk function is an analytical control, not an unlabeled risk-estimation method.

The original Gaussian geometry and exit-time results remain in [gaussian_model.md](gaussian_model.md) and [gaussian_proofs.md](gaussian_proofs.md). Earlier frozen-feature studies and failed stopping tests remain in [experiments.md](experiments.md) with their individual limitations. They motivate calibration sensitivity but do not enlarge the scope to general neural-network adaptation. The present work supplies a prospective probability-prediction theory and a model-specific validation; it does not supply a general unlabeled stopping algorithm.

The visual confirmation conditions on a known official-test empirical baseline and independent draws with replacement from fixed image pools. Its environment-specific weighted label risk differs across tilts, so it is a mechanism and prediction validation rather than a test of the common-risk construction. Automobile/truck does not have a positive simultaneous improvement interval over clipping; no pair was excluded or given extra confirmation budget. Natural corruptions now have an independent direction-calibration comparison, whose primary interval crosses zero. Finite-baseline theoretical guarantees have not been confirmed with actual-SGD or visual experiments.

## References

[1] Y. Bar, S. Shaer and Y. Romano. *Protected Test-Time Adaptation via Online Entropy Matching: A Betting Approach*. NeurIPS, 2024. [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/9b35a0a20d617dc68ae98a7a57df2f51-Paper-Conference.pdf).

[2] T. Weerasinghe et al. *Logit-Adjusted Test-Time Adaptation under Partial Class Imbalance*. WACV, 2026. [Paper](https://openaccess.thecvf.com/content/WACV2026/html/Weerasinghe_Logit-Adjusted_Test-Time_Adaptation_under_Partial_Class_Imbalance_WACV_2026_paper.html).

[3] M. Schirmer et al. *Monitoring Risks in Test-Time Adaptation*. NeurIPS, 2025. [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/746960ad49ddb47248970a0e1404230c-Paper-Conference.pdf).

[4] K. Jhawar and L. Wang. *When Is Test-Time Adaptation Identifiable from Unlabeled Evidence?* arXiv:2609.11235, 2026. [Preprint](https://arxiv.org/html/2609.11235v1).

[5] Y. Bakhtin. *Exit asymptotics for small diffusion about an unstable equilibrium*. Stochastic Processes and their Applications, 118(5):839–851, 2008. [DOI](https://doi.org/10.1016/j.spa.2007.06.003).

[6] M. Nussbaum. *Asymptotic Equivalence of Statistical Experiments*. [Author manuscript](https://pi.math.cornell.edu/~nussbaum/papers/04-1-transl.pdf).

[7] M. van der Laan and S. Gruber. *One-Step Targeted Minimum Loss-based Estimation Based on Universal Least Favorable One-Dimensional Submodels*. International Journal of Biostatistics, 12(1):351–378, 2016. [DOI](https://doi.org/10.1515/ijb-2015-0054).

[8] A. Maurer and M. Pontil. *Empirical Bernstein Bounds and Sample Variance Penalization*. COLT, 2009. [Paper](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf).

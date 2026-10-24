# When Does Entropy Adaptation Stop Helping?
## Risk Rebound and Safe Horizons under Mixed Feature and Prior Shift

Working draft — 24 October 2026. This document separates proved elementary statements, numerical observations, and research targets. No novelty or general neural-network guarantee is claimed.

### Abstract

Entropy minimization can improve a deployed classifier before making it worse than its frozen source state. We study the duration of this improvement using actual entropy-gradient updates of a decision threshold in an anchored Gaussian feature mixture with simultaneous translation and class-prior shift. We derive the population drift and classification-risk geometry and formulate the first crossing of source risk as the central dynamical quantity. A reproducible pilot exhibits a population trajectory whose risk initially decreases and subsequently exceeds the source risk while prediction entropy decreases. We give an elementary finite-horizon tracking bound and a conditional risk certificate to specify what further results must improve upon. A structural drift phase diagram, sharp stochastic crossing-time estimates, and a useful label-free stopping rule remain open tasks in this project. The present results motivate investigation; they do not establish a new general solution to safe test-time adaptation.

### 1. Introduction

A camera can undergo changes in illumination and in the frequency of observed objects at the same time. A threshold learned under a balanced source distribution is then affected by two mechanisms: a common displacement of features and a change in the Bayes decision threshold induced by the target class prior. Entropy adaptation responds to the marginal feature distribution, whereas classification risk depends on the class-conditional assignment. Its preferred direction can therefore change relative to the supervised objective along the optimization path.

This project asks how many actual updates remain safe relative to the deployed source classifier. Existing work already studies collapse, class imbalance, early stopping, and information limitations; these are not new problem statements. Our candidate distinction is quantitative: derive a nonempty parameter region and crossing-time law from the state-dependent entropy recursion, without replacing it by mean recentering or assuming task-gradient alignment throughout the trajectory. Detailed positioning is in [RELATED_WORK.md](RELATED_WORK.md).

Proposed contributions, conditional on future proof: (i) a structural phase diagram for improvement, safe convergence, and risk rebound; (ii) finite-step and finite-batch bounds that remain informative up to the crossing time; (iii) a stopping certificate with explicit information assumptions and demonstrable nonzero improvement. Only the elementary setup and baseline lemmas below are currently derived.

### 2. Anchored model and evaluation

Let Y take values ±1 with P(Y=1)=π, and

\[
 Z=\delta+Ym+\sigma\xi,\qquad \xi\sim N(0,1),\quad m,\sigma,w>0.
\]

The anchor supplies the positive class orientation, m and σ; these are assumed known initially. The feature translation δ and target prior π are unknown. Fix π∈[π_min,1−π_min], π_min>0, and a bounded translation interval when studying uniform guarantees. The deployed classifier is sign(Z−b_0), b_0=0. Its confidence is p_b(z)=sigmoid(w(z−b)). Only b adapts; the positive slope w and feature extractor remain fixed. This is a controlled binary threshold model, not a complete BN network or Tent implementation.

At step t receive a fresh independent batch of n features. With binary prediction entropy H,

\[
 b_{t+1}=b_t-\eta\widehat g_t(b_t),\qquad
 \widehat g_t(b)=\frac1n\sum_i\partial_b H(p_b(Z_{t,i})).
\]

Evaluate R(b_t) on an independent fresh target draw. Labels used to compute synthetic risk are unavailable to the update or a proposed stopping algorithm. Reusing a fixed dataset or introducing dependent/nonstationary streams changes the analysis and is deferred.

The central quantity is

\[
 T_\epsilon=\inf\{t\geq0:R(b_t)>R(b_0)+\epsilon\},
 \quad\inf\varnothing=\infty.
\]

This is the first unsafe iterate. A stopping rule must reject a proposed unsafe update before deploying it. Average risk, training-batch accuracy and cumulative online regret are different metrics.

### 3. Exact identities and elementary baseline results

**Proposition 1 (entropy drift).** Write u=w(z−b), s(u)=sigmoid(u), a(u)=s(u)(1−s(u)). Then

\[
 \partial_bH(p_b(z))=w u a(u),\qquad
 g(b)=w^2\mathbb E[(Z-b)a(w(Z-b))].
\]

The population algorithm is \(\bar b_{t+1}=\bar b_t-\eta g(\bar b_t)\), with the same initial threshold. Proof: dH(s(u))/du=−u a(u) and du/db=−w. In particular the sign in the threshold update differs from a feature-offset parameterization; implementations must preserve this distinction.

**Proposition 2 (risk geometry).** With Φ and φ the standard normal CDF and density,

\[
R(b)=\pi\Phi((b-\delta-m)/\sigma)
+(1-\pi)\Phi((\delta-m-b)/\sigma),
\]

\[
R'(b)=\sigma^{-1}\{\pi\phi((b-\delta-m)/\sigma)
-(1-\pi)\phi((b-\delta+m)/\sigma)\}.
\]

There is a unique finite minimizer

\[
b^*=\delta+\frac{\sigma^2}{2m}\log\frac{1-\pi}{\pi}.
\]

Proof: the ratio of the positive to negative terms in the derivative is \(\pi(1-\pi)^{-1}\exp(2m(b-\delta)/\sigma^2)\), strictly increasing from zero to infinity. Thus R decreases left of b* and increases right of b*. The limits are R(−∞)=1−π and R(+∞)=π. A first-order population update improves risk where R'(b)g(b)>0; finite steps require a remainder bound. π shift can improve the optimal threshold even when δ=0, so a zero-translation rebound example is meaningful but does not alone validate a mixed-shift theory.

**Lemma 3 (a conservative finite-horizon baseline).** The sample gradient satisfies |∂_bH|≤M=w/e, since a(u)≤exp(−|u|). Its derivative in b is bounded by

\[
 L=w^2(1/4+1/e),
\]

because ∂_b(wua)=−w²a[1+u(1−2s)] and |u|a≤1/e. Consequently g is globally L-Lipschitz. For deterministic T≥1 and α∈(0,1), define

\[
q=M\sqrt{2\log(2T/\alpha)/n},\quad
e_t=q\{(1+\eta L)^t-1\}/L.
\]

With probability at least 1−α, simultaneously for t≤T,
\[
 |b_t-\bar b_t|\leq e_t,\qquad
 |R(b_t)-R(\bar b_t)|\leq e_t/(\sigma\sqrt{2\pi}).
\]

Proof: conditional on the past, b_t is fixed and the new batch is independent. Hoeffding's inequality for gradients in [−M,M] and a union bound give |ĝ_t(b_t)−g(b_t)|≤q at all T updates. Iterating d_{t+1}≤(1+ηL)d_t+ηq proves the parameter bound. The risk derivative has absolute value at most 1/(σ√(2π)), giving the risk bound. This is a standard loose baseline, not a candidate principal contribution. Exponential growth can make it vacuous before the empirical rebound; a useful theorem must exploit local contraction or signed drift.

**Conditional certificate, not a completed algorithm.** Suppose nested, label-free confidence sets C_t cover the fixed (δ,π) simultaneously with probability ≥1−α under the anchored model. Given the proposed b', accept only if

\[
 \sup_{\theta\in C_t}\{R_\theta(b')-R_\theta(b_0)\}\leq\epsilon.
\]

Otherwise stop and retain the last accepted threshold. On the coverage event every accepted iterate is safe, regardless of the data dependence of the proposal. This follows directly from the supremum condition and is generic. No confidence-set construction, practical acceptance rate or improvement guarantee is established here. Ordinary fixed-time intervals reused after repeated inspection do not suffice.

The anchored two-component family has additional information beyond an arbitrary observation channel: known ordered means and shared variance substantially constrain the label assignment. Thus distinguishing δ and π is not automatically impossible. The hard practical issue is robustness to anchor error and unknown components; any impossibility statement must specify those enlarged families. We do not repackage existing observational-equivalence results as a new theorem.

### 4. Intended main result and proof obligations

**Target A — structural drift and rebound.** Find explicit inequalities in (m/σ,wσ,δ/σ,π,η), on a nonempty open parameter set including δ≠0, that guarantee an initial strict risk improvement, eventual source-risk crossing by a positive margin, and decreasing entropy along the population iterates. Classify stable entropy stationary points as well: separated mixtures may attract a useful low-density threshold and need not collapse. A proof that merely assumes a favorable sign for g over the desired interval would not explain the phase boundary.

**Target B — first crossing time.** On a monotone population corridor, bound the discrete crossing using its distance and the extrema of |g|. In gradient-flow time the candidate integral is ∫db/(−g(b)), with orientation along the path; this alone is elementary. The substantive goal is explicit parameter dependence near drift zeros and stochastic first-passage fluctuations that distinguish batch size, step size and separation. Prove that the tracking horizon includes the crossing; Lemma 3 does not establish that.

**Target C — useful observable stopping.** Build a sequential uncertainty set under declared anchor knowledge; prove safe acceptance and a strictly positive improvement on a parameter subset separated from the ambiguity boundary. Account for the proposal evaluation and stopping with the same observations. Compare with a one-time calibration batch and frozen estimates, not only with an unconstrained entropy baseline. Unknown anchor errors must appear explicitly in the statement. A universal source-risk guarantee without an information condition is not a reasonable claim.

### 5. Pilot observations

See [experiment record](../experiments/RESULTS.md) and saved arrays. A 30-case grid yields eight population paths with a reduction exceeding .001 followed by risk above the source by .005 within 2500 updates. The illustrated case is selected from the grid and is exploratory: δ=0, π=.35, m=σ=1, w=2, η=.1. Risk starts at .158655, reaches .147598 at step 47, first exceeds source risk plus .005 at step 94, and approaches .35. Entropy decreases from .356316 to .001851. Forty independent fresh-batch SGD trajectories with n=64 show the same late failure in this setting. These simulations support existence checks, not uniform statements or visual-benchmark performance.

### 6. Visual validation plan and decision criteria

Freeze an image encoder; choose two classes and fit the source direction once. Hold the slope fixed and adapt only the scalar threshold. Manipulate held-out stream class priors and common projected feature shifts separately, then jointly. Use target labels only for evaluation. Compare source, mean recentering, population/oracle thresholds, entropy updates, and the eventual stopping rule. Report full trajectories, first crossings, classwise error and useful update counts across held-out seeds. Add image-space corruption only after testing whether its projected class conditionals satisfy the model approximately.

The source anchor requires access to stored source calibration statistics; this differs from strict source-free deployment with weights alone. Measure sensitivity to misestimated separation, variance and orientation. Do not infer full-network guarantees from scalar fits. Stop this line if its theorem reduces to known collapse analysis plus a generic confidence-set filter, if robust finite-sample bounds are vacuous, or if frozen visual representations do not exhibit the predicted phase distinction.

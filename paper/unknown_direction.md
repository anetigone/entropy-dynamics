# Unknown local directions, nuisance, and the least favorable entropy tilt

Part of the [manuscript](DRAFT.md), continuing [the common-risk construction](local_prediction.md) and [the Gaussian decision reduction](gaussian_decision.md). These are supplied proofs for a known baseline and uniformly bounded local perturbations. They do not cover an unknown baseline, arbitrary fixed distribution shifts, or joint vanishing-step limits.

## 1. The model-specific influence function

Use the same baseline Q0, fixed x0=0, safe interval [-b,b], fixed positive eta, and entropy update U as in L.1–L.4. Write

\[
U_0(z)=U(0,z),\quad \lambda=E_0\partial_xU(0,Z)>0,
\quad \sigma_g^2=E_0U_0^2>0,
\]
\[
s^2=\frac{\eta^2\sigma_g^2}{(1+\eta\lambda)^2-1},
\qquad \psi(z)=\frac{U_0(z)}{\lambda s},
\qquad c^2=E_0\psi^2=\frac{2+\eta\lambda}{\eta\lambda}.
\tag{N.1}
\]

The notation psi is distinct from the logistic variance function a used in U. It is the influence function of the normalized root displacement -q(Q)/s at Q0, where s is its fixed baseline value.

### Proposition N.1. Root sensitivity per unit Fisher information

Let h be any bounded, mean-zero score with I_h=E0 h^2>0. Along f_epsilon=f0(1+epsilon h),

\[
q'_h(0)=-\frac{E_0[U_0h]}{\lambda},
\qquad d_h:=\left.\frac{d}{d\epsilon}\frac{-q(f_\epsilon)}s\right|_0
=E_0[\psi h].
\tag{N.2}
\]

The Fisher information for epsilon in one unlabeled observation is I_h, and

\[
\boxed{\frac{d_h^2}{I_h}\le c^2,}
\tag{N.3}
\]

with equality precisely when h is a nonzero scalar multiple of U0 in L2(Q0).

**Proof.** Differentiate E_epsilon U(q_epsilon,Z)=0. At epsilon=0 the state derivative is lambda and the distribution derivative is E0 U0 h, proving (N.2). Differentiating log f_epsilon at zero gives score h and information E0 h^2. Cauchy–Schwarz gives (E0 psi h)^2<=E0 psi^2 E0 h^2. Its equality condition proves the last assertion. □

Thus the original h0=U0/sigma_g is not merely a convenient tilt: it maximizes normalized seed displacement for a given local statistical distinguishability. The term “least favorable” here concerns estimation of this target, after fixing the target window; it is not an ordering of all nonlinear environmental shifts.

If theta=u/sqrt(n) and the score h is known, its limiting seed is d_h u, while the calibration information is rho I_h. At a fixed seed window t=d_h u in [-K,K], the resulting Gaussian noise standard deviation is

\[
r_h=\frac{|d_h|}{\sqrt{\rho I_h}}\le\frac c{\sqrt\rho}.
\tag{N.4}
\]

For d_h=0 the first-order seed is zero, so this reparameterization is not applicable; a second-order analysis would be a different local problem. Comparing different h at a fixed u-window would change the attainable target window and would not be the same least-favorability comparison.

## 2. An infinite-dimensional family with an unknown direction

Fix finite B_loc>0 and K>0, and define

\[
\mathcal K_{B_{\rm loc},K}
=\{k:E_0k=0,\ \|k\|_\infty\le B_{\rm loc},\ |E_0\psi k|\le K\}.
\tag{N.5}
\]

For each n and each unknown k in this class let

\[
f_{n,k}=f_0(1+k/\sqrt n),\qquad t(k)=E_0\psi k.
\]

The predictor knows f0, U, baseline constants, B_loc and K. It does not know k, t(k), the true root, or future SGD samples. The class contains arbitrary measurable bounded perturbation directions, not just a known finite list or a fixed parametric span. It is legitimate for the worst-case k to vary with n.

Keep the same signed density g(z)=kappa tanh(z)f0(z) and joint law

\[
p_{n,k}(z,\pm1)=\tfrac12(f_{n,k}(z)\pm g(z)).
\]

When B_loc/sqrt(n)<=min(1/2,1-kappa), these are legal normalized densities. Every member shares the complete risk function and exactly the same global safe interval [-b,b]. Equal class priors also remain unchanged. The fixed source information and initialization remain identical across the enlarged class.

### Lemma N.2. Uniform real-SGD exit law over unknown directions

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

Thus the entire class lies in one shrinking smooth-field neighborhood. No derivative of k with respect to z is needed. The baseline derivative lower bound and endpoint drift signs persist uniformly. Since v_{n,k}(0)=E0 U0 k/sqrt(n), Taylor expansion at zero and the common positive derivative give q=O(n^{-1/2}) and the root expansion (N.6). Bounded derivatives and bounded U^2 give the remaining expansions. The baseline root noise is nonzero, so its uniform lower bound persists.

For completeness, the probabilistic part can be repeated with exactly the uniform constants of L.2 rather than applying a fixed-law theorem. All maps have inverse contraction and C2 conjugacy bounds in the same smooth neighborhood; their root-to-boundary distances and conjugate exit magnitudes are bounded away from zero. Bounded U and its derivative give common batch second/fourth moments and the state-difference noise bound. For the stopped normalized conjugacy Y and frozen-root seed W, the same orthogonality and Taylor estimates yield

\[
\sup_{k,j}E|Y_j|^2\le C,\quad
\sup_k E A_{n,k}^{-2\tau}\le C/n,\quad
\sup_k E|Y_\tau-W|^2\le C(1+\log n)/n.
\]

Exit is finite for every member because f_{n,k}>=f0/2 supplies a common positive mass in each of the fixed forcing intervals. Frozen seed columns have uniformly bounded third moments and variances bounded below, so Berry–Esseen and seed anti-concentration hold uniformly. Choosing the same anti-concentration cutoff as in L.2 bounds the error of identifying the final exit sign by Ce_n. Finally,

\[
\sqrt n H_{n,k}(0)/s_{n,k}=t(k)+O(n^{-1/2})
\]

uniformly by (N.6) and the normalized C2 bound. The global risk-safe interval is unchanged, so this is actual risk exit without a component-transfer correction. This proves (N.7). □

The proof uses a uniform bounded envelope and a finite number of smooth U derivatives. It does not require entropy or covering-number control of the function class of directions, since no uniform empirical process over k is estimated.

## 3. A direction-free estimator and the minimax value

Define the scalar statistic from independent calibration scores

\[
\widehat t=\frac{\sqrt n}{M}\sum_{j=1}^M\psi(\widetilde Z_j).
\tag{N.8}
\]

It is available without knowing k. Under Q_{n,k},

\[
E\widehat t=t(k),\qquad
\operatorname{Var}(\psi(\widetilde Z))
=c^2+n^{-1/2}E_0[\psi^2 k]-t(k)^2/n
=c^2+O(n^{-1/2})
\tag{N.9}
\]

uniformly. Boundedness of psi gives a uniform triangular-array CLT. For M/n->rho in (0,infinity), the distribution of hat t is uniformly approximated in Kolmogorov distance by N(t(k),c^2/rho).

Let R^K_{n,M} be the minimax expected absolute prediction loss over (N.5), using the exact pi_n as the target. Let

\[
V_G(K,r)=\inf_d\sup_{|t|\le K}E|d(t+rZ)-\Phi(t)|.
\]

### Theorem N.3. Unknown-direction local minimax equivalence

Assume

\[
B_{\rm loc}\ge K\|\psi\|_\infty/c^2.
\tag{N.10}
\]

Then, for M/n->rho in (0,infinity),

\[
\boxed{R^{\mathcal K}_{n,M}\longrightarrow V_G(K,c/\sqrt\rho).}
\tag{N.11}
\]

Uniformly consistent prediction over this enlarged class is possible if and only if M/n->infinity.

**Proof of the lower bound.** The unknown-direction class contains the subfamily

\[
k_t=(t/c^2)\psi=(t/c)h_0,\qquad |t|\le K,
\]

by (N.10). Its seed coordinate is exactly t, and it is the original entropy-gradient tilt with theta=t/(c sqrt(n)). Restricting to this subfamily and even telling the predictor that this restriction holds can only reduce the minimax risk. The experiment-level Gaussian reduction already proved for that subfamily gives liminf R^K>=V_G(K,c/sqrt(rho)). This invokes a full-experiment lower bound, not a lower bound inferred from a moment CLT.

**Proof of the upper bound.** Any fixed bounded Lipschitz Gaussian rule d, applied to hat t, has uniformly convergent risk by (N.9), its uniform CLT, and (N.7). This suffices to approach V_G arbitrarily closely. To see why arbitrary measurable Gaussian rules do not create a gap, start with an epsilon-optimal rule valued in [0,1]. Smooth it by d_a(x)=E[d(x+aZ')], with independent standard-normal Z'. The smoothed rule is bounded Lipschitz. Jensen gives

\[
E_t|d_a(X)-\Phi(t)|
\le E_t|d(X+aZ')-\Phi(t)|.
\]

The last expectation differs from that under X~N(t,r^2) by at most TV(N(0,r^2+a^2),N(0,r^2)), uniformly in t. This tends to zero as a->0. Thus bounded Lipschitz rules are sufficient to approximate the Gaussian minimax value. Apply their uniform risk convergence, then let a and epsilon tend to zero. The upper and lower bounds agree.

For sufficiency when M/n->infinity, use Phi(hat t). Its error is bounded uniformly by

\[
\frac1{\sqrt{2\pi}}\sqrt{\frac nM}
\sup_k\sqrt{\operatorname{Var}_{n,k}\psi}+Ce_n\to0.
\]

For necessity, a bounded-ratio subsequence contains the original nondegenerate least-favorable subfamily, to which the two-point lower bound applies. □

This theorem equates a particular decision problem's minimax value. It does **not** assert that the entire infinite-dimensional experiment is equivalent to a scalar Gaussian observation: the full sample may contain information about orthogonal nuisance directions. The lower bound comes from a subexperiment and the upper bound from a direction-free rule.

If M/n->0, the same subfamily lower bound and the constant forecast upper bound also give R^K_{n,M}->Phi(K)-1/2. This extends the earlier zero-information constant to the enlarged class, without characterizing the entire nuisance experiment.

The fixed finite step rules already computed for the Gaussian problem also transfer uniformly: their finitely many discontinuities have vanishing neighborhood probability under the limiting normal densities. Consequently the existing numerical minimax brackets are relevant to the enlarged local class without rerunning the optimizer or knowing k.

## 4. What nuisance changes, and what it does not

### Proposition N.4. Tangent-space projection

For a finite collection of bounded centered scores H=(h1,...,hd) with nonsingular information I=E0 HH^T, let d=E0[H psi]. The usual local variance associated with estimating the seed functional in this score span is

\[
\frac1\rho d^\top I^{-1}d
=\frac1\rho\|\Pi_{\operatorname{span}(H)}\psi\|_2^2
\le\frac{c^2}{\rho}.
\tag{N.12}
\]

**Proof.** The projection has coefficients I^{-1}d, so its squared L2 norm is d^T I^{-1}d. Projection contracts norm. Equivalently, an empirical mean of d^T I^{-1}H estimates the derivative functional in this local model with that first-order variance. □

For a closed admissible tangent space T, the corresponding sensitivity supremum is

\[
\sup_{h\in T,\,\|h\|_2>0}
\frac{\langle\psi,h\rangle^2}{\|h\|_2^2}
=\|\Pi_T\psi\|_2^2.
\tag{N.13}
\]

Equality as an operational bounded density tilt requires that the maximizing direction be admissible and bounded, or be approximable by admissible bounded scores. Equation (N.13) alone does not supply the geometry and uniform exit proof for a new model class.

If an interest score h is accompanied by nuisance scores spanning N, and those nuisance directions preserve the seed functional to first order, then N is orthogonal to psi. The efficient interest score is h_eff=h-Pi_N h and

\[
\frac{\langle\psi,h\rangle^2}{\|h_{\rm eff}\|_2^2}
=\frac{\langle\psi,h_{\rm eff}\rangle^2}{\|h_{\rm eff}\|_2^2}
\le c^2.
\tag{N.14}
\]

Thus nuisance can reduce information relative to a restricted one-direction model, but cannot exceed the full tangent variance c^2 in this known-baseline first-order problem. In particular, for h=h0 proportional to psi and root-preserving nuisance, h_eff=h0: there is no additional first-order penalty.

If a proposed nuisance direction has E0 psi k nonzero, it changes the target and is not a target-preserving nuisance. It must be included in t(k), as in Theorem N.3, rather than projected out while keeping the target fixed. Moreover, a bounded multidimensional parameter window can permit decision improvements beyond a scalar variance comparison; (N.12) is an information calculation, not a general finite-window minimax formula.

**Concrete nuisance example.** At the symmetric Gaussian baseline, let b(z)=cos(z)-E0 cos(Z). It is bounded, centered and even, whereas psi and h0 are odd. Thus E0 psi b=E0 h0 b=0. The two-direction perturbation k_{t,v}=(t/c)h0+v b has seed t, with an unknown bounded v changing other aspects of the score distribution. It stays in (N.5) whenever |t| H_*/c+|v| ||b||_infinity<=B_loc. This supplies a legal common-risk nuisance family with no additional first-order penalty. No new nuisance simulation is claimed in the present confirmation run.

## 5. Contribution and boundary

The Cauchy–Schwarz and tangent-space projection principles are classical semiparametric efficiency geometry. The added model-specific content is the identification psi=U0/(lambda s), its connection to a uniform true-SGD risk-exit functional, the shared-risk realization of the bounded perturbation class, and the minimax extension beyond a known direction.

This result is local around a known Q0. If Q0, its root, lambda or s must be learned from a separate source sample, that sample's accuracy and its relation to n enter a new information budget. Arbitrary unknown baselines, perturbations outside the n^{-1/2} neighborhood, growing B_loc, online trajectory observation and eta->0 are not covered.

Classical context: van der Laan and Gruber (2016), *One-Step Targeted Minimum Loss-based Estimation Based on Universal Least Favorable One-Dimensional Submodels*, discusses scores aligned with efficient influence curves. This manuscript does not claim that principle or TMLE methodology as a new contribution. https://doi.org/10.1515/ijb-2015-0054

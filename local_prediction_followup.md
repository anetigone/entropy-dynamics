# Local prediction: proof audit and statistical refinements

Companion to the supplied `local_prediction.md`, 29 October 2026. This is an additive review and extension, not a replacement of L.1–L.4. All results below concern the explicit known scalar tilt family, fixed positive step size, fixed initialization, independent calibration samples, and independent future fresh-batch SGD. No new SGD experiment is reported.

## 1. Review verdict and scope

The supplied proofs of L.1–L.4 support the stated necessity-and-sufficiency result within the declared family. I found no counterexample or invalid change from a pointwise exit theorem to a triangular environment sequence in the supplied argument. In particular, the proof of L.2 supplies the uniform constants, controls the stopped nonlinear remainder, and handles seeds near zero before identifying the exit sign.

This review is a mathematical reading of the supplied chapter, not formal verification. The referenced scalar-exit chapter was not supplied for this review. The expansions below make the delicate uses of its conjugacy construction explicit enough to check locally. No literature-priority conclusion is made.

The following distinctions should remain in the main text:

- The family and its constants are known; the local parameter is unknown. This is a lower bound even with substantial common model information, and an upper bound only for that declared information experiment.
- The local window has fixed positive width. The theorem does not apply to the singleton centered environment or to a window shrinking faster without further analysis.
- The theorem concerns prediction of a probability, not prediction of a future realized Bernoulli branch.
- The ratio condition is for fixed eta. A joint small-step result does not follow by substitution.
- The entire global risk-safe set is one interval. There is no unaccounted component-exit/risk-exit transfer.

## 2. Suggested insertions in the proof of L.2

### 2.1. Inverse maps and positivity of the conjugacy derivative

Let K=[-d,d] and G_theta(x)=x+eta v_theta(x). The signs of v_theta at the endpoints imply G_theta(K) contains K. Thus F_theta=G_theta^{-1} is defined on K, maps K into itself, and fixes q_theta. Its derivative satisfies

\[
0<F_\theta'(x)\le \gamma:=(1+\eta\lambda/4)^{-1}<1.
\]

In centered coordinates write

\[
\widetilde F_\theta(y)=A_\theta^{-1}y+r_\theta(y),
\qquad r_\theta(0)=r_\theta'(0)=0.
\]

On a common small neighborhood, uniform smoothness gives bounded third derivatives and a bound beta on the inverse derivative, with A_max beta^2<1. If y_j=F_tilde_theta^j(y), then, uniformly in theta,

\[
|y_j|\le C\beta^j|y|,\qquad
|y_j'|\le C\beta^j,\qquad |y_j''|\le C\beta^j.
\]

For the second inequality use the derivative product. The third follows from

\[
y_{j+1}''=\widetilde F_\theta''(y_j)(y_j')^2
             +\widetilde F_\theta'(y_j)y_j''.
\]

The successive differences in A_theta^j y_j equal
A_theta^{j+1} r_theta(y_j). Their functions, first derivatives and second derivatives have summable bounds C(A_max beta^2)^j. This proves local C^2 convergence. Positivity of the limiting derivative should be justified separately: its derivative product is

\[
\prod_{k=0}^{j-1} A_\theta\widetilde F_\theta'(y_k).
\]

Each factor is positive and differs from one by O(|y_k|). On a sufficiently small common neighborhood the sum of these deviations is uniformly bounded, so the limiting product is bounded away from zero. A fixed number of inverse iterations maps all of K into that neighborhood, since F_theta contracts by gamma. Finite composition gives uniform C^2 bounds and a positive derivative lower bound on K. This also verifies the functional equation wherever its two arguments are in K.

This supplies the detail behind the phrase “the local derivative is uniformly positive by normalization and the series.” Normalization alone would not establish positivity throughout a neighborhood.

### 2.2. The stopped second moment does not use an exit-time bound

Let F_j denote the history before batch j and write I_j=1_{j<tau}. With

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

Therefore sup_t E|Y_t|^2<=C without assuming a logarithmic exit-time estimate. Finite exit and Fatou then imply E|Y_tau|^2<=C. At exit, the uniform root-to-boundary separation and positive derivative of H imply |H_theta(x_tau)|>=b_H>0. Consequently

\[
\mathbb E A_\theta^{-2\tau}\le (nb_H^2)^{-1}\mathbb E|Y_\tau|^2\le C/n.
\]

There is no circular inference from a desired exit approximation to its own moment estimate.

### 2.3. Predictability of the frozen tail

Use the convention that batch j moves x_j to x_{j+1}. At tau, all batches with indices j>=tau remain fresh. Conditional on F_tau, the frozen tail has mean zero and variance

\[
\eta^2\sigma_\theta^2\sum_{j\ge\tau}A_\theta^{-2j-2}
=s_\theta^2 A_\theta^{-2\tau}.
\]

This proves the O(1/n) tail second moment in L.5 of the supplied chapter. It is a stopped-tail independence argument, not a claim that the stopped trajectory is independent of the full frozen seed.

## 3. Additional results for insertion after L.4

Retain the notation of L.1–L.4. Let A0>0 be fixed, let

\[
r_n=\sup_{|u|\le A_0}|\pi_n(Q_{u/\sqrt n})-\Phi(cu)|\le C e_n,
\qquad B_0=\Phi(cA_0)-\tfrac12.
\]

The quantity r_n is a deterministic approximation error, not an observable calibrated confidence bound.

### Proposition L.5. Exact symmetry and calibration moments

For every legal theta,

\[
\mathbb E_\theta h=\theta,\qquad
\mathbb E_\theta h^2=1,\qquad
\operatorname{Var}_\theta h=1-\theta^2.
\tag{L.8}
\]

For every finite batch size n,

\[
\pi_n(Q_{-\theta})=1-\pi_n(Q_\theta),
\qquad \pi_n(Q_0)=\tfrac12.
\tag{L.9}
\]

In particular the moment predictor in L.4 satisfies the sharper bound

\[
\sup_{|u|\le A_0}\mathbb E_{u/\sqrt n}
 |\widehat\pi-\pi_n(Q_{u/\sqrt n})|
\le\frac{c}{\sqrt{2\pi}}\sqrt{n/M}+r_n.
\tag{L.10}
\]

**Proof.** The function h is odd and f0 is even. Therefore E0 h^3=0, and
E_theta h^2=E0 h^2+theta E0 h^3=1. The first identity was already established in L.4, giving the variance formula. In particular E|hat_theta-theta|<=sqrt((1-theta^2)/M)<=1/sqrt(M); apply the Lipschitz constant of Phi.

For symmetry, f_{-theta}(z)=f_theta(-z) and U(-x,-z)=-U(x,z). Reflect every score in a future sample array. Starting at zero, the entire reflected path is the negative of the original path. The symmetric safe interval has the same exit time and the opposite exit side. Since exit is almost surely finite, the two directional probabilities sum to one. This proves (L.9). The exact equality at theta=0 belongs to this symmetric constructed family; it should not be transferred to asymmetric visual safe sets. □

### Proposition L.6. Stronger absolute-loss lower bound

For all sufficiently large n and any 0<a<=A0, put

\[
\Delta_0(a)=2\Phi(ca)-1,\qquad \rho_n=M/n.
\]

Then

\[
\mathfrak R_{n,M}(A_0)
\ge \frac{[\Delta_0(a)-2r_n]_+}{2}
       [1-2a\sqrt{\rho_n}]_+.
\tag{L.11}
\]

Consequently, under the eventual bound and choice of a in L.3, its constant improves to

\[
\liminf_n\mathfrak R_{n,M}(A_0)
\ge [2\Phi(ca)-1]/4.
\tag{L.12}
\]

**Proof.** Let P_+,P_- be the two calibration laws and t_+,t_- their true SGD direction probabilities. Their densities can be taken relative to a common measure. For any forecast T,

\[
\begin{aligned}
\mathbb E_+|T-t_+|+\mathbb E_-|T-t_-|
&\ge\int\min(p_+,p_-)\bigl(|T-t_+|+|T-t_-|\bigr)\\
&\ge |t_+-t_-|\,[1-\operatorname{TV}(P_+,P_-)].
\end{aligned}
\]

Independent randomization can be included in the common measure and leaves TV unchanged. The maximum of the two risks is at least half their sum. L.2 implies |t_+-t_-|>=Delta0(a)-2r_n, and the KL/Pinsker calculation in L.3 gives TV<=2a sqrt(rho_n). This proves (L.11). Under TV<=1/2 the bound is Delta_n/4, rather than the Delta_n/8 obtained by reducing absolute loss to a nearest-target testing error. The original weaker bound remains valid. □

**Rate consequence with an explicit remainder.** Set a_*=min(A0,1/4) and t_{n,M}=min(1,sqrt(n/M)) for M>=1. Apply (L.11) with a=a_* t_{n,M}. Since

\[
\Delta_0(a)\ge 2c\varphi(cA_0)a,
\]

we obtain

\[
\mathfrak R_{n,M}(A_0)
\ge \frac{c\varphi(cA_0)a_*}{2}\,t_{n,M}-\frac{r_n}{2}.
\tag{L.13}
\]

Together with (L.10) and the constant predictor,

\[
\mathfrak R_{n,M}(A_0)
\le\min\left\{B_0+r_n,
\frac{c}{\sqrt{2\pi}}\sqrt{n/M}+r_n\right\}.
\tag{L.14}
\]

Thus matching order follows in sequences where e_n=o(t_{n,M}); this is not a global matching-rate theorem for arbitrarily large M. Once the estimation scale is below the exit-approximation remainder, these bounds no longer determine the rate.

### Theorem L.7. Exact minimax limit when M/n tends to zero

If M/n->0, including bounded M, then

\[
\boxed{\lim_{n\to\infty}\mathfrak R_{n,M}(A_0)
=\Phi(cA_0)-\tfrac12.}
\tag{L.15}
\]

The constant forecast 1/2 is asymptotically minimax in this regime.

**Proof.** For the upper bound, L.2 and monotonicity of Phi give

\[
\sup_{|u|\le A_0}|\tfrac12-\pi_n(Q_{u/\sqrt n})|
\le B_0+r_n.
\]

For the lower bound take a=A0 in (L.11). The calibration TV bound tends to zero and the target separation tends to 2B0, so liminf risk>=B0. These two bounds agree. If M=0 is allowed, use the identical no-observation experiment, with TV exactly zero. □

This is an exact asymptotic constant only in the regime M/n->0. It is not a solution of the positive finite-ratio minimax problem. The result strengthens the interpretation of the sample threshold: a best constant predictor may be optimal when information vanishes, while its nonzero worst-case error still prevents uniform consistency.

### Theorem L.8. Uniform risk limit of the raw moment predictor

Suppose M/n->rho in (0,infinity). Let

\[
\widehat\pi^{\rm mom}=\Phi(c\sqrt n\,\widehat\theta),
\qquad r=c/\sqrt\rho.
\]

Define

\[
D(t,r)=\mathbb E|\Phi(t+rZ)-\Phi(t)|,
\qquad Z\sim N(0,1).
\]

Then

\[
\sup_{|u|\le A_0}
\left|\mathbb E_{u/\sqrt n}
 |\widehat\pi^{\rm mom}-\pi_n(Q_{u/\sqrt n})|
-D(cu,r)\right|\longrightarrow0.
\tag{L.16}
\]

Moreover, D is even in t, is strictly decreasing for t>0 when r>0, and satisfies

\[
\boxed{\lim_n\sup_{|u|\le A_0}
\mathbb E_{u/\sqrt n}
 |\widehat\pi^{\rm mom}-\pi_n(Q_{u/\sqrt n})|
=\frac{\arctan(c/\sqrt\rho)}{\pi}.}
\tag{L.17}
\]

**Proof.** By (L.8), the standardized calibration sum

\[
Z_{n,u}=\frac{\sqrt M(\widehat\theta-u/\sqrt n)}
 {\sqrt{1-u^2/n}}
\]

has a uniform Berry–Esseen bound C/sqrt(M) on the compact u-window: h is bounded and its variance is uniformly bounded away from zero for large n. Hence

\[
c\sqrt n\widehat\theta
=cu+c\sqrt{n/M}\sqrt{1-u^2/n}\,Z_{n,u}.
\]

The coefficient of Z_{n,u} tends uniformly to r. This implies uniform convergence of the bounded loss expectations to D(cu,r). One direct justification is by subsequences: any alleged nonuniformity supplies u_n in the compact window; extract u_n->u_*, apply the triangular CLT and bounded continuous mapping along that subsequence, and use continuity of D. Replacing Phi(cu) by the true pi_n changes expected absolute loss by at most r_n. This proves (L.16).

Pair positive and negative Z to obtain

\[
D(t,r)=\int_0^\infty
 [\Phi(t+rz)-\Phi(t-rz)]\varphi(z)\,dz.
\tag{L.18}
\]

Symmetry makes this even in t. For t>0 and z>0,
|t+rz|>|t-rz|, so differentiation under the integral gives a strictly negative derivative in t. Therefore the maximum over a window containing zero is D(0,r).

Finally,

\[
\frac{\partial}{\partial r}D(0,r)
=2\int_0^\infty z\varphi(rz)\varphi(z)\,dz
=\frac1{\pi(1+r^2)}.
\]

Since D(0,0)=0, integration yields D(0,r)=arctan(r)/pi. Uniform convergence in (L.16) allows passage to the supremum, proving (L.17). □

The arctan expression remains a result for this specified predictor. It is now its worst-case local-window risk limit, rather than a universal lower bound. In this known family, the constant predictor has worst-case limiting risk B0; it can outperform the raw moment predictor for some combinations of window width and calibration ratio.

### A window-aware alternative

When the parameter window is declared known, define

\[
\widehat u^{\rm clip}=\operatorname{proj}_{[-A_0,A_0]}(\sqrt n\widehat\theta),
\qquad \widehat\pi^{\rm clip}=\Phi(c\widehat u^{\rm clip}).
\]

For each u in the window, projection onto the corresponding probability interval gives the pointwise inequality

\[
|\widehat\pi^{\rm clip}-\Phi(cu)|
\le |\widehat\pi^{\rm mom}-\Phi(cu)|.
\]

For loss against the exact SGD probability, its expected risk is at most the raw predictor's risk plus 2r_n. It has the same sufficiency threshold, and is a useful additional comparator. This observation does not establish its minimax optimality.

## 4. Experimental design following these results

Use the explicit family first. Keep x0=0, w, nu, eta, kappa, b and the tolerance fixed. Check the common compact geometry before selecting numerical constants; the phrase “sufficiently small d” must be instantiated for the chosen run.

For an interpretable horizontal axis, choose a grid of seed coordinates t in [-K,K] and generate theta=t/(c sqrt(n)). Here the theoretical u-window is [-K/c,K/c]. The generator knows theta; the predictor receives only its calibration scores and common constants. Do not supply t, u, the true root, or the exact target pool to the predictor.

Exact sampling is possible by proposing Z~f0 and accepting with probability (1+theta h(Z))/(1+|theta|H_*). The proposal procedure is internal to generation and must not become side information for prediction.

Use four predictors: constant .5; raw moment; window-clipped moment; and empirical-root with estimated scale. Define a fallback for every failed empirical root or scale fit, retaining all failures in the primary risk denominator. The moment predictor has no root-solving failure.

Choose calibration budgets through the relative uncertainty r=c sqrt(n/M). For example r in {2,1,.5,.25} corresponds to M approximately c^2 n/r^2. Report both M/n and r. Because c^2 can be large at a small fixed step size, modest nominal ratios M/n may all lie in a poor practical regime despite the asymptotic threshold.

Use a separate sequence with M/n increasing to test consistency, and a sequence with M/n decreasing to test the approach to the constant-predictor benchmark. A fixed-r sequence tests L.8; it does not test consistency.

Estimate each true pi_n from independent future SGD paths. Record complete paths or at least reproducible random-stream state and exit records so capped paths can be continued without changing observed prefixes. With N paths, N_R right exits and N_C censored paths, the eventual right-exit fraction in that sample lies between N_R/N and (N_R+N_C)/N. Report this censoring bracket separately from Monte Carlo uncertainty. If an interval for the population probability is needed, use simultaneous binomial bounds for the fixed-cap events “right exit by cap” and “not left exit by cap”; the former lower bound and latter upper bound bracket pi_n with the declared confidence level. Do not treat nonexit by the cap as a third permanent outcome.

Use the same independent reference estimate for all predictors in a cell. If its error is at most delta, every estimated absolute probability loss differs from its true-target loss by at most delta, by the reverse triangle inequality. Thus reference Monte Carlo uncertainty must be smaller than the scale of predictor differences being interpreted. Conditional on a fixed reference sample, reuse across calibration repetitions is not additional independent evidence about pi_n.

Report the cellwise mean absolute error, calibration-repetition uncertainty, grid maximum error, root failure frequency, root error, and censoring. At finite n the moment predictor need not achieve its maximum exactly at u=0; L.17 is an asymptotic assertion. A finite grid does not certify uniformity over the continuum.

No sequential stopping rule or gain-selection algorithm is required for this experiment.

## 5. Numerical checks performed for this supplement

These are quadrature checks of algebraic formulas, not new entropy-SGD simulations or theorem certification. Using NumPy 2.3.5 and SciPy 1.17.0:

- At w=2 and nu=1, numerical Gaussian integration gives lambda approximately 0.2421101333 and sigma_g^2 approximately 0.1078103527. With eta=.1, the derived c is approximately 9.1436882021. This eta is used here only to illustrate the algebraic scale; the full d-dependent L.1 conditions have not been certified for this numerical choice.
- Tilt moments at theta=-.1, 0, .1 numerically agree with E_theta h=theta and E_theta h^2=1 to tolerance 1e-10.
- Numerical integration of (L.18) at r=.25,.5,1,2,4 agrees with arctan(r)/pi at t=0 to a maximum observed discrepancy of 3.4e-16. The computed D(t,r) decreases on t=0,.1,...,3 for each of these r values.

These checks support transcription and algebra. The mathematical proofs above, rather than numerical agreement, justify the results.

## 6. Suggested main-text contribution paragraph

We study the independent information required to predict the direction of a future risk exit under entropy adaptation. We construct a scalar family with fixed initialization and an identical complete classification-risk function across environments, while the unknown unlabeled marginal moves an unstable entropy root on the mini-batch noise scale. A uniform approximation for the actual nonlinear SGD direction connects this dynamical sensitivity to a statistical estimation problem. Within this known family and at fixed positive step size, uniform consistency is possible exactly when the independent calibration size grows faster than the SGD batch size. When their ratio tends to zero, the exact limiting minimax risk is attained by the constant forecast one half. At positive finite ratios, an explicit moment predictor instead has a nonvanishing worst-case risk limit governed by an arctangent law; this predictor-specific limit is not a minimax lower bound.

## 7. What remains open

The positive finite-ratio minimax constant, global sharp rates when the calibration error is smaller than the exit approximation remainder, robustness to a misspecified known tilt direction, broader unknown score families, joint vanishing-step limits and neural-network adaptation remain separate questions. The current proofs do not resolve them. Before making a novelty claim, compare the completed observation experiment and uniform SGD link with the relevant statistical and unstable-exit literature.

# Risk boundaries, empirical pools, and initialization

Part of [the manuscript](DRAFT.md). E-numbering and section numbers are continuous across [scalar exit](scalar_exit.md), [calibration](calibration.md), and [risk boundaries](risk_boundaries.md). Full statements and proofs are retained. Constants are not numerically calibrated; integration is not a completed independent proof audit. Use \(B_U=w/4\) for the update bound and \(M\) for calibration sample size; the original Gaussian sections use \(M=w/4\) locally.

## 6. Fixed off-peak initialization is a different asymptotic regime

### Proposition E.9. Finite deterministic exit and persistent source-calibration failure

Keep G1–G3 with a fixed interval \(I=[a,c]\), but replace G4 by a fixed initialization \(x_0\in(a,c)\), \(x_0\ne q\). Let
\[
\bar x_{t+1}=G(\bar x_t),\qquad \bar x_0=x_0,
\qquad \tau_{\rm pop}=\inf\{t:\bar x_t\notin I\}.
\]
Then \(\tau_{\rm pop}<\infty\). Assume that all iterates before this exit are strictly inside \(I\), and the exit iterate is strictly outside it. Then
\[
\Pr(\tau_n=\tau_{\rm pop},\ J_n=J_{\rm pop})\to1.
\tag{E.68}
\]
The exit count does not grow like \(\log n\) in this fixed off-peak regime.

Now suppose an independent source calibration converges to a different field with root \(q_C=x_0\), positive multiplier \(A_C>1\), and nonzero seed variance \(s_C^2\). Suppose its root error is \(O_{\mathbb P}(M^{-1/2})\), its conjugacy has a uniform local second-derivative bound, and \(M/n\to\infty\). A source-centered Gaussian-seed direction predictor tends to \(1/2\), whereas the true target direction tends to zero or one. Its absolute direction-probability error therefore tends to \(1/2\).

If its two calibrated boundary coordinates also converge to finite positive values, then the maximum total-exit CDF error tends to one:
\[
\sup_{t\ge0}\left|
\Pr(\tau_n\le t)
-\{\widehat\Psi_{C,n}^+(t)+\widehat\Psi_{C,n}^-(t)\}
\right|\to_{\mathbb P}1.
\tag{E.69}
\]

**Proof.** If \(x_0>q\), the population iterates move rightward. The drift has a positive minimum on \([x_0,c]\), so exit occurs in finitely many steps. The left case is identical. On any fixed time horizon, the Lipschitz recursion and the batch variance bound imply
\[
\max_{t\le T}|x_t-\bar x_t|\to_{\mathbb P}0.
\]
For example, conditioning at each step gives
\(\mathbb E|x_{t+1}-\bar x_{t+1}|^2
\le(1+\eta L_U)^2\mathbb E|x_t-\bar x_t|^2+\eta^2B_U^2/n\).
The finitely many strict margins from the boundaries then prove (E.68). A population iterate exactly on a boundary is excluded because the limiting exit count can then depend on the next-order fluctuations.

For the source predictor, \(x_0=q_C\) and \(M/n\to\infty\) imply that its estimated normalized seed tends to zero. Its direction forecast therefore tends to \(1/2\). At the fixed time \(\tau_{\rm pop}\), its positive boundary factors are proportional to \(\sqrt n A_C^{-\tau_{\rm pop}}\), which diverge. Both forecast joint CDFs tend to zero, whereas the true total CDF tends to one by (E.68). \(\square\)

This proposition separates a source-to-target geometric bias from target-sample calibration noise. Increasing the number of source-distribution calibration observations can remove the latter while preserving the former. It also explains why a fixed source-calibrated cancellation initialization must not be silently treated as a target-critical sequence for all batch sizes.

## 7. Risk boundaries and disconnected empirical safe sets

Let \(R(x)\) be the classification risk used for evaluation, and define
\[
g_n(x)=R(x)-R(x_{0,n})-\epsilon,\qquad
\mathcal S_n=\{x:g_n(x)\le0\},\qquad
T_n=\inf\{t:x_t\notin\mathcal S_n\}.
\tag{E.70}
\]
The update law and this evaluation risk are separate objects. In the Gaussian model both are specified by the joint target distribution. For a fixed empirical pool, the risk can instead be a weighted empirical classification error with a declared prediction rule at score ties.

### Proposition E.10. Regular risk-endpoint perturbations

Suppose \(a_n,c_n\) are roots of \(g_n\), with disjoint neighborhoods of radius \(h>0\), and \(R\) is continuously differentiable on these neighborhoods. Suppose, uniformly in \(n\),
\[
R'(x)\le-g_0<0\quad\text{near }a_n,\qquad
R'(x)\ge g_0>0\quad\text{near }c_n.
\tag{E.71}
\]
Let \(\widehat g_n\) be a continuous estimated risk-difference function with
\[
\sup_{x\text{ in the two neighborhoods}}
|\widehat g_n(x)-g_n(x)|\le\delta_R<g_0h.
\tag{E.72}
\]
Each neighborhood contains a zero of \(\widehat g_n\). Any selected zeros \(\widehat a_n,\widehat c_n\) in these neighborhoods satisfy
\[
\max(|\widehat a_n-a_n|,|\widehat c_n-c_n|)
\le\delta_R/g_0.
\tag{E.73}
\]
Their contribution in Theorem E.6 is therefore at most \(C\delta_R/g_0\).

**Proof.** The derivative signs imply opposite signs of \(g_n\) at each pair of neighborhood endpoints, of magnitude at least \(g_0h\). Equation (E.72) preserves these signs, so continuity gives estimated zeros. At a selected zero \(\widehat c_n\),
\[
|g_n(\widehat c_n)|
=|g_n(\widehat c_n)-\widehat g_n(\widehat c_n)|
\le\delta_R.
\]
The mean-value theorem gives
\(|g_n(\widehat c_n)-g_n(c_n)|\ge g_0|\widehat c_n-c_n|\).
The other boundary is identical. \(\square\)

This is a local endpoint result. It does not by itself establish that the estimated safe set has the correct number of components. If \(\sup_x|\widehat R(x)-R(x)|\le\delta\) and both predictors use the same frozen state, then \(\delta_R\le2\delta\) is valid. The factor two accounts for estimating both the updated and frozen risks.

The transversality constant \(g_0\) is essential. Near a flat risk boundary, or when tolerance shrinks so that the endpoints approach a risk minimizer, it need not stay positive. The inverse-batch tolerance theorem in the preceding Gaussian manuscript is therefore not extended by Proposition E.10. Also, empirical threshold risk is a step function; it must not be inserted into (E.71) as if it had a nonzero ordinary derivative.

### Proposition E.11. Transfer from component exit to actual risk exit

Suppose the connected component \(I_n\) of \(\mathcal S_n\) containing \(x_{0,n}\) is a bounded interval containing \(q\), with its actual endpoint membership retained. Let \(\tau_n\) be its first exit. For a finite integer horizon \(T_{\max}\), put
\[
\pi_{\mathrm{skip},n}(T_{\max})
=\Pr\big(\tau_n\le T_{\max},\
x_{\tau_n}\in\mathcal S_n\setminus I_n\big).
\tag{E.74}
\]
Then \(\tau_n\le T_n\), and
\[
\sup_{0\le t\le T_{\max}}
\left|\Pr(T_n\le t)-\Pr(\tau_n\le t)\right|
\le \pi_{\mathrm{skip},n}(T_{\max}).
\tag{E.75}
\]
If exit directions for both events are defined by the sign of the exit state relative to \(q\), the same bound holds separately for each joint direction-and-time CDF.

Consequently, any joint-CDF approximation bound for component exit transfers to risk exit on this horizon after adding \(\pi_{\mathrm{skip},n}(T_{\max})\).

If
\[
\operatorname{dist}(I_n,\mathcal S_n\setminus I_n)>\eta B_U,
\tag{E.76}
\]
then the skip probability is zero. Alternatively, if
\[
\inf_{x\in I_n}
\operatorname{dist}(G(x),\mathcal S_n\setminus I_n)\ge d>0,
\tag{E.77}
\]
then
\[
\pi_{\mathrm{skip},n}(T_{\max})
\le
\min\left\{1,\
2T_{\max}\exp\!\left(-\frac{nd^2}{2\eta^2B_U^2}\right)
\right\}.
\tag{E.78}
\]
The distance to the empty set is interpreted as infinity.

**Proof.** Before leaving \(I_n\), the process remains risk-safe, so \(T_n\ge\tau_n\). If the first state outside \(I_n\) is unsafe, the two stopping times and directions coincide. If it is safe, it lies in a different component and the two events can differ. For comparisons up to \(T_{\max}\), a discrepancy requires precisely such a skip by that horizon. This coupling proves (E.75) and its joint-CDF version.

The update magnitude is at most \(\eta B_U\), proving (E.76). Under (E.77), landing in another safe component from a state \(x\in I_n\) requires \(|\eta\xi_j(x)|\ge d\). The conditional sub-Gaussian bound in (E.20) gives probability at most \(2\exp[-nd^2/(2\eta^2B_U^2)]\) for that step. Sum over the at most \(T_{\max}\) steps before component exit. \(\square\)

For a discontinuous empirical risk, endpoint inclusion and the classifier's score-tie rule are part of the event definition. The open/half-open version of Theorem E.3 accommodates these conventions. A disconnected safe set, a one-sided unbounded component, or a component that does not contain the selected repeller must not be treated as a two-sided critical interval without an additional argument.

In practice, the skip discrepancy can first be measured on independently sampled adaptation paths conditional on the feature pool. A finite-horizon estimate only concerns skips within the recorded budget. It does not establish an all-time bound, and it is not a population-risk certificate.

## 8. Conditional empirical-pool inference and population risk

The distribution \(Q\) in the preceding results may be the fixed target empirical law \(Q_N\). Conditional on that pool, drawing calibration and adaptation samples independently with replacement satisfies the independence assumptions. Repeated observations count toward \(M\) even when the same score value is seen multiple times. They improve estimation of \(Q_N\); they are not additional independent images from an underlying population.

If a population score law \(Q_\star\) is also considered, the local-root decomposition is
\[
\widehat q_{M,N}-q_\star
=(\widehat q_{M,N}-q_N)+(q_N-q_\star).
\tag{E.79}
\]
Increasing \(M\) with \(N\) fixed removes only the first term. The second is a feature-pool sampling or distribution error. It is not covered by (E.55), and its variance should not be silently identified with the conditional calibration variance. With stratified class pools, its variance also depends on the classwise sampling design.

### Proposition E.12. A separate uniform risk bound for stratified empirical pools

Conditional on an independently fixed score projection, suppose the positive and negative class pools contain \(N_+\) and \(N_-\) iid observations from their respective population score distributions. Let the target positive prior \(p\) be fixed and known. Denote the population threshold risk by \(R_p\) and its classwise weighted empirical counterpart by \(R_{N,p}\), using the same score-tie convention.

For \(0<\alpha<1\), define
\[
d_+=\sqrt{\frac{\log(4/\alpha)}{2N_+}},
\qquad
d_-=\sqrt{\frac{\log(4/\alpha)}{2N_-}}.
\tag{E.80}
\]
With probability at least \(1-\alpha\), simultaneously over all thresholds \(x,x_0\),
\[
\left|
\{R_{N,p}(x)-R_{N,p}(x_0)\}
-\{R_p(x)-R_p(x_0)\}
\right|
\le2\{p d_++(1-p)d_-\}.
\tag{E.81}
\]

**Proof.** Apply the Dvoretzky–Kiefer–Wolfowitz inequality with failure probability \(\alpha/2\) to each classwise empirical CDF. The same uniform bound applies to left limits of CDFs, so either consistent threshold-tie convention is covered. A union bound gives
\[
\sup_x|R_{N,p}(x)-R_p(x)|
\le p d_++(1-p)d_-.
\]
Apply this at \(x\) and \(x_0\). \(\square\)

For \(N_+=N_-=1000\) and \(\alpha=.05\), (E.81) is approximately .09362 for one pair, regardless of \(p\). Simultaneous coverage across multiple pairs requires an appropriate shared confidence event or an additional allocation of the failure probability. The bound is conservative; exceeding a tolerance of .02 does not prove that the actual population error is .09362. It means this bound cannot certify that tolerance at the stated sample size.

A prediction law conditional on \(Q_N\), a confidence statement over the construction of the pool, and a guarantee about every deployed population-risk iterate are three different objects. Theorems E.3 and E.6 address the first. Proposition E.12 supplies one elementary ingredient for the second. None of them by itself supplies the third.

## 9. A separate process: calibration-driven initialization

The distinction between prediction error and process noise can also be formalized positively.

### Proposition E.13. Variance addition when calibration actually sets the initialization

Assume G1–G3 for a fixed interval \(I=[a,c]\). Draw the calibration sample independently of all future adaptation samples, and now change the algorithm by initializing
\[
x_{0,n}^{\mathrm{new}}=\widehat q_M+\frac u{\sqrt n}.
\tag{E.82}
\]
Here \(u\in\mathbb R\) is fixed. Use empirical calibration from the true \(Q\), with \(M\to\infty\) and \(\mathcal R_{M,n}\to r<\infty\). Then, writing \(z=u/s\),
\[
\Pr(J_n^{\mathrm{new}}=+1)
\longrightarrow\Phi\!\left(\frac z{\sqrt{1+r^2}}\right).
\tag{E.83}
\]
Along any subsequence of batch sizes and integer times satisfying \(\sqrt nA^{-t_n}\to\zeta\in(0,\infty)\),
\[
\Pr(\tau_n^{\mathrm{new}}\le t_n,J_n^{\mathrm{new}}=+1)
\longrightarrow
\Phi\!\left(\frac{z-\zeta B^+/s}{\sqrt{1+r^2}}\right),
\]
\[
\Pr(\tau_n^{\mathrm{new}}\le t_n,J_n^{\mathrm{new}}=-1)
\longrightarrow
\Phi\!\left(\frac{-z-\zeta B^-/s}{\sqrt{1+r^2}}\right).
\tag{E.84}
\]
Thus the marginal seed variance of this new process is \(s^2(1+r^2)\).

**Proof.** The root CLT implies tightness of
\(\sqrt n(x_{0,n}^{\mathrm{new}}-q)\), and its limit is \(u+srZ_C\).
Conditional on the calibration sample, apply Theorem E.3 on the event
\(|\sqrt n(x_{0,n}^{\mathrm{new}}-q)|\le K\). Its constants are uniform for each fixed \(K\). The complement has arbitrarily small limiting probability as \(K\to\infty\). The frozen adaptation seed is independent of calibration and converges to \(sZ_S\), where \(Z_C,Z_S\) are independent standard normals. The combined seed is therefore
\[
u+srZ_C+sZ_S\sim N(u,s^2(1+r^2)).
\]
The joint CDF and direction formulas follow either by averaging the conditional Gaussian formulas or by this independent-seed representation. \(\square\)

This proposition concerns the altered initialization (E.82), not the fixed-initialization experiments. It is stated for a fixed exit interval. If the risk comparator and its safe endpoints are recomputed from the random initialization, their uniform geometric validity and convergence must additionally be checked before applying the same argument.

## 11. Scope and theorem dependencies

| Result | What is established here | What remains outside the result |
|---|---|---|
| E.3 | Actual entropy-SGD joint exit law for general scalar score laws under explicit geometry; uniform integer-time CDF error | Automatic validity for every visual marginal or every empirical safe component |
| E.4 | Local quantitative stability of the normalized conjugacy under \(C^3\) field perturbations | Stability across bifurcations or uniformly as curvature vanishes |
| E.5–E.6 | Independent empirical field calibration and its error in predicting actual fixed-initialization SGD | Unlabeled estimation of classification-risk endpoints |
| E.7–E.8 | Nonvanishing calibration-error limits for direction and logarithmic-window time predictions | A universal minimax sample-complexity lower bound |
| E.9 | Fixed off-peak deterministic limit and failure under persistent source-peak miscentering | A logarithmic batch-size delay for that fixed-bias regime |
| E.10–E.11 | Regular endpoint perturbations and an explicit component-skip correction | Unchecked topology of empirical safe sets or flat-boundary limits |
| E.12 | A separate classwise empirical-to-population risk bound under iid pool assumptions | Exact population crossing times inferred from finite-pool exits |
| E.13 | Calibration variance enters the marginal seed when calibration changes initialization | Variance inflation of an unchanged adaptation process |

The general exit theorem is proved for a fixed \(Q\), matching inference conditional on a fixed target pool. For a sequence \(Q_n\), the proof can only be used uniformly after checking common derivative bounds, a positive lower bound on root curvature and root noise variance, uniform root-to-boundary separation, a common inverse-conjugacy domain, and uniform outward-drift control away from the root. A changing distribution family must not be admitted solely by relabeling the constants.

All results keep \(w,\eta\) fixed. They do not cover joint confidence-scale adaptation, momentum, dependent streams, online reuse of calibration data as adaptation batches, or sequentially chosen calibration stopping times. A fixed-time calibration sample independent of SGD is sufficient here; an online predictor repeatedly refitted from the adaptation trajectory would require a different dependence analysis.

The proof tools—local conjugacy, bounded martingale increments, Berry–Esseen approximation, estimating-equation CLTs, and Gaussian anti-concentration—are standard. The paper-level contribution should be stated in terms of the real recursion, risk-event definition, and testable calibration consequences. This extension does not certify priority over all discrete random-map or stochastic-approximation literature.


## References

[E-R1] A. W. van der Vaart. *Asymptotic Statistics*. Cambridge University Press, 1998. Chapter 5, “M- and Z-Estimators.” [Publisher chapter](https://doi.org/10.1017/CBO9780511802256.006).

[E-R2] Y. Bakhtin. *Exit asymptotics for small diffusion about an unstable equilibrium*. arXiv:math/0701569, 2007. [Preprint](https://arxiv.org/abs/math/0701569). Journal DOI: [10.1016/j.spa.2007.06.003](https://doi.org/10.1016/j.spa.2007.06.003).

[E-R3] Y. Bakhtin and A. Gaál. *Exit time asymptotics for dynamical systems with fast random switching near an unstable equilibrium*. arXiv:1901.05513, 2019. [Preprint](https://arxiv.org/abs/1901.05513). Journal DOI: [10.1142/S0219493720500264](https://doi.org/10.1142/S0219493720500264).

[E-R4] P. Massart. *The Tight Constant in the Dvoretzky–Kiefer–Wolfowitz Inequality*. The Annals of Probability 18(3):1269–1283, 1990. [DOI](https://doi.org/10.1214/aop/1176990746).

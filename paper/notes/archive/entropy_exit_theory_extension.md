# Predicting Entropy-Adaptation Risk Exit with Estimated Geometry

## A standalone extension: general scalar marginals, independent calibration, and risk-boundary error

Working theoretical extension, 2026-10-28. This file is intended for later integration with *When Does Entropy Adaptation Stop Helping?* It uses separate E-numbering and includes its assumptions and proofs. The statements concern a fixed confidence slope and scalar threshold. No new visual experiments are reported here. The finite-sample constants are not numerically calibrated.

The sample-update bound is denoted by \(B_U\), reserving \(M\) for the number of calibration observations. This avoids the two uses of \(M\) in the preceding manuscript.

## Abstract

Predicting the risk horizon of entropy adaptation requires both a dynamical model and sufficiently accurate geometry. We first extend the scalar entropy-SGD exit analysis from Gaussian mixtures to arbitrary score distributions satisfying explicit local and interval-wide geometric conditions, including finite empirical distributions conditional on their feature pool. A normalized conjugacy yields a Gaussian-seed approximation for the joint exit direction and integer exit time. A stopped-martingale tail argument makes the approximation uniform over all integer times. We then analyze a separate unlabeled calibration sample that changes the predictor but leaves the actual SGD initialization and updates unchanged. We prove stability of the normalized conjugacy under smooth field perturbations and obtain a joint-CDF prediction bound separating peak, growth-rate, noise, and risk-boundary errors. In the critical regime, root-estimation error is amplified by the square root of the batch size. When calibration uncertainty remains comparable to the SGD seed, the predicted exit probability has a nondegenerate random limit even though the estimated peak is consistent. At exact critical centering, its limiting mean absolute direction-probability error is an explicit arctangent function of the relative calibration scale. We also distinguish fixed off-peak initialization, calibration-driven initialization, regular risk-boundary perturbations, and jumps between disconnected empirical safe components. These results describe the limits of geometric prediction; they do not provide an unlabeled estimator of classification-risk boundaries.

## 1. The process being predicted

Let \(Q\) be a fixed probability distribution on scalar scores \(Z\in\mathbb R\). It may be continuous, discrete, or a fixed weighted empirical distribution. Fix \(w>0\), define
\[
s(u)=(1+e^{-u})^{-1},\qquad a(u)=s(u)(1-s(u)),
\]
and use the actual threshold entropy velocity
\[
U(x,z)=-w^2(z-x)a(w(z-x)),\qquad
v(x)=\mathbb E_Q U(x,Z).
\tag{E.1}
\]
The population map and fresh-batch update are
\[
G(x)=x+\eta v(x),\qquad
x_{t+1}=x_t+\frac{\eta}{n}\sum_{i=1}^n U(x_t,Z_{t,i}),
\tag{E.2}
\]
where the \(Z_{t,i}\) are iid from \(Q\), and the initial state is deterministic unless stated otherwise.

An independent calibration sample
\[
\widetilde Z_1,\ldots,\widetilde Z_M\overset{\mathrm{iid}}{\sim}Q
\tag{E.3}
\]
will be used only to construct a predictor. It does not alter \(x_0\), \(Q\), the adaptation samples, or the recursion (E.2). Conditioning on a fixed feature pool, its source projection, and any externally fixed design parameters is permitted. All probabilities in that case are conditional on those objects.

For intervals \(I_n=[a_n,c_n]\), define
\[
\tau_n=\inf\{t\ge0:x_t\notin I_n\},\qquad
J_n=\begin{cases}+1,&x_{\tau_n}>c_n,\\-1,&x_{\tau_n}<a_n.\end{cases}
\tag{E.4}
\]
The first part of the analysis concerns this interval-exit event. It becomes a classification-risk exit theorem when the relevant risk-safe set is exactly \(I_n\), or when the event discrepancy is separately controlled as in Section 7.

### Lemma E.1. Distribution-free regularity of the entropy update

For every \(x,z\),
\[
|U(x,z)|\le B_U:=w/4,\qquad
|\partial_xU(x,z)|\le L_U:=w^2/4.
\tag{E.5}
\]
Every higher fixed-order derivative in \(x\) is also uniformly bounded. In particular, \(|\partial_x^2 U|\le 3w^3/4\). Consequently \(v\) is smooth for every probability distribution \(Q\), without moment assumptions on \(Z\).

If \(0<\eta<1/L_U\), both \(G\) and every empirical population map \(x+\eta M^{-1}\sum_i U(x,\widetilde Z_i)\) are increasing smooth bijections of the line.

**Proof.** If \(h\) is the binary entropy of \(s\), then \(h'(u)=-ua(u)\) and
\[
h''(u)=a(u)\{u\tanh(u/2)-1\}.
\]
To verify the constants, write \(a(u)=[2(1+\cosh u)]^{-1}\). The inequality \(1+\cosh u\ge2+u^2/2\ge2|u|\) gives \(|u|a(u)\le1/4\). The two nonnegative terms \(ua(u)\tanh(u/2)\) and \(a(u)\) each lie in \([0,1/4]\), proving \(|h''|\le1/4\). Differentiation gives
\[
h'''(u)=2a(u)\tanh(u/2)+ua(u)\{6a(u)-1\},
\]
whose absolute value is at most \(1/2+1/4=3/4\). These bounds give (E.5) and the second-derivative bound after scaling. Further derivatives are sums of polynomial factors times derivatives of the logistic function, and decay at both infinities; hence they are bounded. Dominated differentiation applies under any \(Q\). Finally,
\[
G'(x)\ge1-\eta L_U>0,\qquad |G(x)-x|\le\eta B_U.
\]
Strict monotonicity and the limits at infinity give bijectivity. The empirical map satisfies the same inequalities. \(\square\)

## 2. General scalar exit geometry

Fix a compact interval \(\mathcal K=[\ell,r]\) and \(q\in(\ell,r)\). We impose the following conditions.

**G1: one repeller on the required domain.**
\[
v(q)=0,\qquad \lambda:=v'(q)>0,\qquad
(x-q)v(x)>0\quad(x\in\mathcal K,\ x\ne q).
\tag{E.6}
\]
No uniqueness of the entropy peak outside \(\mathcal K\) is required.

**G2: nondegenerate noise.**
\[
\sigma_g^2:=\operatorname{Var}_Q U(q,Z)>0.
\tag{E.7}
\]

**G3: fixed step size and interior exit boundaries.** Fix \(0<\eta<1/L_U\), and set
\[
A=1+\eta\lambda>1,\qquad
s^2=\frac{\eta^2\sigma_g^2}{A^2-1}.
\tag{E.8}
\]
For fixed \(d_0,\gamma>0\), all sufficiently large \(n\) satisfy
\[
a_n\le q-d_0,\qquad c_n\ge q+d_0,
\]
\[
[a_n-\eta B_U,c_n+\eta B_U]
\subset[\ell+\gamma,r-\gamma].
\tag{E.9}
\]
Thus both the deterministic proposal and the actual one-step overshoot from \(I_n\) remain in the domain where the geometry is controlled.

**G4: critical initialization.**
\[
|x_{0,n}-q|\le K_0/\sqrt n
\tag{E.10}
\]
for a fixed finite \(K_0\). In particular \(x_{0,n}\in I_n\) for large \(n\).

Constants below may depend on \(Q,w,\eta,\mathcal K,d_0,\gamma,K_0\). They do not depend on \(n,t,M\). Uniformity over a family of score distributions additionally requires uniform versions of these geometric and noise bounds; it is not automatic when a curvature or an outward drift bottleneck approaches zero.

### Lemma E.2. Normalized conjugacy on the exit domain

Under G1 and G3, there is an increasing \(C^2\) function \(H\) on \(\mathcal K\), with bounded derivatives and a strictly positive lower bound on \(H'\), such that
\[
H(q)=0,\qquad H'(q)=1,\qquad H(G(x))=A H(x)
\tag{E.11}
\]
whenever \(x,G(x)\in\mathcal K\). It is the unique normalized \(C^1\) conjugacy on this domain.

**Proof.** Since \(G\) is increasing and \(G(q)=q\), its inverse sends each point of \(\mathcal K\) toward \(q\). For example, if \(x>q\), then
\[
q<G^{-1}(x)<x,
\]
using \(G(x)>x\) and monotonicity. The inverse orbit stays in \(\mathcal K\) and converges to \(q\); any other limit would be another zero of \(v\).

Near zero define the centered inverse
\[
g(y)=G^{-1}(q+y)-q=A^{-1}y+O(y^2).
\]
Choose a sufficiently small neighborhood and \(b<1\) such that
\[
\sup |g'|\le b,\qquad Ab^2<1.
\]
Writing \(R(y)=g(y)-A^{-1}y\), the approximants
\[
h_j(y)=A^j g^j(y)
\]
satisfy
\[
h_{j+1}(y)-h_j(y)=A^{j+1}R(g^j(y)).
\tag{E.12}
\]
Here \(g^j\) denotes composition. Iterated differentiation gives
\[
\|g^j\|_{C^2}\le Cb^j
\]
on this local interval. Since \(R(0)=R'(0)=0\), the \(C^2\) norm of the right-hand side of (E.12) is at most \(C(Ab^2)^j\). The sequence converges in \(C^2\) to \(h\), with \(h(0)=0,h'(0)=1\), and \(h'>0\) after decreasing the local interval if necessary.

A fixed finite number \(N\) of inverse steps brings all of \(\mathcal K\) into that local interval. This follows first for the two endpoints and then for every interior point by monotonicity. Set
\[
H(x)=A^N h(G^{-N}(x)-q).
\tag{E.13}
\]
The local functional equation makes this definition independent of a sufficiently large \(N\). It gives (E.11), regularity, and positivity of \(H'\). Compactness supplies the derivative bounds.

For uniqueness, any other normalized \(C^1\) conjugacy \(\widetilde H\) satisfies
\[
\widetilde H(x)=A^j\widetilde H(G^{-j}(x)).
\]
Since \(\widetilde H(q+y)=y+o(y)\) and \(A^j(G^{-j}(x)-q)\to H(x)\), the right-hand side tends to \(H(x)\). \(\square\)

### Theorem E.3. Joint exit approximation for general scalar score laws

Under G1–G4, \(\tau_n<\infty\) almost surely. Define
\[
\mu_n=\sqrt n H(x_{0,n}),\qquad
B_n^+=H(c_n),\qquad B_n^-=-H(a_n),
\tag{E.14}
\]
and
\[
\Psi_n^+(t)=
\Phi\!\left(\frac{\mu_n-\sqrt n A^{-t}B_n^+}{s}\right),
\qquad
\Psi_n^-(t)=
\Phi\!\left(\frac{-\mu_n-\sqrt n A^{-t}B_n^-}{s}\right).
\tag{E.15}
\]
The \(B_n^\pm\) are uniformly bounded above and away from zero. For all integers \(t\ge0\),
\[
\left|\Pr(\tau_n\le t,J_n=\pm1)-\Psi_n^\pm(t)\right|
\le C\left[n^{-1/2}+
\left(\frac{1+\log n}{n}+A^{-2t}\right)^{1/3}\right].
\tag{E.16}
\]
Moreover, with
\[
e_n=n^{-1/2}+\left(\frac{1+\log n}{n}\right)^{1/3},
\tag{E.17}
\]
the stronger uniform-in-time statement holds:
\[
\sup_{t\in\mathbb Z_{\ge0}}
\max_{\pm}\left|\Pr(\tau_n\le t,J_n=\pm1)-\Psi_n^\pm(t)\right|
\le Ce_n.
\tag{E.18}
\]
In particular,
\[
\left|\Pr(J_n=+1)-\Phi(\mu_n/s)\right|\le Ce_n.
\tag{E.19}
\]
The same conclusions hold for open or half-open intervals, with their declared endpoint membership used consistently in the exit event and in \(J_n\).

**Proof.**

**Step 1: finite exit without Gaussian tails.** Since \(\mathbb E U(q,Z)=0\) and its variance is positive, both \(U(q,Z)>0\) and \(U(q,Z)<0\) have positive probability. The sign of \(U(q,z)\) is the sign of \(q-z\). There therefore exist bounded score intervals strictly to the left and right of \(q\), with positive masses \(p_L,p_R\). For states \(x\in[q,c_n]\), samples from the left interval produce a uniformly positive velocity; for states \(x\in[a_n,q]\), samples from the right interval produce a uniformly negative velocity. Uniformity follows from continuity and compactness, and from the positive separation of the selected score intervals from \(q\).

Choose a fixed block length \(N_0\) large enough that either such sequence of updates exits the relevant side. At the start of any block before exit, select the left-score event if the state is at least \(q\), and the right-score event otherwise. The conditional probability of the selected event is at least
\[
p_n:=\min(p_L,p_R)^{nN_0}>0.
\]
Thus \(\Pr(\tau_n>kN_0)\le(1-p_n)^k\). This argument applies to discrete score laws as well.

**Step 2: stopped conjugacy coordinates.** Write
\[
\xi_j(x)=\frac1n\sum_{i=1}^n U(x,Z_{j,i})-v(x).
\]
Conditional on the history before batch \(j\), for predictable \(x\),
\[
\mathbb E\xi_j(x)^2\le B_U^2/n,\qquad
\mathbb E\xi_j(x)^4\le C/n^2,
\]
\[
\mathbb E(\xi_j(x)-\xi_j(q))^2\le L_U^2|x-q|^2/n,
\qquad
\mathbb E e^{u\xi_j(x)}\le e^{u^2B_U^2/(2n)}.
\tag{E.20}
\]
Taylor expansion on the overshoot domain gives, before exit,
\[
H(x_{j+1})=AH(x_j)+
\eta H'(G(x_j))\xi_j(x_j)+R_j,
\qquad |R_j|\le C\xi_j(x_j)^2.
\tag{E.21}
\]
Define
\[
Y_{n,t}=\sqrt n A^{-(t\wedge\tau_n)}
H(x_{t\wedge\tau_n}).
\]
Summation of (E.21) expresses \(Y_{n,t}\) as \(\mu_n\), a stopped martingale \(\mathcal M_{n,t}\), and a remainder \(\mathcal E_{n,t}\). Orthogonality and (E.20) yield
\[
\sup_t\mathbb E|\mathcal M_{n,t}|^2\le C,\qquad
\sup_t\|\mathcal E_{n,t}\|_2\le C/\sqrt n,
\qquad \sup_t\mathbb E|Y_{n,t}|^2\le C.
\tag{E.22}
\]
For the remainder, use Minkowski's inequality and
\(\|\mathbf1_{\{j<\tau_n\}}\xi_j(x_j)^2\|_2\le C/n\)
in the geometrically weighted sum. Fatou's lemma also gives the corresponding estimate at exit. Since \(|H(x_{\tau_n})|\ge b_0>0\),
\[
\mathbb E A^{-2\tau_n}\le C/n.
\tag{E.23}
\]

**Step 3: freeze the noise at the actual root.** On the same infinite independent sample array, define
\[
W_n=\mu_n+\eta\sum_{j\ge0}A^{-j-1}\sqrt n\,\xi_j(q).
\tag{E.24}
\]
Before exit,
\[
D_j=H'(G(x_j))\xi_j(x_j)-\xi_j(q)
\]
has conditional mean zero and conditional variance at most \(C|x_j-q|^2/n\). The lower bound on \(H'\), (E.22), and compactness imply
\[
\mathbb E[\mathbf1_{\{j<\tau_n\}}|x_j-q|^2]
\le C\min(A^{2j}/n,1).
\tag{E.25}
\]
Consequently,
\[
\mathbb E\left|
\sqrt n\eta\sum_{j<t\wedge\tau_n}A^{-j-1}D_j
\right|^2
\le C\sum_{j\ge0}A^{-2j}\min(A^{2j}/n,1)
\le C(1+\log n)/n.
\tag{E.26}
\]
The frozen-noise tail after \(t\wedge\tau_n\) has conditional variance
\(s^2A^{-2(t\wedge\tau_n)}\). Equation (E.23) bounds its expected contribution. Together with the remainder,
\[
\mathbb E|Y_{n,t}-W_n|^2
\le C\left[\frac{1+\log n}{n}+A^{-2t}\right],
\]
\[
\mathbb E|Y_{n,\tau_n}-W_n|^2
\le C(1+\log n)/n.
\tag{E.27}
\]
This comparison controls the actual state-dependent batch updates.

**Step 4: Gaussian seed and event identities.** Grouping (E.24) by sample column gives
\[
W_n-\mu_n=\frac1{\sqrt n}\sum_{i=1}^n V_i,\qquad
V_i=\eta\sum_{j\ge0}A^{-j-1}U(q,Z_{j,i}).
\tag{E.28}
\]
The \(V_i\) are iid, centered, bounded by \(\eta B_U/(A-1)\), and have variance \(s^2>0\). Berry–Esseen therefore gives
\[
\sup_z\left|
\Pr(W_n\le z)-\Phi((z-\mu_n)/s)
\right|\le C/\sqrt n.
\tag{E.29}
\]
For closed intervals the exact stopped-coordinate identities are
\[
\{\tau_n\le t,J_n=+1\}
=\{Y_{n,t}>\sqrt n A^{-t}B_n^+\},
\]
\[
\{\tau_n\le t,J_n=-1\}
=\{Y_{n,t}<-\sqrt n A^{-t}B_n^-\}.
\tag{E.30}
\]
Before exit, the threshold cannot be crossed. After an exit at \(j\le t\), the coordinate is frozen at a value whose magnitude is larger than the boundary scaled by \(A^{-t}\).

For an excluded endpoint, replace the corresponding strict comparison in (E.30) by a weak comparison. The smoothing argument below works for either convention because the comparison Gaussian has a continuous bounded density. No non-atomicity of the finite-batch updates is assumed.

For any threshold, (E.27), Chebyshev's inequality, and (E.29) bound the probability error by
\[
C\{n^{-1/2}+h+\Delta_t/h^2\},\qquad
\Delta_t=(1+\log n)/n+A^{-2t}.
\]
Choosing \(h=\Delta_t^{1/3}\) proves (E.16). At exit, the sign of \(Y_{n,\tau_n}\) is \(J_n\), which proves (E.19).

**Step 5: remove the early-time loss in the bound.** The martingale in (E.22) has a uniform sub-Gaussian proxy. Indeed, each increment has conditional proxy at most
\[
\eta^2 A^{-2j-2}B_U^2\sup_{\mathcal K}|H'|^2.
\]
The sum is finite, so for every deterministic \(t\),
\[
\mathbb E e^{u\mathcal M_{n,t}}\le e^{C u^2}.
\tag{E.31}
\]
For sufficiently large \(y\), (E.22) and (E.31) imply
\[
\Pr(|Y_{n,t}|>y)\le C e^{-c y^2}+\frac{C}{ny^2}.
\tag{E.32}
\]
Uniform boundedness of \(\mu_n\) is used here.

If \(nA^{-2t}>D\log n\), both true exit probabilities in (E.30) and their Gaussian approximations are at most
\[
C e^{-cD\log n}+\frac{C}{nD\log n}.
\]
Choose a fixed sufficiently large \(D\). On all remaining times,
\(A^{-2t}\le D\log n/n\), so (E.16) is bounded by \(Ce_n\).
This proves (E.18). \(\square\)

The logarithmic mechanism and the Gaussian seed are classical unstable-exit phenomena [E-R2, E-R3]. The extension here keeps the actual batch-gradient noise, a declared exit interval, discrete update times, and the geometric quantities needed for the calibration analysis below.

## 3. Quantitative stability of the geometry

Let \(\widetilde v\) be another entropy field, or a smooth perturbation whose map \(\widetilde G=x+\eta\widetilde v\) is an increasing bijection. Compare it with the fixed field \(v\) on \(\mathcal K\). All derivatives through order three are assumed uniformly bounded in the neighborhood of fields under consideration.

### Lemma E.4. Stability of the root and normalized conjugacy

There are constants \(d_*,C>0\) such that, if
\[
d_v:=\|\widetilde v-v\|_{C^3(\mathcal K)}\le d_*,
\tag{E.33}
\]
then \(\widetilde v\) has a unique zero \(\widetilde q\) on \(\mathcal K\), has the same outward sign pattern relative to \(\widetilde q\), and
\[
|\widetilde q-q|\le C d_v,\qquad
|\widetilde\lambda-\lambda|+|\widetilde A-A|\le C d_v.
\tag{E.34}
\]
Here \(\widetilde\lambda=\widetilde v'(\widetilde q)\) and \(\widetilde A=1+\eta\widetilde\lambda\).

The normalized conjugacies satisfy
\[
\|\widetilde H-H\|_{C^1(\mathcal K)}\le C d_v,
\qquad
\|\widetilde H\|_{C^2(\mathcal K)}\le C.
\tag{E.35}
\]
For the critical initialization in G4, writing \(d_q=\widetilde q-q\),
\[
\sqrt n\{\widetilde H(x_{0,n})-H(x_{0,n})\}
=-\sqrt n\,d_q+
O\!\left(d_v+\sqrt n\,d_q^2\right).
\tag{E.36}
\]

**Proof.** Choose a local interval about \(q\) on which \(v'\ge\lambda/2\). A sufficiently small \(C^1\) perturbation has positive derivative there and opposite signs at its endpoints. It therefore has exactly one local zero. Away from this interval, compactness and G1 give a strictly positive lower bound on \(|v|\), so no additional zeros appear under a sufficiently small perturbation. The mean-value theorem gives the root bound; differentiation and bounded \(v''\) give the curvature and multiplier bounds.

For completeness, the quantitative dependence of \(H\) can be proved directly from the inverse construction. Center the inverse maps at their own roots, writing them as \(g,\widetilde g\). On a common sufficiently small local interval,
\[
\|g-\widetilde g\|_{C^2}\le C d_v,\qquad
|g'|,|\widetilde g'|\le b<1,\qquad
\overline A b^2<1,
\tag{E.37}
\]
where \(\overline A\) bounds both multipliers. The inverse-function formulas and the common lower bound on the forward derivatives justify the first inequality. Centering introduces only \(O(d_v)\) changes, including in the second derivatives, by the \(C^3\) hypothesis.

Both inverse maps fix zero. Iterating the difference recursion and its first derivative yields
\[
|\widetilde g^j(y)-g^j(y)|
\le C d_v(j+1)b^{j-1}|y|,
\]
\[
|(\widetilde g^j)'(y)-(g^j)'(y)|
\le C d_v(j+1)^2b^{j-1}.
\tag{E.38}
\]
For the derivative estimate, use the product rule, the common second-derivative bound, and the first estimate. The harmless cases \(j=0,1\) are absorbed in \(C\).

Let \(R_g(y)=g(y)-A^{-1}y\), with the analogous definition for \(\widetilde g\). Their difference has value and derivative zero at zero and \(C^2\) norm at most \(C d_v\). Thus
\[
|R_{\widetilde g}(y)-R_g(y)|\le C d_v y^2,\qquad
|R_{\widetilde g}'(y)-R_g'(y)|\le C d_v|y|.
\]
Use the convergent series
\[
h(y)=y+\sum_{j\ge0}A^{j+1}R_g(g^j(y)),
\tag{E.39}
\]
and its perturbed version. Equations (E.38), the quadratic vanishing of the residuals, and
\[
|\widetilde A^{j+1}-A^{j+1}|
\le C d_v(j+1)\overline A^j
\]
bound the \(C^1\) difference of the \(j\)-th terms by
\[
C d_v(j+1)^3(\overline A b^2)^j.
\]
Summation proves local \(C^1\) Lipschitz dependence. The unperturbed construction of Lemma E.2, with uniform bounds, also proves a common \(C^2\) bound for all the local conjugacies.

A common finite number of inverse steps maps \(\mathcal K\) into the local domain for every sufficiently close field. To see uniformity, outside the local domain the outward drift is bounded away from zero, while inside it the inverse is uniformly contracting. The extension formula (E.13), finite composition, and the inverse derivative bounds give (E.35) on \(\mathcal K\).

Finally, since \(\widetilde H(\widetilde q)=0\) and \(\widetilde H'(\widetilde q)=1\),
\[
\widetilde H(q)=-d_q+O(d_q^2).
\]
The integral of \(\widetilde H'-H'\) between \(q\) and \(x_{0,n}\) is at most \(C d_v/\sqrt n\). Multiplication by \(\sqrt n\) proves (E.36). \(\square\)

This lemma is local in the space of fields. Its constants can be large if the inverse orbit spends a long time away from the root, if a boundary approaches another equilibrium, or if \(\lambda\) is small. It does not assert stable prediction across a change in the number or type of equilibria.

## 4. Independent unlabeled field estimation

Use the sample (E.3) to define
\[
\widehat v_M(x)=\frac1M\sum_{i=1}^M U(x,\widetilde Z_i).
\tag{E.40}
\]
On the event that it has the corresponding root \(\widehat q_M\) on \(\mathcal K\), set
\[
\widehat\lambda_M=\widehat v_M'(\widehat q_M),\qquad
\widehat A_M=1+\eta\widehat\lambda_M,
\]
\[
\widehat\sigma_{g,M}^2
=\frac1M\sum_{i=1}^M U(\widehat q_M,\widetilde Z_i)^2,
\qquad
\widehat s_M^2
=\frac{\eta^2\widehat\sigma_{g,M}^2}{\widehat A_M^2-1}.
\tag{E.41}
\]
The sample mean at an exact root is zero, so the variance expression in (E.41) is the centered empirical variance. Let \(\widehat H_M\) be the normalized conjugacy of the estimated field. Root-selection failures and violations of the required geometry must be reported as inapplicable predictions, rather than removed from a success denominator.

For unconditional weak-convergence and expectation statements, complete otherwise undefined forecasts on the failure event by an arbitrary fixed valid joint exit law, for example exit at time one with equiprobable directions. This mathematical convention makes the random forecasts defined everywhere. Since the failure probability tends to zero, it does not change the limits below; it does not reclassify failed calibrations as successful predictions. Finite-sample bounds remain restricted to their stated good-calibration event.

### Proposition E.5. Field concentration, root concentration, and root CLT

For \(M\ge2\), \(0<\alpha<1/2\), write
\[
d_M(\alpha)=\sqrt{\frac{\log(C_0M/\alpha)}{M}},
\qquad
r_M(\alpha)=\sqrt{\frac{\log(C_0/\alpha)}{M}},
\tag{E.42}
\]
where \(C_0\) is a sufficiently large fixed constant.

With probability at least \(1-\alpha\),
\[
\|\widehat v_M-v\|_{C^3(\mathcal K)}
\le C d_M(\alpha).
\tag{E.43}
\]
For \(M\) sufficiently large that the right-hand side is below the perturbation threshold of Lemma E.4, the same event can be chosen so that
\[
|\widehat q_M-q|\le C r_M(\alpha),
\qquad
|\widehat\sigma_{g,M}^2-\sigma_g^2|\le C r_M(\alpha),
\tag{E.44}
\]
\[
|\widehat s_M-s|+|\log\widehat A_M-\log A|
\le C d_M(\alpha).
\tag{E.45}
\]
Also,
\[
\sqrt M(\widehat q_M-q)
=-\frac1{\lambda\sqrt M}\sum_{i=1}^M U(q,\widetilde Z_i)
+o_{\mathbb P}(1)
\Rightarrow N(0,\sigma_g^2/\lambda^2).
\tag{E.46}
\]
Statements about \(\widehat q_M\) are unaffected by assigning arbitrary values on the root-selection failure event, whose probability tends to zero.

**Proof.** For derivative orders \(j=0,1,2,3\), the variables \(\partial_x^j U(x,\widetilde Z_i)\) are uniformly bounded. Their sample averages and expectations are uniformly Lipschitz in \(x\), using the bounded derivative of order \(j+1\). Take a deterministic grid on \(\mathcal K\) with mesh at most \(1/M\); it contains at most \(C_{\mathcal K}M\) points. Hoeffding's inequality, a union bound over these points and four derivative orders, and an \(O(1/M)\) interpolation error prove (E.43).

On this event, Lemma E.4 selects a unique root near \(q\), and the sample derivative on the segment joining the roots is at least a fixed positive fraction of \(\lambda\). The exact root equation gives
\[
\widehat q_M-q
=-\frac{\widehat v_M(q)}
{\int_0^1\widehat v_M'(q+u(\widehat q_M-q))\,du}.
\tag{E.47}
\]
A separate Hoeffding bound at the single point \(q\) gives the first inequality in (E.44), without the grid logarithm. A pointwise bound for the average of \(U(q,\widetilde Z_i)^2\), together with
\[
|U(x,z)^2-U(q,z)^2|\le2B_UL_U|x-q|,
\]
gives the variance bound. Lemma E.4 and smoothness of the expression in (E.41), with \(\lambda,\sigma_g^2\) bounded away from zero in the fixed neighborhood, give (E.45). Split the confidence budget among these events to obtain the stated probability.

For the CLT, the denominator in (E.47) converges in probability to \(\lambda\). The numerator, multiplied by \(\sqrt M\), converges to \(N(0,\sigma_g^2)\). This proves (E.46). It is the usual estimating-equation expansion [E-R1], applied here to the entropy velocity. \(\square\)

### Theorem E.6. Prediction error for real SGD with independently estimated geometry

Assume G1–G4. All estimated quantities below are measurable with respect to a calibration sigma-field \(\mathcal C_M\) independent of the actual adaptation sample array. The initialization \(x_{0,n}\) remains the original deterministic initialization.

Let \(\widehat a_n,\widehat c_n\) be calibrated risk or interval endpoints. On a calibration event \(\mathcal E\), assume that the estimated field satisfies Lemma E.4, that
\[
d_b=\max(|\widehat a_n-a_n|,|\widehat c_n-c_n|)
\]
is sufficiently small, and that
\(\widehat\sigma_g^2>0\) is sufficiently close to \(\sigma_g^2\). Then the estimated root is inside the estimated interval with positive boundary distances. Define
\[
\widehat\mu_n=\sqrt n\,\widehat H(x_{0,n}),\qquad
\widehat B_n^+=\widehat H(\widehat c_n),\qquad
\widehat B_n^-=-\widehat H(\widehat a_n),
\]
\[
\widehat\Psi_n^\pm(t)
=\Phi\!\left(
\frac{\pm\widehat\mu_n-\sqrt n\,\widehat A^{-t}
\widehat B_n^\pm}{\widehat s}
\right).
\tag{E.48}
\]
These formulas predict the true exit from \(I_n\); they do not replace the true process by SGD driven by the estimated field.

On \(\mathcal E\), uniformly over integer \(t\ge0\),
\[
\max_\pm\left|
\Pr(\tau_n\le t,J_n=\pm1\mid\mathcal C_M)
-\widehat\Psi_n^\pm(t)
\right|
\le Ce_n+C\mathcal D_{M,n},
\tag{E.49}
\]
where
\[
\mathcal D_{M,n}
=|\widehat\mu_n-\mu_n|
+|\widehat s-s|
+\max_\pm|\log\widehat B_n^\pm-\log B_n^\pm|
+(1+\log n)|\log\widehat A-\log A|.
\tag{E.50}
\]
Let \(d_q=\widehat q-q\), \(d_v=\|\widehat v-v\|_{C^3(\mathcal K)}\), and
\(d_\sigma=|\widehat\sigma_g^2-\sigma_g^2|\). A bound in terms of the estimated inputs is
\[
\sup_{t\ge0}\max_\pm\left|
\Pr(\tau_n\le t,J_n=\pm1\mid\mathcal C_M)
-\widehat\Psi_n^\pm(t)
\right|
\le
C\left[
e_n+\sqrt n|d_q|+\sqrt n d_q^2
+(1+\log n)d_v+d_\sigma+d_b
\right].
\tag{E.51}
\]
The right sides may be truncated at one.

For the empirical calibration estimator of Proposition E.5, this yields, on its probability-\((1-\alpha)\) event and any simultaneous endpoint event,
\[
\sup_{t\ge0}\max_\pm\left|
\Pr(\tau_n\le t,J_n=\pm1\mid\mathcal C_M)
-\widehat\Psi_n^\pm(t)
\right|
\le C\left[
e_n+\sqrt n\,r_M(\alpha)
+\sqrt n\,r_M(\alpha)^2
+(1+\log n)d_M(\alpha)+d_b
\right].
\tag{E.52}
\]
An endpoint-event failure probability must be added separately; Proposition E.5 does not estimate risk endpoints.

**Proof.** Independence and the fixed initialization imply
\[
\Pr(\tau_n\le t,J_n=\pm1\mid\mathcal C_M)
=\Pr(\tau_n\le t,J_n=\pm1).
\tag{E.53}
\]
Apply Theorem E.3 to the right side. It remains to compare two deterministic Gaussian formulas, treating a realized calibration sample as fixed.

For \(B>0,a>0,s>0\), put
\[
f_t(\mu,s,B,a)
=\Phi\!\left(\frac{\mu-\sqrt n B e^{-at}}s\right).
\]
On the stated neighborhood, \(s,B,a\) are bounded above and away from zero. The true \(\mu_n\) is bounded. Changing \(\mu\) first costs at most \(C|\widehat\mu-\mu|\), with no bound needed on \(\widehat\mu\). With \(\mu\) thereafter fixed at its true value, the derivatives with respect to \(s\) and \(\log B\) are bounded uniformly in \(t,n\). The latter uses
\[
\sup_{y\ge0} y\,\varphi((\mu-y)/s)<\infty.
\]
For \(a=\log A\), the derivative has magnitude
\[
\frac{t y}{s}\varphi((\mu-y)/s),
\qquad y=\sqrt n B e^{-at}.
\]
Since
\[
t=\frac{\log(\sqrt n B)-\log y}{a},
\]
and both \(y\varphi((\mu-y)/s)\) and
\(y|\log y|\varphi((\mu-y)/s)\) are uniformly bounded, this derivative is at most \(C(1+\log n)\). The same estimates hold for the negative branch and along the parameter interpolation. The mean-value bound proves (E.49).

Lemma E.4 gives
\[
|\widehat\mu_n-\mu_n|
\le\sqrt n|d_q|+C(d_v+\sqrt n d_q^2),
\]
\[
\max_\pm|\widehat B_n^\pm-B_n^\pm|\le C(d_v+d_b),
\qquad |\log\widehat A-\log A|\le Cd_v.
\]
The boundary coordinates are bounded away from zero, so their logarithms satisfy the same local bound. Finally,
\[
|\widehat s-s|\le C(d_v+d_\sigma).
\]
This proves (E.51), and substitution of Proposition E.5 proves (E.52). \(\square\)

**Interpretation.** Peak estimation shifts a seed multiplied by \(\sqrt n\). Growth-rate error accumulates over a logarithmic number of updates. Field-shape and risk-endpoint errors change the transformed distance to the boundary. Equation (E.49) is a bound for the prediction of the actual nonlinear SGD, not a bound comparing two artificial Gaussian processes.

**Consistency consequence.** With exact endpoints, or with \(d_b\to_{\mathbb P}0\), empirical field calibration from the true \(Q\) gives
\[
\sup_{t\ge0}\max_\pm
\left|\Pr(\tau_n\le t,J_n=\pm1)-\widehat\Psi_n^\pm(t)\right|
\to_{\mathbb P}0
\quad\text{if }M/n\to\infty.
\tag{E.54}
\]
Indeed, (E.46) gives \(d_q=O_{\mathbb P}(M^{-1/2})\), while
\(d_v=O_{\mathbb P}(\sqrt{\log M/M})\). Under \(M/n\to\infty\), every term in (E.51) vanishes. The grid logarithm in (E.52) is not a claimed necessary sample-complexity factor.

This conclusion is for the stated empirical-root plug-in predictor under fixed positive \(\eta,\lambda\). It is not a minimax lower bound for all possible predictors.

## 5. A calibration limit law: consistent peaks can give inconsistent predictions

The threshold scale of the frozen SGD seed is \(s/\sqrt n\). The root CLT gives the corresponding calibration scale \(\sigma_g/(\lambda\sqrt M)\). Their squared ratio is
\[
\mathcal R_{M,n}^2
=\frac{n\sigma_g^2}{M\lambda^2s^2}
=\frac nM\frac{2+\eta\lambda}{\eta\lambda}.
\tag{E.55}
\]
The cancellation of \(\sigma_g^2\) follows from comparing two uncertainties generated by the same score law and the same update \(U\). It need not hold for calibration from a different distribution.

### Theorem E.7. Random limits of independently calibrated direction and time predictions

Assume G1–G4 and empirical calibration from the true \(Q\), with \(M=M_n\to\infty\). Suppose
\[
\frac{\sqrt n(x_{0,n}-q)}s\to z\in\mathbb R,\qquad
\mathcal R_{M,n}\to r\in[0,\infty).
\tag{E.56}
\]
Let \(\pi_n=\Pr(J_n=+1)\), and define the calibrated direction predictor
\[
\widehat\pi_{M,n}=\Phi(\widehat\mu_n/\widehat s_M).
\tag{E.57}
\]
Then
\[
\pi_n\to\Phi(z),\qquad
\widehat\pi_{M,n}\Rightarrow\Phi(z-rZ),
\quad Z\sim N(0,1).
\tag{E.58}
\]
The weak convergence in the second statement is over calibration samples, not over additional noise inserted into the adaptation recursion. For \(r>0\), this predictor is not consistent for \(\pi_n\).

For a time statement, suppose also
\[
a_n\to a,\qquad c_n\to c,\qquad a<q<c,\qquad
\max(|\widehat a_n-a_n|,|\widehat c_n-c_n|)
\to_{\mathbb P}0.
\tag{E.59}
\]
Write \(B^+=H(c)\), \(B^-=-H(a)\). Along any subsequence of batch sizes, still indexed by \(n\), and integer times \(t_n\) satisfying
\[
\sqrt n A^{-t_n}\to\zeta\in(0,\infty),
\tag{E.60}
\]
the true joint CDFs satisfy
\[
\Pr(\tau_n\le t_n,J_n=+1)\to\Phi(z-\zeta B^+/s),
\]
\[
\Pr(\tau_n\le t_n,J_n=-1)\to\Phi(-z-\zeta B^-/s).
\tag{E.61}
\]
The corresponding calibrated predictions converge jointly in distribution to
\[
\left(
\Phi(z-rZ-\zeta B^+/s),
\ \Phi(-z+rZ-\zeta B^-/s)
\right).
\tag{E.62}
\]
The same calibration normal \(Z\) appears with opposite signs in the two branches. The integer sequence condition (E.60) retains the discrete-time phase.

**Proof.** Equation (E.46) and (E.55) give
\[
\frac{\sqrt n(\widehat q_M-q)}s\Rightarrow rZ.
\tag{E.63}
\]
Finite \(r\) implies \(M\) is at least of order \(n\) eventually. Thus \(d_v\to_{\mathbb P}0\), \(\sqrt n d_q^2\to_{\mathbb P}0\), and Lemma E.4 implies
\[
\frac{\widehat\mu_n-\mu_n}s
=-\frac{\sqrt n(\widehat q_M-q)}s+o_{\mathbb P}(1).
\tag{E.64}
\]
Also \(\mu_n/s\to z\) and \(\widehat s_M/s\to_{\mathbb P}1\). Theorem E.3 and Slutsky's theorem prove (E.58).

For the time statement, \(t_n=O(\log n)\), and Proposition E.5 implies
\[
t_n|\log\widehat A_M-\log A|\to_{\mathbb P}0.
\]
Hence \(\widehat A_M^{-t_n}/A^{-t_n}\to_{\mathbb P}1\).
Lemma E.4 and (E.59) give \(\widehat B_n^\pm\to_{\mathbb P}B^\pm\). Substitute these facts, (E.63), and (E.64) into the true and calibrated formulas. Joint convergence follows because both calibrated branches use the same estimated root. \(\square\)

**Local bias variant.** The same proof applies if a different calibration scheme satisfies
\[
\frac{\sqrt n(\widehat q-q)}s\Rightarrow b+rZ,
\tag{E.65}
\]
while its other geometry estimates have the convergence and growth-rate control used above. Replace \(z-rZ\) by \(z-b-rZ\) in the positive branch and \(-z+rZ\) by \(-z+b+rZ\) in the negative branch. A fixed nonzero source-to-target peak bias does not satisfy this finite-\(b\) assumption; Section 6 treats that distinct regime.

### Corollary E.8. Explicit direction-prediction error at critical centering

Under Theorem E.7,
\[
\mathbb E_{\rm cal}\widehat\pi_{M,n}
\longrightarrow
\Phi\!\left(\frac z{\sqrt{1+r^2}}\right).
\tag{E.66}
\]
At \(z=0\),
\[
\mathbb E_{\rm cal}|\widehat\pi_{M,n}-\pi_n|
\longrightarrow\frac{\arctan r}{\pi}.
\tag{E.67}
\]
In particular, \(r=1\) gives limiting mean absolute probability error \(1/4\), even though \(\widehat q_M\to q\). At this value the limiting predictor is uniform on \((0,1)\).

**Proof.** All predictors lie in \([0,1]\), so weak convergence in (E.58) also gives convergence of their bounded continuous moments. For an independent standard normal \(Z_0\),
\[
\mathbb E\Phi(z-rZ)
=\Pr(Z_0+rZ\le z)
=\Phi(z/\sqrt{1+r^2}).
\]
At \(z=0\), symmetry gives
\[
f(r):=\mathbb E|\Phi(rZ)-1/2|
=2\int_0^\infty\{\Phi(ru)-1/2\}\varphi(u)\,du.
\]
Differentiation under the integral yields
\[
f'(r)=2\int_0^\infty u\varphi(ru)\varphi(u)\,du
=\frac1{\pi(1+r^2)}.
\]
Since \(f(0)=0\), (E.67) follows. For \(r=1\), \(\Phi(-Z)\) is uniform. \(\square\)

Equation (E.66) has the algebraic appearance of an inflated seed variance, but it is an average of random forecasts. The actual direction probability remains \(\Phi(z)\). Replacing the actual seed variance by \(s^2(1+r^2)\) would therefore answer a different question.

If \(M\to\infty\) but \(M/n\to0\), the same empirical-root predictor instead converges in distribution to a variable taking the values zero and one with probabilities \(1/2,1/2\), for every fixed finite \(z\). To verify this, apply (E.46) on the calibration scale: \(\sqrt M(x_{0,n}-q)\to0\), so the sign of \(x_{0,n}-\widehat q_M\) is asymptotically balanced. Its magnitude multiplied by \(\sqrt n\) diverges in probability away from an arbitrarily small neighborhood of zero. Monotonicity and the local derivative normalization of \(\widehat H_M\) preserve this sign and growth. The predictor becomes extreme across calibration repetitions; it does not become accurate.

| Relative calibration scale | Direction prediction in the critical regime |
|---|---|
| \(\mathcal R_{M,n}\to0\) | Consistent for the actual direction probability |
| \(\mathcal R_{M,n}\to r\in(0,\infty)\) | Nonvanishing random prediction error |
| \(\mathcal R_{M,n}\to\infty\), with \(M\to\infty\) | Random extreme predictions for the empirical-root estimator |

These statements fix \(Q,w,\eta,\lambda\). The factor in (E.55) can be large when \(\eta\lambda\) is small, but the results do not establish a joint limit with \(\eta\to0\) or \(\lambda\to0\).

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

## 10. Consequences for the current empirical evidence

The following numerical statements are taken from Sections 10.6–10.9 of the supplied manuscript revision. They were not rerun for this extension.

**Mixed Gaussian shifts.** The reported local-linear and conjugacy CDF errors, .0336 and .0339, indicate that nonlinear boundary transformation has little additional numerical benefit in that scan. This is consistent with theorems that use conjugacy to control approximation error without asserting that it improves every finite-parameter prediction.

**Non-Gaussian visual score laws.** On the reported common 133-cell subset, full target-geometry local linearization has mean CDF error .4804, compared with .0590 for full conjugacy. Theorem E.3 provides an appropriate analytical framework only after its interval and field assumptions are verified on the relevant empirical laws. It does not turn all cells into valid critical examples. In particular, one-sided safe components, a peak outside the initial safe component, and cross-component jumps require separate treatment.

**Independent calibration.** The reported improvement with increasing target calibration budget is consistent with the scale in (E.55). Theorems E.6–E.8 specify what a new calibration experiment should assess: the distribution of predicted probabilities over independent calibration repetitions, their error relative to the true conditional exit law, and the separate roles of field and endpoint estimation. Repeatedly scoring one stored set of adaptation paths does not create more independent adaptation trajectories.

**Source-to-target peak migration.** The reported cat/dog training and target peaks, .509827 and .527601, differ by a fixed .017774. For an initialization fixed at the training peak, this difference is not an \(O(n^{-1/2})\) target-critical sequence as \(n\to\infty\). The directional sensitivity at the tested batch sizes can be described through the normalized displacement, while Proposition E.9 supplies the appropriate fixed-bias limiting explanation.

**Remaining oracle errors.** The finite-\(n\) constants in Theorem E.3 can depend strongly on the entire exit domain. An accurate median with an inaccurate long-time tail is not ruled out by an uncalibrated asymptotic error rate. Before attributing residuals to one mechanism, distinguish numerical conjugacy error, a slow outward-drift region, a large transformed boundary, finite-batch noise away from the root, and disagreement between component exit and risk exit.

**Censoring and the evaluation horizon.** The supplied visual study records 13,102 of 108,000 paths without exit by the 2,000-update cap. For a deterministic cap \(T_{\max}\), let \(F^\pm(t)=\Pr(T_n\le t,J_n=\pm1)\) for the declared exit event. The observable joint empirical CDF uses all \(L\) paths:
\[
\widehat F_L^\pm(t)=\frac1L\sum_{i=1}^L
\mathbf1\{T_{n,i}\le t,\ J_{n,i}=\pm1\},
\qquad t\le T_{\max}.
\]
Each indicator is known even for a censored path, where it is zero. Dropping censored paths instead estimates a conditional distribution. Neither a small discrepancy on this recorded time range nor Theorem E.3's uncalibrated constant validates the unobserved tail. In particular, before accounting for Monte Carlo error, the eventual right-exit probability is only bounded by
\[
F^+(T_{\max})\le \Pr(T_n<\infty,J_n=+1)
\le F^+(T_{\max})+1-F^+(T_{\max})-F^-(T_{\max}).
\]
Thus a right-exit fraction at the cap is not automatically the eventual direction probability. If the recorded total CDF has not reached one half, its median is right-censored; an exit-only median answers a different question. All reported CDF errors in the supplied manuscript concern the recorded budget, whereas (E.18) is a mathematical all-integer-time statement under G1–G4.

**Risk information.** Target-oracle endpoints may be used to isolate field-estimation error in Theorem E.6. They must remain labeled as such. The theorem supplies no way to infer a label-dependent risk boundary from a general unlabeled marginal without additional information.

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

## 合并说明（供作者使用，不属于投稿正文）

本文件以现有英文稿为基础，给出独立的陈述与证明；不修改原稿，也不新增视觉实验结果。建议按以下方式合并。

| 现有位置 | 对应扩展 | 合并方式 |
|---|---|---|
| §8 的 Lemma 8 / Theorem 9 | E.1–E.3 | 用一般分布、显式几何条件重述核心退出定理；原高斯混合部分负责验证条件。E.3 的统一时间界需要保留 Step 5 的早时刻尾部论证 |
| §10.9 的根 CLT 与尺度比 | E.4–E.8 | 提升为“独立校准下的预测误差”理论部分；主文放 E.6、E.7 及 E.8 的直观推论，较长稳定性证明放附录 |
| 固定 source cancellation 的解释 | E.9 | 明确区分固定偏差与真正的临界参数序列，避免把所有 batch 扫描都解释为同一个对数极限 |
| 风险端点与经验安全集合局限 | E.10–E.12 | 增加端点扰动及跨分量跳跃的正式接口，继续区分经验分布事件与总体风险事件 |
| 校准方差是否进入种子 | E.13 | 可放附录或省略；其初始化被校准改变，与当前复用旧轨迹的实验不同 |

这里已经写出的定理都带有证明，但仍需要作者在整合时独立复核，并与实现中的具体事件、取整、根选择及端点约定逐项对齐。尤其需要核对：

1. 每个视觉设置是否在整个必要区间内满足 G1–G3，而不只是根附近曲率为正。没有通过的设置保持“不适用”，不能用对成功子集的理论解释替代总体适用率。
2. 当前保存的“越界”到底是风险首次超过比较器，还是离开初始安全分量；必要时报告预算内的跳跃事件比例，并使用 E.11。
3. 校准样本与 SGD 样本在条件于固定池之后是否独立；当前“仅预测、初始化不变”的实验使用 E.6–E.8，不能套用 E.13。
4. 风险端点误差单独保留。无标签熵场估计不提供其标签信息；使用 oracle 端点的结果继续标成 oracle。
5. 原稿的 \(M=w/4\) 建议统一改为 \(B_U=w/4\)，校准样本量使用 \(M\)。其他符号可在合并时重编号。
6. CDF 验证继续保留未退出轨迹的分母；预算内右退出比例不自动等于最终方向概率，删失中位数也不能由“仅已退出”中位数替代。E.7、E.13 的时间极限沿保留离散相位的子序列陈述。

建议主文形成“已知几何下的真实退出分布—独立估计几何下的预测误差—风险边界与经验池的适用范围”这一条理论线。原有弱/强分离几何与容忍度结果仍可保留；本扩展没有把一般非高斯边际下的微小容忍度、完整相边界或学习率联合极限写成已证明结论。

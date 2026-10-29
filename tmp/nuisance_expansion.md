### 6.1. Root sensitivity and the information bound

 Differentiate E_epsilon U(q_epsilon,Z)=0. At epsilon=0 the state derivative is lambda and the distribution derivative is E0 U0 h, proving (N.2). Differentiating log f_epsilon at zero gives score h and information E0 h^2. Cauchy–Schwarz gives (E0 psi h)^2<=E0 psi^2 E0 h^2. Its equality condition proves the last assertion. □

Thus the original h0=U0/sigma_g is not merely a convenient tilt: it maximizes normalized seed displacement for a given local statistical distinguishability. The term “least favorable” here concerns estimation of this target, after fixing the target window; it is not an ordering of all nonlinear environmental shifts.

If theta=u/sqrt(n) and the score h is known, its limiting seed is d_h u, while the calibration information is rho I_h. At a fixed seed window t=d_h u in [-K,K], the resulting Gaussian noise standard deviation is

\[
r_h=\frac{|d_h|}{\sqrt{\rho I_h}}\le\frac c{\sqrt\rho}.
\tag{N.4}
\]

For d_h=0 the first-order seed is zero, so this reparameterization is not applicable; a second-order analysis would be a different local problem. Comparing different h at a fixed u-window would change the attainable target window and would not be the same least-favorability comparison.

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

### 6.3. Upper and lower minimax arguments

Here B_loc denotes the envelope B used in the main statement, and the class has the same target window K.

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

Let R^K_{n,M} be the minimax expected absolute prediction loss over the declared unknown-direction class, using the exact pi_n as the target. Let

\[
V_G(K,r)=\inf_d\sup_{|t|\le K}E|d(t+rZ)-\Phi(t)|.
\]

**Lower bound.** The unknown-direction class contains the subfamily

\[
k_t=(t/c^2)\psi=(t/c)h_0,\qquad |t|\le K,
\]

by the envelope condition of Theorem 4. Its seed coordinate is exactly t, and it is the original entropy-gradient tilt with theta=t/(c sqrt(n)). Restricting to this subfamily and even telling the predictor that this restriction holds can only reduce the minimax risk. The experiment-level Gaussian reduction already proved for that subfamily gives liminf R^K>=V_G(K,c/sqrt(rho)). This invokes a full-experiment lower bound, not a lower bound inferred from a moment CLT.

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

### 6.4. Tangent projection and target-preserving nuisance

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

**Concrete nuisance example.** At the symmetric Gaussian baseline, let b(z)=cos(z)-E0 cos(Z). It is bounded, centered and even, whereas psi and h0 are odd. Thus E0 psi b=E0 h0 b=0. The two-direction perturbation k_{t,v}=(t/c)h0+v b has seed t, with an unknown bounded v changing other aspects of the score distribution. It stays in the declared unknown-direction class whenever |t| H_*/c+|v| ||b||_infinity<=B_loc. This supplies a legal common-risk nuisance family with no additional first-order penalty. No new nuisance simulation is claimed in the present confirmation run.

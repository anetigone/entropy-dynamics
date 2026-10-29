# Local prediction with unknown entropy dynamics

Part of the [manuscript](DRAFT.md). This chapter gives an explicit common-risk family, a uniform direction-exit proof, and statistical necessity and sufficiency within that family. Numbering L.1–L.8 is local to this chapter. The model is scalar, the step size is fixed, and the calibration and future adaptation samples are independent.

## 8B.1. Observation experiment

Write s(u)=(1+exp(-u))^{-1}, a(u)=s(u)(1-s(u)), and
\[
U(x,z)=-w^2(z-x)a(w(z-x)),\quad
x_{t+1}=x_t+\frac\eta n\sum_{i=1}^n U(x_t,Z_{t,i}),\quad x_0=0.
\]
Fix w>0. Future batches are fresh iid scores from an unknown environment Q. A predictor observes only M independent calibration scores, x0, the common model family and its constants, and, if desired, the entire common classification-risk function. Future SGD samples, trajectories, root, and environment identity are unavailable. Any source information is fixed across environments. Randomized predictors may use independent randomization.

For a positive fixed tolerance epsilon define
\[
T_\epsilon=\inf\{t\ge0:R_Q(x_t)>R_Q(0)+\epsilon\}.
\]
Below the safe set is exactly [-b,b], T_epsilon is finite almost surely, and
\[
\pi_n(Q)=\Pr_Q(x_{T_\epsilon}>b).
\]
The probability integrates over independent future SGD; prediction loss averages over calibration randomness. Online trajectory prediction is a different experiment.

Known exact centering admits the asymptotically correct constant forecast .5. Thus the centered plug-in error arctan(r)/pi in [calibration](calibration.md) is not a universal lower bound. Here initialization stays fixed while the unknown environment moves the root.

## 8B.2. Explicit common-risk family

### Proposition L.1. Legal environments and local geometry

Fix nu>0 and 0<kappa<1. Let f0 be the N(0,nu²) density, and put
\[
v_0(x)=E_0 U(x,Z),\quad \lambda=v_0'(0),\quad
\sigma_g^2=E_0 U(0,Z)^2,\quad h(z)=U(0,z)/\sigma_g.
\]
Then
\[
\lambda=\frac{w^2}{\nu^2}E_0[Z^2a(wZ)]>0,\qquad \sigma_g>0,
\quad E_0h=0,\quad E_0h^2=1,\quad H_*:=\|h\|_\infty<\infty.
\]
Choose a sufficiently small d>0 such that v0'(x)>=lambda/2 on K=[-d,d], and set b=d/2. Choose and then fix
\[
0<\eta<\min\{1/L_U,d/(4B_U)\},\quad B_U=w/4,\quad L_U=w^2/4.
\]
There exists bar_theta>0 such that for |theta|<=bar_theta the joint densities
\[
f_\theta=f_0(1+\theta h),\qquad g(z)=\kappa\tanh(z)f_0(z),
\qquad p_\theta(z,\pm1)=(f_\theta(z)\pm g(z))/2
\]
are nonnegative and normalized, with equal class priors. The score law Q_theta has density f_theta. All environments share the entire risk function
\[
R(x)=\frac12+\int_{-\infty}^x g(z)\,dz,
\quad R(x)-R(0)=\kappa\int_0^x\tanh(z)f_0(z)\,dz.
\]
In particular, with the fixed choice
\[
\epsilon=\kappa\int_0^b\tanh(z)f_0(z)\,dz>0,
\quad \{x:R(x)\le R(0)+\epsilon\}=[-b,b].
\]
The field v_theta has exactly one root q_theta in K, with |q_theta|<=b/2 and v_theta'>=lambda/4 throughout K. Moreover,
\[
q_\theta=-\frac{\sigma_g}{\lambda}\theta+O(\theta^2).
\tag{L.1}
\]
All constants and the remainder are uniform on a sufficiently small parameter interval.

**Proof.** Symmetry makes U(0,z) odd and v0(0)=0. Since U(x,z) is a translate of U(0,z), Gaussian integration by parts gives
v0'(0)=-E0[partial_z U(0,Z)]=-E0[Z U(0,Z)]/nu², yielding the displayed positive formula. The variance is positive because U(0,Z) is not constant. Boundedness and smoothness follow from Lemma E.1 in [scalar exit](scalar_exit.md).

Require bar_theta H_*<=min(1/2,1-kappa). Then f_theta>=kappa f0>=|g|, and E0h=0 normalizes f_theta. Since g is odd, its integral is zero and each class has mass 1/2. Integrating the two misclassification regions gives the displayed risk. Its derivative g is negative on the negative half-line and positive on the positive half-line, so its even safe set is exactly the stated interval, globally. Thus risk exit equals interval exit path by path, including discrete overshoots; there is no second safe component.

Write v_theta=v0+theta k, k(x)=E0[U(x,Z)h(Z)]. All fixed-order derivatives of k are bounded, and k(0)=sigma_g. Shrinking bar_theta gives v_theta'>=lambda/4 on K and opposite signs at +/-b/2. The implicit-function theorem gives (L.1), with a bounded second derivative on the reduced closed parameter interval. Root noise variance is continuous and stays bounded below by a positive constant after another reduction. The full one-step overshoot interval [-b-eta B_U,b+eta B_U] lies in [-3d/4,3d/4], inside K. These choices establish common geometry rather than assuming it. This family is a general scalar joint model; its label-conditionals are not the original Gaussian mixture. □

## 8B.3. Uniform direction exit for actual entropy SGD

Define
\[
\lambda_\theta=v_\theta'(q_\theta),\quad A_\theta=1+\eta\lambda_\theta,
\quad \sigma_\theta^2=\operatorname{Var}_\theta U(q_\theta,Z),
\quad s_\theta^2=\frac{\eta^2\sigma_\theta^2}{A_\theta^2-1}.
\]
At theta=0 write A=1+eta lambda, s=s0, and
\[
c=\frac{\sigma_g}{\lambda s}=\sqrt{\frac{2+\eta\lambda}{\eta\lambda}},
\qquad e_n=n^{-1/2}+((1+\log n)/n)^{1/3}.
\]

### Lemma L.2. Uniform local direction law

For every fixed finite A0>0 there are C and n0 such that
\[
\sup_{|u|\le A_0}|\pi_n(Q_{u/\sqrt n})-\Phi(cu)|\le C e_n
\quad(n\ge n_0).
\tag{L.2}
\]
This statement concerns fixed eta; its constants are not uniform as eta tends to zero.

**Proof.** We supply the common constants and the stopped-seed argument, so this is not an application of a pointwise exit limit to changing laws.

Construct normalized C² conjugacies H_theta with H_theta(q_theta)=0, H_theta'(q_theta)=1 and H_theta(G_theta(x))=A_theta H_theta(x) whenever both arguments are in K. The uniform inverse construction is as follows.

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
A_theta^{j+1} r_theta(y_j). Their functions, first derivatives and second derivatives have summable bounds C(A_max beta^2)^j. This proves local C^2 convergence. To establish positivity of the limiting derivative, use its derivative product:

\[
\prod_{k=0}^{j-1} A_\theta\widetilde F_\theta'(y_k).
\]

Each factor is positive and differs from one by O(|y_k|). On a sufficiently small common neighborhood the sum of these deviations is uniformly bounded, so the limiting product is bounded away from zero. A fixed number of inverse iterations maps all of K into that neighborhood, since F_theta contracts by gamma. Finite composition gives uniform C^2 bounds and a positive derivative lower bound on K. This also verifies the functional equation wherever its two arguments are in K.

Thus, for constants independent of theta,
\[
\|H_\theta\|_{C^2(K)}\le C,\quad \inf_K H_\theta'\ge m>0,
\quad 1<A_{\min}\le A_\theta\le A_{\max},\quad
0<s_{\min}\le s_\theta\le s_{\max}.
\tag{L.3}
\]
The inverse derivative bounds use the uniform smoothness of U and the positive derivative of G_theta. The root-to-boundary distances are at least b/2. These facts also give uniform bounds above and below for the absolute conjugate boundary values.

Let tau be exit from [-b,b] and xi_j(x)=n^{-1}sum_i U(x,Z_{j,i})-v_theta(x). Each theta has finite tau almost surely: choose two fixed bounded score intervals respectively left and right of K. Their masses are uniformly positive since f_theta>=f0/2. Repeated batches entirely in the left interval drive any state at least q_theta rightward by a positive minimum speed until exit; the right interval does the converse. Compactness supplies a fixed finite block length N. The conditional chance of an appropriate exiting block is at least p^{nN}>0; geometric blocking proves finiteness. This is not a bound on the typical exit time.

For |theta|<=A0/sqrt(n), (L.1) implies |q_theta|<=C/sqrt(n) and mu=sqrt(n)H_theta(0) is bounded. Conditional batch moments give
\[
E[\xi_j(x)^2\mid\mathcal F_j]\le C/n,\quad
E[\xi_j(x)^4\mid\mathcal F_j]\le C/n^2,\quad
E[(\xi_j(x)-\xi_j(q_\theta))^2\mid\mathcal F_j]
\le C|x-q_\theta|^2/n.
\]
All constants are distribution independent, by bounded U and partial_x U. Taylor expansion on the established overshoot domain gives, for j<tau,
\[
H_\theta(x_{j+1})=A_\theta H_\theta(x_j)
 +\eta H_\theta'(G_\theta(x_j))\xi_j(x_j)+r_j,
\quad |r_j|\le C\xi_j(x_j)^2.
\]
Set Y_t=sqrt(n) A_theta^{-(t wedge tau)}H_theta(x_{t wedge tau}).

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

Before exit, H_theta'>=m and bounded K also imply
\[
E[\mathbf1_{j<\tau}|x_j-q_\theta|^2]\le C\min(A_\theta^{2j}/n,1).
\tag{L.4}
\]

On the same infinite sample array introduce the frozen seed
\[
W=\mu+\eta\sum_{j\ge0}A_\theta^{-j-1}\sqrt n\,\xi_j(q_\theta).
\]
The difference H_theta'(G_theta(x_j))xi_j(x_j)-xi_j(q_theta) has conditional mean zero and variance at most C|x_j-q_theta|²/n, using H_theta'(q_theta)=1 and (L.3). Consequently its stopped weighted sum has second moment at most
\[
C\sum_{j\ge0}A_\theta^{-2j}\min(A_\theta^{2j}/n,1)
\le C(1+\log n)/n.
\]
Use the convention that batch j moves x_j to x_{j+1}. At tau, all batches with indices j>=tau remain fresh. Conditional on F_tau, the frozen tail has mean zero and variance

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
Grouping W-mu by sample column gives n^{-1/2}sum_i V_i with iid centered
V_i=eta sum_{j>=0} A_theta^{-j-1}U(q_theta,Z_{j,i}). These variables have uniform absolute bounds eta B_U/(A_min-1), and variance s_theta² bounded below. Berry–Esseen gives a uniform C/sqrt(n) Kolmogorov error relative to N(mu,s_theta²). In particular P(|W|<=delta)<=C delta+C/sqrt(n).

The sign of Y_tau is exactly the exit direction. Hence (L.5), Chebyshev, and the preceding anti-concentration bound give
\[
|\pi_n(Q_\theta)-\Phi(\mu/s_\theta)|
\le C\{n^{-1/2}+\delta+(1+\log n)/(n\delta^2)\}\le C e_n,
\]
choosing delta=((1+log n)/n)^{1/3}. This explicitly controls seeds near zero, rather than identifying their signs without an error bound. Finally normalization and the uniform C² bound imply H_theta(0)=-q_theta+O(q_theta²); smoothness gives s_theta=s+O(|theta|). Uniformly on |u|<=A0,
\[
\mu/s_\theta=cu+O(n^{-1/2}).
\]
The Lipschitz property of Phi proves (L.2). □

## 8B.4. Statistical threshold within the same family

Let
\[
\mathfrak R_{n,M}(A_0)=\inf_{\widehat\pi}
\sup_{|u|\le A_0}E_{u/\sqrt n}
|\widehat\pi-\pi_n(Q_{u/\sqrt n})|,
\]
where A0>0 and predictors have the information in §8B.1. The family and its constants are known; u is unknown.

### Theorem L.3. Necessity with explicit constants

If eventually M/n<=C0<infinity, choose
\[
0<a\le\min\{A_0,1/(4\sqrt{\max(C_0,1)})\}.
\]
Then
\[
\liminf_n\mathfrak R_{n,M}(A_0)\ge [2\Phi(ca)-1]/4>0.
\tag{L.6}
\]

**Proof.** At theta=+/-a/sqrt(n), positivity gives
\[
D_{KL}(f_\theta\|f_{-\theta})\le\chi^2(f_\theta,f_{-\theta})
=4\theta^2 E_0\frac{h^2}{1-\theta h}\le8\theta^2.
\]
Product KL is at most 8a²M/n. Pinsker gives total variation at most 2a sqrt(M/n)<=1/2. Lemma L.2 implies the target separation Delta_n tends to Delta=2Phi(ca)-1>0. Let p_+,p_- be the calibration densities and t_+,t_- their true exit probabilities. For any forecast T, the triangle inequality on their common density overlap gives
\[
E_+|T-t_+|+E_-|T-t_-|
\ge\int\min(p_+,p_-)(|T-t_+|+|T-t_-|)
\ge\Delta_n(1-\operatorname{TV}).
\]
The maximum risk is at least half this sum, hence at least Delta_n/4 when TV<=1/2. This proves (L.6). Common risk and fixed source information have identical values and cannot change this comparison. Independent randomization cannot improve the testing bound. □

### Theorem L.4. Sufficiency and equivalence

The predictor
\[
\widehat\theta=M^{-1}\sum_{j=1}^M h(\widetilde Z_j),\qquad
\widehat\pi=\Phi(c\sqrt n\,\widehat\theta)
\]
satisfies, for M>=1 and large n,
\[
\sup_{|u|\le A_0} E_{u/\sqrt n}|\widehat\pi-\pi_n(Q_{u/\sqrt n})|
\le \frac{c}{\sqrt{2\pi}}\sqrt{n/M}+Ce_n.
\tag{L.7}
\]
Thus uniform consistency within this family is possible if and only if M/n tends to infinity.

**Proof.** E_theta h=theta exactly, and Var_theta h=1-theta²<=1 by oddness of h (see L.5). Hence E|hat_theta-theta|<=1/sqrt(M). Apply the global Lipschitz constant 1/sqrt(2pi) of Phi and (L.2). No clipping or root oracle is used. Conversely, if M/n does not diverge, some infinite subsequence has bounded M/n; Theorem L.3 on that subsequence excludes risk tending to zero. □

These consistency statements are specific to the known family and fixed eta. The following refinements distinguish minimax conclusions from the risk of an individual predictor.

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

Consequently, under the eventual bound and choice of a in L.3, this recovers

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

Independent randomization can be included in the common measure and leaves TV unchanged. The maximum of the two risks is at least half their sum. L.2 implies |t_+-t_-|>=Delta0(a)-2r_n, and the KL/Pinsker calculation in L.3 gives TV<=2a sqrt(rho_n). This proves (L.11). Under TV<=1/2 the bound is Delta_n/4, as used in L.3. A nearest-target testing reduction would lose a factor of two. □

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

## 8B.5. Empirical interpretation and contribution boundary

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

## 8B.6. Contribution and open questions

We study the independent information required to predict the direction of a future risk exit under entropy adaptation. We construct a scalar family with fixed initialization and an identical complete classification-risk function across environments, while the unknown unlabeled marginal moves an unstable entropy root on the mini-batch noise scale. A uniform approximation for the actual nonlinear SGD direction connects this dynamical sensitivity to a statistical estimation problem. Within this known family and at fixed positive step size, uniform consistency is possible exactly when the independent calibration size grows faster than the SGD batch size. When their ratio tends to zero, the exact limiting minimax risk is attained by the constant forecast one half. At positive finite ratios, an explicit moment predictor instead has a nonvanishing worst-case risk limit governed by an arctangent law; this predictor-specific limit is not a minimax lower bound.

The finite-ratio problem is continued in §8B.7 below. Global sharp rates below the exit-approximation remainder, robustness to a misspecified tilt direction, broader unknown score families, joint vanishing-step limits and neural-network adaptation remain separate questions. Literature-priority assessment remains necessary.

## 8B.7. Finite-information minimax precision

Read the [Gaussian decision extension](../gaussian_decision/gaussian_decision_extension.md) next as a continuation of this manuscript. It retains the full proofs of Theorem L.9 and Corollary L.10, with its own G-equation numbering, and the numerical analysis in its original location. The [reproduction entry](../gaussian_decision/README.md) documents the scripts and evidence boundaries.

Theorem L.9 constructs parameter-independent Markov kernels in both directions and proves asymptotic equivalence, uniformly over the fixed local window, between the full calibration experiment and Y=u+Z/sqrt(rho) when M/n tends to rho in (0,infinity). Its likelihood-ratio L1 control is stronger than the moment-statistic CLT in L.8. Corollary L.10 transfers bounded absolute loss and the uniform exit approximation to obtain
\[
\mathfrak R_{n,M}(A_0)\longrightarrow
V_G(cA_0,c/\sqrt\rho),\qquad
V_G(K,r)=\inf_d\sup_{|t|\le K} E_t|d(t+rZ)-\Phi(t)|.
\]
The extension gives finite-decimal prior lower bounds and continuous-window rule upper bounds for six Gaussian problems, with directed probability calculations and crossing brackets. For K=r=1 the outward six-decimal bracket is [0.164401,0.164587], compared with the raw moment risk .25. The certified candidate rule therefore improves Gaussian worst-case MAE by at least 34.16%; this is not pointwise dominance. It does not require the searched prior to be exactly least favorable.

The independent entropy-SGD pilot contains ten cells, 10,240 reference paths and no censored paths. The largest empirical five-point MAEs for raw moment versus optimized rules are .253404 versus .167207 at n=128 and .267974 versus .178871 at n=512. These empirical maxima support finite-batch relevance but are neither continuous-window bounds nor validation of the narrow Gaussian minimax brackets. Reference-probability uncertainty is additional to conditional paired calibration standard errors. The full tables and their interpretation remain in the [original analysis](../gaussian_decision/gaussian_decision_extension.md#7-results).

The minimax convergence is asymptotic: the proof does not numerically calibrate the experiment deficiency or the SGD exit remainder. The pilot's geometry quadrature is diagnostic, separate from the directed Gaussian certificate. Implementation details, prior/rule specifications, original outputs and raw pilot arrays stay in the original directory.

## 8B.8. Unknown local directions and fixed-design confirmation

Continue with [N.1–N.4 and their complete proofs](../entropy_prediction_revision/contribution_precision/unknown_direction_theory.md). Around the same known baseline, let f_{n,k}=f0(1+k/sqrt(n)), E0 k=0, ||k||_infinity<=B_loc, and |E0 psi k|<=K, with psi=U(0,.)/(lambda s) and E0 psi²=c². Provided B_loc>=K ||psi||_infinity/c², the class contains k_t=t psi/c² for every |t|<=K. Its finite-ratio minimax value is the same V_G(K,c/sqrt(rho)), and its uniform-consistency threshold remains M/n tending to infinity.

The lower bound restricts to the contained least-favorable subfamily. The upper bound applies Gaussian rules to the direction-free statistic sqrt(n) mean(psi). A uniform exit proof and uniform scalar CLT suffice for this upper bound; no equivalence of the entire infinite-dimensional experiment is asserted. The sensitivity bound <psi,h>²/E0 h²<=c² identifies the entropy-gradient direction as least favorable for a fixed seed window. Known baseline constants, a fixed bounded envelope and n^(-1/2) locality are essential. The nuisance projection formulas and their admissibility qualifications remain in N.4.

The [independent confirmation analysis](../entropy_prediction_revision/contribution_precision/precision_report.md) supersedes the pilot as the precision evidence: it uses the frozen Gaussian rule and ten fixed cells, with simultaneous intervals that account for both reference trajectories and calibration repetitions. It tests the original tilt family, not the new unknown-direction or nuisance class. Its full tables and raw data remain at their original location. The [contribution comparison](../entropy_prediction_revision/contribution_precision/contribution_and_literature.md) separates these results from prior entropy-collapse, monitoring and unlabeled action-selection results.

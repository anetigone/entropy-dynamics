# Gaussian-model proofs

Proof appendix to [the main manuscript](DRAFT.md). Definitions, assumptions, and original result numbers are retained there; the general scalar extension has its own E-numbering.

## Proposition 1. Risk geometry and uniform update bounds

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

## Lemma 2. The entropy kernel is log-concave

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

## Theorem 3. Weak separation gives a unique unstable entropy peak

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

## Corollary 4. An explicit mixed-shift rebound region

**Proof.** Here \(x_0=-\delta\in[0,x_*)\) and \(q_p<0\). Theorem 3 gives rightward escape. The first step is at most \(\eta M<x_*-x_0\), so it stays in the strict risk-decrease region. Proposition 1 gives entropy descent. Finally,
\[
r_p(x_0)\le r_p(0)=\Phi(-m/\sigma)<p-\epsilon.
\]
Convergence of risk to \(p\) therefore forces a finite strict crossing. \(\square\)

## Theorem 5. A sufficiently separated safe population basin

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

## Theorem 6. Fixed and shrinking tolerances have different time scales

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

## Theorem 7. Two-sided first-exit control

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

## Lemma 8. A normalized deterministic conjugacy

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

## Theorem 9. State-dependent SGD admits a Gaussian-seed exit approximation

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

## Corollary 10. The critical joint limit

**Proof.** Peak and conjugacy expansions give
\(\sqrt nH_{\rho_n}(x_{0,n})\to\mu\). Continuity gives
\(s_n\to s_\eta\) and \(B_n^\pm\to C_\epsilon\).
Since \(\log A_{\rho_n}-\log A=O(\rho_n)\) and \(t_n=O(\log n)\),
\(A_{\rho_n}^{-t_n}/A^{-t_n}\to1\). Substitute into Theorem 9. Evaluating the two tails a sufficiently large fixed number of steps on either side of the center proves tightness. \(\square\)

## Theorem 11. Inverse-batch tolerance gives finite-step AR exit

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

## Proposition 12. Zero tolerance and a limitation of shrinking the step

**Proof.** At exact balance, zero is the unique risk minimum. The analytic nonconstant function \(U(0,z)\), applied to a continuously distributed feature, has no atoms: each level set has Lebesgue measure zero. An independent sample average is therefore nonzero almost surely, proving first-step violation.

For positive prior bias, the risk minimizer is right of zero. Any negative empirical velocity moves left and strictly increases risk for every positive step size. The CLT
\[
\sqrt n\,n^{-1}\sum_iU(0,z_i)\Rightarrow N(\beta c,s_0^2)
\]
proves the lower bound independently of \(\eta_n\). For fixed \(\eta\), improvement requires the limiting scaled first iterate to lie between \(0\) and \(2k_0c\), giving the displayed Gaussian interval probability. \(\square\)

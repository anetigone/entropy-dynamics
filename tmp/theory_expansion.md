### 4.1. Proof of the uniform direction law

We supply the common constantsand the stopped-seed argument, so this is not an application of a pointwise exit limit to changing laws.

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

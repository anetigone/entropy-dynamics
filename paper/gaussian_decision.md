# Finite-information prediction: Gaussian reduction and minimax certificates

Part of the [manuscript](DRAFT.md), continuing [L.1–L.8](local_prediction.md). This chapter contains L.9–L.10 and the Gaussian decision certificate argument. Numerical results and pilot analysis are in [the result directory](../experiments/results/gaussian_decision/analysis.md).

## 1. Observation experiment and notation

Let the calibration observations have density f_theta=f0(1+theta h), with bounded h, E0 h=0 and E0 h^2=1. In the explicit entropy construction, f0 is a centered Gaussian and h(z)=U(0,z)/sigma_g. For each n the parameter is u in the fixed compact interval [-A0,A0], theta=u/sqrt(n), and the calibration experiment is

\[
\mathcal E_n=\{P_{n,u}=Q_{u/\sqrt n}^{\otimes M}:|u|\le A_0\}.
\]

Assume M/n->rho in (0,infinity). The true target is the probability pi_n(Q_{u/sqrt(n)}) of right risk exit by a future, independent entropy-SGD trajectory. The supplied uniform exit lemma gives

\[
r_n:=\sup_{|u|\le A_0}|\pi_n(Q_{u/\sqrt n})-\Phi(cu)|\to0.
\]

Here c=sqrt((2+eta lambda)/(eta lambda)) is fixed. Define the Gaussian experiment

\[
\mathcal G_\rho=\{Y=u+\rho^{-1/2}Z:|u|\le A_0\},\qquad Z\sim N(0,1).
\]

Throughout this extension TV means sup_B |P(B)-Q(B)|, or half the L1 distance of densities. Nussbaum's reference uses the full L1 convention, so constants must be translated when comparing definitions. We use deficiency as the infimum over parameter-independent Markov kernels of the supremum TV approximation error, and Delta as the maximum of the two deficiencies.

Reference: Michael Nussbaum, *Asymptotic Equivalence of Statistical Experiments*, especially the definitions of deficiency and its decision-theoretic interpretation, pp. 4–6: https://pi.math.cornell.edu/~nussbaum/papers/04-1-transl.pdf . The proof below is specialized to the present bounded tilt family; the reference is not being cited as proving this model-specific result.

## 2. Theorem L.9: equivalence of the entire local experiment

For the explicit family and every fixed A0<infinity,

\[
\boxed{\Delta(\mathcal E_n,\mathcal G_\rho)\to0.}
\tag{G.1}
\]

The assertion supplies both directions of risk transfer, not merely a CLT for the empirical moment.

**Proof.** Write P_{n,0}=Q0^{otimes M}, rho_n=M/n, and

\[
S_n=M^{-1/2}\sum_{j=1}^M h(Z_j),\qquad
V_n=M^{-1}\sum_{j=1}^M h(Z_j)^2.
\]

Let F_M be the CDF of S_n under P_{n,0}. In the present family h is a nonconstant real-analytic function, so the preimage of any value is a Lebesgue-null set. Since f0 has a density, h(Z) has no atoms; the sum also has no atoms. Therefore F_M is continuous. Define

\[
T_n=\Phi^{-1}(F_M(S_n)).
\]

This parameter-independent statistic has the exact N(0,1) law under P_{n,0}; events where the argument is zero or one have probability zero. The baseline law and the transform are known because the model family is known.

The CLT and continuity of Phi give sup_x|F_M(x)-Phi(x)|->0. On a compact interval for S_n the inverse normal CDF is uniformly continuous after restricting to a slightly larger compact probability interval. Tightness of S_n then proves

\[
S_n-T_n\longrightarrow0\quad\text{in }P_{n,0}\text{-probability}.
\tag{G.2}
\]

This use of a CLT is only one part of the proof. The essential next step controls the full likelihood ratios.

Let Lambda_n(u)=dP_{n,u}/dP_{n,0}. Boundedness of h and compactness of u allow a uniform Taylor expansion, valid for all sufficiently large n:

\[
\log\Lambda_n(u)
=u\sqrt{\rho_n}S_n-\tfrac12u^2\rho_n V_n+R_n(u),
\qquad
\sup_{|u|\le A_0}|R_n(u)|\le C M n^{-3/2}.
\tag{G.3}
\]

Under P_{n,0}, V_n->1 in probability, and rho_n->rho. Equations (G.2)–(G.3) imply

\[
\sup_{|u|\le A_0}
\left|\log\Lambda_n(u)-\left(\sqrt\rho\,uT_n-\tfrac12\rho u^2\right)\right|
\longrightarrow0
\tag{G.4}
\]

in probability. Define g_u(t)=exp(sqrt(rho)u t-rho u^2/2). We now strengthen convergence of log likelihoods to uniform L1 convergence of likelihoods.

For any fixed p>1, log(1+x)<=x gives

\[
\sup_{|u|\le A_0}\Lambda_n(u)
\le\exp\left(A_0\left|\sum_jh(Z_j)\right|/\sqrt n\right).
\]

The centered bounded summands have a sub-Gaussian exponential-moment bound. Since M/n is bounded, the pth moments of the last display are uniformly bounded. Also sup_u g_u(T_n)<=exp(sqrt(rho)A0|T_n|), whose pth moment is bounded because T_n is exactly standard normal. Equation (G.4), tightness, and these bounds give uniform integrability and hence

\[
\varepsilon_n:=\tfrac12\mathbb E_{n,0}
\sup_{|u|\le A_0}|\Lambda_n(u)-g_u(T_n)|\longrightarrow0.
\tag{G.5}
\]

Define an auxiliary experiment on the original sample space by

\[
d\widetilde P_{n,u}=g_u(T_n)\,dP_{n,0}.
\]

These are probability measures, because E_{n,0}g_u(T_n)=1 exactly. Equation (G.5) bounds sup_u TV(P_{n,u},Ptilde_{n,u}) by epsilon_n.

Under Ptilde_{n,u}, T_n has law N(sqrt(rho)u,1). Conversely, use the regular conditional distribution P_{n,0}(d z|T_n=t) to turn such a normal observation back into a full calibration sample. This kernel is independent of u; tilting by a function of T_n leaves the conditional law unchanged. Regular conditional distributions exist on the Euclidean sample spaces, and arbitrary versions on baseline-null sets do not matter because all normal shifts are mutually absolutely continuous. Thus the auxiliary experiment is exactly equivalent to observing T~N(sqrt(rho)u,1), or Y=T/sqrt(rho).

Applying the statistic T_n in one direction and this baseline conditional kernel in the other gives both deficiencies bounded by epsilon_n. This proves (G.1). □

The proof supplies asymptotic equivalence on a fixed compact local parameter interval. It does not give a calibrated numerical value of epsilon_n, nor global equivalence over an expanding parameter range. The conditional kernel is an existence construction; implementing it is not required to use a decision rule in the forward direction.

## 3. Corollary L.10: the finite-ratio minimax limit

Let

\[
V(A_0,c,\rho)=\inf_\delta\sup_{|u|\le A_0}
 \mathbb E_u|\delta(Y)-\Phi(cu)|.
\]

Then the original minimax risk for predicting the true future SGD direction probability satisfies

\[
\boxed{\mathfrak R_{n,M}(A_0)\longrightarrow V(A_0,c,\rho).}
\tag{G.6}
\]

**Proof.** Project all decision values to [0,1], which cannot increase loss. Absolute loss is then bounded by one, so transporting any randomized rule through either Markov kernel changes each parameterwise risk by at most epsilon_n. Replacing the target pi_n by Phi(cu) changes each loss by at most r_n. Taking supremum risks and then infima in both directions gives

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

Actions may be restricted further to [Phi(-K),Phi(K)]. Reflection and averaging permit a symmetric rule d(-x)=1-d(x) without increasing worst-case risk, although the certificates below do not require a proof that the candidate optimizer is minimax.

## 4. Certified numerical brackets without claiming optimizer optimality

### 4.1. A prior produces a valid lower bound

Choose any finite prior with ordered support t_1<...<t_m in [-K,K] and positive weights w_i. Its posterior median under absolute loss is a Bayes rule taking the values Phi(t_i). The boundary between actions k and k+1 solves

\[
\sum_{i\le k}w_i e^{x t_i/r^2-t_i^2/(2r^2)}
=\sum_{i>k}w_i e^{x t_i/r^2-t_i^2/(2r^2)}.
\tag{G.8}
\]

The ratio of the left sum to the right sum is strictly decreasing: its log derivative is the difference of two tilted means divided by r^2, and every left support point is smaller than every right support point. Its limits are infinity and zero. Thus every crossing is unique; the ordered crossings define the posterior-median rule.

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

Indeed the target's first and second derivatives are bounded in absolute value by one; the sum of absolute bin-probability derivatives is bounded by E|Z|/r<=1/r, and that of second derivatives by E|Z^2-1|/r^2<=2/r^2. All action losses are bounded by one. The bound is deliberately conservative.

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

## 5. Using the computed Gaussian rule on calibration data

For an arbitrary measurable Gaussian rule, the direct risk-transfer construction applies it to c T_n/sqrt(rho). This uses the known baseline CDF F_M, not the unknown environment parameter.

For each of the fixed finite step rules produced here, a simpler implementation is also asymptotically justified: apply it to Xhat=c sqrt(n) hat_theta, the raw empirical moment coordinate. Under the baseline, Xhat-c T_n/sqrt(rho) tends to zero in probability. The likelihood-ratio moment bounds used in (G.5) transfer this to the compact local family uniformly. The Gaussian statistic has a continuous density uniformly bounded on the local window, so the probability of lying near one of the finitely many thresholds tends to zero with the threshold neighborhood. Therefore the two implementations have uniformly vanishing risk difference.

This argument is for a fixed finite rule. Refining the rule with n so that the number of thresholds diverges requires additional control. No finite-n risk certificate for the SGD experiment follows merely from the Gaussian numerical certificate.



## 6. Numerical evidence

The [Gaussian certificate and pilot analysis](../experiments/results/gaussian_decision/analysis.md) contains the numerical tables and evidence limits. The independent [fixed-rule confirmation](../experiments/results/precision_confirmation/analysis.md) uses fresh samples. Continue with [unknown local directions](unknown_direction.md) for N.1–N.4.

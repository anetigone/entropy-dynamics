### 7.4. Constructing a certified minimax bracket

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

### 7.5. Using the fixed rule without the local parameter

For an arbitrary measurable Gaussian rule, the direct risk-transfer construction applies it to c T_n/sqrt(rho). This uses the known baseline CDF F_M, not the unknown environment parameter.

For each of the fixed finite step rules produced here, a simpler implementation is also asymptotically justified: apply it to Xhat=c sqrt(n) hat_theta, the raw empirical moment coordinate. Under the baseline, Xhat-c T_n/sqrt(rho) tends to zero in probability. The likelihood-ratio moment bounds used in (G.5) transfer this to the compact local family uniformly. The Gaussian statistic has a continuous density uniformly bounded on the local window, so the probability of lying near one of the finitely many thresholds tends to zero with the threshold neighborhood. Therefore the two implementations have uniformly vanishing risk difference.

This argument is for a fixed finite rule. Refining the rule with n so that the number of thresholds diverges requires additional control. No finite-n risk certificate for the SGD experiment follows merely from the Gaussian numerical certificate.

### 7.6. Simultaneous reference and calibration uncertainty

At the fixed cap, let p_R be the probability of a right exit by the cap and p_NL the probability of not having exited left by the cap. The eventual right-exit probability satisfies

\[
p_R\le\pi_n\le p_{NL}.
\]

For each cell use a one-sided exact Clopper–Pearson lower bound on p_R and upper bound on p_NL, each with error alpha_ref/(2C). Their endpoints [l_i,u_i] jointly cover every cell's eventual pi_n with probability at least 1-alpha_ref. This remains valid if a cell has censored paths; lack of an observed censor is not treated as proof that its population censoring probability is zero.

Set phat_i=N_R/N and e_i=max(phat_i-l_i,u_i-phat_i). On the reference event, |phat_i-pi_i|<=e_i.

#### 2.2. Calibration loss uncertainty conditional on the reference sample

For each rule and cell, score calibration repetition j by L_j=|forecast_j-phat_i|. Conditional on the independent reference sample, these are iid in [0,1]. Paired baseline-minus-optimized loss differences are iid in [-1,1].

For m observations taking values in an interval of width B, with unbiased sample variance S_m^2, use the two-sided empirical Bernstein radius

\[
b_m(B,\delta)=\sqrt{\frac{2S_m^2\log(4/\delta)}m}
+\frac{7B\log(4/\delta)}{3(m-1)}.
\]

This follows by scaling and applying both tails of Maurer–Pontil (2009), Theorem 4. Allocate delta=alpha_cal/(7C) to each of 40 rule-risk estimates and 30 paired gains. Their conditional coverage is at least 1-alpha_cal by a union bound; integration over the reference sample preserves that bound. Independence between the different rules is not required.

For each rule risk, add e_i to its calibration radius. For a paired gain add 2e_i, by the reverse triangle inequality. For the deterministic constant rule, propagate [l_i,u_i] through |.5-p| exactly instead of using an unnecessary calibration bound.

The resulting risk and gain intervals have **simultaneous coverage at least 95%** over all reported cells and rules, under the declared iid simulation experiment. They include both uncertainty layers. They are not just conditional standard errors or pointwise normal intervals. Evaluation of beta quantiles uses SciPy floating-point routines; these are statistical confidence statements, distinct from the previous directed-arithmetic Gaussian decision certificates.

#### 2.3. Finite-grid maximum risks

For each n let R_rule^grid be the maximum of the five true expected calibration losses at the fixed t cells. If cellwise intervals are [L_i,U_i], then

\[
\max_iL_i\le R_{\rm rule}^{\rm grid}\le\max_iU_i.
\]

For the improvement in grid maximum risk, subtract intervals: its lower bound is max_i L_baseline,i minus max_i U_optimized,i. The same simultaneous event covers all these derived comparisons. This is a five-cell statement, not a continuous-window maximum or finite-n minimax theorem.

Reference: Andreas Maurer and Massimiliano Pontil, *Empirical Bernstein Bounds and Sample Variance Penalization*, COLT 2009, Theorem 4. https://www.cs.mcgill.ca/~colt2009/papers/012.pdf

### 7.7. Cellwise confirmation and interpretation

| n | t | Reference right probability | Simultaneous reference interval | Moment MAE | Clipped MAE | Optimized MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | -1 | 0.16321 | [0.1545, 0.1721] | 0.18038 | 0.12645 | 0.15757 |
| 128 | -0.5 | 0.30817 | [0.2973, 0.3192] | 0.23400 | 0.20093 | 0.16552 |
| 128 | 0 | 0.49817 | [0.4863, 0.5101] | 0.24813 | 0.22381 | 0.16306 |
| 128 | 0.5 | 0.69037 | [0.6793, 0.7013] | 0.23015 | 0.19765 | 0.16299 |
| 128 | 1 | 0.84259 | [0.8338, 0.8511] | 0.18340 | 0.13257 | 0.16626 |
| 512 | -1 | 0.16205 | [0.1534, 0.1710] | 0.18425 | 0.13184 | 0.16259 |
| 512 | -0.5 | 0.30695 | [0.2960, 0.3180] | 0.22602 | 0.19352 | 0.16187 |
| 512 | 0 | 0.50385 | [0.4920, 0.5157] | 0.24760 | 0.22282 | 0.16288 |
| 512 | 0.5 | 0.68506 | [0.6739, 0.6960] | 0.23076 | 0.19884 | 0.16429 |
| 512 | 1 | 0.84711 | [0.8384, 0.8556] | 0.18185 | 0.13448 | 0.16837 |

The interval construction supports positive paired improvements over the raw moment rule in 6/10 cells and over the clipped rule in 2/10 cells. At endpoint cells clipping is often better, so there is no claim of uniform pointwise improvement. At t=0 the target equals .5 exactly by symmetry in this family; for consistency of the primary analysis, that cell was scored with the independent reference estimate, without using the exact value to narrow its interval.

The largest observed exit step was 720. Raw exit arrays, forecast arrays, fixed-design metadata and the frozen rule are retained. The checksum of the rule and the aggregate counts and MAEs were independently checked against the retained NPZ files.

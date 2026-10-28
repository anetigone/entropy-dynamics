# Numerical evidence

Experimental part of [the manuscript](DRAFT.md). Section 10 numbering is retained.

## 10. Numerical evidence

The experiments comprise Gaussian model checks, a mixed-shift scan, and controlled threshold adaptation on frozen visual features. Stochastic trajectories use the actual nonlinear entropy update with independent batches; Gaussian autoregressions are separately identified reference limits. Labels enter generators, risk evaluation, and the explicitly identified offline risk calibrations, not the entropy update. The target-geometry diagnostics and independent field-calibration study score existing visual trajectories. All results precede this manuscript revision; no new simulation is implied by the presentation below. Reported CDF errors are maximum absolute discrepancies over the observed integer-time budget; joint errors compare the left/right exit subdistributions.

### 10.1. Population rebound and safe convergence

For \(m=\sigma=1,w=2,\eta=.1,\epsilon=.005\), with \(p=.35\), Gaussian quadrature gives:

| Translation \(\delta\) | Frozen target risk | Minimum risk | Minimum-risk step | \(T_{.005}\) |
|---:|---:|---:|---:|---:|
| 0 | .158655254 | .147597606 | 47 | 94 |
| -.1 | .152603983 | .147597604 | 30 | 67 |
| .1 | .167122203 | .147597778 | 67 | 124 |

Entropy decreases throughout 2500 updates. The first two rows satisfy the explicit sufficient rebound region. The last row is an additional numerical example, not a claim that the same simple sufficient inequalities cover it.

An initial exploratory pilot considered 30 cells:
\(\delta\in\{-.6,-.3,0,.3,.6,1\}\) and
\(p\in\{.2,.35,.5,.65,.8\}\), using 160-point Gauss–Hermite quadrature. Eight cells improved risk by more than .001 and later exceeded frozen risk by .005 within 2500 steps. The illustrated zero-translation case was selected from this grid. This is exploratory evidence, not a preregistered estimate of how frequently rebound occurs.

For safe convergence, \(m=2,\sigma=1,w=5\) satisfies the curvature condition: its right-hand side is approximately 17.0163, below \((w\sigma)^2=25\). The explicit radius is about .0406252. Separate numerical trajectories with \(\delta=.2,\eta=.02\), beyond that conservative radius, reduce risk from .0249169 to .0227501 at \(p=.5\), and from .0251372 to .0227471 at \(p=.49\), over 6000 steps. These larger-initialization examples do not enlarge the proved basin.

### 10.2. Population tolerance contrast

For \(m=\sigma=1,w=2,\eta=.1,x_0=0\),
\[
\beta=.1835111816,\qquad\lambda=.0825771460,\qquad
1/\log A=121.598196.
\]

| \(\rho\) | \(T_{.005}\) | \(T_0\) |
|---:|---:|---:|
| .1 | 112 | 79 |
| .01 | 288 | 79 |
| .001 | 552 | 79 |
| .0001 | 830 | 79 |
| .00001 | 1110 | 79 |
| .000001 | 1390 | 79 |

For small \(\rho\), \(T_{.005}-121.598196\log(1/\rho)\) stays near \(-290\), with integer oscillations. The zero-tolerance limiting index before rounding is about 78.046262, predicting step 79. Near zero tolerance, risk differences are evaluated by a scaled integral rather than subtraction of nearly equal CDF values. A plotted asymptote uses the theoretical slope and a fitted intercept from the smallest four imbalance values; it is a slope check, not an independently predicted absolute time.

### 10.3. A nonvacuous positive-drift bound

For the \(p=.35,\delta=0\) example, the safe interval is approximately
\[
[a,c]=[-.062394603,\ .713054205].
\]
Conservative constants are
\[
d\ge.062,\quad .713\le D\le.714,\quad
\nu=.049,\quad V=.098,\quad M=.5.
\]
The drift calculation uses standard-normal truncation to \([-8,8]\), 8192 midpoint subintervals, and 801 threshold grid points over \([-.063,.714]\). Its discretization, tail, and interpolation bounds give the enclosure
\[
[.0492117163,\ .0977563734].
\]
For clarity, if \(Q\) is sample velocity times standard-normal density,
\[
\|Q''\|_\infty\le
(3w^3/4)\sigma^2\varphi(0)+
2L\sigma/\sqrt{2\pi e}+M\varphi(0).
\]
The midpoint error is at most \(16h^2\|Q''\|_\infty/24\), \(h=16/8192\); the tail error is at most \(2M\Phi(-8)\), and threshold interpolation contributes at most \(L\Delta x/2\). An additional \(10^{-8}\) floating-point guard is included. This is not formal interval certification of all rounding errors.

With \(n=64,t_-=50,t_+=210\), Theorem 7 gives failure terms approximately
\[
1.7569\times10^{-7},\quad2.9591\times10^{-6},\quad.00236258.
\]
Conditional on the stated drift enclosure, the coverage lower bound for
\(50<T_{.005}\le210\) with right exit is .99763428.

A separate 2000-path simulation with seed 20261024 gives minimum 80, maximum 112, median 94, and 2.5%/97.5% quantiles 85/104, with no observed left exits. This checks typical behavior; absence of rare failures in a simulation does not prove the probability bound.

### 10.4. Critical SGD: finite geometry and direction

Use \(m=\sigma=1,w=2,\eta=.1,\delta=0\), and set
\[
\rho=(\lambda s_\eta/\beta)\kappa/\sqrt n.
\]
Each fixed-tolerance cell uses 600 actual SGD paths, seed 20261026, and a 2500-step cap. All paths exit before the cap. The model gives
\[
s_0^2=.0935303760,\quad s_\eta=.2374852733,\quad
\lambda s_\eta/\beta=.1068646385.
\]
For \(\epsilon=.005\), \(c_\epsilon=.2039969278\) and
\(C_\epsilon=.2038878864\).

The balanced-limit prediction uses Corollary 10. The finite-geometry prediction uses the true \(q_\rho,A_\rho,\mu_n,s_n,B_n^\pm\) in Theorem 9. Neither prediction fits a time offset or empirical quantile.

| \(n\) | \(\kappa\) | Observed median | Balanced median | Finite-geometry median | Balanced CDF error | Finite-geometry CDF error |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0 | 233 | 241 | 241 | .0450 | .0450 |
| 32 | 1 | 209 | 187 | 210 | .1482 | .0254 |
| 32 | 2 | 162 | 108 | 167 | .4026 | .0445 |
| 128 | 0 | 322.5 | 325 | 325 | .0295 | .0295 |
| 128 | 1 | 298.5 | 271 | 284 | .1015 | .0595 |
| 128 | 2 | 224 | 193 | 225 | .2478 | .0155 |
| 512 | 0 | 412 | 409 | 409 | .0216 | .0216 |
| 512 | 1 | 361.5 | 355 | 362 | .0600 | .0238 |
| 512 | 2 | 293 | 277 | 294 | .1235 | .0212 |

CDF errors are empirical maximum discrepancies on an integer grid and include Monte Carlo noise. A single-cell 95% DKW band for 600 paths is approximately .0554; it is neither simultaneous across all cells nor a theorem-approximation guarantee. The corrected \(n=128,\kappa=1\) error remains above that single-cell value.

| \(\kappa\) | Predicted right-exit probability | Observed, \(n=32\) | \(n=128\) | \(n=512\) |
|---:|---:|---:|---:|---:|
| 0 | .5000 | .5150 | .4867 | .5117 |
| 1 | .8413 | .8533 | .8383 | .8483 |
| 2 | .9772 | .9750 | .9817 | .9750 |

Accurate direction prediction does not ensure accurate time prediction. Both exit directions are unsafe relative to the chosen comparator.

### 10.5. Microscopic tolerance and checks with limited benefit

With \(\rho=.2/\sqrt n\) and \(\epsilon=.005/n\), 2000 actual paths per batch size are compared with 150000 separately simulated AR reference paths.

| \(n\) | Actual median | AR reference median | Empirical CDF difference |
|---:|---:|---:|---:|
| 32 | 45 | 45 | .0127 |
| 128 | 47 | 45 | .0243 |
| 512 | 45 | 45 | .0228 |

The actual risk endpoints are solved from the original risk, not substituted from its quadratic limit. Single-sample-set 95% DKW bands are about .0304 for the actual paths and .00351 for the AR reference. The nearly unchanged medians contrast with the logarithmic fixed-tolerance shift.

For a wider interval, \(\epsilon=.15,n=128,\kappa=1\), 1000 actual paths have median 499. Replacing the balanced linear boundary with the conjugacy boundary changes the CDF discrepancy only from .02903 to .02893; both predicted medians are 493. This does not support a claim that the conjugacy boundary alone produces a substantial numerical improvement.

At \(n=512\), 12000 independent first batches per setting give improvement probabilities .5497, .7196, and .8838 for \(c=.2,1,2\), compared with .5478, .7258, and .8849 from Proposition 12. At \(c=0\), improvement is exactly impossible. These probability comparisons are distinct from bounds on the magnitude of the first-step risk change.

### 10.6. Mixed-shift cancellation and prediction ablations

We evaluate 45 cells: \(n\in\{32,128,512\}\), \(\kappa\in\{0,1,2\}\), and five translation offsets around cancellation, with 1000 fresh-batch paths per cell. We use \(m=\sigma=1,w=2,\eta=.1,\epsilon=.005\) and a 2500-step cap. All 45000 paths exit within budget. Translation offsets are expressed in seed-standard-deviation units; a positive offset increases translation and moves the centered initialization leftward.

For \(\kappa=2,n=128\), the direction reverses and the median exit time increases near cancellation:

| Translation offset | Observed right-exit fraction | Conjugacy prediction | Observed median | Predicted median |
|---:|---:|---:|---:|---:|
| -2 | .971 | .977 | 227 | 225 |
| -1 | .845 | .841 | 291 | 296 |
| 0 | .472 | .499 | 325 | 327 |
| 1 | .169 | .158 | 244 | 244 |
| 2 | .025 | .023 | 169 | 168 |

At cancellation, observed medians for \(n=32,128,512\) are 241/325/414, versus predictions 250/327/410. Unequal risk endpoints permit asymmetric time curves. Across all 45 cells, the mean maximum absolute total-exit CDF errors are .057445 for the balanced limit, .033616 for finite-parameter local linearization, and .033852 for full conjugacy. Removing the actual risk-endpoint correction from local linearization raises the error to .057995; removing the peak, curvature, or noise correction individually leaves it near .034. Thus the main numerical gain in this scan comes from risk endpoints. The scan supports cancellation and finite-risk geometry, but does not show an overall numerical advantage of conjugacy over local linearization. The single-cell 95% Monte Carlo DKW width is .042947; 35/45 cells fall within it for each finite-parameter predictor, without implying simultaneous coverage.

### 10.7. Frozen visual features: protocol and independent-prediction failure

We use a frozen ImageNet-1k-pretrained ResNet-18 in evaluation mode, extracting 512-dimensional features from the official CIFAR-10 train/test splits. The extraction protocol uses its paired ConvNextImageProcessor. Four prespecified binary pairs are cat/dog, automobile/truck, deer/horse, and airplane/ship, with the first class positive. For each pair, the projection direction is the normalized difference of training-class feature means; the center is their midpoint. Projected scores are divided by the square root of the average training-class score variance. Official training features determine this source projection and labeled offline calibration under controlled shifts; official test features supply the fixed target empirical distributions and risk evaluation. No additional dataset split is introduced. This is a controlled threshold-adaptation experiment, not full normalization adaptation or strict weights-only deployment.

The grid contains four pairs, three priors \(p\in\{.5,.4,.3\}\), three batches \(n\in\{32,128,512\}\), and five translation offsets, giving 180 cells. Each uses 600 paths, \(\eta=.1,w=2,\epsilon=.02\), and 2000 updates. Only threshold-mode results are available; joint-scale and logit-adjusted runs have not yet supplied results. Updates draw independently with replacement from the target empirical mixture and do not read labels. Labels define the controlled mixture and evaluate risk; training labels also supply the original risk calibration. No exit curves are used to fit parameters.

All 108000 paths run without recorded numerical failure; 13102 (12.13%) remain unexited at the cap, across 37 cells. Unexited paths remain in CDF denominators. They are not permanently safe, and exit-only conditional medians are not substituted for censored medians.

Source calibration yields a two-sided prediction in 141/180 cells; 39 have a risk-safe component unbounded on at least one side. Across the 141 applicable cells, mean maximum absolute total-exit CDF errors are .5411 for local linearization and .4479 for conjugacy. Only 3/141 conjugacy errors are within the single-cell Monte Carlo DKW width .05544. Conjugacy errors averaged by batch are .3193/.4564/.5732, with slightly different applicable subsets. The fully matched cat/dog subset also worsens with batch. These results do not establish accurate independent prediction.

Direction is a particularly informative failure. At the source-calibrated cancellation point, cat/dog with \(p=.5\) has observed right-exit fractions .2933/.1383/.0117 for \(n=32/128/512\), against a prediction of .5 throughout. The training-calibrated peak is .509827 and the target empirical peak is .527601. At \(n=512\), the displacement is -2.246 target seed standard deviations; using target geometry gives a conjugacy right-exit probability .01186, close to the observed .01167. Small peak migration can dominate the shrinking mini-batch noise scale. Correcting direction alone, however, does not ensure a correct time distribution.

### 10.8. Target empirical geometry: diagnostic recovery and residuals

We recompute the entropy field, peak, curvature, noise, and risk boundaries directly from the target feature pool and score the existing trajectories. This is a post hoc conditional oracle diagnostic: it fits no exit times, but uses the same target pool and target labels for risk boundaries. It is not independent calibration. The original CDFs recomputed from saved paths agree with their summaries to a maximum difference of \(2.22\times10^{-16}\).

On the same 133 cells where all compared predictors apply:

| Predictor | Mean total-exit CDF error | Mean left/right joint CDF error |
|---|---:|---:|
| Training-calibrated local linearization | .5255 | .5094 |
| Training-calibrated conjugacy | .4315 | .4940 |
| Target peak only, local linearization | .5517 | .5485 |
| Target field conjugacy, training risk endpoints | .1899 | .1858 |
| Full target geometry, local linearization | .4804 | .4821 |
| Full target geometry, conjugacy | .0590 | .0578 |

Unlike the Gaussian mixed-shift scan, these visual marginals show a substantial numerical role for the full nonlinear geometry. Peak correction explains important directional biases but is insufficient for time prediction. Replacing the field while retaining training endpoints leaves appreciable error; replacing the endpoints as well reduces it further. Accurate entropy geometry and accurate risk geometry are separate requirements.

Target conjugacy applies in 148/180 cells: 21 have a one-sided unbounded safe component and 11 have the target peak outside the initial safe component. Among applicable cells, 102/148 errors lie within the single-cell Monte Carlo band; the largest error remains about .3600. Automobile/truck exhibits a long-time tail discrepancy even with oracle geometry. For \(p=.5,n=32,\mathrm{offset}=-2\), observed 10/50/90% quantiles are 1449/1729/not observed by the cap, versus predictions 1681/1739/1822. A close median does not establish tail accuracy.

Direct inverse-iteration checks at 12 representative coordinates give a maximum relative discrepancy of approximately \(1.26\times10^{-4}\) from interpolation; tightening the local cutoff from \(10^{-5}\) to \(10^{-6}\) changes them by at most approximately \(1.01\times10^{-5}\). These checks support those representative numerical predictions, without certifying the whole domain or every large conjugacy boundary.

Each pair contains 1000 official test images per class. The classwise empirical risk can be computed exactly, but a conservative uniform bound for its difference from frozen risk is about .09362, exceeding \(\epsilon=.02\). This does not invalidate finite-pool exits or assert that actual population error is that large; it prevents interpreting them as precise population-risk crossing predictions. Some empirical safe sets also have multiple components, so departure from the initial component and the first discrete iterate with excessive risk need not be identical. These are additional limits on applying the two-sided approximation.

### 10.9. Independent unlabeled field calibration

We next estimate the entropy field from a new unlabeled calibration stream sampled independently with replacement from the same fixed target feature pool. The estimator receives scores and multiplicities, not labels, the target peak, or exit outcomes. Independence is conditional on the fixed pool. We reuse the original SGD paths and keep their original initialization; no SGD is resimulated and no new official split is created. Target labels are used for the controlled sampler, oracle endpoints, and evaluation, not for field fitting.

The pilot covers all four pairs, \(p=.5\), original offset zero, \(n=32/128/512\), and four calibration repetitions per pair and budget. Budgets 2048/16384/131072 use nested prefixes; the 1048576-observation check is a separate run. The latter is not a paired continuation of the smaller-budget prefixes. For \(n=128\), errors below average each applicable calibration repetition separately:

| Calibration observations \(M\) | Applicable repetitions / 16 | Training risk endpoints | Target oracle risk endpoints |
|---:|---:|---:|---:|
| 2048 | 13 | .6818 | .6644 |
| 16384 | 16 | .3566 | .2972 |
| 131072 | 16 | .2290 | .1585 |
| 1048576 | 16 | .2015 | .0910 |

All smallest-budget fields have a unique unstable peak, but three estimated peaks fall outside the relevant safe interval; their inapplicability is retained. The successful-only mean is not an unconditional success claim. On these initializations, original training-calibrated conjugacy has mean error .3599 and exact target geometry has .0713. Small calibration budgets can therefore worsen prediction. Larger budgets approach the conditional oracle when oracle endpoints are supplied, but retaining training endpoints leaves a substantial residual. At the largest budget, automobile/truck still has oracle-endpoint error .1887.

Calibration budgets count repeated stream observations from a pool of 2000 images per pair, not distinct new images. Four repetitions do not establish confidence coverage, and repeatedly scoring the same SGD paths adds no independent adaptation trajectories. Calibration cost is separate from the reported SGD update count; an online policy including this cost is not evaluated. These results support finite independent field estimation on a fixed empirical distribution, not general population prediction or an unlabeled risk certificate.

The following first-order calculation clarifies the required calibration scale; the full real-SGD prediction-error theorem and random calibration limits are in [Theorems E.6–E.8](calibration.md). For a fixed target field with unstable root \(q\), let \(\lambda=v'(q)>0\), \(A=1+\eta\lambda\), and \(\sigma_g^2=\operatorname{Var}(U(q,Z))\). If a consistent local root of the sample field is selected and its derivative converges uniformly near \(q\), Taylor expansion of the root equation and the iid CLT give
\[
\sqrt M(\widehat q_M-q)\Rightarrow N(0,\sigma_g^2/\lambda^2).
\]
The frozen-noise SGD seed variance is \(s^2=\eta^2\sigma_g^2/(A^2-1)\). Consequently the squared ratio of peak-calibration uncertainty to the threshold noise scale \(s/\sqrt n\) is
\[
\mathcal R_{M,n}^2
=\frac{n\sigma_g^2}{M\lambda^2s^2}
=\frac nM\frac{2+\eta\lambda}{\eta\lambda}.
\]
Negligible peak uncertainty therefore requires a budget large relative to \(n(2+\eta\lambda)/(\eta\lambda)\) in this first-order comparison. At \(n=128\), the four target empirical fields give equal-scale budgets of approximately 17600/48600/38300/20200 observations. This is not a finite-sample exit-CDF guarantee: it omits field-shape, curvature, noise, and risk-boundary errors. Because calibration changes only the predictor in this experiment, its variance cannot simply be added to the actual SGD seed variance. Random initialization centered at the estimated peak would define a different process requiring a separate real-SGD argument.

### 10.10. Numerical scope

The geometry and critical experiments record Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0. Independent adaptive-integration and finite-difference diagnostics support quadrature and gradient consistency. Conjugacy endpoint values change by at most about \(3.71\times10^{-12}\) when the inverse-iteration cutoff is tightened from \(10^{-3}\) to \(10^{-4}\); checked functional-equation residuals are at most about \(5.55\times10^{-17}\). These are floating-point diagnostics, not certified probability constants.

The Gaussian experiments establish controlled-model checks. The frozen visual experiments add empirical evidence for nonlinear geometry and calibration sensitivity on non-Gaussian score distributions; they do not extend the Gaussian theorems to these marginals. They do not establish full visual TTA benchmark performance, unknown-anchor robustness, or a practical unlabeled stopping procedure. The separately integrated [calibration theory](calibration.md) now gives such a theorem under explicit conditions; these experiments do not verify all its assumptions or its finite-sample constants.


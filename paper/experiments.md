# Numerical evidence

Experimental part of [the manuscript](DRAFT.md). Section 10 numbering is retained.

## 10. Numerical evidence

The experiments comprise Gaussian model checks, a mixed-shift scan, and controlled threshold adaptation on frozen visual features. Stochastic trajectories use the actual nonlinear entropy update with independent batches; Gaussian autoregressions are separately identified reference limits. Labels enter generators, risk evaluation, and the explicitly identified offline risk calibrations, not the entropy update. The target-geometry diagnostics and independent field-calibration study score existing visual trajectories. The legacy studies below are retained; Section 10.15 adds the unknown-local-environment visual study and its new independent confirmation. Reported CDF errors are maximum absolute discrepancies over the observed integer-time budget; joint errors compare the left/right exit subdistributions.

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

The subsequent expanded study covers 144 cells: four pairs, three priors, three batches, and offsets -2/-1/0/1 (not +2), with 20 calibration repetitions per pair/prior and budget. Repetitions are shared across batch and offset. On 1822 records (106 cells) applicable for both endpoints and all budgets, source-endpoint CDF errors are .5902/.3686/.2354/.1935 and oracle-endpoint errors .5709/.3349/.1668/.0899. At the largest budget, predictions succeed in 2136/2880 source-endpoint and 2364/2880 oracle-endpoint attempts. The expanded evidence strengthens the budget trend without eliminating inapplicability, endpoint migration, or finite-pool limitations. Source offset zero is not exact target centering and does not validate the arctan law. Full analysis remains in [the calibration result directory](../experiments/results/calibration_grid/analysis.md).

### 10.10. Numerical scope

The geometry and critical experiments record Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0. Independent adaptive-integration and finite-difference diagnostics support quadrature and gradient consistency. Conjugacy endpoint values change by at most about \(3.71\times10^{-12}\) when the inverse-iteration cutoff is tightened from \(10^{-3}\) to \(10^{-4}\); checked functional-equation residuals are at most about \(5.55\times10^{-17}\). These are floating-point diagnostics, not certified probability constants.

The Gaussian experiments establish controlled-model checks. The frozen visual experiments add empirical evidence for nonlinear geometry and calibration sensitivity on non-Gaussian score distributions; they do not extend the Gaussian theorems to these marginals. They do not establish full visual TTA benchmark performance, unknown-anchor robustness, or a practical unlabeled stopping procedure. The separately integrated [calibration theory](calibration.md) now gives such a theorem under explicit conditions; these experiments do not verify all its assumptions or its finite-sample constants.


### 10.11. Noise ablations explain a finite-parameter tail failure

We further test automobile/truck at p=.5, translation offset -2, and n=32/128/512. Each setting uses 600 paths, a 4000-update cap, and seed 20261030. Four coupled processes share sampled batches: actual entropy SGD; the same population drift with centered batch noise evaluated at the root throughout; state-dependent noise retained only within distance r of the root; and noise retained only outside that region. The last three are diagnostic recursions, not the original algorithm or proposed adaptation methods. We use r=.5 and .25. Actual-SGD and full root-noise exit arrays agree exactly across radii, so these are not additional independent replicates. All paths exit within budget.

| Batch | Actual 10–90% width | Full root-noise width | Near-only width, r=.5 | Far-only width, r=.5 |
|---:|---:|---:|---:|---:|
| 32 | 590 | 590 | 245 | 535 |
| 128 | 333 | 310 | 171 | 266 |
| 512 | 170 | 172 | 116 | 131 |

At r=.25, near-only widths are 185/148/102 and far-only widths are 553/277/134. Thus removing late noise substantially narrows the time distribution under both partitions; retaining far noise recovers much, though not all, of its width. Full root-noise empirical CDFs differ from actual SGD by .0667/.0500/.0483, even though their noise law does not depend on the current state. Freezing noise at the root while continually injecting it is different from representing the entire passage by an early Gaussian seed.

For this new actual-SGD sample, the original target-geometry conjugacy predictor has total-CDF errors .3740/.2903/.2086. A separate, unfitted first-order population-orbit diagnostic propagates variance by \(V_{t+1}=G^{\prime}(\bar x_t)^2V_t+\eta^2\operatorname{Var}(U(\bar x_t,Z))/n\), starting at zero. Dividing terminal standard deviation by the magnitude of the deterministic crossing velocity gives time standard deviations approximately 230/124/67 updates; a normal approximation centered at deterministic exit gives CDF errors .0527/.0388/.0569. This is a single-branch diagnostic, not a proved joint exit law. It does not fit a center or variance to observed exits. In the corresponding linearized variance decomposition, noise beyond distance .5 contributes approximately 82%/72%/62%.

The evidence identifies continuing noise during slow far-field passage as an important finite-parameter source of dispersion. Strong state dependence of the noise is not required to reproduce the broad distributions. It does not establish that all residuals have this cause, nor invalidate a critical asymptotic theorem: the fixed-source initialization scan is not automatically a uniformly target-critical sequence, and the constants can depend strongly on the whole exit domain. Modified-noise processes use an 8193-point drift interpolation with maximum midpoint discrepancy about 2.04e-8; actual SGD uses exact sampled gradients. This numerical check is not a full grid-convergence certification. The empirical-CDF comparisons use coupled paths and are descriptive, not independent two-sample significance tests. Further theory should connect the early random seed to accumulated passage noise before extending the diagnostic to mixed exit directions.


### 10.12. Target-centered calibration scale

We fix four visual empirical score laws at prior .5 and initialize actual entropy SGD at the exact target peak q. For n=32/128/512 and eta=.1, calibration budgets are chosen from the exact target curvature to keep R=.5/1/2, where R²=(n/M)(2+eta lambda)/(eta lambda). A second sequence sets M proportional to n², reducing R from1 to .5 to .25. The design uses geometry, not observed exits. Each unique budget has128 calibration repetitions; prefixes are nested and reused when settings have the same budget. The estimator receives only unlabeled scores and multiplicities.

This experiment evaluates the local empirical-root predictor Phi(sqrt(n)(q-qhat)/shat), rather than the full conjugacy time forecast. Across the four pairs, fixed R=1 gives centered MAE approximately .237–.269 against .5 despite decreasing absolute root error, close to arctan(1)/pi=.25. On the decreasing-R sequence, MAE ranges are:

| n | R | Centered MAE across pairs | arctan(R)/pi |
|---:|---:|---:|---:|
| 32 | 1 | .237–.257 | .2500 |
| 128 | .5 | .139–.143 | .1476 |
| 512 | .25 | .080–.082 | .0780 |

These are successful-root means; the lowest individual-setting success rate is .9141. Failed estimates and failure-inclusive MAE bounds are reported in the result analysis. Repetitions sharing prefixes are not independent additional observations. The entire experiment is conditional on fixed pools, not increasing numbers of independent images.

Independent actual SGD uses256 paths per pair and batch. Three pairs have at most one censored path by1500 steps. Automobile/truck has129–132 censored paths; replaying the same paths to4000 steps preserves all earlier observed exits and resolves every censor on the right. Final right-exit counts for n=32/128/512 are132/132/129 out of256. This extension is a continuation, not an independent replication; only its summary was retained. Its cause is the asymmetric risk geometry: peak-to-left distance about .100 and peak-to-right distance1.884, with representative deterministic right passages taking2071–2342 steps. It is consistent with the slow-passage diagnostics in §10.11.

The centered errors test a specific forecast limit. If the predictor knows exact centering, a constant .5 forecast bypasses this error. Unknown local displacement and a lower bound over a local family are separate questions defined in [§8B](local_prediction.md). Detailed protocol, per-setting estimates, uncertainty and censoring remain in the [result analysis](../experiments/results/centered_calibration/analysis.md).

### 10.13. Safety-horizon stopping as an application stress test

Two independent random-seed runs use four pairs, prior .5, n=128 and offsets -2/0/2:12 cases,128 paths each and1500 steps per run. All cases remain in the denominator; unavailable forecasts freeze at initialization. The stopping horizon is the last prefix whose approximate exit CDF is no greater than alpha, without fitting a safety margin to test exits. A disagreement gate checks candidates against the exact unlabeled empirical marginal before deployment. It is not a finite-sample marginal estimator or a reproduction of a published risk monitor.

| Rule | Run1 path violation / gain(pp) | Run2 path violation / gain(pp) |
|---|---:|---:|
| Exact-geometry forecast, alpha=.01 | .0130 / .651 | .0124 / .701 |
| Fixed100 steps | .0020 / .652 | .0026 / .639 |
| Exact-pool disagreement gate | 0 / .572 | 0 / .569 |

These runs do not show a stable utility advantage from treating a safety upper horizon as the desired stopping point. They do not disprove risk-exit forecasting or all stopping methods. Target-endpoint forecasts use labels; source endpoints are an uncontrolled transfer ablation. Nominal alpha is not a proved safety bound. The runs share the same feature pools and therefore do not establish cross-environment generalization. Full results are in the [first analysis](../experiments/results/stopping_pilot/analysis.md) and [seed repetition](../experiments/results/stopping_repeat/analysis.md); all evidence is indexed by the [results README](../experiments/results/README.md).

### 10.14. Fixed-rule finite-batch confirmation

The [original confirmation report](../experiments/results/precision_confirmation/analysis.md) gives the fixed design, two-layer simultaneous confidence construction, all cellwise and five-cell maximum-risk comparisons, and evidence boundaries. It excludes pilot observations and retains the previously selected Gaussian rule. The unknown-direction extension is theoretical; no nuisance simulation is claimed. The report and its raw outputs remain in their original directory.


### 10.15. Unknown local visual environments and independent confirmation

**Setup and information.** We next test prospective probability prediction on real visual scores, using the frozen ImageNet-pretrained ResNet-18 features and four prespecified CIFAR-10 class pairs. Official training features determine a fixed projection. For each pair, the official test empirical baseline determines one root \(q_0\) and its constants; all environments retain \(x_0=q_0\). With centered, unit-variance baseline entropy velocity \(h_0(z)\), we draw images with weights proportional to \(1+\theta h_0(z)\), where \(\theta=t/(c\sqrt n)\) and \(c=\sigma_0/(\lambda_0s_0)\). The generator knows \(t\); predictors receive only independent unlabeled calibration scores and the declared baseline constants. They do not receive target roots, labels, generating coordinates or reference exits. Unlike the theoretical construction, reweighting labeled images changes the complete classification-risk function: risk is evaluated under each environment's actual weighted label law.

**Exploration and independent confirmation.** The exploratory study covered four pairs, \(n\in\{32,128,512\}\) and \(t\in\{-1,-.5,0,.5,1\}\), with 1,024 reference paths per environment and 256 calibration repetitions per budget. Fixed relative calibration noise \(r=c\sqrt{n/M}\in\{.5,1,2\}\) gave an error plateau across batch sizes; the sequence \(r=2,1,.5\) gave decreasing error. These are finite-grid trends conditional on fixed image pools, not new independent images or a consistency theorem for a finite collection of frozen rules. All 36 fixed-ratio settings favored the optimized rule in maximum-MAE point estimates, but their simultaneous improvement intervals crossed zero. We therefore fixed a new confirmation design before generating its samples, rather than pooling exploration into confirmation.

**Fixed design and scoring.** Confirmation retains all four pairs, fixes \(n=128\), \(K=r=1\), and uses the same five-point \(t\) grid. Each of the 20 environments has 16,384 fresh reference paths and 4,096 independent calibration repetitions, totaling 327,680 paths and 81,920 repetitions. We use \(\eta=.1\), slope \(w=2\), tolerance \(\epsilon=.02\), baseline prior \(.5\), \(M=\operatorname{round}(c^2n)\), and a fixed 4,000-update cap for every pair. Five predictors share each calibration repetition: constant \(.5\), empirical-root plug-in, raw influence-function moment, window-clipped moment, and the previously frozen Gaussian step rule. Root failures would fall back to \(.5\) and remain in the score. The primary comparison is clipped minus optimized five-point maximum expected MAE, separately for each pair.

**Uncertainty and results.** We allocate error \(.0125\) to each pair, giving at least 95% simultaneous coverage across all four pairs, five rules and recorded paired comparisons by a union bound. Censor-compatible exact one-sided binomial reference bounds and conditional empirical-Bernstein calibration bounds account for both uncertainty layers. Maximum-risk differences are obtained by subtracting maximum-risk intervals, not by maximizing cellwise paired gains. The optimized rule has a lower grid maximum MAE in every pair. Its improvement over clipping has a positive interval in three pairs; automobile/truck remains unresolved, with interval \([-.0030,.1032]\). Comparisons with constant, empirical-root and raw-moment baselines have positive improvement intervals in all four pairs. These are finite-grid comparisons, with no claim of pointwise dominance.

**Dynamics diagnostics and scope.** There are three censored reference paths and no empirical-root failures. The maximum point-estimate discrepancy between actual right-risk-exit probability and \(\Phi(t)\) is \(.01046\). Initial-safe-component exit and actual risk exit have different times on 18,378 paths, but no direction disagreement among paths with observed risk exits; censored paths remain in the bounds. The optimized rule trades endpoint accuracy for a smaller central peak. This confirms a reproducible prediction improvement on the declared finite visual grid. It does not validate shared risk on images, unknown-baseline robustness, corruption transfer, continuous-window finite-\(n\) minimax optimality or exit-time accuracy.

| Pair | M | Moment | Clipped | Optimized | Clipped minus optimized | Simultaneous interval |
|---|---:|---:|---:|---:|---:|---|
| Cat/dog | 17625 | 0.2530 | 0.2266 | 0.1682 | 0.0585 | [0.0052, 0.1117] |
| Automobile/truck | 48557 | 0.2552 | 0.2288 | 0.1787 | 0.0501 | [-0.0030, 0.1032] |
| Deer/horse | 38254 | 0.2536 | 0.2276 | 0.1676 | 0.0600 | [0.0068, 0.1129] |
| Airplane/ship | 20213 | 0.2502 | 0.2246 | 0.1680 | 0.0566 | [0.0036, 0.1095] |

The four pair-specific random seeds are 20261030, 20261031, 20261032 and 20261033, respectively; the reference and calibration streams are independent within each pair. Sampling is with replacement conditional on the official test pool; it does not increase the number of distinct images. The reference and calibration error allocations are each \(.025\) across \(C=20\) environments. Each one-sided reference tail receives \(.05/(4C)\); each of five risk estimates and four paired gains per environment receives \(\delta=.05/(18C)\). These are the same two-layer formulas as the earlier confirmation, with nine rather than seven calibration quantities per environment. No exploration or timing samples enter these intervals.


| Pair | t | Right exits | Censored | Root MAE | Moment MAE | Clipped MAE | Optimized MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cat/dog | -1 | 2662 | 0 | 0.18434 | 0.18434 | 0.13394 | 0.16584 |
| Cat/dog | -0.5 | 5145 | 0 | 0.22452 | 0.22446 | 0.19203 | 0.15725 |
| Cat/dog | 0 | 8280 | 0 | 0.25302 | 0.25300 | 0.22662 | 0.16644 |
| Cat/dog | 0.5 | 11274 | 0 | 0.23200 | 0.23199 | 0.20005 | 0.16436 |
| Cat/dog | 1 | 13818 | 0 | 0.18382 | 0.18382 | 0.13406 | 0.16817 |
| Automobile/truck | -1 | 2428 | 2 | 0.18711 | 0.18720 | 0.14584 | 0.17871 |
| Automobile/truck | -0.5 | 4974 | 0 | 0.23115 | 0.23121 | 0.19892 | 0.16575 |
| Automobile/truck | 0 | 8156 | 0 | 0.25529 | 0.25524 | 0.22881 | 0.16799 |
| Automobile/truck | 0.5 | 11161 | 1 | 0.22560 | 0.22558 | 0.19323 | 0.15735 |
| Automobile/truck | 1 | 13714 | 0 | 0.18358 | 0.18359 | 0.13297 | 0.16499 |
| Deer/horse | -1 | 2526 | 0 | 0.18295 | 0.18289 | 0.13437 | 0.16762 |
| Deer/horse | -0.5 | 5021 | 0 | 0.23019 | 0.23015 | 0.19880 | 0.16501 |
| Deer/horse | 0 | 8129 | 0 | 0.25361 | 0.25358 | 0.22760 | 0.16685 |
| Deer/horse | 0.5 | 11303 | 0 | 0.22802 | 0.22801 | 0.19637 | 0.16172 |
| Deer/horse | 1 | 13806 | 0 | 0.18272 | 0.18272 | 0.13153 | 0.16557 |
| Airplane/ship | -1 | 2477 | 0 | 0.17882 | 0.17886 | 0.13311 | 0.16803 |
| Airplane/ship | -0.5 | 5063 | 0 | 0.23066 | 0.23067 | 0.19842 | 0.16405 |
| Airplane/ship | 0 | 8138 | 0 | 0.25027 | 0.25025 | 0.22461 | 0.16451 |
| Airplane/ship | 0.5 | 11303 | 0 | 0.22784 | 0.22775 | 0.19617 | 0.16296 |
| Airplane/ship | 1 | 13751 | 0 | 0.18414 | 0.18406 | 0.13305 | 0.16607 |

All reference denominators are 16,384. Cellwise MAEs are point estimates; simultaneous intervals are in the original summaries. See [the complete visual confirmation analysis](../experiments/results/local_confirmation/analysis.md) and [the exploratory analysis](../experiments/results/local_scaling/analysis.md). The main text presents the risk figure and primary maximum-risk table.

## 10.16. Natural corruptions: independent direction-calibration confirmation

We retain the direction target, not the probability of any deterioration. On natural corruptions the target is the unconditional event P(T<infinity, x_T>x0); conditioning on finite exit requires a separate justification. Complete labeled risk geometry is used only for offline applicability and evaluation, not as an unlabeled deployment detector.

Using frozen ResNet-18 scores for four CIFAR-10 class pairs, we consider clean and six CIFAR-10-C corruption conditions, retaining the source projection and initialization. Before new sampling, complete empirical risk enumeration fixes 22 environments with an attainable unsafe state; six environments have no unsafe state and are retained separately. Geometric attainability does not prove almost-sure dynamical exit. Each selected environment has 1,024 fresh reference paths, a 4,000-update cap, and 256 independent calibration repetitions at M=2,048/16,384/131,072. Budgets share nested prefixes within repetitions. The fixed-pool sampling protocol counts replacement draws, not new independent images. This is a controlled binary-threshold study on standard corruption data, not a full multiclass Tent evaluation.

The primary predictor estimates the target entropy root and noise scale from unlabeled calibration scores. The sole primary comparison is equal-environment mean MAE at the lowest minus highest budget. Conditional empirical-Bernstein bounds for paired losses and simultaneous censor-compatible binomial reference bounds jointly give at least 95% coverage. Average MAE decreases from .108344 to .047292 and .016906. The gain is .091438, with interval [-.037798,.220674]: the independent confirmation remains unresolved. No paths are censored. The calibration and reference contributions to the gain radius are .101054 and .028182; the run used the illustrative 1,024/256 budgets, without a prior precision calculation.

Offline displacement strata contain 2, 3 and 17 environments for absolute t at most 1, between 1 and 3, and above 3. Their low/high-budget mean MAEs are .369887/.103069, .361697/.054515 and .032865/.000132. This descriptive pattern is compatible with greater difficulty near ambiguous directions, but does not establish a universal difficulty law. Target noise ratios are also reported offline. Exploration found almost identical errors for untrimmed source-influence moments and target-root estimates, so no unique benefit of nonlinear root estimation is claimed. A fixed K=1 Gaussian rule cannot output outside [.158655,.841345]; its transfer error on extreme probabilities does not refute the local theorem. Full protocols, failures and the six no-exit environments remain in the [confirmation analysis](../experiments/results/corrupt_confirmation/analysis.md).

## 10.17. Bounded nonaligned local directions

We check the direction-free prediction interface beyond the original tilt using k=t*psi/c^2+a*(tanh(z)^2-E0 tanh(Z)^2), with t in {-1,0,1}, a in {0,-.5,.5}, and n in {128,512}. The even centered perturbation is orthogonal to the odd influence function, so it preserves the first-order target t while changing the environment. All members retain the same signed label density and complete risk; continuous rejection sampling drives the actual nonlinear SGD. The predictor uses only the baseline influence and independent calibration scores, with no direction or generating coordinate input.

The 18 environments each have 1,024 reference paths and 256 calibration repetitions at each of three relative noise levels. All 18,432 paths exit within 1,500 updates. The maximum point-estimate discrepancy from Phi(t) is .023962. All 18 simultaneous reference intervals contain Phi(t), which is not an equivalence test. Across each batch and nuisance amplitude, averaging over the three targets gives MAE ranges .297-.321, .202-.212 and .111-.120 for r=2,1,.5. These are descriptive mechanism checks. They do not test an infinite direction class, compare optimized rules, or certify finite-n minimax optimality. Geometry quadrature and slope grids are diagnostics rather than rigorous bounds. See the [direction analysis](../experiments/results/unknown_directions/analysis.md).

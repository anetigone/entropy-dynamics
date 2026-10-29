# Contribution statement and literature comparison

Part of the [manuscript](DRAFT.md). The statistical results are in [local prediction](local_prediction.md), [Gaussian decision](gaussian_decision.md) and [unknown directions](unknown_direction.md). The earlier Gaussian risk/time comparison is retained in [the archive](notes/archive/gaussian_comparison.md). This note is based on the supplied manuscript, the local prediction extensions, and the primary sources linked below. It is a focused comparison, not an exhaustive priority search or a claim that no overlapping theorem exists.

## 1. Main question and contribution


One-sentence question: **With the initialization and the complete classification-risk function fixed and known, how accurately can independent unlabeled observations predict which side a future entropy-SGD trajectory will exit?**

The complete risk function identifies unsafe states; it does not identify the unlabeled dynamics that reach them. This distinction isolates a dynamical estimation difficulty from uncertainty about labels, safe endpoints, or the frozen comparator. The predictor observes M independent calibration samples before a separate SGD run with batch size n. It cannot observe that run's batches, trajectory, exit time or branch.

The proposed contribution has three parts:

1. **Risk-known dynamical uncertainty.** Construct legal target environments sharing the complete risk function, fixed initialization, source information and a connected risk-safe set. Prove a uniform approximation for the actual nonlinear fresh-batch entropy-SGD exit functional on local environments. Risk exit equals interval exit path by path.
2. **Information and best achievable precision.** Connect that functional to a Gaussian decision problem through experiment-level risk transfer. Establish the calibration threshold M/n->infinity and characterize the finite-ratio minimax value. Extend the same value to an unknown bounded local perturbation direction around a known baseline, identifying the entropy-gradient tilt as least favorable for the normalized root target.
3. **Quantified practical improvement within the model.** Obtain Gaussian minimax brackets from a prior lower bound and a continuous-window rule upper bound, then evaluate the fixed rule on independent real entropy-SGD runs. Report calibration and reference-path uncertainty jointly. Do not turn finite-grid experiments into a theorem about continuum worst-case risk.

Classical ingredients should be credited explicitly: unstable-equilibrium exit mechanisms, two-point testing, local asymptotic normality and experiment equivalence, influence-function geometry, posterior medians under absolute loss, and finite-prior numerical decision methods. The claim concerns their model-specific connection and its consequences, not invention of those tools.

## 2. Focused literature comparison

The distinctions in the last column are comparisons with this manuscript's declared observation experiment, not assertions that the cited work is deficient.

| Work | Question and information available | Established role | Precise distinction here |
|---|---|---|---|
| Weerasinghe et al., WACV 2026, Logit-Adjusted TTA [R1] | Adaptation under partial class imbalance; modifies the entropy objective using class-prior information | Long-sequence degradation and a stabilization method | Our object is an independent forecast of a fixed update rule's exit probability. Degradation itself is not a contribution. |
| Bar, Shaer and Romano, NeurIPS 2024, POEM [R2] | Sequential entropy-shift detection coupled to adaptation | Betting-based detection and entropy matching; also a scalar Gaussian illustration of entropy collapse | A scalar collapse example and entropy/risk disagreement are already present. Our new burden is the shared-risk statistical experiment and true-SGD exit functional. |
| Schirmer et al., NeurIPS 2025, risk monitoring [R3] | Observes the evolving test stream and adapted model, with source calibration; uses a proxy condition | Sequential risk alarms and false-alarm control | Our forecast precedes an independent trajectory and predicts its branch probability. Their proxy assumption must be described rather than treating the monitor as assumption-free. |
| Jhawar and Wang, 2026 preprint, action identifiability [R4] | Selects among TTA actions from a declared unlabeled evidence channel | Indistinguishable-evidence constructions, testing bounds and a finite-batch recentering boundary | Their equal-marginal example changes action risks. Here the complete risk function is shared while marginals differ locally, and the target is a stochastic path probability. |
| Zhou et al., ICML 2026, recovery complexity [R5] | Learns under nonstationary test streams | Bounds post-shift recovery and longer-term learnability | We estimate an independent fixed-environment SGD exit functional. This row is based on the official abstract/framework page, not a theorem-by-theorem proof audit. |
| Kifer/Bakhtin unstable-exit theory [R6] | Studies random perturbations near an unstable equilibrium with specified dynamics | Logarithmic escape and random-seed behavior | It supplies the classical mechanism. The adaptation recursion, common risk geometry and unknown-environment information budget require an additional bridge. |
| Le Cam framework, Nussbaum exposition [R7] | Compares full statistical experiments using parameter-independent randomizations | Transfers bounded-loss decision risks | A moment CLT alone is insufficient. The manuscript proves the needed uniform likelihood and kernel statement for its tilt family. |
| Efficient influence-function and least-favorable-submodel theory [R8] | Relates target derivatives, information and model tangent spaces | Explains why influence-aligned directions are least favorable | The model-specific calculation is psi=U0/(lambda s) and its realization in legal shared-risk environments with uniform actual-SGD exits. |

The most important nearby distinctions are therefore the **forecast target**, **time at which prediction is made**, **allowed information**, and **which objects are identical across environments**. Different assumptions alone are not a novelty argument; the paper needs the completed quantitative result under its own experiment.

## 3. Relationship to prior work

Recent TTA research already addresses performance deterioration, statistically controlled monitoring, and limitations of action selection from unlabeled evidence [R1–R5]. Our setting asks a different prospective question: from an independent calibration sample, estimate the direction probability of a future run of a fixed entropy-SGD recursion. The construction deliberately supplies the same complete classification-risk function in every environment, so the difficulty persists after risk-boundary uncertainty is removed. Classical unstable-exit theory explains the underlying amplification mechanism [R6], while statistical-experiment and efficiency theory provide the comparison tools [R7–R8]. The contribution is their rigorous connection through a uniform approximation of the true entropy-SGD risk-exit functional, including finite-information minimax precision and its extension to unknown local perturbation directions.

## 4. Contribution boundaries

The main manuscript centers on independent prediction. Earlier rebound, tolerance and stopping results serve as supporting geometry and motivation; earlier visual failures motivate sensitivity to root calibration, and centered arctan experiments concern a specified predictor.

The statistical result should not be described as:

- a universal impossibility of unlabeled safety or TTA;
- a lower bound equal to the raw moment predictor's arctangent error;
- the first observation that entropy adaptation can deteriorate;
- a new general LAN theorem, semiparametric efficiency principle or minimax algorithm;
- proof that additional nuisance always worsens the leading information limit;
- a general unknown-target-distribution result without the known-baseline and bounded-locality restrictions.

The new unknown-direction theorem is stronger than the old known-tilt theorem, but its baseline remains known. Orthogonal local nuisance can be irrelevant at first order; unknown nonlocal baseline geometry is a different problem.

## 5. Source comparison records

[R1] Thilina Weerasinghe, Ruwan Tennakoon, WeiQin Chuah and Alireza Bab-Hadiashar. *Logit-Adjusted Test-Time Adaptation under Partial Class Imbalance*. WACV 2026, pp. 5977–5985. Official CVF abstract and paper record checked; direct HTML retrieval was intermittently forbidden. https://openaccess.thecvf.com/content/WACV2026/html/Weerasinghe_Logit-Adjusted_Test-Time_Adaptation_under_Partial_Class_Imbalance_WACV_2026_paper.html

[R2] Yarin Bar, Shalev Shaer and Yaniv Romano. *Protected Test-Time Adaptation via Online Entropy Matching: A Betting Approach*. NeurIPS 2024. Main text, including its scalar Gaussian example, checked. https://proceedings.neurips.cc/paper_files/paper/2024/file/9b35a0a20d617dc68ae98a7a57df2f51-Paper-Conference.pdf

[R3] Mona Schirmer, Metod Jazbec, Christian A. Naesseth and Eric Nalisnick. *Monitoring Risks in Test-Time Adaptation*. NeurIPS 2025. Setting, Assumption 1 and online threshold calibration checked. https://proceedings.neurips.cc/paper_files/paper/2025/file/746960ad49ddb47248970a0e1404230c-Paper-Conference.pdf

[R4] Kartik Jhawar and Lipo Wang. *When Is Test-Time Adaptation Identifiable from Unlabeled Evidence?* arXiv:2609.11235v1, 10 September 2026. Observation channel, equal-marginal construction, testing bound and recentering section checked. https://arxiv.org/html/2609.11235v1

[R5] Zhi Zhou et al. *On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective*. ICML 2026, PMLR 306:166172–166195. Official publication page and author framework page checked; full-PDF retrieval failed in this run, so no detailed theorem-overlap claim is made. https://proceedings.mlr.press/v306/zhou26be.html ; https://zhouz.dev/TTA-Learnability/

[R6] Yuri Bakhtin. *Exit asymptotics for small diffusion about an unstable equilibrium*. Stochastic Processes and their Applications 118(5):839–851, 2008; arXiv:math/0701569v3. Main setting and introductory comparison with Kifer (1981) checked. https://arxiv.org/html/math/0701569v3

[R7] Michael Nussbaum. *Asymptotic Equivalence of Statistical Experiments*. Definitions of deficiency and its decision interpretation checked. https://pi.math.cornell.edu/~nussbaum/papers/04-1-transl.pdf

[R8] Mark van der Laan and Susan Gruber. *One-Step Targeted Minimum Loss-based Estimation Based on Universal Least Favorable One-Dimensional Submodels*. International Journal of Biostatistics 12(1):351–378, 2016. Publisher abstract and institutional working-paper record checked for the influence-aligned submodel principle; not used as a substitute for the present proof. https://doi.org/10.1515/ijb-2015-0054 ; https://biostats.bepress.com/ucbbiostat/paper347/

[R9] Andreas Maurer and Massimiliano Pontil. *Empirical Bernstein Bounds and Sample Variance Penalization*. COLT 2009, Theorem 4. Used for the confirmation experiment's calibration uncertainty. https://www.cs.mcgill.ca/~colt2009/papers/012.pdf

### 5.1. Proof of the information threshold

Write r_n for the uniform error in Theorem 1, so r_n<=Ce_n. We retain source equation labels to connect each step to the complete proof chapters.

 At theta=+/-a/sqrt(n), positivity gives
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

The matching consistency argument uses the raw moment estimator already defined above. Its exact variance is at most 1/M, so the Lipschitz bound gives risk at most c sqrt(n/M)/sqrt(2pi)+Ce_n. If the budget ratio does not diverge, a bounded-ratio subsequence and the two-point lower bound exclude consistency.

### 5.2. Exact low-information minimax limit

 For the upper bound, L.2 and monotonicity of Phi give

\[
\sup_{|u|\le A_0}|\tfrac12-\pi_n(Q_{u/\sqrt n})|
\le B_0+r_n.
\]

For the lower bound take a=A0 in the density-overlap bound. The calibration TV bound tends to zero and the target separation tends to 2B0, so liminf risk>=B0. These two bounds agree. If M=0 is allowed, use the identical no-observation experiment, with TV exactly zero. □

This is an exact asymptotic constant only in the regime M/n->0. It is not a solution of the positive finite-ratio minimax problem. The result strengthens the interpretation of the sample threshold: a best constant predictor may be optimal when information vanishes, while its nonzero worst-case error still prevents uniform consistency.

### 5.3. Full-experiment equivalence: likelihoods and both kernels

Write P_{n,0}=Q0^{otimes M}, rho_n=M/n, and

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

### 5.4. Minimax risk transfer

 Project all decision values to [0,1], which cannot increase loss. Absolute loss is then bounded by one, so transporting any randomized rule through either Markov kernel changes each parameterwise risk by at most epsilon_n. Replacing the target pi_n by Phi(cu) changes each loss by at most r_n. Taking supremum risks and then infima in both directions gives

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

### 5.5. Uniform risk of the raw moment predictor

 By the exact calibration moment identities, the standardized calibration sum

\[
Z_{n,u}=\frac{\sqrt M(\widehat\theta-u/\sqrt n)}
 {\sqrt{1-u^2/n}}
\]

has a uniform Berry–Esseen bound C/sqrt(M) on the compact u-window: h is bounded and its variance is uniformly bounded away from zero for large n. Hence

\[
c\sqrt n\widehat\theta
=cu+c\sqrt{n/M}\sqrt{1-u^2/n}\,Z_{n,u}.
\]

The coefficient of Z_{n,u} tends uniformly to r. This implies uniform convergence of the bounded loss expectations to D(cu,r). One direct justification is by subsequences: any alleged nonuniformity supplies u_n in the compact window; extract u_n->u_*, apply the triangular CLT and bounded continuous mapping along that subsequence, and use continuity of D. Replacing Phi(cu) by the true pi_n changes expected absolute loss by at most r_n. This proves the uniform moment-risk convergence.

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

Since D(0,0)=0, integration yields D(0,r)=arctan(r)/pi. Uniform convergence in the uniform moment-risk convergence allows passage to the supremum, proving (L.17). □

The arctan expression remains a result for this specified predictor. It is now its worst-case local-window risk limit, rather than a universal lower bound. In this known family, the constant predictor has worst-case limiting risk B0; it can outperform the raw moment predictor for some combinations of window width and calibration ratio.

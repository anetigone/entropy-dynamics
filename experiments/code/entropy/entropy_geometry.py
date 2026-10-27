#!/usr/bin/env python3
"""Reproduce the theory note's checks; synthetic diagnostics, not a proof assistant.

Run this file directly to compute results; run geometry_plot.py separately to plot.
Dependencies: numpy and scipy; plotting also uses matplotlib.
Outputs are saved in experiments/results/geometry.
All stochastic batches are fresh; labels enter only the synthetic generator and
the analytic evaluation risk. No target-label stopping rule is evaluated here.
"""
from pathlib import Path
import json
import platform
import numpy as np
import scipy
from scipy.special import expit, ndtr, roots_hermitenorm, zeta
from scipy.integrate import quad
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parents[3] / "experiments" / "results" / "geometry"
SEED = 20261024


class Model:
    def __init__(self, m=1.0, sigma=1.0, w=2.0, prior=.35, order=240):
        self.m, self.sigma, self.w, self.prior = m, sigma, w, prior
        z, weights = roots_hermitenorm(order)
        self.normal_nodes = z
        self.normal_weights = weights / np.sqrt(2*np.pi)
        self.z = np.r_[m+sigma*z, -m+sigma*z]
        self.weights = np.r_[prior*self.normal_weights,
                             (1-prior)*self.normal_weights]

    def velocity(self, x):
        u = self.w*(self.z-x)
        return -np.dot(self.weights, self.w*u*expit(u)*expit(-u))

    def entropy(self, x):
        q = np.abs(self.w*(self.z-x))
        h = np.logaddexp(0, -q) + q*expit(-q)
        return np.dot(self.weights, h)

    def risk(self, x):
        return (self.prior*ndtr((x-self.m)/self.sigma)
                +(1-self.prior)*ndtr((-x-self.m)/self.sigma))

    def beta_lambda(self):
        u = self.w*(self.m+self.sigma*self.normal_nodes)
        a = expit(u)*expit(-u)
        beta = np.dot(self.normal_weights, self.w*u*a)
        lam = np.dot(self.normal_weights,
                     self.w**2*a*(1-u*np.tanh(u/2)))
        return float(beta), float(lam)

    def crossing_boundary(self, x0, epsilon):
        xstar = self.sigma**2/(2*self.m)*np.log((1-self.prior)/self.prior)
        target = self.risk(x0)+epsilon
        assert target < self.prior
        return brentq(lambda x: self.risk(x)-target, xstar, 30.0)


def crossing(model, x0, boundary, eta=.1, max_steps=100000):
    x = x0
    for t in range(max_steps+1):
        if x > boundary:
            return t
        x += eta*model.velocity(x)
    raise RuntimeError("Population crossing not reached")


def trajectory(model, delta=0.0, eta=.1, steps=2500):
    x = -delta
    xs, risks, entropies = [], [], []
    for t in range(steps+1):
        xs.append(x)
        risks.append(model.risk(x))
        entropies.append(model.entropy(x))
        x += eta*model.velocity(x)
    return np.array(xs), np.array(risks), np.array(entropies)


def zero_tolerance_boundary(model, rho):
    # Work in x=rho*u and divide the risk difference by rho^2. This
    # avoids subtracting nearly equal CDF values when rho is very small.
    m, sigma = model.m, model.sigma
    k = sigma*sigma/m
    prefactor = np.exp(-.5*(m/sigma)**2)/(sigma*np.sqrt(2*np.pi))
    def scaled_risk_difference(u):
        def integrand(t):
            a = m*rho*t/(sigma*sigma)
            return (np.exp(-.5*(rho*t/sigma)**2)
                    *(np.sinh(a)/rho-np.cosh(a)))
        return prefactor*quad(integrand, 0, u, epsabs=1e-13)[0]
    return rho*brentq(scaled_risk_difference, k, 4*k)


def drift_enclosure():
    """Analytic discretization error + floating point numerical evaluation.

    This is not an interval-arithmetic certificate. The separate 1e-8 pad is
    a conservative numerical guard, not a proved IEEE rounding error bound.
    The stated midpoint and spatial interpolation errors are analytic bounds.
    """
    m = sigma = 1.0
    w, prior = 2.0, .35
    M, L, B3 = w/4, w*w/4, .75*w**3
    trunc = 8.0
    nq = 8192
    h = 2*trunc/nq
    z = -trunc+(np.arange(nq)+.5)*h
    weights = np.exp(-z*z/2)/np.sqrt(2*np.pi)*h
    xs = np.linspace(-.063, .714, 801)
    vals = []
    for x in xs:
        up = w*(m+sigma*z-x)
        un = w*(-m+sigma*z-x)
        vals.append(-np.dot(weights, prior*w*up*expit(up)*expit(-up)
                           +(1-prior)*w*un*expit(un)*expit(-un)))
    vals = np.array(vals)
    phi_max = 1/np.sqrt(2*np.pi)
    phi_prime_max = 1/np.sqrt(2*np.pi*np.e)
    integrand_second_bound = (B3*sigma*sigma*phi_max
                              +2*L*sigma*phi_prime_max+M*phi_max)
    quadrature_error = 2*trunc*h*h/24*integrand_second_bound
    tail_error = M*2*ndtr(-trunc)
    interpolation_error = L*(xs[1]-xs[0])/2
    rounding_guard = 1e-8
    error = quadrature_error+tail_error+interpolation_error+rounding_guard
    lower, upper = float(vals.min()-error), float(vals.max()+error)
    assert lower > .049 and upper < .098
    return dict(interval=[-.063, .714], raw_min=float(vals.min()),
                raw_max=float(vals.max()), quadrature_error=quadrature_error,
                tail_error=tail_error, interpolation_error=interpolation_error,
                numerical_rounding_guard=rounding_guard,
                lower=lower, upper=upper,
                status="analytic discretization error; no formal floating-point certification")


def stochastic_check(paths=2000, batch=64, max_steps=350):
    rng = np.random.default_rng(SEED)
    model = Model()
    eta, epsilon = .1, .005
    left = brentq(lambda x: model.risk(x)-model.risk(0)-epsilon, -3, 0)
    right = model.crossing_boundary(0, epsilon)
    assert -.063 < left < -.062 and .713 < right < .714
    x = np.zeros(paths)
    times = np.full(paths, -1, dtype=int)
    exit_left = np.zeros(paths, dtype=bool)
    for t in range(1, max_steps+1):
        active = np.flatnonzero(times < 0)
        if len(active) == 0:
            break
        y = np.where(rng.random((len(active), batch)) < model.prior, 1., -1.)
        features = model.m*y + model.sigma*rng.standard_normal(y.shape)
        u = model.w*(features-x[active, None])
        x[active] -= eta*np.mean(model.w*u*expit(u)*expit(-u), axis=1)
        out = (x[active] < left) | (x[active] > right)
        indices = active[out]
        times[indices] = t
        exit_left[indices] = x[indices] < left
    assert np.all(times > 0)
    M, nu, V, d, Dlo, Dhi = .5, .049, .098, .062, .713, .714
    low, high = 50, 210
    p_left = np.exp(-2*batch*nu*d/(eta*M*M))
    p_early = np.exp(-batch*(Dlo-eta*V*low)**2/(2*eta**2*M*M*low))
    p_late = np.exp(-batch*(eta*nu*high-Dhi)**2/(2*eta**2*M*M*high))
    summary = dict(paths=paths, batch=batch, seed=SEED, left_root=float(left),
                   right_root=float(right), observed_min=int(times.min()),
                   observed_max=int(times.max()),
                   percentiles=dict(zip(["2.5", "25", "50", "75", "97.5"],
                                        np.percentile(times,[2.5,25,50,75,97.5]).tolist())),
                   observed_left_exits=int(exit_left.sum()),
                   observed_outside_bound=int(np.sum((times<=low)|(times>high))),
                   proposed_interval=[low,high],
                   failure_terms=dict(left=float(p_left), early=float(p_early), late=float(p_late)),
                   computed_coverage_lower_bound=float(1-p_left-p_early-p_late),
                   numerical_caveat="coverage instantiation uses the separately checked drift constants")
    return summary, times


def main():
    model = Model()
    beta, lam = model.beta_lambda()
    eta, epsilon = .1, .005
    A = 1+eta*lam
    slope = 1/np.log(A)
    zero_index = np.log(1+2*lam/beta)/np.log(A)
    pilot = []
    for delta in [0., -.1, .1]:
        xs, rr, hh = trajectory(model, delta)
        first = int(np.flatnonzero(rr > rr[0]+epsilon)[0])
        pilot.append(dict(delta=delta, initial_risk=float(rr[0]),
                          min_risk=float(rr.min()), min_step=int(rr.argmin()),
                          T_epsilon=first,
                          entropy_initial=float(hh[0]), entropy_final=float(hh[-1]),
                          max_entropy_increment=float(np.diff(hh).max())))
    rhos = np.array([.3,.1,.03,.01,.003,.001,.0003,.0001,.00003,.00001,.000003,.000001])
    log_rows = []
    for rho in rhos:
        mm = Model(prior=(1-rho)/2)
        c_eps = mm.crossing_boundary(0, epsilon)
        c_zero = zero_tolerance_boundary(mm, rho)
        te = crossing(mm, 0, c_eps)
        tz = crossing(mm, 0, c_zero)
        log_rows.append(dict(rho=float(rho), T_fixed_epsilon=te, T_zero=tz,
                             centered_fixed_time=float(te-slope*np.log(1/rho))))
    safe = []
    C0, C2 = 2*zeta(2,1), 14*zeta(4,1)
    for prior in [.5, .49]:
        mm = Model(m=2.,w=5.,prior=prior,order=512)
        xs, rr, hh = trajectory(mm, delta=.2, eta=.02, steps=6000)
        b, l = mm.beta_lambda()
        safe.append(dict(m=2., sigma=1., w=5., prior=prior, delta=.2, eta=.02,
                         initial_risk=float(rr[0]), final_risk=float(rr[-1]),
                         max_risk_excess=float(np.max(rr-rr[0])),
                         final_centered_threshold=float(xs[-1]),
                         symmetric_entropy_curvature=-l,
                         sufficient_wsigma_squared=(3*C2/(2*C0))*np.exp(2)/3,
                         actual_wsigma_squared=25.))
    enclosure = drift_enclosure()
    sgd, times = stochastic_check()
    checks = dict(pilot_matches=(pilot[0]['min_step']==47 and pilot[0]['T_epsilon']==94),
                  small_imbalance_zero_crossing_is_79=all(r['T_zero']==79 for r in log_rows[-5:]),
                  safe_examples_no_risk_increase=all(r['max_risk_excess']<=1e-12 for r in safe))
    assert all(checks.values())
    results = dict(environment=dict(python=platform.python_version(), numpy=np.__version__,
                                    scipy=scipy.__version__),
                   beta=beta, lambda_=lam, log_slope=slope, zero_continuous_index=zero_index,
                   pilot=pilot, log_law=log_rows, stable_examples=safe,
                   drift_enclosure=enclosure, stochastic=sgd, checks=checks)
    (OUT/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    np.savez_compressed(OUT/'sgd_crossing_times.npz', times=times)

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    main()

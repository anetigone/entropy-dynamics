#!/usr/bin/env python3
"""Independent numerical checks, not interval-certified proof constants."""
import json
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit
from entropy_critical import Geometry, ROOT, M, SIGMA, W, ETA, finite_cdf, finite_sample_scaled_endpoints


def density(z, rho=0.0):
    return ((1-rho)/2*np.exp(-.5*((z-M)/SIGMA)**2)
            +(1+rho)/2*np.exp(-.5*((z+M)/SIGMA)**2))/(SIGMA*np.sqrt(2*np.pi))


def U(x,z):
    u=W*(z-x)
    return -W*u*expit(u)*expit(-u)


def expectation(f,rho=0.0):
    return quad(lambda z:f(z)*density(z,rho),-np.inf,np.inf,epsabs=2e-12,epsrel=2e-12)[0]


def velocity(x,rho=0.0):
    return expectation(lambda z:U(x,z),rho)


def main():
    g=Geometry()
    beta=-quad(lambda z:U(0,z)*np.exp(-.5*((z-M)/SIGMA)**2)/(SIGMA*np.sqrt(2*np.pi)),
               -np.inf,np.inf,epsabs=2e-12,epsrel=2e-12)[0]
    lam=expectation(lambda z:W**2*expit(W*z)*expit(-W*z)*(1-W*z*np.tanh(W*z/2)))
    sg2=expectation(lambda z:U(0,z)**2)
    errors={name:abs(a-b) for name,a,b in [('beta',beta,g.beta),('lambda',lam,g.lam),('variance',sg2,g.sg2)]}
    peak_checks=[]
    for rho in [0.,.02,.04]:
        peak=brentq(lambda x:velocity(x,rho),-M,M,xtol=1e-13)
        fg=g.finite_geometry(rho,32,.005)
        peak_checks.append(dict(rho=rho,quad_peak=peak,hermite_peak=fg['q'],difference=abs(peak-fg['q'])))
    micro_errors=[]
    for n in [32,128,512]:
        lo,hi=finite_sample_scaled_endpoints(n,.2,.005)
        rho=.2/np.sqrt(n)
        err=max(abs(n*(g.risk(y/np.sqrt(n),rho)-g.risk(0.,rho))-.005) for y in [lo,hi])
        micro_errors.append(dict(n=n,scaled_risk_error=err))
    results=json.loads((ROOT/'critical_results.json').read_text())
    for row in results['macro']:
        cdf=finite_cdf(np.arange(0,3001),row['n'],row['finite_geometry'])
        assert np.all(cdf>=0) and np.all(cdf<=1)
        assert np.all(np.diff(cdf)>=-1e-14)
    # Independently differentiate the entropy, rather than differentiating U again.
    z=np.random.default_rng(7654).normal(size=200)
    def entropy(x):
        u=np.abs(W*(z-x))
        return np.mean(np.logaddexp(0,-u)+u*expit(-u))
    grad_errors=[]
    for x in [-.2,0.,.2]:
        h=1e-5
        numerical_velocity=-(entropy(x+h)-entropy(x-h))/(2*h)
        grad_errors.append(abs(numerical_velocity-np.mean(U(x,z))))
    audit=dict(constant_comparison=errors,peak_comparison=peak_checks,
               max_entropy_gradient_error=max(grad_errors),micro_boundary_checks=micro_errors,
               max_cutoff_comparison=max(r['finite_geometry']['cutoff_check'] for r in results['macro']),
               max_conjugacy_residual=max(r['finite_geometry']['conjugacy_residual'] for r in results['macro']),
               all_nine_finite_parameter_cdfs_valid=True,
               interpretation='Floating-point diagnostic only; not interval-certified error bounds or validation of novelty.')
    assert max(errors.values())<1e-9
    assert max(r['difference'] for r in peak_checks)<1e-9
    assert max(grad_errors)<1e-8
    assert max(r['scaled_risk_error'] for r in micro_errors)<1e-10
    (ROOT/'geometry_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()

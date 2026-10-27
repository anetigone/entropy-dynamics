"""Gaussian-mixture threshold adaptation; CPU, fixed seed, no target labels used in updates."""
from pathlib import Path
import json, csv
import numpy as np
from scipy.special import expit, ndtr, roots_hermitenorm

OUT = Path(__file__).resolve().parents[3] / "experiments"
X, Q = roots_hermitenorm(160)
Q /= np.sqrt(2*np.pi)
M, SIGMA, W, ETA, T = 1., 1., 2., .1, 2500

def risk(b, delta, pi):
    return pi*ndtr((b-delta-M)/SIGMA)+(1-pi)*ndtr((delta-M-b)/SIGMA)

def grad(z, b):
    u = W*(z-b)
    return W*u*expit(u)*expit(-u)

def drift(b, delta, pi):
    return pi*np.dot(Q, grad(delta+M+SIGMA*X,b))+(1-pi)*np.dot(Q,grad(delta-M+SIGMA*X,b))

def population(delta, pi):
    b = np.zeros(T+1)
    for t in range(T):
        b[t+1] = b[t]-ETA*drift(b[t],delta,pi)
    return b, risk(b,delta,pi)

def main():
    rows = []
    for delta in [-.6,-.3,0.,.3,.6,1.]:
        for pi in [.2,.35,.5,.65,.8]:
            b,r = population(delta,pi)
            bad = np.flatnonzero(r>r[0]+.005)
            rows.append(dict(delta=delta,pi=pi,r0=float(r[0]),r_min=float(r.min()),r_final=float(r[-1]),
                             t_min=int(r.argmin()),first_exceed_005=int(bad[0]) if len(bad) else None,b_final=float(b[-1])))
    candidates = [r for r in rows if r['r_min']<r['r0']-.001 and r['r_final']>r['r0']+.005]
    chosen = max(candidates,key=lambda r:r['r_final']-r['r_min']) if candidates else rows[0]
    delta,pi = chosen['delta'],chosen['pi']
    b,r = population(delta,pi)
    rng = np.random.default_rng(20261024)
    paths = np.zeros((40,T+1))
    for t in range(T):
        y = np.where(rng.random((40,64))<pi,1.,-1.)
        z = delta+M*y+SIGMA*rng.normal(size=y.shape)
        paths[:,t+1] = paths[:,t]-ETA*grad(z,paths[:,t,None]).mean(axis=1)
    rr = risk(paths,delta,pi)
    zplus,zminus=delta+M+SIGMA*X,delta-M+SIGMA*X
    def ent(z):
        u=W*(z[None,:]-b[:,None])
        return np.logaddexp(0,u)-u*expit(u)
    entropy=pi*(ent(zplus)@Q)+(1-pi)*(ent(zminus)@Q)
    # Independent quadrature order checks and finite-difference sign check.
    x2,q2=roots_hermitenorm(256);q2/=np.sqrt(2*np.pi)
    err=max(abs(drift(v,delta,pi)-(pi*np.dot(q2,grad(delta+M+x2,v))+(1-pi)*np.dot(q2,grad(delta-M+x2,v)))) for v in np.linspace(-4,4,41))
    u=W*(zplus-.13); h=lambda bb:np.dot(Q,np.logaddexp(0,W*(zplus-bb))-W*(zplus-bb)*expit(W*(zplus-bb)))
    fd=(h(.130001)-h(.129999))/.000002
    fd_error=abs(fd-np.dot(Q,grad(zplus,.13)))
    report=dict(settings=dict(m=M,sigma=SIGMA,w=W,eta=ETA,steps=T,n=64,seeds=40,epsilon=.005,seed=20261024),
                selected=chosen,population_rebound_cases=len(candidates),grid_cases=len(rows),
                quadrature_max_abs_error=float(err),gradient_finite_difference_error=float(fd_error),
                entropy_start=float(entropy[0]),entropy_final=float(entropy[-1]),
                stochastic_final_risk_quantiles=np.quantile(rr[:,-1],[.1,.5,.9]).tolist())
    (OUT/'pilot_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (OUT/'population_grid.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    np.savez_compressed(OUT/'pilot_paths.npz',b=b,risk=r,entropy=entropy,b_sgd=paths,risk_sgd=rr)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    main()

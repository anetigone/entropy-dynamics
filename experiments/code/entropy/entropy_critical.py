#!/usr/bin/env python3
"""Nonlinear entropy SGD in the weak-imbalance, large-batch regime.

All empirical trajectories use actual fresh-sample entropy gradients. Gaussian
AR(1) simulations are separately labeled reference limits, never substituted
for the nonlinear SGD. No time offset or direction parameter is fitted.
"""
from pathlib import Path
import json
import platform
import time
import numpy as np
import scipy
from scipy.special import expit, ndtr, roots_hermitenorm
from scipy.integrate import quad
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[3] / "experiments" / "results" / "critical"
SEED = 20261026
ETA = .1
M = SIGMA = 1.
W = 2.


class Geometry:
    def __init__(self, order=240):
        self.z, self.ww = roots_hermitenorm(order)
        self.ww /= np.sqrt(2*np.pi)
        self.nodes = np.r_[M+SIGMA*self.z, -M+SIGMA*self.z]
        self.weights = np.r_[.5*self.ww, .5*self.ww]
        u = W*(M+SIGMA*self.z)
        a = expit(u)*expit(-u)
        self.beta = float(np.dot(self.ww, W*u*a))
        self.lam = float(np.dot(self.ww, W*W*a*(1-u*np.tanh(u/2))))
        u = W*self.nodes
        uv = -W*u*expit(u)*expit(-u)
        self.sg2 = float(np.dot(self.weights,uv*uv))
        self.sg = np.sqrt(self.sg2)
        self.A = 1+ETA*self.lam
        self.seed_sd = np.sqrt(ETA**2*self.sg2/(self.A**2-1))
        self.rho_scale = self.lam*self.seed_sd/self.beta
        # v(x) = lambda*x + v3*x^3 + ... at rho=0.
        q = np.abs(W*(M+SIGMA*self.z))
        h = np.logaddexp(0,-q)+q*expit(-q)
        k4 = np.dot(self.ww,h*(self.z**4-6*self.z**2+3))/SIGMA**4
        self.v3 = float(-k4/6)
        self.h3 = -ETA*self.v3/(self.A**3-self.A)

    def velocity(self, x):
        if abs(x) < 1e-5:
            return self.lam*x+self.v3*x**3
        u = W*(self.nodes-x)
        return -float(np.dot(self.weights,W*u*expit(u)*expit(-u)))

    @staticmethod
    def risk(x, rho=0.):
        p=(1-rho)/2
        return p*ndtr((x-M)/SIGMA)+(1-p)*ndtr((-x-M)/SIGMA)

    def endpoints(self, rho, epsilon, x0=0.):
        xstar=SIGMA**2/(2*M)*np.log((1+rho)/(1-rho))
        target=self.risk(x0,rho)+epsilon
        left=brentq(lambda x:self.risk(x,rho)-target,-30.,xstar)
        right=brentq(lambda x:self.risk(x,rho)-target,xstar,30.)
        return left,right

    def conjugacy(self, x, cutoff=1e-4):
        sign=np.sign(x)
        y=abs(x)
        count=0
        while y>cutoff:
            target=y
            y=brentq(lambda t:t+ETA*self.velocity(t)-target,0.,target,xtol=5e-16)
            count+=1
        return sign*self.A**count*(y+self.h3*y**3),count

    def boundary_constant(self, epsilon):
        _,c=self.endpoints(0.,epsilon)
        h1,j1=self.conjugacy(c,1e-3)
        h2,j2=self.conjugacy(c,1e-4)
        return dict(epsilon=epsilon,risk_boundary=c,H_boundary=h2,
                    cutoff_comparison_error=abs(h1-h2),inverse_steps=j2)

    def finite_geometry(self,rho,n,epsilon,x0=0.):
        weights=np.r_[((1-rho)/2)*self.ww,((1+rho)/2)*self.ww]
        def derivatives(x):
            u=W*(self.nodes-x); a=expit(u)*expit(-u); t=np.tanh(u/2)
            v=-np.dot(weights,W*u*a)
            vp=np.dot(weights,W**2*a*(1-u*t))
            vpp=np.dot(weights,W**3*a*(2*t+u*(2*a-t*t)))
            vppp=-np.dot(weights,W**4*a*(6*a-3*t*t+u*t*(t*t-8*a)))
            return float(v),float(vp),float(vpp),float(vppp)
        q=brentq(lambda x:derivatives(x)[0],-M,M,xtol=1e-14)
        _,lam,vpp,vppp=derivatives(q)
        A=1+ETA*lam
        b=ETA*vpp/2; c3=ETA*vppp/6
        h2=-b/(A*A-A)
        h3=-(c3+2*h2*A*b)/(A**3-A)
        def H(x,cutoff=1e-4):
            y=x; count=0
            while abs(y-q)>cutoff:
                target=y
                lo,hi=sorted([q,y])
                y=brentq(lambda z:z+ETA*derivatives(z)[0]-target,lo,hi,xtol=5e-16)
                count+=1
            d=y-q
            return A**count*(d+h2*d*d+h3*d**3)
        lo,hi=self.endpoints(rho,epsilon,x0)
        u=W*(self.nodes-q); uv=-W*u*expit(u)*expit(-u)
        sg2=float(np.dot(weights,uv*uv)-np.dot(weights,uv)**2)
        variance=ETA*ETA*sg2/(A*A-1)
        values=dict(q=q,A=A,lambda_=lam,seed_mean=np.sqrt(n)*H(x0),
                    seed_sd=np.sqrt(variance),left_H=-H(lo),right_H=H(hi),
                    left_boundary=lo,right_boundary=hi)
        values['cutoff_check']=max(abs(H(lo,1e-3)+values['left_H']),
                                   abs(H(hi,1e-3)-values['right_H']))
        values['conjugacy_residual']=max(
            abs(H(x+ETA*derivatives(x)[0])-A*H(x)) for x in [lo,x0,hi])
        return values


def cdf_prediction(t, n, kappa, H_boundary, g):
    # kappa is the limiting Gaussian seed mean divided by its SD.
    threshold=H_boundary*np.sqrt(n)*np.exp(-np.asarray(t)*np.log(g.A))/g.seed_sd
    return ndtr(kappa-threshold)+ndtr(-kappa-threshold)


def finite_cdf(t,n,geometry):
    factor=np.sqrt(n)*np.exp(-np.asarray(t)*np.log(geometry['A']))
    mu,sd=geometry['seed_mean'],geometry['seed_sd']
    return ndtr((mu-factor*geometry['right_H'])/sd)+ndtr((-factor*geometry['left_H']-mu)/sd)


def refine_saved_results():
    results=json.loads((ROOT/'critical_results.json').read_text())
    arrays=dict(np.load(ROOT/'critical_samples.npz'))
    g=Geometry()
    for row in results['macro']:
        n,kappa=row['n'],row['kappa']
        fg=g.finite_geometry(row['rho'],n,row['epsilon'])
        tt=arrays[f'macro_n{n}_k{int(kappa)}_times']
        grid=np.arange(0,int(tt.max())+500)
        predicted=finite_cdf(grid,n,fg)
        empirical=np.searchsorted(np.sort(tt),grid,side='right')/len(tt)
        row['finite_geometry']=fg
        row['refined_cdf_error']=float(np.max(abs(empirical-predicted)))
        row['refined_median']=int(grid[np.flatnonzero(predicted>=.5)[0]])
        print(json.dumps({k:row[k] for k in ['n','kappa','median','predicted_median','refined_median','cdf_sup_error','refined_cdf_error']}),flush=True)
    results['refinement_note']='All finite-parameter geometry computed from the model; no fit to SGD outcomes. Asymptotically equivalent correction, no finite-n error guarantee.'
    (ROOT/'critical_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


def simulate_actual(n,paths,rho,epsilon,rng,geometry,max_steps=2500,scaled_endpoints=None,x0=0.):
    if scaled_endpoints is None:
        lo,hi=geometry.endpoints(rho,epsilon,x0)
    else:
        lo,hi=np.array(scaled_endpoints)/np.sqrt(n)
    x=np.full(paths,x0,dtype=float)
    times=np.full(paths,-1,dtype=np.int32)
    directions=np.zeros(paths,dtype=np.int8)
    for t in range(1,max_steps+1):
        ids=np.flatnonzero(times<0)
        if not len(ids):break
        z=rng.standard_normal((len(ids),n))*SIGMA
        z+=np.where(rng.random(z.shape)<(1-rho)/2,M,-M)
        u=W*(z-x[ids,None])
        x[ids]-=ETA*np.mean(W*u*expit(u)*expit(-u),axis=1)
        exited=(x[ids]<lo)|(x[ids]>hi)
        out=ids[exited]
        times[out]=t
        directions[out]=np.where(x[out]>hi,1,-1)
    return times,directions


def finite_sample_scaled_endpoints(n,c,e):
    # Exact n*[r_{c/sqrt(n)}(y/sqrt(n))-r(...0)] evaluated without CDF cancellation.
    pref=np.exp(-M*M/(2*SIGMA*SIGMA))/(SIGMA*np.sqrt(2*np.pi))
    def delta(y):
        def integrand(t):
            a=M*t/(SIGMA*SIGMA*np.sqrt(n))
            return np.exp(-t*t/(2*n*SIGMA*SIGMA))*(np.sqrt(n)*np.sinh(a)-c*np.cosh(a))
        return pref*quad(integrand,0,y,epsabs=1e-12)[0]-e
    rho=c/np.sqrt(n)
    star=np.sqrt(n)*SIGMA**2/(2*M)*np.log((1+rho)/(1-rho))
    lo=brentq(delta,-10,star)
    hi=brentq(delta,star,10)
    return lo,hi


def simulate_gaussian_ar(paths,c,e,rng,g,max_steps=2500):
    k=SIGMA*SIGMA/M
    B=M*np.exp(-M*M/(2*SIGMA*SIGMA))/(SIGMA**3*np.sqrt(2*np.pi))
    radius=np.sqrt((k*c)**2+2*e/B)
    lo,hi=k*c-radius,k*c+radius
    y=np.zeros(paths)
    times=np.full(paths,-1,dtype=np.int32)
    for t in range(1,max_steps+1):
        ids=np.flatnonzero(times<0)
        if not len(ids):break
        y[ids]=g.A*y[ids]+ETA*g.beta*c+ETA*g.sg*rng.standard_normal(len(ids))
        out=ids[(y[ids]<lo)|(y[ids]>hi)]
        times[out]=t
    return times


def main():
    started=time.monotonic()
    g=Geometry()
    constants={name:float(getattr(g,name)) for name in ['beta','lam','sg2','sg','A','seed_sd','rho_scale','v3','h3']}
    constants['doubling_delay']=.5*np.log(2)/np.log(g.A)
    constants['batch_log_slope']=.5/np.log(g.A)
    boundaries={str(e):g.boundary_constant(e) for e in [.005,.15]}
    rng=np.random.default_rng(SEED)
    rows=[]
    arrays={}
    for n in [32,128,512]:
        for kappa in [0.,1.,2.]:
            paths=600
            rho=g.rho_scale*kappa/np.sqrt(n)
            tt,dd=simulate_actual(n,paths,rho,.005,rng,g)
            key=f'macro_n{n}_k{int(kappa)}'
            arrays[key+'_times']=tt
            arrays[key+'_directions']=dd
            assert np.all(tt>0), 'Censored paths must not silently be dropped'
            grid=np.arange(0,int(tt.max())+500)
            pred=cdf_prediction(grid,n,kappa,boundaries['0.005']['H_boundary'],g)
            empirical=np.mean(tt[:,None]<=grid[None,:],axis=0)
            qpred=[int(grid[np.flatnonzero(pred>=q)[0]]) for q in [.1,.5,.9]]
            row=dict(n=n,kappa=kappa,rho=rho,paths=paths,epsilon=.005,
                     right_fraction=float(np.mean(dd>0)),right_prediction=float(ndtr(kappa)),
                     median=float(np.median(tt)),predicted_median=qpred[1],
                     quantiles_observed=np.quantile(tt,[.1,.5,.9]).tolist(),
                     quantiles_predicted=qpred,cdf_sup_error=float(np.max(abs(empirical-pred))),
                     cdf_sampling_band_95=float(np.sqrt(np.log(40)/(2*paths))),
                     censored=int(np.sum(tt<0)))
            rows.append(row)
            print(json.dumps(row),flush=True)
    # A wider risk interval checks the nonlinear boundary correction without fitting it.
    n,kappa,paths=128,1.,1000
    rho=g.rho_scale*kappa/np.sqrt(n)
    tt,dd=simulate_actual(n,paths,rho,.15,rng,g)
    assert np.all(tt>0)
    arrays['wide_times']=tt
    arrays['wide_directions']=dd
    grid=np.arange(0,int(tt.max())+500)
    pred=cdf_prediction(grid,n,kappa,boundaries['0.15']['H_boundary'],g)
    naive=cdf_prediction(grid,n,kappa,boundaries['0.15']['risk_boundary'],g)
    empirical=np.mean(tt[:,None]<=grid[None,:],axis=0)
    wide=dict(n=n,kappa=kappa,paths=paths,epsilon=.15,median=float(np.median(tt)),
              predicted_median=int(grid[np.flatnonzero(pred>=.5)[0]]),
              linear_boundary_median=int(grid[np.flatnonzero(naive>=.5)[0]]),
              nonlinear_cdf_error=float(np.max(abs(empirical-pred))),
              linear_boundary_cdf_error=float(np.max(abs(empirical-naive))))
    print(json.dumps({'wide_interval':wide}),flush=True)
    c,e=.2,.005
    reference=simulate_gaussian_ar(150000,c,e,rng,g)
    assert np.all(reference>0)
    arrays['gaussian_ar_reference_times']=reference
    micro=[]
    for n in [32,128,512]:
        endpoints=finite_sample_scaled_endpoints(n,c,e)
        tt,dd=simulate_actual(n,2000,c/np.sqrt(n),e/n,rng,g,scaled_endpoints=endpoints)
        assert np.all(tt>0)
        arrays[f'micro_n{n}_times']=tt
        grid=np.arange(0,max(int(tt.max()),int(reference.max()))+1)
        refcdf=np.searchsorted(np.sort(reference),grid,side='right')/len(reference)
        empirical=np.searchsorted(np.sort(tt),grid,side='right')/len(tt)
        row=dict(n=n,c=c,e=e,paths=2000,epsilon=e/n,
                 median=float(np.median(tt)),reference_median=float(np.median(reference)),
                 cdf_sup_error=float(np.max(abs(empirical-refcdf))),
                 reference_sampling_error_95=float(np.sqrt(np.log(40)/(2*len(reference)))),
                 actual_sampling_error_95=float(np.sqrt(np.log(40)/(2*len(tt)))))
        micro.append(row)
        print(json.dumps({'micro':row}),flush=True)
    one_step=[]
    n,paths=512,12000
    for c in [0.,.05,.1,.2,.5,1.,2.]:
        rho=c/np.sqrt(n)
        z=rng.standard_normal((paths,n))+np.where(rng.random((paths,n))<(1-rho)/2,1.,-1.)
        u=W*z
        x=-ETA*np.mean(W*u*expit(u)*expit(-u),axis=1)
        if c==0:
            actual=float(np.mean(g.risk(x,rho)<g.risk(0.,rho)))
            assert actual==0., 'Balanced Bayes initialization cannot improve in risk'
        else:
            # epsilon=0, x0=0: the other safe boundary is the positive risk root.
            endpoint=finite_sample_scaled_endpoints(n,c,0.)[1]/np.sqrt(n)
            actual=float(np.mean((x>0)&(x<endpoint)))
        prediction=float(ndtr((2*c-ETA*g.beta*c)/(ETA*g.sg))-ndtr(-g.beta*c/g.sg))
        one_step.append(dict(c=c,n=n,paths=paths,improvement_fraction=actual,prediction=prediction))
    results=dict(date='2026-10-26',seed=SEED,eta=ETA,model=dict(m=M,sigma=SIGMA,w=W),
                 environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                 constants=constants,boundaries=boundaries,macro=rows,wide_interval=wide,
                 micro=micro,one_step=one_step,seconds=time.monotonic()-started,
                 interpretation='Asymptotic formulas; no finite-n confidence guarantee or fitted offsets.')
    (ROOT/'critical_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    np.savez_compressed(ROOT/'critical_samples.npz',**arrays)
    print(json.dumps({'completed':True,'seconds':results['seconds']}),flush=True)


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    main()
    refine_saved_results()

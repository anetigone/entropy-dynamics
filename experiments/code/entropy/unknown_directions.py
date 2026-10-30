"""Known Gaussian baseline, unknown bounded local directions, actual entropy SGD."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, ndtr, roots_hermitenorm


sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'frozen_visual'))
from local_calibration import velocity


def constants(eta):
    z, weights = roots_hermitenorm(160)
    weights /= np.sqrt(2*np.pi)
    u = 2*z
    lam = float(weights @ (4*expit(u)*expit(-u)*(1-u*np.tanh(u/2))))
    variance = float(weights @ velocity(0, z)**2)
    sd = np.sqrt(eta**2*variance/((1+eta*lam)**2-1))
    c2 = variance/(lam*sd)**2
    center = quad(lambda x: np.tanh(x)**2*np.exp(-x*x/2)/np.sqrt(2*np.pi),
                  -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
    return z, weights, lam, float(sd), float(c2), center


def draw(rng, shape, n, t, amplitude, base):
    _, _, lam, sd, c2, center = base
    bound = abs(t)*.5/(lam*sd*c2)+abs(amplitude)
    size = int(np.prod(shape))
    out = np.empty(size)
    filled = 0
    while filled < size:
        z = rng.normal(size=max(32, int((size-filled)*(1+bound/np.sqrt(n))*1.05)))
        psi = velocity(0, z)/(lam*sd)
        k = t*psi/c2+amplitude*(np.tanh(z)**2-center)
        keep = rng.random(len(z))*(1+bound/np.sqrt(n)) < 1+k/np.sqrt(n)
        accepted = z[keep][:size-filled]
        out[filled:filled+len(accepted)] = accepted
        filled += len(accepted)
    return out.reshape(shape)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    values = [args.eta, args.boundary, args.envelope, *args.targets, *args.amplitudes, *args.ratios]
    if (not np.isfinite(values).all() or not 0 < args.eta < 1
            or min(args.boundary, args.envelope, *args.ratios) <= 0
            or min(*args.batches, args.paths, args.repeats, args.steps, args.block) < 1
            or args.repeats < 2 or args.seed < 0
            or any(len(x) != len(set(x)) for x in [args.targets, args.amplitudes, args.batches, args.ratios])):
        raise ValueError('Invalid numerical experiment parameters')
    base = constants(args.eta)
    z, weights, lam, sd, c2, center = base
    psi = velocity(0, z)/(lam*sd)
    # Even bounded nuisance is centered and orthogonal to the odd influence.
    nuisance = np.tanh(z)**2-center
    K = max(abs(t) for t in args.targets)
    if K*.5/(lam*sd*c2) > args.envelope:
        raise ValueError('Envelope must include the complete least-favorable target window')
    cells = []
    for n in args.batches:
        if args.envelope/np.sqrt(n) > .5:
            raise ValueError('Require B/sqrt(n) <= .5 for common joint density with kappa=.5')
        for t in args.targets:
            for amplitude in args.amplitudes:
                bound = abs(t)*.5/(lam*sd*c2)+abs(amplitude)
                if bound > args.envelope:
                    raise ValueError('Direction exceeds declared common envelope')
                k = t*psi/c2+amplitude*nuisance
                target_weights = weights*(1+k/np.sqrt(n))
                field = lambda x: float(target_weights @ velocity(x, z))
                q = brentq(field, -args.boundary/2, args.boundary/2)
                grid = np.linspace(-2*args.boundary, 2*args.boundary, 101)
                h = 1e-5
                slopes = [(field(x+h)-field(x-h))/(2*h) for x in grid]
                if min(slopes) <= 0 or field(-args.boundary) >= 0 or field(args.boundary) <= 0:
                    raise ValueError('Local field geometry diagnostic failed')
                if args.eta*.5 >= args.boundary/2:
                    raise ValueError('Step bound must keep boundary overshoots inside checked domain')
                cells.append(dict(key=f'cell_{len(cells)}', n=n, t=t, amplitude=amplitude,
                    envelope_bound=bound, mean_k=float(weights @ k),
                    influence_projection=float(weights @ (psi*k)),
                    information=float(weights @ k**2), exact_root_quadrature=q,
                    baseline_scaled_root=float(-np.sqrt(n)*q/sd),
                    minimum_checked_slope=float(min(slopes))))
    epsilon = .5*quad(lambda x: np.tanh(x)*np.exp(-x*x/2)/np.sqrt(2*np.pi),
                       0, args.boundary)[0]
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(lambda_=lam, seed_sd=sd, c_squared=c2, nuisance_center=center,
        kappa=.5, risk_tolerance=epsilon,
        family='f0*(1+k/sqrt(n)), k=t*psi/c²+a*(tanh²-E0 tanh²)',
        common_risk='g=.5*tanh(z)*f0; global safe interval [-boundary,boundary]',
        predictor='Phi(sqrt(n)*mean(psi)); no direction, root or generating t input',
        diagnostics='160-node Gaussian quadrature and finite-difference slope grid; not proof certification')
    args.output.mkdir(parents=True)
    def save(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    save(args.output/'design.json', dict(protocol=protocol, cells=cells))
    result = dict(protocol=dict(protocol, complete=False), cells=[])
    save(args.output/'results.json', result)
    for index, cell in enumerate(cells):
        n, t, amplitude = cell['n'], cell['t'], cell['amplitude']
        reference_rng = np.random.default_rng(np.random.SeedSequence([args.seed, index, 0]))
        times = np.full(args.paths, -1, dtype=np.int32)
        sides = np.zeros(args.paths, dtype=np.int8)
        for start in range(0, args.paths, args.block):
            stop = min(args.paths, start+args.block)
            x = np.zeros(stop-start)
            rt, rs = times[start:stop], sides[start:stop]
            for step in range(1, args.steps+1):
                ids = np.flatnonzero(rt < 0)
                if not len(ids):
                    break
                sample = draw(reference_rng, (len(ids), n), n, t, amplitude, base)
                x[ids] += args.eta*velocity(x[ids, None], sample).mean(axis=1)
                if not np.isfinite(x).all():
                    raise ValueError('Nonfinite state')
                ids = ids[abs(x[ids]) > args.boundary]
                rt[ids], rs[ids] = step, np.sign(x[ids]).astype(np.int8)
        arrays = dict(risk_times=times, risk_sides=sides)
        budgets = sorted(set(max(1, round(c2*n/r**2)) for r in args.ratios))
        cal_rng = np.random.default_rng(np.random.SeedSequence([args.seed, index, 1]))
        settings = []
        for m in budgets:
            forecasts = np.empty(args.repeats)
            for j in range(args.repeats):
                sample = draw(cal_rng, (m,), n, t, amplitude, base)
                coordinate = np.sqrt(n)*np.mean(velocity(0, sample)/(lam*sd))
                forecasts[j] = ndtr(coordinate)
            arrays[f'm{m}'] = forecasts
            settings.append(dict(budget=m, r=float(np.sqrt(c2*n/m))))
        np.savez_compressed(args.output/(cell['key']+'.npz'), **arrays)
        result['cells'].append(dict(cell, right=int((sides > 0).sum()),
            censored=int((times < 0).sum()), settings=settings))
        save(args.output/'results.json', result)
        print(cell['key'], n, t, amplitude, flush=True)
    result['protocol']['complete'] = True
    save(args.output/'results.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batches', type=int, nargs='+', default=[128, 512])
    p.add_argument('--targets', type=float, nargs='+', default=[-1., 0., 1.])
    p.add_argument('--amplitudes', type=float, nargs='+', default=[0., -.5, .5])
    p.add_argument('--ratios', type=float, nargs='+', default=[.5, 1., 2.])
    p.add_argument('--envelope', type=float, default=2.)
    p.add_argument('--eta', type=float, default=.1)
    p.add_argument('--boundary', type=float, default=.25)
    p.add_argument('--paths', type=int, default=1024)
    p.add_argument('--repeats', type=int, default=256)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--block', type=int, default=32)
    p.add_argument('--seed', type=int, default=73192)
    run(p.parse_args())

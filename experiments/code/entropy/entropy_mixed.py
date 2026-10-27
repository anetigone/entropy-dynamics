"""Mixed translation/prior cancellation and prediction ablations. Run directly.

Edit constants here for a separate server run; existing output is never replaced.
No parameters are fitted to simulated exits. delta changes x0 and the frozen
risk comparator, not the centered entropy field.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import ndtr
from entropy_critical import Geometry, ETA, M, SIGMA, W, simulate_actual
from entropy_prediction import joint_prediction, observed_cdf, summarize

OUT = Path(__file__).resolve().parents[3]/'experiments/results/mixed_shift'
SEED = 20261027
BATCHES = [32, 128, 512]
KAPPAS = [0., 1., 2.]
# delta = beta/lambda*rho + offset*seed_sd/sqrt(n).
OFFSETS = [-2., -1., 0., 1., 2.]
PATHS = 1000
MAX_STEPS = 2500
EPSILON = .005


def ablations(g, rho, n, epsilon, x0):
    fg = g.finite_geometry(rho, n, epsilon, x0)
    lo, hi = fg['left_boundary'], fg['right_boundary']
    balanced_radius = g.endpoints(0., epsilon)[1]
    u = W*(g.nodes-fg['q'])
    from scipy.special import expit
    weights = np.r_[((1-rho)/2)*g.ww, ((1+rho)/2)*g.ww]
    score = -W*u*expit(u)*expit(-u)
    actual_variance = float(weights@(score*score)-(weights@score)**2)

    def linear(peak=True, curvature=True, noise=True, boundary=True):
        q = fg['q'] if peak else -g.beta/g.lam*rho
        A = fg['A'] if curvature else g.A
        sg2 = actual_variance if noise else g.sg2
        a, b = (lo, hi) if boundary else (-balanced_radius, balanced_radius)
        return dict(q=q, A=A, seed_mean=float(np.sqrt(n)*(x0-q)),
                    seed_sd=float(np.sqrt(ETA**2*sg2/(A*A-1))),
                    left_H=q-a, right_H=b-q)

    predictions = {'balanced_limit': linear(False, False, False, False),
                   'actual_linear': linear(), 'conjugate': fg,
                   'without_peak': linear(peak=False),
                   'without_curvature': linear(curvature=False),
                   'without_noise': linear(noise=False),
                   'without_boundaries': linear(boundary=False)}
    return predictions


def run(out=OUT, batches=BATCHES, kappas=KAPPAS, offsets=OFFSETS,
        paths=PATHS, max_steps=MAX_STEPS):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    g = Geometry()
    grid = np.arange(max_steps+1)
    rows, arrays = [], {'grid': grid}
    for ni, n in enumerate(batches):
        for ki, kappa in enumerate(kappas):
            rho = g.rho_scale*kappa/np.sqrt(n)
            for oi, offset in enumerate(offsets):
                delta = g.beta/g.lam*rho+offset*g.seed_sd/np.sqrt(n)
                x0 = -delta
                predictions = ablations(g, rho, n, EPSILON, x0)
                rng = np.random.default_rng(np.random.SeedSequence([SEED, ni, ki, oi]))
                tt, dd = simulate_actual(n, paths, rho, EPSILON, rng, g,
                                         max_steps=max_steps, x0=x0)
                key = f'case_{len(rows)}'
                el, er = observed_cdf(tt, dd, grid)
                arrays.update({key+'_times': tt, key+'_directions': dd,
                               key+'_observed_left': el, key+'_observed_right': er})
                row = dict(key=key, n=n, kappa=kappa, rho=float(rho),
                           delta=float(delta), offset=offset, x0=float(x0),
                           linear_displacement=float(x0+g.beta/g.lam*rho),
                           actual_displacement=float(x0-predictions['conjugate']['q']),
                           predictions={})
                for name, geometry in predictions.items():
                    try:
                        left, right = joint_prediction(grid, n, geometry)
                    except ValueError as exc:
                        row['predictions'][name] = {'unavailable': str(exc)}
                        continue
                    arrays[key+'_'+name+'_left'] = left
                    arrays[key+'_'+name+'_right'] = right
                    row['predictions'][name] = dict(
                        geometry=geometry,
                        eventual_right=float(ndtr(geometry['seed_mean']/geometry['seed_sd'])),
                        **summarize(tt, dd, grid, left, right))
                rows.append(row)
                print(f'n={n}, kappa={kappa}, offset={offset}, censored={sum(tt<0)}', flush=True)
    result = dict(protocol=dict(seed=SEED, batches=list(batches), kappas=list(kappas),
                  offsets=list(offsets), paths=paths, max_steps=max_steps,
                  epsilon=EPSILON, eta=ETA, m=M, sigma=SIGMA, w=W,
                  calibration='Exact distribution; no use of exit observations',
                  event='First risk > frozen risk + epsilon; -1 means right censoring',
                  ablation='Leave one correction out of actual_linear; conjugate adds only the coordinate change'),
                  cases=rows)
    out.mkdir(parents=True)
    (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(out/'samples.npz', **arrays)
    return result


if __name__ == '__main__':
    run()

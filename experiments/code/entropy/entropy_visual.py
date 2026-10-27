"""Independent frozen-feature prediction and controlled scale/LA probes.

The official training set supplies the source anchor AND a labeled oracle
calibration distribution under imposed shifts. The official test set supplies
the target empirical distribution: independent draws with replacement for SGD,
and exact class-weighted empirical risk. No new dataset split is made. This is
an offline mechanism experiment, not an unlabeled deployment estimator.

No observed exit or evaluation risk enters a prediction. Empirical geometry is
used directly, so this does not assume Gaussian visual marginals. Failure to
find an unstable peak or a safe interval is saved as an applicability failure.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit, ndtr
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
from entropy_prediction import joint_prediction, observed_cdf, summarize

BASE = Path(__file__).resolve().parents[3]/'experiments/results'
CACHE = BASE/'visual_features/features.npz'
OUT = BASE/'visual_shift'
# Preselected before inspecting target exits; positive, negative CIFAR-10 IDs.
PAIRS = [(3,5), (1,9), (4,7), (0,8)]
BATCHES = [32,128,512]
PRIORS = [.5,.4,.3]
OFFSETS = [-2.,-1.,0.,1.,2.]
MODES = ['threshold', 'joint_scale', 'la_threshold', 'la_joint_scale', 'oracle_la_threshold']
ETA = .1
W = 2.
EPSILON = .02
PATHS = 600
MAX_STEPS = 2000
SEED = 20261027
LA_ALPHA = .1
LA_TAU = 1.
LA_PRIOR_FLOOR = 1e-6


class Marginal:
    def __init__(self, positive, negative, prior):
        self.pos = np.sort(np.asarray(positive, dtype=float))
        self.neg = np.sort(np.asarray(negative, dtype=float))
        self.p = float(prior)
        if not 0 < self.p < 1 or min(len(self.pos),len(self.neg)) < 2:
            raise ValueError('Two nonempty classes and an interior prior are required')
        if not np.isfinite(self.pos).all() or not np.isfinite(self.neg).all():
            raise ValueError('Features must be finite')

    def risk(self, x):
        return (self.p*np.searchsorted(self.pos, x, side='left')/len(self.pos)
                +(1-self.p)*(1-np.searchsorted(self.neg, x, side='left')/len(self.neg)))

    def moments(self, x, w=W):
        result = []
        for z in [self.pos, self.neg]:
            u = w*(z-x)
            a = expit(u)*expit(-u)
            v = -w*u*a
            result.append((np.mean(v), np.mean(w*w*a*(1-u*np.tanh(u/2))), np.mean(v*v)))
        mean, derivative, second = self.p*np.array(result[0])+(1-self.p)*np.array(result[1])
        return float(mean), float(derivative), float(max(0., second-mean*mean))

    def safe_component(self, x0, epsilon):
        # Enumerate every empirical risk plateau; do not smooth label risk.
        knots = np.unique(np.r_[self.pos, self.neg, x0])
        probes = np.r_[np.nextafter(knots[0], -np.inf), np.nextafter(knots, np.inf)]
        safe = self.risk(probes) <= self.risk(x0)+epsilon
        # x0 itself belongs to the plateau immediately before an equal knot.
        index = int(np.searchsorted(knots, x0, side='left'))
        left = index
        while left >= 0 and safe[left]:
            left -= 1
        right = index
        while right < len(safe) and safe[right]:
            right += 1
        if left < 0 or right == len(safe):
            raise ValueError('Risk-safe component is unbounded on at least one side')
        lo, hi = float(knots[left]), float(knots[right-1])
        components = int(safe[0])+int(np.sum(safe[1:] & ~safe[:-1]))
        return lo, hi, components

    def calibrate(self):
        # Search independently of target outcomes; multiple peaks are not hidden.
        mesh = np.linspace(min(self.pos[0],self.neg[0])-2, max(self.pos[-1],self.neg[-1])+2, 513)
        v = np.array([self.moments(x)[0] for x in mesh])
        peaks = []
        for i in range(len(mesh)-1):
            if v[i] <= 0 < v[i+1]:
                q = brentq(lambda x:self.moments(x)[0], mesh[i], mesh[i+1])
                if self.moments(q)[1] > 0:
                    peaks.append(float(q))
        if len(peaks) != 1:
            raise ValueError(f'Expected one unstable peak, found {len(peaks)}')
        q = peaks[0]
        _, lam, sg2 = self.moments(q)
        A = 1+ETA*lam
        return dict(q=q, lambda_=lam, A=A, seed_sd=float(np.sqrt(ETA**2*sg2/(A*A-1))))

    def prediction(self, calibration, n, x0, epsilon):
        lo, hi, components = self.safe_component(x0, epsilon)
        q, A = calibration['q'], calibration['A']
        if not lo < q < hi:
            raise ValueError('Calibrated peak lies outside the frozen-risk safe component')
        # Monotone interpolation of the deterministic update, not of exit data.
        mesh = np.unique(np.r_[np.linspace(lo, hi, 4097), q])
        mapped = mesh+ETA*np.array([self.moments(x)[0] for x in mesh])
        if np.any(np.diff(mapped) <= 0):
            raise ValueError('Update is not monotone on the calibrated interval')
        inverse = PchipInterpolator(mapped, mesh, extrapolate=False)
        def H(x):
            count = 0
            while abs(x-q) > 1e-5:
                x = float(inverse(x))
                count += 1
                if not np.isfinite(x) or count > 10000:
                    raise ValueError('Inverse conjugacy did not approach the peak')
            return A**count*(x-q)
        linear = dict(**calibration, seed_mean=float(np.sqrt(n)*(x0-q)),
                      left_H=q-lo, right_H=hi-q)
        full = dict(**calibration, seed_mean=float(np.sqrt(n)*H(x0)),
                    left_H=-H(lo), right_H=H(hi))
        probes = np.linspace(lo, hi, 17)[1:-1]
        residual = max(abs(H(x+ETA*self.moments(x)[0])-A*H(x))
                       for x in probes if lo <= x+ETA*self.moments(x)[0] <= hi)
        return dict(actual_linear=linear, conjugate=full), dict(
            left_boundary=lo, right_boundary=hi, safe_components=components,
            conjugacy_residual=float(residual), inverse_grid_points=len(mesh))


def project_pair(train, train_y, test, test_y, pair):
    a, b = pair
    pos, neg = train[train_y == a], train[train_y == b]
    if min(len(pos),len(neg)) < 2:
        raise ValueError(f'Missing source classes: {pair}')
    direction = pos.mean(axis=0)-neg.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm <= 0:
        raise ValueError('Degenerate source direction')
    direction /= norm
    center = (pos.mean(axis=0)+neg.mean(axis=0))/2
    pp, nn = (pos-center)@direction, (neg-center)@direction
    scale = np.sqrt((np.var(pp)+np.var(nn))/2)
    if scale <= 0:
        raise ValueError('Degenerate source scale')
    pp, nn = pp/scale, nn/scale
    evp = (test[test_y == a]-center)@direction/scale
    evn = (test[test_y == b]-center)@direction/scale
    anchor = dict(pair=list(pair), direction=direction.tolist(), center=center.tolist(),
                  projection_scale=float(scale), w=W,
                  m_over_sigma=float((np.mean(pp)-np.mean(nn))/2),
                  positive_std=float(np.std(pp)), negative_std=float(np.std(nn)),
                  orientation='First class positive; source midpoint threshold zero')
    return pp, nn, evp, evn, anchor


def simulate_visual(target, n, paths, delta, mode, rng, max_steps=MAX_STEPS):
    """True sampled entropy gradients; labels only select the controlled mixture."""
    if mode not in ['threshold','joint_scale','la_threshold','la_joint_scale',
                    'oracle_la_threshold','oracle_la_joint_scale']:
        raise ValueError(f'Unknown adaptation mode: {mode}')
    x = np.full(paths, -delta)
    logw = np.full(paths, np.log(W))
    oracle = mode.startswith('oracle_la_')
    use_la = oracle or mode.startswith('la_')
    prior = np.full(paths, target.p if oracle else .5)
    # LA adjusts only the entropy objective. Evaluation uses raw logits. The
    # estimated baseline uses square-root class counts and EMA (RELATED_WORK
    # section 2.2); the oracle version separately uses the imposed true prior.
    baseline = float(target.risk(-delta))
    times = np.full(paths, -1, dtype=np.int32)
    directions = np.zeros(paths, dtype=np.int8)
    robust = times.copy()
    possible = times.copy()
    invalid = np.zeros(paths, dtype=bool)
    peak_logw = logw.copy()
    # Simultaneous classwise DKW bound, uniform over thresholds; delta-risk
    # uncertainty is at most twice this risk bound. No claim for full networks.
    band = (target.p*np.sqrt(np.log(80)/(2*len(target.pos)))
            +(1-target.p)*np.sqrt(np.log(80)/(2*len(target.neg))))
    checkpoint_steps = np.unique(np.r_[0, np.arange(1,max_steps+1, max(1,max_steps//100)),max_steps])
    traces = []
    for t in range(max_steps+1):
        with np.errstate(over='ignore', invalid='ignore'):
            w = np.exp(logw)
            threshold = x
        valid = np.isfinite(w) & np.isfinite(threshold)
        invalid |= ~valid
        risks = np.full(paths, np.nan)
        risks[valid] = target.risk(threshold[valid])
        excess = risks-baseline
        if t > 0:
            out = (times < 0) & valid & (excess > EPSILON)
            times[out] = t
            directions[out] = np.where(threshold[out] > -delta, 1, -1)
            robust[(robust < 0) & valid & (excess > EPSILON+2*band)] = t
            possible[(possible < 0) & valid & (excess > EPSILON-2*band)] = t
        if t in checkpoint_steps:
            traces.append([t, float(np.mean(risks[valid])) if valid.any() else None,
                           float(np.median(w[valid])) if valid.any() else None,
                           float(np.mean(prior[valid])) if valid.any() else None])
        if t == max_steps:
            break
        ids = np.flatnonzero(valid)
        zpos = target.pos[rng.integers(len(target.pos),size=(len(ids),n))]
        zneg = target.neg[rng.integers(len(target.neg),size=(len(ids),n))]
        z = np.where(rng.random((len(ids),n)) < target.p, zpos, zneg)
        if use_la and not oracle:
            count = np.sum(z >= x[ids,None],axis=1)
            batch_prior = np.sqrt(count)/(np.sqrt(count)+np.sqrt(n-count))
            prior[ids] = np.clip(LA_ALPHA*prior[ids]+(1-LA_ALPHA)*batch_prior,
                                 LA_PRIOR_FLOOR,1-LA_PRIOR_FLOOR)
        logodds = (LA_TAU*np.log(prior[ids]/(1-prior[ids])))[:,None] if use_la else 0.
        u = w[ids,None]*(z-x[ids,None])+logodds
        a = expit(u)*expit(-u)
        x[ids] -= ETA*np.mean(w[ids,None]*u*a,axis=1)
        if mode.endswith('joint_scale'):
            # d entropy / d log(w) = -(u-logodds)*u*a for adjusted logits.
            logw[ids] += ETA*np.mean((u-logodds)*u*a,axis=1)
        peak_logw[ids] = np.maximum(peak_logw[ids], logw[ids])
    return dict(times=times, directions=directions, robust_times=robust,
                possible_times=possible, invalid=invalid, max_logw=peak_logw,
                final_prior=prior,
                trace=np.array(traces,dtype=float)), dict(
                    frozen_risk=baseline, risk_band_95=float(band),
                    epsilon_below_risk_uncertainty=bool(EPSILON <= 2*band),
                    classwise_risk_jumps=[target.p/len(target.pos),(1-target.p)/len(target.neg)],
                    invalid_paths=int(invalid.sum()), censored=int(np.sum(times<0)),
                    initial_inference_risk=baseline)


def run(cache=CACHE, out=OUT, pairs=PAIRS, batches=BATCHES, priors=PRIORS,
        offsets=OFFSETS, modes=MODES, paths=PATHS, max_steps=MAX_STEPS):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    with np.load(cache, allow_pickle=False) as saved:
        train, ty = saved['train_features'], saved['train_labels']
        test, vy = saved['test_features'], saved['test_labels']
    if (train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]
            or len(train) != len(ty) or len(test) != len(vy)
            or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError('Expected finite aligned official train/test feature arrays')
    grid = np.arange(max_steps+1)
    arrays, rows, anchors, failures = {'grid':grid}, [], [], []
    for pi, pair in enumerate(pairs):
        pp, nn, ep, en, anchor = project_pair(train, ty, test, vy, pair)
        anchors.append(anchor)
        for ri, p in enumerate(priors):
            calibration, target = Marginal(pp,nn,p), Marginal(ep,en,p)
            calibration_error = None
            try:
                cg = calibration.calibrate()
            except ValueError as exc:
                failures.append(dict(pair=list(pair),prior=p,reason=str(exc)))
                # Keep actual trajectories of every preselected pair even when
                # unstable-peak theory fails. Use a declared source-unit grid.
                cg, calibration_error = None, str(exc)
            for ni, n in enumerate(batches):
                for oi, offset in enumerate(offsets):
                    delta = (-cg['q']+offset*cg['seed_sd']/np.sqrt(n)
                             if cg is not None else offset/np.sqrt(n))
                    x0 = -delta
                    try:
                        if cg is None:
                            raise ValueError(calibration_error)
                        predictions, diagnostic = calibration.prediction(cg,n,x0,EPSILON)
                    except ValueError as exc:
                        predictions, diagnostic = {}, {'unavailable':str(exc)}
                    for mode in modes:
                        # Common random batches across adaptation modes.
                        rng = np.random.default_rng(np.random.SeedSequence([SEED,pi,ri,ni,oi]))
                        sample, summary = simulate_visual(target,n,paths,delta,mode,rng,max_steps)
                        key = f'case_{len(rows)}'
                        for name, values in sample.items():
                            arrays[key+'_'+name] = values
                        el, er = observed_cdf(sample['times'],sample['directions'],grid)
                        arrays[key+'_observed_left'], arrays[key+'_observed_right'] = el, er
                        row = dict(key=key,pair=list(pair),prior=p,n=n,offset=offset,
                                   delta=float(delta),mode=mode, calibration=cg,
                                   offset_units='seed_sd' if cg is not None else 'source_within_class_sd',
                                   diagnostic=diagnostic, evaluation=summary,predictions={})
                        # Fixed-slope theory is compared to scale/LA probes as a
                        # transfer diagnostic; no claim that its assumptions hold.
                        for name, geometry in predictions.items():
                            left, right = joint_prediction(grid,n,geometry)
                            arrays[key+'_'+name+'_left'] = left
                            arrays[key+'_'+name+'_right'] = right
                            row['predictions'][name] = dict(geometry=geometry,
                                eventual_right=float(ndtr(geometry['seed_mean']/geometry['seed_sd'])),
                                **summarize(sample['times'],sample['directions'],grid,left,right))
                        rows.append(row)
                        print(f'pair={pair}, p={p}, n={n}, offset={offset}, mode={mode}',flush=True)
    protocol = dict(cache=str(cache),pairs=[list(p) for p in pairs],batches=list(batches),
        priors=list(priors),offsets=list(offsets),modes=list(modes),paths=paths,
        max_steps=max_steps,seed=SEED,eta=ETA,w=W,epsilon=EPSILON,
        calibration='Official train labels, source projection and shifted empirical geometry; oracle offline calibration',
        evaluation='Official test weighted empirical risk; fresh iid mixture draws with replacement from that finite pool',
        translation='Input score z+delta, original frozen threshold 0; centered initialization x0=-delta',
        translation_grid='delta=-q+offset*seed_sd/sqrt(n) when calibration has one unstable peak; otherwise delta=offset/sqrt(n) in source-standardized units, with prediction unavailable',
        la='Add tau*log prior to logits for entropy only; raw logits for evaluation. Estimated prior uses sqrt raw-argmax counts and EMA; oracle_la uses the imposed true prior.',
        la_alpha=LA_ALPHA, la_tau=LA_TAU, la_prior_floor=LA_PRIOR_FLOOR,
        transfer='Fixed-slope prediction applied to other modes as an assumption stress test',
        censoring='times=-1 retained in CDF denominator; invalid paths separately reported',
        risk_uncertainty='Classwise DKW bands require iid class samples; empirical risk itself is evaluated exactly',
        event='First t>=1 risk > frozen risk + epsilon; all modes start from the same frozen model',
        conjugacy='Numerical inverse interpolation; residual diagnostic, no exact numerical guarantee')
    result = dict(protocol=protocol,anchors=anchors,applicability_failures=failures,cases=rows)
    out.mkdir(parents=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'samples.npz', **arrays)
    return result


if __name__ == '__main__':
    run()

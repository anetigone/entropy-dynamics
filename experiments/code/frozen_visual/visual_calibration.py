"""Independent unlabeled calibration draws, scored against saved SGD paths.

Sampling is iid with replacement from the original controlled target pool.
Labels specify that experimental generator only; the estimator receives scalar
scores and their multiplicities. No official split, new SGD or time fitting.
Oracle risk endpoints and target peaks are used only in labeled diagnostics.
"""
from pathlib import Path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.special import expit, ndtr
from entropy_visual import CACHE, BASE, OUT as RESULTS, ETA, W, project_pair, class_pair
from field_geometry import calibrate_peak, conjugate_prediction
from entropy_prediction import joint_prediction, summarize

OUT = BASE/'visual_calibration'
BUDGETS = [2048,16384,131072,1048576]
SEED = 20261028


class UnlabeledField:
    def __init__(self, scores, counts):
        scores, counts = np.asarray(scores,dtype=float),np.asarray(counts)
        if len(scores) != len(counts) or np.any(counts < 0) or counts.sum() <= 0:
            raise ValueError('Calibration requires aligned scores and positive sample counts')
        keep = counts > 0
        self.nodes, inverse = np.unique(scores[keep],return_inverse=True)
        self.weights = np.bincount(inverse,weights=counts[keep])/counts.sum()
        if not np.isfinite(self.nodes).all():
            raise ValueError('Nonfinite calibration scores')

    def moments(self, x):
        u = W*(self.nodes-x)
        a = expit(u)*expit(-u)
        gradient = -W*u*a
        mean = self.weights@gradient
        derivative = self.weights@(W*W*a*(1-u*np.tanh(u/2)))
        variance = self.weights@(gradient*gradient)-mean*mean
        return float(mean),float(derivative),float(max(variance,0.))

    def velocities(self, xs):
        xs = np.asarray(xs, dtype=float)
        result = np.empty(xs.shape)
        for start in range(0, len(xs), 64):
            u = W*(self.nodes[None, :]-xs[start:start+64, None])
            result[start:start+64] = np.sum((-W*u*expit(u)*expit(-u))*self.weights, axis=1)
        return result

    def calibrate(self):
        return calibrate_peak(self,self.nodes[0],self.nodes[-1],ETA)


def _calibrate_repeat(task):
    pool, probabilities, pair, prior, repeat, budgets, seed, rows, true, oracle, data = task
    calibration_rows, fits, averages = [], [], {}
    grid = data["grid"]
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed,pair[0],pair[1],int(round(prior*100000)),repeat]))
    counts = np.zeros(len(pool),dtype=np.int64)
    previous = 0
    for budget in budgets:
        # Multinomial counts exactly represent an iid unlabeled
        # calibration prefix, retaining repeated samples as weights.
        counts += rng.multinomial(budget-previous,probabilities)
        previous = budget
        field = UnlabeledField(pool,counts)
        cal_key = f'calibration_{len(calibration_rows)}'
        cr = dict(key=cal_key,pair=pair,prior=prior,budget=budget,repeat=repeat)
        calibration_rows.append(cr)
        try:
            cg = field.calibrate()
        except ValueError as exc:
            cr['unavailable'] = str(exc)
            continue
        variance = field.moments(cg['q'])[2]
        se_q = np.sqrt(variance/budget)/cg['lambda_']
        lo_q,hi_q = cg['q']-1.959963984540054*se_q,cg['q']+1.959963984540054*se_q
        cr.update(geometry=cg,root_se=float(se_q),root_interval=[float(lo_q),float(hi_q)],
                  target_q=true['q'],root_error=float(cg['q']-true['q']),
                  target_q_in_interval=bool(lo_q <= true['q'] <= hi_q))
        geometry_cache = {}
        for row in rows:
            n,x0,key = row['n'],-row['delta'],row['key']
            old = next(r for r in oracle['cases'] if r['key']==key)
            fit = dict(calibration=cal_key,case=key,budget=budget,repeat=repeat,
                       theoretical_ratio=float(np.sqrt(n/budget*(2+ETA*true['lambda_'])/(ETA*true['lambda_']))),
                       target_standardized_initialization=float(np.sqrt(n)*(x0-true['q'])/true['seed_sd']),
                       target_local_direction=float(ndtr(np.sqrt(n)*(x0-true['q'])/true['seed_sd'])),
                       root_noise_ratio=float(np.sqrt(n)*se_q/cg['seed_sd']),
                       root_standardized_error=float(np.sqrt(n)*(cg['q']-true['q'])/true['seed_sd']),
                       local_direction=float(ndtr(np.sqrt(n)*(x0-cg['q'])/cg['seed_sd'])),
                       local_direction_interval=[float(ndtr(np.sqrt(n)*(x0-hi_q)/cg['seed_sd'])),
                                                 float(ndtr(np.sqrt(n)*(x0-lo_q)/cg['seed_sd']))],
                       predictions={},unavailable={})
            fits.append(fit)
            endpoints = {}
            if 'left_boundary' in row['diagnostic']:
                endpoints['source_risk'] = (row['diagnostic']['left_boundary'],row['diagnostic']['right_boundary'])
            else:
                fit['unavailable']['source_risk'] = row['diagnostic'].get('unavailable','No source interval')
            if 'target_risk' in old:
                endpoints['oracle_risk'] = (old['target_risk']['left_boundary'],old['target_risk']['right_boundary'])
            else:
                fit['unavailable']['oracle_risk'] = old['unavailable'].get('target_risk','No target interval')
            for name,(lo,hi) in endpoints.items():
                cache_key = (x0,lo,hi)
                if cache_key not in geometry_cache:
                    try:
                        models,_ = conjugate_prediction(field,cg,1,x0,lo,hi,ETA)
                        geometry_cache[cache_key] = models['conjugate']
                    except (ValueError,OverflowError) as exc:
                        geometry_cache[cache_key] = str(exc)
                base_geometry = geometry_cache[cache_key]
                if isinstance(base_geometry,str):
                    fit['unavailable'][name] = base_geometry
                    continue
                geom = dict(base_geometry,seed_mean=float(np.sqrt(n)*base_geometry['seed_mean']))
                left,right = joint_prediction(grid,n,geom)
                score = summarize(data[key+'_times'],data[key+'_directions'],grid,left,right)
                score['geometry'] = geom
                score['eventual_right'] = float(ndtr(geom['seed_mean']/geom['seed_sd']))
                bounds = score['right_probability_bounds']
                score['direction_bound_distance'] = max(bounds[0]-score['eventual_right'],
                                                        score['eventual_right']-bounds[1],0.)
                fit['predictions'][name] = score
                avg_key = f'{key}_m{budget}_{name}'
                if avg_key not in averages:
                    averages[avg_key] = dict(key=avg_key,case=key,budget=budget,model=name,
                        count=0,cdf_errors=[],joint_errors=[],direction_errors=[],forecasts=[],
                        left=np.zeros_like(left),right=np.zeros_like(right))
                avg = averages[avg_key]
                avg['count'] += 1
                avg['cdf_errors'].append(score['cdf_error'])
                avg['joint_errors'].append(score['joint_cdf_error'])
                avg['direction_errors'].append(score['direction_bound_distance'])
                avg['forecasts'].append(score['eventual_right'])
                avg['left'] += left
                avg['right'] += right
    return calibration_rows, fits, averages


def _repeat_name(task):
    return f'pair_{task[2][0]}_{task[2][1]}_prior_{task[3]:.8g}_repeat_{task[4]}'


def _save_repeat(folder, task, value):
    crs, fs, averages = value
    arrays, meta = {}, {}
    for key, avg in averages.items():
        meta[key] = {k: v for k, v in avg.items() if k not in ['left', 'right']}
        for direction in ['left', 'right']:
            arrays[key+'_'+direction] = avg[direction]
    name = _repeat_name(task)
    temporary = folder/(name+'.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(folder/(name+'.npz'))
    temporary = folder/(name+'.tmp.json')
    temporary.write_text(json.dumps(dict(calibrations=crs, fits=fs, averages=meta), allow_nan=False), encoding='utf-8')
    temporary.replace(folder/(name+'.json'))


def _load_repeat(folder, task):
    name = _repeat_name(task)
    saved = json.loads((folder/(name+'.json')).read_text(encoding='utf-8'))
    with np.load(folder/(name+'.npz'), allow_pickle=False) as arrays:
        for key, avg in saved['averages'].items():
            for direction in ['left', 'right']:
                avg[direction] = arrays[key+'_'+direction]
    return saved['calibrations'], saved['fits'], saved['averages']


def run(features=CACHE, results=RESULTS/'results.json', samples=RESULTS/'samples.npz',
        diagnosis=RESULTS/'diagnosis/results.json', out=OUT, budgets=BUDGETS,
        repeats=20, pairs=None, priors=(.5,), batches=(32,128,512), offsets=(0.,),seed=SEED, workers=1, resume=False):
    out = Path(out)
    if out.exists() and not resume:
        raise FileExistsError(f'Choose a new output directory: {out}')
    budgets = sorted(set(budgets))
    if not budgets or budgets[0] <= 0 or repeats <= 0 or workers <= 0:
        raise ValueError('Calibration budgets and repetitions must be positive')
    original = json.loads(Path(results).read_text(encoding='utf-8'))
    oracle = json.loads(Path(diagnosis).read_text(encoding='utf-8'))
    if original['protocol']['modes'] != ['threshold']:
        raise ValueError('Use fixed-slope threshold trajectories')
    if original['protocol']['eta'] != ETA or original['protocol']['w'] != W:
        raise ValueError('Saved update parameters must match the field calculation')
    selected = [r for r in original['cases'] if r['prior'] in priors and r['n'] in batches
                and r['offset'] in offsets and (pairs is None or tuple(r['pair']) in {tuple(p) for p in pairs})]
    if not selected:
        raise ValueError('No saved cases match the requested parameters')
    with np.load(features,allow_pickle=False) as saved:
        train,ty,test,vy = [saved[k] for k in
            ['train_features','train_labels','test_features','test_labels']]
    with np.load(samples,allow_pickle=False) as saved:
        data = dict(saved)
    grid = data['grid']
    curves = {'grid':grid}
    cases, calibration_rows, fits, averages = [],[],[],{}
    for row in selected:
        old = next(r for r in oracle['cases'] if r['key']==row['key'])
        cases.append(dict(key=row['key'],pair=row['pair'],prior=row['prior'],n=row['n'],
                          offset=row['offset'],x0=-row['delta'],evaluation=row['evaluation'],
                          baseline={k:old['predictions'][k] for k in
                                    ['source_conjugate','target_conjugate'] if k in old['predictions']}))
        for direction in ['left','right']:
            curves[row['key']+'_observed_'+direction] = data[row['key']+'_observed_'+direction]
    tasks = []
    for pair in original['protocol']['pairs']:
        pair_rows = [r for r in selected if r['pair']==pair]
        if not pair_rows:
            continue
        _,_,positive,negative,_ = project_pair(train,ty,test,vy,pair)
        pool = np.r_[positive,negative]
        for prior in sorted({r['prior'] for r in pair_rows}):
            rows = [r for r in pair_rows if r['prior']==prior]
            probabilities = np.r_[np.full(len(positive),prior/len(positive)),
                                   np.full(len(negative),(1-prior)/len(negative))]
            true = next(g['target'] for g in oracle['geometry'] if g['pair']==pair and g['prior']==prior)
            for repeat in range(repeats):
                keys = {r['key'] for r in rows}
                task_data = {'grid': grid}
                for key in keys:
                    for suffix in ['_times', '_directions']:
                        task_data[key+suffix] = data[key+suffix]
                task_oracle = {'cases': [r for r in oracle['cases'] if r['key'] in keys]}
                tasks.append((pool, probabilities, pair, prior, repeat, budgets, seed,
                              rows, true, task_oracle, task_data))
    settings = dict(features=str(Path(features).resolve()), results=str(Path(results).resolve()),
        samples=str(Path(samples).resolve()), diagnosis=str(Path(diagnosis).resolve()),
        budgets=budgets, repeats=repeats, seed=seed, cases=[r['key'] for r in selected])
    progress = out/'repeats'
    settings_file = out/'settings.json'
    if resume:
        if not settings_file.exists():
            raise ValueError('No resumable settings found; old versions have no intermediate checkpoints')
        if json.loads(settings_file.read_text(encoding='utf-8')) != settings:
            raise ValueError('Resume requires the same inputs, selected cases, budgets, repeats and seed; workers may change')
    else:
        progress.mkdir(parents=True)
        settings_file.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    pending = [task for task in tasks if not (progress/(_repeat_name(task)+'.json')).exists()]
    print(f'{len(tasks)-len(pending)}/{len(tasks)} repetitions saved; {len(pending)} pending', flush=True)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 and pending else None
    try:
        completed = executor.map(_calibrate_repeat, pending) if executor else map(_calibrate_repeat, pending)
        for task, value in zip(pending, completed):
            _save_repeat(progress, task, value)
            print(f'saved pair={task[2]}, prior={task[3]}, repeat={task[4]+1}/{repeats}', flush=True)
    finally:
        if executor:
            executor.shutdown(cancel_futures=True)
    # Reload in deterministic design order, so resume and process scheduling
    # cannot change calibration keys or summation order.
    for task in tasks:
            crs, fs, avgs = _load_repeat(progress, task)
            names = {cr['key']: f'calibration_{len(calibration_rows)+i}' for i, cr in enumerate(crs)}
            for cr in crs:
                cr['key'] = names[cr['key']]
            for fit in fs:
                fit['calibration'] = names[fit['calibration']]
            calibration_rows.extend(crs)
            fits.extend(fs)
            for key, avg in avgs.items():
                if key not in averages:
                    averages[key] = avg
                else:
                    dst = averages[key]
                    for field in ['count', 'left', 'right']:
                        dst[field] += avg[field]
                    for field in ['cdf_errors','joint_errors','direction_errors','forecasts']:
                        dst[field].extend(avg[field])
            print(f'calibrated pair={task[2]}, prior={task[3]}, repeat={task[4]+1}/{repeats}', flush=True)
    aggregate = []
    for avg in averages.values():
        forecasts = avg.pop('forecasts')
        avg['direction_forecast_quantiles'] = np.quantile(forecasts,[.1,.5,.9]).tolist()
        avg['direction_forecast_std'] = float(np.std(forecasts))
        for direction in ['left','right']:
            curves[avg['key']+'_'+direction] = avg.pop(direction)/avg['count']
        avg.update(availability=avg['count']/repeats,
                   mean_cdf_error=float(np.mean(avg.pop('cdf_errors'))),
                   mean_joint_error=float(np.mean(avg.pop('joint_errors'))),
                   mean_direction_bound_distance=float(np.mean(avg.pop('direction_errors'))))
        aggregate.append(avg)
    result = dict(protocol=dict(features=str(features),results=str(results),samples=str(samples),
        diagnosis=str(diagnosis),budgets=budgets,repeats=repeats,seed=seed,workers=workers,
        pairs=pairs,priors=list(priors),batches=list(batches),offsets=list(offsets),
        calibration='Independent iid with-replacement prefixes from the fixed target empirical mixture; estimator receives scores and counts only',
        independence='New calibration RNG, independent of saved adaptation RNG conditional on the fixed feature pool; no new official split',
        initialization='Original saved initialization; it is never recentered using the new calibration',
        source_risk='Training-label risk endpoints retained: target labels not used for field fitting or these endpoints',
        oracle_risk='Target-label endpoints retained solely to isolate field calibration: not an unlabeled risk predictor',
        uncertainty='Root Wald interval is asymptotic, estimated curvature/noise are plug-in; direction interval varies root only',
        variance='Do not inflate actual SGD seed variance: calibration affects the predictor, not the saved initialization',
        aggregation='Scores average each successful calibration separately; curves average successful calibrations only, availability reported; all reuse the same saved SGD paths',
        no_new_sgd=True,no_fit_to_exits=True),cases=cases,calibrations=calibration_rows,
        fits=fits,aggregate=aggregate)
    out.mkdir(parents=True, exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'curves.npz',**curves)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,default=CACHE)
    parser.add_argument('--results',type=Path,default=RESULTS/'results.json')
    parser.add_argument('--samples',type=Path,default=RESULTS/'samples.npz')
    parser.add_argument('--diagnosis',type=Path,default=RESULTS/'diagnosis/results.json')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--budgets',type=int,nargs='+',default=BUDGETS)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--pairs',type=class_pair,nargs='+')
    parser.add_argument('--priors',type=float,nargs='+',default=[.5])
    parser.add_argument('--batches',type=int,nargs='+',default=[32,128,512])
    parser.add_argument('--offsets',type=float,nargs='+',default=[0.])
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--workers',type=int,default=1,help='Parallel independent calibration repetitions')
    parser.add_argument('--resume',action='store_true',help='Reuse completed repetitions in the same output directory')
    args = parser.parse_args()
    run(args.features,args.results,args.samples,args.diagnosis,args.output,args.budgets,
        args.repeats,args.pairs,args.priors,args.batches,args.offsets,args.seed,args.workers,args.resume)

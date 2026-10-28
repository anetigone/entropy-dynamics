"""Re-evaluate saved exits using target empirical geometry; no new SGD runs.

Target geometry uses the same feature pool as adaptation and is an oracle
post-hoc diagnostic, never an independent prediction or a fit to exit curves.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.special import ndtr
from entropy_visual import CACHE, OUT as RESULTS, Marginal, project_pair
from entropy_prediction import joint_prediction, observed_cdf, summarize

OUT = RESULTS/'diagnosis'


def linear_geometry(cg, n, x0, lo, hi):
    return dict(**cg, seed_mean=float(np.sqrt(n)*(x0-cg['q'])),
                left_H=cg['q']-lo, right_H=hi-cg['q'])


def empirical_summary(times, grid):
    cdf = np.searchsorted(np.sort(times[times >= 0]), grid, side='right')/len(times)
    quantiles = []
    for q in [.1,.5,.9]:
        ids = np.flatnonzero(cdf >= q)
        quantiles.append(int(grid[ids[0]]) if len(ids) else None)
    return dict(censored=int(np.sum(times < 0)), quantiles=quantiles,
                exit_fraction=float(cdf[-1]))


def run(features=CACHE, results=RESULTS/'results.json', samples=RESULTS/'samples.npz', out=OUT):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Choose a new diagnostic output directory: {out}')
    original = json.loads(Path(results).read_text(encoding='utf-8'))
    if original['protocol']['modes'] != ['threshold']:
        raise ValueError('This diagnosis compares fixed-slope threshold trajectories only')
    with np.load(features, allow_pickle=False) as cache:
        train, ty, test, vy = [cache[k] for k in
            ['train_features','train_labels','test_features','test_labels']]
    with np.load(samples, allow_pickle=False) as saved:
        data = dict(saved)
    grid = data['grid']
    geometry, rows, arrays = [], [], {'grid':grid}
    reconstruction_error = 0.
    epsilon = original['protocol']['epsilon']
    # All cases share a geometry for a given pair/prior. Geometry is computed
    # from features only; saved exit outcomes are used exclusively for scoring.
    for pair in original['protocol']['pairs']:
        pp, nn, ep, en, anchor = project_pair(train, ty, test, vy, pair)
        source_anchor = next(a for a in original['anchors'] if a['pair']==pair)
        anchor_error = max(float(np.max(abs(np.asarray(source_anchor[k])-np.asarray(anchor[k]))))
                           for k in ['direction','center','projection_scale'])
        for prior in original['protocol']['priors']:
            source, target = Marginal(pp,nn,prior), Marginal(ep,en,prior)
            source_rows = [r for r in original['cases'] if r['pair']==pair and r['prior']==prior]
            cg = source_rows[0]['calibration']
            tg = target.calibrate()
            mesh = np.linspace(min(pp.min(),nn.min(),ep.min(),en.min()),
                               max(pp.max(),nn.max(),ep.max(),en.max()),1025)
            gkey = f'geometry_{len(geometry)}'
            arrays[gkey+'_x'] = mesh
            arrays[gkey+'_source_drift'] = np.array([source.moments(x)[0] for x in mesh])
            arrays[gkey+'_target_drift'] = np.array([target.moments(x)[0] for x in mesh])
            gr = dict(key=gkey,pair=pair,prior=prior,source=cg,target=tg,
                      peak_difference=float(cg['q']-tg['q']),
                      target_drift_at_source_peak=target.moments(cg['q'])[0],
                      source_drift_at_saved_peak=source.moments(cg['q'])[0],
                      anchor_max_difference=anchor_error)
            geometry.append(gr)
            print(f'pair={pair}, prior={prior}, q_source={cg["q"]:.6f}, q_target={tg["q"]:.6f}',flush=True)
            for row in source_rows:
                key, n, x0 = row['key'], row['n'], -row['delta']
                tt, dd = data[key+'_times'], data[key+'_directions']
                el, er = observed_cdf(tt,dd,grid)
                reconstruction_error = max(reconstruction_error,
                    float(np.max(abs(el-data[key+'_observed_left']))),
                    float(np.max(abs(er-data[key+'_observed_right']))))
                arrays[key+'_observed_left'], arrays[key+'_observed_right'] = el, er
                rr = dict(key=key,pair=pair,prior=prior,n=n,offset=row['offset'],x0=x0,
                          geometry_key=gkey,source_evaluation=row['evaluation'],
                          observed=empirical_summary(tt,grid),
                          robust=empirical_summary(data[key+'_robust_times'],grid),
                          possible=empirical_summary(data[key+'_possible_times'],grid),
                          critical_peak_bias=float(np.sqrt(n)*(cg['q']-tg['q'])/tg['seed_sd']),
                          target_local_escape_right=float(ndtr(np.sqrt(n)*(x0-tg['q'])/tg['seed_sd'])),
                          predictions={},unavailable={})
                models = {}
                for name, p in row['predictions'].items():
                    if 'geometry' in p:
                        models['source_'+name] = p['geometry']
                if 'left_boundary' in row['diagnostic']:
                    lo, hi = row['diagnostic']['left_boundary'],row['diagnostic']['right_boundary']
                    peak = dict(cg,q=tg['q'])
                    models['target_peak_only'] = linear_geometry(peak,n,x0,lo,hi)
                    models['target_field_only'] = linear_geometry(tg,n,x0,lo,hi)
                    try:
                        field_predictions, _ = target.prediction(tg,n,x0,epsilon,boundaries=(lo,hi))
                        models['target_field_conjugate'] = field_predictions['conjugate']
                    except ValueError as exc:
                        rr['unavailable']['target_field_conjugate'] = str(exc)
                try:
                    lo, hi, components = target.safe_component(x0,epsilon)
                    rr['target_risk'] = dict(left_boundary=lo,right_boundary=hi,
                                             safe_components=components,
                                             frozen_risk=float(target.risk(x0)))
                    models['target_boundary_only'] = linear_geometry(cg,n,x0,lo,hi)
                    models['target_linear'] = linear_geometry(tg,n,x0,lo,hi)
                except ValueError as exc:
                    rr['unavailable']['target_risk'] = str(exc)
                try:
                    predictions, diagnostic = target.prediction(tg,n,x0,epsilon)
                    models['target_conjugate'] = predictions['conjugate']
                    rr['target_diagnostic'] = diagnostic
                except ValueError as exc:
                    rr['unavailable']['target_conjugate'] = str(exc)
                for name, geom in models.items():
                    try:
                        left, right = joint_prediction(grid,n,geom)
                    except ValueError as exc:
                        rr['unavailable'][name] = str(exc)
                        continue
                    score = summarize(tt,dd,grid,left,right)
                    score['geometry'] = geom
                    score['eventual_right'] = float(ndtr(geom['seed_mean']/geom['seed_sd']))
                    bounds = score['right_probability_bounds']
                    score['direction_bound_distance'] = max(bounds[0]-score['eventual_right'],
                                                            score['eventual_right']-bounds[1],0.)
                    if name.startswith('source_'):
                        old = row['predictions'][name.removeprefix('source_')]
                        reconstruction_error = max(reconstruction_error,
                            abs(score['cdf_error']-old['cdf_error']),
                            abs(score['joint_cdf_error']-old['joint_cdf_error']))
                    rr['predictions'][name] = score
                    arrays[key+'_'+name+'_left'], arrays[key+'_'+name+'_right'] = left,right
                rows.append(rr)
    if reconstruction_error > 1e-10:
        raise ValueError(f'Saved summary does not match reconstructed samples: {reconstruction_error}')
    result = dict(protocol=dict(features=str(features),results=str(results),samples=str(samples),
                  interpretation='Target models use exact target empirical geometry: post-hoc oracle diagnostic, not independent prediction',
                  no_new_sgd=True, no_fit_to_exits=True,
                  original_protocol=original['protocol']),
                  reconstruction_max_difference=reconstruction_error,geometry=geometry,cases=rows)
    out.mkdir(parents=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'curves.npz',**arrays)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,default=CACHE)
    parser.add_argument('--results',type=Path,default=RESULTS/'results.json')
    parser.add_argument('--samples',type=Path,default=RESULTS/'samples.npz')
    parser.add_argument('--output',type=Path,default=OUT)
    args = parser.parse_args()
    run(args.features,args.results,args.samples,args.output)

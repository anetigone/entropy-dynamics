"""Separate mean motion from run variation in existing BN development states."""
import argparse
import json
from pathlib import Path

import numpy as np


def energy(a):
    return float(np.sum(np.square(a, dtype=np.float64)))


def axis(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1, a.shape[-1])
    gram = a @ a.T
    values, vectors = np.linalg.eigh(gram)
    v = vectors[:, -1] @ a
    v /= np.linalg.norm(v)
    return v, float(values[-1] / np.trace(gram))


def fraction(a, v):
    return energy(a @ v) / energy(a)


def analyze(path):
    with np.load(path) as d:
        states = d['states']  # Already displacements, not absolute parameters.
        risks = d['risks']
        environments = d['environments']
        global_axis = d['direction'].astype(np.float64)
    if states.ndim != 3 or risks.shape != states.shape[:2]:
        raise ValueError('Incompatible states and risks')
    if not np.isfinite(states).all() or not np.isfinite(risks).all():
        raise ValueError('Nonfinite states or risks')
    if not np.allclose(states[:, 0], 0):
        raise ValueError('Expected displacements from initialization')
    horizon = states.shape[1] - 1
    sample = np.arange(25, horizon + 1, 25)
    if len(sample) < 2:
        raise ValueError('Need at least 50 recorded steps')
    out = dict(shape=list(states.shape), sampled_steps=sample.tolist(), environments={})
    for name in np.unique(environments):
        a = states[environments == name]
        r = risks[environments == name].astype(np.float64)
        mean = a.mean(axis=0, dtype=np.float64)
        residual = a - mean
        total = energy(a)
        centered = energy(residual)
        uncentered_axis, uncentered_fraction = axis(a[:, sample])
        residual_axis, residual_fraction = axis(residual[:, sample])
        mean_axis, mean_fraction = axis(mean[sample])
        split_axes = []
        heldout = []
        for train, test in ((a[::2], a[1::2]), (a[1::2], a[::2])):
            tr = train - train.mean(axis=0, dtype=np.float64)
            te = test - test.mean(axis=0, dtype=np.float64)
            v, _ = axis(tr[:, sample])
            split_axes.append(v)
            heldout.append(fraction(te[:, sample], v))
        checkpoints = {}
        for t in (25, 50, 100, 200, 300, 400, 500):
            if t > horizon:
                continue
            e = energy(a[:, t])
            er = energy(residual[:, t])
            checkpoints[str(t)] = dict(
                mean_energy_fraction=len(a)*energy(mean[t])/e,
                residual_rms_norm=float(np.sqrt(er/len(a))),
                mean_norm=float(np.linalg.norm(mean[t])),
                global_axis_residual_fraction=fraction(residual[:, t], global_axis),
                risk_residual_sd_pp=float(100*r[:, t].std(ddof=1)))
        # Leave one entire run out. Center train and held-out risks at each step
        # using the TRAINING mean; regression cannot exploit time trend alone.
        prediction_error = baseline_error = 0.
        for i in range(len(a)):
            mask = np.arange(len(a)) != i
            x = a @ global_axis
            mx, mr = x[mask].mean(0), r[mask].mean(0)
            tx, tr = x[mask]-mx, r[mask]-mr
            beta = float(np.sum(tx*tr)/np.sum(tx*tx))
            pred = mr + beta*(x[i]-mx)
            prediction_error += energy(pred-r[i])
            baseline_error += energy(mr-r[i])
        increments = np.diff(a, axis=1)
        im = increments.mean(axis=0, dtype=np.float64)
        ir = increments-im
        out['environments'][str(name)] = dict(
            runs=len(a), mean_energy_fraction=len(a)*energy(mean)/total,
            mean_energy_fraction_bias_corrected=(len(a)*len(a)*energy(mean)/total-1)/(len(a)-1),
            residual_energy_fraction=centered/total,
            energy_identity_relative_error=abs(total-len(a)*energy(mean)-centered)/total,
            uncentered_first_fraction_sampled=uncentered_fraction,
            mean_first_fraction_sampled=mean_fraction,
            residual_first_fraction_sampled=residual_fraction,
            residual_axis_split_cosine=float(abs(split_axes[0]@split_axes[1])),
            residual_axis_heldout_fractions=heldout,
            residual_axis_mean_axis_cosine=float(abs(residual_axis@mean_axis)),
            uncentered_axis_residual_fraction=fraction(residual, uncentered_axis),
            global_axis_residual_fraction=fraction(residual, global_axis),
            global_axis_mean_fraction=fraction(mean, global_axis),
            increment_mean_energy_fraction=len(a)*energy(im)/energy(increments),
            increment_global_axis_residual_fraction=fraction(ir, global_axis),
            risk_residual_loro_r2=1-prediction_error/baseline_error,
            risk_residual_loro_rmse_pp=100*np.sqrt(prediction_error/r.size),
            risk_time_mean_loro_rmse_pp=100*np.sqrt(baseline_error/r.size),
            checkpoints=checkpoints)
    return out


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = json.dumps(analyze(args.input), indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(result+'\n', encoding='utf-8')
    else:
        print(result)


if __name__ == '__main__':
    main()

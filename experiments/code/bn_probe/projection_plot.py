"""Summarize independent projection diagnostics and exploratory Brier comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from bn_dynamics import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.result/'protocol.json').read_text(encoding='utf-8'))
    if not protocol['complete']:
        raise ValueError('Calculation is incomplete')
    environments = protocol['corruptions']
    fig, axes = plt.subplots(len(environments), 3, figsize=(12, 3.3*len(environments)), squeeze=False)
    summary = {}
    rng = np.random.default_rng(9311)
    for row, name in enumerate(environments):
        d = np.load(args.result/(name+'.npz'))
        forecasts, events = d['forecasts'], d['events'].astype(float)
        constant = float(d['development_constant'])
        losses = (forecasts[:, None]-events[None, :])**2
        gain = (constant-events)**2-losses
        # Resample independent calibration and future-run axes, never C*R fake iid pairs.
        boot = []
        for _ in range(2000):
            ci = rng.integers(len(forecasts), size=len(forecasts))
            ri = rng.integers(len(events), size=len(events))
            boot.append(gain[np.ix_(ci, ri)].mean())
        x, risk, residual = [d[k] for k in ['projections', 'risk_increases', 'residuals']]
        coef = d['risk_coef']
        mapped = coef[0]*x+coef[1]*x*x
        local = d['local_errors']
        slopes = d['coefficients'][:, 1]
        intercepts = d['coefficients'][:, 0]
        root_inside = abs(intercepts) <= abs(slopes)*float(d['offset'])
        summary[name] = dict(event_rate=float(events.mean()), forecast_mean=float(forecasts.mean()),
            confirmation_axis_cosines=d['confirmation_axis_cosines'].tolist(),
            brier=float(losses.mean()), constant_brier=float(np.mean((constant-events)**2)),
            brier_gain=float(gain.mean()), exploratory_bootstrap_interval=np.quantile(boot, [.025, .975]).tolist(),
            drift_slope_mean=float(d['coefficients'][:, 1].mean()),
            positive_slope_fraction=float(np.mean(slopes>0)),
            expanding_root_in_probe_range_fraction=float(np.mean((slopes>0)&root_inside)),
            risk_map_confirmation_rmse=float(np.sqrt(np.mean((mapped-risk)**2))),
            off_axis_fraction=float(np.sum(residual**2)/(np.sum(residual**2)+np.sum(x*x)+1e-30)),
            local_prediction_rmse=float(np.sqrt(np.mean(local**2))),
            probe_noise_sd=float(np.sqrt(d['coefficients'][:, 2].mean())),
            projection_outside_development_offset_fraction=float(np.mean(abs(x)>float(d['offset']))))
        axes[row, 0].plot(x.T, alpha=.35)
        axes[row, 0].set(title=name+' projected trajectories', xlabel='update', ylabel='BN projection')
        axes[row, 1].scatter(x.ravel(), risk.ravel(), s=3, alpha=.2)
        grid = np.linspace(x.min(), x.max(), 200)
        axes[row, 1].plot(grid, coef[0]*grid+coef[1]*grid**2, color='black')
        axes[row, 1].axhline(protocol['epsilon'], color='red', linestyle='--')
        axes[row, 1].set(xlabel='BN projection', ylabel='risk increase')
        scale = d['batch_scaling']
        axes[row, 2].plot(scale[0, :, 0], scale[:, :, 2].mean(0), 'o-', label='projected variance')
        axes[row, 2].plot(scale[0, :, 0], scale[:, :, 3].mean(0), 'o-', label='full covariance trace')
        axes[row, 2].set(xlabel='batch size', ylabel='update variance', xscale='log', yscale='log')
        axes[row, 2].legend()
        d.close()
    fig.tight_layout()
    fig.savefig(args.result/'projection.png', dpi=160)
    plt.close(fig)
    save_json(args.result/'summary.json', summary)


if __name__ == '__main__':
    main()

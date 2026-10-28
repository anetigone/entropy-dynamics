"""Prediction and censored summaries shared by the mixed/visual experiments.

All times are first risk exceedances, not risk minimizers. Predictions are
asymptotic plug-in diagnostics, not finite-sample confidence guarantees.
"""
import numpy as np
from scipy.special import ndtr


def joint_prediction(grid, n, geometry):
    """Unconditional left/right CDFs; seed SD excludes the 1/sqrt(n) factor."""
    t = np.asarray(grid)
    if geometry['A'] <= 1 or geometry['seed_sd'] <= 0:
        raise ValueError('Prediction requires an unstable peak and nonzero noise')
    if min(geometry['left_H'], geometry['right_H']) <= 0:
        raise ValueError('The peak must be inside the risk-safe component')
    scale = np.sqrt(n)*np.exp(-t*np.log(geometry['A']))
    mu, sd = geometry['seed_mean'], geometry['seed_sd']
    left = ndtr((-scale*geometry['left_H']-mu)/sd)
    right = ndtr((mu-scale*geometry['right_H'])/sd)
    # A Gaussian infinite-horizon seed approximation has a small spurious t=0
    # tail. Retain it as a diagnostic instead of masking it to improve the fit.
    return left, right


def observed_cdf(times, directions, grid):
    times, directions = np.asarray(times), np.asarray(directions)
    def cdf(mask):
        return np.searchsorted(np.sort(times[mask]), grid, side='right')/len(times)
    return cdf((times >= 0) & (directions < 0)), cdf((times >= 0) & (directions > 0))


def summarize(times, directions, grid, left, right):
    """Administrative censoring stays in the denominator of every CDF."""
    el, er = observed_cdf(times, directions, grid)
    def quantile(cdf, q):
        ids = np.flatnonzero(cdf >= q)
        return int(grid[ids[0]]) if len(ids) else None
    exited = times >= 0
    count = len(times)
    # Count all exits directly. Adding left/right floating CDFs can put an
    # exact empirical quantile just below its level and shift it by a step.
    total = np.searchsorted(np.sort(times[exited]), grid, side='right')/count
    return dict(paths=count, censored=int(np.sum(~exited)),
                observed_quantiles=[quantile(total, q) for q in [.1,.5,.9]],
                predicted_quantiles=[quantile(left+right, q) for q in [.1,.5,.9]],
                observed_right_by_horizon=float(er[-1]),
                right_probability_bounds=[float(er[-1]), float(er[-1]+np.mean(~exited))],
                predicted_right_by_horizon=float(right[-1]),
                cdf_error=float(np.max(np.abs(total-left-right))),
                joint_cdf_error=float(max(np.max(abs(el-left)), np.max(abs(er-right)))),
                cdf_sampling_band_95=float(np.sqrt(np.log(40)/(2*count))))

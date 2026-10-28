"""Peak and conjugacy calculations for labeled or unlabeled entropy fields."""
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator


def calibrate_peak(field, lower, upper, eta):
    mesh = np.linspace(lower-2,upper+2,513)
    velocity = np.array([field.moments(x)[0] for x in mesh])
    peaks = []
    for i in range(len(mesh)-1):
        if velocity[i] <= 0 < velocity[i+1]:
            q = brentq(lambda x:field.moments(x)[0],mesh[i],mesh[i+1])
            if field.moments(q)[1] > 0:
                peaks.append(float(q))
    if len(peaks) != 1:
        raise ValueError(f'Expected one unstable peak, found {len(peaks)}')
    q = peaks[0]
    _, lam, variance = field.moments(q)
    A = 1+eta*lam
    if A <= 1 or variance <= 0:
        raise ValueError('Peak requires positive curvature and noise variance')
    return dict(q=q,lambda_=lam,A=A,seed_sd=float(np.sqrt(eta**2*variance/(A*A-1))))


def conjugate_prediction(field, calibration, n, x0, lo, hi, eta):
    q, A = calibration['q'],calibration['A']
    if not lo < q < hi:
        raise ValueError('Calibrated peak lies outside the frozen-risk safe component')
    mesh = np.unique(np.r_[np.linspace(lo,hi,4097),q])
    mapped = mesh+eta*np.array([field.moments(x)[0] for x in mesh])
    if np.any(np.diff(mapped) <= 0):
        raise ValueError('Update is not monotone on the calibrated interval')
    inverse = PchipInterpolator(mapped,mesh,extrapolate=False)

    def H(x):
        count = 0
        while abs(x-q) > 1e-5:
            x = float(inverse(x))
            count += 1
            if not np.isfinite(x) or count > 10000:
                raise ValueError('Inverse conjugacy did not approach the peak')
        return A**count*(x-q)

    linear = dict(**calibration,seed_mean=float(np.sqrt(n)*(x0-q)),left_H=q-lo,right_H=hi-q)
    full = dict(**calibration,seed_mean=float(np.sqrt(n)*H(x0)),left_H=-H(lo),right_H=H(hi))
    probes = np.linspace(lo,hi,17)[1:-1]
    residual = max(abs(H(x+eta*field.moments(x)[0])-A*H(x))
                   for x in probes if lo <= x+eta*field.moments(x)[0] <= hi)
    return dict(actual_linear=linear,conjugate=full),dict(
        left_boundary=lo,right_boundary=hi,conjugacy_residual=float(residual),
        inverse_grid_points=len(mesh))

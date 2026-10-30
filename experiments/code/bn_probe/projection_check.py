"""Independent frozen-projection checks, batch probes and finite-horizon forecasts."""
import json
import numpy as np
from bn_dynamics import Dynamics, parser, save_json, scalar_forecast


def main():
    p = parser(__doc__)
    p.add_argument('--development', type=__import__('pathlib').Path, required=True)
    p.add_argument('--runs', type=int, default=32)
    p.add_argument('--calibrations', type=int, default=16)
    p.add_argument('--probe-batches', type=int, default=32)
    p.add_argument('--simulations', type=int, default=4096)
    args = p.parse_args()
    if min(args.runs, args.calibrations, args.probe_batches, args.simulations) < 2:
        p.error('All repetition counts must be >=2')
    if args.output.exists():
        raise FileExistsError(args.output)
    dev = json.loads((args.development/'protocol.json').read_text(encoding='utf-8'))
    for key in ['batch_size', 'eval_batch_size', 'horizon', 'lr', 'epsilon', 'severity', 'mean', 'std', 'corruptions']:
        if getattr(args, key) != dev[key]:
            p.error(f'Frozen design mismatch: {key}')
    frozen = np.load(args.development/'projection.npz')
    dyn = Dynamics(args)
    direction, origin, coef = [frozen[k] for k in ['direction', 'origin', 'risk_coef']]
    if dyn.names != dev['parameter_names'] or not np.array_equal(dyn.origin, origin):
        raise ValueError('Checkpoint BN parameters differ from development')
    offset = float(frozen['offset'])
    frozen.close()
    streams = np.random.SeedSequence([dev['seed'], args.seed, 1]).spawn(3)
    cal_rng, future_rng, sim_rng = [np.random.default_rng(s) for s in streams]
    args.output.mkdir(parents=True)
    protocol = {k: str(v) if hasattr(v, '__fspath__') else v for k, v in vars(args).items()}
    protocol.update(complete=False, development=dev,
                    predictor_information='Frozen development projection, labelled development risk map and event rate; independent unlabelled batch probes. No confirmation labels or trajectories.',
                    event='Full-panel batch-statistics classification error increase > epsilon at any update <=H',
                    sampling='Independent replacement draws from full official corruption block; not new images')
    save_json(args.output/'protocol.json', protocol)
    for corruption in args.corruptions:
        images, labels = dyn.load(corruption)
        predictions, coefficients, batch_scaling = [], [], []
        # Calibration probes never inspect labels or confirmation trajectories.
        for rep in range(args.calibrations):
            increments = []
            for location in [-offset, 0., offset]:
                dyn.set_vector(origin + location*direction)
                increments.append([dyn.update(images, cal_rng) @ direction for _ in range(args.probe_batches)])
            increments = np.asarray(increments)
            forecast, a, b, var = scalar_forecast(offset, increments, coef,
                    args.horizon, args.epsilon, args.simulations, sim_rng)
            predictions.append(forecast)
            coefficients.append([a, b, var])
            # Actual complete-batch updates at theta0, not rescaled single-image gradients.
            dyn.set_vector(origin)
            scale = []
            for size in [args.batch_size//2, args.batch_size, args.batch_size*2]:
                if size < 2:
                    raise ValueError('Batch scaling requires base batch >=4')
                updates = np.asarray([dyn.update(images, cal_rng, size) for _ in range(args.probe_batches)])
                projected = updates @ direction
                scale.append([size, projected.mean(), projected.var(ddof=1),
                              np.var(updates, axis=0, ddof=1).sum(),
                              np.linalg.norm(updates.mean(0) - projected.mean()*direction)])
            batch_scaling.append(scale)
        projections, risks, residuals, events, local_errors, displacements = [], [], [], [], [], []
        a, b, var = np.mean(coefficients, axis=0)
        for run in range(args.runs):
            states, r = dyn.trajectory(images, labels, future_rng)
            displacement = states-origin
            displacements.append(displacement)
            x = displacement @ direction
            residual = np.linalg.norm(displacement-x[:, None]*direction, axis=1)
            projections.append(x)
            risks.append(r-r[0])
            residuals.append(residual)
            events.append(np.max(r-r[0]) > args.epsilon)
            # Diagnostic only: confirmation paths cannot refit predictor coefficients.
            local_errors.append(np.diff(x) - (a+b*x[:-1]))
            print(corruption, run, bool(events[-1]), flush=True)
        # Post-hoc diagnostic directions never replace the frozen forecasting axis.
        def diagnostic_axis(values):
            _, _, vt = np.linalg.svd(np.concatenate(values), full_matrices=False)
            return vt[0]
        axis_left = diagnostic_axis(displacements[::2])
        axis_right = diagnostic_axis(displacements[1::2])
        stability = [abs(axis_left @ direction), abs(axis_right @ direction), abs(axis_left @ axis_right)]
        np.savez_compressed(args.output/(corruption+'.npz'), forecasts=predictions,
                            coefficients=coefficients, batch_scaling=batch_scaling,
                            projections=projections, risk_increases=risks, residuals=residuals,
                            events=events, local_errors=local_errors,
                            development_constant=dev['event_rates'][corruption],
                            risk_coef=coef, offset=offset, confirmation_axis_cosines=stability)
    protocol['complete'] = True
    save_json(args.output/'protocol.json', protocol)


if __name__ == '__main__':
    main()

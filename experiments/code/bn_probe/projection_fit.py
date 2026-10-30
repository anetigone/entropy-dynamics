"""Develop and freeze one BN projection using independent Tent trajectories."""
import numpy as np
from bn_dynamics import Dynamics, parser, save_json


def main():
    p = parser(__doc__)
    p.add_argument('--runs', type=int, default=12)
    args = p.parse_args()
    if args.runs < 4:
        p.error('Need >=4 development runs')
    if args.output.exists():
        raise FileExistsError(args.output)
    dyn = Dynamics(args)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0]))
    states, risks, environments = [], [], []
    for corruption in args.corruptions:
        images, labels = dyn.load(corruption)
        for run in range(args.runs):
            s, r = dyn.trajectory(images, labels, rng)
            states.append(s - dyn.origin)
            risks.append(r - r[0])
            environments.append(corruption)
            print(corruption, run, float(np.max(r-r[0])), flush=True)
    states, risks = np.asarray(states), np.asarray(risks)
    # Equal weighting of runs/states; PCA of displacement about the declared origin.
    def leading(a):
        _, singular, vt = np.linalg.svd(a.reshape(-1, a.shape[-1]), full_matrices=False)
        return vt[0], float(singular[0]**2 / np.sum(singular**2))
    direction, fraction = leading(states)
    if np.dot(direction, states[:, -1].mean(0)) < 0:
        direction = -direction
    left, _ = leading(states[::2])
    right, _ = leading(states[1::2])
    x = states @ direction
    design = np.stack([x.ravel(), x.ravel()**2], axis=1)
    coef = np.linalg.lstsq(design, risks.ravel(), rcond=None)[0]
    residual = np.linalg.norm(states - x[..., None]*direction, axis=-1)
    offset = max(float(np.std(x)), 1e-6)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'projection.npz', origin=dyn.origin, direction=direction,
                        risk_coef=coef, offset=offset, states=states, risks=risks,
                        environments=environments)
    protocol = {k: str(v) if hasattr(v, '__fspath__') else v for k, v in vars(args).items()}
    protocol.update(parameter_names=dyn.names, dimension=len(direction), explained_fraction=fraction,
                    split_direction_abs_cosine=float(abs(left @ right)),
                    risk_map_rmse=float(np.sqrt(np.mean((design@coef-risks.ravel())**2))),
                    residual_norm_mean=float(residual.mean()),
                    event_rates={c: float(np.mean(np.max(risks[np.array(environments)==c], axis=1)>args.epsilon))
                                 for c in args.corruptions},
                    target='max over updates 0..H of fixed full-test-panel error minus initial error > epsilon',
                    normalization='User must match checkpoint training preprocessing',
                    complete=True)
    save_json(args.output/'protocol.json', protocol)


if __name__ == '__main__':
    main()

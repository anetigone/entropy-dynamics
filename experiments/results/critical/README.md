# Critical SGD: historical results

This directory retains critical_results.json, critical_samples.npz, critical_checks.png, geometry_audit.json, run_log.txt, and the associated [analysis](analysis.md). Current evidence selection and next steps live in [STATUS.md](../../STATUS.md).

Code is organized by purpose under [experiments/code/entropy/](../../experiments/code/entropy/). See the [experiment README](../../experiments/README.md) for all commands and dependencies. From the repository root:

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_critical.py --phase compute
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_critical.py --phase plot
```

To update finite-parameter geometry using saved trajectories, run --phase refine, then --phase plot in a separate process. Optional diagnostics use entropy_check.py. All entries accept --output; default output remains here. Commands may overwrite same-name outputs. Plotting also writes critical_checks.svg and does not rerun stochastic simulations.

Proofs: [critical SGD](../../paper/notes/critical_sgd.md), [geometry](../../paper/notes/entropy_geometry.md), and the [integrated draft](../../paper/DRAFT.md).

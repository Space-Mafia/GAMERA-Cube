# Output, diagnostics and troubleshooting

The launcher writes `config_receipt.json` with the executable SHA256,
Git commit/dirty state, full command, grid/decomposition, output times,
optimization switches and driver settings. Preserve it with your results.

- `run.log`: actual solver progress and failure diagnostics.
- `validation.log`: the original HDF5 validator's report.
- `run_complete.txt`: created only after simulation and validation succeed.
- `analysis_grid_p{0..5}.h5`: static coordinates/vertices in RE.
- `analysis_p{face}_{sequence:06d}.h5`: cell-centered state.
- `mi_ionosphere_{sequence:06d}.h5`: coupled hemispheric diagnostics.

The state fields are rho, p, vx, vy, vz, Bx, By, Bz in **code units**.
B is total field (background plus residual), not perturbation B.
Each frame includes actual time/step and the normalization vector:
mu0, length, speed, time, density, pressure, B, dipole moment, frequency,
conductance (SI). Multiply code values by the relevant normalization.
The plotter does this conversion; never label raw code pressure as nPa.
See [detailed schema](../cubedsphere/docs/OUTPUT_DATA_MODEL.md).

The validator checks complete six-face frames and MI files, finite fields,
positive density/pressure and requested temporal coverage. A successful job
exit alone is insufficient. Numerical guards in the driver are retained;
inspect `run.log` too. Neither validator nor a smooth plot proves convergence.

## Common problems

| Symptom | Action |
|---|---|
| Production executable missing | Use parallel HDF5, not serial HDF5; configure with `GAMERA_REQUIRE_PRODUCTION=ON`. |
| MPI launch/rank-count failure | Match ranks to six times the per-face topology and check scheduler allocation. |
| OpenMP oversubscription | Reserve the requested CPUs/task; do not stack an outer threaded launcher. |
| Test process cannot bind sockets on macOS | Allow local MPI networking; distinguish launch errors from numerical failures. |
| Python NumPy/h5py import failure | Use the plotting virtual environment; check PYTHONHOME/PYTHONPATH and interpreter. |
| Run directory already exists | Choose a new directory. Never delete evidence just to reuse a name. |
| Driver stops early or validator fails | Preserve logs and receipt; do not loosen tolerances or call incomplete data a completed run. |
| Native-cell plot looks faceted | This is the actual mesh, not a rendering failure. No smoothing is applied. |
| Colours saturate | Fixed ranges deliberately support time comparisons; inspect data before adjusting scales. |

For bug reports include Git SHA, compiler/MPI/HDF5 versions, exact command,
receipt, and the relevant log tail. Do not upload credentials or multi-GB
raw outputs to an issue.

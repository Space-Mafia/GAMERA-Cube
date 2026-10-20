# Reproduce a southward-IMF magnetosphere

## 1. Inspect before launching

```bash
python3 scripts/run_earth.py --profile reference --output runs/earth-3h --dry-run
```

| Profile | Active cells per face | MPI topology per face | MPI × OMP | Default duration/output |
|---|---|---|---|---|
| smoke | 72 × 16 × 16 | 1 × 1 × 1 | 6 × 1 | 1 s / 1 s |
| teaching | 72 × 24 × 24 | 2 × 2 × 2 | 48 × 2 | 10,800 s / 300 s |
| reference | 128 × 32 × 32 | 4 × 2 × 2 | 96 × 2 | 10,800 s / 300 s |

There are six faces, so MPI ranks = 6 × radial tiles × angular tiles²
(or the product of the two angular tile counts when they differ).
The reference case contains 786,432 active cells, excluding ghosts.
The teaching profile is cheaper but is not interchangeable with the
reference grid or a grid-convergence result.

## 2. Run a short installation check

```bash
python3 scripts/run_earth.py --profile smoke --output runs/smoke
```

Require exit status zero, `run_complete.txt`, and `EARTH_HDF5_OUTPUT_PASS`
in `validation.log`. This validates an actual MPI/HDF5 launch, NOT a
developed magnetosphere. Each output directory must be new.

## 3. Run three physical hours

On a 192-core allocation:

```bash
python3 scripts/run_earth.py --profile reference --output runs/earth-3h
```

For Slurm, from the repository root:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION examples/earth/reference.slurm
```

Load the same compiler/MPI/HDF5 stack used for compilation before submitting.
The template requests 96 ranks with two threads each. Sites may require a
specific `srun --mpi` plugin or CPU placement policy; consult the cluster
documentation. The launcher checks allocation counts but does not guess
site-specific modules, account names or PMI-library paths.

The default reference case requests 37 output times including t=0.
Expect several wall-clock hours, depending on hardware and startup timesteps.
Historical seven-hour Earth runs cost 13.89 h before optimization and 11.10 h
with two halo optimizations on 192 cores; these are NOT a measured runtime
promise for this new constant-IMF configuration. Reserve at least several GB
for output. State arrays alone for the reference example are approximately
0.93 GB decimal (37 × 6 × 128 × 32 × 32 × 8 × 4 bytes), plus static geometry,
MI files, metadata and logs. Memory also includes large halo/geometry caches;
do not estimate RAM from state arrays alone.

For a shorter development run:

```bash
python3 scripts/run_earth.py --profile teaching --stop 600 --interval 60 \
  --output runs/earth-600
```

Duration must be an integer multiple of output interval. The default maximum
step count is one million; failure to reach the requested time fails validation.
Snapshots occur on actual steps, so timestamps may slightly exceed nominal
output times. HDF5 files are analysis output, **not restart checkpoints**.

## 4. Inspect, then interpret

```bash
python3 scripts/plot_earth.py runs/earth-3h --output runs/earth-3h/final.png
python3 scripts/plot_earth.py runs/earth-3h --sequence 12 \
  --extent 15 --output runs/earth-3h/t1h-near.png
```

At 300 s cadence sequence 12 is approximately one hour. Read the actual time
shown in the plot. Default axes span ±30 RE; `--extent 200` shows the whole
spherical domain at lower visual detail. Colours are fixed across snapshots;
the colourbar extensions indicate clipped values. The grey disk is the
excluded inner region, not a simulated solid Earth surface.

Do not identify a steady state from a single frame. Compare several late-time
snapshots and inspect FAC/potential and solver diagnostics before making
scientific claims. The three-hour exercise is a reproducibility baseline,
not proof of asymptotic equilibrium or realistic space-weather prediction.

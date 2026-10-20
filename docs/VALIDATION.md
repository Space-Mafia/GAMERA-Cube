# Validation and provenance

## Source lineage

Initial source is the isolated optimized Earth snapshot
`GAMERA-OP-cubedsphere-perf-halo-parallel-20261015`.
Original MIT copyright and GAMERA-OP citation are preserved. Shared numerical
dependencies remain under `nonorthogonal/src/`; private deployment recipes,
outputs, build directories and Io/Mars experiments are not bundled.
The old development narrative is retained in `DEVELOPMENT_HISTORY.md` for
historical context only; its statements about unfinished features and commands
are not current student instructions.

Inherited evidence:

- Earth 128×32×32×6, 96 MPI×2 OMP, seven hours: 85 frames,
  final 25200.046 s, 353114 steps, reported final divB 1.329e−11.
- Compact/index-cache seven-hour verification completed against the accepted
  baseline; 39951 wall seconds versus the original 50008.
- Five-optimization repeated same-allocation benchmark: 1.4444× driver
  speedup, eight output-bit comparisons passed.
- Parallel-halo tests: radial/angular OMP1/2, original build regressions and
  four accepted-output bit comparisons passed.
- These histories are NOT acceptance of this new constant-IMF release.

## Release-specific changes

1. Earth interior radial tiles now refresh current and old cell-centered B
   from synchronized face fluxes instead of returning before recovery.
   No physical boundary formula, conserved state or face flux is changed
   by this interior-tile operation. `earth_interior_recovery` exercises
   both time levels and checks flux/state preservation. It failed on the
   original code and is rerun after the correction.
2. Optional `GAMERA_CS_EARTH_CONSTANT_BZ_NT` selects a one-record driving
   series and matching upstream initial IMF. Absence retains the old
   0/+5/−5 nT switched driver. The student launcher selects −5.
3. Production dependency check, clean launcher, docs and plotting.

The synchronization correction may change a multi-radial-tile trajectory.
Do not promise bitwise identity to the historical uncorrected seven-hour run.

## Current acceptance

Release-specific test results are recorded in `RELEASE_STATUS.md`.
Until the exact packaged three-hour reference run has passed, this repository
is a **release candidate**, not a newly validated production science release.
The reference driver, HDF5 checker and regression thresholds are retained;
failures must be investigated rather than hidden by tolerance changes.

## Maintainer release checklist

- Build with MPI, actual OpenMP and parallel HDF5; record compiler/library versions.
- Run original selected regressions, MPI tile and MI parity tests.
- Exercise the interior recovery regression (old fail / new pass).
- Run the packaged smoke command, validator and plotter.
- Run constant −5 nT reference for 10800 s, output300 s: 37 complete frames.
- Check original numerical guards, final time, MI diagnostics and native-cell figures.
- Record hashes/jobs/timings; tag only the tested commit.
- Keep future planet examples independently gated.

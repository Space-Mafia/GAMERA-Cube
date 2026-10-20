# Release status — 2026-10-20

Initial student release candidate; not yet a production acceptance tag.

- Local AppleClang Release build: passed (without OpenMP/parallel HDF5).
- New Earth interior-recovery regression: original code failed; corrected
  code passed. This checks both current/old B caches and unchanged face fluxes
  and conserved state.
- 28 selected local tests passed initially. MPI-launched tests were initially
  blocked by sandbox communication; permitted reruns passed the radial/angular
  halo checks and explicit radial/angular Earth-MI checks. This machine does
  not have detected OpenMP, so these do not establish actual OMP2 acceptance.
- Launcher profile/environment-isolation and invalid-duration unit tests: passed.
- Linux MPI/OpenMP/parallel-HDF5 build and full regression suite: pending.
- Packaged 1-second smoke, HDF5 validation and native-cell plotting: pending.
- Exact constant −5 nT, 128×32×32×6, 3-hour reference: pending.

See `docs/VALIDATION.md` for historical evidence and the acceptance checklist.
No complete reference run is implied by the presence of runnable examples.

Cluster packaging gate 28979457 stopped before compilation because a Python
distribution's `mpicc` shadowed the cluster MPI compiler. Its evidence is
preserved. Gate 28979471 selects Python by absolute path without changing the
compiler PATH; no numerical setting or threshold was changed.

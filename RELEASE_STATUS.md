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
- Public Linux CI run 39392107337: passed production build, selected
  numerical tests, launcher tests, MPI/parallel-HDF5 smoke and plotting.
  Smoke reached 1.14811266 s in six steps, with two complete frames,
  `EARTH_HDF5_OUTPUT_PASS` and final divB 2.152e−12.
  The generated native-cell image was inspected.
- Cluster full regression suite (actual OpenMP/Intel MPI): in progress,
  gate 28979471. Do not claim this gate passed until its terminal result.
- Additional 64-step, four-radial-tile Earth/MI comparison against the
  six-face reference: job 28979573, dependent on the gate. Same scaled
  1e−12 comparator; specifically exercises interior radial tiles.
- Exact constant −5 nT, 128×32×32×6, three-hour reference: job 28979574,
  dependent on both gates succeeding. Target 37 complete frames.
  This is not yet a completed/accepted magnetosphere result.

See `docs/VALIDATION.md` for historical evidence and the acceptance checklist.
No complete reference run is implied by the presence of runnable examples.

Cluster packaging gate 28979457 stopped before compilation because a Python
distribution's `mpicc` shadowed the cluster MPI compiler. Its evidence is
preserved. Gate 28979471 selects Python by absolute path without changing the
compiler PATH; no numerical setting or threshold was changed.

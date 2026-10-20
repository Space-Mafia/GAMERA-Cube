# Release status — 2026-10-20

Initial student release candidate; not yet a production acceptance tag.

- Local AppleClang Release build: passed (without OpenMP/parallel HDF5).
- New Earth interior-recovery regression: original code failed; corrected
  code passed. This checks both current/old B caches and unchanged face fluxes
  and conserved state.
- 28 selected local tests passed initially; eight MPI-launched tests were
  blocked by sandbox communication and are being rerun with networking allowed.
- Linux MPI/OpenMP/parallel-HDF5 build and full regression suite: pending.
- Packaged 1-second smoke, HDF5 validation and native-cell plotting: pending.
- Exact constant −5 nT, 128×32×32×6, 3-hour reference: pending.

See `docs/VALIDATION.md` for historical evidence and the acceptance checklist.
No complete reference run is implied by the presence of runnable examples.

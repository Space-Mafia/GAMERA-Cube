# Contributing

Open an issue describing the physical or numerical problem before changing
solver algorithms. Preserve existing tests and acceptance thresholds.
Include a focused regression with bug fixes and record compiler/MPI/HDF5
versions. Keep planetary adapters separate from shared numerical kernels.

Performance changes need same-allocation comparisons and numerical equivalence
tests, not just shorter elapsed time from different nodes. Never alter
physics or smooth plots to make an instability appear/disappear.

Documentation and examples must not contain user-specific accounts,
absolute cluster paths, credentials or raw simulation output.
Retain the upstream license/citation and identify shared-kernel changes.

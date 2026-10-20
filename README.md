# GAMERA-Cube

Optimized, six-block **cubed-sphere magnetohydrodynamics** for planetary
magnetospheres. The first student example is Earth's magnetosphere driven
by a constant **Bz = −5 nT** solar wind, with aligned-dipole
magnetosphere–ionosphere coupling.

**Release status: initial release candidate.** The inherited solver has
seven-hour Earth verification and output-equivalent optimization benchmarks.
This package adds an Earth synchronization correction and a constant-IMF
example; consult [validation](docs/VALIDATION.md) for what has actually
passed in this revision. Do not confuse historical validation with acceptance
of a new three-hour simulation.

## Start here

1. [Install dependencies and build](docs/INSTALL.md).
2. [Run the smoke test, then a three-hour magnetosphere](docs/EARTH_TUTORIAL.md).
3. [Understand the model, coordinates and limitations](docs/MODEL.md).
4. [Inspect results and troubleshoot](docs/OUTPUT_AND_TROUBLESHOOTING.md).
5. [Read validation and source provenance](docs/VALIDATION.md).

```bash
git clone https://github.com/Space-Mafia/GAMERA-Cube.git
cd GAMERA-Cube
bash scripts/build.sh -DCMAKE_C_COMPILER=mpicc
python3 scripts/run_earth.py --profile smoke --output runs/smoke
# On an allocation with 96 MPI ranks x 2 CPU threads:
python3 scripts/run_earth.py --profile reference --output runs/earth-bzminus5
python3 -m pip install -r requirements-plot.txt
python3 scripts/plot_earth.py runs/earth-bzminus5 --output runs/earth-bzminus5/final.png
```

The three-hour run is an **HPC exercise**, not a quick laptop demo.
Never run a full magnetosphere on a shared login node. For Slurm use
[`examples/earth/reference.slurm`](examples/earth/reference.slurm), after
loading your site's compiler/MPI/parallel-HDF5 modules.

## Repository organization

- `cubedsphere/`: geometry, interfaces, MPI tiling, Earth boundary/MI adapters,
  analysis output and regression tests.
- `nonorthogonal/src/`: inherited GAMERA numerical kernels and dependencies.
  The folder name describes the numerical implementation; it does not select
  a Yin–Yang grid. This release builds the cubed-sphere application only.
- `scripts/`: supported build, reproducible launch and native-cell plotting.
- `examples/earth/`: portable scheduler template.
- `docs/`: student guides, validation and historical development record.

No Io/Mars experiment settings, private cluster credentials, raw simulation
data or precompiled executables are needed. Future Mars/Venus/Io configurations
must receive separate physical-model documentation and validation before
being advertised as supported examples.

## License and citation

MIT; original copyright is retained in [LICENSE](LICENSE). This package
derives from GAMERA-OP and its non-orthogonal/cubed-sphere development.
Please retain and use [CITATION.cff](CITATION.cff) for the underlying published
GAMERA-OP method, and record the GAMERA-Cube Git revision in any results.
That citation is not a claim that all cubed-sphere extensions were validated
in the original orthogonal-grid paper.

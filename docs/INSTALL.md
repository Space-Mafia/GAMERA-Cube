# Installation

## Supported production build

Use Linux with a C99 compiler, CMake >=3.16, MPI C, OpenMP, and **parallel**
HDF5 built against the same MPI implementation. Python 3.9+ runs the launcher.
NumPy, h5py and Matplotlib are needed only for plotting.

On Ubuntu/Debian a typical administrator-managed installation is:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake git openmpi-bin libopenmpi-dev \
  libhdf5-openmpi-dev python3 python3-venv
```

Do not run these administrator commands on a managed cluster; load its modules.
Do not mix Open MPI libraries with an Intel MPI executable or vice versa.
Use `h5pcc -showconfig` to check `Parallel HDF5: yes`.

```bash
bash scripts/build.sh -DCMAKE_C_COMPILER=mpicc
# If CMake chooses serial HDF5, specify the correct installation:
bash scripts/build.sh -DCMAKE_C_COMPILER=mpicc -DHDF5_ROOT=/path/to/parallel-hdf5
```

The wrapper configures `-O3 -DNDEBUG -fno-fast-math`, builds the code, and
runs the baseline selected regressions. Some inherited test names do not
contain `mpi` although they launch multiple ranks: run the build/test wrapper
on an allocation supporting at least 24 MPI processes, or configure/build
first and run CTest on such an allocation. Use `ctest --test-dir build -N -V`
to inspect commands. Do not enable fast-math or silently remove failing tests.
The wrapper clears experiment-related variables before testing.

For the full suite (includes MPI and longer wave/parity tests):

```bash
OMP_NUM_THREADS=1 ctest --test-dir build --output-on-failure --timeout 900 -j 1
```

Confirm both `build/cubedsphere_earth_mpi_drive` and
`build/check_cubedsphere_earth_hdf5` exist. The production requirement fails
configuration rather than silently substituting CSV output.

## Local development without parallel HDF5

```bash
cmake -S cubedsphere -B build-local -DCMAKE_BUILD_TYPE=Release
cmake --build build-local -j 4
ctest --test-dir build-local -R '^earth_interior_recovery$' --output-on-failure
```

This builds numerical tests but may NOT build the production driver.
AppleClang may not discover OpenMP automatically; local serial tests do not
validate a two-thread production run. Linux/HPC is the reference production
environment. No requirement to install Intel MPI for laptop development.

Plotting dependencies in an isolated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-plot.txt
```

If a cluster module sets `PYTHONHOME` or `PYTHONPATH` for another interpreter,
unset those before activating a virtual environment. Never combine one
Python executable with another version's standard library.

# Cubed-sphere Earth analysis output

The production drive uses the accepted Yin–Yang GAMERA compact-analysis
HDF5 layout. A cubed-sphere face is a patch (`p0` … `p5`). The output
interval is configurable; one file is written per patch **at each analysis
output**, not at every MHD integration step. The actual MHD step and time are
saved in each file.

| File | Contents |
| --- | --- |
| `analysis_grid_p0.h5` … `analysis_grid_p5.h5` | Static `/coordinates` float64 `[3,NI,NJ,NK]` and `/vertices` float64 `[3,NI+1,NJ+1,NK+1]`, in Cartesian Earth radii. |
| `analysis_p0_000000.h5` … `analysis_p5_000000.h5` | `/state` float32 `[8,NI,NJ,NK]` in `rho,p,vx,vy,vz,Bx,By,Bz` order, matching `/field_names`. `B` is the total field, residual plus analytic dipole. |
| `mi_ionosphere_000000.h5` | North/south FAC and potential, longitude and colatitude, constant Pedersen conductance, and native sample-shell current, with explicit units. |

`/analysis_schema`, `/patch_id`, `/sequence`, `/global_cells`,
`/time_code`, `/time_seconds`, and `/normalization` follow the Yin–Yang
analysis metadata contract. `/step` is an additional 64-bit MHD step number.
The coordinate-backend ID is 3 for the six-face cube sphere. MI uses a
documented subset of the Yin–Yang MI science fields: there is no electron
precipitation or Hall conductance in this aligned-dipole test, and neither is
silently invented in the output.

All ranks of one face collectively write non-overlapping HDF5 hyperslabs of
that face's file. Only the highest tile in each axis writes the final vertex
plane. No rank-zero global MHD gather, temporary per-rank CSV, or full-volume
replication is used. Rank zero writes the much smaller MI file after the
already-established MI shell gather and solve. This output is for analysis,
not restart. The local parity-test executable retains its legacy CSV
diagnostics, but `cubedsphere_earth_mpi_drive` requires parallel HDF5 and
never falls back to CSV.

For Python postprocessing, open each face's static grid once and each
`analysis_p{face}_{sequence:06d}.h5` with `h5py`; a 2-D slice can then be
assembled from the six faces using the Cartesian coordinates. The HDF5
validator checks all faces and MI files at every expected output time,
including finite state and positive density/pressure.
On Sugon, `module load anaconda3/2023.09` provides Python with `h5py`
and NumPy for this postprocessing; the simulation itself uses the separate
parallel HDF5 module.

```python
import h5py

sequence = 12
for face in range(6):
    with h5py.File(f"analysis_grid_p{face}.h5") as grid, \
         h5py.File(f"analysis_p{face}_{sequence:06d}.h5") as frame:
        xyz = grid["coordinates"][:]       # (3, NI, NJ, NK), in RE
        rho = frame["state"][0]             # code units
        bz_total = frame["state"][7]        # code units, includes dipole
        time_s = frame["time_seconds"][0]
        # Select or interpolate each face's Cartesian cells here.
```

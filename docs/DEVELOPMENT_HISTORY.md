# Cubed-sphere GAMERA MHD (development branch)

This directory is the isolated six-block cubed-sphere MHD backend on branch
`feature/cubed-sphere-mhd-v1`. It is built against the existing
`nonorthogonal/src/nonorthogonal/` numerical kernels. The frozen Earth
`nonorthogonal/` application target, its Yin-Yang connectivity and its
production configuration are not changed.

## Geometry and numerical path

- Six conforming equiangular gnomonic faces cover a spherical shell. Each
  face has a right-handed `(r,a,b)` frame. Physical vertices and interface
  areas match on all twelve seams; there is no overset/ownership mask.
- Four ghost cells per side expose the original eight-cell, volume-weighted
  Up7/PDM GAMERA reconstruction in every local logical direction. Tangential
  ghost states are sampled from the *physical owning face* with an 8×8 donor
  stencil. At discontinuities, only a nonphysical interpolated ghost state
  falls back to its nearest physical donor cell.
- The existing GAMERA kinetic Reynolds/Maxwell fluxes and CT edge-EMF sweeps
  run on each face. Every physical inter-face flux is reconciled to one
  oriented value; every physical seam edge has one signed EMF. The EMF
  reduction includes the three-face radial edges at cube corners. No
  independent per-face CT update can introduce a mismatched shared edge.
- On twisted faces, the existing scalar face area and centerline normal are
  not exactly the integrated area vector used by finite-volume fluxes. The
  cubed-sphere target calibrates its kinetic face frame against that area
  vector and calibrates face-flux-to-cell-B recovery against three uniform
  Cartesian fields. These corrections are linked **only** into this target;
  the frozen Earth executable retains its original kernels.

Face order is `+X, -X, +Y, -Y, +Z, -Z`. Each face's tangent pair satisfies
`a × b = outward normal`. The active face resolution is `(nr,n,n)` plus four
ghost cells per side. In this first implementation `n` must be equal on all
faces and `n ≥ 9`; the radial ghost extension must remain at positive radius.

## Build and run the current MHD gates

From the repository root:

```sh
cmake -S cubedsphere -B build-cubedsphere -DCMAKE_BUILD_TYPE=Release
cmake --build build-cubedsphere -j
ctest --test-dir build-cubedsphere --output-on-failure
```

The standalone `cubedsphere/` MHD library has no HDF5 or MPI dependency.
If an MPI C installation is available, CMake also builds a separate
six-rank communication-oracle library and test. The Earth production CMake
target is untouched. The present test suite covers:

1. All six right-handed geometries, positive cell volumes, reciprocal edge
   topology, identical seam vertices, opposite outward magnetic flux, 8×8
   tangential ghost fill, conservative face-flux reconciliation and shared
   two-/three-face EMFs.
2. One-step uniform Cartesian MHD on each cube face, including face-field
   recovery and CT.
3. Three synchronized six-block uniform MHD steps.
4. A seam-centered magnetized pressure blast with x/y reflection symmetry.
5. A radial magnetized Riemann discontinuity crossing all six blocks.
6. A divergence-free circularly polarized Alfvén wave initialized from a
   vector potential and propagated across the six-block shell.
7. A 300-step radial shock and 200-step Alfvén extension of the short tests.
8. A three-resolution order measurement for the *initial CT-to-center
   magnetic representation* and, when MPI is available, a six-rank active
   cell exchange compared bitwise with the serial ghost interpolation.
9. A traveling Alfvén wave with time-dependent analytic radial ghost states,
   measured at three resolutions, plus six-rank shared magnetic/fluid/Maxwell
   flux and CT edge-EMF communication compared bitwise with serial
   reconciliation.
10. A pure-HD angular pressure blast initialized uniformly in radius on a
    six-face spherical shell, including a longer run until all six faces
    respond. This uses the same high-order fluid flux path with MHD disabled.
11. A 32×32-per-face magnetized angular blast, initially confined to the +X
    face, evolved through adjacent seams with shared CT edge EMFs. This gate
    uses the non-orthogonal driver values `alpha=0.5`, HOGS coefficients
    `0.25/0.25`, and EMF CFL coefficient `0.3`; its adaptive CFL step,
    positivity, discrete div B, and final seam-pressure jump are checked.

The tests check physical positivity and finite values, and measure discrete
`div B` as net magnetic face flux divided by cell volume. They intentionally
do not claim a production magnetosphere or a parallel MHD solver.

### Earth magnetosphere integration gate

The six-face step now accepts the existing non-orthogonal 12-point integrated
background-field data on each cube face. The same GAMERA sweep, Boris,
background Maxwell-force, EMF and CT kernels are used; the cube-sphere layer
only reconciles shared face fluxes and edge EMFs. The
`cubedsphere_dipole_background` gate constructs an analytic dipole on all six
faces, verifies that the shared CFL kernel responds to Boris limiting, and
advances one split-field MHD step with positive density/pressure and zero
residual discrete divergence. This is a short numerical integration gate,
**not** a solar-wind magnetosphere validation.

The six faces can also use the unchanged Earth production radial-map v4
function. At 72 radial cells, the `cubedsphere_earth_radial_map` gate checks
2.5 RE at the wall, exactly 23 half-RE cells through 14 RE, 200 RE at the
outer shell, and positive-radius inner ghost vertices. The default linear
shell generator remains available for the earlier numerical tests.

The local Earth integration gate uses **72×16×16 active cells per face**
(110,592 active cells in six faces), retaining the accepted 0.5 RE inner
radial spacing. `cubedsphere_dipole_background_72x16x16` checks one
split-field dipole/Boris MHD step at that resolution using the accepted
Problem 21 upstream-front primitive initialization, cut-dipole vector
potential, and background-to-residual conversion. These application formulas
are isolated in `cubedsphere_earth_startup.c`; they call the unchanged
non-orthogonal initialization, background and MHD/CT kernels. Since the
radially cut background field is not divergence-free by itself, this gate
measures the divergence of **total** B = B0 + B1. The six-face step now
calls the existing non-orthogonal `fluid_flux_sync` and `edge_emf_sync`
hooks at the same solver phases as the Yin-Yang driver, with cube-face
flux/EMF reconciliation around them. The gate invokes the existing
`gamera_no_trap_inner_outward_mass_energy_flux` routine and a zero inner-wall
EMF condition on all six faces. This is an interface/one-step test, not a
solar-wind magnetosphere run.

The following remain to be adapted from the accepted Yin-Yang Earth driver
before a northward-IMF run is meaningful: its upstream initial state,
physical inner-wall and solar-wind ghost states at both time
levels, inner-wall fluid-flux and EMF conditions, outer diffusive solar-wind
EMF, global CFL/MPI advance, and production diagnostics. MI coupling follows
only after the no-MI magnetosphere passes. No Earth boundary has been
approximated by the present radial-outflow test fill.

### Current measured gates (AppleClang Release, local machine)

- At angular resolutions `n=9,18,36`, one tiny-time-step Alfvén probe gives
  full-domain RMS errors `9.120391e-5`, `2.299546e-5`, `5.761182e-6`.
  The corresponding orders are approximately 1.99 and 2.00. This is the
  second-order error floor of the present CT-face-to-center-field
  representation; **it is not a measured seventh-order MHD convergence
  result**.
- At physical time `0.04` with 20 steps, the same resolutions give full
  RMS errors `1.984623e-4`, `1.304715e-4`, `1.045471e-4`; excluding the
  outer two radial cells on each side gives `9.527090e-5`,
  `3.061791e-5`, `1.923053e-5`. The simple zero-gradient radial ghosts are
  incompatible with the traveling Cartesian Alfvén wave, so this test does
  not establish formal evolution order. The next gate replaces those ghosts
  with analytic traveling-wave data.
- With analytic traveling-wave values imposed in the nonphysical radial
  ghost cells/faces, the `t=0.04` full-domain RMS errors at `n=9,18,36`
  become `9.195415e-5`, `2.492605e-5`, `7.009773e-6`: observed orders
  about 1.88 and 1.83. Halving `dt` at fixed final time changes the
  `n=18` and `n=36` errors by less than 0.12%, so time error is not the
  leading term here. This is an **end-to-end approximately second-order
  MHD+CT wave result**, despite the retained eight-cell Up7/PDM face
  reconstruction. The analytic ghost magnetic flux currently uses B at the
  ghost-face centroid times its area vector; it is not an exact surface
  integral, so further spatial-order diagnosis remains open.
- The 300-step radial shock remains positive (`min rho=0.181741`,
  `min p=0.216545`) with max discrete `div B=7.13e-15`. The 200-step wave
  remains finite with RMS `2.560899e-3` and max discrete
  `div B=3.59e-14`. These are longer **stability gates**, not physical
  validation over magnetospheric times.
- The optional six-rank MPI test assigns one face to each rank, exchanges
  active conserved states at both time levels plus Cartesian B, and checks
  every reconstructed seam/corner ghost against the serial oracle bitwise.
  It also exchanges only O(nr×n) interface entries for shared magnetic,
  fluid and Maxwell face fluxes and for two-/three-face CT edge EMFs; the
  reconciled results match the serial oracle bitwise. It passes locally.
  The exchange test alone does not advance MHD; the separate MPI time-step
  adapter below now advances each face on its owning rank while reusing
  these exact interface collectives.
- The existing smooth donor-halo probe gives maximum errors
  `1.571391e-3`, `9.390630e-5`, `2.419998e-5` at `n=10,18,26`. They
  decrease rapidly, but this three-point non-asymptotic probe is not a
  formal high-order seam convergence proof.
- The HD blast centered on the common `+X/+Y/+Z` corner uses a shell
  `r=2.0–2.8`, `nr=8`, and `24×24` angular cells per face. It reaches
  `t=1.5` in 262 adaptive-CFL steps with final minimum density `0.16653`
  and minimum pressure `0.98154`; all six faces have a pressure response.
  A 120° cyclic rotation about the blast axis maps face cell centers to
  within `7.7e-15` and pressure to within `4.7e-15`. The largest mid-shell
  seam pressure jump is `0.3225`, or `2.66` times the local one-cell
  pressure-gradient scale. Native-cell plots show a propagating front without
  an obvious seam break, but this is not a formal seam-convergence result.
  The radial-outflow shell loses approximately 2.49% of its initial mass
  and develops radial velocity (maximum `0.511` at the end), so this is an
  angular blast in **3-D HD**, not a strictly constrained 2-D Euler solver.
- The long magnetized +X-centered blast uses a uniform Cartesian initial
  `B=(0.5,0,0.5)` and the same MHD flux, HOGS, EMF and CT kernels as the
  non-orthogonal backend. With the standard EMF/HOGS coefficients listed
  above, it reaches `t=1.5` in 231 adaptive-CFL steps. Final minimum density
  is `0.17737`, minimum thermal pressure `0.94996`, maximum discrete `div B`
  is `3.94e-14`, and the largest mid-shell seam pressure jump is `0.02288`.
  The jump/local-gradient ratio is printed but is not an acceptance criterion
  where the local gradient is nearly zero. Previous experimental `alpha=0`
  and `alpha=0.05` results are rejected as non-standard parameter runs; they
  are not MHD acceptance evidence. This shock test is not a claim of formal
  convergence or a production magnetosphere.

Reproduce the long HD gate and native-cell figures without resampling
(plotter requires NumPy, SciPy and Matplotlib):

```sh
build-cubedsphere/test_cubedsphere_hd_spherical_blast \
  blast_n24.csv 24 1.5
python3 cubedsphere/tests/plot_hd_spherical_blast.py \
  blast_n24.csv blast_figures
```

For a 32×32-per-face run with the initial high-pressure cap strictly inside
the `+X` face (angular support ≤0.22 rad; nearest seam at π/4 rad), use:

```sh
build-cubedsphere/test_cubedsphere_hd_spherical_blast \
  blast_n32_facecenter.csv 32 1.5 face-center 0.1
python3 cubedsphere/tests/plot_hd_spherical_blast_3d_movie.py \
  blast_n32_facecenter.csv sphere_frames
ffmpeg -framerate 5/4 -i sphere_frames/blast_%03d.png -r 10 \
  -c:v libx264 -pix_fmt yuv420p blast_n32_facecenter.mp4
```

The movie holds each recorded snapshot without temporal interpolation. Its
color is the pressure enhancement normalized independently at each snapshot;
the title states that frame's actual maximum pressure.

Run the magnetized counterpart (same cap and grid, with the standard
non-orthogonal EMF coefficient `alpha=0.5`) using:

```sh
build-cubedsphere/test_cubedsphere_hd_spherical_blast \
  blast_mhd_n32_facecenter.csv 32 1.5 face-center-mhd 0.1
python3 cubedsphere/tests/plot_hd_spherical_blast_3d.py \
  blast_mhd_n32_facecenter.csv blast_mhd_n32_final.png --label MHD
python3 cubedsphere/tests/plot_hd_spherical_blast_3d_movie.py \
  blast_mhd_n32_facecenter.csv blast_mhd_frames --label MHD
ffmpeg -framerate 5/4 -i blast_mhd_frames/blast_%03d.png -r 10 \
  -c:v libx264 -pix_fmt yuv420p blast_mhd_n32_facecenter.mp4
```

The CSV records mid-shell density and thermal pressure on native face cells;
the solver prints positivity, mass, div B and seam diagnostics. This is a
cross-seam numerical gate, not yet a magnetosphere application.

## Current boundary and scope

`gamera_cs_advance_six` is a **serial six-block standard-MHD** step. Its
default radial ghost boundary remains simple zero-gradient outflow for the
generic MHD gates. The separate `earth-local` development adapter reuses the
non-orthogonal Earth startup, split dipole/Boris kernels and solar-wind
sampling. With MI disabled, its physical inner shell has zero tangential
EMF, reflected normal flow, and zero radial gradients of density, thermal
pressure and residual magnetic field: a hard wall at 2.5 RE. The optional
`earth-mi-local` adapter adds aligned-dipole MI coupling, as described below.
Neither adapter has precipitation, multi-fluid species, a restart/output
driver; neither is **yet a production Earth magnetosphere**. An optional
six-rank distributed MHD/CT time-step gate is documented below, but still
replicates full donor states and is not high-resolution scalable. The current
8×8 donor interpolation uses physical
cell-center samples; its exact smooth-solution order across warped seam cells
remains to be measured and, if needed, made conservative. The current cubed-sphere-only
magnetic recovery calibration also needs a convergence/performance study
before any production use.

For the later magnetosphere stage, retain the existing Earth boundary,
solar-wind and MI physics behind cubed-sphere geometry/communication
adapters. Do not enable those modules by merely changing `COORD_TYPE` or
`PROBLEM` in the frozen Earth target.

The low-resolution no-MI Earth development run uses the v4 radial map
(2.5–14 RE in 0.5 RE cells, stretched to 200 RE), 72 radial layers and
16×16 angular cells on each of six cube faces. It uses the accepted Earth
normalization, 0.31 G dipole, 100 km/s velocity unit, CA=10 Boris correction,
and an idealized +X solar wind launched from x=-30 RE. The shell is the
geometric adapter; it does not change the GAMERA MHD flux, HOGS, EMF, CT or
Boris formulas. For a short gate and a native-cell slice:

```sh
cmake -S cubedsphere -B build-cubedsphere -DCMAKE_BUILD_TYPE=Release
cmake --build build-cubedsphere --target test_cubedsphere_dipole_background
OMP_NUM_THREADS=8 build-cubedsphere/test_cubedsphere_dipole_background \
  earth-local 500 earth_no_mi_500.csv
python3 cubedsphere/tests/plot_earth_no_mi_native.py \
  earth_no_mi_500.csv earth_no_mi_500.svg --time-code 1.50174312
```

The plot script intersects each original hexahedral cell with the XY or XZ
plane and colors that polygon; it does not interpolate the simulation onto a
finer Cartesian mesh. A 500-step snapshot is only about 96 physical seconds
and mainly tests the upstream launch, not a formed magnetosphere.

For a faster *development visualization* of the wind/obstacle interaction,
`GAMERA_CS_EARTH_FRONT_X=-12` moves only the initial wind front closer to the
Earth; it does not alter any MHD or boundary algorithm. One 3200-step local
run reached 558.9 physical seconds with finite active-cell outputs and
`max |div B|=1.875e-12` in code units. Its day-side compression is visible,
but the tail retains startup/seam-aligned sectors, and 336 outer cells were
at the pressure floor. Thus this is a no-MI *development snapshot*, not an
accepted steady-state or production magnetosphere. Reproduce it with:

```sh
GAMERA_CS_EARTH_FRONT_X=-12 OMP_NUM_THREADS=8 \
  build-cubedsphere/test_cubedsphere_dipole_background \
  earth-local 3200 earth_no_mi_front12_3200.csv
python3 cubedsphere/tests/plot_earth_no_mi_native.py \
  earth_no_mi_front12_3200.csv earth_no_mi_front12_3200.svg \
  --time-code 8.77262721 --xy-field density
```

### Aligned-dipole MI development gate

`earth-mi-local` uses the accepted non-orthogonal Stokes-current, dipole
field-line mapping, constant-Pedersen potential solve and ghost-velocity
composition. The cubed-sphere adapter samples FAC from the six-face MHD
shell and maps the ionospheric potential back to inner-wall edge EMFs by
potential differences. It applies the corresponding E×B ghost drift. This
baseline has **no dipole tilt, Hall conductance, or precipitation feedback**.
The ionosphere has 192 longitude × 22 colatitude samples per hemisphere,
constant SigmaP=5 S and SigmaH=0 S; the potential is solved at a natural
10-second cadence without clipping MHD timesteps. The FAC sample shell is
the second active radial cell, and the footpoint radius is 1 RE.

```sh
GAMERA_CS_EARTH_FRONT_X=-12 OMP_NUM_THREADS=8 \
  build-cubedsphere/test_cubedsphere_dipole_background \
  earth-mi-local 3200 earth_mi_3200.csv
python3 cubedsphere/tests/plot_earth_no_mi_native.py \
  earth_mi_3200.csv earth_mi_density.svg --time-code 8.93810508 \
  --xy-field density --mi
python3 cubedsphere/tests/plot_earth_mi_dipole.py \
  earth_mi_3200.csv.mi_north.csv earth_mi_3200.csv.mi_south.csv \
  earth_mi_polar.png
```

The 3200-step local gate completed to 569.4 physical seconds with finite
outputs and maximum total-field divergence of 1.922e-12 in code units.
At the last MI update (560.1 s), the north/south CPCP values were 52.948
and 52.946 kV. The corresponding FAC maps were opposite to within 0.33%
relative RMS, while the potential maps matched to within 0.19% relative
RMS. These are expected aligned, symmetric zero-IMF baseline checks, not
proof of a steady magnetosphere: the short-time MHD slices still exhibit
the no-MI startup/tail sectors, and some cells touch the pressure floor.
The `cubedsphere_earth_mi_dipole_zero_fac` and
`cubedsphere_earth_mi_dipole_single_fluid_reduction` CTests check the
initial zero-current coupling path. In a paired one-step run from the same initial
state, MI-on and MI-off produced identical density and pressure in all
110,592 active cells; maximum component differences were 1.15e-18 in
velocity and 5.69e-14 in magnetic field (code units). The production-level
next gate is the full
distributed run, restart continuity and longer-time MHD/MI morphology.

For a cheaper six-face development run, select 48 radial cells without
changing the 16×16 angular mesh or the inner 2.5–14 RE spacing. The first
23 radial cells remain 0.5 RE each; only the 14–200 RE stretched segment
uses fewer cells. This is a separate resolution-specific v4 map, not the
unaltered 72-cell production-v4 geometry. The native-cell plotter reads the
radial count from the snapshot and supports both layouts:

```sh
GAMERA_CS_EARTH_RADIAL_CELLS=48 GAMERA_CS_EARTH_INNER_UNIFORM_CELLS=23 \
  GAMERA_CS_EARTH_FRONT_X=-12 \
  OMP_NUM_THREADS=8 build-cubedsphere/test_cubedsphere_dipole_background \
  earth-mi-local 3200 earth_mi_48_3200.csv
python3 cubedsphere/tests/plot_earth_no_mi_native.py \
  earth_mi_48_3200.csv earth_mi_48_density.svg --time-code 8.93751077 \
  --xy-field density --mi
python3 cubedsphere/tests/plot_earth_mi_dipole.py \
  earth_mi_48_3200.csv.mi_north.csv earth_mi_48_3200.csv.mi_south.csv \
  earth_mi_48_polar.png
```

This 48-cell gate completed to 569.3 physical seconds in 800.0 wall
seconds on the same eight local OpenMP threads, versus 1140.7 wall seconds
for the 72-cell gate (1.43× faster). The final output had 73,728 finite
active cells and maximum total-field divergence 1.898e-12 in code units.
At the 560.1-second MI update, CPCP was 52.734/52.744 kV (north/south).
Relative to the 72-cell run at the same MI time, the FAC maps differed by
3.1–3.2% RMS and the potential maps by less than 1% RMS. The north/south
FAC antisymmetry residual was 0.76% RMS. This is a useful inexpensive MI
development baseline; the tail startup sectors and pressure-floor cells
still prevent a production claim.

#### Cube-corner FAC diagnostic correction

The MI current remains the existing non-orthogonal Stokes curl of the
**residual** cell-centered magnetic field, projected along the background
dipole and mapped to the ionosphere by the dipole field-strength ratio.
On a cube-sphere, differentiating the unlimited eighth-order interpolated
magnetic halo and then evaluating that curl in a receiver-face ghost cell
produced isolated FAC spikes at the three-face corners. For the 48-cell
sample shell at 3.256 RE, these corners map to 09/15 MLT at 63.1° MLAT.

The MI adapter now copies magnetic values into a **private diagnostic
buffer**, obtains its one-cell magnetic fringe by local bilinear
interpolation from physical donor cells, computes the unchanged Stokes
curl only in physical cells, and interpolates the resulting FAC into the
one-cell fringe. This does not change the MHD halo, reconstruction, CT,
or timestep. In a 3200-step local rerun of the corrected diagnostic fringe,
the 61.98° MLAT north FAC values
at 09/15 MLT changed from 0.0448/0.1034 to 0.00084/0.00755 µA m⁻²;
the south showed the corresponding sign reversal. The run completed with
finite outputs and maximum total-field divergence 1.843e-12 in code units.
The final private-buffer implementation reproduces its MI output bitwise
on the saved MHD snapshot, and its MHD and MI outputs match bitwise in a
paired 300-step evolution test.
This is a seam-current development gate, not a long-time production test.

To overlay the cube seams on the native MI FAC map, pass the final MHD
snapshot to the seam plotter. To inspect native MHD-shell J_parallel,
replay the saved snapshot through the same MI diagnostic without evolving
MHD:

```sh
python3 cubedsphere/tests/plot_earth_mi_seams.py \
  earth_mi_48_3200.csv.mi_north.csv \
  earth_mi_48_3200.csv.mi_south.csv \
  earth_mi_48_3200.csv earth_mi_48_fac_seams.png
GAMERA_CS_EARTH_RADIAL_CELLS=48 GAMERA_CS_EARTH_INNER_UNIFORM_CELLS=23 \
  build-cubedsphere/test_cubedsphere_dipole_background \
  earth-mi-replay earth_mi_48_3200.csv replay_48
python3 cubedsphere/tests/plot_mi_shell_current.py \
  replay_48.shell_j.csv earth_mi_48_shell_j.png
```

#### Fixed-grid halo geometry plan (local performance gate)

The cube-face ghost interpolation must use each cell/face centroid's **own**
donor coordinate. Reusing one angular stencil across all radial layers was
measurably faster but changed the MHD result after only two steps, so that
approximation was rejected. The accepted optional plan instead evaluates the
original 8×8 donor stencil once for **every actual ghost geometry point** and
reuses only its indices/weights over MHD steps. The donor conserved states and
magnetic fields are still read from the current/old time levels at every step.
The GAMERA flux, Boris, CT, MI, and CFL formulas are unchanged.

On the 48×16×16/face local Earth gate, the initial selected timestep is
`0.00300348621` code units (`0.191352` physical seconds) with Boris enabled
and a 0.5 CFL fraction. A same-host eight-thread, 300-step comparison took
80.63 s without the plan and 56.31 s with it (`1.432×` faster). The complete
MHD snapshot and both MI hemispheres were byte-identical. The plan costs
about 81 MiB of additional memory at this resolution. The development runner
creates it by default; set `GAMERA_CS_EARTH_DISABLE_HALO_PLAN=1` for an exact
unplanned A/B on the same executable. This is a framework optimization only;
the independent Yin-Yang production source is untouched.

The full 3200-step planned gate completed 569.409 physical seconds in 571.78
wall seconds (`0.996×` real time). Its final MHD CSV and both MI hemisphere
CSVs are byte-identical to the corrected-FAC 3200-step baseline. The final
MHD timestep is `0.00293528224` code units (`0.187007` physical seconds),
and maximum total-field divergence is `1.84287e-12` in code units. This
establishes numerical parity and local performance, not a long-duration
magnetosphere production acceptance.

The planned cell and magnetic-face halo entries are independent: each reads
only active donor cells and writes one unique receiving ghost location. With
OpenMP and more than one thread, both planned halo loops therefore run in
parallel by default. Set `GAMERA_CS_PARALLEL_HALO=0` to retain the serial
planned-halo path for an A/B comparison. This changes only framework halo
evaluation order across independent targets, not the eighth-order donor
stencil or any MHD/MI formula. On the same eight-thread 48×16×16-per-face
gate, 100 steps took 22.16 s serial versus 14.05 s parallel (`1.58×`), and
300 steps took 58.90 versus 38.88 s (`1.52×`). The complete MHD snapshot,
north/south MI maps, and sampled shell-current outputs were byte-identical
in both A/B runs. The parallel speedup is a local 48-cell result; high-
resolution scaling and the six-rank communication path need separate gates.
The full 3200-step, eight-thread run with parallel planned halos completed
the same 569.409 physical seconds in 368.27 wall seconds (`1.546×` real
time, `1.553×` faster than the 571.78-second serial-planned-halo gate).
Its final MHD, north/south MI and sampled shell-current CSVs were all
byte-identical to that prior 3200-step gate; maximum total-field divergence
remained `1.84287e-12` in code units. These are framework performance and
numerical-parity results, not a production physics acceptance. At this
small angular resolution, 73,728 active cells coexist with 92,160 planned
cell-ghost and 368,256 planned magnetic-face-ghost targets per step, each
using an 8×8 donor stencil. This high guard-zone/active-cell ratio explains
why face-halo work was a major bottleneck. The MPI prototype still uses
replicated active-cell all-gathers and requires a separate communication
optimization before high-resolution scaling.
For a 72×16×16-per-face, 100-step check, the same A/B took 29.57 s with
serial planned halos and 20.55 s with parallel planned halos (`1.44×`);
its MHD, both MI and shell-current outputs were also byte-identical.

#### Time-dependent idealized solar-wind drive

The `earth-mi-drive` development runner uses the same MHD, Boris, aligned-
dipole MI and cubed-sphere boundary routines as `earth-mi-local`. At the
upstream reference plane `GAMERA_CS_EARTH_FRONT_X` (default −12 RE), it
holds IMF Bz at 0 for the first hour, +5 nT for two hours, then −5 nT for
three hours. The existing solar-wind boundary transports that time series
to each boundary point; the switch times refer to the reference plane, not
the outermost radial ghost cell. No timestep is clipped to the switching,
MI, or output cadence. The default end time is 21600 physical seconds and
the output interval is 600 seconds; environment variables
`GAMERA_CS_EARTH_FIRST_SWITCH_SECONDS`,
`GAMERA_CS_EARTH_SECOND_SWITCH_SECONDS`,
`GAMERA_CS_EARTH_STOP_SECONDS`, and
`GAMERA_CS_EARTH_OUTPUT_SECONDS` override these positive times for smoke
tests. The first three must remain strictly increasing.

For a local build, run `earth-mi-drive 300000 /path/to/frame` using the
six-rank MPI executable or the OpenMP executable. The positional number is
the **maximum MHD steps**, not the requested duration. Outputs are an index
`/path/to/frame.index.csv` plus numbered MHD, north/south MI, and sampled
shell-current CSVs. Validate a complete run with
`python3 cubedsphere/tests/check_earth_drive_outputs.py
/path/to/frame.index.csv --radial 72 --angular 16 --stop-seconds 21600
--output-seconds 600`.

The Sugon examples in `cubedsphere/examples/sugon_earth_72/` use a fresh
source copy and run root. Submit `build.slurm`, then `smoke.slurm` with an
`afterok` dependency on the build, then `gate.slurm` after the smoke,
then `run.slurm` after the 1800-second gate. Export `GAMERA_CS_SOURCE_ROOT` (directory containing
`cubedsphere/` and `nonorthogonal/src/`) and `GAMERA_CS_RUN_ROOT` to all
three jobs. The 72×16×16-per-face job uses six MPI ranks, eight OpenMP
threads per rank, and non-exclusive allocation. The smoke compresses the
three phases into 120 seconds to verify the execution/output path; it is
**not** a long-time physical stability test. The 1800-second gate checks
stability through the earlier pressure-rise window before the full run.
The six-hour example is an
experimental physics gate, not yet an accepted production model. This
runner has no restart/resume implementation: preserve all outputs if the
walltime or numerical gate fails, and do not call a finite intermediate
snapshot a completed result.

### Six-rank MPI correctness path

The OpenMP `earth-mi-local` runner above is unchanged. When CMake finds MPI,
it also builds `test_cubedsphere_dipole_background_mpi`. Launch it with
**exactly six ranks**, rank `f` advancing cube face `f`. For example, on a
local Open MPI installation:

```sh
cmake -S cubedsphere -B build-cubedsphere -DCMAKE_BUILD_TYPE=Release
cmake --build build-cubedsphere --target \
  test_cubedsphere_dipole_background \
  test_cubedsphere_dipole_background_mpi
GAMERA_CS_EARTH_RADIAL_CELLS=48 GAMERA_CS_EARTH_INNER_UNIFORM_CELLS=23 \
  GAMERA_CS_EARTH_FRONT_X=-12 \
  OMP_NUM_THREADS=1 mpiexec --oversubscribe --map-by slot -n 6 \
  build-cubedsphere/test_cubedsphere_dipole_background_mpi \
  earth-mi-local 100 earth_mi_mpi_100.csv
```

Omit the Open MPI-specific `--oversubscribe --map-by slot` flags on MPI
installations that do not support them. The MPI time step uses the same
predictor, GAMERA directional sweeps, Boris update, shared edge-EMF/CT and
aligned-dipole MI routines as the OpenMP path; it computes only the owning
face's MHD/CT update. The exact parity gate is
`ctest --test-dir build-cubedsphere -R cubedsphere_earth_mi_mpi_parity
--output-on-failure`. At 48×16×16 per face, the 20- and 100-step MPI MHD,
north/south MI and sampled FAC outputs were byte-identical to the OpenMP
runner, with owner-reduced total-field divergence about `1.8e-12` code units.

This remains a six-rank, replicated-grid implementation: each rank allocates
all six grids, and the MI potential solve runs on every rank. For the
`earth-mi-drive` MPI runner, set `GAMERA_CS_SPARSE_ACTIVE=1` to opt into a
communication-only optimization. The immutable halo plan selects exactly the
active donor cells read by the existing 8×8 reconstruction stencils; full
six-face exchange still occurs before MI updates, output, and global
diagnostics. Cached donor row indices remove repeated index calculations.
No GAMERA reconstruction, MHD flux, Boris, CT, or MI equation changes.

The Sugon 72×16×16×6, 6 MPI×8 OpenMP, 1320-s gate completed with 48/48
MHD/MI/current output files byte-identical to the full-exchange path. On a
busy node, the opt-in sparse/cached version ran at about 1.95× real time;
building that same source with GCC `-O3 -fno-fast-math -flto` using matching
`gcc-ar`/`gcc-ranlib` reached about 2.73× real time on a less-busy node.
These absolute rates depend on node load; matched-node short gates showed
the isolated effects of sparse exchange, donor-index caching, and LTO.
The optimization is opt-in so existing production jobs are unchanged.
Further high-resolution scaling still needs less replicated interface data
and a rank-owned MI solve/broadcast.

#### Resolution-independent tiled MPI work (isolated, not production)

The Earth driver now accepts any radial and per-face angular cell counts of
at least eight rather than a 48/72 or 16/32 whitelist. Its default physical
v4 radial map retains the accepted 72-cell geometry. To reproduce the older
48-cell experiment's 23 uniform inner cells exactly, explicitly set
`GAMERA_CS_EARTH_INNER_UNIFORM_CELLS=23` together with
`GAMERA_CS_EARTH_RADIAL_CELLS=48`. This optional map parameter is independent
of grid size; it is not inferred from a special-case cell count.

The new `cubedsphere_decomposition` and `cubedsphere_mpi_tile` components
derive each tile's active cells, global offsets and MPI neighbors from the
global `radial × angular × angular` grid and three process dimensions. The
Up7 minimum of eight active cells per tile and four ghost cells are
algorithmic requirements, not named resolutions. Mesh vertices use global
logical offsets, so a tile has the same coordinates as its region of an
undivided cube face. Tests cover radial and angular 12-rank exchanges,
48-rank `2×2×2` exchanges, a non-whitelisted `64×24×24` mesh, rectangular
local angular tiles, and sampled `256×256×256` global geometry without
allocating an entire global mesh.

The isolated tiled adapter also resolves every global-face 8×8 cube-face
donor to its radial/angular MPI tile. Remote values use the already accepted
Yin–Yang sparse MPI request/response implementation; same-rank donors stay
local. Current/old conserved state and Cartesian B are exchanged together.
The actual cube-face **cell** ghost fill uses the original interpolation
weights and the original nonphysical-state nearest-donor fallback. Magnetic
face ghosts likewise interpolate Cartesian B and project onto the local face
area vector, as in the original full-face halo. Physical cube-edge magnetic
and MHD face fluxes retain the original orientation and arithmetic average;
a cached global-cell owner map routes values to the correct radial/angular
tile, including unequal angular process counts. The edge exchange uses
cached point-to-point MPI requests because the local Open MPI graph
communicator stalled on the disconnected high-rank edge topology. Tests
exercise smooth-state ghosts and face-flux reconciliation at 12, 24, and 48
ranks, including `72×32×32` with `2×2×2` ranks per face. A separate
owner-sharded physical cube-edge CT EMF plan reproduces the original
endpoint-key, orientation-sign, and arithmetic group mean. Duplicate edges
on internal tile boundaries are canonicalized before this reconciliation,
then the existing internal-edge sync propagates the result to those tiles.
The first tiled GAMERA update calls the existing predictor, MHD/EMF sweeps,
stress, active update, and CT recovery with these exchanges; three-step
uniform-MHD tests pass on small 12-, 24-, and 48-rank layouts. The Earth
radial ghost and EMF functions have been factored so only tiles touching a
physical radial end apply that end's accepted boundary condition. The
existing 6-rank Earth/MI parity gate and non-MPI tests remain the regression
oracle. A new two-step, zero-IMF Earth test reuses the accepted dipole startup,
Boris/HOGS options, SW boundary and strict wall on real 12-rank radial and
24-rank angular tiles; both finish finite with total `div B` around
`6e-13`. A direct six-face oracle comparison found that the first radial
split discrepancy was at a CT edge where an internal radial ghost and a
physical cube-face angular ghost intersected. The initial tile exchange
had not propagated the *refreshed* cube-face ghost through that radial tile
boundary; current/old magnetic face-flux ghost planes were also missing.
The tiled adapter now exchanges those magnetic face-flux layers and
completes the internal cell/face-flux corner exchange after cube-face halo
fill. It does not change any GAMERA flux, EMF, Boris, HOGS, or CT formula.
At `16×16×16` per face, one- and two-step `12`-rank radial, `24`-rank
angular, and `48`-rank `2×2×2` physical Earth runs have exactly the same
`x,y,z,rho,p,v,B` values in all 24,576 active cells as the accepted six-face
oracle; the previously mismatched CT edge EMF is also bitwise identical.
These are short, zero-IMF, no-MI interface-parity gates, not a long-time
Earth/MI stability claim.

For MI, a lightweight worker view holds the original solver's two
hemisphere potentials without a radial MHD grid. The distributed adapter
gathers only the three physical radial cells needed by the existing FAC
curl, builds six *three-layer* grids on rank zero, runs the unchanged
aligned-dipole MI solver there, then broadcasts its potential and cadence.
The gather is indexed by global cell and rank topology, including sample
shells that cross a radial tile boundary. On analytic nonzero residual B,
12-rank radial, 12-rank radial-crossing, and 24-rank angular tests reproduce
the full six-face MI potential **bitwise**. These are MI sampling/solve
regressions, not yet a coupled physical Earth/MI run.

The same distributed MI state is now supplied to the existing Earth radial
ghost-velocity and CT wall-EMF callbacks in a short physical tiled MHD run.
Two-step `72×16×16×6` tests pass at 12 radial-split and 24 angular-split
ranks with finite positive gas state and total `div B` around `1.8e-12`.
The 12-rank physical Earth/MI two-step snapshot was compared cell-by-cell
with the existing six-face serial solver: all 110,592 cell positions,
density, pressure and total magnetic fields match exactly; 132 cells differ
only in velocity roundoff (maximum absolute difference `2.1e-30` in code
units). The northern/southern MI and sampled FAC CSV files are byte-identical.
These steps contain the initial MI solve. A further 64-step six-face-versus-
12-rank comparison crosses the first 10-second coupling update: MHD and
MI/FAC agree to a maximum scaled `1e-12` roundoff tolerance, without
claiming bitwise identity at that later time. A toy 16- or 48-radial-cell
version with the 72-cell radial
map is unsuitable for this particular MI wall test because its deepest
extrapolated ghost cell lies inside the configured 1-RE ionosphere, where the
existing dipole potential mapping correctly rejects the point.

The tiled path builds as `cubedsphere_earth_mpi_drive` only when Parallel
HDF5 is available. `GAMERA_CS_TILE_DRIVE=1` selects the accepted 0/+5/−5 nT
upstream series, with switch, stop, and output times set by
`GAMERA_CS_EARTH_*_SECONDS`. Its production analysis I/O now follows the
accepted Yin–Yang per-patch HDF5 contract: six static grid files, six
collective patch files per analysis time, and one MI HDF5 file. Each rank
writes only its hyperslab; no rank-zero global CSV merge is performed.
See [OUTPUT_DATA_MODEL.md](docs/OUTPUT_DATA_MODEL.md) for schema, units,
the `h5py` reader, and validation.

The 72×32×32×6 Earth/MI solver completed the separate five-hour 0/+5/−5 nT
gate with 48 MPI×2 OpenMP and no dt collapse. That historical run retains
its original CSV output; it has not been deleted or relabeled. The new I/O
path completed an isolated 48-rank, 72×32×32×6 three-stage 30-second gate:
four complete analysis times, 24 patch HDF5 files, six static grid files,
four MI files, no CSV, all fields finite, positive density/pressure, and
total-field divB 7.154e-12. The numerical MHD/CT/cube-interface sources
are byte-identical to the five-hour solver build. The old full-face 8-point
halo plan must never receive a local tile dimension: its interpolation
coordinates are global-face coordinates.

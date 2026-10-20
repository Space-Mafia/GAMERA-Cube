# Model and numerical scope

## Earth example

The computational shell is 2.5–200 Earth radii with an aligned dipole and
the inherited Earth-production-v4 stretched radial map. Six conforming
equiangular gnomonic faces cover the sphere; this is not an overset grid.
Four ghost layers support the existing eight-cell GAMERA reconstruction.
The optimized code retains the MHD fluxes, Boris treatment, CT update and
pressure-control path of the selected Earth baseline.

Solar wind is 5 proton-mass equivalents/cm³, speed **+400 km/s in X**,
sonic Mach 10, gamma 5/3. Pressure is about 0.00803 nPa.
The IMF is **(0,0,−5) nT from t=0** in the new student example.
Upstream is negative X and downstream positive X in this implementation:
do not label these axes as conventional GSM without an explicit transformation.
The initialization retains the accepted front at X=−30 RE, width 1.5 RE,
ambient plasma and cut-dipole transition (radius 16 RE, length 8 RE).
It is NOT a uniform-flow-through-the-entire-domain initialization.
The initialized upstream field and the driving series use the same −5 nT.

MI coupling uses 192 × 22 points per hemisphere, a 10 s cadence, a
1 RE ionosphere, and constant Pedersen conductance 5 S.
It maps along an aligned dipole. This cube implementation does not include
the Yin–Yang production model's full precipitation/conductance treatment:
no Hall conductance or precipitation is implied.
The inner boundary retains the strict gas-wall and MI electric-field
feedback. Outer ghost/EMF conditions retain the inherited advected-wind and
outflow treatment. Pressure control is enabled explicitly by the launcher.

## Optimizations enabled by the supported launcher

1. Compact halo interpolation.
2. Donor-index caching.
3. Block halo access.
4. OpenMP halo interpolation.
5. EMF-face caching.
6. Primitive-state caching.

They change data access/reuse, not reconstruction coefficients, CFL or
physical equations. The inherited A/B tests check numerical output equality;
do not multiply gains from separate allocations or assume identical speedups
on a laptop and a cluster. LTO is not enabled by the supported build recipe.

## Intentional limits

This is single-fluid ideal MHD with numerical dissipation, not kinetic or
multi-fluid plasma physics. No formal seventh-order end-to-end convergence
is claimed merely from the reconstruction stencil. No arbitrary-decomposition
bitwise identity, exact physical steady state, reconnection-rate convergence,
or operational forecasting accuracy is claimed. Analysis snapshots are float32;
geometry/normalizations are float64. Output rounding matters in comparisons.

Mars, Venus and Io require different boundary conditions and source physics.
They are future model adapters, not supported merely by changing a planet name.

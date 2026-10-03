# Single-particle T-matrices

This module turns Mie coefficients into T-matrices for multilayer chiral
spheres and cylinders, and measures their helicity response.

[sphere.rs](sphere.rs) places one coefficient block per spherical degree into
the full matrix. [cylinder.rs](cylinder.rs) places blocks by axial wavenumber
and azimuthal order, reusing mirrored boundary solves when possible.
[metric.rs](metric.rs) computes circular dichroism, duality breaking and
electromagnetic chirality. Each operation also provides analytic gradients.

The [coeffs](../coeffs/README.md) module supplies boundary coefficients in
negative-positive helicity order. T-matrices reverse that order to match the
basis: polarization 1 precedes 0. [mod.rs](mod.rs) defines these conventions;
[cluster](../cluster/README.md) couples the resulting particle matrices.

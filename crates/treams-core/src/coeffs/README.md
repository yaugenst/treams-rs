# Particle coefficients and materials

This module computes Mie coefficients for concentric chiral spheres and
cylinders. Material parameters and boundary sizes enter a sequence of
interface calculations; the result is a 2 × 2 helicity coefficient matrix.

[material.rs](material.rs) defines isotropic reciprocal materials, refractive
indices and layer checks. [mie.rs](mie.rs) handles spheres and
[mie_cyl.rs](mie_cyl.rs) handles cylinders. Both save interface states and
radial functions for analytic gradients, avoiding another Bessel evaluation
when differentiating.

Coefficient matrices order helicities as negative, then positive.
[tmatrix](../tmatrix/README.md) reverses both axes when placing them into the
basis order. Planar Fresnel coefficients live in
[smatrix](../smatrix/README.md). [mod.rs](mod.rs) lists the public functions.

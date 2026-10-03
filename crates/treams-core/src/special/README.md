# Special functions

This module supplies the functions used by wave expansions, lattice sums and
scattering coefficients, together with their analytic derivatives.

[bessel.rs](bessel.rs) evaluates radial functions.
[legendre.rs](legendre.rs) and [ferrers.rs](ferrers.rs) handle associated
Legendre functions, including real non-integer degrees.
[wigner.rs](wigner.rs) computes angular-momentum coupling and rotations.
[harmonics.rs](harmonics.rs) evaluates Cartesian solid harmonics.
[integrals](integrals/README.md) supplies incomplete gamma and Kambe functions
for Ewald sums. [coordinates.rs](coordinates.rs) transforms points and vector
components; [polarization.rs](polarization.rs) combines helicity and parity
waves.

[mod.rs](mod.rs) collects the public functions and defines label limits.
Radial evaluations retain first and second derivatives, which the vector-wave
calculations need. Reference values live in
[references](../../references/README.md).

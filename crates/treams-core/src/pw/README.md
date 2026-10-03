# Plane waves

This module computes plane-wave fields, translation phases, changes of Cartesian
axes, and regular spherical or cylindrical expansions, with analytic gradients.

- [polarization.rs](polarization.rs) derives polarization vectors and the normal
  wavenumber from a complex wavevector.
- [field.rs](field.rs) combines polarization with the phase `exp(i k·r)` to evaluate
  fields and translations.
- [expand.rs](expand.rs) combines angular coefficients with position phases to
  build expansion matrices.
- [permute.rs](permute.rs) computes polarization changes under cyclic axis
  permutations.

Complex wavevectors support evanescent waves. Their wavenumber is
`sqrt(kx² + ky² + kz²)`, without complex conjugation. [mod.rs](mod.rs) collects the
public functions and defines polarization and gradient conventions.

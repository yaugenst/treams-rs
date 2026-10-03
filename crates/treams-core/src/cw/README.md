# Cylindrical waves

This module translates cylindrical waves, builds ordinary and periodic expansion
matrices, and converts regular cylindrical waves to spherical waves.

- [mod.rs](mod.rs) defines mode labels, bases and the transverse wavenumber branch.
- [polar.rs](polar.rs) and [cartesian.rs](cartesian.rs) compute translation
  coefficients and derivatives in their respective coordinates.
- [expansion.rs](expansion.rs) assembles matrices, reusing coefficients shared by
  multiple entries.
- [to_sw.rs](to_sw.rs) combines angular coefficients and translations for conversion
  to a spherical basis.

Matrices map source coefficients to destination coefficients. Cylindrical
translations couple only modes with equal axial wavenumbers and polarization
indices. Ordinary translations use `destination - source`; lattice sums use the
opposite displacement. The transverse wavenumber has nonnegative imaginary part.

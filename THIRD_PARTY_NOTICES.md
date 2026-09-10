# Third-party notices

This project ports numerical algorithms and conventions from
[treams](https://github.com/tfp-photonics/treams), released under the MIT license.
The upstream copyright and license are retained in `LICENSE.treams`.

Scientific reference: D. Beutel, I. Fernandez-Corbaton, and C. Rockstuhl,
*treams — a T-matrix-based scattering code for nanophotonics*, Computer Physics
Communications 297, 109076 (2024).

Repository tooling and the framework-neutral derivative boundary follow the
Photonoodle project. The numerical core uses `complex-bessel` for
complex Bessel functions, `nalgebra` for small matrices, and `faer` with
`rayon` for dense CPU algebra and parallel particle coupling; see Cargo.lock for
the exact dependency versions.

Ewald lattice sums and Kambe recurrences follow the MIT-licensed treams
implementation and its cited scientific references. The incomplete-gamma
continued fraction and exponential-integral series follow DLMF 8.9 and 6.6;
complex complementary error functions use the MIT-licensed `errorfunctions`
crate (a Rust implementation of Steven G. Johnson's Faddeeva algorithms).

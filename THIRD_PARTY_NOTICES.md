# Third-party notices

This project ports numerical algorithms and conventions from
[treams](https://github.com/tfp-photonics/treams), released under the MIT license.
The upstream copyright and license are retained in `LICENSE.treams`.

The examples in `docs/examples/upstream/` and the descriptions of the example
pages adapt the MIT-licensed treams documentation examples (`LICENSE.treams`).

Scientific reference: D. Beutel, I. Fernandez-Corbaton, and C. Rockstuhl,
*treams — a T-matrix-based scattering code for nanophotonics*, Computer Physics
Communications 297, 109076 (2024).

The numerical core uses `complex-bessel` for complex Bessel functions, `nalgebra`
for small matrices, and `faer` with `rayon` for dense CPU algebra and parallel
particle coupling; see Cargo.lock for the exact dependency versions.

Ewald lattice sums and Kambe recurrences follow the MIT-licensed treams
implementation and its cited scientific references. The incomplete-gamma
continued fraction and exponential-integral series follow DLMF 8.9 and 6.6;
complex complementary error functions use the MIT-licensed `errorfunctions`
crate (a Rust implementation of Steven G. Johnson's Faddeeva algorithms).

Chiral Fresnel coefficients and plane-wave power-flux conventions are ported from
the MIT-licensed treams implementation. Native S-matrix composition uses the
Redheffer product, eliminating both internal fields with one LU factorization and
reusing it in the analytic adjoint.

Real-degree Ferrers/associated Legendre functions adapt SciPy XSF's `specfun::lpmv`
and `lpmv0` endpoint series and recurrence, with scaled high-order arithmetic and
analytic argument derivatives. The BSD-3-Clause SciPy developer notice and the
original Zhang/Jin attribution are retained in `LICENSE.xsf`. XSF and SciPy are
not runtime dependencies.

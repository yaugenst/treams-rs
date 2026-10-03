# Third-party notices

This project ports numerical algorithms and conventions from
[treams](https://github.com/tfp-photonics/treams), released under the MIT license.
The upstream copyright and license are retained in `LICENSE.treams`.

The examples in `docs/examples/upstream/` and the descriptions of the example
pages adapt the MIT-licensed treams documentation examples (`LICENSE.treams`).

Scientific reference: D. Beutel, I. Fernandez-Corbaton, and C. Rockstuhl,
*treams — a T-matrix-based scattering code for nanophotonics*, Computer Physics
Communications 297, 109076 (2024).

The numerical core uses `complex-bessel` for
complex Bessel functions, `nalgebra` for dense and small matrices, and `faer` with
`rayon` for dense CPU algebra and parallel particle coupling; see Cargo.lock for
the exact dependency versions. Release wheels statically link these crates and
their dependencies; `THIRD_PARTY_LICENSES.txt` carries each linked crate's
license text, and the `RUST_STDLIB_*` files carry the Rust standard-library
notices (`just licenses` regenerates them). treams-rs itself is MIT licensed;
the linked crates are under permissive licenses that require keeping their
notices: Apache-2.0 (nalgebra, simba, approx), BSD-2-Clause (numpy,
atomic-wait), the Rust standard library's MIT and Unicode-3.0 terms, and the
MIT option of every dual-licensed crate. faer's ports of Eigen (MPL-2.0) and
LAPACK routines keep their notices in the same file. Keep the wheel's
`.dist-info/licenses` files when redistributing it.

Ewald lattice sums and Kambe recurrences follow the MIT-licensed treams
implementation and its cited scientific references. The incomplete-gamma
continued fraction and exponential-integral series follow DLMF 8.9 and 6.6;
complex complementary error functions use the MIT-licensed `errorfunctions`
crate (a Rust implementation of Steven G. Johnson's Faddeeva algorithms).

Chiral Fresnel coefficients and plane-wave power-flux conventions are ported from
the MIT-licensed treams implementation. Native S-matrix composition uses the
Redheffer product, eliminating both internal fields with one LU factorization and
reusing it in the analytic adjoint.

Real-degree Ferrers functions are an original implementation of the DLMF
formulas cited in `crates/treams-core/src/special/ferrers.rs` (hypergeometric series
§14.3/§15.8, degree recurrences §14.10 and connection formulas §14.9).

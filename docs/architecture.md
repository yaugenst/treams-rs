# Architecture

The Rust core is independent of Python and autodiff frameworks. It owns special
functions, scattering coefficients, wave translation, lattice sums, scattering
solves, observables, and the corresponding analytic derivatives.

The PyO3 crate validates array dimensions and physical inputs at entry, releases
the GIL for numerical work, and transfers arrays and opaque residuals. Python
provides typed physics objects and convenient operations. During development,
`treams_rs` coexists with the upstream `treams` reference package. Public naming,
normalization, helicity ordering, and basis ordering follow treams. Full import
compatibility is a release decision after numerical and API parity is verified.

The differentiation contract is `forward(parameters) -> (outputs, residual)`
and `pullback(residual, output_cotangents) -> input_cotangents`. The native
convention is the real pairing `dL = Re(sum(conj(g) * dx))`. Framework adapters
perform their own convention conversion. No dense parameter Jacobian is part
of this contract. Static mode counts and discrete topology are not differentiable.

First-order derivatives are implemented analytically at numerical boundaries.
Finite differences and upstream treams are test oracles, never production
fallbacks. The first target is CPU execution; GPU execution and higher-order AD
require separate implementations rather than an implicit framework promise.

Reference: tfp-photonics/treams commit
`1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39` (2026-08-24).

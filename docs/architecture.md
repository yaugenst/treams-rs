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
fallbacks. Advect, JAX and PyTorch adapt this same first-order boundary. JAX
callbacks recompute the native residual during reverse evaluation; PyTorch
retains it for the first backward and recomputes it for repeated backwards.
Neither framework adapter provides GPU tensor execution or higher-order AD.

Dense requested-illumination solves factor `I - T C` once and solve only the
requested right-hand sides. Immutable factors can be shared across independently
owned residuals. The matrix-free sphere solver instead applies pair translations
on demand and uses restarted GMRES for both the forward and adjoint systems.
It contracts geometry/material cotangents per pair, avoiding a global coupling
matrix and its gradient. Dense and iterative paths share Mie coefficients,
translation plans and the native real-pairing convention.

`treams-wasm` uses the same core with serial dense algebra and selected JavaScript
exports. `treams-cuda` and `treams-cuda-tile` are optional crates: CPU/WASM builds
contain neither GPU dependency. cuBLAS/cuSOLVER supply complex128 dense algebra;
cuTile supplies the fused plane-wave field kernel. Operator assembly, wave
polarization and equilibration remain shared core mathematics. GPU memory
ownership and transfers are explicit, and no backend silently lowers precision.

Reference: tfp-photonics/treams commit
`1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39` (2026-08-24).

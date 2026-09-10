# Test strategy

Rust tests are the primary numerical contract. They run without Python. Use
proptest for bounded physical domains, algebraic identities, and adjoint checks.
Hypothesis exercises the installed Python/native boundary, broadcasting and
strides, parameter validation, opaque residuals, and complete user workflows.

Use independent evidence:

- Physical invariants: zero-contrast scattering, lossless optical theorem,
  passivity where applicable, invariance under global translation, and invariance
  under splitting a homogeneous layer into identical sublayers.
- Analytic identities: Bessel Wronskians and spherical differential equations,
  Wigner selection rules and orthogonality, and matrix residual equations.
- Differentiation: arbitrary cotangents and parameter directions, complex
  real-pairing adjoint identities, directional finite-difference convergence,
  radius/material/position/wavenumber sensitivities, and residual consumption.
- Independent numerical oracle: pinned upstream treams plus SciPy special
  functions. Oracle agreement is additional evidence, not the sole definition
  of physical correctness.

Use finite bounded input domains reflecting each property's assumptions. Keep
shrinking enabled, save regression cases when a property exposes a bug, and
include polar-axis and small-argument cases explicitly. Never loosen a numerical
tolerance merely to make a failing implementation pass. Benchmark only optimized
builds, separately from correctness checks.

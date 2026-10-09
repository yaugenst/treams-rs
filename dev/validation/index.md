# Validation

Rust tests are the main numerical check and run without Python. They use
proptest to sample bounded physical domains and check algebraic and adjoint
identities. Hypothesis tests exercise the installed Python package: array shapes
and memory layouts, input validation, reusable derivative contexts,
and complete user workflows.

## Kinds of evidence

- **Physical invariants**: zero-contrast scattering, lossless energy balance
  (unitarity, which implies the optical theorem), reciprocity, mirror and duality
  symmetries, passivity, invariance under global translation, covariance under
  rotation, and invariance under splitting a homogeneous layer.
- **Analytic identities**: Bessel Wronskians, differential equations and
  reflection formulas, Legendre recurrences, Wigner selection rules and
  orthogonality, independent algorithms for one function (Wigner d by recurrence
  and by the generator exponential), the group laws of translations, split, Bloch
  and primitive-basis independence of Ewald sums, and matrix residual equations.
- **Differentiation**: arbitrary cotangents (gradients with respect to an output)
  and parameter directions, complex real-pairing adjoint identities, directional
  finite-difference convergence and reusable derivative contexts. Symmetries give
  exact identities of complete pullbacks without finite differences: Euler scaling of
  lengths against wavenumbers, and similarity or unitary orbits of spectra.
  Framework adapters reproduce the native pullback exactly
  (`tests/autodiff/test_adjoint_identities.py`,
  `tests/autodiff/test_adapter_bridges.py`).
- **Independent references**: fixed versions of treams, SciPy and mpmath.
  Agreement adds evidence but does not define correctness.

Property tests sample finite domains that match each property's assumptions,
reduce failing inputs to simpler examples and save them for future tests.
Polar-axis, origin and small-argument points are added explicitly. No numerical tolerance is loosened to
make an implementation pass. Benchmarks run only on optimized builds, separately
from correctness checks.

## Rust properties and proofs

Hosted CI checks Rust on the minimum and stable toolchains and Python on
CPython 3.12–3.15. The complete reference suite runs on 3.12 and 3.13; later
versions omit tests whose reference dependencies are unavailable
([development](../development/index.md)). Native proptest
properties, mostly in `crates/treams-core/src/properties/`, check:

- the group laws of translations;
- Bessel, Legendre and Wigner identities, including the Wigner small-d recurrence
  against the generator exponential up to degree 128;
- split, Bloch, primitive-basis and point-group invariances of every Ewald family,
  in value and every derivative;
- Lorentz reciprocity of particles and of dense, heterogeneous, matrix-free
  (`iterative::tests`) and cylindrical clusters;
- the lossless energy balance of sphere and cylinder T-matrices and of sphere
  clusters;
- EBCM against Mie;
- implicit-adjoint closed forms of the solves against an independent LU;
- composition, reciprocity, unitarity and layer-split invariance of planar stacks.

Finite differences are used only as test references, never to compute pullbacks.

[Lean proofs](../design/formal-proofs.md) establish, over exact arithmetic, that:

- diffraction-order enumeration is complete and duplicate-free;
- lattice shells partition the integer lattice;
- the translation degree loop matches the Wigner 3j selection rules;
- equilibrated LU solves are exact solves;
- the dense requested-illumination pullback is the exact derivative.

The proofs do not cover rounding. In the rounding case that the diffraction-order
proof exposed, a cutoff equal to the magnitude of an order, the enumeration keeps
that order (`diffraction_cutoff_on_an_order_keeps_it`). The proofs cover
hand-written models, not the Rust source; Rust tests compare `cube`, `degrees` and
`harmonics` with outputs generated from those models.

## Accuracy evidence

Errors are relative unless stated otherwise. The workflow comparisons and the
2,032 finite-output reference cases come from one recorded Linux run (Ryzen 9
9950X, Python 3.13.1); the [performance](../performance/index.md) page lists
its build.

Development tests compare against treams 0.4.7. Archived workflow and gallery
measurements retain their recorded treams 0.4.5 identity. The results below
belong to those measurements, not automatically to a later release build.

| Evidence | Scope | Outcome | Source |
| --- | --- | --- | --- |
| Rust property tests | Physical, analytic and adjoint identities above, on bounded random domains with saved regressions | Pass in CI | [`crates/treams-core/src/properties/`](https://github.com/yaugenst/treams-rs/tree/e464d5fea94749223e22fe80c2ba5a53b7426950/crates/treams-core/src/properties) |
| Python Hypothesis and reference tests | Broadcasting, strides, validation, derivative contexts and workflows through the installed extension, against SciPy, treams and mpmath | Pass in CI on Python 3.12 and 3.13 | [`tests/special/`](https://github.com/yaugenst/treams-rs/tree/e464d5fea94749223e22fe80c2ba5a53b7426950/tests/special), [`tests/lattice/`](https://github.com/yaugenst/treams-rs/tree/e464d5fea94749223e22fe80c2ba5a53b7426950/tests/lattice), [`tests/waves/`](https://github.com/yaugenst/treams-rs/tree/e464d5fea94749223e22fe80c2ba5a53b7426950/tests/waves), [`tests/tmatrix/`](https://github.com/yaugenst/treams-rs/tree/e464d5fea94749223e22fe80c2ba5a53b7426950/tests/tmatrix), [`tests/smatrix/`](https://github.com/yaugenst/treams-rs/tree/e464d5fea94749223e22fe80c2ba5a53b7426950/tests/smatrix), [`tests/plane/`](https://github.com/yaugenst/treams-rs/tree/e464d5fea94749223e22fe80c2ba5a53b7426950/tests/plane), [`tests/autodiff/`](https://github.com/yaugenst/treams-rs/tree/e464d5fea94749223e22fe80c2ba5a53b7426950/tests/autodiff) |
| High precision: incomplete gamma | 1494 40-digit mpmath values; 60-digit sweeps over \|n\| <= 128 | Within 1e-13 relative; sweeps within 1.5e-13 | [`incgamma.txt`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/crates/treams-core/references/incgamma.txt), [limits](numerical-limits.md#special-functions-and-waves) |
| High precision: Kambe integrals | 181 quadrature values; 1249 70-digit values at the lattice-sum arguments | 1e-13 relative; even orders within 1e-11 | [`kambe.txt`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/crates/treams-core/references/kambe.txt), [`kambe_lattice.txt`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/crates/treams-core/references/kambe_lattice.txt) |
| High precision: lattice sums | Ewald sums and derivatives against mpmath, near and off the plane or axis | Values within 5e-14 to 2e-13 near the plane or axis; 1D spherical sums off the axis within 1.4e-13 | [`lattice_sums.txt`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/crates/treams-core/references/lattice_sums.txt), [`lattice_chain.txt`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/crates/treams-core/references/lattice_chain.txt), [`generate_references.py`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/scripts/generate_references.py) |
| High precision: fractional Legendre, archived pre-release implementation | 530 finite value and derivative cases and two expected overflows against 70-digit hypergeometric values | Largest relative error 2.17e-13 in that record | [`qualify_legendre.py`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/scripts/qualify_legendre.py), [record](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/results/fractional-legendre-physical.json) |
| High precision: reference collector | 2,032 finite-output reference cases, including a metallic sphere of size parameter 80; 18 inputs beyond the float64 range are excluded | All pass | [`qualify_references.py`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/scripts/qualify_references.py), [`linux-core-qualification.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/linux-core-qualification.json) |
| treams workflow comparisons | 527 complete workflows against treams 0.4.5, plus 51 complete-gradient cases | All pass | [`qualify_upstream.py`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/scripts/qualify_upstream.py), [`linux-core-qualification.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/linux-core-qualification.json) |
| Published applications | Electron-beam spectra, treams paper spectra and thermal radiation | Within the stated tolerances of the author data, for example all 300 thermal absorption values within 0.577% | [Published applications](published-applications.md), [`benchmarks/papers/`](https://github.com/yaugenst/treams-rs/tree/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/papers) |
| Lean proofs | The five statements above, over exact arithmetic | Proved; Rust tests compare `cube`, `degrees` and `harmonics` with model outputs | [formal proofs](../design/formal-proofs.md) |
| Floating-point environment | Every scalar binding and ufunc loop, records with their pullbacks, solves and a slab on a thread that flushes subnormals to zero | Equal to the results of an IEEE thread, bit for bit | [`test_float_environment.py`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/tests/bindings/test_float_environment.py), [`float_environment.py`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/scripts/float_environment.py), [floating-point environment](../design/floating-point.md) |
| Thread count | Values and gradients of every parallel reduction, dense solves and decompositions, a cluster solve and ufuncs at one, two and three threads; the Python suite on one thread and on every CPU | Equal at every thread count, bit for bit | [`test_thread_pool.py`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/tests/bindings/test_thread_pool.py), [parallelism](../design/parallelism.md) |

[Testing](../development/testing.md) explains how to add a test.

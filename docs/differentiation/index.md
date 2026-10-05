---
description: First-order forward and reverse differentiation with analytic Rust pushforwards and pullbacks.
---

# Differentiation

treams-rs computes first-order forward and reverse derivatives in Rust,
analytically. Every
differentiable function is a record:

> A record is a function that returns a value and a reusable context. The context stores what is needed to compute derivatives later: `context.pullback(g)` takes the gradient `g` of a real-valued loss with respect to the value and returns the gradients with respect to the inputs, one for each differentiable input, in the order of the arguments. Gradients follow the convention dL = Re Σ conj(g)·dx.

The same context also provides `context.pushforward(*input_tangents)`: a
Jacobian–vector product (JVP), which maps an input perturbation to the resulting
output perturbation. Pass one tangent per dynamic input, in pullback order.
The output tangent has the primal output's shape and tuple structure. Both
derivative directions can be called repeatedly on the same context.

The pullback maps the gradient with respect to the output, called the
cotangent, back to the inputs. The
[glossary](../reference/glossary.md#records-and-gradients) defines record,
context, pullback and cotangent for the whole project.

```python exec
import numpy as np
from treams_rs import diff
from treams_rs.testing import check_pullback, check_pushforward

# One sphere of size parameter x = k0 r = 0.6, then the vacuum around it.
x = np.array([0.6])
epsilon = np.array([4.0 + 0.1j, 1.0])
mu = np.array([1.0 + 0j, 1.0])
kappa = np.array([0j, 0j])

coefficients, context = diff.mie(1, x, epsilon, mu, kappa)

# The loss L = sum |a|^2 has the gradient g = 2 a with respect to the
# coefficients a. The pullback returns one gradient per input, in order.
d_x, d_epsilon, d_mu, d_kappa = context.pullback(2 * coefficients)
assert d_x.shape == x.shape and d_epsilon.shape == epsilon.shape

# Compare the pullback with central differences. The degree 1 stays fixed.
check_pullback(lambda *inputs: diff.mie(1, *inputs), x, epsilon, mu, kappa)

# Sensitivity of every coefficient to the size parameter, with materials fixed.
tangent = context.pushforward(
    np.ones_like(x), np.zeros_like(epsilon), np.zeros_like(mu), np.zeros_like(kappa)
)
assert tangent.shape == coefficients.shape
check_pushforward(lambda *inputs: diff.mie(1, *inputs), x, epsilon, mu, kappa)
```

The [`diff` reference](../reference/python/diff.md) lists every record with the
order of its dynamic inputs. [Gradient checks](gradient-checks.md) explains
`check_pullback` and `check_pushforward`.

Forward mode is useful for a few input directions and many outputs, such as an
entire field map's sensitivity to radius or frequency. Reverse mode is useful
for a few scalar objectives and many design parameters. Both compute first
derivatives; combining them does not enable second derivatives of native calls.

## Rules

- **Records.** A record returns `(value, context)`. Module-level records live
  only in `treams_rs.diff`. Objects that record have a method named `record`:
  `InteractionFactor.record` and `iterative.SphereCluster.record`.
- **Gradient order.** `context.pullback(g)` returns the gradients of the
  differentiable inputs in the order of the forward arguments:
  `diff.sphere_cluster(lmax, k0, radii, epsilon, positions)` returns the
  gradients `(k0, radii, epsilon, positions)`.
- **Tangent order.** `context.pushforward(k0, radii, epsilon, positions)` takes
  tangents in that same order. Each tangent has its input's original shape;
  broadcasting follows the primal call. Real inputs require real tangents.
- **Pairing.** The gradient `g` of a complex value `x` satisfies
  `dL = Re(sum(conj(g) * dx))`, in NumPy `np.vdot(g, dx).real`. Its real part is
  the derivative with respect to `Re x`, its imaginary part the derivative with
  respect to `Im x`. A real input gets a real gradient.
- **Fixed inputs.** Mode labels, cutoffs such as `lmax`, basis sizes, quadrature
  nodes, material topology and the order of eigenvalues have no gradient. A
  record takes them as integers or keywords, not as differentiable inputs.
- **Reuse.** Pushforwards and pullbacks borrow the same immutable residual.
  Call either method repeatedly for different directions or cotangents.
- **Shape checks first.** A tangent or cotangent of the wrong shape, or with
  non-finite entries, raises `ValueError` and leaves the context unchanged.
- **Ownership.** The context owns copies of the data it needs. Changing the input
  arrays after the forward call leaves the gradients unchanged.
- **First order on the CPU.** Pushforwards and pullbacks give first derivatives
  on the CPU. Higher derivatives remain unsupported.

```python exec
import numpy as np
import pytest
from treams_rs import diff

operator = np.array([[2.0 + 0.1j, 0.2], [0.1j, 3.0]])
solution, context = diff.solve(operator, np.ones((2, 1), complex))

with pytest.raises(ValueError, match="shape"):
    context.pullback(np.ones((3, 1)))  # the context stays usable
d_operator, d_rhs = context.pullback(np.ones_like(solution))
again_operator, again_rhs = context.pullback(np.ones_like(solution))
np.testing.assert_allclose(again_operator, d_operator)
np.testing.assert_allclose(again_rhs, d_rhs)
tangent = context.pushforward(np.zeros_like(operator), np.ones_like(solution))
np.testing.assert_allclose(tangent, np.linalg.solve(operator, np.ones_like(solution)))
```

Cylindrical records expose `pushforward_axial`, appending the axial-wavenumber
tangent just as `pullback_axial` appends its gradient. Factors built from
particle blocks expose `pushforward_blocks`, whose first input is a sequence
of block tangents. The corresponding ordinary methods cover fixed axial
wavenumbers and dense local matrices.

[Analytic pullbacks](../design/pullbacks.md) gives the reasons for these rules.

## Frameworks

Advect, JAX, PyTorch and HIPS Autograd differentiate whole objectives built
from physics objects. Use the ordinary `treams_rs` API: framework inputs select
their adapter before conversion, and the adapter calls these records underneath.
Plain Python and NumPy inputs retain NumPy behavior. Explicit adapter namespaces
remain available for expert operations and custom records.
[Framework adapters](frameworks.md) shows each framework, and
[custom records](custom-records.md) shows how to wrap a record for JAX, PyTorch or
HIPS Autograd.

## Points without a gradient

- **Normal incidence.** Observables with a smooth limit, such as cross sections
  and port powers, have limiting pullbacks there.
- **Polarization axis.** The derivative of a plane-wave direction is undefined
  on the polarization axis. Keep the direction fixed when other inputs vary.
- **Thresholds and degeneracies.** Exact diffraction thresholds, a change of the
  direct-shell grouping in a lattice sum and individual degenerate eigenmodes
  have no smooth derivative.
- **Spectral outputs.** Full eigensystem JVPs require distinct eigenvalues and
  unique phase pivots. Individual singular-value JVPs require distinct positive
  values. Some smooth spectral sums still have pullbacks at degeneracies.

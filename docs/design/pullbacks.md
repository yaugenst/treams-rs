---
description: Analytic first-order pushforwards and pullbacks, shared residuals and the complex-gradient convention.
---

# Analytic pushforwards and pullbacks

Every differentiable operation of treams-rs has a record:

> A record is a function that returns a value and a reusable context. The context stores what is needed to compute derivatives later: `context.pullback(g)` takes the gradient `g` of a real-valued loss with respect to the value and returns the gradients with respect to the inputs, one for each differentiable input, in the order of the arguments. Gradients follow the convention dL = Re Σ conj(g)·dx.

The same residual supports a pushforward, `context.pushforward(*tangents)`,
which maps input directions to output directions. Both actions can reuse the
context. Forward and reverse mode share local analytic derivatives, numerical
factors and physical conventions; they do not share a materialized Jacobian.

[Differentiation](../differentiation/index.md) shows how to use records; the
[glossary](../reference/glossary.md) defines pullback, context and cotangent (the
gradient with respect to an output).

```python exec
import numpy as np
from treams_rs import diff

operator = np.array([[2.0 + 0.1j, 0.2], [0.1j, 3.0]])
x, context = diff.solve(operator, np.ones((2, 1), complex))
d_operator, d_rhs = context.pullback(np.ones_like(x))  # forward order
dx = context.pushforward(np.zeros_like(operator), np.ones_like(x))
np.testing.assert_allclose(dx, np.linalg.solve(operator, np.ones_like(x)))
again_operator, again_rhs = context.pullback(np.ones_like(x))
np.testing.assert_allclose(again_operator, d_operator)
np.testing.assert_allclose(again_rhs, d_rhs)
```

## Why analytic

- **Accuracy.** Analytic derivatives avoid the step-size choice and subtraction
  error of finite differences. Their accuracy still depends on the conditioning
  of the forward and adjoint problems and on floating-point rounding.
- **Cost.** A pushforward computes one input direction, and a pullback computes
  one objective cotangent. Their work depends on the numerical operation,
  without multiplying by the number of design parameters. Computing a full
  Jacobian still requires many directions. Framework callbacks can add extra
  forward evaluations; the [adapter contracts](adapters.md#how-long-a-context-lives)
  make that cost explicit.
- **Reuse.** Both derivative directions reuse work from the primal call. For
  `A X = B`, the pushforward solves `A dX = dB - dA X`, and the pullback solves
  the adjoint system. Both use the recorded LU factors. Requested-illumination
  records use the factor of `I - T C` in the same way. The derivative does not
  differentiate the factorization algorithm.

## Why a value and a reusable context

- **Ownership.** The context owns copies of what its derivatives need. Changing the
  input arrays after the forward call does not change the gradients.
- **No global registry.** Nothing outside the context remembers a forward call.
  Python, or the framework that holds the context, decides how long it lives, and
  nothing leaks when a context is dropped unused.
- **Memory.** The context keeps its immutable residual, such as a dense LU
  factor, for its lifetime. Rust derivative methods borrow it, allowing repeated
  directions and cotangents without another factorization. Dropping the context
  releases that data. Invalid tangent or cotangent shapes and non-finite entries
  raise without changing the residual.

Frameworks control the lifetime of their saved context; their transform and
recomputation behavior is described in [framework adapters](adapters.md).

## Why the real pairing

The loss is real and most inputs are complex. With the pairing defined above,
the gradient of a complex input `x` is one complex number: its real part is the
derivative with respect to `Re x` and its imaginary part the derivative with
respect to `Im x`. For a real input it is the ordinary real gradient. Advect and
PyTorch use the same rule. JAX and HIPS Autograd pair complex numbers without the
conjugate, so their adapters conjugate the cotangent and the gradients around
every pullback. Input tangents are ordinary increments `dx`, so pushforwards
need no convention conversion in any framework. A real input has a real tangent.

## Inputs, labels and order

- **Forward order.** A pullback returns one gradient per differentiable input, in
  the order of the forward's arguments; a pushforward takes tangents in that
  same order. A docstring states any other order. Input tangents have their
  original shapes, and the output tangent follows the primal shape and tuple
  structure. Broadcasting remains the native record's responsibility.
- **Static labels.** Degrees, orders, polarization indices, `lmax`, basis sizes,
  quadrature nodes and lattice dimensions are integers or fixed choices. They have
  no gradient and stay outside the record's inputs. Continuous values that the
  numerical method chooses, such as the Ewald split parameter, are held fixed
  too.
- **No dense Jacobians.** A pushforward contracts a direction with local
  derivatives; a pullback contracts a cotangent with their real adjoint.
  The Jacobian of an M × M interaction matrix with respect to 3N
  positions has 3N M² entries: 1.7e11 complex numbers, or 2.8 TB, for 100 spheres
  at `lmax = 10`. No record forms it.

## What treams-rs does not do

- **No finite-difference fallback.** Missing derivative methods are unsupported;
  adapters never approximate them or reconstruct a full Jacobian from another
  derivative direction. Finite differences appear only in tests, as an independent check
  (see [gradient checks](../differentiation/gradient-checks.md)).
- **First order only.** Pushforwards and pullbacks give first derivatives.
  Differentiating either again, including forward-over-reverse Hessian products,
  remains unsupported. Fixed-size jets (`numerics::Jet`) provide local first
  derivatives inside Rust; adapters expose analytic directional contractions.
- **No derivative across singularities.** Full eigenpair pushforwards require
  distinct eigenvalues and unique phase pivots; singular-value pushforwards
  require distinct positive values. Smooth spectral sums can have pullbacks
  where those individual output derivatives do not exist.

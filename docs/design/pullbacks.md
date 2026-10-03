---
description: Analytic first derivatives, one-use gradient contexts and the complex-gradient convention.
---

# Analytic pullbacks

Every differentiable operation of treams-rs has a record:

> A record is a function that returns a value and a context. The context stores what is needed to compute gradients later and can be used once: `context.pullback(g)` takes the gradient `g` of a real-valued loss with respect to the value and returns the gradients with respect to the inputs, one for each differentiable input, in the order of the arguments. Gradients follow the convention dL = Re Σ conj(g)·dx.

[Differentiation](../differentiation/index.md) shows how to use records; the
[glossary](../reference/glossary.md) defines pullback, context and cotangent (the
gradient with respect to an output).

```python exec
import numpy as np
import pytest
from treams_rs import diff

operator = np.array([[2.0 + 0.1j, 0.2], [0.1j, 3.0]])
x, context = diff.solve(operator, np.ones((2, 1), complex))
d_operator, d_rhs = context.pullback(np.ones_like(x))  # forward order
with pytest.raises(ValueError, match="consumed"):
    context.pullback(np.ones_like(x))
```

## Why analytic

- **Accuracy.** Analytic pullbacks avoid the step-size choice and subtraction
  error of finite differences. Their accuracy still depends on the conditioning
  of the forward and adjoint problems and on floating-point rounding.
- **Cost.** A pullback costs about one forward call, whatever the number of
  inputs. A central difference costs two forward calls per real input: 2 × 3N for
  the positions of N spheres.
- **Reuse.** A pullback reuses work from the forward call. The pullback of
  `diff.solve` solves the adjoint system with the LU factors of the forward
  solve. The requested-illumination records of one cluster share one factor of
  `I - T C`.

## Why a value and a one-use context

- **Ownership.** The context owns copies of what its pullback needs. Changing the
  input arrays after the forward call does not change the gradients.
- **No global registry.** Nothing outside the context remembers a forward call.
  Python, or the framework that holds the context, decides how long it lives, and
  nothing leaks when a context is dropped unused.
- **Memory.** The pullback consumes the context and frees its data, such as a
  dense LU factor. In Rust, `pullback(self, ...)` takes the residual (the Rust
  form of the context) by value, so the compiler rules out a second use. In
  Python, a second call raises
  `ValueError`. A cotangent of the wrong shape raises before the context is
  consumed, so the call can be retried.

A framework that needs a second reverse pass records the forward again (see
[framework adapters](adapters.md)).

## Why the real pairing

The loss is real and most inputs are complex. With the pairing defined above,
the gradient of a complex input `x` is one complex number: its real part is the
derivative with respect to `Re x` and its imaginary part the derivative with
respect to `Im x`. For a real input it is the ordinary real gradient. Advect and
PyTorch use the same rule. JAX and HIPS Autograd pair complex numbers without the
conjugate, so their adapters conjugate the cotangent and the gradients around
every pullback.

## Inputs, labels and order

- **Forward order.** A pullback returns one gradient per differentiable input, in
  the order of the forward's arguments. A docstring states any other order.
- **Static labels.** Degrees, orders, polarization indices, `lmax`, basis sizes,
  quadrature nodes and lattice dimensions are integers or fixed choices. They have
  no gradient and stay outside the record's inputs. Continuous values that the
  numerical method chooses, such as the Ewald split parameter, are held fixed
  too.
- **No dense Jacobians.** A pullback maps one cotangent to the input gradients
  directly. The Jacobian of an M × M interaction matrix with respect to 3N
  positions has 3N M² entries: 1.7e11 complex numbers, or 2.8 TB, for 100 spheres
  at `lmax = 10`. No record forms it.

## What treams-rs does not do

- **No finite-difference fallback.** An operation without an analytic pullback
  has no record. Finite differences appear only in tests, as an independent check
  (see [gradient checks](../differentiation/gradient-checks.md)).
- **First order only.** Pullbacks give first derivatives in reverse mode. Forward
  mode and second derivatives are not available. Inside the Rust core, fixed-size
  forward-mode jets (`numerics::Jet`) give some kernels their local first
  derivatives; they never reach Python.

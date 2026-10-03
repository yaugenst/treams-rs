---
description: Check the pullback of a record or the gradient of a scalar objective against finite differences with treams_rs.testing.
---

# Gradient checks

`treams_rs.testing` compares first-order derivatives with central finite
differences, using NumPy only. `check_pullback` checks a record and
`check_gradient` a scalar objective. Both return `None` on success and raise an
error that names the input on disagreement. Finite differences serve only as
this check; every record computes its gradients analytically in Rust.

## Records

Pass a [record](index.md): a function returning `(output, context)`, whose
`context.pullback(...)` returns one gradient per input, in the order of the
inputs. The pullback maps the gradient of a loss with respect to the output,
called the cotangent, back to these input gradients; the
[glossary](../reference/glossary.md#records-and-gradients) defines both
terms. A pullback function can replace the context. Keep fixed labels and
options in a closure.

```python exec
import numpy as np
from treams_rs import diff
from treams_rs.testing import check_pullback

matrix = np.array([[2.0 + 0.1j, 0.2], [-0.1j, 1.7]])
incident = np.array([[0.3 + 0.2j], [0.5]])
check_pullback(diff.solve, matrix, incident)

# The Bessel order is fixed; only z is differentiated.
check_pullback(lambda z: diff.bessel([0, 1, 2], z), np.asarray(0.8 + 0.3j))
```

`check_pullback` draws one random direction per input and random cotangents
from `seed=0`. Each input is checked separately, so errors in different
inputs cannot cancel. Repeat with other seeds to probe other directions. The
record runs once at the given inputs and twice per input for the central
differences. Only the first context's pullback is called, exactly once, because
a pullback consumes its context.

## Scalar objectives

For a real scalar objective, pass a function that returns its gradient. It may
come from a framework adapter or from an independent analytic formula:

```python exec
import numpy as np
from treams_rs.testing import check_gradient


def energy(coefficients):
    return np.vdot(coefficients, coefficients).real


def gradient(coefficients):
    return 2 * coefficients


check_gradient(energy, gradient, np.array([0.3 + 0.2j, -0.1j]))
```

Complex gradients follow the [pairing](index.md#rules) of the records, also
for real objectives of complex inputs such as squared norms. JAX and HIPS
Autograd pair complex numbers without the conjugate: pass the conjugate of
their raw gradient. The gradient function runs once per check. For several inputs, return a tuple with
one gradient per input; a list counts as one array gradient.

## Explicit probes and errors

```python exec
import numpy as np
from treams_rs.testing import check_pullback


def square_record(z):
    return z * z, lambda cotangent: 2 * z.conj() * cotangent


check_pullback(
    square_record,
    np.asarray(0.4 + 0.3j),
    directions=(np.asarray(1j),),
    cotangents=np.asarray(1.0 + 0j),
    step=1e-6,
    rtol=1e-5,
    atol=1e-7,
)
```

`directions` is always a tuple with one item per input, each of that input's
shape. Inputs must be float64 or complex128 arrays or scalars. Real inputs
require real directions; a complex gradient of a real input counts with its
real part.

For tuple outputs, pass a tuple of cotangents; the pullback then takes one
cotangent per output. For a single output, pass one cotangent of its shape.
Outputs are one array or a flat tuple, and the check covers first derivatives
only.

A mismatch reports the input index and shape, the analytic directional
derivative, the finite difference, the absolute error, the tolerance, the step
and the seed. Shape errors name the gradient or output. The check passes when
the error is at most `atol + rtol * abs(finite_difference)`; this bounds the
agreement, not the accuracy of either value. Choose a step that suits the scale
of the input, keep both perturbed inputs inside the physical domain, and expect
poor conditioning near resonances and thresholds. A pass holds for the checked
directions at one point only. Functions must be deterministic and must leave
their inputs unchanged.

[Testing](../development/testing.md) describes how the test suite uses these
checks.

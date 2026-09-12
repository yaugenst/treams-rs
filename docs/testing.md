# Check a gradient or native pullback

`treams_rs.testing` provides two NumPy-only checks for first-order derivatives.
They raise an actionable error on disagreement and return `None` on success.
Finite differences are diagnostic oracles; production derivatives remain analytic
in Rust.

## Native records

Pass a function returning `(output, context)`, where `context.pullback(...)`
returns gradients in the same order as the dynamic inputs. A callable pullback
can replace the context. Capture static labels and options in a closure.

```python exec
import numpy as np
from treams_rs import diff
from treams_rs.testing import check_pullback

matrix = np.array([[2.0 + 0.1j, 0.2], [-0.1j, 1.7]])
incident = np.array([[0.3 + 0.2j], [0.5]])
check_pullback(diff.solve, matrix, incident)

# The Bessel order is static; only z is differentiated.
check_pullback(lambda z: diff.bessel([0, 1, 2], z), np.asarray(0.8 + 0.3j))
```

One random direction per parameter and random output cotangents are generated
from `seed=0`. Each parameter is tested separately, so errors in different
parameters cannot cancel. Repeat with other seeds to probe additional directions.
The record runs once at the original parameters and twice per parameter for the
central differences. Only the original context's pullback is called, exactly
once, supporting native contexts that are consumed on use.

## Scalar objectives

For a real scalar objective, supply its gradient callable. It may come from an
AD adapter or an independent analytic implementation:

```python exec
import numpy as np
from treams_rs.testing import check_gradient


def energy(coefficients):
    return np.vdot(coefficients, coefficients).real


def gradient(coefficients):
    return 2 * coefficients


check_gradient(energy, gradient, np.array([0.3 + 0.2j, -0.1j]))
```

Complex gradients follow `dL = Re(vdot(gradient, direction))`, including
nonholomorphic real objectives such as squared norms. Convert a framework's
complex convention before passing its raw gradient. The supplied gradient is
evaluated once per check. For multiple inputs, return a tuple with one gradient
per input; a list represents a single array gradient.

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

`directions` is always a tuple with one shape-matched item per input. Inputs
must be float64 or complex128 arrays/scalars. Real inputs require real directions;
a complex native cotangent for a real input is projected onto its real part.
For tuple outputs, supply a tuple of cotangents and accept one positional
cotangent per output in the pullback. Cotangents must otherwise match the
single-output structure and shape. Nested trees and higher derivatives are not
supported.

A mismatch reports the parameter index and shape, analytic contraction, finite
difference, absolute error, tolerance, step, and seed. Shape errors identify the
relevant gradient or output. Tolerances compare against
`atol + rtol * abs(finite_difference)`; they are not a guarantee of numerical
accuracy. Choose a step appropriate to parameter scale, keep both perturbations
inside the physical domain, and check conditioning near resonances or thresholds.
These local checks do not establish differentiability at discontinuities or prove
correctness in every direction. Functions must be deterministic and must not mutate
their inputs.

# Framework adapters

The optional Advect, JAX, and PyTorch adapters compose Rust forward operations and
analytic first-order pullbacks. They do not implement numerical derivatives in
Python. Install the corresponding `treams-rs[advect]`, `treams-rs[jax]`, or
`treams-rs[torch]` extra; importing the core package needs none of these frameworks.

JAX and PyTorch expose `sphere`, `bessel`, `solve`, `interaction`, and `illuminate`
conveniences. Their `wrap` function exposes any existing `diff` operation through
the same boundary, including operations returning several arrays. Static basis metadata,
mode labels and algorithm options belong in the recording function's closure.

## Advect

Import `treams_rs.advect` alongside `advect.numpy` to compose native numerical
operations with ordinary array objectives. The [README example](../README.md#differentiation)
optimizes sphere radii in a cluster; the [API reference](api.md#treams_rsadvect)
lists the direct native operations and gradient argument ordering.

Advect uses the native one-use residual contract. A new forward call is required
for another VJP. Forward mode, higher derivatives, staging, checkpointing, and
Jacobian helpers that repeatedly invoke one residual are unsupported. Use
[gradient checks](testing.md) to qualify complete objectives independently of the
framework's trace.

## JAX

```python
import jax
import jax.numpy as jnp
from treams_rs import jax as tr

jax.config.update("jax_enable_x64", True)


@jax.jit
def scattering(radius):
    matrix = tr.sphere(2, 1.2, jnp.reshape(radius, (1,)), jnp.array([3.0 + 0.1j, 1.0]))
    return jnp.sum(jnp.abs(matrix) ** 2)


value, gradient = jax.value_and_grad(scattering)(0.3)
```

`jit`, reverse differentiation and sequential `vmap` are supported on the CPU
backend. Every native call returns float64 or complex128; dynamic inputs must
also use these dtypes, and `jax_enable_x64` must be enabled explicitly. The
adapter never silently truncates a native result to float32. Set `JAX_PLATFORMS=cpu`
before importing JAX on a machine whose default JAX backend is a GPU. Concrete
GPU arrays and a GPU default backend are rejected. Explicitly overriding a
compiled function to a GPU backend is unsupported; callbacks would still execute
on the CPU, with transfers, and do not provide a native GPU adapter.

Prepare other operations once outside the traced function:

```python
import numpy as np
from treams_rs import diff

matrix = np.array([[2.0 + 0.1j, 0.2], [0.1j, 3.0]])
eigensystem = tr.wrap(diff.eig, matrix)
values, vectors = jax.jit(eigensystem)(jnp.asarray(matrix))
```

`wrap` executes one example forward to determine output shapes and dtypes, and
requires subsequent input shapes/dtypes to match. Examples must be valid physical
inputs. Output structure must stay fixed. Use immutable static configuration and
a deterministic recording function: JAX's `pure_callback` may eliminate or repeat
calls. It must not send messages, mutate state or depend on a random generator.

The residual contains only JAX primals. Each reverse call recomputes a native
forward and immediately consumes its pullback. This supports repeated VJPs and
checkpointing without a global context registry or leaked native handles; the
cost is one additional native forward per reverse call. `vmap` runs callbacks
sequentially; use a native batched input when the underlying operation supports it.

## PyTorch

```python
import torch
from treams_rs import torch as tr

radius = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
matrix = tr.sphere(2, 1.2, radius, [3.0 + 0.1j, 1.0])
loss = matrix.abs().square().sum()
loss.backward()
gradient = radius.grad
```

Dynamic tensors must be CPU float64/complex128. CUDA, MPS and reduced-precision
tensors are rejected rather than implicitly copied through the CPU. Ordinary
Python/NumPy constants are accepted when they represent float64/complex128 values.
`tr.wrap(diff.solve)` prepares other native boundaries; unlike JAX it needs no
example execution because PyTorch discovers outputs eagerly.

The first backward consumes the retained native context, preserving its LU
factorization and other native reuse. A second backward with `retain_graph=True`
recomputes the native forward before applying another pullback. The bridge keeps
owned input snapshots for this purpose, so it adds their storage to the native
residual. Saved tensor versions detect ordinary in-place tensor mutation; NumPy
aliases cannot alter the recomputation snapshots. Noncontiguous and conjugated
CPU tensor views are supported. Outputs own their arrays. Snapshot storage is
released with PyTorch's saved tensors after the final backward.

`create_graph=True`, higher derivatives, forward-mode AD, `torch.func` transforms
and `torch.compile` are outside the supported contract. The native first-order
rules are not an implementation of these features.

## Requested illuminations

Both adapters expose `illuminate(local, coupling, incident)`. For `incident`
shaped `(channels, illuminations)`, Rust solves only those illumination columns:
`(I - local @ coupling) @ scattered = local @ incident`. It avoids forming the
full interacting response and differentiates local matrices, coupling and
incidence. Use this in objectives that need a few incident fields. Static reuse of
an LU across independent forward calls is available separately through
`diff.factor_interaction`; the framework convenience records a fresh operation
so changed geometry/material parameters receive the correct gradients.

## Recording function contract

`record(*parameters)` returns `(outputs, context)` or `(outputs, pullback)`, where
`outputs` is one numeric array/scalar or a flat tuple of arrays/scalars. A context
supplies `context.pullback(*output_cotangents)`. The pullback returns one input
gradient or a tuple with one gradient per dynamic parameter, in exactly that
parameter order. Shapes must agree, and the bridge projects gradients onto the
real part for real parameters. It converts complex conventions: Rust and PyTorch
use `Re(vdot(gradient, direction))`, while JAX uses the bilinear convention.

Read the individual `diff` docstring before wrapping: some native contexts include
gradients for optional inputs or embedded basis coordinates, and their order can
differ from the public forward signature. For example, cluster gradients are
ordered radii, positions, epsilon, k0:

```python
from treams_rs import diff


def record_cluster(radii, positions, epsilon, k0):
    return diff.cluster(1, float(k0), radii, epsilon, positions)
```

Pass these four arrays, in this order, to `wrap`. A sphere context always returns
all five material/geometry gradients, even if `mu` or `kappa` used defaults; the
`sphere` conveniences handle those constants. For a custom selection, return a
small pullback closure that selects/reorders the native gradients. This also
supports cylindrical `pullback_axial` and flattened heterogeneous particle
inputs. The adapter does not guess which hidden coordinates should be dynamic.

Numerical qualification includes native VJP comparisons, complex directional
differences, eigensystems with unused output cotangents, broadcast lattice sums,
multicolumn illumination, Hypothesis sphere scaling, repeated reverse calls,
ownership and dtype contracts. Native singularity/conditioning limitations still
apply; no finite-difference derivative fallback is used.

References: [JAX external callbacks](https://docs.jax.dev/en/latest/external-callbacks.html)
and [PyTorch custom autograd functions](https://docs.pytorch.org/docs/stable/notes/extending.html).

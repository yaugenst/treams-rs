# Framework adapters

Use `import treams_rs as tr` with Advect, JAX, PyTorch or
[HIPS Autograd](https://github.com/HIPS/autograd). Framework arrays and traced
values select their adapter automatically, so an objective written with the
ordinary API supports first-order forward and reverse differentiation within
the [limits](#limits) below. Rust supplies
analytic derivatives at native boundaries through the records of
[Differentiation](index.md);
shared array operations stay in the selected framework. The adapters add no
finite-difference derivatives. [Advect](https://yaugenst.github.io/advect/) and HIPS
Autograd differentiate NumPy-style code.

```python
import advect
import treams_rs as tr


def scattering(radius):
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=3)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
    return sphere.cross_sections(incident).scattering


value, gradient = advect.value_and_grad(scattering)(advect.numpy.asarray(0.2))
assert gradient > 0
value, tangent = advect.jvp(scattering)(
    advect.numpy.asarray(0.2), tangents=advect.numpy.asarray(1.0)
)
assert tangent > 0
```

The same objective in JAX:

```python
import jax
import jax.numpy as jnp
import treams_rs as tr


def scattering(radius):
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=3)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
    return sphere.cross_sections(incident).scattering


value, gradient = jax.jit(jax.value_and_grad(scattering))(0.2)
assert gradient > 0
radius = jnp.asarray(0.2)
value, tangent = jax.jvp(scattering, (radius,), (jnp.ones_like(radius),))
derivative = jax.jacfwd(scattering)(radius)
assert tangent > 0 and derivative > 0
```

And in PyTorch:

```python
import torch
import treams_rs as tr


def scattering(radius):
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=3)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
    return sphere.cross_sections(incident).scattering


radius = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
scattering(radius).backward()
assert radius.grad > 0
value, tangent = torch.func.jvp(scattering, (radius,), (torch.ones_like(radius),))
assert tangent > 0
```

And in HIPS Autograd:

```python
from autograd import make_jvp, value_and_grad
import treams_rs as tr


def scattering(radius):
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=3)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
    return sphere.cross_sections(incident).scattering


value, gradient = value_and_grad(scattering)(0.2)
assert gradient > 0
value, tangent = make_jvp(scattering)(0.2)(1.0)
assert tangent > 0
```

The JVP in each example uses a unit radius perturbation. For an array-valued
function, it returns the sensitivity of every output to that input direction.
JVPs are often useful for field maps, frequency sensitivity and correlated
geometry perturbations; they do not construct a full Jacobian.

## Frameworks compared

| | Advect | JAX | PyTorch | HIPS Autograd |
|---|---|---|---|---|
| Import | `import treams_rs as tr` | `import treams_rs as tr` | `import treams_rs as tr` | `import treams_rs as tr` |
| Extra | `treams-rs[advect]` | `treams-rs[jax]` | `treams-rs[torch]` | `treams-rs[autograd]` |
| Reverse context lifetime | Owned by the Advect tape | Built-ins keep numerical state or inputs; unannotated custom records repeat the forward | Kept through the first backward, recorded again for later backwards | Reused for repeated VJPs |
| Native work per direct JVP | One forward; its context computes the tangent | One record for built-ins; unannotated custom records repeat it for the tangent | The forward context computes the tangent | The forward context computes the tangent |
| Dtypes | float32/64 and complex64/128 inputs; double-precision outputs | float32/64 and complex64/128 inputs; single-precision outputs by default, double with `jax_enable_x64` | float32/64 and complex64/128 inputs; double-precision outputs | float64 and complex128; Python numbers work as constants |
| Transforms | `grad`, `value_and_grad`, `jvp` | `grad`, `value_and_grad`, `vjp`, `jvp`, `jacfwd`, `jacrev`, `linearize`, `jit`, `checkpoint`, sequential `vmap` | `backward`, `torch.autograd.grad`, repeated backward with `retain_graph=True`, `torch.func.jvp`, `torch.autograd.forward_ad` | `grad`, `value_and_grad`, `make_vjp`, `make_jvp` |
| Devices | CPU | CPU; GPU arrays and a GPU default backend raise `ValueError` | CPU; CUDA and MPS tensors raise `ValueError` | CPU |
| Higher order | No | No | No | No |

Native calculations use double precision. Gradients retain their input dtype;
no adapter changes a framework's global precision settings. For JAX, enable
`jax_enable_x64` when the objective needs double-precision arrays and gradients.

Higher derivatives are unsupported, including nested gradients and differentiating
a JVP or VJP with respect to the original model parameters. Combining first
derivatives at fixed parameters is possible, as shown below. Advect also rejects
`stage` and `checkpoint`; PyTorch rejects
`create_graph=True`. The supported PyTorch functional transform is
`torch.func.jvp`; other `torch.func` transforms and `torch.compile` are outside
the supported contract.
[How long a context lives](../design/adapters.md#how-long-a-context-lives) gives
the reasons for these context lifetimes.

## Use both modes together

When fitting a model to measurements, an optimizer may need to follow a
parameter change to the predicted measurements, then send that change back to
the parameters. If `J` is the matrix of first derivatives and `v` is the
parameter change, the result is `J.T @ J @ v`, often called a
**Gauss–Newton product**. Neither step needs to build `J`.

In JAX, keep the parameters fixed with `linearize`, then transpose that linear
map. This small example uses spherical Bessel values; `predictions` can instead
return the real measurements from your scattering model.

```python
import jax
import jax.numpy as jnp
import treams_rs as tr


def predictions(parameters):
    return tr.special.spherical_jn(2, parameters).real


parameters = jnp.array([0.4, 0.7])
direction = jnp.array([0.2, -0.3])
_, linear = jax.linearize(predictions, parameters)
change, transpose = jax.vjp(linear, direction)
(result,) = transpose(change)
assert result.shape == parameters.shape
```

In PyTorch, use the original outputs for the reverse step. The forward and
reverse steps share one native calculation. `detach()` treats the output change
as a fixed weight, so the reverse step does not differentiate the tangent.

```python
import torch
import treams_rs as tr


def predictions(parameters):
    return tr.special.spherical_jn(2, parameters).real


parameters = torch.tensor([0.4, 0.7], dtype=torch.float64, requires_grad=True)
direction = torch.tensor([0.2, -0.3], dtype=torch.float64)
value, change = torch.func.jvp(predictions, (parameters,), (direction,))
(result,) = torch.autograd.grad(value, parameters, change.detach())
assert result.shape == parameters.shape
```

These are two first-order actions at the same parameters. The result is useful
for least-squares fitting, but is not a general second derivative: it leaves
out how `J` itself changes with the parameters. General Hessians and nested
higher derivatives remain unsupported. PyTorch also rejects differentiating
the tangent directly; use the original `value` as above.

## Automatic selection and constants

The ordinary API examines inputs before converting arrays. An Advect tracer,
JAX array or tracer, PyTorch tensor, or HIPS Autograd box selects that framework.
Lists, tuples, dictionaries and material components are inspected too. Numeric
constants and compatible NumPy physics objects, such as a fixed incident wave,
are promoted when they meet framework values. Mixing frameworks in one
operation raises `TypeError` rather than detaching gradients.

Python numbers and NumPy arrays alone keep their existing NumPy return types,
even if optional frameworks are installed or a surrounding objective is being
differentiated. No global backend setting is needed, and the package imports
only the selected adapter. If you need framework outputs from constants alone,
use a framework array as an input or the explicit `tr.advect`, `tr.jax`,
`tr.torch` or `tr.autograd` namespace. These namespaces also retain expert
record helpers such as `wrap`.

Keep changing values as framework values in your own code. A prior `float(...)`,
`complex(...)` or NumPy conversion can lose a trace before treams-rs sees it;
use the framework's array operations for your objective's arithmetic.
Direct calls to differentiable numerical functions also select the adapter.
Their NumPy ufunc attributes and methods remain available for NumPy use, but
public wrappers need not be `numpy.ufunc` instances. Differentiated calls do
not support `out` or masked `where`: apply your framework's `where` to the
returned value instead. Ufunc methods such as `reduce` remain NumPy operations.

## Physical objects

All four adapters offer the same constructors: `Material`, `sphere_tmatrix`,
`multilayer_sphere_tmatrix`, `cylinder_tmatrix`, `multilayer_cylinder_tmatrix`,
`plane_wave`, `wave`, `tmatrix`, `smatrix`, `Cluster`, `interface`, `slab`,
`multilayer_slab`, `propagation`, `stack` and `solve_periodic`. They return the
shared classes `TMatrix`, `Wave`, `PlaneWave`, `SMatrix`, `Cluster` and
`PeriodicResponse`:

- a T-matrix scatters waves, gives cross sections and response metrics, and
  supports `rotate`, `translate` and `in_basis`;
- a wave gives its fields at points (`efield`, `hfield`, `dfield`, `bfield`,
  `gfield`, `ffield`) and converts with `in_basis`;
- a `Cluster` separates assembly from `solve()`, and its `scatter()` solves only
  for the requested illumination;
- an `SMatrix` gives `cascade`, `scatter`, `power` and `bands`;
- `solve_periodic(...)` returns a `PeriodicResponse`, whose `to_smatrix()`
  converts without a second interaction solve. `to_cylindrical()` converts
  a spherical chain along z to cylindrical modes.

T-matrix properties include `average_cross_sections`, `average_cross_widths`,
`circular_dichroism`, `duality_breaking` and `electromagnetic_chirality`. Averages
require a global basis and a nonabsorbing propagating medium; response metrics
require a global helicity basis.

Build changing geometry and materials inside the differentiated function.
Constants may be created outside it. Under `jax.jit`, return arrays, not
physics objects; the objects are not JAX pytrees.

```python
import jax
import treams_rs as tr

jax.config.update("jax_enable_x64", True)


def intensity(radius):
    sphere = tr.sphere_tmatrix(
        radius=radius, material=tr.Material(3.0 + 0.1j), k0=1.2, lmax=2
    )
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    field = sphere.scatter(incident).efield([[0.3, 0.2, 1.5]])
    return (abs(field) ** 2).sum()


value, gradient = jax.jit(jax.value_and_grad(intensity))(0.2)
```

## Fixed and differentiable inputs

Frequency, amplitudes, materials, radii, positions, lattice vectors, the
Bloch vector and real plane-wave directions can carry gradients. These inputs
stay fixed:

- bases, mode labels, cutoffs such as `lmax`, and polarization conventions;
- plane-wave ports built from a `PlaneWavePorts` basis;
- the axial wavenumbers of a cylindrical basis in the physics objects. A plane
  wave illuminates a cylinder only with `kz = 0` and transverse incidence.

Plane-wave direction derivatives are defined away from the polarization axis.
For `SMatrix.power`, supply `side="negative"` or `side="positive"` when the
incident direction is differentiated; fixed directions retain automatic side
selection.

Axial wavenumbers use two keywords. `kz` holds one value per mode, like
`basis.kz` in treams; `kzs` holds the sorted distinct values, like
`TMatrixC.cylinder(kzs, ...)` in treams. The Advect functions `field`,
`hfield`, `gfield`, `ffield` and `field_operator` take `kz=`, and `expansion`,
`lattice_expansion` and `cylinder` take `kzs`; those make the axial
wavenumbers differentiable.

## Periodic geometry

Ports given by a `PlaneWavePorts` basis have fixed transverse wavevectors. To
differentiate with respect to the lattice, convert with integer diffraction
orders instead:

```python
import advect
import advect.numpy as anp
import numpy as np
import treams_rs as tr

k0 = 2 * np.pi / 400  # vacuum wavelength 400 nm
kpar = [0.0, 0.3 * k0]


def transmission(period):
    sphere = tr.sphere_tmatrix(k0=k0, lmax=2, radius=100.0, material=4)
    response = tr.solve_periodic(sphere, lattice=period * anp.eye(2), kpar=kpar)
    array = response.to_smatrix(orders=[[0, 0]])
    incident = tr.plane_wave([0, 0.3, np.sqrt(0.91)], "positive_helicity", k0=k0)
    return array.power(incident).transmission


value, gradient = advect.value_and_grad(transmission)(anp.asarray(500.0))
step = 1e-3
difference = (transmission(500.0 + step) - transmission(500.0 - step)) / (2 * step)
np.testing.assert_allclose(gradient, difference, rtol=1e-6)
```

The objective is the power transmitted into the zeroth order. Each row of
`orders` is one diffraction order, with the transverse wavevector
`q = kpar + 2*pi*orders @ inverse(cell).T`, so the ports move with the cell and
the Bloch vector. The gradient includes the cell area, the lattice coupling, the
incidence, the radiation into the orders and the power normalization. Each order
carries both helicities. `smatrix.transverse_wavevectors` returns the `q`
values; `smatrix.basis` is `None`, because a fixed basis cannot hold traced
wavevectors. Cylindrical order ports take a vector of integer orders and one
fixed axial wavenumber. The order labels themselves are fixed integers.

`smatrix.scatter(negative=incident)` returns the outgoing waves on the
`positive` and `negative` sides; their fields carry gradients. `power()` finds
the incident side of a plane wave; pass `side="negative"` or `side="positive"`
otherwise. The incident wave must travel toward the stack and match its ports,
outer medium and frequency.

## Requested illuminations

For physics objects, use `Cluster.scatter(incident)`. Expert array code can use
`illuminate(local, coupling, incident)` from `tr.advect`, `tr.jax`, `tr.torch`
or `tr.autograd`. It solves
`(I - local @ coupling) @ scattered = local @ incident` for the columns of
`incident`, shaped `(channels, illuminations)`. It never forms the full
interacting T-matrix and differentiates `local`, `coupling` and `incident`. Use
it when an objective needs a few incident fields. `diff.factor_interaction`
reuses one LU factor across many forward calls with fixed matrices;
`illuminate` records a fresh solve, so changed geometry and materials receive
their gradients.

`Cluster.factor()`, `TMatrix.interaction.factor()` and
`TMatrix.latticeinteraction.factor(...)` prepare coupling for repeated
illuminations. With framework inputs, each solve records its own native
factorization; these helpers do not retain LU factors across differentiated
calls. NumPy factors can reuse them when geometry and materials stay fixed.

## Advect

The adapter requires Advect 0.3.1 or later. Pass arrays to Advect transforms, scalars included:
`advect.grad(f)(advect.numpy.asarray(0.2))`. The Advect tape owns the native
contexts and reuses them across tangent directions. Besides
the physics objects, `treams_rs.advect` offers an array function for every
`diff` record except the reusable factors; [custom records](custom-records.md)
maps them.

```python
import advect
import advect.numpy as anp
import treams_rs.advect as tr


def size(radius):
    matrix = tr.sphere(
        2, 1.2, anp.reshape(radius, (1,)), anp.asarray([3.0 + 0.1j, 1.0])
    )
    return anp.sum(anp.abs(matrix) ** 2)


gradient = advect.grad(size)(anp.asarray(0.3))
```

`advect.jvp(size)(radius, tangents=direction)` computes a first-order tangent.
Advect passes the forward invocation's saved context to its JVP callback, so
each direction reuses its numerical factors and a direct JVP needs one native forward.

## JAX

JAX 0.10 and 0.11 work with their default single-precision arrays. Enable
`jax_enable_x64` globally, as below, when you need double-precision outputs and
gradients. A scoped `with jax.enable_x64(True)` can lose its setting on JAX's
callback worker threads. On a machine
whose default JAX backend is a GPU, set `JAX_PLATFORMS=cpu` before importing JAX. Native calls
run in `jax.pure_callback` on the CPU; `vmap` calls them once per batch element.

```python
import jax
import jax.numpy as jnp
from treams_rs import jax as tr

jax.config.update("jax_enable_x64", True)


@jax.jit
def size(radius):
    matrix = tr.sphere(2, 1.2, jnp.reshape(radius, (1,)), jnp.array([3.0 + 0.1j, 1.0]))
    return jnp.sum(jnp.abs(matrix) ** 2)


value, gradient = jax.value_and_grad(size)(0.3)
```

JAX keeps computed numerical state, such as matrix factors, for repeated
`jax.vjp` and `jax.linearize` calls. Functions whose derivatives need only the
inputs retain those inputs instead. Both paths reconstruct the derivative
context without evaluating the values again. `jax.jacfwd` evaluates tangent
seeds sequentially; for input-only contexts it can omit value evaluation entirely.
JAX owns the arrays; there is no global native-context registry.
`jax.checkpoint` may recompute the record according to its rematerialization policy.

Other records, including unannotated custom records, retain the inputs and
reconstruct a native record for every derivative application, including a direct
JVP. Built-in records supply their value and saved data with one native forward.
Saving and restoring this data adds memory and copying costs; the timing
comparison includes those costs as well as the numerical work.
These first-order
transforms compose with `jit`, `checkpoint` and sequential `vmap`. `tr.wrap`
makes another record a JAX function; see [custom records](custom-records.md).

## PyTorch

```python
import torch
from treams_rs import torch as tr

radius = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
matrix = tr.sphere(2, 1.2, radius, [3.0 + 0.1j, 1.0])
matrix.abs().square().sum().backward()
gradient = radius.grad
```

The first backward uses the context of the forward, so the pullback of
`tr.solve` reuses its LU factors. A later backward with
`retain_graph=True` records the forward again from snapshots of the inputs. The
snapshots add their memory to that of the context and are freed with PyTorch's
saved tensors. An in-place change of an input tensor raises at the next
backward, as for any PyTorch operation. Noncontiguous and conjugated CPU views
work, and outputs own their memory. `tr.wrap` makes any `diff` record a PyTorch
function; see [custom records](custom-records.md).

Forward mode uses `torch.func.jvp` or the dual-tensor API. Both use the native
context saved by the forward call for its tangent. A subsequent backward
through the returned primal reuses that context too; the first backward then
releases it. For example:

```python
import torch
from treams_rs import torch as tr

operator = torch.tensor([[2.0, 0.2], [0.1, 1.5]], dtype=torch.float64)
rhs = torch.ones((2, 1), dtype=torch.float64)
with torch.autograd.forward_ad.dual_level():
    dual_rhs = torch.autograd.forward_ad.make_dual(rhs, torch.ones_like(rhs))
    value, tangent = torch.autograd.forward_ad.unpack_dual(tr.solve(operator, dual_rhs))
    torch.testing.assert_close(tangent, value)
```

## HIPS Autograd

Use `autograd.numpy` for arithmetic around the treams-rs calls. Traced scalar,
array and nested material inputs select the adapter automatically:

```python
import autograd.numpy as anp
from autograd import grad
import treams_rs as tr


def scattering(parameters):
    radius, epsilon = parameters
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=epsilon)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
    return sphere.cross_sections(incident).scattering


gradient = grad(scattering)(anp.array([0.2, 3.0]))
assert gradient.shape == (2,)
assert anp.all(anp.isfinite(gradient))
```

The adapter uses Autograd primitives and the shared native pullback contract.
It keeps and reuses the invocation's native context for repeated VJPs.
Like JAX, HIPS Autograd uses the unconjugated complex pairing;
the adapter converts cotangents and gradients at the native boundary. Its
`tr.autograd.wrap(record)` also accepts custom records without example inputs.

`autograd.make_jvp(function)(inputs)(direction)` returns the value and its
tangent. The adapter uses the context of that forward evaluation, so one
native record supplies both. Input directions use ordinary real or complex
increments, without the conjugations needed for reverse cotangents.

## Limits

- Derivatives are first order, in forward and reverse mode, on the CPU. Higher
  derivatives, `torch.compile` and `torch.func` transforms other than `jvp` are
  unsupported. Under `jax.jit`,
  return arrays rather than physics objects. The
  [comparison above](#frameworks-compared) lists supported transforms and dtypes.
- Cylindrical physics objects keep their basis axial wavenumbers fixed;
  `cw.to_sw` also cannot differentiate its `kz` label. Framework plane waves
  require real directions. See [fixed and differentiable inputs](#fixed-and-differentiable-inputs)
  for the lower-level axial-gradient interfaces.
- `sw.periodic_to_pw` and `cw.periodic_to_pw` do not support independent
  wavevector derivatives. Use `PeriodicResponse.to_smatrix()` with
  [diffraction orders](#periodic-geometry) to differentiate physical ports with
  respect to lattice geometry and the Bloch vector.
- EBCM, custom lattice tables, chirality densities and the matrix-free cluster
  solver are outside the shared framework physics objects.
  [Custom records](custom-records.md#records-by-family) lists their array and
  explicit-gradient interfaces.
- Differentiated numerical calls do not support mutable `out`, masked `where`
  or ufunc methods such as `reduce`. Apply the framework's array operations to
  the returned values instead.
- With traced inputs, physics objects have only the members listed under
  [physical objects](#physical-objects) and in the framework references, such
  as [`treams_rs.torch`](../reference/python/torch.md). These members exist only
  on NumPy objects: `SMatrix.circular_dichroism`, `transfer_matrix`, `rotate`,
  `translate`, `permute`, `double`, `block` and the treams names `tr`, `cd`,
  `illuminate`, `add`, `periodic`, `bands_kz`, `changepoltype`, `poltype` and
  `material`; `TMatrix.select`, `permute`, `expand`, `expandlattice`,
  `valid_points` and `modetype`; `Wave.changepoltype` and `expand`;
  `PlaneWave.basis`, `kvecs` and `expand`.

Cutoffs, discrete labels and topology stay fixed. Eigenvalue crossings,
diffraction thresholds and the polarization axis have no smooth derivative;
see [points without a gradient](index.md#points-without-a-gradient).

References: [JAX external callbacks](https://docs.jax.dev/en/latest/external-callbacks.html)
and [PyTorch custom autograd functions](https://docs.pytorch.org/docs/stable/notes/extending.html).

---
description: Differentiate the ordinary API with Advect, JAX, PyTorch or HIPS Autograd, with their transforms, dtypes and limits.
---

# Framework adapters

Use `import treams_rs as tr` with Advect, JAX, PyTorch or
[HIPS Autograd](https://github.com/HIPS/autograd). Framework arrays and traced
values select their adapter automatically, so an objective written with the
ordinary API differentiates within the [limits](#limits) below. Rust supplies
analytic derivatives at native boundaries through the records of
[Differentiation](index.md);
shared array operations stay in the selected framework. The adapters add no
finite-difference derivatives. [Advect](https://yaugenst.github.io/advect/) and HIPS
Autograd differentiate NumPy-style code.

```python exec
import advect
import treams_rs as tr


def scattering(radius):
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=3)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
    return sphere.cross_sections(incident).scattering


value, gradient = advect.value_and_grad(scattering)(advect.numpy.asarray(0.2))
assert gradient > 0
```

The same objective in JAX:

```python exec jax
import jax
import treams_rs as tr


def scattering(radius):
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=3)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
    return sphere.cross_sections(incident).scattering


value, gradient = jax.jit(jax.value_and_grad(scattering))(0.2)
assert gradient > 0
```

And in PyTorch:

```python exec torch
import torch
import treams_rs as tr

radius = torch.tensor(0.2, requires_grad=True)
sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=3)
incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
scattering = sphere.cross_sections(incident).scattering
scattering.backward()
assert radius.grad > 0
```

And in HIPS Autograd:

```python exec autograd
from autograd import value_and_grad
import treams_rs as tr


def scattering(radius):
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=3)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
    return sphere.cross_sections(incident).scattering


value, gradient = value_and_grad(scattering)(0.2)
assert gradient > 0
```

## Frameworks compared

| | Advect | JAX | PyTorch | HIPS Autograd |
|---|---|---|---|---|
| Import | `import treams_rs as tr` | `import treams_rs as tr` | `import treams_rs as tr` | `import treams_rs as tr` |
| Extra | `treams-rs[advect]` | `treams-rs[jax]` | `treams-rs[torch]` | `treams-rs[autograd]` |
| Context lifetime | One use: each reverse pass consumes the context of its forward | Recorded again for every reverse pass | Kept for the first backward, recorded again for each further backward | Kept for the first reverse pass, recorded again for each further pass |
| Dtypes | float32/64 and complex64/128 inputs; double-precision outputs | float32/64 and complex64/128 inputs; single-precision outputs by default, double with `jax_enable_x64` | float32/64 and complex64/128 inputs; double-precision outputs | float64 and complex128; Python numbers work as constants |
| Transforms | `grad`, `value_and_grad` | `grad`, `value_and_grad`, `vjp`, `jit`, `checkpoint`, sequential `vmap` | `backward`, `torch.autograd.grad`, repeated backward with `retain_graph=True` | `grad`, `value_and_grad`, `make_vjp` |
| Devices | CPU | CPU; GPU arrays and a GPU default backend raise `ValueError` | CPU; CUDA and MPS tensors raise `ValueError` | CPU |
| Higher order | No | No | No | No |

Native calculations use double precision. Gradients retain their input dtype;
no adapter changes a framework's global precision settings. For JAX, enable
`jax_enable_x64` when the objective needs double-precision arrays and gradients.

Unsupported transforms raise: `jvp`, nested `grad`, `stage` and `checkpoint` in
Advect, `jvp` in JAX and `create_graph=True` in PyTorch. `torch.func` and
`torch.compile` are not supported. HIPS Autograd supports first-order reverse
mode only; nested gradients and forward-mode transforms are unsupported.
[How long a context lives](../design/adapters.md#how-long-a-context-lives) gives
the reasons for these context lifetimes.

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

```python exec jax
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

```python exec
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

Pass arrays to Advect transforms, scalars included:
`advect.grad(f)(advect.numpy.asarray(0.2))`. Each reverse pass consumes its
contexts, so call the transformed function again for every gradient. Besides
the physics objects, `treams_rs.advect` offers an array function for every
`diff` record except the reusable factors; [custom records](custom-records.md)
maps them.

```python exec
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

## JAX

JAX works with its default single-precision arrays. Enable `jax_enable_x64`
when you need double-precision outputs and gradients, as below. On a machine
whose default JAX backend is a GPU, set `JAX_PLATFORMS=cpu` before importing JAX. Native calls
run in `jax.pure_callback` on the CPU; `vmap` calls them once per batch element.

```python exec jax
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

JAX keeps only the inputs of a native call. Each reverse pass records the
forward again and consumes the new context at once. Repeated `jax.vjp` calls
and `jax.checkpoint` work, and nothing outside JAX holds a context; the price
is one extra native forward per reverse pass. `tr.wrap` makes any `diff` record
a JAX function; see [custom records](custom-records.md).

## PyTorch

```python exec torch
import torch
from treams_rs import torch as tr

radius = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
matrix = tr.sphere(2, 1.2, radius, [3.0 + 0.1j, 1.0])
matrix.abs().square().sum().backward()
gradient = radius.grad
```

The first backward consumes the context of the forward, so the pullback of
`tr.solve` reuses the LU factors of its forward. A second backward with
`retain_graph=True` records the forward again from snapshots of the inputs. The
snapshots add their memory to that of the context and are freed with PyTorch's
saved tensors. An in-place change of an input tensor raises at the next
backward, as for any PyTorch operation. Noncontiguous and conjugated CPU views
work, and outputs own their memory. `tr.wrap` makes any `diff` record a PyTorch
function; see [custom records](custom-records.md).

## HIPS Autograd

Use `autograd.numpy` for arithmetic around the treams-rs calls. Traced scalar,
array and nested material inputs select the adapter automatically:

```python exec autograd
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
It keeps a native context for the first reverse pass and records again for a
repeated VJP. Like JAX, HIPS Autograd uses the unconjugated complex pairing;
the adapter converts cotangents and gradients at the native boundary. Its
`tr.autograd.wrap(record)` also accepts custom records without example inputs.

## Limits

- Gradients are first order and reverse mode, on the CPU. Forward mode, higher
  derivatives, `torch.func` and `torch.compile` are unsupported. Under `jax.jit`,
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

## Known framework issues

**Advect 0.2.0: Python floats promoted to complex.** If the input of an Advect
transform is a Python float, an expression such as
`advect.numpy.abs(epsilon + 0.1j)` makes
the reverse pass raise `RuntimeError: Transpose rule for 'array.absolute'
failed`. An array input works. Advect 0.3.0, now selected in `uv.lock`, returns
the correct gradient for both inputs. Use an array when supporting Advect 0.2.0:

```python exec
import advect
import advect.numpy as anp


def loss(epsilon):
    return anp.abs(epsilon + 0.1j) ** 2


assert advect.grad(loss)(anp.asarray(3.0)) == 6.0
```

References: [JAX external callbacks](https://docs.jax.dev/en/latest/external-callbacks.html)
and [PyTorch custom autograd functions](https://docs.pytorch.org/docs/stable/notes/extending.html).

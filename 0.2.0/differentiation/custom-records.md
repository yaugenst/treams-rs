# Custom records

`treams_rs.jax.wrap`, `treams_rs.torch.wrap` and `treams_rs.autograd.wrap` turn
any record into a function that the framework differentiates. A context with
both derivative methods supports forward and reverse mode; a pullback closure
supports reverse mode. Use them for the `diff` records that the
physics objects do not call, or to combine several records into one operation.

```python
import jax
import jax.numpy as jnp
import numpy as np
from treams_rs import diff
from treams_rs import jax as tr

jax.config.update("jax_enable_x64", True)

matrix = np.array([[2.0 + 0.1j, 0.2], [0.1j, 3.0]])
eigensystem = tr.wrap(diff.eig, matrix)  # one example call fixes the shapes


def spread(operator):
    values, _ = eigensystem(operator)
    return jnp.sum(jnp.abs(values) ** 2)


values, vectors = jax.jit(eigensystem)(jnp.asarray(matrix))
gradient = jax.grad(spread)(jnp.asarray(matrix))
values_and_vectors, tangents = jax.jvp(
    eigensystem, (jnp.asarray(matrix),), (jnp.ones_like(jnp.asarray(matrix)),)
)
```

Advect has no public `wrap`. `treams_rs.advect` offers an array function for
every `diff` record except the reusable factors; the
[table below](#records-by-family) lists them.

## Writing a record

A record for `wrap` follows these rules:

- `record(*inputs)` returns `(outputs, context)` or `(outputs, pullback)`. The
  outputs are one array or scalar, or a flat tuple of them.
- `context.pullback(*cotangents)`, or `pullback(*cotangents)`, takes one
  cotangent per output: the gradient of the loss with respect to that output.
  The pullback maps these back to one gradient per input, in the order of the
  inputs: a tuple for several inputs, an array for one. The
  [glossary](../reference/glossary.md#records-and-gradients) defines both
  terms.
- Each gradient has the shape of its input. A native gradient of shape `(1,)`
  is also accepted for a scalar input and converted to shape `()`.
  The adapter keeps the real part of the gradient of a real input.
- Forward mode requires `context.pushforward(*tangents)`: one tangent per
  dynamic input, in the same order as the pullback's results. Each tangent has
  the input's shape and real or complex domain. The returned tangent has the
  primal output's shape and tuple structure. There is no conjugation of JVP
  inputs or outputs, in any framework.
- A native context is reusable. Its saved residual must correspond to the
  current record inputs; a context from different inputs gives incorrect
  derivatives. A pullback-only closure remains supported for reverse mode,
  but a JVP through it raises `NotImplementedError`.
- Gradients follow the [pairing](index.md#rules) of the `diff` records. JAX and
  HIPS Autograd pair complex numbers without the conjugate, so their adapters conjugate the
  cotangents and the gradients around the pullback; a record never converts.
- The record is deterministic and free of side effects: no messages, no state
  changes, no random numbers. JAX may skip or repeat the call.

The JAX `wrap(record, *examples)` runs the record once on the examples to learn
the output shapes and dtypes. Later calls must use the same input shapes and
dtypes, and the examples must be valid physical inputs. The PyTorch and HIPS Autograd
`wrap(record)` needs no examples: it learns the outputs on each call.
Built-in records expose numerical saved state to JAX. A custom record without
that metadata runs again for every derivative application, including a direct
JVP; keep the record deterministic and account for this extra primal work.

## Fixed configuration

The inputs of a record are its differentiable arrays. Bind fixed choices such
as `lmax`, bases, labels and keywords with `functools.partial`. This also keeps
a built-in record's saved-state support, so JAX can reuse its numerical work:

```python
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from treams_rs import diff
from treams_rs import jax as tr

jax.config.update("jax_enable_x64", True)


record_cluster = partial(diff.sphere_cluster, 1)  # lmax = 1


k0 = np.asarray(1.0)
radii = np.array([0.3, 0.3])
epsilon = np.array([4.0 + 0.1j, 4.0 + 0.1j])
positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
cluster = tr.wrap(record_cluster, k0, radii, epsilon, positions)


def size(radii, positions):
    matrix = cluster(jnp.asarray(k0), radii, jnp.asarray(epsilon), positions)
    return jnp.sum(jnp.abs(matrix) ** 2)


d_radii, d_positions = jax.grad(size, argnums=(0, 1))(
    jnp.asarray(radii), jnp.asarray(positions)
)
```

`diff.sphere_cluster` returns its gradients in the order of its arguments,
`(k0, radii, epsilon, positions)`, so the record passes them on unchanged.
A Python closure also works, but JAX treats it as a custom record and repeats
its forward calculation when applying a derivative.

## Choosing the gradients

A record differentiates every input it takes. Some contexts return gradients for
more inputs than an objective needs. The context of `diff.sphere` always returns
five gradients, `(k0, radii, epsilon, mu, kappa)`, even when `mu` and `kappa`
take their defaults. Select the pullback results and insert zero tangents for
fixed inputs in the pushforward:

```python
import torch
import numpy as np
from types import SimpleNamespace
from treams_rs import diff
from treams_rs import torch as tr


def record_sphere(radii, epsilon):
    matrix, context = diff.sphere(2, 1.2, radii, epsilon)

    def pullback(cotangent):
        _, d_radii, d_epsilon, _, _ = context.pullback(cotangent)
        return d_radii, d_epsilon

    def pushforward(d_radii, d_epsilon):
        fixed_material = np.zeros_like(epsilon)
        return context.pushforward(
            0.0, d_radii, d_epsilon, fixed_material, fixed_material
        )

    return matrix, SimpleNamespace(pullback=pullback, pushforward=pushforward)


sphere = tr.wrap(record_sphere)
radii = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
epsilon = torch.tensor([3.0 + 0.1j, 1.0], dtype=torch.complex128, requires_grad=True)
sphere(radii, epsilon).abs().square().sum().backward()
value, tangent = torch.func.jvp(
    sphere, (radii, epsilon), (torch.ones_like(radii), torch.zeros_like(epsilon))
)
assert tangent.shape == value.shape
```

Returning only `matrix, pullback` from this example preserves reverse-mode
behavior, but removes forward-mode support: the adapter does not infer a JVP
from a pullback or construct a dense Jacobian.

The same pattern selects the axial gradients of cylindrical records.
`context.pullback_axial` appends them to the other gradients:

- `diff.field` and `diff.field_operator` return one gradient per mode, matching
  `kz`, the axial wavenumber of each mode (`basis.kz` in treams);
- `diff.expansion` and `diff.lattice_expansion` return one gradient per distinct
  value, matching `kzs`, the sorted distinct axial wavenumbers of both bases
  (`TMatrixC.cylinder(kzs, ...)` in treams).

The docstring of each `diff` record states its gradients. The adapter never
guesses which inputs are differentiable.

The matching `pushforward_axial` takes those axial tangents last. For factors
built from particle blocks, `pushforward_blocks` takes a sequence of local block
tangents, followed by coupling and incident-field tangents. A wrapper that
flattens or selects dynamic inputs must adapt both derivative methods.

## Records by family

Every `diff` record has the same name in `treams_rs.advect`, except the reusable
factors. JAX, PyTorch and HIPS Autograd offer five records directly: `solve`, `interaction`,
`illuminate`, `sphere` and `bessel`; `wrap` covers the others. The
[API reference](../reference/python/index.md) lists every signature and gradient
order.

The adapter columns below describe explicit namespaces such as `tr.jax`.
[Differentiable numerical functions](../guide/numerical-namespaces.md) also
select their adapter through ordinary calls such as `tr.special.jv(...)`.

| Family | `diff` record | Advect function | JAX, PyTorch and HIPS Autograd | Physics object |
|---|---|---|---|---|
| Sphere T-matrix | `sphere` | `sphere` | `sphere` | `sphere_tmatrix`, `multilayer_sphere_tmatrix` |
| Mie coefficients | `mie`, `mie_cyl` | `mie`, `mie_cyl` | `wrap` | none |
| Cylinder T-matrix | `cylinder` | `cylinder` | `wrap` | `cylinder_tmatrix`, `multilayer_cylinder_tmatrix` |
| Sphere cluster (dense) | `sphere_cluster` | `sphere_cluster` | `wrap` | none (same result: `Cluster(...).solve()`) |
| Heterogeneous cluster | `particle_cluster` | `particle_cluster` | `wrap` | `Cluster(...).solve()` |
| Interaction and illumination | `interaction`, `illuminate` | `interaction`, `illuminate` | `interaction`, `illuminate` | `TMatrix.interaction`, `TMatrix.latticeinteraction` |
| Reusable factors | `factor_interaction`, `factor_interaction_blocks`, `sphere_cluster_factor`; their `InteractionFactor.record` | none | `wrap` of `factor.record` | `Cluster.factor()`, `TMatrix.interaction.factor()`, `TMatrix.latticeinteraction.factor(...)`; see below |
| Matrix-free sphere cluster | `iterative.SphereCluster.record` | none | none | none |
| Fields | `field`, `field_operator`, `plane_field` | `field`, `hfield`, `gfield`, `ffield`, `field_operator`, `plane_field` | `wrap` | `efield`, `hfield`, `dfield`, `bfield`, `gfield`, `ffield` of waves |
| Expansion and translation | `expansion`, `spherical_translation`, `cylindrical_translation`, `plane_expansion`, `plane_phases` | the same names | `wrap` | `in_basis` of waves and T-matrices; `TMatrix.translate(...)` |
| Rotation and permutation | `rotation`, `plane_permutation` | `rotation`, `plane_permutation` | `wrap` | `TMatrix.rotate(...)` for rotations |
| Periodic expansion | `lattice_expansion` | `lattice_expansion` | `wrap` | `solve_periodic(...)` |
| Lattice sums and tables | `lattice_sum`, `lattice_expansion_from_table` | the same names | `wrap` | none |
| Spherical chain to cylindrical modes | `periodic_to_cw` | `periodic_to_cw` | `wrap` | `PeriodicResponse.to_cylindrical(...)` |
| Radiation channels | `spherical_channels`, `cylindrical_channels` | the same names | `wrap` | `PeriodicResponse.to_smatrix(...)` |
| Interfaces and stacks | `fresnel`, `interface_coefficients`, `propagation_matrix`, `layer_stack` | the same names | `wrap` | `interface`, `slab`, `multilayer_slab`, `propagation`, `stack` |
| S-matrix networks | `smatrix_add`, `smatrix_illuminate`, `smatrix_tr`, `smatrix_from_array`, `smatrix_periodic`, `bands` | the same names, and `smatrix_cd` | `wrap` | `cascade`, `scatter`, `power`, `bands` of `SMatrix` |
| Chirality densities | `chirality_density`, `oriented_chirality` | the same names | `wrap` | none |
| Response metrics | `tmatrix_metric` | `tmatrix_metric` | `wrap` | `TMatrix.circular_dichroism`, `duality_breaking`, `electromagnetic_chirality` |
| Axisymmetric EBCM | `ebcm_qmat` | `ebcm_qmat` | `wrap` | none |
| Special functions | `bessel`, `incgamma`, `intkambe`, `angular`, `wignerd`, `sph_harm`, `vector_wave` | the same names | `bessel`; `wrap` | none |
| Coordinates | `coordinates`, `vector_coordinates` | the same names | `wrap` | none |
| Linear algebra | `solve`, `eig`, `svdvals` | `solve`, `eig`, `svdvals` | `solve`; `wrap` | none |

`Cluster.solve()` records the coupled particle blocks with `particle_cluster`.
`Cluster.scatter()` factors the interaction for the requested incident columns.
Framework factor helpers reuse prepared coupling, but each differentiated solve
records a new native factorization. Reuse of native LU factors across solves is
available through NumPy factors and the explicit factor records.

The Advect functions `interface_coefficients` and `propagation_matrix` return
coefficient arrays; `interface` and `propagation` build physical S-matrices.

---
description: Wrap a diff record for JAX or PyTorch, and find the record, adapter function and physics object for each family.
---

# Custom records

`treams_rs.jax.wrap` and `treams_rs.torch.wrap` turn any record into a function
that the framework differentiates. Use them for the `diff` records that the
physics objects do not call, or to combine several records into one operation.

```python exec jax
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
- Each gradient has the shape of its input. The adapter keeps the real part of
  the gradient of a real input.
- Gradients follow the [pairing](index.md#rules) of the `diff` records. JAX
  pairs complex numbers without the conjugate, so the JAX adapter conjugates the
  cotangents and the gradients around the pullback; a record never converts.
- The record is deterministic and free of side effects: no messages, no state
  changes, no random numbers. JAX may skip or repeat the call.

The JAX `wrap(record, *examples)` runs the record once on the examples to learn
the output shapes and dtypes. Later calls must use the same input shapes and
dtypes, and the examples must be valid physical inputs. The PyTorch
`wrap(record)` needs no examples: it learns the outputs on each call.

## Fixed configuration in closures

The inputs of a record are its differentiable arrays. Everything else, such as
`lmax`, bases, labels and keywords, belongs in the record's closure:

```python exec jax
import jax
import jax.numpy as jnp
import numpy as np
from treams_rs import diff
from treams_rs import jax as tr

jax.config.update("jax_enable_x64", True)


def record_cluster(k0, radii, epsilon, positions):
    return diff.sphere_cluster(1, float(k0), radii, epsilon, positions)  # lmax = 1


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

## Choosing the gradients

A record differentiates every input it takes. Some contexts return gradients for
more inputs than an objective needs. The context of `diff.sphere` always returns
five gradients, `(k0, radii, epsilon, mu, kappa)`, even when `mu` and `kappa`
take their defaults. A small pullback closure selects and orders the gradients:

```python exec torch
import torch
from treams_rs import diff
from treams_rs import torch as tr


def record_sphere(radii, epsilon):
    matrix, context = diff.sphere(2, 1.2, radii, epsilon)

    def pullback(cotangent):
        _, d_radii, d_epsilon, _, _ = context.pullback(cotangent)
        return d_radii, d_epsilon

    return matrix, pullback


sphere = tr.wrap(record_sphere)
radii = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
epsilon = torch.tensor([3.0 + 0.1j, 1.0], dtype=torch.complex128, requires_grad=True)
sphere(radii, epsilon).abs().square().sum().backward()
```

The same pattern selects the axial gradients of cylindrical records.
`context.pullback_axial` appends them to the other gradients:

- `diff.field` and `diff.field_operator` return one gradient per mode, matching
  `kz`, the axial wavenumber of each mode (`basis.kz` in treams);
- `diff.expansion` and `diff.lattice_expansion` return one gradient per distinct
  value, matching `kzs`, the sorted distinct axial wavenumbers of both bases
  (`TMatrixC.cylinder(kzs, ...)` in treams).

The docstring of each `diff` record states its gradients. The adapter never
guesses which inputs are differentiable.

## Records by family

Every `diff` record has the same name in `treams_rs.advect`, except the reusable
factors. JAX and PyTorch offer five records directly: `solve`, `interaction`,
`illuminate`, `sphere` and `bessel`; `wrap` covers the others. The
[API reference](../reference/python/index.md) lists every signature and gradient
order.

| Family | `diff` record | Advect function | JAX and PyTorch | Physics object |
|---|---|---|---|---|
| Sphere T-matrix | `sphere` | `sphere` | `sphere` | `sphere_tmatrix`, `multilayer_sphere_tmatrix` |
| Mie coefficients | `mie`, `mie_cyl` | `mie`, `mie_cyl` | `wrap` | none |
| Cylinder T-matrix | `cylinder` | `cylinder` | `wrap` | `cylinder_tmatrix`, `multilayer_cylinder_tmatrix` |
| Sphere cluster (dense) | `sphere_cluster` | `sphere_cluster` | `wrap` | none (same result: `Cluster(...).solve()`) |
| Heterogeneous cluster | `particle_cluster` | `particle_cluster` | `wrap` | none (same result: `Cluster(...).solve()`) |
| Interaction and illumination | `interaction`, `illuminate` | `interaction`, `illuminate` | `interaction`, `illuminate` | `Cluster.solve()`, `Cluster.scatter()` |
| Reusable factors | `factor_interaction`, `factor_interaction_blocks`, `sphere_cluster_factor`; their `InteractionFactor.record` | none | `wrap` of `factor.record` | none |
| Matrix-free sphere cluster | `iterative.SphereCluster.record` | none | none | none |
| Fields | `field`, `field_operator`, `plane_field` | `field`, `hfield`, `gfield`, `ffield`, `field_operator`, `plane_field` | `wrap` | `efield`, `hfield`, `dfield`, `bfield`, `gfield`, `ffield` of waves |
| Expansion and translation | `expansion`, `spherical_translation`, `cylindrical_translation`, `plane_expansion`, `plane_phases` | the same names | `wrap` | `in_basis` of waves |
| Rotation and permutation | `rotation`, `plane_permutation` | `rotation`, `plane_permutation` | `wrap` | none |
| Periodic expansion | `lattice_expansion` | `lattice_expansion` | `wrap` | `solve_periodic(...)` |
| Lattice sums and tables | `lattice_sum`, `lattice_expansion_from_table`, `periodic_to_cw` | the same names | `wrap` | none |
| Radiation channels | `spherical_channels`, `cylindrical_channels` | the same names | `wrap` | `PeriodicResponse.to_smatrix(...)` |
| Interfaces and stacks | `fresnel`, `interface_coefficients`, `propagation_matrix`, `layer_stack` | the same names | `wrap` | `interface`, `slab`, `multilayer_slab`, `propagation`, `stack` |
| S-matrix networks | `smatrix_add`, `smatrix_illuminate`, `smatrix_tr`, `smatrix_from_array`, `smatrix_periodic`, `bands` | the same names, and `smatrix_cd` | `wrap` | `cascade`, `scatter`, `power`, `bands` of `SMatrix` |
| Chirality and response metrics | `chirality_density`, `oriented_chirality`, `tmatrix_metric` | the same names | `wrap` | none |
| Axisymmetric EBCM | `ebcm_qmat` | `ebcm_qmat` | `wrap` | none |
| Special functions | `bessel`, `incgamma`, `intkambe`, `angular`, `wignerd`, `sph_harm`, `vector_wave` | the same names | `bessel`; `wrap` | none |
| Coordinates | `coordinates`, `vector_coordinates` | the same names | `wrap` | none |
| Linear algebra | `solve`, `eig`, `svdvals` | `solve`, `eig`, `svdvals` | `solve`; `wrap` | none |

`Cluster.solve()` builds its result from `interaction` and `expansion`.
`Cluster.scatter()` factors the interaction internally for one solve; reuse of
a factor across solves needs the factor records.

The Advect functions `interface_coefficients` and `propagation_matrix` return
coefficient arrays; `interface` and `propagation` build physical S-matrices.

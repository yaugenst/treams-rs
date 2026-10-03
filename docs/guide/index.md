---
description: The physics objects of treams-rs, their metadata and read-only arrays, and which computation path fits a problem.
---

# User guide

## Objects carry their physics

treams-rs describes a computation with objects that keep their physical
metadata as attributes:

| Object | Holds | Made by |
| --- | --- | --- |
| `TMatrix`, `CylindricalTMatrix` | the response of one particle or a solved cluster | `sphere_tmatrix`, `cylinder_tmatrix`, `Cluster.solve` |
| `Wave`, `PlaneWave` | wave coefficients with their basis and `kind` | `plane_wave`, `spherical_wave`, `scatter` |
| `Cluster` | particles and positions, not yet solved | `Cluster(particles, positions=...)` |
| `PeriodicResponse`, `PeriodicWave` | the response of a lattice and its waves | `solve_periodic`, `PeriodicResponse.scatter` |
| `SMatrix` | a planar two-port network | `interface`, `slab`, `propagation`, `stack`, `PeriodicResponse.to_smatrix` |

Each object knows its basis, `k0`, surrounding `medium` and polarization
convention. Operations check that these agree and return objects again:
`scatter` returns a `Wave`, `power` returns named `transmission` and
`reflection`, `cross_sections` returns named `scattering` and `extinction`.

```python exec
import numpy as np
import treams_rs as tr

sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3, medium=1.21)
assert (sphere.k0, sphere.medium.epsilon) == (1.3, 1.21)
assert sphere.polarization == "helicity"  # the default convention
assert sphere.array.shape == (16, 16)  # 2 pol x 8 (l, m) pairs for lmax = 2
assert not sphere.array.flags.writeable  # call .copy() to edit
parity = sphere.with_polarization("parity")
np.testing.assert_allclose(
    parity.with_polarization("helicity").array, sphere.array, atol=1e-15
)
```

`.array` and `.coefficients` give read-only NumPy arrays. NumPy arithmetic on
them returns plain arrays without metadata. The polarization convention is an
argument of each factory, with `"helicity"` as the default, never a global
setting.

Units are up to you: lengths and `1 / k0` share one unit. Numbers passed as a
material are relative permittivities; `Material(epsilon, mu, kappa)` adds
permeability and chirality. `material` names the particle or layer and `medium`
the space around it. See [Conventions](../coming-from-treams/conventions.md) for
the units, mode ordering and polarization in detail.

## Choose a computation path

| Need | Use | Guide |
| --- | --- | --- |
| One particle: fields and cross sections | `sphere_tmatrix`, `cylinder_tmatrix`, `plane_wave`, `scatter` | [Particles and waves](particles-and-waves.md) |
| A few incident waves on a cluster | `Cluster.scatter`, or `Cluster.factor()` to reuse one LU factorization | [Clusters](clusters.md) |
| The full response of a cluster | `Cluster.solve()` | [Clusters](clusters.md) |
| A large cluster of homogeneous spheres in vacuum | `iterative.SphereCluster`: GMRES without a dense matrix | [Large clusters](large-clusters.md) |
| A periodic array | `solve_periodic`, then `to_smatrix` or `scatter` | [Periodic arrays](periodic.md) |
| Interfaces, slabs and layer stacks | `interface`, `slab`, `propagation`, `stack` | [Planar layers](planar.md) |
| treams functions such as `sw.translate` | `special`, `sw`, `cw`, `pw`, `lattice`, `coeffs`, `misc`, `ebcm`, `operators` | [Numerical namespaces](numerical-namespaces.md) |
| HDF5 T-matrix files | `io.save_hdf5`, `io.load_hdf5` (extra `[io]`) | [HDF5 and Gmsh](io.md) |
| Gradients | `treams_rs.advect`, `.jax`, `.torch`; `diff` records | [Differentiation](../differentiation/index.md) |
| Fewer threads, process pools, threadpoolctl | `set_num_threads`, `threads`, `thread_info` | [Threads and process pools](threads.md) |
| The installed API from a program | `python -m treams_rs`, `support_catalog()` | [API discovery](api-discovery.md) |

The [Python reference](../reference/python/index.md) lists every signature.

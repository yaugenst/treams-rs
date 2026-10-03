---
description: T-matrix electromagnetic scattering with a Rust core, a typed Python API and analytic gradients, in the conventions of treams.
---

# treams-rs

treams-rs computes electromagnetic scattering with T-matrices: spheres,
cylinders and layered or chiral particles, finite clusters, periodic arrays and
planar stacks, with their fields, cross sections and power. A T-matrix is the
linear map from the multipole coefficients of an incident wave to those of the
scattered wave. A Rust core does the numerical work and a typed Python API
describes the physics.

treams-rs follows the numerical conventions of
[treams](https://github.com/tfp-photonics/treams) 0.4.5 and keeps its numerical
namespaces (`special`, `sw`, `cw`, `pw`, `lattice`, `coeffs`, `misc`, `ebcm`,
`io`) with the same function names. The physics objects are its own; the
[Coming from treams](coming-from-treams/index.md) section maps one to the
other.

## In 30 seconds

```python exec
import numpy as np
import treams_rs as tr

sphere = tr.sphere_tmatrix(k0=2.0, lmax=3, radius=0.5, material=4)
incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=2.0)
cross_sections = sphere.cross_sections(incident)
# A lossless sphere scatters all the power it removes from the incident wave.
np.testing.assert_allclose(cross_sections.extinction, cross_sections.scattering)
# The extinction cross section of treams.TMatrix.sphere for the same sphere.
np.testing.assert_allclose(cross_sections.extinction, 0.6258290233441384, rtol=1e-12)
```

## What treams-rs adds

- **Analytic gradients.** `treams_rs.advect`, `treams_rs.jax` and
  `treams_rs.torch` provide the physics objects and factories for each
  framework, so `grad` works through T-matrices, clusters, periodic arrays and
  S-matrices. Rust computes each derivative analytically, without finite
  differences. See [Differentiation](differentiation/index.md).
- **Speed.** The median speedup over treams 0.4.5 is 5.1× across 527 measured
  Linux CPU cases; see [Performance](performance/index.md).
- **Requested illuminations.** `Cluster.scatter` and `Cluster.factor` solve
  only for the incident waves you ask for, without building the full T-matrix
  of the cluster.
- **Matrix-free solves.** `iterative.SphereCluster` solves sphere clusters
  with GMRES, an iterative solver that needs only matrix-vector products, and
  never stores the dense coupling matrix. For 1,024 spheres
  with `lmax=1` on four Linux threads, the solve takes 2.2 s and its gradient
  2.8 s, within 48 MiB. See [Large clusters](guide/large-clusters.md).

## What it does not do

- **No GPU.** All computations run on the CPU, in parallel
  [threads](guide/threads.md).
- **First-order gradients only.** No forward mode and no second derivatives.
- **No annotated arrays.** NumPy arithmetic on `.array` returns plain arrays;
  physical metadata lives on the objects. The treams `PhysicsArray` is
  available in `operators` for operator evaluation.
- **No global configuration.** Each object and function takes its
  polarization convention as an argument; there is no `config.POLTYPE`.

## Where to go next

| Section | Contents |
| --- | --- |
| [Getting started](getting-started/install.md) | Install and four short examples |
| [Coming from treams](coming-from-treams/index.md) | Workflows, the name map, conventions and differences from treams |
| [User guide](guide/index.md) | Particles, clusters, periodic arrays, planar stacks and the numerical namespaces |
| [Examples](examples/index.md) | The treams gallery in treams-rs, plus gradient-based design |
| [Differentiation](differentiation/index.md) | Gradients through Advect, JAX, PyTorch and `diff` |
| [Design](design/index.md) | Why treams-rs is built the way it is |
| [Validation](validation/index.md) | How the results are tested, and their limits |
| [Performance](performance/index.md) | Timings and memory against treams |
| [Reference](reference/index.md) | Every public function and class, and the glossary |
| [Development](development/index.md) | Setup and checks for contributors |

---
description: A Rust port of treams with a Python interface and analytic gradients.
---

# treams-rs

treams-rs is a Rust port of [treams](https://github.com/tfp-photonics/treams)
with a Python interface and analytic gradients.

**All credit for treams belongs to the original treams contributors.**
Their work is described in the
[treams paper](https://doi.org/10.1016/j.cpc.2023.109076); please cite it when
using this port.

treams-rs computes electromagnetic scattering with T-matrices: spheres,
cylinders and layered or chiral particles, finite clusters, periodic arrays and
planar stacks, with their fields, cross sections and power. A T-matrix is the
linear map from the multipole coefficients of an incident wave to those of the
scattered wave.

treams-rs follows the numerical conventions of
[treams](https://github.com/tfp-photonics/treams) 0.4.5 and keeps its numerical
modules (`special`, `sw`, `cw`, `pw`, `lattice`, `coeffs`, `misc`, `ebcm`,
`io`) with the same function names. Its Python interface differs;
[Coming from treams](coming-from-treams/index.md) shows the corresponding
functions and objects.

## Example

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
  `treams_rs.torch` let you differentiate calculations involving T-matrices,
  clusters, periodic arrays and S-matrices with each library's differentiation
  functions.
  Rust computes each derivative analytically, without finite
  differences. See [Differentiation](differentiation/index.md).
- **Measured performance.** Archived Linux CPU benchmarks recorded a median
  speedup of 5.1× over treams 0.4.5 across 527 cases. The results apply to the
  measured builds and machine; see [Performance](performance/index.md).
- **Requested illuminations.** `Cluster.scatter` and `Cluster.factor` solve
  only for the incident waves you ask for, without building the full T-matrix
  of the cluster.
- **Matrix-free solves.** `iterative.SphereCluster` solves sphere clusters
  with GMRES, an iterative solver that needs only matrix-vector products, and
  does not store the dense coupling matrix. An archived benchmark of 1,024
  spheres with `lmax=1` on four Linux threads recorded 2.2 s for the solve and
  2.8 s for its gradient, within 48 MiB. See [Large clusters](guide/large-clusters.md).

## Limits

- **CPU execution.** Computations use CPU [threads](guide/threads.md), not GPUs.
- **First-order gradients.** Forward-mode differentiation and second derivatives
  are unsupported.
- **NumPy arithmetic.** Operations on `.array` return ordinary arrays. The
  objects retain the basis, wavenumber and media. The treams `PhysicsArray` is
  available in `operators` for evaluating operators.
- **Explicit polarization.** Each object and function takes its polarization
  convention as an argument; there is no global `config.POLTYPE` setting.

## Where to go next

| Section | Contents |
| --- | --- |
| [Getting started](getting-started/install.md) | Install and four short examples |
| [Coming from treams](coming-from-treams/index.md) | Workflows, the name map, conventions and differences from treams |
| [User guide](guide/index.md) | Particles, clusters, periodic arrays, planar stacks and numerical functions |
| [Examples](examples/index.md) | The treams gallery in treams-rs, plus gradient-based design |
| [Differentiation](differentiation/index.md) | Gradients through Advect, JAX, PyTorch and `diff` |
| [Design](design/index.md) | Why treams-rs is built the way it is |
| [Validation](validation/index.md) | How the results are tested, and their limits |
| [Performance](performance/index.md) | Timings and memory against treams |
| [Reference](reference/index.md) | Every public function and class, and the glossary |
| [Development](development/index.md) | Setup and checks for contributors |

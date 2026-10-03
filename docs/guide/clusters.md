---
description: Solve a finite cluster for its full response, scatter requested illuminations, reuse a factor and expand into one global basis.
---

# Clusters

A `Cluster` stores particles and their positions. Choose how to solve their
multiple scattering by the response you need:

| Method | Computes | Returns |
| --- | --- | --- |
| `cluster.scatter(incident)` | the response to the given incident waves | the scattered `Wave` |
| `cluster.solve()` | the response to every incident mode | the `TMatrix` of the cluster |
| `cluster.factor()` | an LU factorization for later illuminations | a `ScatteringFactor` with `scatter` |

```python exec
import numpy as np
import treams_rs as tr

particles = [
    tr.sphere_tmatrix(k0=1.3, lmax=2, radius=r, material=3) for r in (0.15, 0.2)
]
cluster = tr.Cluster(particles, positions=[[0, 0, 0], [0.8, 0, 0]])
incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.3)
requested = cluster.scatter(incident)
complete = cluster.solve()
np.testing.assert_allclose(
    requested.coefficients, complete.scatter(incident).coefficients
)
factor = cluster.factor()
np.testing.assert_allclose(
    factor.scatter(incident).coefficients, requested.coefficients
)
```

Use `scatter` for one or a few incident waves, `factor` when more incident
waves follow later, and `solve` when you need the response to every incident
mode, for example for orientation averages.

## Different particles and a global basis

The particles of a cluster may differ in size, material and `lmax`. They must
share `k0`, the surrounding medium and the polarization convention. The
solved T-matrix uses one expansion centre per particle. `in_basis` expands it
around a single origin, which orientation averages need:

```python exec
import numpy as np
import treams_rs as tr

sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
chiral = tr.sphere_tmatrix(
    k0=1.3, lmax=3, radius=0.25, material=tr.Material(epsilon=4, kappa=0.05)
)
cluster = tr.Cluster([sphere, chiral], positions=[[0, 0, -0.35], [0, 0, 0.35]])
local = cluster.solve()
assert len(local.basis) == 16 + 30  # each particle keeps its own modes
global_ = local.in_basis(tr.SphericalBasis.default(8))
incident = tr.plane_wave(direction=[1, 0, 0], pol="positive_helicity", k0=1.3)
np.testing.assert_allclose(
    global_.cross_sections(incident).scattering,
    local.cross_sections(incident).scattering,
    rtol=1e-9,
)
print(global_.average_cross_sections)
```

The global expansion converges with its `lmax`: 4, 6 and 8 give relative
errors of 1e-4, 2e-7 and 1e-10 here. The radius of the smallest sphere that
encloses all particles sets how large `lmax` must be.

## Using a solved T-matrix

The solved `TMatrix` keeps its metadata. `select(basis)`, `in_basis(basis)`,
`rotate(...)` and `with_polarization(...)` return physical objects;
`.array[...]` gives plain NumPy slices.

For clusters of many homogeneous spheres, the matrix-free solver avoids the
dense matrices: see [Large clusters](large-clusters.md). Cylinders form
clusters the same way, with a `CylindricalTMatrix` per particle.

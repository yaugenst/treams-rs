# Requested illuminations without a dense cluster matrix

`iterative.SphereCluster` solves finite clusters of homogeneous nonmagnetic
spheres in vacuum. It computes only the supplied incident multipole columns.
It never builds the global interaction matrix or the full interacting T-matrix.

```python
import numpy as np
from treams_rs.iterative import SphereCluster

cluster = SphereCluster(
    lmax=2,
    k0=1.3,
    radii=[0.2, 0.3],
    epsilon=[2.4, 3.1 + 0.02j],
    positions=[[0, 0, 0], [1.1, 0.2, 0]],
)
# One selected regular-multipole illumination; arbitrary complex columns work.
incident = np.zeros(cluster.dimension, dtype=complex)
incident[0] = 1
solution, context = cluster.solve_with_pullback(incident, rtol=1e-11, restart=30)
assert solution.coefficients.shape == incident.shape
assert all(r.residual_norm <= 1e-11 * r.rhs_norm for r in solution.convergence)

# Gradient of the squared norm of these scattered multipole coefficients.
gradient = context.pullback(2 * solution.coefficients)
print(gradient.radii, gradient.positions, gradient.epsilon, gradient.k0)
```

The incident coefficients use the same local helicity ordering as the native
sphere cluster: particle index, then spherical modes with positive helicity
before negative helicity. A vector describes one illumination; a matrix of
shape `(dimension, P)` describes P requested illuminations. Changing geometry
holds these supplied coefficients fixed. If an incident field itself depends
on geometry, compose its separate pullback with `gradient.incident`.

The solver applies `A = I - T C` and solves `A X = T B` using restarted GMRES.
Each particle stores only its 2×2 Mie blocks per multipole degree. Rayon workers
evaluate pair translations and immediately apply them to vectors. The adjoint
solves `Aᴴ Λ = G`, then contracts low-rank pair cotangents directly into
position and wavenumber derivatives. It retains no Krylov history and no global
square cotangent. Multiple recorded solves share the native geometry and
translation plan.

For N particles, M local modes, P requested columns and restart R, Krylov and
solution workspace scales as `O(N M (R + P) + R²)`. Each worker has one `M²`
pair block. The geometry-independent angular translation plan is shared and
scales with local multipole order, independently of particle count.

Every forward and adjoint column checks the actual residual after convergence,
using `||residual|| <= max(atol, rtol * ||rhs||)`. Exhausting `max_iterations`,
a singular Arnoldi system, or a nonfinite residual raises `ValueError`.
Tighter tolerances can be necessary near resonances; a small residual does not
bound solution error in an ill-conditioned system. Pair translations are
recomputed on every iteration, so this is a memory-saving option whose speed
and convergence must be measured for the intended physical configuration.

# Large clusters

The full T-matrix of a cluster gives the response to every incident mode.
For a few incident waves, two solvers compute only the responses you need:

| Solver | Stores | Best for |
| --- | --- | --- |
| `Cluster.factor()`, `diff.sphere_cluster_factor` | the dense coupling matrix and its LU factorization | many incident waves on one geometry, while the dense matrix fits in memory |
| `iterative.SphereCluster` | one 2×2 Mie block per sphere and degree | many spheres, when memory matters more than the time per incident wave |

For 512 spheres with `lmax=1` and one incident wave, the dense solve for that
wave takes 0.38 s against 0.94 s for the full T-matrix. The matrix-free
gradient peaks at 46 MiB of memory, the gradient through the full T-matrix at
1,368 MiB. See [Large problems](../performance/large-problems.md) for the
measurements.

## Matrix-free sphere clusters

`iterative.SphereCluster` solves clusters of homogeneous, nonmagnetic spheres
in vacuum. It takes the incident field as a physical wave or as coefficient
columns, and agrees with the dense `Cluster`:

```python
import numpy as np
import treams_rs as tr
from treams_rs.iterative import SphereCluster

radii, epsilon = [0.2, 0.3], [2.4, 3.1 + 0.02j]
positions = [[0, 0, 0], [1.1, 0.2, 0]]
cluster = SphereCluster(2, 1.3, radii, epsilon, positions)  # lmax, k0, ...
incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.3)
solution = cluster.scatter(incident, rtol=1e-11)
spheres = [
    tr.sphere_tmatrix(k0=1.3, lmax=2, radius=r, material=e)
    for r, e in zip(radii, epsilon)
]
dense = tr.Cluster(spheres, positions=positions).scatter(incident)
np.testing.assert_allclose(solution.wave.coefficients, dense.coefficients, atol=1e-12)
```

`scatter` returns a `ScatteringSolution(wave, convergence)`. For coefficient
columns, `solve` returns a `Solution(coefficients, convergence)`, and `record`
returns the solution together with an `IterativeContext` for gradients.

## Coefficient columns and gradients

The incident coefficients are ordered by particle, then by the spherical modes
of each particle, with pol 1 (positive helicity) before pol 0. A vector is one
incident wave; a matrix of shape `(dimension, P)` holds P of them.

```python
import numpy as np
from treams_rs.iterative import Gradient, IterativeContext, SphereCluster

cluster = SphereCluster(
    lmax=2,
    k0=1.3,
    radii=[0.2, 0.3],
    epsilon=[2.4, 3.1 + 0.02j],
    positions=[[0, 0, 0], [1.1, 0.2, 0]],
)
incident = np.zeros(cluster.dimension, dtype=complex)
incident[0] = 1  # the first mode of the first sphere
solution, context = cluster.record(incident, rtol=1e-11, restart=30)
assert isinstance(context, IterativeContext)
assert all(c.residual_norm <= 1e-11 * c.rhs_norm for c in solution.convergence)

# Gradient of the squared norm of the scattered coefficients.
gradient = context.pullback(2 * solution.coefficients)
assert isinstance(gradient, Gradient)
fields = ("k0", "radii", "epsilon", "positions", "incident", "convergence")
assert Gradient._fields == fields
assert gradient.positions.shape == (2, 3)
```

`context.pullback(g)` takes the gradient `g` of a real loss with respect to the
solution and returns a `Gradient` with one entry per input of `SphereCluster`
and `record`, in their order, followed by the convergence of the adjoint
solves. An adjoint solve is the linear system with the conjugate transpose
matrix that carries the gradient back to the inputs.
[Differentiation](../differentiation/index.md) defines how complex gradients
pair with their values. A context allows one pullback; record again for a second.

The gradients hold the incident coefficients fixed while the geometry changes.
If the incident field depends on the geometry, apply its own pullback to
`gradient.incident` and add the result.

## The dense factor for coefficient columns

`diff.sphere_cluster_factor` builds the dense coupling matrix of the same
spheres and factors it once. Every `solve` reuses that factorization:

```python
import numpy as np
from treams_rs import diff
from treams_rs.iterative import SphereCluster

lmax, k0 = 1, 1.3
radii = np.array([0.2, 0.25])
epsilon = np.array([2.4, 3.1 + 0.02j])
positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0]])
incident = np.ones((12, 2), dtype=complex)

factor = diff.sphere_cluster_factor(lmax, k0, radii, epsilon, positions)
selected = factor.solve(incident)
other = factor.solve(incident * (0.3 + 0.2j))
np.testing.assert_allclose(other, selected * (0.3 + 0.2j), rtol=1e-12)

matrix_free = SphereCluster(lmax, k0, radii, epsilon, positions)
solution = matrix_free.solve(incident, rtol=1e-11)
np.testing.assert_allclose(solution.coefficients, selected, rtol=1e-9, atol=1e-13)
```

`factor.record(incident)` returns the solution and an `IlluminateContext`,
whose pullback gives the gradients with respect to the T-matrix blocks of the
spheres, the coupling matrix and the incident coefficients. Chain them with
the pullbacks of `diff.sphere` and `diff.expansion` for gradients with respect
to radii, permittivities and positions.

## Convergence

The solver applies `A = I - T C` to vectors, where `T` holds the Mie blocks and
`C` the translations between spheres, and solves `A X = T B` with GMRES. GMRES
is an iterative solver that needs only products of `A` with vectors. It builds
its approximation from at most `restart` stored vectors (the Krylov vectors),
then starts again from the current solution. After GMRES stops, the solver
computes the residual `T B - A X` of every column, which is zero for an exact
solution, and accepts the column only if

```text
||residual|| <= max(atol, rtol * ||rhs||)
```

The defaults are `rtol = 1e-10`, `atol = 0`, `restart = 30` and
`max_iterations = 300` (also available as constants such as
`iterative.DEFAULT_RTOL`). The adjoint solves `Aᴴ Λ = G` with the same check.
A solve that reaches
`max_iterations`, breaks down because GMRES cannot extend its stored vectors,
or produces a non-finite residual raises `ValueError`. Each `Convergence(iterations, residual_norm,
rhs_norm)` reports one column.

A small residual does not bound the error of an ill-conditioned system: near
resonances, tighten `rtol` and compare with a dense solve on a smaller cluster.

## Memory and time

For N spheres, M modes per sphere, P incident columns and restart R, the
storage for the Krylov vectors and the solution grows as `O(N M (R + P) + R²)`. Each thread holds
one `M × M` pair block. The angular part of the translations is computed once
per `lmax` and shared by every pair, so it does not grow with N.

The solver recomputes the pair translations in every iteration and keeps no
Krylov vectors for the adjoint solve. It trades time for memory: a reused dense
factor computes another response in milliseconds, while the matrix-free solver
runs GMRES again. Measure both for your geometry.

The dense `diff.sphere_cluster` forward peaks at about four complex `NM × NM`
buffers, `64 (NM)²` bytes: the coupling, the LU factors, the solution and its
NumPy copy. These four allocations raise `MemoryError` when refused. Other
allocations, including gradient buffers and matrix products, can still abort
when refused. On Linux, memory overcommit can grant a reservation before the
out-of-memory killer later ends the process as its pages are written.

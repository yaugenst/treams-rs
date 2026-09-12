# Requested illuminations and larger clusters

Use `diff.cluster_factor` when several illuminations share a geometry and its
dense LU factorization fits in memory. Use [`iterative.SphereCluster`](iterative.md) when avoiding
the dense coupling and factorization matters more than repeated-solve latency.
Both compute only the incident columns requested by the caller.

```python
import numpy as np
from treams_rs import diff
from treams_rs.iterative import SphereCluster

lmax, k0 = 1, 1.3
radii = np.array([0.2, 0.25])
epsilon = np.array([2.4, 3.1 + 0.02j])
positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0]])
incident = np.ones((12, 2), dtype=complex)

# Native assembly and one LU, without computing the full interacting T matrix.
factor = diff.cluster_factor(lmax, k0, radii, epsilon, positions)
selected = factor.solve(incident)
other = factor.solve(incident * (0.3 + 0.2j))
np.testing.assert_allclose(other, selected * (0.3 + 0.2j), rtol=1e-12)

# No global dense coupling, LU or dense gradient matrix.
operator = SphereCluster(lmax, k0, radii, epsilon, positions)
solution, context = operator.solve_with_pullback(incident, rtol=1e-11)
np.testing.assert_allclose(solution.coefficients, selected, rtol=1e-9, atol=1e-13)
gradient = context.pullback(2 * solution.coefficients)
```

The matrix-free adjoint returns radius, position, permittivity, wavenumber and
incident-coefficient cotangents. `cluster_factor.record` returns local T-block,
coupling and incident cotangents. Compose the sphere and expansion pullbacks
when physical derivatives are needed from that dense factor. The benchmark
includes this complete chain; its additional geometry contexts and assembly
cost are reported in the adjoint phase. Forward-only selected LU uses the
optimized native cluster assembly directly.

## Reproduce the qualification

```sh
just build-ext-release
uv run --no-sync python scripts/benchmark_illumination.py \
  --particles 512 --lmax 1 --columns 1 --threads 4 \
  --output benchmarks/results/illumination-local-n512-l1-p1.json
uv run --no-sync python scripts/benchmark_illumination.py \
  --particles 1024 --lmax 1 --columns 1 --threads 4 \
  --output benchmarks/results/illumination-local-n1024-l1-p1.json
```

The script uses separate fresh processes for every backend and forward/adjoint
phase, limits BLAS and Rayon workers, runs one warmup and three timing samples,
and captures binary, Python-source and benchmark SHA-256 hashes. It reports
geometry/assembly/factorization setup, a fresh end-to-end solve, reused-geometry
or reused-factor solve, and the complete native physical pullback separately.
The common plane-wave coefficient construction is timed separately and excluded
from solver comparisons. Peak RSS includes imports, inputs, warmup and actual
recorded execution; finite-difference validation happens after the RSS capture.

All paths use complex128, the same particle geometry and multipole cutoff,
and the same requested plane-wave coefficient columns. The strongest available
native full-cluster API provides the full-T reference. Small cases are also
checked against upstream treams 0.4.5. Matrix-free forward and adjoint solves
independently recompute and enforce their true residual tolerances. Physical
cotangents are compared across backends, checked against translation/scaling
and illumination-amplitude invariants, and checked with a radius-direction
finite difference in the largest matrix-free case as well.

Dense comparisons are explicitly limited to dimension 3,072 by default. Above
that bound, only matrix-free runtime, RSS, residuals, invariants and directional
derivatives are measured. No same-size dense speedup is claimed for skipped
cases. A multipole cutoff of one is a specified dipole model, not a claim of
multipole convergence for arbitrary spheres.

## Results

The final grid contains nine cases and 46 isolated worker measurements. The Mac
was an Apple M3 (8 cores, 24 GiB RAM), with four numerical workers. Linux used
an AMD Ryzen 9 9950X, with four workers pinned to CPUs 8–11. Both used the same
Python sources, benchmark, complex128 precision and GMRES tolerance `1e-10`.
The default Linux extension had CUDA completely compiled out.

For 512 weakly scattering spheres, `lmax=1`, one illumination and 3,072 channels:

| Host | Method | Fresh solve (s) | Reuse (ms) | Physical pullback (s) | Forward peak RSS (MiB) | Adjoint peak RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|
| [Linux](../benchmarks/results/illumination-linux-n512-l1-p1.json) | Full T | 0.940 | 2.684 | 1.885 | 640.4 | 1367.7 |
| [Linux](../benchmarks/results/illumination-linux-n512-l1-p1.json) | Selected LU | 0.381 | 4.317 | 0.764 | 351.6 | 861.6 |
| [Linux](../benchmarks/results/illumination-linux-n512-l1-p1.json) | Matrix-free | 0.544 | 541.090 | 0.711 | 44.9 | 45.6 |
| [Mac](../benchmarks/results/illumination-mac-n512-l1-p1.json) | Full T | 2.902 | 2.495 | 6.342 | 690.7 | 1461.0 |
| [Mac](../benchmarks/results/illumination-mac-n512-l1-p1.json) | Selected LU | 0.959 | 6.571 | 0.741 | 367.1 | 919.5 |
| [Mac](../benchmarks/results/illumination-mac-n512-l1-p1.json) | Matrix-free | 0.794 | 854.252 | 1.070 | 41.0 | 42.1 |

On Linux, selected LU reduced a fresh solve by **2.47×**. Matrix-free used
**30× less peak RSS for the complete physical adjoint** than the full-T path,
and 18.9× less than the selected-LU physical chain. On the Mac, the corresponding
full-T adjoint memory reduction was 34.7×. These are process high-water marks,
including the Python runtime, not estimates of matrix storage or steady-state RSS.

The larger case doubled the particle/channel count to 1,024 spheres and 6,144
channels. Both forward and adjoint GMRES converged in six iterations:

| Host | Fresh solve (s) | Physical pullback (s) | Adjoint peak RSS (MiB) | Radius-direction derivative relative error |
|---|---:|---:|---:|---:|
| [Linux](../benchmarks/results/illumination-linux-n1024-l1-p1.json) | 2.192 | 2.826 | 47.9 | 2.19e-08 |
| [Mac](../benchmarks/results/illumination-mac-n1024-l1-p1.json) | 3.377 | 4.411 | 44.9 | 2.19e-08 |

This larger case was qualified by true residuals, physical gradient invariants
and an independent finite difference. Its dense reference was deliberately
skipped under the explicit dimension limit; no same-size dense speedup is claimed.

Matrix-free is not the fastest choice for every workload. The moderate-scattering
128-sphere, four-illumination case took 0.208 s matrix-free versus 0.0183 s with a
fresh selected LU on Linux. With a stored full T matrix or LU, repeated responses
can take milliseconds while matrix-free must recompute translations. Choose the
method using the measured memory budget, number of illuminations and convergence.

The rest of the grid is retained, including the slower matrix-free cases:

- 128 spheres, dipole order, one weak-scattering illumination:
  [Linux](../benchmarks/results/illumination-linux-n128-l1-p1.json),
  [Mac](../benchmarks/results/illumination-mac-n128-l1-p1.json).
- 128 spheres, dipole order, four moderate-scattering illuminations:
  [Linux](../benchmarks/results/illumination-linux-n128-l1-p4-moderate.json),
  [Mac](../benchmarks/results/illumination-mac-n128-l1-p4-moderate.json).
- 64 spheres, `lmax=2`, two moderate-scattering illuminations:
  [Mac](../benchmarks/results/illumination-mac-n64-l2-p2-moderate.json).

Across the reference-comparable cases, the largest relative discrepancy between
native physical gradients was `6.58e-11`. The largest upstream output discrepancy
was `1.99e-15`; radius-direction finite differences agreed within `3.02e-8`
relative across all nine cases. These checks qualify the stated finite set of
physical configurations, not arbitrary high-order or resonant clusters.

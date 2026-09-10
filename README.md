# treams-rs

A personal Rust rewrite of treams, with a typed Python API and native analytic
pullbacks. The numerical core runs without Python or an autodiff framework.

This is an active rewrite, not yet a replacement for all of treams. See
[implementation status](docs/status.md) for verified coverage and remaining work.

```sh
uv sync --group dev
just build-ext
just verify
uv run pre-commit install
```

The [upstream inventory](docs/upstream.md) records codebase size and rewrite scope.
Source and derivative conventions are described in
[architecture](docs/architecture.md). Numerical comparisons use treams only as
a development dependency. See [third-party notices](THIRD_PARTY_NOTICES.md).

```python
import treams_rs as tr

particles = [tr.TMatrix.sphere(3, 1.3, radius, [4 + 0.1j, 1]) for radius in (0.2, 0.25)]
cluster = tr.TMatrix.cluster(particles, [[0, 0, 0], [0, 0, 0.8]])
solution = cluster.interaction.solve()
global_matrix = solution.expand(tr.SphericalWaveBasis.default(8))
print(global_matrix.xs_sca_avg)
```

For derivatives, use `tr.diff.sphere`, `tr.diff.cluster`, `tr.diff.expansion`, or
`tr.diff.interaction` / `tr.diff.field`. Each returns `(array, context)`; call
`context.pullback(output_cotangent)` once. Complex cotangents satisfy
`dL = real(vdot(cotangent, doutput))`. See [test strategy](docs/test-strategy.md).

CPU parallelism uses Rayon for particle coupling and faer dense algebra. Set
`RAYON_NUM_THREADS` **before importing** to choose the thread budget. Build with
`just build-ext-release` before benchmarking; development builds are not comparable.
The benchmark validates output parity before timing isolated solver processes:

```sh
just build-ext-release
uv run --no-sync python scripts/benchmark_cluster.py --particles 16 --lmax 3 --threads 4
```

Measured scope and caveats are in [benchmarks](docs/benchmarks.md).


Optional Advect integration composes ordinary objectives around the native solver:

```python
import advect.numpy as np
from advect import grad
from treams_rs import advect as ad


def loss(radii):
    t, positions = 1.3, np.array([[0, 0, 0], [0, 0, 0.8]])
    matrix = ad.cluster(2, t, radii, np.array([4.0, 3.0]), positions)
    return np.sum(np.real(matrix * np.conj(matrix)))


print(grad(loss)(np.array([0.2, 0.25])))
```

Install `treams-rs[advect]` when using a wheel, or `uv sync --extra advect`
from source. This is a first-order, one-use VJP contract. A new forward call is
required for another VJP; forward mode, higher-order derivatives, staging, checkpointing, and Jacobian-building
helpers that repeatedly call one residual are unsupported.

Batched Cartesian electric fields use `tr.diff.field(coefficients, points, basis, ks)`.
The output has shape `(number_of_points, 3)`. Its pullback returns cotangents for
amplitudes, sample coordinates, basis origins, and the two helicity wavenumbers.
`tr.advect.field` composes this calculation with scattering and ordinary NumPy
objectives; `tr.advect.expansion` similarly differentiates basis translations.

Plane-wave illumination can be passed directly to either matrix family:

```python
wave = tr.plane_wave([0, 0, 1], [1, 0, 0], k0=1.3)
sphere = tr.TMatrix.sphere(4, 1.3, 0.25, [4.0, 1.0])
scattered = sphere @ wave
print(sphere.xs(wave))
```

The illumination may also be expanded explicitly with `wave.expand(basis)`.
`plane_wave_angle(theta, phi, pol, ...)` accepts angles in radians. Illumination
parameters currently remain ordinary constants when used in an Advect objective;
scattering and field parameters retain their native VJPs.

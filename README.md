# treams-rs

T-matrix electromagnetic scattering with a Rust numerical core, a typed Python
API, and native analytic derivatives. The core runs independently of Python and
autodiff frameworks; there is no runtime fallback to treams, SciPy, or Cython.

The package covers spherical, cylindrical, and plane waves; layered and chiral
particles; finite clusters and periodic arrays; planar stacks; fields and power
observables. Python constructors and numerical conventions follow treams 0.4.5,
with an explicit-object API described in the [capability reference](docs/status.md).

This branch preserves experimental CUDA support. `main` is the authoritative
CPU core; the browser playground is isolated on
`experimental/browser`. CUDA development and hardware qualification
remain deferred.

## Getting started

From a source checkout, install the tools listed in the
[development guide](docs/development.md), then build the Python extension:

```sh
uv sync --locked --group dev
just build-ext-release
```

Run Python with `uv run --no-sync python` in this environment. The installed
package needs only NumPy at runtime. A two-sphere calculation is:

```python
import treams_rs as tr

particles = [tr.TMatrix.sphere(3, 1.3, radius, [4 + 0.1j, 1]) for radius in (0.2, 0.25)]
cluster = tr.TMatrix.cluster(particles, [[0, 0, 0], [0, 0, 0.8]])
solution = cluster.interaction.solve()
global_matrix = solution.expand(tr.SphericalWaveBasis.default(8))
print(global_matrix.xs_sca_avg)
```

Objects carry explicit basis and material metadata; `.array` exposes read-only
numerical storage. Ordinary array arithmetic does not infer physical metadata.
See the [API reference](docs/api.md) for signatures, shapes, and conventions.

## Differentiation

Rust implements first-order pullbacks at numerical boundaries. A `diff` call
returns `(output, context)`; `context.pullback(cotangent)` consumes its residual
once. Complex derivatives use `dL = Re(vdot(cotangent, doutput))`.

Optional Advect, JAX, and PyTorch adapters compose these pullbacks with scalar
objectives. For example, using `treams-rs[advect]`:

```python
import advect.numpy as np
from advect import grad
from treams_rs import advect as ad


def loss(radii):
    positions = np.array([[0, 0, 0], [0, 0, 0.8]])
    matrix = ad.cluster(2, 1.3, radii, np.array([4.0, 3.0]), positions)
    return np.sum(np.real(matrix * np.conj(matrix)))


print(grad(loss)(np.array([0.2, 0.25])))
```

The [adapter guide](docs/adapters.md) defines supported transforms, CPU execution,
complex conventions, and residual ownership. Mode labels and topology are static;
forward mode and higher derivatives are outside the contract. Use the
[gradient-checking helpers](docs/testing.md) to validate a composed objective.

## Choose a computation path

| Need | Guide |
| --- | --- |
| Reuse a factorization for a few incident waves | [Requested illuminations and larger clusters](docs/large-problems.md) |
| Avoid dense storage for a sphere cluster | [Matrix-free forward and adjoint solves](docs/iterative.md) |
| Run on an NVIDIA GPU | [Optional CUDA execution](docs/gpu.md), including [repeated sampling and coefficient pullbacks](docs/gpu-sampling.md) |
| Exchange T matrices through HDF5 | `treams-rs[io]` and [I/O API](docs/api.md#treams_rsio) |
| Discover the installed API programmatically | [Agent guide](docs/agents.md) and [llms.txt](llms.txt) |

CPU parallelism uses Rayon and faer. Set `RAYON_NUM_THREADS` before importing to
choose the thread budget. CUDA is opt-in and absent from default CPU builds;
backend selection and transfers are explicit.

## Numerical qualification

Rust unit/property tests and Python reference, Hypothesis, and complete-workflow
tests check physical invariants and analytic derivatives. The
[benchmark guide](docs/benchmarks.md) records 527 runtime and 525 peak-RSS CPU
comparisons, including exact inputs, binary/source hashes, and two recorded-adjoint
memory exceptions. These finite measurements are not universal performance
or accuracy guarantees.

[Published-application reproductions](docs/paper-qualification.md) compare original
source data, complete spectra, and convergence checks. Known reference defects
are listed in [upstream findings](docs/upstream-findings.md). Platform-specific
qualification and numerical limits are collected in the
[capability reference](docs/status.md).

For implementation work, start with [Contributing](CONTRIBUTING.md),
[architecture](docs/architecture.md), and the [test strategy](docs/test-strategy.md).
The upstream source inventory is recorded in [rewrite scope](docs/upstream.md).
License and scientific attribution are preserved in
[third-party notices](THIRD_PARTY_NOTICES.md).

# treams-rs

T-matrix electromagnetic scattering with a Rust numerical core, a typed Python
API, and native analytic derivatives. The core runs independently of Python and
autodiff frameworks; there is no runtime fallback to treams, SciPy, or Cython.

The package covers spherical, cylindrical, and plane waves; layered and chiral
particles; finite clusters and periodic arrays; planar stacks; fields and power
observables. The Python API preserves physical meaning through scattering and field operations,
with explicit solve boundaries and named results. Numerical conventions follow
treams; upstream API compatibility is not a design requirement. Start with the
[physics-first guide](docs/user-guide.md).

`main` is the authoritative CPU core and Python implementation. Experimental
browser work lives on `experimental/browser`; CUDA work lives on
`experimental/gpu`. Each branch contains only its own experiment.

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

particles = [
    tr.sphere_tmatrix(k0=1.3, lmax=3, radius=r, material=4 + 0.1j) for r in (0.2, 0.25)
]
cluster = tr.Cluster(particles, positions=[[0, 0, 0], [0, 0, 0.8]])
incident = tr.plane_wave(direction=[0, 0, 1], polarization="positive_helicity", k0=1.3)
scattered = cluster.scatter(incident)
print(scattered.efield([[0.1, 0.2, 1.2]]))
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
from advect import grad
from treams_rs import advect as tr


def loss(radius):
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=3)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
    return sphere.cross_sections(incident).scattering


print(grad(loss)(0.2))
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
| Exchange T matrices through HDF5 | `treams-rs[io]` and [I/O API](docs/api.md#treams_rsio) |
| Discover the installed API programmatically | [Agent guide](docs/agents.md) and [llms.txt](llms.txt) |

CPU parallelism uses Rayon and faer. Set `RAYON_NUM_THREADS` before importing to
choose the thread budget.

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

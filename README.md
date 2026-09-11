# treams-rs

A personal Rust rewrite of treams, with a typed Python API and native analytic
pullbacks. The numerical core runs without Python or an autodiff framework.

The documented CPU rewrite is complete for the pinned treams 0.4.5 numerical
inventory, with explicit Python objects and first-order Advect integration. See
[implementation status](docs/status.md) for the API contract and qualified limits.
The complete performance grid passes 527 runtime and 525 peak-RSS comparisons;
[benchmarks](docs/benchmarks.md) records the inputs and two recorded-adjoint memory
exceptions. There is no runtime fallback to treams, SciPy or Cython.

```sh
uv sync --locked --group dev
just verify
just build-ext-release
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

Batched Cartesian electric fields for either multipole family use
`tr.diff.field(coefficients, points, basis, ks)`.
The output has shape `(number_of_points, 3)`. Its pullback returns cotangents for
amplitudes, sample coordinates, basis origins, and the two helicity wavenumbers.
For cylinders, `residual.pullback_axial` also returns axial-wavenumber cotangents;
pass `kzs` to the Advect field functions when optimizing them.
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
`plane_wave_angle(theta, phi, pol, ...)` accepts angles in radians. Wave-object
constructors take ordinary arrays. Use `ad.plane_expansion` for differentiable
illumination wavevectors and origins, and `ad.plane_field` for differentiable
plane-wave amplitudes, sample points and wavevectors.

Planar stacks use explicit up/down plane-wave channels:

```python
basis = tr.PlaneWaveBasisByComp.default([[0, 0]])
slab = tr.SMatrices.slab(0.4, basis, 1.3, [1, 2.5 + 0.1j, 1])
print(slab.tr(tr.plane_wave([0, 0, 1], 1, k0=1.3)))
```

Use `ad.fresnel`, `ad.propagation`, and `ad.smatrix_add` to differentiate complete
stacks. Their arrays have shape `(2, 2, number_of_modes, number_of_modes)`, indexing
outgoing direction, incoming direction, output mode, and input mode. Directions
are ordered up/down; low-level Fresnel polarization indices are ordered 0/1.

`ad.smatrix_tr` differentiates transmitted/reflected power, including port media
and incident amplitudes. It accepts independent illuminations as columns and
returns T/R rows. `ad.smatrix_cd` evaluates both helicities in one native batch.

```python
def transmittance(impedance):
    ks = np.array([[1.3, 1.3], [2.0, 2.0]])  # below, above; polarizations 0, 1
    zs = np.stack([1.0, impedance])
    q = np.array([[0.2, 0.3]])
    blocks = ad.interface(ks, zs, q[0])
    powers = ad.smatrix_tr(
        blocks,
        np.array([[1.0], [0.2j]]),
        ks[::-1],
        zs[::-1],
        q,
        modes=[(0, 0), (0, 1)],  # transverse-direction index, polarization
    )
    return np.real(powers[0, 0])


print(grad(transmittance)(np.array(0.7)))  # approximately 0.24412137
```

Interface media are ordered below/above; S-matrix power ports are ordered
above/below, hence the reversals. At normal incidence use `fixed_q=True` when
the transverse direction is held fixed.

Periodic spherical unit cells use the same S-matrix interface:

```python
cell = [[1.7, 0], [0, 1.8]]
basis = tr.PlaneWaveBasisByComp.diffr_orders([0, 0], cell, 4)
particle = tr.TMatrix.sphere(3, 2.1, 0.2, [3, 1])
array = tr.SMatrices.from_array(particle, basis, lattice=cell, kpar=[0, 0])
print(array.tr(tr.plane_wave([0, 0, 1], 1, k0=2.1)))
```

For inverse design, compose `ad.sphere`, `ad.lattice_expansion`, `ad.interaction`,
`ad.spherical_channels` and `ad.smatrix_from_array`. All numerical pullbacks run in
Rust. Channel derivatives cover origins, complex wavenumbers, transverse vectors
and cell area. At exact normal incidence use `fixed_q=True` to hold the undefined
polarization azimuth fixed; frequency, shape, material and cell gradients remain
available. See `tests/test_channels.py` for a complete reflected-power objective.

Local multipole rotations use `tr.rotate(phi, theta, psi, basis=basis)` or
`matrix.rotate(phi, theta, psi)`. `ad.rotation(angles, destination=basis)`
differentiates all three spherical Euler angles. Cylindrical bases permit only
rotation around their fixed axis (`theta=0`). Basis origins are not displaced.

`tr.efield`, `tr.hfield`, `tr.dfield` and `tr.bfield` return full Cartesian
field-operator arrays for spherical or cylindrical bases. Their shape is
`(..., 3, modes)`, so `tr.efield(points, basis=basis, k0=k0) @ coefficients`
gives electric samples. Use `ad.field_operator` for operator derivatives or the
lower-memory `ad.field`/`ad.hfield` when only weighted field samples are needed.

Single multipole sources and superpositions carry explicit metadata:

```python
source = tr.spherical_wave(1, 0, 1, k0=1.3)
particle = tr.TMatrix.sphere(3, 1.3, 0.2, [3, 1])
scattered = tr.MultipoleWave(
    particle @ source, basis=particle.basis, k0=1.3, modetype="singular"
)
print(scattered.efield([[0.5, 0.3, 0.1]]))
```

`tr.cylindrical_wave(kz, m, pol, ...)` follows the same pattern. Source E/H/D/B/G/F
methods use weighted native evaluation; `.array` exposes ordinary coefficients.

Install `treams-rs[io]` for HDF5 interchange:

```python
import h5py
from treams_rs import io

with h5py.File("particle.h5", "w") as handle:
    io.save_hdf5(handle, particle, lunit="nm")
restored = io.load_hdf5("particle.h5", lunit="nm")
```

The adapter accepts a single spherical T matrix or rectangular parameter sweeps.
It preserves embedding chirality and mode origins and writes matrices one at a
time. The core package and native solver do not require HDF5.

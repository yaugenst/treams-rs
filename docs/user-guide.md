# Physics-first Python API

Physical operations return physical objects. Use `.array` or `.coefficients`
when you want numerical arrays. Arrays are read-only in the NumPy API; call
`.copy()` before editing. No automatic ndarray annotation propagation is involved.
All lengths and inverse `k0` use the same units. Polarization defaults to helicity
and never depends on mutable global configuration.

## Particles, illumination, and fields

```python exec
import numpy as np
import treams_rs as tr

sphere = tr.sphere_tmatrix(
    k0=1.3,
    lmax=3,
    radius=0.2,
    material=tr.Material(epsilon=3.0),
)
incident = tr.plane_wave(
    direction=[0, 0, 1],
    polarization="positive_helicity",
    k0=sphere.k0,
)
scattered = sphere.scatter(incident)
points = [[0.3, 0.2, 0.7], [-0.4, 0.1, 0.8]]
total_e = incident.efield(points) + scattered.efield(points)
assert total_e.shape == (2, 3)
power = sphere.cross_sections(incident)
np.testing.assert_allclose(power.scattering, power.extinction, rtol=1e-11)
assert abs(power.absorption) < 1e-12
```

`material` describes the particle; `medium` describes its exterior.
`multilayer_sphere_tmatrix` takes `radii` and one material per layer, ordered
inside-out, with the exterior passed separately as `medium`.
The corresponding cylinder constructors take axial `kz` and azimuthal cutoff
`mmax`; use `cross_widths` for a result in length units, rather than sphere
`cross_sections` in area units.

Fields belong to waves. A T-matrix response does not expose bound field
operators: use `response.scatter(incident).efield(points)` for scattered fields,
and add `incident.efield(points)` for total fields. Applying a response-level
field operator to already scattered coefficients would apply the response twice.
Use `operators.efield(...)` when an explicit numerical field matrix is needed.

`Wave` retains coefficients, basis, frequency, medium, polarization convention,
and regular/outgoing or plane direction. It supports E/H/D/B/G/F evaluation,
`in_basis`, and `with_polarization`. Coefficient batches have shape
`(modes, illuminations)`. `SphericalBasis` and `CylindricalBasis` describe multipole
channels. `PlaneWaveBasis` stores directions; `PlaneWavePorts` stores transverse
wavevectors and the port normal. Their constructors, mode selections, diffraction
orders, and geometry operations appear in the generated API reference.

## Finite multiple scattering

An unsolved collection is a `Cluster`, not a T-matrix that looks solved.

```python exec
import numpy as np
import treams_rs as tr

particles = [
    tr.sphere_tmatrix(k0=1.3, lmax=2, radius=r, material=3) for r in (0.15, 0.2)
]
cluster = tr.Cluster(particles, positions=[[0, 0, 0], [0.8, 0, 0]])
incident = tr.plane_wave(direction=[0, 0, 1], polarization=1, k0=1.3)
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

`scatter` solves only requested incident columns; `solve` builds the complete
response. A retained `factor` reuses the LU for new illuminations.
`select(basis)`, `in_basis(basis)`, `rotate(...)`, and `with_polarization(...)`
retain physical objects. Numerical slices use `.array[...]`.
The restricted matrix-free `iterative.SphereCluster.scatter` returns a named
`ScatteringSolution(wave, convergence)`. Its numerical `solve` and analytic
`solve_with_pullback` remain available for incident-column arrays; convergence
is always checked. See [iterative solver contracts](iterative.md).

## Periodic arrays and planar layers

Periodic response is distinct from isolated-particle response. Solve once,
then convert that response into the output channels you need.

```python exec
import numpy as np
import treams_rs as tr

cell = tr.Lattice.square(1.7)
sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
response = tr.solve_periodic(sphere, lattice=cell, kpar=[0, 0])
ports = tr.PlaneWavePorts.diffr_orders([0, 0], cell, 0.1)
array = response.to_smatrix(ports)
layer = tr.slab(basis=ports, k0=1.3, thickness=0.1, material=2)
gap = tr.propagation(distance=0.3, basis=ports, k0=1.3)
stack = tr.stack([layer, gap, array])
incident = tr.plane_wave(direction=[0, 0, 1], polarization=1, k0=1.3)
power = stack.power(incident, side="negative")
np.testing.assert_allclose(power.transmission + power.reflection, 1, atol=1e-10)
outputs = stack.scatter(negative=incident)
assert outputs.positive.coefficients.shape == (len(ports),)
```

`SMatrix` is the complete two-sided network; `ScatteringBlock` is a single port
block. Layers are ordered from the negative toward the positive side of their
basis normal. Exterior media are named `negative_medium` and `positive_medium`.
`power` returns named transmission/reflection/absorption; `scatter` returns named
negative/positive outgoing waves. `bands(period=...)` returns named wavenumbers
and eigenvectors. Internal-layer illumination and explicit numerical transfer
matrices remain available on the S-matrix object.

A `PeriodicWave` represents every cell, not just its local coefficients. Choose
an exterior plane/cylindrical expansion or a local regular multipole expansion
with `in_basis(..., kind=...)` before evaluating fields. Evaluating the reference
cell alone would produce the wrong periodic field. Do not apply isolated-particle
cross-section formulas to periodic responses.

## Numerical tools and interchange

`operators` owns numerical matrix builders and operator objects: field matrices,
rotations, translations, finite/periodic expansions and polarization changes.
These operations return arrays. `special`, `coeffs`, `sw`, `cw`, `pw`, `lattice`,
`misc`, and `ebcm` retain the complete mathematical toolbox, including corrected
EBCM quadrature and its explicit reference-comparison mode. Mathematical function
names and normalization conventions are preserved.

Optional `io.save_hdf5` / `io.load_hdf5` exchange spherical T-matrices and sweeps,
including units, origins, material and reproducibility metadata. A loaded response
supports the same `scatter`, field and observable methods. `io.mesh_spheres`
keeps Gmsh lifecycle ownership with the caller.

## Differentiation and discovery

Choose `treams_rs.advect`, `.jax`, or `.torch` explicitly for differentiable
physical objects. The same sphere, cluster, wave and planar vocabulary composes
with framework scalar objectives; continuous values remain framework arrays.
Read [adapter contracts](adapters.md) for static choices and supported derivatives.
The numerical `diff` API remains the explicit forward/pullback boundary, and
`testing` provides independent objective and pullback checks.

`support_catalog()` and `python -m treams_rs --format markdown` describe the
installed source without importing optional frameworks. [API reference](api.md)
is generated from those same source definitions. The complete surface review is
recorded in the [physics map](api-physics-map.md) and
[autodiff map](api-autodiff-map.md); those explain design decisions rather than
establish a second capability registry.

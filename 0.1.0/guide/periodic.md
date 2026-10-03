# Periodic arrays

`solve_periodic` couples a unit cell to all its copies in a lattice at a
fixed Bloch wavevector `kpar` (the wavevector component in the lattice plane;
`[0, 0]` at normal incidence). The result is a `PeriodicResponse`. Convert it
to the output you need without solving again:

| Method | Returns |
| --- | --- |
| `to_smatrix(ports)` | an `SMatrix` over plane-wave ports, to stack with planar layers |
| `to_cylindrical(basis)` | a `CylindricalTMatrix`, for spheres repeated along z |
| `scatter(incident)` | a `PeriodicWave`: the scattered wave of every cell |

The unit cell is a `TMatrix`, a `CylindricalTMatrix` or an unsolved `Cluster`.
`Lattice.square`, `Lattice.hexagonal` and `Lattice(...)` set the lattice; a
two-dimensional lattice lies in the xy plane. The lattice sums use the Ewald
method: each sum splits into a real-space part and a reciprocal-space part that
both converge quickly. The split parameter `eta` defaults to 0, which picks the
split automatically.

## An array on a slab

```python
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
incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.3)
power = stack.power(incident, side="negative")
np.testing.assert_allclose(power.transmission + power.reflection, 1, atol=1e-10)
outputs = stack.scatter(negative=incident)
assert outputs.positive.coefficients.shape == (len(ports),)
```

The S-matrix of the array places its ports at the plane of the particle
centres, so a `propagation` before or after it measures the distance from that
plane.

## Diffraction orders

`PlaneWavePorts.diffr_orders(kpar, lattice, bmax)` lists the diffraction orders
`kpar + G` whose reciprocal lattice vector `G` is at most `bmax` long. At normal
incidence, `bmax = k0` holds every propagating order; at oblique incidence, use
`bmax = k0 + |kpar|`. Power is conserved only when the ports include all
propagating orders:

```python
import numpy as np
import treams_rs as tr

lattice, k0 = tr.Lattice.square(1.7), 5.0  # the period exceeds the wavelength
sphere = tr.sphere_tmatrix(k0=k0, lmax=4, radius=0.3, material=4)
response = tr.solve_periodic(sphere, lattice=lattice, kpar=[0, 0])
incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=k0)
zeroth = tr.PlaneWavePorts.diffr_orders([0, 0], lattice, 0.1)
assert len(zeroth) == 2  # one order, two pol values
open_orders = tr.PlaneWavePorts.diffr_orders([0, 0], lattice, k0)
assert len(open_orders) == 10  # five orders
power = response.to_smatrix(open_orders).power(incident)
np.testing.assert_allclose(power.transmission + power.reflection, 1, rtol=1e-12)
partial = response.to_smatrix(zeroth).power(incident)
assert partial.transmission + partial.reflection < 0.6
kpar = [1.5, 0]  # oblique incidence
oblique = tr.solve_periodic(sphere, lattice=lattice, kpar=kpar)
direction = [1.5, 0, np.sqrt(k0**2 - 1.5**2)]
tilted = tr.plane_wave(direction=direction, pol="positive_helicity", k0=k0)
ports = tr.PlaneWavePorts.diffr_orders(kpar, lattice, k0 + np.linalg.norm(kpar))
power = oblique.to_smatrix(ports).power(tilted)
np.testing.assert_allclose(power.transmission + power.reflection, 1, rtol=1e-12)
```

## Fields of the whole array

A `PeriodicWave` describes the scattered waves of every cell. The reference
cell's coefficients alone give the wrong field, so expand the wave with
`in_basis` first:

- `in_basis(ports, kind="up")` or `kind="down"`: plane waves above or below
  the array.
- `in_basis(SphericalBasis.default(lmax, positions=[point]), kind="regular")`:
  regular multipoles about a point between the particles. Evaluated at that
  point, they give the scattered field there.

```python
import numpy as np
import treams_rs as tr

lattice, k0 = tr.Lattice.square(1.7), 1.3
sphere = tr.sphere_tmatrix(k0=k0, lmax=2, radius=0.3, material=4)
response = tr.solve_periodic(sphere, lattice=lattice, kpar=[0, 0])
incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=k0)
wave = response.scatter(incident)
ports = tr.PlaneWavePorts.diffr_orders([0, 0], lattice, k0)
down = wave.in_basis(ports, kind="down")
reflected = response.to_smatrix(ports).scatter(negative=incident).negative
np.testing.assert_allclose(down.coefficients, reflected.coefficients, atol=1e-15)
point = [0.4, 0.3, 6.0]  # far above the array, out of reach of the evanescent orders
about_point = tr.SphericalBasis.default(1, positions=[point])
local = wave.in_basis(about_point, kind="regular")
up = wave.in_basis(ports, kind="up")
np.testing.assert_allclose(local.efield([point]), up.efield([point]), atol=1e-9)
```

Close to the array, the plane-wave form needs the evanescent orders too
(a larger `bmax`); the local expansion needs none. Cross sections of isolated
particles do not apply to a periodic response; use `power` of its S-matrix.

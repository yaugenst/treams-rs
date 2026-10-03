# Quickstart

Lengths and `1 / k0` share one unit; `k0` is the vacuum wavenumber. Numbers
passed as `material` are relative permittivities.

## A sphere

`sphere_tmatrix` builds the T-matrix of a sphere. `scatter` turns an incident
wave into the scattered wave, and `cross_sections` gives areas:

```python
import numpy as np
import treams_rs as tr

sphere = tr.sphere_tmatrix(k0=2.0, lmax=3, radius=0.5, material=4)
incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=2.0)
scattered = sphere.scatter(incident)
points = [[0, 0, 2.0], [1.0, 0, 1.0]]
total = incident.efield(points) + scattered.efield(points)
assert total.shape == (2, 3)  # (points, Cartesian components)
power = sphere.cross_sections(incident)
# The sphere is lossless: it scatters all the power it removes.
np.testing.assert_allclose(power.extinction, power.scattering, rtol=1e-12)
```

`pol="positive_helicity"` sets the polarization state of the plane wave. The
polarization convention of the T-matrix, `polarization="helicity"` or
`"parity"`, is a separate argument.

## A cluster

A `Cluster` holds particles and their positions. `scatter` solves only for the
incident wave you pass; `solve` returns the T-matrix of the whole cluster;
`factor` keeps an LU factorization for more incident waves:

```python
import numpy as np
import treams_rs as tr

spheres = [tr.sphere_tmatrix(k0=1.3, lmax=2, radius=r, material=3) for r in (0.2, 0.25)]
cluster = tr.Cluster(spheres, positions=[[0, 0, 0], [0, 0, 0.8]])
incident = tr.plane_wave(direction=[1, 0, 0], pol="positive_helicity", k0=1.3)
direct = cluster.scatter(incident)
tmatrix = cluster.solve()
factor = cluster.factor()
for wave in (tmatrix.scatter(incident), factor.scatter(incident)):
    np.testing.assert_allclose(wave.coefficients, direct.coefficients, rtol=1e-10)
```

## A periodic array

`solve_periodic` couples a particle to all its copies in a lattice.
`to_smatrix` expresses the response in the diffraction orders of the array:

```python
import numpy as np
import treams_rs as tr

lattice = tr.Lattice.square(1.7)
sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=0.3, material=4)
response = tr.solve_periodic(sphere, lattice=lattice, kpar=[0, 0])
orders = tr.PlaneWavePorts.diffr_orders([0, 0], lattice, 1.3)
array = response.to_smatrix(orders)
incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.3)
power = array.power(incident)
# Lossless spheres: transmitted and reflected power add up to the incident power.
np.testing.assert_allclose(power.transmission + power.reflection, 1, rtol=1e-12)
```

`kpar` is the Bloch wavevector in the lattice plane: `[0, 0]` for normal
incidence.

## A slab stack

`slab` and `propagation` return S-matrices; `stack` combines them from the
negative side (z < 0) to the positive side:

```python
import numpy as np
import treams_rs as tr

ports = tr.PlaneWavePorts.default([0, 0])  # normal incidence, pol 1 and 0
layer = tr.slab(basis=ports, k0=1.0, thickness=0.5, material=2.25)
gap = tr.propagation(distance=0.7, basis=ports, k0=1.0)
incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.0)
# Fabry-Perot reflectance of a slab with refractive index 1.5.
r, phase = (1 - 1.5) / (1 + 1.5), np.exp(2j * 1.5 * 1.0 * 0.5)
expected = abs(r * (1 - phase) / (1 - r**2 * phase)) ** 2
np.testing.assert_allclose(layer.power(incident).reflection, expected, rtol=1e-12)
power = tr.stack([layer, gap, layer]).power(incident)
np.testing.assert_allclose(power.transmission + power.reflection, 1, rtol=1e-12)
```

## Next steps

- [Coming from treams](../coming-from-treams/index.md) translates treams
  workflows.
- The [user guide](../guide/index.md) covers each topic in depth.
- [Differentiation](../differentiation/index.md) shows gradients through
  Advect, JAX, PyTorch and HIPS Autograd.

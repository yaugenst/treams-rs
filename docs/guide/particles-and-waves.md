---
description: Sphere and cylinder T-matrices, materials and media, plane and multipole waves, fields and cross sections.
---

# Particles and waves

## A sphere and its fields

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
    pol="positive_helicity",
    k0=sphere.k0,
)
scattered = sphere.scatter(incident)
assert scattered.kind == "singular"
points = [[0.3, 0.2, 0.7], [-0.4, 0.1, 0.8]]
total_e = incident.efield(points) + scattered.efield(points)
assert total_e.shape == (2, 3)
power = sphere.cross_sections(incident)
np.testing.assert_allclose(power.scattering, power.extinction, rtol=1e-11)
assert abs(power.absorption) < 1e-12
assert sphere.array.shape == (30, 30)  # 2 lmax (lmax + 2) modes
average = sphere.average_cross_sections
np.testing.assert_allclose(average.extinction, average.scattering, rtol=1e-11)
```

Fields belong to waves. `sphere.scatter(incident)` returns the scattered wave,
and `scattered.efield(points)` evaluates it; add `incident.efield(points)` for
the total field. `hfield`, `dfield`, `bfield`, `gfield` and `ffield` give the
other fields. A T-matrix has no field methods: applying one to scattered
coefficients would apply the response twice. `operators.efield` builds an
explicit field matrix when you need one.

`cross_sections(incident)` returns areas in the length unit squared;
`average_cross_sections` gives the averages over all incident directions and
both polarizations.

## Materials and media

`material` is the particle or layer; `medium` is the space around it (vacuum
by default). A number is a relative permittivity; `Material(epsilon, mu, kappa)`
adds the relative permeability and the chirality parameter.
`multilayer_sphere_tmatrix` takes increasing `radii` and one material per
layer, from the inside out:

```python exec
import numpy as np
import treams_rs as tr

core_shell = tr.multilayer_sphere_tmatrix(
    k0=1.3, lmax=2, radii=[0.1, 0.2], materials=[3, 3], medium=1.69
)
solid = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3, medium=1.69)
# Two layers of the same material make one solid sphere.
np.testing.assert_allclose(core_shell.array, solid.array, atol=1e-15)
# Only a chiral material (kappa != 0) treats the two helicities differently.
chiral = tr.sphere_tmatrix(
    k0=1.3, lmax=2, radius=0.2, material=tr.Material(epsilon=3, kappa=0.1)
)
assert abs(solid.circular_dichroism) < 1e-12 < abs(chiral.circular_dichroism)
```

## Cylinders

Cylinders are infinite along z. `cylinder_tmatrix` takes the axial
wavenumbers `kz` and the largest azimuthal order `mmax` in place of `lmax`.
Their cross sections are widths per unit length, so use `cross_widths`:

```python exec
import numpy as np
import treams_rs as tr

cylinder = tr.cylinder_tmatrix(k0=1.0, kz=[0.0], mmax=3, radius=0.4, material=4)
incident = tr.plane_wave(direction=[1, 0, 0], pol="positive_helicity", k0=1.0)
widths = cylinder.cross_widths(incident)
np.testing.assert_allclose(widths.extinction, widths.scattering, rtol=1e-12)
```

## Waves and bases

A `Wave` stores coefficients with their basis, `k0`, `medium`, polarization
convention and `kind`:

| Basis | Modes | `kind` |
| --- | --- | --- |
| `SphericalBasis` | `l`, `m`, `pol` at one or more positions | `"regular"` (finite at the origin) or `"singular"` (radiating, with the Hankel function of the first kind) |
| `CylindricalBasis` | `kz`, `m`, `pol` at one or more axes | `"regular"` or `"singular"` |
| `PlaneWaveBasis` | unit directions and `pol` | `"up"` or `"down"` |
| `PlaneWavePorts` | transverse wavevectors and `pol` | `"up"` or `"down"` |

`plane_wave` builds a plane wave, `spherical_wave` and `cylindrical_wave` a
single multipole. `pol` sets the polarization state: `"positive_helicity"`,
`"negative_helicity"`, an index 0 or 1, two amplitudes `[a0, a1]` or the three
Cartesian components of the electric field.
`in_basis` expands a wave in another basis; `with_polarization` changes the
polarization convention:

```python exec
import numpy as np
import treams_rs as tr

plane = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.3)
local = plane.in_basis(tr.SphericalBasis.default(8))
assert local.kind == "regular"
point = [[0.1, 0.2, 0.3]]
# Near the origin, the multipole expansion reproduces the plane wave.
np.testing.assert_allclose(local.efield(point), plane.efield(point), atol=1e-9)
```

Coefficient batches have the shape `(modes, illuminations)`. The
[Python reference](../reference/python/treams_rs.md) lists the basis
constructors, mode selections and geometry operations.

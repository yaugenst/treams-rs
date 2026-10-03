---
description: The numerical namespaces special, sw, cw, pw, lattice, coeffs, misc, ebcm and operators, with one checked example each.
---

# Numerical namespaces

These modules keep the function and argument names of treams, so a call such
as `treams.sw.translate(...)` runs as
`treams_rs.sw.translate(...)` with the same arguments. They take and return
NumPy arrays, and most broadcast like NumPy ufuncs. The
[differences page](../coming-from-treams/differences.md) lists every case where
a result differs from treams; the
[Python reference](../reference/python/index.md) lists every signature.

The functions take the polarization convention as `poltype="helicity"` or
`"parity"` and the radial kind as `singular=True` (radiating waves) or
`singular=False` (regular waves), as in treams.

## special

Mirrors `treams.special`: Bessel and Hankel functions, Legendre functions,
Wigner symbols, vector spherical, cylindrical and plane waves, translation
coefficients and coordinate transforms.

```python exec
import numpy as np
from treams_rs import special

x = np.array([0.5, 2.0, 7.5])
np.testing.assert_allclose(
    special.spherical_jn(1, x), np.sin(x) / x**2 - np.cos(x) / x, rtol=1e-13
)
np.testing.assert_allclose(
    special.spherical_hankel1(0, x), -1j * np.exp(1j * x) / x, rtol=1e-13
)
```

Bessel, Hankel and wave functions return complex128, also for real inputs. An
evaluation that treams returns as NaN raises `ValueError`; see
[deliberate differences](../coming-from-treams/differences.md#deliberate-differences).

## sw

Mirrors `treams.sw`: coefficients of spherical waves. In
`sw.translate(lambda_, mu, pol, l, m, qol, kr, theta, phi)`, the first three
labels belong to the destination mode, the next three to the source mode, and
`kr`, `theta`, `phi` give the translation vector in spherical coordinates, its
length multiplied by the wavenumber.

```python exec
import numpy as np
from treams_rs import sw

# A rotation by phi about z multiplies a mode of order m by exp(-i m phi).
np.testing.assert_allclose(sw.rotate(1, 1, 1, 1, 1, 1, 0.3), np.exp(-0.3j))
# A regular translation along z (theta = 0) keeps the order m.
assert sw.translate(1, 1, 1, 1, 0, 1, 2.0, 0.0, 0.0, singular=False) == 0
np.testing.assert_allclose(
    sw.translate(1, 0, 1, 1, 0, 1, 0.0, 0.0, 0.0, singular=False), 1
)
```

## cw

Mirrors `treams.cw`: coefficients of cylindrical waves, labelled by the axial
wavenumber `kz`, the order `m` and `pol`.

```python exec
import numpy as np
from treams_rs import cw

kz, z = 0.5, 2.0
# A shift along the axis multiplies a cylindrical wave by exp(i kz z).
shifted = cw.translate(kz, 1, 1, kz, 1, 1, 0.0, 0.0, z, singular=False)
np.testing.assert_allclose(shifted, np.exp(1j * kz * z))
```

## pw

Mirrors `treams.pw`: plane-wave phases, expansions into spherical and
cylindrical waves and coordinate permutations.

```python exec
import numpy as np
from treams_rs import pw

k, r = np.array([1.0, 0.0, 2.0]), np.array([0.3, 0.0, 0.5])
np.testing.assert_allclose(pw.translate(*k, *r), np.exp(1j * k @ r))
```

## lattice

Mirrors `treams.lattice`: lattice sums of spherical and cylindrical waves in one,
two and three dimensions, and lattice geometry. The `lsum*` functions evaluate
a sum with the Ewald method, which splits it into a real-space part
(`realsum*`) and a reciprocal-space part (`recsum*`); `eta` sets the split and
`eta=0` picks it automatically. The arguments `kpar` (Bloch wavevector), `a`
(lattice vectors as rows) and `r` (shift) follow treams.

```python exec
import numpy as np
from treams_rs import lattice

a = 1.7 * np.eye(2)
args = (2, 0, 1.3, [0.1, 0.0], a, [0.0, 0.0])  # l, m, k, kpar, a, r
split = lattice.realsumsw2d(*args, 0.6) + lattice.recsumsw2d(*args, 0.6)
np.testing.assert_allclose(lattice.lsumsw2d(*args, eta=0.6), split, rtol=1e-14)
np.testing.assert_allclose(lattice.lsumsw2d(*args), split, rtol=1e-12)
np.testing.assert_allclose(lattice.reciprocal(a), 2 * np.pi / 1.7 * np.eye(2))
```

## coeffs

Mirrors `treams.coeffs`: Mie coefficients of layered spheres and cylinders and
Fresnel coefficients. `mie(l, x, epsilon, mu, kappa)` takes the size parameters
`x = k0 r` and the materials from the inside out, followed by the surrounding
medium. It takes one degree `l` per call, where treams broadcasts over degrees.

```python exec
import numpy as np
from treams_rs import coeffs

t = coeffs.mie(1, [1.3 * 0.5], [4, 1], [1, 1], [0, 0])  # 2x2 helicity block
s = np.eye(2) + 2 * t
# A lossless sphere conserves energy: S = 1 + 2 T is unitary.
np.testing.assert_allclose(s.conj().T @ s, np.eye(2), atol=1e-14)
```

## misc

Mirrors `treams.misc`: refractive indices, normal wavenumbers, mode selection
and reduction to the first Brillouin zone.

```python exec
import numpy as np
from treams_rs import misc

# The two helicities see n - kappa and n + kappa.
np.testing.assert_allclose(misc.refractive_index(4, 1, 0.1), [1.9, 2.1])
np.testing.assert_allclose(misc.wave_vec_z(3, 0, 5), 4)
np.testing.assert_allclose(misc.wave_vec_z(0, 2, 1), 1j * np.sqrt(3))  # evanescent
```

## ebcm

Mirrors `treams.ebcm`: surface integrals of the extended boundary condition
method for particles with rotational symmetry about z. `qmat(r, dr, ks, zs,
out, in_)` takes the radius `r(theta)`, its derivative, the wavenumbers and
impedances (inside, then outside) and the destination and source modes.

```python exec
import numpy as np
import treams_rs as tr

basis = tr.SphericalBasis.default(2)
media = [tr.Material(4), tr.Material(1)]
ks = [m.ks(1.3) for m in media]
zs = [m.impedance for m in media]
q = tr.ebcm.qmat(lambda _: 0.3, lambda _: 0, ks, zs, basis)
q_regular = tr.ebcm.qmat(lambda _: 0.3, lambda _: 0, ks, zs, basis, singular=False)
tmatrix = -np.linalg.solve(q, q_regular)
# For a sphere, EBCM reproduces the Mie T-matrix.
sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=0.3, material=4)
np.testing.assert_allclose(tmatrix, sphere.array, atol=1e-12)
```

`qmat` includes the radial factor of the surface element;
`radial_area_factor=False` reproduces treams, which omits it. `order` sets the
number of Gauss-Legendre nodes, where treams integrates adaptively. See the
[differences page](../coming-from-treams/differences.md#behavioral-defects).

## operators and PhysicsArray

Mirrors the operators of treams (`treams.efield`, `treams.Rotate`, ...) and
`treams.PhysicsArray`. `efield` and the other field operators, `rotate`,
`translate`, `expand`, `expandlattice`, `changepoltype` and `permute` return
explicit matrices. `PhysicsArray` holds an array with its basis, `k0`, material
and conventions, and provides these operations as methods:

```python exec
import numpy as np
import treams_rs as tr
from treams_rs import operators

basis = tr.SphericalBasis.default(1)
array = operators.PhysicsArray(np.eye(6), basis=basis, k0=1.3)
rotation = array.rotate.eval(0.3, 0, 0)  # Euler angles phi, theta, psi
np.testing.assert_allclose(np.diag(rotation), np.exp(-0.3j * basis.m))
larger = tr.SphericalBasis.default(2)
expansion = array.expand.eval(basis=larger)
np.testing.assert_allclose(expansion, operators.expand((larger, basis), k0=1.3))
assert isinstance(array * 2, np.ndarray)  # arithmetic returns plain arrays
```

Arithmetic with a `PhysicsArray` returns plain arrays without metadata.
`operators.expand` and `operators.expandlattice` between spherical and cylindrical waves add the terms
of every pair of positions, where treams pairs only equal particle indices;
see [spheres and cylinders at several positions](../coming-from-treams/differences.md#spheres-and-cylinders-at-several-positions).
Use the methods of `TMatrix`, `Wave` and `SMatrix` to retain physical metadata.

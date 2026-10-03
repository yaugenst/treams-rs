---
description: Units, time convention, polarization, mode ordering, planar sides, field normalization, branches and label bounds, as in treams.
---

# Conventions

treams-rs follows the numerical conventions of treams. Arrays from one package
can be compared entry by entry with arrays from the other.

## Units and frequency

- Lengths and `1 / k0` share one unit. With lengths in nm, `k0` is in 1/nm.
- `k0 = ω / c` is the vacuum angular wavenumber. In a medium the wavenumber is
  `k0 n`, and in a chiral medium `k0 (n - κ)` for pol 0 and `k0 (n + κ)` for
  pol 1.
- A `Material` holds the relative permittivity `epsilon`, the relative
  permeability `mu` and the chirality parameter `kappa`. `Material(4)` has
  `epsilon = 4`, `mu = 1` and `kappa = 0`.
- Cross sections are areas and cross widths are lengths, in the length unit.

## Time convention

Fields vary in time as `exp(-iωt)`, as in the
[treams theory page](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/theory.rst).
Outgoing waves therefore use the Hankel function of the first kind, and lossy
media have `Im(epsilon) > 0`.

## Polarization

The polarization convention is `"helicity"` (the default) or `"parity"`, which
treams calls `poltype`. The index `pol` of a mode is 0 or 1:

| Convention | pol 0 | pol 1 |
| --- | --- | --- |
| `"helicity"` | negative helicity | positive helicity |
| `"parity"` | TE, the M multipoles | TM, the N multipoles |

A helicity mode combines the parity modes as `(N + (2 pol - 1) M) / sqrt(2)`.
For a plane wave, `pol=[a0, a1]` gives the amplitudes of pol 0 and pol 1, as
in treams.

## Mode ordering

The default bases order their modes as in treams:

| Basis | Order, slowest index first |
| --- | --- |
| `SphericalBasis.default(lmax, nmax)` | position index, degree `l`, order `m`, then pol 1 before pol 0 |
| `CylindricalBasis.default(kzs, mmax, nmax)` | position index, `kz` in input order, order `m`, then pol 1 before pol 0 |
| `PlaneWavePorts.default(kpars)` | transverse wavevector in input order, then pol 1 before pol 0 |

```python exec
import numpy as np
import treams_rs as tr

basis = tr.SphericalBasis.default(1)
labels = np.column_stack([basis.l, basis.m, basis.pol])
np.testing.assert_array_equal(
    labels, [[1, -1, 1], [1, -1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]]
)
```

`plane_wave(...)` builds a two-mode `PlaneWaveBasis` with pol 0 first; treams
orders it pol 1 first. The field is the same, and `pol=[a0, a1]` means the same
in both.

The incident coefficients of a sphere cluster list the modes of each particle in
turn, in the order above. A vector describes one illumination; a matrix of shape
`(modes, P)` describes P illuminations. See
[large clusters](../guide/large-clusters.md).

## Degree, order and axial wavenumbers

- The degree is `l` and the order is `m`; `lmax` and `mmax` bound them. The
  `diff` functions and the framework adapters spell them `degree=` and
  `order=`.
- `kz` is the axial wavenumber of one cylindrical mode: `basis.kz` has one value
  per mode, as in treams.
- `kzs` are the distinct axial wavenumbers of a cylindrical T-matrix, one per
  block, as in treams' `TMatrixC.cylinder(kzs, ...)`. `diff.cylinder(kzs, ...)`
  requires distinct values and returns the gradient of `kzs` in input order.
  `context.pullback_axial` of the expansion, lattice-expansion and field
  records returns one gradient per distinct `kz`, in increasing order.

## Planar sides and S-matrix blocks

| Term | Meaning |
| --- | --- |
| negative side | below, at smaller coordinates along the basis normal (z by default) |
| positive side | above, at larger coordinates along the basis normal |
| `negative_medium`, `positive_medium` | the media on the two sides of an `interface`, `slab` or `SMatrix` |
| `kind="up"`, index 0 | propagation towards the positive side |
| `kind="down"`, index 1 | propagation towards the negative side |
| `stack([a, b])`, `a.cascade(b)` | `a` is the lower layer, `b` sits on its positive side |
| `SMatrix.material` | `(positive_medium, negative_medium)`, the treams order |

An `SMatrix` array has shape `(2, 2, n, n)`. Entry `[i, j]` maps waves that
arrive travelling in direction `j` to waves that leave in direction `i`, with
0 for up and 1 for down. So `[0, 0]` transmits upwards, `[1, 0]` reflects
light from below, `[0, 1]` reflects light from above and `[1, 1]` transmits
downwards, as in treams. Port 0, the up direction, leaves through the positive
side.

```python exec
import numpy as np
import treams_rs as tr

ports = tr.PlaneWavePorts.default([0, 0])
glass = tr.interface(basis=ports, k0=1.0, negative_medium=1, positive_medium=2.25)
upward = glass.block(outgoing="positive", incoming="negative")
np.testing.assert_array_equal(upward.array, glass.array[0, 0])
```

## Field normalization

All fields come in the unit of the electric field:

| Method | Quantity |
| --- | --- |
| `efield` | E |
| `hfield` | Z0 H, with the vacuum impedance Z0 |
| `dfield` | D / ε0 |
| `bfield` | c B |
| `gfield`, `ffield` | Riemann–Silberstein fields G and F, combinations of E and i Z H |

`gfield(pol, r)` and `ffield(pol, r)` keep the scale factors of treams, which
differ between wave families and conventions. For a single positive-helicity
wave, `gfield(1, r)` returns:

| Wave | `"helicity"` | `"parity"` |
| --- | --- | --- |
| plane or cylindrical | E | 2 E |
| spherical | sqrt(2) E | 2 sqrt(2) E |

`pol` of `gfield` and `ffield` is 1, 0 or -1; 0 means -1, as in treams.
`ffield` also weights each helicity mode by `n_pol / n`, the ratio of its
helicity refractive index to the mean index.

## Complex branches

- The normal wavenumber `kz = sqrt(k^2 - kx^2 - ky^2)` takes the root with
  `Im(kz) >= 0`, and `Re(kz) >= 0` when it is real. `misc.wave_vec_z` returns
  it. Evanescent waves then decay away from their source.
- The radial wavenumber of cylindrical waves, `krho = sqrt(k^2 - kz^2)`, takes
  the same branch.

## Label bounds

treams-rs limits mode labels with three constants of the Rust core
(`treams-core`); treams has no limits. A label beyond its bound raises
`ValueError`.

| Rust constant | Value | Bounds |
| --- | --- | --- |
| `MAX_DEGREE` | 128 | degree `l` of multipoles, `special.wignerd` and angular functions; the size of cylindrical orders `m` |
| `MAX_ORDER` | 256 | radial orders and cylindrical order differences inside translations |
| `MAX_LABEL` | 260 | the labels of `special.wigner3j`, orders of Kambe integrals and Wigner labels inside translations |

Bessel and Hankel functions in `special` take any finite order.

## Reference version

treams-rs follows treams 0.4.7 at commit
[`1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39`](https://github.com/tfp-photonics/treams/tree/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39)
(2026-08-24), one documentation commit after the v0.4.7 tag. The tests compare
with treams 0.4.7, and the [differences page](differences.md) describes it.

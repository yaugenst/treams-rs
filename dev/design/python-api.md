# Python API design

treams-rs uses physics objects where treams uses annotated arrays. The
[user guide](../guide/index.md) shows the objects at work; the
[name map](../coming-from-treams/names.md) lists every treams name.

## Explicit objects

A physics object holds its numbers and its metadata side by side. A `TMatrix` has
an `.array`, a `basis`, `k0`, a `medium` and a `polarization` convention. Each
operation reads the metadata it needs and checks that two objects agree, so a
mismatch raises an error instead of producing numbers in the wrong basis.

```python
import treams_rs as tr

sphere = tr.sphere_tmatrix(k0=2.0, lmax=2, radius=0.3, material=4.0 + 0.1j)
incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=2.0)
scattered = sphere.scatter(incident)
cross = sphere.cross_sections(incident)
assert cross.extinction > cross.scattering > 0
```

- **Unsolved and solved objects differ.** `Cluster(particles, positions=...)`
  only collects particles. `Cluster.solve()` returns the coupled `TMatrix`,
  `Cluster.factor()` prepares repeated illumination solves and
  `solve_periodic(...)` returns a `PeriodicResponse`. NumPy clusters reuse a
  `ScatteringFactor`; framework clusters prepare coupling and record a native
  factorization for each solve. treams instead calls `TMatrix.cluster`, which
  returns a matrix that is not yet a response.
- **`scatter()` returns waves.** `TMatrix.scatter`, `Cluster.scatter` and
  `PeriodicResponse.scatter` return a `Wave` or `PeriodicWave`; fields come from
  the wave, as in `response.scatter(incident).efield(points)`. A field operator
  bound to a T-matrix would apply the response twice when given scattered
  coefficients, so treams-rs has none.
- **Results have names.** `cross_sections` returns `CrossSections` with
  `scattering`, `extinction` and `absorption`; `SMatrix.power` returns
  `PowerBalance`; `SMatrix.bands` returns `BandModes`. A field name replaces a
  tuple position.

## One name per class and concept

Each class is defined under its public name, so `type(x).__name__` is
the name to import. Where treams uses another name, `treams_rs._upstream` records
the replacement, and accessing the treams name raises an `AttributeError` that names
it:

```python
import pytest
import treams_rs as tr

with pytest.raises(AttributeError, match="SMatrix"):
    tr.SMatrices
```

`_upstream` holds one table that drives both these errors and the
[name map](../coming-from-treams/names.md). For example,
treams-rs `SMatrix` is the full two-port network (treams `SMatrices`), and
`ScatteringBlock` is one block (treams `SMatrix`).

## treams names as delegations

Physics classes use physics names: `cross_sections`, `circular_dichroism`,
`with_polarization`, `kind`, `polarization`, `scatter`, `cascade`. The treams names
(`xs`, `cd`, `changepoltype`, `modetype`, `poltype`, `illuminate`, `add`, ...) remain
as one-line methods or properties that call the corresponding physics member.
Each class groups them at its end under the comment `# treams-compatible names`.
They add no behaviour of their own. The modules
`special`, `sw`, `cw`, `pw`, `lattice`, `coeffs`, `misc`, `ebcm` and `io` keep the
treams function and argument names unchanged.

## Material and medium, polarization and pol

| Word | Meaning | Example |
|---|---|---|
| `material` | What a particle or layer is made of | `sphere_tmatrix(material=4.0)` |
| `medium` | The space around an object | `TMatrix.medium`, `plane_wave(medium=...)` |
| `negative_medium`, `positive_medium` | The media below and above a planar structure | `SMatrix.negative_medium` |
| `polarization` | The convention, `'helicity'` or `'parity'` | `sphere_tmatrix(polarization="parity")` |
| `pol` | The index 0 or 1 of a mode, or the state of a plane wave | `plane_wave(..., pol=1)`, `basis.pol` |

The treams-style modules keep treams' `poltype=`. The `TMatrix` constructor,
`plane_wave` and `operators.PhysicsArray` also accept treams' `material=` and
`poltype=`; there `material=` means the surrounding medium, as in treams.
`TMatrix.material` and `TMatrix.poltype` delegate to `medium` and `polarization`.

## Arrays and metadata

The NumPy implementations follow these rules. Framework objects retain their
framework arrays and its mutation rules; see [framework adapters](adapters.md).

- **`.array` is read-only.** Objects own their storage and mark it read-only, so
  no caller can change a T-matrix behind its metadata. Copy the array to edit it.
- **Plain indexing gives plain values.** Indexing `.array`, or arithmetic on an
  `operators.PhysicsArray`, returns NumPy values without metadata. Indexing a
  `TMatrix` with a basis returns a `TMatrix` in those modes.
  `SMatrix.block(outgoing, incoming)` returns a `ScatteringBlock` that shares the
  storage and knows its ports.
- **Rotations keep metadata.** Rotating a plane-wave basis about z also rotates
  its lattice and Bloch vector. An xy lattice, a z axis and a 3D cell stay
  aligned with the Cartesian axes under any z rotation. A rotation that would turn
  another lattice, such as a chain along x, into an oblique one raises `ValueError`
  rather than drop the lattice. Quarter turns are exact.

## NumPy ufuncs

Every native ufunc follows one rule set, checked by
[`test_ufunc_contract.py`](https://github.com/yaugenst/treams-rs/blob/a2e66b9c60f9aa26cc602847e9bf1ae761fab0c3/tests/bindings/test_ufunc_contract.py): a
vectorized call equals the per-element calls bit for bit. Results do not depend on
the memory layout of the operands (broadcast, reversed, strided, unaligned or
Fortran-ordered), on masked or aliased outputs, on the number of threads, or on
which dtype loop serves a value-preserving cast. Invalid mode labels raise `ValueError` in every layout.

Public differentiable functions wrap these native implementations to select an
adapter from framework inputs. NumPy calls retain the native behavior and ufunc
attributes and methods, but public wrappers need not be `numpy.ufunc` instances.
Only direct function calls support autodiff; ufunc methods and `out`/`where`
remain on the NumPy path. See [framework adapters](adapters.md).

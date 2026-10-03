---
description: The treams names that treams-rs spells differently, member by member, and the keyword changes.
---

# Name map

Most treams names work unchanged: the numerical namespaces keep their function
and argument names, and physics objects answer to the treams member names. The
tables list every treams name that treams-rs spells differently.

A treams class name raises an `AttributeError` that names the replacement:

```python exec
import treams_rs as tr

try:
    tr.TMatrixC
except AttributeError as error:
    assert "treams_rs.CylindricalTMatrix" in str(error)
```

<!-- generated: upstream-names -->

## treams top-level names

| treams | treams-rs | Note |
| --- | --- | --- |
| `treams.SphericalWaveBasis` | `treams_rs.SphericalBasis` |  |
| `treams.CylindricalWaveBasis` | `treams_rs.CylindricalBasis` |  |
| `treams.PlaneWaveBasisByUnitVector` | `treams_rs.PlaneWaveBasis` |  |
| `treams.PlaneWaveBasisByComp` | `treams_rs.PlaneWavePorts` |  |
| `treams.TMatrixC` | `treams_rs.CylindricalTMatrix` |  |
| `treams.SMatrices` | `treams_rs.SMatrix` | treams-rs SMatrix is the full two-port network (treams SMatrices); one block is ScatteringBlock (treams SMatrix). |
| `treams.BField` | `treams_rs.operators.BField` |  |
| `treams.ChangePoltype` | `treams_rs.operators.ChangePoltype` |  |
| `treams.DField` | `treams_rs.operators.DField` |  |
| `treams.EField` | `treams_rs.operators.EField` |  |
| `treams.Expand` | `treams_rs.operators.Expand` |  |
| `treams.ExpandLattice` | `treams_rs.operators.ExpandLattice` |  |
| `treams.FField` | `treams_rs.operators.FField` |  |
| `treams.GField` | `treams_rs.operators.GField` |  |
| `treams.HField` | `treams_rs.operators.HField` |  |
| `treams.Permute` | `treams_rs.operators.Permute` |  |
| `treams.PhysicsArray` | `treams_rs.operators.PhysicsArray` |  |
| `treams.Rotate` | `treams_rs.operators.Rotate` |  |
| `treams.Translate` | `treams_rs.operators.Translate` |  |
| `treams.bfield` | `treams_rs.operators.bfield` |  |
| `treams.changepoltype` | `treams_rs.operators.changepoltype` |  |
| `treams.dfield` | `treams_rs.operators.dfield` |  |
| `treams.efield` | `treams_rs.operators.efield` |  |
| `treams.expand` | `treams_rs.operators.expand` |  |
| `treams.expandlattice` | `treams_rs.operators.expandlattice` |  |
| `treams.ffield` | `treams_rs.operators.ffield` |  |
| `treams.gfield` | `treams_rs.operators.gfield` |  |
| `treams.hfield` | `treams_rs.operators.hfield` |  |
| `treams.permute` | `treams_rs.operators.permute` |  |
| `treams.rotate` | `treams_rs.operators.rotate` |  |
| `treams.translate` | `treams_rs.operators.translate` |  |
| `treams.config` | none | treams-rs has no global settings; pass poltype= or polarization= explicitly (default helicity). |
| `treams.util` | none | treams-rs has no util module: bases are ordered sets, and PhysicsArray is treams_rs.operators.PhysicsArray. |

## Members of treams objects

| Class | treams member | treams-rs name or note |
| --- | --- | --- |
| `TMatrix` | `material` | `medium`. |
| `TMatrix` | `poltype` | `polarization`. |
| `TMatrix` | `modetype` | `modetype`. treams-rs keeps the treams name. Always ('singular', 'regular'): outgoing rows, incident columns. |
| `TMatrix` | `changepoltype` | `with_polarization`. |
| `TMatrix` | `expand` | `in_basis`. |
| `TMatrix` | `xs` | `cross_sections`. |
| `TMatrix` | `xs_ext_avg` | `average_cross_sections`. Read its extinction field. |
| `TMatrix` | `xs_sca_avg` | `average_cross_sections`. Read its scattering field. |
| `TMatrix` | `cd` | `circular_dichroism`. |
| `TMatrix` | `db` | `duality_breaking`. |
| `TMatrix` | `chi` | `electromagnetic_chirality`. |
| `TMatrix` | `sphere` | `sphere`. treams-rs keeps the treams name. sphere_tmatrix and multilayer_sphere_tmatrix take keywords. |
| `TMatrix` | `interaction` | `interaction`. treams-rs keeps the treams name. |
| `TMatrix` | `latticeinteraction` | `latticeinteraction`. treams-rs keeps the treams name. |
| `TMatrix` | `expandlattice` | `expandlattice`. treams-rs keeps the treams name. |
| `TMatrix` | `permute` | `permute`. treams-rs keeps the treams name. It raises TypeError: permutations act on plane-wave ports. |
| `TMatrix` | `cluster` | Use Cluster(particles, positions=xyz); Cluster.solve() returns the coupled T-matrix. |
| `TMatrix` | `ann` | treams-rs objects keep basis, k0, medium and polarization as attributes instead of array annotations. |
| `TMatrix` | `relax` | treams-rs objects keep basis, k0, medium and polarization as attributes instead of array annotations. |
| `TMatrix` | `efield` | Fields belong to waves: use response.scatter(incident).efield(points); treams_rs.operators.efield builds a field matrix. |
| `TMatrix` | `hfield` | Fields belong to waves: use response.scatter(incident).efield(points); treams_rs.operators.efield builds a field matrix. |
| `TMatrix` | `dfield` | Fields belong to waves: use response.scatter(incident).efield(points); treams_rs.operators.efield builds a field matrix. |
| `TMatrix` | `bfield` | Fields belong to waves: use response.scatter(incident).efield(points); treams_rs.operators.efield builds a field matrix. |
| `TMatrix` | `gfield` | Fields belong to waves: use response.scatter(incident).efield(points); treams_rs.operators.efield builds a field matrix. |
| `TMatrix` | `ffield` | Fields belong to waves: use response.scatter(incident).efield(points); treams_rs.operators.efield builds a field matrix. |
| `CylindricalTMatrix` | `material` | `medium`. |
| `CylindricalTMatrix` | `poltype` | `polarization`. |
| `CylindricalTMatrix` | `modetype` | `modetype`. treams-rs keeps the treams name. Always ('singular', 'regular'): outgoing rows, incident columns. |
| `CylindricalTMatrix` | `changepoltype` | `with_polarization`. |
| `CylindricalTMatrix` | `expand` | `in_basis`. |
| `CylindricalTMatrix` | `xw` | `cross_widths`. |
| `CylindricalTMatrix` | `xw_ext_avg` | `average_cross_widths`. Read its extinction field. |
| `CylindricalTMatrix` | `xw_sca_avg` | `average_cross_widths`. Read its scattering field. |
| `CylindricalTMatrix` | `cylinder` | `cylinder`. treams-rs keeps the treams name. cylinder_tmatrix and multilayer_cylinder_tmatrix take keywords. |
| `CylindricalTMatrix` | `interaction` | `interaction`. treams-rs keeps the treams name. |
| `CylindricalTMatrix` | `latticeinteraction` | `latticeinteraction`. treams-rs keeps the treams name. |
| `CylindricalTMatrix` | `expandlattice` | `expandlattice`. treams-rs keeps the treams name. |
| `CylindricalTMatrix` | `permute` | `permute`. treams-rs keeps the treams name. It raises TypeError: permutations act on plane-wave ports. |
| `CylindricalTMatrix` | `cluster` | Use Cluster(particles, positions=xyz); Cluster.solve() returns the coupled T-matrix. |
| `CylindricalTMatrix` | `ann` | treams-rs objects keep basis, k0, medium and polarization as attributes instead of array annotations. |
| `CylindricalTMatrix` | `relax` | treams-rs objects keep basis, k0, medium and polarization as attributes instead of array annotations. |
| `CylindricalTMatrix` | `efield` | Fields belong to waves: use response.scatter(incident).efield(points); treams_rs.operators.efield builds a field matrix. |
| `CylindricalTMatrix` | `hfield` | Fields belong to waves: use response.scatter(incident).efield(points); treams_rs.operators.efield builds a field matrix. |
| `CylindricalTMatrix` | `dfield` | Fields belong to waves: use response.scatter(incident).efield(points); treams_rs.operators.efield builds a field matrix. |
| `CylindricalTMatrix` | `bfield` | Fields belong to waves: use response.scatter(incident).efield(points); treams_rs.operators.efield builds a field matrix. |
| `CylindricalTMatrix` | `gfield` | Fields belong to waves: use response.scatter(incident).efield(points); treams_rs.operators.efield builds a field matrix. |
| `CylindricalTMatrix` | `ffield` | Fields belong to waves: use response.scatter(incident).efield(points); treams_rs.operators.efield builds a field matrix. |
| `CylindricalTMatrix` | `from_array` | Use solve_periodic(...).to_cylindrical(basis) for a z-periodic array of spheres. |
| `SMatrix` | `material` | Returns (positive_medium, negative_medium), the treams order of the pair. |
| `SMatrix` | `poltype` | `polarization`. |
| `SMatrix` | `changepoltype` | `with_polarization`. |
| `SMatrix` | `add` | `cascade`. |
| `SMatrix` | `illuminate` | `scatter`. |
| `SMatrix` | `tr` | `power`. |
| `SMatrix` | `cd` | `circular_dichroism`. |
| `SMatrix` | `periodic` | `transfer_matrix`. |
| `SMatrix` | `bands_kz` | `bands`. |
| `SMatrix` | `from_array` | Use solve_periodic(...).to_smatrix(ports) for a periodic array. |
| `Wave` | `material` | `medium`. |
| `Wave` | `poltype` | `polarization`. |
| `Wave` | `modetype` | `kind`. |
| `Wave` | `changepoltype` | `with_polarization`. |
| `Wave` | `expand` | `in_basis`. |
| `PlaneWave` | `material` | `medium`. |
| `PlaneWave` | `poltype` | `polarization`. |
| `PlaneWave` | `modetype` | `kind`. |
| `PlaneWave` | `expand` | `in_basis`. |
| `PlaneWave` | `changepoltype` | `with_polarization`. |
| `Material` | `from_n` | `from_refractive_index`. |
| `Material` | `from_nmp` | `from_helicity_indices`. |

<!-- end generated -->

## Reading the table

- **treams top-level names** lists module-level names. `none` means treams-rs
  has no counterpart, and the note says what to use instead.
- **Members of treams objects** lists members of the treams-rs classes. A name
  in code font is the treams-rs member that the treams member calls; both give
  the same result.
- **treams-rs keeps the treams name** marks a member that has no separate
  treams-rs name, such as `TMatrix.interaction`,
  `latticeinteraction`, `expandlattice`, `permute` and `sphere`, and
  `CylindricalTMatrix.cylinder`.
- A row without a name in code font has no treams-rs member: the note shows the
  treams-rs way, and the member raises an `AttributeError` with that note.

## Keywords

The numerical namespaces, `operators` and the class constructors such as
`TMatrix(array, k0=..., basis=..., material=..., poltype=...)` keep the treams
keywords. The factories and methods of the physics objects use these keywords
instead:

| treams keyword | treams-rs keyword | Note |
| --- | --- | --- |
| `poltype` | `polarization` | The convention, `"helicity"` or `"parity"`. The numerical namespaces, `operators` and `diff` keep `poltype`. |
| `modetype` | `kind` | `"regular"` or `"singular"` for multipole waves, `"up"` or `"down"` for plane-wave ports. |
| `pol` | `pol` | The state of a plane wave or the index 0 or 1 of a mode, as in treams. |
| `materials` (a list ending with the embedding medium) | `material` and `medium` | `material` is the particle or layer, `medium` the space around it. |
| `materials` (two media of an interface) | `negative_medium` and `positive_medium` | The media below and above, along the basis normal. |
| `radii` | `radius` | Single-layer factories such as `sphere_tmatrix`; the multilayer factories keep `radii`. |
| `kvec` | `direction` | Any length; `k0` and the medium set the wavenumber. |
| `lmax`, `mmax` | `lmax`, `mmax` | Unchanged. |

# Upstream findings

Findings from comparison with treams 0.4.5 and the reference source pinned in
architecture.md. These are reproduced behavior, not claims that all current
upstream branches remain affected. No upstream issues or maintainer messages
have been sent from this project.

The skew-cell omission is also tracked upstream in
[issue #14](https://github.com/tfp-photonics/treams/issues/14), which was open when
checked on 2026-09-11. This was independently reproduced during the rewrite;
it is not claimed as a new discovery.

## Behavioral defects

| Finding | Evidence and impact | Regression |
| --- | --- | --- |
| HDF5 embedding chirality is lost | `save_hdf5` writes `embedding/chirality`, while `load_hdf5` reads `embedding/chirality_parameter`. A material with kappa=0.08 reloads with kappa=0, so the numerical T matrix acquires incorrect embedding metadata. | [HDF5 tests](../tests/test_io.py) check chirality round trips and legacy names; minimal upstream example below. The rewrite writes compatible hard links to one dataset. |
| HDF5 single-matrix save fails | Passing one TMatrix directly to `save_hdf5` raises IndexError because it slices a zero-dimensional object array with `[:]`. Passing `[tm]` avoids this particular error. | Same example below: replace `[tm]` with `tm` in the save call. `test_hdf5_single_matrix_and_path` checks the rewrite's scalar case. |
| HDF5 origins and local-mode indices | Upstream's loader discards origins when forming the union of incident/scattered bases: an origin (0.7,0.2,0.1) reloads as (0,0,0). Its writer and reader also disagree on local-mode index names (`pidx` versus `position_index`); multiple-origin files can fail to load. | [HDF5 tests](../tests/test_io.py) exercise multiple origins, legacy index names, rectangular mode sets and unit conversion. Compatibility with upstream's reader is qualified for global zero-origin matrices; its local-basis union still fails even with corrected index names. |
| Multi-direction parity interface | `SMatrices.interface` applies a fixed 2-by-2 polarization mask to an N-by-N matrix. For transverse directions (0.2,0.1), (-0.3,0.5), k0=1.7 and materials [1,2.5], direct parity construction raises an IndexError (mask axis 2 versus matrix axis 4). | `test_slab_reference` in [S-matrix tests](../tests/test_smatrix.py) checks our direct parity construction against upstream helicity construction followed by polarization conversion, including multiple directions. |
| Plane-wave chirality interval | For a single propagating vacuum wave with transverse k=(0.2,0.3), k0=1.3, the positive-helicity coefficient should stay 2 under averaging. Upstream returns 2 at z=(0,0), 2.56210 at (0,1), and 4.83423 at (0,2): its hyperbolic average uses Re(kz) instead of Im(kz). It also drops the complex up/down interference phase for shifted intervals and ignores the position of a zero-width interval. | [Chirality-density tests](../tests/test_chirality_density.py) check direct Cartesian E/H quadrature, single-wave invariance, native interval additivity and full k/normal/endpoint adjoints. Corrected xy-basis forms preserve the origin convention and retain the coherent cross phase. |
| EBCM radial area factor | Upstream `ebcm.qmat` uses `sin(theta) * [r, -dr, 0]`; the surface element requires another factor of r. For r=0.3(1+0.23 cos²(theta)), lossless eps=3.1, kappa=0.07 and k0=1.3, max abs(SᴴS-I) stays about 0.00135 at degrees 2, 4 and 6. Restoring r gives 6.39e-6, 2.02e-7 and 1.22e-8. With identical vacuum inside/outside, degree 2 produces spurious max abs(T)=5.88e-4; the corrected integral gives 8.41e-18. | [EBCM tests](../tests/test_ebcm.py) cover sphere/Mie agreement, lossless convergence, Hypothesis zero-contrast shapes and both adjoints. The default includes r; explicit `legacy=True` reproduces upstream. |
| Complex plane-wave branch | For k=(0.2+i, 0.1+0.3i, 1.3-0.8i), an extra principal square root in the angular coefficients can disagree with the Cartesian polarization branch. Reconstructed fields differ by about 1.25 on the recorded samples; the corrected expansion is accurate to about 1e-13. | `test_complex_transverse_branch_reconstruction_and_gradient` in [plane-expansion tests](../tests/test_plane_expansion.py). |
| Missing skew-cell diffraction orders | For lattice rows (2,-3), (0,1.7), zero Bloch vector and cutoff 4, upstream omits integer reciprocal orders (-2,1) and (2,-1). Both lie inside the cutoff. | `test_diffraction_basis_skew_completeness` in [channel tests](../tests/test_channels.py), checked against independent integer enumeration. |

## Numerical accuracy limitations

| Finding | Evidence and impact | Regression |
| --- | --- | --- |
| Near-axis spherical translation | Forming the angular coordinate through cos(theta) can round small transverse offsets to the axis, losing nonzero terms and derivatives. Cartesian evaluation is checked against a resolved-angle extrapolation down to offsets of 1e-300. | `test_near_axis_translation_against_resolved_angle_limit` in [invariant tests](../tests/test_invariants.py). |
| Small-angle spherical rotation | For degree 1 and theta=1e-10, upstream rounds an off-diagonal coefficient to zero; the angular generator requires theta/sqrt(2)=7.071e-11. | `test_small_angle_generator` in [rotation tests](../tests/test_rotation.py) checks first-order entries down to theta=1e-100. |
| Axial complex-medium illumination | Rounding of complex kz/k away from exactly 1 leaks forbidden azimuthal orders for an exactly axial plane wave. For k0=1.3, material (2+0.1i,1.1,0.02), negative helicity and degree 4, the largest forbidden coefficient is 3.55e-7. | `test_plane_to_spherical_reference` in [plane-wave tests](../tests/test_plane.py) requires these modes to be exactly zero and uses scale-invariant angular coefficients as an independent oracle. |
| Cylindrical Ewald automatic split | At order -6, k=sqrt(1.3²-0.2²), Bloch k=0.1, period=7.2 and displacement=0.8, the automatic and converged eta=0.7 sums differ by 0.0067592. In the nine-cylinder benchmark this becomes about 8.4e-8 in the final S matrix. Both benchmark implementations use eta=0.7, retaining the comparison tolerance. | `test_large_cylindrical_cell_ewald_split_invariance` in [lattice tests](../tests/test_lattice.py) checks automatic and explicit splits against the converged reference; details in [benchmarks](benchmarks.md#complete-cylindrical-arrays). |

The G/F field normalization also differs between spherical and cylindrical/plane
families and between helicity and parity. This is documented as a preserved
convention, not classified here as a confirmed physical defect. Likewise, normal-
incidence polarization gauges, diffraction thresholds and eigenvalue degeneracies
are mathematical restrictions, not automatically upstream bugs.

`SMatrices.cd` is also a terminology caveat: upstream's second return value is
the normalized contrast of total outgoing power T+R, although its docstring calls
it absorption CD. The rewrite preserves this formula and names it explicitly.

## HDF5 reproducer

Run with treams 0.4.5 and h5py installed. The file stays in memory.

```python
import h5py
import treams
import treams.io

medium = treams.Material(1.3, 1.1, 0.08)
tm = treams.TMatrix.sphere(1, 1.3, 0.2, [treams.Material(3), medium])
with h5py.File("roundtrip.h5", "w", driver="core", backing_store=False) as f:
    treams.io.save_hdf5(f, [tm])
    loaded = treams.io.load_hdf5(f)[0]
    print(tm.material.kappa, loaded.material.kappa)  # 0.08 0
```

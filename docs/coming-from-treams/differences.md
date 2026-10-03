---
description: Where treams-rs and treams 0.4.5 give different results, why, and the tests that show it.
---

# Differences from treams 0.4.5

The comparison point is treams 0.4.5 at the
[reference commit](conventions.md#reference-version); later treams versions may
differ. Three entries are also treams issues:
[skew-cell diffraction orders (#14)](https://github.com/tfp-photonics/treams/issues/14),
[evanescent scattering widths (#20)](https://github.com/tfp-photonics/treams/issues/20)
and [chiral-sphere parity coupling (#27)](https://github.com/tfp-photonics/treams/issues/27).

Each difference has tests in treams-rs; most compare with treams or with an
independent reference.

## Deliberate differences

- **Complex results.** `special` functions return complex128, also for real
  inputs: `special.jv(1, 0.5)` is `(0.2422684576748739+0j)`, where treams
  returns a float. `lpmv` with real arguments, `wigner3j` and the coordinate
  transforms of real inputs return float64.
- **Errors instead of NaN.** A special-function evaluation that treams returns
  as NaN raises `ValueError`: `special.hankel1(0, 0)` is `nan+nanj` in treams.
- **Label bounds.** Degrees and cylindrical orders go up to 128; see
  [label bounds](conventions.md#label-bounds). treams has no bound.
- **Wigner 3j symbols.** `special.wigner3j` continues each recurrence for as
  long as the symbols grow in its direction. treams switches direction at a
  point that ignores the orders and loses accuracy for extreme orders; see
  [the accuracy table](#numerical-accuracy-limitations).
- **EBCM surface element.** `ebcm.qmat` includes the radial factor r of the
  surface element by default. `radial_area_factor=False` reproduces treams,
  which omits it; see [the defect table](#behavioral-defects).
- **Duplicate mode labels.** `sw.translate_periodic` and `cw.translate_periodic`
  build bases from their label arrays and drop repeated modes. With
  `out=([1, 1, 1], [0, 0, 1], [0, 0, 0])`, treams returns a (3, 3) matrix and
  treams-rs a (2, 2) matrix.
- **One degree per call in `coeffs`.** treams' `coeffs.mie`, `mie_cyl` and
  `fresnel` broadcast over leading axes. treams-rs `mie` takes one degree
  `l` and `mie_cyl` one order `m`; an array raises `ValueError`. `fresnel` takes
  one interface stack. Loop in Python for several.
- **No global configuration.** treams-rs has no `config.POLTYPE`. Every
  function and object takes its convention explicitly, with `"helicity"` as the
  default.
- **Partial slab ports.** A slab retains both polarizations inside every layer
  and selects the requested external ports after solving the complete stack.
  treams projects at each interface instead, discarding internal reflections
  through unrepresented polarizations. Selected outgoing powers need not sum to
  one even for a lossless slab: power in omitted external channels is not counted.
  [Layer tests](../../tests/smatrix/test_layers.py) compare with the projection of
  a complete treams response and bound each selected power by its complete value.
- **Gain media and planar material branches.** High-level interfaces and slabs
  match tangential electric and magnetic fields in the same plane-wave convention.
  The closed-form `coeffs.fresnel` remains the treams formula; it differs for some
  gain media. Plane-wave polarizations use the principal norm `sqrt(k dot k)`,
  including exactly axial waves. Electric displacement and magnetic induction
  follow the constitutive equations instead of a signed-index shortcut.
  [Layer physics tests](../../tests/smatrix/test_smatrix.py) check boundary
  continuity, Maxwell's curl equations, constitutive relations and an independent
  TE admittance solution; framework tests also check their gradients.
  Material and normal wavenumbers retain their nonnegative-imaginary-part
  branches. This convention does not select a causal branch for an arbitrary
  dispersive gain model.

  High-level plane waves and planar layers reject material branches incompatible
  with these polarizations. With principal `n = sqrt(epsilon * mu)` and
  `Z = sqrt(mu / epsilon)`, they require `mu / Z = n` and both `n - kappa` and
  `n + kappa` on the principal square-root branch: positive real part, or zero
  real part and nonnegative imaginary part. This excludes strong chirality with
  a negative real helicity index and double-negative-index materials. The numerical
  wave and coefficient functions still accept literal wavevectors and wavenumbers.
- **Compatibility power forms.** `poynting_avg_z` retains treams' normalization
  and complex-medium conventions, including a factor of one half for parity
  channels. It is not a physical flux matrix for arbitrary media.
  `SMatrix.power` computes physical transmission and reflection from the native
  tangential fields; use `0.5 * real(E cross conj(H))` for a local flux.
- **Subnormal Hankel orders.** Orders below the smallest normal float give the
  zero-order limit; SciPy returns NaN; see
  [the accuracy table](#numerical-accuracy-limitations).
- **Spheres and cylinders at several positions.** `operators.expand` and
  `operators.expandlattice` between spherical and cylindrical bases add every
  pair of positions; treams pairs only equal particle indices. See
  [below](#spheres-and-cylinders-at-several-positions).
- **Circular dichroism of S-matrices.** treams' `SMatrices.cd` returns as its
  second value the contrast of the total outgoing power T + R, although its
  docstring calls it absorption CD. `SMatrix.circular_dichroism` keeps the
  formula and names the value `outgoing_power`.

## Defects in treams 0.4.5 {#behavioral-defects}

| Finding | Evidence and impact | Tests |
| --- | --- | --- |
| S-matrix plane rotation fails during annotation propagation | Rotating a two-polarization xy S-matrix by 0.3 radians gives incompatible input and output basis annotations; the reconstruction raises `AttributeError: 'tuple' object has no attribute 'bycomp'`. The plane-wave coefficient transform is the identity while both channel bases rotate. | `test_smatrix_sparse_coordinate_transforms` in [matrix-method tests](../../tests/api/test_matrix_methods.py) checks unchanged coefficients and the inverse basis rotation. |
| Bound G/F field operators miss a constructor parameter | `GField` and `FField` inherit the one-argument `FieldOperator(r)` constructor, although their functions need both the helicity and the coordinates. `tm.gfield(1, r)` and `tm.ffield(1, r)` raise `TypeError`. | The G/F cases in [matrix-method tests](../../tests/api/test_matrix_methods.py) compare the treams-rs bound methods with the free treams functions followed by T-matrix multiplication. |
| Saved operator attributes change owner | Save `a.rotate` from a degree-1 `PhysicsArray`, then access `b.rotate` of a degree-2 array. The saved attribute then returns a 16-by-16 operator instead of the 6-by-6 one: the shared descriptor overwrites its bound object. | `test_physics_array_bound_attribute_ownership_and_keyword_arguments` in [operator tests](../../tests/api/test_operators.py) checks independent bindings and numerical transformation identities. |
| Gmsh physical surface groups use volume tags | In a model that already holds a box with volume tag 101 and surface tags 1 to 6, treams creates sphere volumes 1 and 2 but assigns their physical surface groups to box surfaces 1 and 2. Reproduced with Gmsh 4.15.2. treams-rs reads the boundary surfaces of each sphere and lets Gmsh allocate tags. | `test_gmsh_helper_actual_surface_tags_and_complete_geometry` in [I/O tests](../../tests/api/test_io.py), plus a surface-mesh generation check in a populated model. |
| Computation metadata without a mesh raises NameError | `_save_computation_hdf5` refers to `issues`, which exists only in its caller. A computation dictionary without a mesh or the semi-analytical keyword raises `NameError: name 'issues' is not defined`. treams-rs writes the available metadata and omits the v1 version marker in that case. | `test_mesh_and_reproducibility_files` in [I/O tests](../../tests/api/test_io.py) covers mesh paths, inline meshes, named source files and the case without a mesh. |
| Tiny nonzero Bloch vectors in axial reciprocal sums | `recsumcw1d` divides by beta squared in its coefficient recurrence. With k = 2+0.15i and kpar = 1e-240, beta is nonzero but its square underflows, and treams raises `ZeroDivisionError`, also at order zero. treams-rs uses nonnegative powers of beta and never divides by it. | `test_axial_reciprocal_tiny_bloch_has_the_zero_bloch_limit` in [lattice tests](../../tests/lattice/test_lattice_decomposition.py) checks the finite limit at q = 0 up to order 12. Random comparisons with treams stay where treams is defined. |
| Direct spherical 1D axis shortcut | `dsumsw1d_shift` tests y = z = 0 and treats x as an axial displacement, although its lattice runs along z. With l = 2, m = -1, k = 2.1+0.2i, kpar = 0.13, a = 1.7, r = (0.2, 0, 0) and shell 2, treams returns zero instead of -0.00110290923-0.00270676694i. The scalar 1D half-cell branch also discards the sign of negative shifts, which breaks the reflection symmetry. treams-rs pairs equidistant images in half-cell shells and keeps the shift and Bloch phase; at exactly half a cell the grouping changes, so the sum has no derivative there. | [Direct-shell tests](../../tests/lattice/test_lattice_decomposition.py) compare with independent Cartesian shell sums and check the reflection at negative half cells. |
| HDF5 embedding chirality is lost | `save_hdf5` writes `embedding/chirality` and `load_hdf5` reads `embedding/chirality_parameter`. A medium with kappa = 0.08 reloads with kappa = 0. treams-rs writes hard links to one dataset under both names. | [HDF5 tests](../../tests/api/test_io.py) check chirality round trips and the treams names; see the [reproducer](#hdf5-reproducer). |
| HDF5 single-matrix save fails | Passing one `TMatrix` to `save_hdf5` raises `IndexError`: it slices a zero-dimensional object array with `[:]`. Passing `[tm]` avoids the error. | In the [reproducer](#hdf5-reproducer), replace `[tm]` with `tm`. `test_hdf5_single_matrix_and_path` checks the single-matrix case in treams-rs. |
| HDF5 positions and local-mode indices | The treams loader discards positions when it forms the union of the incident and scattered bases: a position (0.7, 0.2, 0.1) reloads as (0, 0, 0). Its writer and reader also disagree on the name of the local-mode index (`pidx` against `position_index`), so files with several positions can fail to load. | [HDF5 tests](../../tests/api/test_io.py) cover several positions, the treams index names, rectangular mode sets and unit conversion. treams reads treams-rs files of global matrices at the origin; its local-basis union fails even with corrected index names. |
| Multi-direction parity interface | `SMatrices.interface` applies a fixed 2-by-2 polarization mask to an N-by-N matrix. For the transverse directions (0.2, 0.1) and (-0.3, 0.5), k0 = 1.7 and materials [1, 2.5], the parity construction raises `IndexError` (mask axis 2 against matrix axis 4). | `test_slab_reference` in [S-matrix tests](../../tests/smatrix/test_smatrix.py) compares the treams-rs parity construction with the treams helicity construction followed by a polarization change, also for several directions. |
| Plane-wave chirality interval | For one propagating vacuum wave with transverse k = (0.2, 0.3) and k0 = 1.3, the positive-helicity coefficient must stay 2 under averaging. treams returns 2 on z = (0, 0), 2.56210 on (0, 1) and 4.83423 on (0, 2): its hyperbolic average uses Re(kz) instead of Im(kz). It also drops the complex up/down interference phase for shifted intervals and ignores the position of a zero-width interval. | `test_chirality_interval_average_is_invariant_unlike_upstream` and the other [chirality-density tests](../../tests/smatrix/test_chirality_density.py) check direct Cartesian E/H quadrature, single-wave invariance, interval additivity and the gradients with respect to k, the normal and the endpoints. |
| EBCM radial area factor | `ebcm.qmat` uses `sin(theta) * [r, -dr, 0]`; the surface element needs another factor r. For r = 0.3 (1 + 0.23 cos²(theta)), lossless eps = 3.1, kappa = 0.07 and k0 = 1.3, max abs(SᴴS - I) stays about 0.00135 at degrees 2, 4 and 6. With the factor it is 6.39e-6, 2.02e-7 and 1.22e-8. With equal vacuum inside and outside, degree 2 gives a spurious max abs(T) = 5.88e-4; the corrected integral gives 8.41e-18. | [EBCM tests](../../tests/tmatrix/test_ebcm.py) cover sphere/Mie agreement, lossless convergence, Hypothesis checks at zero contrast and both gradients. The default includes r; `radial_area_factor=False` reproduces treams. |
| Complex plane-wave branch | For k = (0.2+i, 0.1+0.3i, 1.3-0.8i), an extra principal square root in the angular coefficients can disagree with the branch of the Cartesian polarization. Reconstructed fields differ by about 1.25 on the recorded samples; the treams-rs expansion is accurate to about 1e-13. | `test_complex_transverse_branch_reconstruction_and_gradient` in [plane-expansion tests](../../tests/plane/test_plane_expansion.py). |
| Missing skew-cell diffraction orders (#14) | For lattice rows (2, -3) and (0, 1.7), a zero Bloch vector and cutoff 4, treams omits the reciprocal orders (-2, 1) and (2, -1). Both lie inside the cutoff. | `test_diffraction_basis_skew_completeness` in [diffraction-channel tests](../../tests/plane/test_diffraction_channels.py) compares with an independent integer enumeration. |
| Evanescent cylindrical scattering widths (#20) | Closed outgoing orders count as radiated power. For a lossless eps = 3.1 sphere of radius 0.2, z period 1.7, k0 = 1.3 and axial orders 0 and ±2π/period, the scattering width is 0.00409310 against an extinction width of 0.000248301. Without the closed orders, scattering equals extinction. Averaged widths must also exclude evanescent incident channels and normalize by the remaining ones. treams-rs excludes closed channels in illuminated and averaged widths. | `test_array_cross_width_with_evanescent_orders` in [periodic-conversion tests](../../tests/lattice/test_periodic_conversion.py) checks passivity, lossless conservation, explicit angular averages, chiral helicity cutoffs and invariance to closed channels. |
| Chiral sphere in the parity basis (#27) | The parity construction drops the magnetoelectric blocks of a chiral sphere. For radius 0.2, k0 = 1.3, eps = 3.1, mu = 1 and kappa = 0.12 in vacuum, it differs by 2.86987e-4 from the helicity construction followed by a change to parity. | `test_chiral_sphere_parity_retains_magnetoelectric_coupling` in [sphere tests](../../tests/tmatrix/test_sphere.py) checks the converted treams reference and the lossless optical theorem. |
| Brillouin reduction rejects hexagonal cells | `firstbrillouin2d` requires a minimal reciprocal basis and compares squared lengths with an absolute tolerance of 1e-14. A hexagonal basis ties exactly (the sum or difference of its rows is as long as the rows), so rounding decides. For reciprocal rows `pitch * [[1, 0], [0.5, sqrt(3)/2]]`, treams raises "Lattice vectors are not of minimal length" at pitch 0.3. treams-rs compares relative to the row lengths and keeps the absolute floor, so the reduction is the same at every scale. | `hexagonal_brillouin_zones_are_accepted_at_every_scale` and `brillouin_reduction_is_a_covariant_voronoi_projection` (scales 1e-3 to 1e8) in `lattice/geometry.rs`, and `test_brillouin_accepts_hexagonal_lattices_at_any_pitch` in [misc tests](../../tests/api/test_misc.py). |
| Plane-to-plane expansion mixes polarizations | Expanding a two-polarization basis into itself returns a 2-by-2 matrix of ones instead of the identity: the matching mask compares wavevectors but not polarizations. Reproduced for unit-vector and component bases in vacuum with k0 = 1.3. | `test_plane_basis_conversion_and_translation_preserve_fields` in [translation tests](../../tests/waves/test_translation.py) checks the identity, the inverse conversion, polarization preservation and Cartesian field reconstruction for propagating and evanescent modes in both conventions; see the [reproducer](#plane-expansion-reproducer). |
| Component-plane slicing loses alignment | Slicing a yz or zx `PlaneWaveBasisByComp` builds a new basis with the default xy alignment. The stored transverse components then refer to other axes. | `test_component_selection_keeps_alignment_unlike_upstream` and the Hypothesis field-column reconstruction for all families in [basis tests](../../tests/api/test_basis.py). |

## Numerical accuracy limitations

For lattice sums, "off by x" means an error of x times max(|S|, 1), where S is
an extended-precision value of the sum. An ulp is one unit in the last place of
a double.

| Finding | Evidence and impact | Tests |
| --- | --- | --- |
| Exact periodic diffraction poles | At the 350 nm end of the spectrum of the companion array, treams substitutes `1e-20+1e-20j` for a zero radiation denominator and `1e-7` in the singular lattice expression. Its finite value there is a regularized number, not a reference at the threshold. The mathematical pole itself is no defect. | The [published-application checks](../validation/published-applications.md) exclude the endpoint and approach it from one side at four points, including power conservation. [Threshold tests](../../tests/plane/test_diffraction_threshold.py) check finite tangential interfaces without changing the wavelength or adding a pseudoinverse. |
| Fractional Legendre cutoff, kept for compatibility | The docstring of the real-argument `lpmv` says that it calls SciPy, but treams first returns zero when abs(m) > degree. For m = 4, degree 2.3 and x = 0.3 it returns 0; SciPy and an independent 70-digit hypergeometric evaluation give about -1.9871849213085072. | treams-rs keeps this zero, as `test_real_degree_broadcast_poles_and_owned_pullback` in [fractional Legendre tests](../../tests/special/test_fractional_legendre.py) checks. Its tested Ferrers domain is abs(m) <= degree; the function beyond it is not implemented. The `ferrers` reference in the [high-precision reproducer](../../scripts/qualify_legendre.py) evaluates the unrestricted value. |
| Fractional-degree Legendre accuracy | For a fractional degree and real x, treams calls SciPy's `lpmv`. In the archived comparison, m = 11, degree 20.0625 and x = -0.875 gives 412007816248.8; a 50-digit mpmath value is 412007810333.4865, a relative error of 1.4e-8. treams-rs uses an original implementation of DLMF hypergeometric series, degree and order recurrences, and connection formulas, with scaled arithmetic for large intermediate values. It follows independent high-precision references where SciPy loses accuracy. | `test_real_degree_reference_and_advect` in [fractional Legendre tests](../../tests/special/test_fractional_legendre.py) includes this comparison. [Native Ferrers tests](../../crates/treams-core/src/special/ferrers.rs) check high-precision values, endpoint limits and recurrences across the numerical methods. |
| Subnormal Hankel orders in dependencies | With order = 5e-324 and z = 1+0j, SciPy 1.16.3 (which treams uses) returns NaN for `hankel2`. complex-bessel 0.2.0 gives a wrong finite value, about 0.97428+0.06122j instead of the zero-order limit 0.76520-0.08826j. These are defects of the dependencies, not of treams. | `test_hankel_subnormal_order_continuous_limit` and Hypothesis reflection and derivative checks in [special tests](../../tests/special/test_special.py). The native layer maps subnormal orders to their zero-order limit; the reference checks use the same limit only where SciPy fails. |
| Near-axis spherical translation | Forming the angular coordinate through cos(theta) can round small transverse offsets onto the axis and lose nonzero terms and derivatives. The Cartesian evaluation is checked against a resolved-angle extrapolation down to offsets of 1e-300. The treams `special` functions also return zero for the dipole A and B coefficients at theta = 1e-10, kr = 1.4+0.2i and phi = 0.4; closed forms give -2.20566e-11+1.53401e-10i and 6.35401e-11+4.87093e-11i. | `test_near_axis_translation_against_resolved_angle_limit` in [translation tests](../../tests/waves/test_translation.py), and `test_tiny_angle_translation_against_closed_dipole_form` in [translation-coefficient tests](../../tests/waves/test_translation_coefficients.py), for singular and regular coefficients down to theta = 1e-100. |
| Wigner 3j symbols in the classically forbidden regions | `wigner3j` recurs upward in j3 below a switch point that ignores the orders, and downward above it. With extreme orders either forbidden region can reach past the switch, and recurring into it is unstable. Downward into the lower region: (47 60 39; -47 8 39) is exactly 1.33e-11 (Racah formula in rational arithmetic), but treams returns -2.06e-8, and (88 66 56; -10 66 -56) = 5.79e-17 comes back as -2.16e-3. Upward into the upper region: (260 130 190; -260 129 131) = -1.51e-12 comes back as -2.77e-9, and (128 64 95; -128 64 64) = -6.912539e-8 as -6.912476e-8. These symbols weight every spherical translation coefficient. treams-rs continues each recurrence for as long as the symbols grow in its direction. Over sampled extreme-order symbols with j <= 260, its error times sqrt(2 j3 + 1) stays below 5e-13; elsewhere it agrees with the treams recurrence to rounding. | Exact values in `wigner3j_matches_exact_symbols_in_the_forbidden_regions` (`special/wigner.rs`) and in [Wigner tests](../../tests/special/test_wigner.py), which also compare random extreme-order symbols with exact Racah values, and the permutation, reflection and orthogonality properties in `properties/special.rs`, which fail for the treams recurrence. |
| Small-angle spherical rotation | For degree 1 and theta = 1e-10, treams rounds an off-diagonal coefficient to zero; the angular generator requires theta/sqrt(2) = 7.071e-11. | `test_small_angle_generator` in [rotation tests](../../tests/waves/test_rotation.py) checks first-order entries down to theta = 1e-100. |
| Axial complex-medium illumination | Rounding of the complex kz/k away from exactly 1 leaks forbidden azimuthal orders into an exactly axial plane wave. For k0 = 1.3, material (2+0.1i, 1.1, 0.02), negative helicity and degree 4, the largest forbidden coefficient is 3.55e-7. | `test_plane_to_spherical_reference` in [plane-wave tests](../../tests/plane/test_plane.py) requires these modes to be exactly zero and uses scale-invariant angular coefficients as an independent reference. |
| Cylindrical Ewald automatic split | The Ewald split parameter eta divides a lattice sum into a real-space and a reciprocal-space part; the exact sum does not depend on it. At order -6, k = sqrt(1.3² - 0.2²), Bloch k = 0.1, period 7.2 and displacement 0.8, the treams sums at the automatic split and at the converged eta = 0.7 differ by 0.0067592. In the nine-cylinder benchmark this becomes about 8.4e-8 in the final S-matrix. The 16-cylinder coupling matrix with period 12.8 also differs by up to 0.007466 at the automatic split; eta = 0.5, 0.8 and 1.0 agree with treams-rs within about 1.5e-11. Both benchmark implementations use eta = 0.7 and keep the comparison tolerance. | `test_large_cylindrical_cell_ewald_split_invariance` in [lattice tests](../../tests/lattice/test_lattice.py) checks the automatic and explicit splits against the converged reference; details in the [performance evidence](../performance/evidence.md#complete-cylindrical-arrays). |
| Ewald sums shifted slightly off their lattice plane or axis | `lsumsw2d_shift`, `lsumcw1d_shift` and `lsumsw1d_shift` evaluate their reciprocal orders from Kambe integrals at eta = -i / (k s eta) for the distance s from the plane or axis; these cancel as s falls. With k = 2.1, the lattice diag(1.7, 1.8) and the in-plane shift (0.6, 0.2), the degree-4 sum is 4e4 off at s = 1e-12 and not finite from about 1e-50 to 1e-100. Over eight 2D spherical, five 1D cylindrical and three 1D spherical probe configurations of degree up to 12, treams is off by up to 1e117 from s = 1e-100 to 1e-3 and by 4e-8 from 1e-2 to 0.1; it is exact only below about 1e-101. Periodic couplings of particles at almost equal heights inherit the error. | The native reciprocal parts sum the series in t = (k s eta)^2 of their reduced integrals near the plane or axis (see [numerical limits](../validation/numerical-limits.md#lattice-sums)): `test_tiny_out_of_plane_shift_is_continuous`, `test_periodic_coupling_at_almost_equal_heights` and `test_tiny_normal_shift_pullbacks_are_continuous` in [lattice tests](../../tests/lattice/test_lattice.py), and `sums_tend_to_their_values_on_the_plane_and_axis` in `properties/lattice/mod.rs`. |
| Small explicit Ewald splits | treams sums a fixed number of real-space shells. At splits far below the automatic one this leaves 3D sums up to 4e-2 and 2D sums up to 0.11 off, and below every automatic split cancelling Ewald parts leave errors up to 2.5e6. | See [Small explicit Ewald splits](#small-explicit-ewald-splits). |
| Degree-6 EBCM cancellation floor | On the equatorially symmetric benchmark surface, the m = 0 entries with odd l_out + l_in are exactly zero. An independent 80-digit calculation gives magnitudes below 8e-82; treams leaves up to 6.62e-10 and treams-rs up to 3.94e-10 with 48, 96 and 192 nodes. The absolute integrand reaches a scale where machine epsilon times its integral is 1.62e-10. Tighter SciPy tolerances report rounding without changing the result. Both packages reach this double-precision floor; it is separate from the missing radial area factor. | [Reproducer](../../scripts/qualify_ebcm_cancellation.py), [raw results](../../benchmarks/results/ebcm-l6-cancellation.json) and the [performance evidence](../performance/evidence.md#axisymmetric-ebcm). The comparison tolerance stays strict, and no degree-6 speedup is claimed. |

The G/F field normalization differs between spherical and cylindrical or plane
waves and between helicity and parity (see
[field normalization](conventions.md#field-normalization)). treams-rs keeps it
as a convention. Normal-incidence polarization gauges, diffraction thresholds
and eigenvalue degeneracies are mathematical restrictions of both packages.

### Small explicit Ewald splits

treams sums at most 10 (3D) or 20 (2D) real-space shells and stops once a pair
of shells changes the sum by less than 1e-10. Splits with |k eta| far below the
automatic one need more shells:

- At k = 0.5i and eta = -0.5i (k eta = 0.25), its 3D sums in the unit cube are
  up to 4e-2 off. At k = 0.15+0.03i and eta = 0.6, its 2D sums in the unit square
  are up to 0.11 off.
- Below every automatic split its Ewald parts cancel. A 2D spherical sum in the
  unit square at eta = 0.1 is up to 2.5e6 off for k from 3.5 to 8.
- The closed forms of its real-space Kambe integrals lose about |a|^2 ulps each.
  This puts 1D spherical sums of degrees 5 and 7 on the axis at k = 0.42 and
  eta = 0.137 3.4e-7 to 7.1e-7 off.
- Sums that vanish by symmetry (odd degrees or orders at a zero Bloch vector, at
  a lattice point or half a lattice vector from one) come back as the rounding of
  their parts, up to 8.5e10 off at splits of 0.14 to 0.3.

treams-rs sums these cases as follows:

- The real-space shell limit grows with 1 / |k eta| and with e^Re(1 / (2 eta^2)),
  the size of the nearest terms, where (k eta)^2 lies at most 45 degrees off the
  real axis. 1D and 2D sums below every automatic split grow it at every angle;
  3D sums, and sums above every automatic split, keep the fixed shell limit at
  splits turned farther.
- No shell sum stops before the moduli of its far terms peak.
- Below every automatic split both Ewald parts continue to 2e-13 of the complete
  sum, and the real-space terms take the series of their Kambe integrals where
  the closed forms would lose more. The degree-5 and degree-7 sums above are
  within 2.2e-13.
- Each sum estimates the rounding error of its cancelling parts. It fails with
  "Ewald split too small" where that error exceeds 1e-3 of the value or one of
  its derivatives.
- Complete sums whose real-space shells settle at the shell limit are returned
  where they match the automatic split.
- Sums that vanish by symmetry are exactly zero at every split.

treams stays more accurate in two cases:

- 1D spherical sums of degree 0 on the axis, where the native sums keep a few
  ulps of their parts: up to 54 times, 8.3e-6 against 1.5e-7.
- treams returns some sums that fail in treams-rs with "did not converge":
    - sums at small |k eta| d (d is the smallest height of the reduced unit cell)
      whose real-space shells do not settle. Above every automatic split treams
      is 5e-4 to 15 off there at degrees 0 and 1, and 3e-15 to 0.4 at higher
      degrees, mostly within 1e-6 from degree 4. The native 3D sums fail within
      about 2.4 s.
    - 2D spherical sums of lossy k at real splits of 0.35 to 0.7, above every
      automatic split, that turn (k eta)^2 more than 45 degrees off the real
      axis. treams is within 1.2e-10 and 6.8e-8 there.
    - 1D spherical sums on the axis whose far real-space terms peak beyond the
      shell limit. treams is 5.7e-7 off there.

The [numerical limits](../validation/numerical-limits.md#lattice-sums) give the
rules in full. Tests:

- Rust, in `properties/lattice/tables.rs`: the `SPLITS`, `ZEROS` and `RECORDED` tables
  of `explicit_splits_agree_with_the_automatic_split_or_fail`,
  `sums_that_vanish_by_symmetry_are_exactly_zero` and
  `recorded_cases_keep_their_properties`; the references of
  `sums_match_high_precision_references`; and the properties
  `small_splits_keep_their_sums_or_fail`,
  `small_splits_keep_the_sums_next_to_a_lattice_point_or_fail`,
  `small_split_jets_keep_the_derivatives_of_vanishing_sums_or_fail` and
  `turned_small_split_jets_keep_their_derivatives_or_fail`.
- Python, in [lattice tests](../../tests/lattice/test_lattice.py):
  `test_small_splits_reach_the_sum`,
  `test_small_splits_fail_only_where_their_parts_cancel` and
  `test_sums_that_vanish_by_symmetry_are_exactly_zero`.

## Companion-paper findings

The cylinder notebook of the electron-beam spectroscopy repository omits
`.changepoltype("parity")` when it pairs its helicity T-matrix with a parity
electron illumination. Its own regression test includes the conversion. The
unmodified notebook can differ by 50% in EELS; with the one-line conversion it
agrees with the stored regression spectra. The defect is in the notebook, not
in treams. The [published-application checks](../validation/published-applications.md)
link the unchanged source and keep the results of both versions.

The thermal-radiation reproduction has strongly unbalanced multipole linear
systems. treams-rs rescales their rows and columns before the LU factorization,
with the same scales in the forward solve and its gradient. The published
absorption tables differ from the reproduced spectrum
by up to 0.577%, and the report keeps that difference. It is not classified as
a treams defect.

## Spheres and cylinders at several positions

`operators.expand((spheres, axes))` adds the contribution of every axis to
every sphere. `Wave.in_basis` and `TMatrix.scatter` call it for cylindrical
waves. `operators.expandlattice(..., basis=(axes, spheres))` likewise adds the
radiation of every chain of spheres to every axis. treams pairs each sphere
only with the axis of the same particle index (`pidx`) and sets every other
entry to zero. treams-rs keeps all pairs, as both packages do for
sphere-to-sphere and cylinder-to-cylinder expansions.

The difference shows when the cylindrical coefficients describe one field more
than once. A wave expanded around each of two axes holds the same field twice,
so its expansion into two spheres counts the field twice. Expand the wave from
its one-axis basis to count it once. To reproduce treams, multiply the operator
by the mask `spheres.pidx[:, None] == axes.pidx`. treams ignores the positions
in these expansions, so the two agree when each sphere lies at the origin of its
own axis:

```python exec
import numpy as np
import treams
import treams_rs as tr

k0, positions = 1.3, [[0, 0, 0], [1.5, 0, 0]]
spheres = tr.SphericalBasis.default(1, 2, positions=positions)
axes = tr.CylindricalBasis.default([0.2], 12, 2, positions=positions)
own_axis = spheres.pidx[:, None] == axes.pidx

# One regular cylindrical wave, expanded around each of the two axes.
single = tr.cylindrical_wave(0.2, 0, 1, k0=k0)
wave = single.in_basis(axes)
expansion = tr.operators.expand((spheres, axes), k0=k0)
once = (expansion * own_axis) @ wave.coefficients
np.testing.assert_allclose(wave.in_basis(spheres).coefficients, 2 * once, atol=1e-13)
np.testing.assert_allclose(single.in_basis(spheres).coefficients, once, atol=1e-13)

up_spheres = treams.SphericalWaveBasis.default(1, 2, positions=positions)
up_axes = treams.CylindricalWaveBasis.default([0.2], 12, 2, positions=positions)
upstream = treams.expand((up_spheres, up_axes), k0=k0, poltype="helicity")
np.testing.assert_allclose(expansion * own_axis, upstream, atol=1e-13)

# Radiation of sphere chains with period 2 along z into the two axes.
radiation = tr.operators.expandlattice(2.0, 0.2, basis=(axes, spheres), k0=k0)
upstream = treams.expandlattice(
    2.0, 0.2, basis=(up_axes, up_spheres), k0=k0, poltype="helicity"
)
np.testing.assert_allclose(radiation * own_axis.T, upstream, atol=1e-13)
```

The axes carry azimuthal orders up to 12, enough for the wave translated from
one axis to the other to converge; with orders up to 1 the doubled result is
off by about 13%.

The periodic conversions follow treams: `PeriodicResponse.to_cylindrical` and
`PeriodicWave.in_basis` with a cylindrical basis pair each axis with its own
particle. A basis with several axes needs one axis per particle; axis i then
collects the waves of particle i only. The
[chain of sphere pairs](../examples/cylindrical_chain.md) uses one axis through
each sphere.

## HDF5 reproducer

Run with treams 0.4.5 and h5py installed. The file stays in memory.

```python no-exec
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

## Plane expansion reproducer

```python no-exec
import numpy as np
import treams

basis = treams.PlaneWaveBasisByComp.default([[0.2, 0.3]])
print(np.asarray(treams.expand(basis, k0=1.3, poltype="helicity")))
# [[1, 1], [1, 1]]; a basis expanded into itself must give the identity.
```

# Upstream findings

Findings from comparison with treams 0.4.5 and the reference source pinned in
architecture.md. These are reproduced behavior, not claims that all current
upstream branches remain affected. No upstream issues or maintainer messages
have been sent from this project.

## Behavioral defects

| Finding | Evidence and impact | Regression |
| --- | --- | --- |
| EBCM radial area factor | Upstream `ebcm.qmat` uses `sin(theta) * [r, -dr, 0]`; the surface element requires another factor of r. For r=0.3(1+0.23 cos²(theta)), lossless eps=3.1, kappa=0.07 and k0=1.3, max abs(SᴴS-I) stays about 0.00135 at degrees 2, 4 and 6. Restoring r gives 6.39e-6, 2.02e-7 and 1.22e-8. With identical vacuum inside/outside, degree 2 produces spurious max abs(T)=5.88e-4; the corrected integral gives 8.41e-18. | [EBCM tests](../tests/test_ebcm.py) cover sphere/Mie agreement, lossless convergence, Hypothesis zero-contrast shapes and both adjoints. The default includes r; explicit `legacy=True` reproduces upstream. |
| Complex plane-wave branch | For k=(0.2+i, 0.1+0.3i, 1.3-0.8i), an extra principal square root in the angular coefficients can disagree with the Cartesian polarization branch. Reconstructed fields differ by about 1.25 on the recorded samples; the corrected expansion is accurate to about 1e-13. | `test_complex_transverse_branch_reconstruction_and_gradient` in [plane-expansion tests](../tests/test_plane_expansion.py). |
| Missing skew-cell diffraction orders | For lattice rows (2,-3), (0,1.7), zero Bloch vector and cutoff 4, upstream omits integer reciprocal orders (-2,1) and (2,-1). Both lie inside the cutoff. | `test_diffraction_basis_skew_completeness` in [channel tests](../tests/test_channels.py), checked against independent integer enumeration. |

## Numerical accuracy limitations

| Finding | Evidence and impact | Regression |
| --- | --- | --- |
| Near-axis spherical translation | Forming the angular coordinate through cos(theta) can round small transverse offsets to the axis, losing nonzero terms and derivatives. Cartesian evaluation is checked against a resolved-angle extrapolation down to offsets of 1e-300. | `test_near_axis_translation_against_resolved_angle_limit` in [invariant tests](../tests/test_invariants.py). |
| Cylindrical Ewald automatic split | The nine-cylinder benchmark's automatic split differs from a converged result by about 8.4e-8 in the final S matrix. Both benchmark implementations use eta=0.7, retaining the original comparison tolerance. | Large-cell split-independence check in [lattice tests](../tests/test_lattice.py); details in [benchmarks](benchmarks.md#complete-cylindrical-arrays). |

The G/F field normalization also differs between spherical and cylindrical/plane
families and between helicity and parity. This is documented as a preserved
convention, not classified here as a confirmed physical defect. Likewise, normal-
incidence polarization gauges, diffraction thresholds and eigenvalue degeneracies
are mathematical restrictions, not automatically upstream bugs.

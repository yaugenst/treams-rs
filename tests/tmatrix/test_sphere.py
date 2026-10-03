"""Single spheres through the upstream-compatible API against treams.

Checks TMatrix.sphere with its cross sections and polarization changes, the chiral
parity coupling, and the complete diff.sphere pullback; test_physics_api checks the
same optical theorem through the typed factories.
"""

import numpy as np
import pytest
import treams
from hypothesis import given
from hypothesis import strategies as st

import treams_rs as tr
from treams_rs import diff
from treams_rs.testing import check_pullback


@pytest.mark.physics
@pytest.mark.workflows
@pytest.mark.reference
@given(
    degree=st.integers(1, 4),
    radius=st.floats(0.1, 1.5),
    epsilon=st.floats(1.1, 8),
    parity=st.booleans(),
)
def test_sphere_api_and_optical_theorem(degree, radius, epsilon, parity):
    poltype = "parity" if parity else "helicity"
    actual = tr.TMatrix.sphere(degree, 1.3, radius, [tr.Material(epsilon), 1], poltype)
    expected = treams.TMatrix.sphere(degree, 1.3, radius, [epsilon, 1], poltype)
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=2e-11)
    np.testing.assert_allclose(
        actual.xs_ext_avg, expected.xs_ext_avg, atol=1e-12, rtol=3e-11
    )
    np.testing.assert_allclose(
        actual.xs_sca_avg, expected.xs_sca_avg, atol=1e-12, rtol=3e-11
    )
    np.testing.assert_allclose(
        actual.xs_ext_avg, actual.xs_sca_avg, atol=1e-12, rtol=3e-11
    )
    np.testing.assert_array_equal(actual.basis.lms, expected.basis.lms)
    np.testing.assert_allclose(
        actual.changepoltype().changepoltype(), actual, atol=2e-14
    )


@pytest.mark.workflows
@pytest.mark.reference
@given(radius=st.floats(0.1, 0.4), kappa=st.floats(0.03, 0.2))
def test_chiral_sphere_parity_retains_magnetoelectric_coupling(radius, kappa):
    actual = tr.TMatrix.sphere(2, 1.3, radius, [(3.1, 1, kappa), 1], "parity")
    # Upstream #27: direct parity construction erases the off-diagonal EM terms.
    expected = treams.TMatrix.sphere(
        2, 1.3, radius, [(3.1, 1, kappa), 1], "helicity"
    ).changepoltype("parity")
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=3e-13)
    assert np.max(abs(actual.array[0::2, 1::2])) > 1e-7
    np.testing.assert_allclose(
        actual.xs_ext_avg, actual.xs_sca_avg, rtol=3e-11, atol=1e-12
    )


@pytest.mark.gradients
def test_complete_sphere_directional_derivative():
    # Frequency, radii and every layer's material, with the exterior included.
    check_pullback(
        lambda k0, radii, epsilon, mu, kappa: diff.sphere(
            1, float(k0), radii, epsilon, mu, kappa
        ),
        np.asarray(1.2),
        np.array([0.18, 0.3]),
        np.array([3.2 + 0.1j, 2.1 + 0.05j, 1.0 + 0j]),
        np.array([1.1 + 0.02j, 1.2 + 0.01j, 1.0 + 0j]),
        np.array([0.1 + 0.01j, 0.05 + 0.02j, 0.0 + 0j]),
        step=2e-6,
        rtol=8e-6,
        atol=3e-8,
    )

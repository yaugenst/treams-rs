"""Vector spherical waves checked against treams and Maxwell identities."""

import numpy as np
import pytest
import treams.special as sc
from hypothesis import given
from hypothesis import strategies as st

from treams_rs import _native


@pytest.mark.oracle_numerical
@pytest.mark.filterwarnings(
    "ignore:`scipy.special.sph_harm` is deprecated.*:DeprecationWarning"
)
@pytest.mark.parametrize(
    "position",
    [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0.2, -0.1, 0.3), (1e-9, -1e-9, 1e-9)],
)
@pytest.mark.parametrize("outgoing", [False, True])
@pytest.mark.parametrize("helicity", [False, True])
def test_vector_waves_reference(position, outgoing, helicity):
    if outgoing and np.linalg.norm(position) == 0:
        with pytest.raises(ValueError, match="singular"):
            _native.spherical_wave((1, 0, 1), 1.2 + 0.1j, position, helicity, outgoing)
        return
    spherical = sc.car2sph(position)
    for degree in range(1, 5):
        for order in range(-degree, degree + 1):
            for pol in (0, 1):
                args = (degree, order, (1.2 + 0.1j) * spherical[0], *spherical[1:])
                if helicity:
                    function = sc.vsw_A if outgoing else sc.vsw_rA
                    expected = function(*args, pol)
                else:
                    function = {
                        (False, 0): sc.vsw_rM,
                        (False, 1): sc.vsw_rN,
                        (True, 0): sc.vsw_M,
                        (True, 1): sc.vsw_N,
                    }[outgoing, pol]
                    expected = function(*args)
                expected = sc.vsph2car(expected, spherical)
                actual = _native.spherical_wave(
                    (degree, order, pol), 1.2 + 0.1j, position, helicity, outgoing
                )[0]
                # Cartesian components can cancel near the outgoing singularity.
                # Judge error relative to the whole vector, including zero components.
                error = np.linalg.norm(np.asarray(actual) - expected)
                assert error <= 3e-13 + 2e-11 * np.linalg.norm(expected)


@pytest.mark.physics
@given(
    degree=st.integers(1, 6),
    selector=st.integers(0, 20),
    x=st.floats(-1, 1),
    y=st.floats(-1, 1),
    z=st.floats(0.5, 2),
    pol=st.integers(0, 1),
    outgoing=st.booleans(),
)
def test_helicity_eigenfield_and_zero_divergence(
    degree, selector, x, y, z, pol, outgoing
):
    k = 1.2 + 0.1j
    order = selector % (2 * degree + 1) - degree
    value, derivative, _ = _native.spherical_wave(
        (degree, order, pol), k, (x, y, z), True, outgoing
    )
    derivative = np.array(derivative)
    curl = np.array(
        [
            derivative[2, 1] - derivative[1, 2],
            derivative[0, 2] - derivative[2, 0],
            derivative[1, 0] - derivative[0, 1],
        ]
    )
    scale = max(1, np.linalg.norm(value))
    np.testing.assert_allclose(
        curl, (2 * pol - 1) * k * np.array(value), rtol=2e-10, atol=2e-10 * scale
    )
    np.testing.assert_allclose(np.trace(derivative), 0, atol=2e-10 * scale)


@pytest.mark.ad_contract
@pytest.mark.parametrize("position", [(0, 0, 0), (0, 0, 1), (0.2, -0.3, 0.4)])
@pytest.mark.parametrize("mode", [(1, 0, 0), (1, -1, 1), (2, 1, 1), (3, -2, 0)])
def test_field_position_and_wavenumber_derivatives(position, mode):
    k = 1.2 + 0.1j
    _, jacobian, dk = _native.spherical_wave(mode, k, position, True, False)
    step = 1e-6
    for axis in range(3):
        direction = np.eye(3)[axis]
        plus = _native.spherical_wave(
            mode, k, tuple(np.asarray(position) + step * direction), True, False
        )[0]
        minus = _native.spherical_wave(
            mode, k, tuple(np.asarray(position) - step * direction), True, False
        )[0]
        numerical = (np.array(plus) - minus) / (2 * step)
        np.testing.assert_allclose(
            np.array(jacobian)[:, axis], numerical, atol=2e-10, rtol=3e-8
        )
    direction = 0.3 - 0.2j
    plus = _native.spherical_wave(mode, k + step * direction, position, True, False)[0]
    minus = _native.spherical_wave(mode, k - step * direction, position, True, False)[0]
    np.testing.assert_allclose(
        np.array(dk) * direction,
        (np.array(plus) - minus) / (2 * step),
        atol=2e-10,
        rtol=3e-8,
    )

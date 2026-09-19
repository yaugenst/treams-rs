"""Upstream numerical oracles and independent checks of native pullbacks."""

import numpy as np
import pytest
import treams.coeffs as reference
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import spherical_jn, spherical_yn

from treams_rs import _native, coeffs


@pytest.mark.oracle_numerical
@pytest.mark.parametrize("degree", [1, 3, 10])
@pytest.mark.parametrize("loss", [0.0, 0.2])
def test_multilayer_chiral_mie(degree, loss):
    args = ([1, 2, 4], [1, 1.3 + loss * 1j, 3, 2], [1, 2, 1, 1.5], [0.1, 0, 0.2, 0.1])
    np.testing.assert_allclose(
        coeffs.mie(degree, *args), reference.mie(degree, *args), atol=4e-13, rtol=3e-12
    )


@pytest.mark.physics
def test_lossless_sphere_optical_theorem():
    matrix = coeffs.mie(3, [2.0], [4, 1], [1, 1], [0, 0])
    np.testing.assert_allclose(
        -np.trace(matrix).real, np.sum(abs(matrix) ** 2), atol=1e-13
    )


@pytest.mark.oracle_numerical
@pytest.mark.parametrize("degree", [0, 1, 3, 10, 30])
@pytest.mark.parametrize("z", [0.0, 0.01 + 0.02j, 1.2, 8.0 + 0.5j])
def test_spherical_bessel(degree, z):
    f, df, _ = _native.radial(degree, z, False)
    np.testing.assert_allclose(f, spherical_jn(degree, z), atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(
        df, spherical_jn(degree, z, derivative=True), atol=1e-14, rtol=1e-12
    )
    if z != 0:
        h, dh, _ = _native.radial(degree, z, True)
        np.testing.assert_allclose(
            h,
            spherical_jn(degree, z) + 1j * spherical_yn(degree, z),
            atol=1e-13,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            dh,
            spherical_jn(degree, z, True) + 1j * spherical_yn(degree, z, True),
            atol=1e-13,
            rtol=1e-12,
        )


@pytest.mark.ad_contract
@pytest.mark.parametrize("parameter", range(4))
def test_mie_pullback(parameter):
    args = [
        np.array([0.7, 1.4]),
        np.array([2 + 0.2j, 1.4 + 0.1j, 1]),
        np.array([1.1, 1.3, 1], dtype=complex),
        np.array([0.12, 0.05, 0], dtype=complex),
    ]
    rng = np.random.default_rng(12)
    cotangent = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    _, context = coeffs.mie_with_context(2, *args)
    gradient = context.pullback(cotangent)[parameter]
    direction = rng.normal(size=args[parameter].shape)
    if parameter != 0:
        direction = direction + 1j * rng.normal(size=direction.shape)
    step = 1e-6
    plus, minus = list(args), list(args)
    plus[parameter] = args[parameter] + step * direction
    minus[parameter] = args[parameter] - step * direction
    finite_difference = np.vdot(
        cotangent, coeffs.mie(2, *plus) - coeffs.mie(2, *minus)
    ).real / (2 * step)
    np.testing.assert_allclose(
        np.vdot(gradient, direction).real, finite_difference, atol=2e-8, rtol=2e-7
    )
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(cotangent)


@pytest.mark.python_contract
def test_invalid_shapes_and_radii():
    with pytest.raises(ValueError, match="strictly increasing"):
        coeffs.mie(1, [2, 1], [1, 2, 3], [1, 1, 1], [0, 0, 0])
    with pytest.raises(ValueError, match="equal lengths"):
        coeffs.mie(1, [1], [1, 2], [1], [0, 0])


@pytest.mark.oracle_numerical
@settings(max_examples=32, deadline=None)
@given(
    size=st.floats(60, 100),
    degree=st.integers(1, 100),
    epsilon=st.floats(-9, -7),
    loss=st.floats(0.1, 1),
    scale=st.floats(0.5, 2),
)
def test_absorbing_mie_reference_and_scale_invariance(
    size, degree, epsilon, loss, scale
):
    epsilon = np.array([epsilon + 1j * loss, 1])
    matrix, context = coeffs.mie_with_context(degree, [size], epsilon, [1, 1], [0, 0])
    np.testing.assert_allclose(
        matrix,
        reference.mie(degree, [size], epsilon, [1, 1], [0, 0]),
        atol=2e-13,
        rtol=2e-11,
    )
    # Rescaling size and all refractive indices inversely leaves both radial
    # arguments and interface impedance ratios unchanged.
    np.testing.assert_allclose(
        matrix,
        coeffs.mie(degree, [size * scale], epsilon / scale**2, [1, 1], [0, 0]),
        atol=2e-13,
        rtol=2e-11,
    )
    cotangent = np.array([[0.3 + 0.7j, -0.2j], [0.1 - 0.4j, -0.5 + 0.2j]])
    gx, ge, _, _ = context.pullback(cotangent)
    np.testing.assert_allclose(
        gx[0] * size - 2 * np.vdot(ge, epsilon).real, 0, atol=2e-10
    )


@pytest.mark.ad_contract
@pytest.mark.parametrize("degree", [1, 3, 80, 99])
@pytest.mark.parametrize("parameter", range(4))
def test_absorbing_mie_pullback(degree, parameter):
    args = [
        np.array([80.0]),
        np.array([-8 + 0.4j, 1]),
        np.ones(2, dtype=complex),
        np.zeros(2, dtype=complex),
    ]
    cotangent = np.array([[0.3 + 0.7j, -0.2j], [0.1 - 0.4j, -0.5 + 0.2j]])
    _, context = coeffs.mie_with_context(degree, *args)
    gradient = context.pullback(cotangent)[parameter]
    direction = (
        np.array([0.3]) if parameter == 0 else np.array([0.3 + 0.1j, -0.2 + 0.4j])
    )
    step = 1e-6
    plus, minus = list(args), list(args)
    plus[parameter] = args[parameter] + step * direction
    minus[parameter] = args[parameter] - step * direction
    finite_difference = np.vdot(
        cotangent, coeffs.mie(degree, *plus) - coeffs.mie(degree, *minus)
    ).real / (2 * step)
    np.testing.assert_allclose(
        np.vdot(gradient, direction).real, finite_difference, atol=2e-8, rtol=2e-7
    )

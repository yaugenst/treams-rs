"""Upstream numerical oracles and independent checks of native pullbacks."""

import numpy as np
import pytest
import treams.coeffs as upstream_coeffs
from hypothesis import given, settings
from hypothesis import strategies as st

from treams_rs import coeffs, diff
from treams_rs.testing import check_pullback

from _support import complex_normal


@pytest.mark.reference
@pytest.mark.parametrize("degree", [1, 3, 10])
@pytest.mark.parametrize("loss", [0.0, 0.2])
def test_multilayer_chiral_mie(degree, loss):
    args = ([1, 2, 4], [1, 1.3 + loss * 1j, 3, 2], [1, 2, 1, 1.5], [0.1, 0, 0.2, 0.1])
    np.testing.assert_allclose(
        coeffs.mie(degree, *args),
        upstream_coeffs.mie(degree, *args),
        atol=4e-13,
        rtol=3e-12,
    )


@pytest.mark.gradients
def test_mie_pullback():
    args = [
        np.array([0.7, 1.4]),
        np.array([2 + 0.2j, 1.4 + 0.1j, 1]),
        np.array([1.1, 1.3, 1], dtype=complex),
        np.array([0.12, 0.05, 0], dtype=complex),
    ]
    cotangent = complex_normal(np.random.default_rng(12), (2, 2))

    def record(*layers):
        return diff.mie(2, *layers)

    check_pullback(
        record,
        *args,
        cotangents=cotangent,
        step=1e-6,
        rtol=2e-7,
        atol=2e-8,
        seed=12,
    )


@pytest.mark.interface
def test_invalid_shapes_and_radii():
    with pytest.raises(ValueError, match="strictly increasing"):
        coeffs.mie(1, [2, 1], [1, 2, 3], [1, 1, 1], [0, 0, 0])
    with pytest.raises(ValueError, match="equal lengths"):
        coeffs.mie(1, [1], [1, 2], [1], [0, 0])
    # treams broadcasts over degrees and orders; treams-rs takes one at a time.
    with pytest.raises(ValueError, match="one integer degree; loop over degrees"):
        coeffs.mie([1, 2], [1], [2, 1], [1, 1], [0, 0])
    with pytest.raises(ValueError, match="one integer order; loop over orders"):
        coeffs.mie_cyl(0.2, [0, 1], 1.1, [1], [2, 1], [1, 1], [0, 0])


@pytest.mark.physics
@pytest.mark.gradients
@pytest.mark.reference
@settings(max_examples=32)
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
    matrix, context = diff.mie(degree, [size], epsilon, [1, 1], [0, 0])
    np.testing.assert_allclose(
        matrix,
        upstream_coeffs.mie(degree, [size], epsilon, [1, 1], [0, 0]),
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


@pytest.mark.gradients
@pytest.mark.parametrize("degree", [1, 3, 80, 99])
def test_absorbing_mie_pullback(degree):
    # Hand-picked directions: some random ones exceed the tolerance at degree 80.
    layer = np.array([0.3 + 0.1j, -0.2 + 0.4j])
    check_pullback(
        lambda *layers: diff.mie(degree, *layers),
        np.array([80.0]),
        np.array([-8 + 0.4j, 1]),
        np.ones(2, dtype=complex),
        np.zeros(2, dtype=complex),
        directions=(np.array([0.3]), layer, layer, layer),
        cotangents=np.array([[0.3 + 0.7j, -0.2j], [0.1 - 0.4j, -0.5 + 0.2j]]),
        step=1e-6,
        rtol=2e-7,
        atol=2e-8,
    )

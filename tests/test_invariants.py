"""Shrinking property tests against physical invariants and the upstream oracle."""

from itertools import pairwise

import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st

from treams_rs import _native, coeffs, diff

settings.register_profile("native", max_examples=30, deadline=None)
settings.load_profile("native")
positive = st.floats(0.2, 3.0, allow_nan=False, allow_infinity=False)


@pytest.mark.physics
@given(size=positive, epsilon=st.floats(1.0, 8.0), degree=st.integers(1, 8))
def test_lossless_optical_theorem(size, epsilon, degree):
    value = coeffs.mie(degree, [size], [epsilon, 1], [1, 1], [0, 0])
    np.testing.assert_allclose(
        -np.trace(value).real, np.sum(abs(value) ** 2), atol=1e-12, rtol=1e-11
    )


@pytest.mark.physics
@given(size=positive, epsilon=positive, degree=st.integers(1, 8))
def test_zero_contrast(size, epsilon, degree):
    value = coeffs.mie(degree, [size], [epsilon, epsilon], [1, 1], [0, 0])
    np.testing.assert_allclose(value, 0, atol=2e-13)


@pytest.mark.physics
@given(
    size=positive,
    epsilon=positive,
    fraction=st.floats(0.2, 0.8),
    degree=st.integers(1, 6),
)
def test_identical_layer_split(size, epsilon, fraction, degree):
    plain = coeffs.mie(degree, [size], [epsilon, 1], [1, 1], [0.03, 0])
    split = coeffs.mie(
        degree,
        [fraction * size, size],
        [epsilon, epsilon, 1],
        [1, 1, 1],
        [0.03, 0.03, 0],
    )
    np.testing.assert_allclose(split, plain, atol=3e-12, rtol=3e-11)


@pytest.mark.oracle_numerical
@given(
    degree=st.integers(1, 4),
    order=st.integers(-1, 1),
    # Upstream angular functions lose small transverse components through cos(theta).
    # The independent resolved-angle limit below covers the near-axis regime.
    x=st.one_of(st.just(0.0), st.floats(-1.0, -0.02), st.floats(0.02, 1.0)),
    z=positive,
    outgoing=st.booleans(),
)
def test_translation_reference(degree, order, x, z, outgoing):
    displacement = np.array([x, 0.0, z])
    spherical = treams.special.car2sph(displacement)
    to, source = (degree, order, 1), (2, -1, 1)
    actual = _native.translation(
        to, source, 1.2 + 0.1j, tuple(displacement), True, outgoing
    )[0]
    expected = treams.sw.translate(
        *to,
        *source,
        (1.2 + 0.1j) * spherical[0],
        spherical[1],
        spherical[2],
        singular=outgoing,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-11)


@pytest.mark.ad_contract
@pytest.mark.parametrize(
    "position", [(0.0, 0.0, 1.0), (0.2, -0.5, 1.0), (0.0, 0.0, -1.0)]
)
def test_translation_position_pullback_on_and_off_axis(position):
    to, source = (3, -1, 1), (2, 1, 1)
    _, derivative, _ = _native.translation(to, source, 1.2 + 0.1j, position, True, True)
    for axis in range(3):
        plus, minus = list(position), list(position)
        plus[axis] += 1e-5
        minus[axis] -= 1e-5
        finite_difference = (
            _native.translation(to, source, 1.2 + 0.1j, tuple(plus), True, True)[0]
            - _native.translation(to, source, 1.2 + 0.1j, tuple(minus), True, True)[0]
        ) / 2e-5
        np.testing.assert_allclose(
            derivative[axis], finite_difference, atol=2e-6, rtol=2e-7
        )


@pytest.mark.public_e2e
@pytest.mark.oracle_numerical
def test_cluster_native_against_upstream():
    radii = [0.2, 0.3]
    epsilon = [3 + 0.1j, 2 + 0.2j]
    positions = [[0.0, 0.0, 0.0], [0.4, -0.2, 1.4]]
    actual, _ = diff.cluster(2, 1.1, radii, epsilon, positions)
    spheres = [
        treams.TMatrix.sphere(2, 1.1, radius, [eps, 1])
        for radius, eps in zip(radii, epsilon, strict=True)
    ]
    expected = treams.TMatrix.cluster(spheres, positions).interaction.solve()
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=2e-10)


@pytest.mark.ad_contract
@given(offset=st.tuples(st.floats(-5, 5), st.floats(-5, 5), st.floats(-5, 5)))
def test_cluster_global_translation_invariance(offset):
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    args = (1, 1.2, [0.2, 0.3], [2.0, 3.0])
    base, context = diff.cluster(*args, positions)
    shifted, _ = diff.cluster(*args, positions + offset)
    np.testing.assert_allclose(shifted, base, atol=3e-13, rtol=3e-12)
    gradient = context.pullback(np.ones_like(base))[1]
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=2e-13)


@pytest.mark.ad_contract
@pytest.mark.parametrize("parameter", ["radii", "positions", "epsilon", "k0"])
def test_complete_cluster_directional_derivative(parameter):
    args = {
        "lmax": 2,
        "k0": 1.1,
        "radii": np.array([0.2, 0.3]),
        "epsilon": np.array([3 + 0.1j, 2 + 0.2j]),
        "positions": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
    }
    value, context = diff.cluster(**args)
    rng = np.random.default_rng(7)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradients = dict(
        zip(["radii", "positions", "epsilon", "k0"], context.pullback(g), strict=True)
    )
    direction = rng.normal(size=np.shape(args[parameter]))
    if parameter == "epsilon":
        direction = direction + 1j * rng.normal(size=direction.shape)
    plus, minus = dict(args), dict(args)
    step = 2e-6
    plus[parameter] = args[parameter] + step * direction
    minus[parameter] = args[parameter] - step * direction
    finite_difference = np.vdot(
        g, diff.cluster(**plus)[0] - diff.cluster(**minus)[0]
    ).real / (2 * step)
    np.testing.assert_allclose(
        np.vdot(gradients[parameter], direction).real,
        finite_difference,
        atol=2e-8,
        rtol=2e-6,
    )


@pytest.mark.parametrize("outgoing", [False, True])
@pytest.mark.parametrize("z", [-0.5, 0.5])
@given(exponent=st.integers(8, 300), y_fraction=st.floats(-1, 1))
def test_near_axis_translation_against_resolved_angle_limit(
    outgoing, z, exponent, y_fraction
):
    # For this azimuthal difference the translation is (x-i*y)*slope + O(rho**3).
    # Extrapolate the slope from four resolvable upstream angles, where its
    # associated Legendre evaluation has not rounded to the axis.
    k = 1.2 + 0.1j
    estimates = []
    for h in (0.01, 0.005, 0.0025, 0.00125):
        spherical = treams.special.car2sph([h, 0, z])
        estimates.append(
            treams.sw.translate(
                4,
                0,
                1,
                2,
                -1,
                1,
                k * spherical[0],
                spherical[1],
                spherical[2],
                singular=outgoing,
            )
            / h
        )
    for order in range(1, 4):
        estimates = [
            (4**order * b - a) / (4**order - 1) for a, b in pairwise(estimates)
        ]
    slope = estimates[0]
    scale = 10.0 ** (-exponent)
    value, gradient, _ = _native.translation(
        (4, 0, 1), (2, -1, 1), k, (scale, scale * y_fraction, z), True, outgoing
    )
    np.testing.assert_allclose(
        value / scale, slope * (1 - 1j * y_fraction), rtol=2e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        gradient[:2], [slope, -1j * slope], rtol=2e-9, atol=1e-12
    )

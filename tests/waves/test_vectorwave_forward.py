"""Vector-wave JVPs through harmonics, spherical, cylindrical and plane waves."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from treams_rs import diff
from treams_rs.testing import check_pushforward

pytestmark = pytest.mark.gradients


@pytest.mark.parametrize(
    ("function", "values"),
    [
        ("sph_harm", (0.8, 0.3)),
        ("vsh_X", (0.8, 0.3)),
        ("vsh_Y", (0.8, 0.3)),
        ("vsh_Z", (0.8, 0.3)),
        ("vsw_rM", (0.7 + 0.1j, 0.8, 0.3)),
        ("vsw_N", (0.7 + 0.1j, 0.8, 0.3)),
        ("vsw_rA", (0.7 + 0.1j, 0.8, 0.3)),
        ("vcw_rM", (0.2, 0.7 + 0.1j, 0.3, 0.4)),
        ("vcw_N", (0.2, 0.7 + 0.1j, 0.3, 0.4, 1.3)),
        ("vcw_rA", (0.2, 0.7 + 0.1j, 0.3, 0.4, 1.3)),
        ("vpw_M", (0.2, 0.3, 1.0 + 0.1j, 0.1, 0.2, 0.3)),
        ("vpw_N", (0.2, 0.3, 1.0 + 0.1j, 0.1, 0.2, 0.3)),
        ("vpw_A", (0.2, 0.3, 1.0 + 0.1j, 0.1, 0.2, 0.3)),
    ],
)
def test_vectorwave_pushforward_all_families(function, values):
    check_pushforward(
        lambda *a: diff.vector_wave(*a, function=function, degree=3, order=1, pol=1),
        *(np.asarray(value, dtype=complex) for value in values),
    )


def test_vectorwave_pushforward_broadcasts_arguments_before_cartesian_axis():
    check_pushforward(
        lambda *a: diff.vector_wave(
            *a, function="vsw_rA", degree=np.array([2, 3])[:, None], order=1, pol=1
        ),
        np.array([0.4 + 0.1j, 0.7 - 0.1j, 1.2 + 0.2j])[None, :],
        np.array([[0.7], [1.0]]),
        np.asarray(0.3),
    )


@pytest.mark.parametrize("position_only", [False, True])
def test_plane_wave_shared_polarization_pushforward(position_only):
    values = (
        np.asarray(0.2 + 0.03j),
        np.asarray(0.3 - 0.01j),
        np.asarray(1.0 + 0.05j),
        np.linspace(-0.2, 0.4, 5),
        np.asarray(0.2),
        np.asarray(0.3),
    )
    directions = tuple(np.full_like(value, 0.1) for value in values)
    if position_only:
        directions = (*(np.zeros_like(value) for value in values[:3]), *directions[3:])
    check_pushforward(
        lambda *a: diff.vector_wave(*a, function="vpw_A", pol=1),
        *values,
        directions=directions,
    )


def test_wave_pushforward_validates_arguments_and_reuses_context():
    value, context = diff.sph_harm(np.array([0.4, 0.8]), 0.3, degree=2, order=1)
    with pytest.raises(ValueError, match="tangent"):
        context.pushforward(np.ones(2))
    with pytest.raises(ValueError, match="tangent"):
        context.pushforward(np.ones(3), 0.1)
    tangent = context.pushforward(np.ones(2), 0.1)
    gradients = context.pullback(np.ones_like(value))
    assert_allclose(context.pushforward(-np.ones(2), -0.1), -tangent)
    for actual, expected in zip(
        context.pullback(-np.ones_like(value)), gradients, strict=True
    ):
        assert_allclose(actual, -expected)
    assert_allclose(context.pushforward(np.zeros(2), 0.0), 0.0)

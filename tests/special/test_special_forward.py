"""Directional derivatives of broadcast special-function and wave records."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from treams_rs import diff, special
from treams_rs.testing import check_pushforward

pytestmark = pytest.mark.gradients


@pytest.mark.parametrize("function", ["j", "y", "h1", "h2"])
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("derivative", [False, True])
def test_bessel_pushforward_broadcast(function, spherical, derivative):
    orders = np.arange(3)[:, None]
    z = np.array([0.7 + 0.2j, 1.2 + 0.1j])[None, :]
    check_pushforward(
        lambda z: diff.bessel(
            orders, z, function=function, spherical=spherical, derivative=derivative
        ),
        z,
    )


@pytest.mark.parametrize("function", ["legendre", "pi", "tau"])
def test_angular_pushforward_broadcast(function):
    check_pushforward(
        lambda z: diff.angular(np.arange(2, 5)[:, None], 1, z, function=function),
        np.array([0.2 + 0.1j, 0.4 - 0.2j])[None, :],
    )


def test_integral_and_wigner_pushforwards():
    check_pushforward(
        lambda z: diff.incgamma(np.array([0.5, 1.0, 1.5])[:, None], z),
        np.array([0.7 + 0.2j, 1.2 + 0.1j])[None, :],
    )
    check_pushforward(
        lambda z, eta: diff.intkambe(np.array([-3, -2, 0])[:, None], z, eta),
        np.array([0.7 + 0.2j, 1.2 + 0.1j])[None, :],
        np.asarray(0.9 + 0.1j),
    )
    check_pushforward(
        lambda *angles: diff.wignerd(np.array([2, 3])[:, None], 1, -1, *angles),
        np.array([0.2 + 0.1j, 0.4 - 0.1j])[None, :],
        np.asarray(0.6 + 0.1j),
        np.array([[0.1], [0.3]]),
    )


@pytest.mark.parametrize(
    ("record", "args"),
    [
        (lambda z: diff.bessel(2, z), (0.8 + 0.2j,)),
        (lambda z: diff.angular(3, 1, z), (0.3 + 0.1j,)),
        (lambda z: diff.incgamma(0.5, z), (0.8 + 0.2j,)),
        (lambda z, eta: diff.intkambe(-3, z, eta), (0.8 + 0.2j, 0.9 + 0.1j)),
        (lambda *a: diff.wignerd(3, 1, -1, *a), (0.2, 0.6, 0.3)),
    ],
)
def test_scalar_records_keep_forward_mode(record, args):
    check_pushforward(lambda *a: record(*(np.asarray(x).item() for x in a)), *args)


def test_pushforward_validates_tangents_and_reuses_context_with_strided_inputs():
    z = np.array([[0.7 + 0.2j, 1.2 + 0.1j]])
    orders = np.arange(3)[:, None]
    _, context = diff.bessel(orders, z, spherical=True)
    with pytest.raises(ValueError, match="tangent"):
        context.pushforward(np.ones((3, 2)))
    with pytest.raises(ValueError, match="tangent"):
        context.pushforward(np.full_like(z, np.nan))
    tangent = np.array([[0.3 + 0.2j, 0.1 - 0.2j]])[:, ::-1]
    actual = context.pushforward(tangent)
    assert_allclose(actual, special.spherical_jn_d(orders, z) * tangent)
    cotangent = np.ones_like(actual)
    gradient = context.pullback(cotangent)
    expected_gradient = diff.bessel(orders, z, spherical=True)[1].pullback(cotangent)
    assert_allclose(gradient, expected_gradient)
    assert_allclose(context.pushforward(-tangent), -actual)
    assert_allclose(context.pullback(-cotangent), -gradient)
    assert_allclose(context.pushforward(tangent), actual)


def test_pushforward_accepts_empty_broadcasts_and_zero_singular_directions():
    _, context = diff.bessel(np.empty((0, 3)), np.asarray(0.8))
    assert context.pushforward(np.asarray(0.2)).shape == (0, 3)
    _, context = diff.angular(3, 1, 1.0)
    assert context.pushforward(0.0) == 0.0
    # The z derivative at the cusp is undefined, but eta still has a derivative.
    eta, tangent = 0.9, 0.3
    _, context = diff.intkambe(-2, 0.0, eta)
    assert_allclose(
        context.pushforward(0.0, tangent),
        -(eta**-2) * np.exp(0.5 / eta**2) * tangent,
    )

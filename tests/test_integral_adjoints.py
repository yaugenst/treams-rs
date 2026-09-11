"""Integral adjoints: reference values, owned broadcast tapes and analytic identities."""

import advect as ad
import numpy as np
import pytest
import treams.special as reference
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import advect, diff, special


@pytest.mark.parametrize("n", [-8, -2.5, 0, 0.5, 1, 2, 8])
def test_gamma_owned_complex_argument(n):
    z = np.array([0.6 + 0.2j, 1.4 - 0.1j, 2.7 + 0.3j])[::-1]
    original = z.copy()
    value, context = diff.incgamma(n, z)
    assert_allclose(value, reference.incgamma(n, z), rtol=3e-12, atol=2e-14)
    z[:] = 9
    g = np.array([0.3 + 0.2j, -0.2j, 0.4])
    with pytest.raises(ValueError, match="shape"):
        context.pullback(np.ones((3, 1), complex))
    gradient = context.pullback(g)
    assert_allclose(gradient, -g * (original ** (n - 1) * np.exp(-original)).conj())
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)
    direction = np.array([0.1 + 0.4j, 0.3j, -0.2])
    h = 1e-5
    fd = (
        special.incgamma(n, original + h * direction)
        - special.incgamma(n, original - h * direction)
    ) / (2 * h)
    assert_allclose(
        np.vdot(gradient, direction).real, np.vdot(g, fd).real, rtol=2e-7, atol=1e-10
    )


@pytest.mark.parametrize("n", [-8, -3, -2, -1, 0, 1, 5])
def test_kambe_owned_broadcast_arguments(n):
    z = np.array([[0.8 + 0.1j], [1.3 - 0.05j]])
    eta = np.array([[0.9 + 0.03j, 1.1 - 0.1j, 1.4 + 0.04j]])
    original = (z.copy(), eta.copy())
    value, context = diff.intkambe(n, z, eta)
    assert_allclose(value, reference.intkambe(n, z, eta), rtol=3e-10, atol=1e-12)
    z[:] = 7
    eta[:] = 6
    g = np.arange(6).reshape(2, 3) * (0.1 + 0.04j)
    gradients = context.pullback(g)
    for j, gradient in enumerate(gradients):
        direction = np.full(original[j].shape, 0.2 + 0.3j)
        plus, minus = list(original), list(original)
        h = 1e-5
        plus[j] = original[j] + h * direction
        minus[j] = original[j] - h * direction
        fd = (special.intkambe(n, *plus) - special.intkambe(n, *minus)) / (2 * h)
        assert_allclose(gradient.shape, original[j].shape)
        assert_allclose(
            np.vdot(gradient, direction).real, np.vdot(g, fd).real, rtol=4e-6, atol=1e-8
        )
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


@given(n=st.integers(-6, 12), x=st.floats(0.8, 2.5), y=st.floats(-0.3, 0.3))
@settings(max_examples=30, deadline=None)
def test_gamma_recurrence_has_zero_value_and_gradient(n, x, y):
    n = n / 2
    z = np.asarray(x + 1j * y)

    def identity(z):
        return ad.numpy.real(
            advect.incgamma(z, n=n + 1)
            - n * advect.incgamma(z, n=n)
            - z**n * ad.numpy.exp(-z)
        )

    assert_allclose(identity(z), 0, atol=5e-13)
    assert_allclose(ad.grad(identity)(z), 0, atol=5e-12)


@given(n=st.integers(-5, 3), z=st.floats(0.8, 1.8), eta=st.floats(0.9, 1.6))
@settings(max_examples=25, deadline=None)
def test_kambe_integration_by_parts_gradient(n, z, eta):
    inputs = np.array([z + 0.1j, eta + 0.04j])

    def identity(v):
        z, eta = v[0], v[1]
        return ad.numpy.real(
            (n + 1) * advect.intkambe(z, eta, n=n)
            - z**2 * advect.intkambe(z, eta, n=n + 2)
            - advect.intkambe(z, eta, n=n - 2)
            + eta ** (n + 1) * ad.numpy.exp((eta**-2 - z**2 * eta**2) / 2)
        )

    assert_allclose(identity(inputs), 0, atol=2e-10)
    assert_allclose(ad.grad(identity)(inputs), 0, atol=2e-9)


def test_integral_empty_scalar_broadcast_and_singular_derivatives():
    for n in (2, np.arange(4).reshape(4, 1) / 2 + 1):
        value, context = diff.incgamma(n, np.empty((0,), complex))
        assert value.size == 0
        assert context.pullback(np.empty(value.shape, complex)).shape == (0,)
    degrees = np.array([1, 2, 3])
    value, context = diff.incgamma(degrees, 0)
    assert_allclose(value, [1, 1, 2])
    assert_allclose(context.pullback(np.ones(3, complex)), -1)
    for n in (-6, -3):
        _, context = diff.intkambe(n, 0, 1)
        dz, deta = context.pullback(np.asarray(1, complex))
        assert_allclose(dz, 0)
        assert_allclose(deta, -np.exp(0.5))
    _, context = diff.intkambe(-2, 0, 1)
    with pytest.raises(ValueError, match="singular"):
        context.pullback(np.asarray(1, complex))
    _, context = diff.intkambe(-2, 0, 1)
    assert_allclose(context.pullback(np.asarray(0, complex)), [0, 0])
    _, context = diff.incgamma(0.5, 0)
    with pytest.raises(ValueError, match="singular"):
        context.pullback(np.asarray(1, complex))


def test_integral_parallel_matches_partitioned_and_degree_broadcast():
    z = np.linspace(0.8, 2, 2048) + 0.1j
    eta = 1.2 + 0.02j
    for function, arguments in (
        (diff.incgamma, (2.5, z)),
        (diff.intkambe, (-3, z, eta)),
    ):
        value, context = function(*arguments)
        expected = np.concatenate(
            [
                function(
                    *(a[sl] if isinstance(a, np.ndarray) else a for a in arguments)
                )[0]
                for sl in (slice(0, 1024), slice(1024, None))
            ]
        )
        assert_allclose(value, expected, rtol=2e-14)
        gradients = context.pullback(np.ones(value.shape, complex))
        if function is diff.intkambe:
            assert gradients[0].shape == z.shape
            assert gradients[1].shape == ()
        else:
            assert gradients.shape == z.shape
    value, context = diff.intkambe(
        np.array([-3, -2])[:, None], 1.1, np.array([1.0, 1.2, 1.3])
    )
    assert value.shape == (2, 3)
    dz, deta = context.pullback(np.ones((2, 3), complex))
    assert dz.shape == ()
    assert deta.shape == (3,)

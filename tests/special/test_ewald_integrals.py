"""Incomplete gamma and Kambe integrals of the Ewald sums against treams.special.

Checks the _native scalar functions (the fast path of special.incgamma and
special.intkambe for Python scalars), the broadcasting special wrappers, the diff
records with their owned adjoints, and recurrence identities.
"""

import advect
import numpy as np
import pytest
import treams.special as upstream_special
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import (
    _native,  # the Ewald integral references check the native scalar functions
    diff,
    special,
)
from treams_rs import advect as ad

from _support import assert_one_use_context


@pytest.mark.reference
@pytest.mark.parametrize("n", [-10, -3, -1.5, -0.5, 0, 0.5, 1, 4, 10])
@pytest.mark.parametrize(
    "z", [0.3, 1.5, 2 + 4j, 14 + 2j, complex(-3, 0.0), complex(-3, -0.0)]
)
def test_gamma_reference(n, z):
    assert_allclose(
        _native.incgamma_scalar(n, z),
        upstream_special.incgamma(n, z),
        rtol=2e-10,
        atol=1e-14,
    )


@pytest.mark.reference
@pytest.mark.parametrize("n", [-10, -9, -4, -3, -2, -1, 0, 1, 9, 10])
@pytest.mark.parametrize(
    "z,eta", [(0.8, 1.2), (1 + 1j, 2 - 1.1j), (1.4 + 0.01j, 3 - 0.01j), (3, 2)]
)
def test_kambe_reference(n, z, eta):
    assert_allclose(
        _native.intkambe_scalar(n, z, eta),
        upstream_special.intkambe(n, z, eta),
        rtol=2e-8,
        atol=1e-13,
    )


@pytest.mark.physics
@pytest.mark.reference
def test_public_ewald_integral_broadcasts_and_identities():
    n = np.arange(-6, 8)[:, None] / 2
    z = np.array([0.3 + 0.1j, 1.3 - 0.2j, 4.1 + 0.3j])
    actual = special.incgamma(n, z)
    assert_allclose(actual, upstream_special.incgamma(n, z), rtol=3e-10, atol=1e-13)
    assert_allclose(
        special.incgamma(n + 1, z),
        n * actual + z**n * np.exp(-z),
        rtol=3e-11,
        atol=1e-13,
    )
    orders = np.arange(-7, 5)[:, None]
    eta = np.array([0.5, 0.7 - 0.1j, 0.9])
    kambe = special.intkambe(orders, z, eta)
    assert_allclose(
        kambe, upstream_special.intkambe(orders, z, eta), rtol=3e-8, atol=1e-12
    )
    # The public wrappers forward out= and where= to the native ufuncs.
    for call, args, expected in (
        (special.incgamma, (n, z), actual),
        (special.intkambe, (orders, z, eta), kambe),
    ):
        out = np.full(expected.shape, 9 + 0j)
        assert call(*args, out=out, where=[True, False, True]) is out
        assert_allclose(out[:, [0, 2]], expected[:, [0, 2]])
        assert_allclose(out[:, 1], 9)


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize("n", [-8, -2.5, 0, 0.5, 1, 2, 8])
def test_gamma_owned_complex_argument(n):
    z = np.array([0.6 + 0.2j, 1.4 - 0.1j, 2.7 + 0.3j])[::-1]
    original = z.copy()
    value, context = diff.incgamma(n, z)
    assert_allclose(value, upstream_special.incgamma(n, z), rtol=3e-12, atol=2e-14)
    z[:] = 9
    g = np.array([0.3 + 0.2j, -0.2j, 0.4])
    # d/dz Gamma(n, z) = -z^(n-1) exp(-z).
    assert_one_use_context(
        context,
        g,
        -g * (original ** (n - 1) * np.exp(-original)).conj(),
        wrong_shape=np.ones((3, 1), complex),
    )


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize("n", [-8, -3, -2, -1, 0, 1, 5])
def test_kambe_owned_broadcast_arguments(n):
    z = np.array([[0.8 + 0.1j], [1.3 - 0.05j]])
    eta = np.array([[0.9 + 0.03j, 1.1 - 0.1j, 1.4 + 0.04j]])
    original = (z.copy(), eta.copy())
    value, context = diff.intkambe(n, z, eta)
    assert_allclose(value, upstream_special.intkambe(n, z, eta), rtol=3e-10, atol=1e-12)
    z[:] = 7
    eta[:] = 6
    # With I_n(z, eta) the integral of t^n exp(-z^2 t^2 / 2 + 1 / (2 t^2)) over
    # t > eta: dI/dz = -z I_(n+2) and dI/deta = -eta^n exp(...) at t = eta.
    z, eta = original
    dz = -z * special.intkambe(n + 2, z, eta)
    deta = -(eta**n) * np.exp((eta**-2 - z**2 * eta**2) / 2)
    g = np.arange(6).reshape(2, 3) * (0.1 + 0.04j)
    expected = (
        np.sum(g * dz.conj(), axis=1, keepdims=True),
        np.sum(g * deta.conj(), axis=0, keepdims=True),
    )
    assert_one_use_context(context, g, expected, rtol=1e-12)


@pytest.mark.gradients
@pytest.mark.physics
@given(n=st.integers(-6, 12), x=st.floats(0.8, 2.5), y=st.floats(-0.3, 0.3))
def test_gamma_recurrence_has_zero_value_and_gradient(n, x, y):
    n = n / 2
    z = np.asarray(x + 1j * y)

    def identity(z):
        return advect.numpy.real(
            ad.incgamma(z, n=n + 1)
            - n * ad.incgamma(z, n=n)
            - z**n * advect.numpy.exp(-z)
        )

    assert_allclose(identity(z), 0, atol=5e-13)
    assert_allclose(advect.grad(identity)(z), 0, atol=5e-12)


@pytest.mark.gradients
@pytest.mark.physics
@given(n=st.integers(-5, 3), z=st.floats(0.8, 1.8), eta=st.floats(0.9, 1.6))
@settings(max_examples=25)
def test_kambe_integration_by_parts_gradient(n, z, eta):
    inputs = np.array([z + 0.1j, eta + 0.04j])

    def identity(v):
        z, eta = v[0], v[1]
        return advect.numpy.real(
            (n + 1) * ad.intkambe(z, eta, n=n)
            - z**2 * ad.intkambe(z, eta, n=n + 2)
            - ad.intkambe(z, eta, n=n - 2)
            + eta ** (n + 1) * advect.numpy.exp((eta**-2 - z**2 * eta**2) / 2)
        )

    assert_allclose(identity(inputs), 0, atol=2e-10)
    assert_allclose(advect.grad(identity)(inputs), 0, atol=2e-9)


@pytest.mark.gradients
@pytest.mark.interface
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


@pytest.mark.gradients
@pytest.mark.interface
def test_integral_parallel_matches_partitioned_and_degree_broadcast():
    z = np.linspace(0.8, 2, 2048) + 0.1j
    eta = 1.2 + 0.02j
    halves = (slice(0, 1024), slice(1024, None))
    for function, arguments in (
        (diff.incgamma, (2.5, z)),
        (diff.intkambe, (-3, z, eta)),
    ):
        value, context = function(*arguments)
        parts = [
            function(*(a[half] if isinstance(a, np.ndarray) else a for a in arguments))
            for half in halves
        ]
        assert_allclose(value, np.concatenate([v for v, _ in parts]), rtol=2e-14)
        gradients = context.pullback(np.ones(value.shape, complex))
        expected = [c.pullback(np.ones(v.shape, complex)) for v, c in parts]
        if function is diff.intkambe:
            # The broadcast scalar eta collects the sum over both halves.
            concatenated = np.concatenate([g[0] for g in expected])
            assert_allclose(gradients[0], concatenated, rtol=1e-14)
            assert gradients[1].shape == ()
            assert_allclose(gradients[1], sum(g[1] for g in expected), rtol=1e-13)
        else:
            assert_allclose(gradients, np.concatenate(expected), rtol=1e-14)
    orders = np.array([-3, -2])[:, None]
    eta = np.array([1.0, 1.2, 1.3])
    value, context = diff.intkambe(orders, 1.1, eta)
    assert value.shape == (2, 3)
    dz, deta = context.pullback(np.ones((2, 3), complex))
    assert dz.shape == ()
    assert deta.shape == (3,)
    # The scalar z and the row of eta, broadcast over the orders, collect the
    # sums of the gradients of an explicitly broadcast record.
    _, broadcast = diff.intkambe(
        orders, np.full((2, 3), 1.1 + 0j), np.broadcast_to(eta, (2, 3)).copy()
    )
    full_dz, full_deta = broadcast.pullback(np.ones((2, 3), complex))
    assert_allclose(dz, full_dz.sum(), rtol=1e-13)
    assert_allclose(deta, full_deta.sum(axis=0), rtol=1e-13)

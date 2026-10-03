"""Real-degree Ferrers values and analytic argument pullbacks."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams.special as oracle
from hypothesis import example, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from treams_rs import advect as ad
from treams_rs import diff, special

from _support import assert_one_use_context, degree_order

mp = pytest.importorskip("mpmath", reason="high-precision oracle")


@pytest.mark.gradients
@pytest.mark.reference
@settings(max_examples=60)
@given(
    label=degree_order(2, 20, margin=1),
    fraction=st.floats(0.05, 0.95),
    x=st.floats(-0.9, 0.9),
)
@example(label=(20, 11), fraction=0.0625, x=-0.875)
def test_real_degree_reference_and_advect(label, fraction, x):
    # The degree recurrence and the derivative identity are checked natively
    # (real_degree_legendre_recurrence_and_derivative, over a wider domain), the
    # derivative also by the 70-digit fixtures below; the adapter must pass the
    # native pullback through unchanged.
    integer, order = label
    degree = integer + fraction
    value = special.lpmv(order, degree, x)
    with mp.workdps(30):
        exact = float(mp.legenp(degree, order, x, type=2))
    assert_allclose(value, exact, rtol=5e-10, atol=2e-11 * (1 + abs(value)))
    # treams calls SciPy's lpmv, whose upward recursion in the degree and
    # 100-term series lose up to 2.9e-7 relative accuracy for |m| >= 7.
    upstream = oracle.lpmv(order, degree, x)
    assert_allclose(value, upstream, rtol=1e-6, atol=2e-11 * (1 + abs(value)))

    def objective(argument):
        return anp.real(ad.angular(argument, degree=degree, order=order))

    _, context = diff.angular(degree, order, np.array(x))
    native = context.pullback(np.array(1 + 0j))
    assert native.imag == 0
    assert_allclose(
        advect.grad(objective)(np.array(x)), native.real, rtol=1e-13, atol=0
    )


# Independently evaluated at 70 decimal digits with mpmath's hypergeometric
# representation. Includes near-integer cancellation and high-order endpoint scaling.
@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize(
    "degree,order,x,value,derivative",
    [
        (0.2, 0, -0.999999, -1.7976872815135916, 187098.1884756791),
        (10.000000000000002, 6, -0.9, 76642.47046048816, 1738989.8378666244),
        (32.3, 12, -0.7, 1.931539353538286e16, -8.220038417299478e18),
        (65.2, -64, -0.9999, 1336260.7162305901, -427579033201.5145),
        (65.2, 64, -0.9999, 1.5960939897585381e224, -5.107209369776433e229),
        (127.2, -126, -0.999999, 2.2868712333767252e105, -1.4407281056659067e113),
        (127.2, 64, 0.999999, 2.1332851219934673e-24, -6.82648904372342e-17),
    ],
)
def test_scaled_endpoint_high_precision_fixtures(degree, order, x, value, derivative):
    actual, context = diff.angular(degree, order, x)
    assert_allclose(actual, value, rtol=3e-12, atol=0)
    assert_allclose(context.pullback(np.array(1 + 0j)), derivative, rtol=3e-12, atol=0)


@pytest.mark.gradients
@pytest.mark.interface
@pytest.mark.reference
def test_real_degree_broadcast_poles_and_owned_pullback():
    degrees = np.array([2.3, 5.7])[:, None]
    orders = np.array([-2, 0, 2])
    x = np.array([0.1, 0.4, 0.7])
    expected = oracle.lpmv(orders, degrees, x)
    assert_allclose(special.lpmv(orders, degrees, x), expected, rtol=2e-12, atol=2e-12)
    value, context = diff.angular(degrees, orders, x)
    _, duplicate = diff.angular(degrees, orders, x.copy())
    x[:] = 0
    assert_one_use_context(
        context, np.ones_like(value), duplicate.pullback(np.ones_like(value))
    )
    for degree in (0.3, 2.3, 127.3):
        _, context = diff.angular(degree, 0, 1.0)
        assert_allclose(
            context.pullback(np.array(1 + 0j)), degree * (degree + 1) / 2, rtol=2e-13
        )
    assert_allclose(special.lpmv(1, 2.3, 1.0), 0)
    _, context = diff.angular(2.3, 1, 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        context.pullback(np.array(1 + 0j))
    assert special.lpmv(3, 2.3, 0.1) == 0
    for kind in ("pi", "tau"):
        with pytest.raises(ValueError, match="noninteger"):
            diff.angular(2.3, 1, 0.1, function=kind)
    with pytest.raises(ValueError, match="real Legendre"):
        diff.angular(2.3, 1, 0.1 + 0.2j)
    assert_allclose(special.lpmv(0, 5e-324, -0.8), 1)


@pytest.mark.interface
@pytest.mark.reference
@settings(max_examples=60)
@given(degree=st.integers(0, 12), seed=st.integers(0, 100), x=st.floats(-3, 3))
def test_real_lpmv_is_the_real_restriction_of_the_complex_continuation(degree, seed, x):
    """Scalar and array real calls agree; only an imaginary continuation raises."""
    order = seed % (2 * degree + 5) - degree - 2  # includes the zero extension
    continuation = special.lpmv(order, degree, complex(x))
    assert continuation == special.lpmv([order], degree, complex(x))[0]
    if continuation.imag == 0:
        scalar = special.lpmv(order, degree, x)
        assert type(scalar) is float
        assert scalar == continuation.real
        array = special.lpmv([order], degree, np.array([x]))
        assert array.dtype == np.float64
        assert_array_equal(array, [continuation.real])
    else:
        assert abs(x) > 1 and order % 2 == 1
        for argument in (x, np.float64(x), np.array([x]), np.array(x)):
            with pytest.raises(ValueError, match="complex z"):
                special.lpmv(order, degree, argument)


@pytest.mark.interface
@pytest.mark.reference
def test_real_lpmv_outside_the_unit_interval():
    # P_3(1.5) and the even-order product (1 - x^2) P_3''(x) are real.
    assert special.lpmv(0, 3, 1.5) == pytest.approx(6.1875, rel=1e-15)
    assert special.lpmv(2, 3, 1.2) == special.lpmv(2, 3, 1.2 + 0j).real
    assert_allclose(special.lpmv(1, 2, 1.5 + 0j), -5.031152949374527j, rtol=1e-15)
    for argument in (1.5, [1.5], np.float32(1.5)):
        with pytest.raises(ValueError, match="odd orders"):
            special.lpmv(1, 2, argument)
    # Odd orders above the degree keep the reference's real zero extension.
    assert special.lpmv(3, 2, 1.5) == 0.0
    assert_array_equal(special.lpmv(-3, 2, [1.5, -2.0]), [0.0, 0.0])


@pytest.mark.interface
@settings(max_examples=60)
@given(
    kind=st.sampled_from(["legendre", "pi", "tau"]),
    degree=st.one_of(st.integers(0, 8), st.floats(0, 8), st.booleans()),
    order=st.one_of(st.integers(-3, 3), st.floats(-3, 3)),
    shape=st.sampled_from([(), (1,), (5,), (2, 3)]),
)
def test_python_number_labels_equal_0d_label_arrays(kind, degree, order, shape):
    # Python-number labels skip NumPy broadcasting; values, the z gradient and
    # errors equal those of the same labels as 0-d arrays.
    z = np.linspace(-0.8, 0.8, int(np.prod(shape))).reshape(shape) + 0.1j

    def recorded(labels):
        try:
            value, context = diff.angular(*labels, z, function=kind)
            return value, context.pullback(np.full(value.shape, 0.4 - 0.2j))
        except ValueError as error:
            return str(error)

    actual = recorded((degree, order))
    expected = recorded((np.asarray(degree), np.asarray(order)))
    if isinstance(expected, str):
        assert actual == expected
        return
    assert_array_equal(actual[0], expected[0])
    assert actual[1].shape == z.shape
    assert_array_equal(actual[1], expected[1])

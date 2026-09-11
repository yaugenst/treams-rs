"""Real-degree Ferrers values and analytic argument pullbacks."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams.special as oracle
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import advect as ad
from treams_rs import diff, special


@settings(max_examples=60)
@given(
    integer=st.integers(2, 20),
    fraction=st.floats(0.05, 0.95),
    x=st.floats(-0.9, 0.9),
    seed=st.integers(0, 40),
)
def test_real_degree_recurrence_reference_and_advect(integer, fraction, x, seed):
    degree = integer + fraction
    order = seed % (2 * integer - 1) - (integer - 1)
    value = special.lpmv(order, degree, x)
    assert_allclose(
        value, oracle.lpmv(order, degree, x), rtol=5e-10, atol=2e-11 * (1 + abs(value))
    )
    lower = special.lpmv(order, degree - 1, x)
    upper = special.lpmv(order, degree + 1, x)
    scale = 1 + degree * abs(value) + abs(lower) + abs(upper)
    assert_allclose(
        (degree - order + 1) * upper,
        (2 * degree + 1) * x * value - (degree + order) * lower,
        atol=2e-11 * scale,
        rtol=2e-11,
    )

    def objective(argument):
        return anp.real(ad.angular(argument, degree=degree, order=order))

    gradient = advect.grad(objective)(np.array(x))
    assert_allclose(
        (1 - x * x) * gradient,
        (degree + order) * lower - degree * x * value,
        rtol=2e-10,
        atol=2e-11 * scale,
    )


# Independently evaluated at 70 decimal digits with mpmath's hypergeometric
# representation. Includes near-integer cancellation and high-order endpoint scaling.
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


def test_real_degree_broadcast_strides_poles_and_owned_pullback():
    degrees = np.array([2.3, 5.7])[:, None]
    orders = np.array([-2, 0, 2])
    x = np.array([0.1, 0.4, 0.7])
    expected = oracle.lpmv(orders, degrees, x)
    storage = np.zeros((3, 4))
    out = storage[:, ::2].T
    assert special.lpmv(orders, degrees, x, out=out) is out
    assert_allclose(out, expected, rtol=2e-12, atol=2e-12)
    value, context = diff.angular(degrees, orders, x)
    _, duplicate = diff.angular(degrees, orders, x.copy())
    x[:] = 0
    assert_allclose(
        context.pullback(np.ones_like(value)),
        duplicate.pullback(np.ones_like(value)),
        atol=0,
    )
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(np.ones_like(value))
    for degree in (0.3, 2.3, 127.3):
        _, context = diff.angular(degree, 0, 1.0)
        assert_allclose(
            context.pullback(np.array(1 + 0j)), degree * (degree + 1) / 2, rtol=2e-13
        )
    assert_allclose(special.lpmv(1, 2.3, 1.0), 0)
    _, context = diff.angular(2.3, 1, 1.0)
    with pytest.raises(ValueError, match="nonfinite"):
        context.pullback(np.array(1 + 0j))
    assert special.lpmv(3, 2.3, 0.1) == 0
    for kind in ("pi", "tau"):
        with pytest.raises(ValueError, match="noninteger"):
            diff.angular(2.3, 1, 0.1, kind=kind)
    with pytest.raises(ValueError, match="real Legendre"):
        diff.angular(2.3, 1, 0.1 + 0.2j)
    assert_allclose(special.lpmv(0, 5e-324, -0.8), 1)

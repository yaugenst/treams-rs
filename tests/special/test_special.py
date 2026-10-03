"""Bessel, Hankel and angular functions (treams_rs.special ufuncs, diff.bessel and
diff.angular): SciPy and upstream references, jets, branch cuts and pullbacks."""

from concurrent.futures import ThreadPoolExecutor

import advect
import advect.numpy as anp
import numpy as np
import pytest
import scipy.special as scipy_special
import treams.special as oracle
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

# _native: the cylindrical Bessel jets are a test hook without a public wrapper.
from treams_rs import _native, diff, special
from treams_rs import advect as ad
from treams_rs.testing import check_gradient, check_pullback

from _support import assert_one_use_context, complex_normal, degree_order


@pytest.mark.reference
@pytest.mark.parametrize(
    "name",
    [
        "jv",
        "yv",
        "hankel1",
        "hankel2",
        "jv_d",
        "yv_d",
        "hankel1_d",
        "hankel2_d",
        "spherical_jn",
        "spherical_yn",
        "spherical_hankel1",
        "spherical_hankel2",
        "spherical_jn_d",
        "spherical_yn_d",
        "spherical_hankel1_d",
        "spherical_hankel2_d",
    ],
)
def test_bessel_broadcast_reference(name):
    degrees = (
        np.arange(12)[:, None]
        if name.startswith("spherical_")
        else np.array([-3.4, -2, -0.2, 0, 0.2, 0.5, 3, 5.3])[:, None]
    )
    z = np.array(
        [0.1 + 0.03j, 1.2 - 0.7j, 2.4 + 0.2j, -1.3 + 0.1j, -1.3 - 0.1j, 8.3 + 0.4j]
    )
    assert_allclose(
        getattr(special, name)(degrees, z),
        getattr(oracle, name)(degrees, z),
        rtol=2e-12,
        atol=3e-14,
    )
    assert_allclose(
        getattr(special, name)(0, 1.2 + 0.1j),
        getattr(oracle, name)(0, 1.2 + 0.1j),
        rtol=2e-13,
    )


@pytest.mark.reference
@pytest.mark.parametrize("degree", [0, 1, 3, 10, 30])
@pytest.mark.parametrize("z", [0.0, 0.01 + 0.02j, 0.3 - 0.3j, 1.2, 8.0 + 0.5j])
def test_spherical_bessel_jets(degree, z):
    # |z| < 0.5 takes the regular series jet that the Mie kernel also uses; the
    # series second derivative is checked natively (spherical_series_and_wronskian).
    assert_allclose(
        special.spherical_jn(degree, z),
        scipy_special.spherical_jn(degree, z),
        atol=1e-14,
        rtol=1e-12,
    )
    assert_allclose(
        special.spherical_jn_d(degree, z),
        scipy_special.spherical_jn(degree, z, True),
        atol=1e-14,
        rtol=1e-12,
    )
    if z != 0:
        expected = scipy_special.spherical_jn(
            degree, z
        ) + 1j * scipy_special.spherical_yn(degree, z)
        assert_allclose(
            special.spherical_hankel1(degree, z), expected, atol=1e-13, rtol=1e-12
        )
        expected = scipy_special.spherical_jn(
            degree, z, True
        ) + 1j * scipy_special.spherical_yn(degree, z, True)
        assert_allclose(
            special.spherical_hankel1_d(degree, z), expected, atol=1e-13, rtol=1e-12
        )


@pytest.mark.physics
def test_regular_spherical_jet_is_continuous_across_series_switch():
    # Below |z| = 0.5 the regular jet comes from its power series, above from
    # Bessel evaluations. Across the switch the value and first derivative agree
    # after a Taylor step whose higher derivatives follow from the spherical Bessel
    # equation f'' = -2 f'/z + (n/z^2 - 1) f with n = l (l + 1).
    for degree in range(58):
        n = degree * (degree + 1)
        for angle in np.linspace(-np.pi, np.pi, 13):
            inner, outer = 0.5 * np.exp(1j * angle) * (1 + np.array([-1, 1]) * 1e-12)
            f = special.spherical_jn(degree, inner)
            d1 = special.spherical_jn_d(degree, inner)
            d2 = -2 * d1 / inner + (n / inner**2 - 1) * f
            d3 = -2 * d2 / inner + 2 * d1 / inner**2 - 2 * n * f / inner**3
            d3 += (n / inner**2 - 1) * d1
            step = outer - inner
            expected = [
                f + d1 * step + d2 * step**2 / 2,
                d1 + d2 * step + d3 * step**2 / 2,
            ]
            actual = [
                special.spherical_jn(degree, outer),
                special.spherical_jn_d(degree, outer),
            ]
            assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("order", [-8, -1, 0, 1, 2, 10, 30])
@pytest.mark.parametrize("z", [1e-7 + 1e-8j, 0.2 + 0.1j, 5 + 0.3j])
def test_cylindrical_bessel_jets(order, z):
    for singular in (False, True):
        actual = _native.cylindrical_radial_jet(order, z, singular)
        function = scipy_special.h1vp if singular else scipy_special.jvp
        expected = [function(order, z, n) for n in range(3)]
        assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)


def _bessel_ufuncs(spherical, kind):
    """The public value and derivative ufuncs of one Bessel kind."""
    names = {"j": "jv", "y": "yv", "h1": "hankel1", "h2": "hankel2"}
    if spherical:
        names = {"j": "spherical_jn", "y": "spherical_yn"} | {
            h: "spherical_" + names[h] for h in ("h1", "h2")
        }
    return getattr(special, names[kind]), getattr(special, names[kind] + "_d")


@pytest.mark.physics
@pytest.mark.parametrize("zero", [None, 0.0, -0.0], ids=["float", "+0j", "-0j"])
@pytest.mark.parametrize("derivative", [0, 1])
@pytest.mark.parametrize(
    "kind,mirror,odd",
    [
        ("jn", "jn", 0),
        ("yn", "yn", 1),
        ("hankel1", "hankel2", 0),
        ("hankel2", "hankel1", 0),
    ],
)
def test_spherical_bessel_parity_on_the_negative_real_axis(
    kind, mirror, odd, derivative, zero
):
    # j_n and y_n are single-valued with parity (-1)^n and (-1)^(n+1), and
    # h1_n(-x) = (-1)^n h2_n(x); derivatives have the opposite parity. This holds
    # on the negative axis with either sign of zero too: inside the power series
    # of j_n (|x| < 0.5) and where sqrt(pi / (2 z)) multiplies the AMOS function
    # of order n + 1/2. AMOS leaves rounding-size imaginary parts at real
    # arguments, so the tolerance scales with |f| + |f'|, as natively.
    degrees = np.arange(6)
    x = np.array([0.05, 0.6, 2.0, 8.0])[:, None]
    z = -x
    if zero is not None:
        z = z.astype(complex)
        z.imag = zero
    reflected = [
        getattr(special, f"spherical_{mirror}{suffix}")(degrees, x)
        for suffix in ("", "_d")
    ]
    scale = np.abs(reflected[0]) + np.abs(reflected[1])
    sign = (-1.0) ** (degrees + odd + derivative)
    function = getattr(special, f"spherical_{kind}{'_d' if derivative else ''}")
    assert_allclose(
        function(degrees, z) / scale,
        sign * reflected[derivative] / scale,
        rtol=0,
        atol=1e-13,
    )


@pytest.mark.gradients
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("kind", ["j", "y", "h1", "h2"])
@pytest.mark.parametrize("derivative", [False, True])
def test_bessel_native_broadcast_vjp(spherical, kind, derivative):
    # The pullback of f is g conj(f'), summed over the broadcast orders. For
    # derivative=True, f'' follows from the (spherical) Bessel equation.
    kwargs = {"spherical": spherical, "function": kind, "derivative": derivative}
    function, first = _bessel_ufuncs(spherical, kind)

    def slope(order, z):
        if not derivative:
            return first(order, z)
        f, d = function(order, z), first(order, z)
        if spherical:
            return -2 * d / z - (1 - order * (order + 1) / z**2) * f
        return -d / z - (1 - order**2 / z**2) * f

    order = np.arange(4)[:, None, None]
    z = np.array([[[1.2 + 0.1j], [1.4 - 0.2j]]])
    value, context = diff.bessel(order, z, **kwargs)
    assert_allclose(value, (first if derivative else function)(order, z), rtol=1e-13)
    g = complex_normal(np.random.default_rng(38), value.shape)
    gradient = context.pullback(g)
    assert gradient.shape == z.shape
    expected = np.sum(g * slope(order, z).conj(), axis=0, keepdims=True)
    assert_allclose(gradient, expected, rtol=1e-13, atol=1e-15)
    order, z = np.arange(4), np.array(1.3 + 0.2j)
    _, context = diff.bessel(order, z, **kwargs)
    g = np.array([0.2 + 0.1j, 0.3, 0.4j, -0.1])
    gradient = context.pullback(g)
    assert gradient.shape == ()
    expected = np.sum(g * slope(order, z).conj())
    assert_allclose(gradient, expected, rtol=1e-13, atol=1e-15)


@pytest.mark.gradients
def test_bessel_origin_limits():
    degrees = np.arange(5)
    for spherical in (False, True):
        value, context = diff.bessel(degrees, 0, spherical=spherical)
        assert_allclose(value, [1, 0, 0, 0, 0])
        assert_allclose(
            context.pullback(np.ones(5, complex)), 1 / 3 if spherical else 0.5
        )
        value, context = diff.bessel(degrees, 0, spherical=spherical, derivative=True)
        assert_allclose(value, [0, 1 / 3 if spherical else 0.5, 0, 0, 0])
        assert_allclose(
            context.pullback(np.ones(5, complex)), -0.2 if spherical else -0.25
        )


@pytest.mark.gradients
@pytest.mark.interface
@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_bessel_empty_broadcast(shape):
    z = np.ones(shape) * 1.3
    value, context = diff.bessel(np.empty((0, 3)), z)
    assert value.shape == (0, 3)
    gradient = context.pullback(np.empty((0, 3), complex))
    assert gradient.shape == shape
    assert_allclose(gradient, 0)
    z = np.empty((0, 1))
    value, context = diff.bessel(np.arange(3), z)
    assert context.pullback(np.empty_like(value)).shape == z.shape


@pytest.mark.gradients
@pytest.mark.interface
def test_bessel_ownership_strides_parallel_and_retry():
    order = np.tile(np.arange(4), 256)
    z = np.linspace(1.1, 2.3, 2048)[::2].astype(complex)
    value, context = diff.bessel(order, z)
    g = np.full_like(value, 0.2 + 0.1j)[::-1]
    expected = context.pullback(g)
    value, context = diff.bessel(order[::-1], z[::-1])
    order[:] = 1
    z[:] = 0
    assert_one_use_context(context, g, expected[::-1])


@pytest.mark.interface
@pytest.mark.reference
def test_bessel_ufunc_error_and_mask_contract():
    for argument in (0j, complex(np.nan), np.zeros(256, complex)):
        with pytest.raises(ValueError):
            special.hankel1(1, argument)
    out = np.array([123, 0], dtype=complex)
    special.hankel1(1, [0, 1.3], out=out, where=[False, True])
    assert out[0] == 123
    assert_allclose(out[1], oracle.hankel1(1, 1.3), rtol=1e-13)
    for function in (special.hankel1, special.spherical_jn):
        assert function([], 1.3).shape == (0,)


@pytest.mark.interface
@pytest.mark.reference
@given(order=st.integers(0, 10), size=st.integers(0, 160), backwards=st.booleans())
def test_bessel_ufunc_inplace_derivative(order, size, backwards):
    z = np.linspace(0.7, 3.0, size).astype(complex) + 0.2j
    if backwards:
        z = z[::-1]
    expected = oracle.jv_d(order, z)
    special.jv_d(order, z, out=z)
    assert_allclose(z, expected, rtol=2e-12, atol=1e-13)


@pytest.mark.interface
@pytest.mark.reference
def test_bessel_ufunc_concurrent_calls():
    z = np.linspace(0.6, 3, 256) + 0.2j
    expected = oracle.hankel1(np.arange(8)[:, None], z)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda order: special.hankel1(order, z), range(8)))
    assert_allclose(values, expected, rtol=2e-12, atol=1e-13)


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize("kind", ["h1", "h2"])
@given(
    order=st.floats(-8, 8, allow_nan=False, allow_infinity=False),
    x=st.floats(0.3, 8),
    y=st.floats(-2, 2),
)
@settings(max_examples=35)
def test_hankel_sequence_derivatives_and_reflection(kind, order, x, y):
    z = x + 1j * y
    function = special.hankel1 if kind == "h1" else special.hankel2
    scipy_reference = oracle.hankel1 if kind == "h1" else oracle.hankel2

    def reference(order, z):
        # SciPy 1.16.3 returns NaN for subnormal order on this branch. Its
        # continuous zero-order limit differs by far less than double precision.
        return scipy_reference(0 if abs(order) < np.finfo(float).tiny else order, z)

    derivative = special.hankel1_d if kind == "h1" else special.hankel2_d
    value, context = diff.bessel(order, z, function=kind, derivative=True)
    expected = 0.5 * (reference(order - 1, z) - reference(order + 1, z))
    assert_allclose(value, expected, rtol=3e-12, atol=1e-12)
    assert_allclose(derivative(order, z), value, rtol=3e-12, atol=1e-12)
    second = 0.25 * (
        reference(order - 2, z) - 2 * reference(order, z) + reference(order + 2, z)
    )
    assert_allclose(
        context.pullback(np.array(0.3 + 0.2j)),
        (0.3 + 0.2j) * second.conjugate(),
        rtol=5e-12,
        atol=1e-12,
    )
    assert_allclose(
        function(-order, z),
        np.exp((1j if kind == "h1" else -1j) * np.pi * order) * function(order, z),
        rtol=3e-12,
        atol=1e-12,
    )


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize(
    "name,kind", [("lpmv", "legendre"), ("pi_fun", "pi"), ("tau_fun", "tau")]
)
def test_angular_reference_and_broadcast(name, kind):
    labels = [(degree, m) for degree in range(13) for m in range(-degree, degree + 1)]
    degrees, orders = np.array(labels).T[:, :, None]
    z = np.array([-1, -0.9, 0, 0.7, 1, 1.3 + 0.2j, -1.3 - 0.2j, 0.2 + 0.5j])
    a, b = (orders, degrees) if name == "lpmv" else (degrees, orders)
    function = getattr(special, name)
    expected = getattr(oracle, name)(a, b, z.astype(complex))
    # Upstream's m=0 pi at degree=0 on the axis is finite, as are all integers.
    actual = function(a, b, z)
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    value, context = diff.angular(degrees, orders, z, function=kind)
    assert_allclose(value, actual, rtol=2e-13, atol=1e-13)
    assert_allclose(context.pullback(np.zeros_like(value)), 0)


@pytest.mark.physics
@pytest.mark.gradients
def test_pi_and_tau_follow_from_legendre():
    # pi = m P / sin(theta) and tau = dP/dtheta = -sin(theta) dP/dz, with dP/dz
    # from the native Legendre pullback, for every label up to degree 12. Next
    # to the poles, `sine` repeats the implementation's cancelling
    # sqrt(1 - z^2): those points check consistency with its sin(theta), not
    # the accuracy of P there (docs/validation/numerical-limits.md).
    labels = [
        (degree, m) for degree in range(1, 13) for m in range(-degree, degree + 1)
    ]
    degrees, orders = np.array(labels).T[:, :, None]
    z = np.array([-1 + 1e-6, -0.8, -0.3 + 0.2j, 0.1 - 0.3j, 0.5, 0.8 + 0.1j, 1 - 1e-6])
    z = np.broadcast_to(z, (len(labels), len(z))).copy()
    sine = np.sqrt(1 - z * z)
    legendre, context = diff.angular(degrees, orders, z)
    slope = context.pullback(np.ones_like(legendre)).conj()
    for actual, expected in [
        (special.pi_fun(degrees, orders, z) * sine, orders * legendre),
        (special.tau_fun(degrees, orders, z), -sine * slope),
    ]:
        error = np.abs(actual - expected)
        assert np.all(error <= 1e-12 * (np.abs(actual) + np.abs(expected)))


@pytest.mark.interface
@given(label=degree_order(0, 30), x=st.floats(-1, 1))
def test_real_legendre_loop_matches_complex_loop(label, x):
    # A float64 argument selects the real-degree Ferrers loop; for integer
    # degrees on [-1, 1] it must agree with the complex loop.
    degree, order = label
    value = special.lpmv(order, degree, np.array([x]))
    assert value.dtype == np.float64
    expected = special.lpmv(order, degree, np.array([x + 0j]))
    assert_allclose(value, expected, rtol=1e-13, atol=1e-300)


@pytest.mark.gradients
@pytest.mark.parametrize("kind", ["legendre", "pi", "tau"])
@given(label=degree_order(1, 10), x=st.floats(-0.8, 0.8), y=st.floats(-0.3, 0.3))
@settings(max_examples=35)
def test_angular_complex_argument_adjoint(kind, label, x, y):
    degree, m = label
    z = np.array(complex(x, y))
    value = diff.angular(degree, m, z, function=kind)[0]
    check_pullback(
        lambda z: diff.angular(degree, m, z, function=kind),
        z,
        directions=(np.array(0.17 + 0.11j),),
        cotangents=np.array(0.3 + 0.2j),
        step=2e-6,
        rtol=2e-6,
        atol=2e-8 * max(1, abs(value)),
    )


@pytest.mark.gradients
@pytest.mark.interface
def test_angular_polar_derivatives_and_branch_points():
    for z in (-1.0, 1.0):
        # P_3 = (5z^3-3z)/2 and pi_3^1 = -(15z^2-3)/2.
        for kind, m, expected in [
            ("legendre", 0, 6),
            ("pi", 1, -15 * z),
            ("tau", 1, -51),
            ("legendre", 2, -30),
        ]:
            _, context = diff.angular(3, m, z, function=kind)
            assert_allclose(context.pullback(np.array(1 + 0j)), expected)
        for kind, m in [("legendre", 1), ("pi", 2), ("tau", 0)]:
            _, context = diff.angular(3, m, z, function=kind)
            with pytest.raises(ValueError, match="undefined"):
                context.pullback(np.array(1 + 0j))
            _, context = diff.angular(3, m, z, function=kind)
            assert_allclose(context.pullback(np.array(0j)), 0)
        for m in (-3, 3):
            _, context = diff.angular(3, m, z)
            assert_allclose(context.pullback(np.array(1 + 0j)), 0)


@pytest.mark.gradients
@pytest.mark.parametrize("kind", ["legendre", "pi", "tau"])
def test_angular_broadcast_ownership_and_advect(kind):
    degrees = np.array([2, 3, 4])[:, None, None]
    orders = np.array([1, -1])[:, None]
    z = np.array([0.2 + 0.1j, 0.4 - 0.2j])
    value, context = diff.angular(degrees, orders, z, function=kind)
    g = np.full_like(value, 0.2 + 0.3j)
    gradient = context.pullback(g)
    value, context = diff.angular(degrees, orders, z, function=kind)
    saved = z.copy()
    z[:] = 0
    assert_one_use_context(context, g, gradient)
    for arguments in (saved, np.array(0.2 + 0.1j)):

        def objective(argument):
            result = ad.angular(argument, degree=degrees, order=orders, function=kind)
            return anp.sum(anp.abs(result) ** 2)

        check_gradient(
            objective,
            advect.grad(objective),
            arguments,
            directions=(np.ones(arguments.shape),),
            step=1e-6,
            rtol=1e-7,
            atol=1e-8,
        )
    value, context = diff.angular(np.empty((0, 2)), 1, np.ones((1, 1)))
    assert value.shape == (0, 2)
    assert_allclose(context.pullback(np.empty_like(value)), np.zeros((1, 1)))


@pytest.mark.reference
def test_hankel_subnormal_order_continuous_limit():
    for name in ("hankel1", "hankel2", "hankel1_d", "hankel2_d"):
        function = getattr(special, name)
        for order in (5e-324, -5e-324, 1e-310):
            assert_allclose(
                function(order, 1 + 0j), getattr(oracle, name)(0, 1 + 0j), rtol=2e-14
            )

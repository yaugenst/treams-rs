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


@pytest.mark.parametrize("spherical", [False, True])
@given(x=st.floats(0.7, 5.0), y=st.floats(-0.5, 0.5), order=st.integers(0, 10))
@settings(max_examples=30)
def test_bessel_wronskian_and_hankel_identities(spherical, x, y, order):
    z = x + 1j * y

    def value(kind, derivative=False):
        return diff.bessel(
            order, z, kind=kind, spherical=spherical, derivative=derivative
        )[0]

    j, y = value("j"), value("y")
    assert_allclose(value("h1"), j + 1j * y, rtol=2e-12, atol=1e-13)
    assert_allclose(value("h2"), j - 1j * y, rtol=2e-12, atol=1e-13)
    wronskian = j * value("y", True) - value("j", True) * y
    assert_allclose(
        wronskian, 1 / z**2 if spherical else 2 / (np.pi * z), rtol=3e-12, atol=1e-13
    )


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("kind", ["j", "y", "h1", "h2"])
@pytest.mark.parametrize("derivative", [False, True])
def test_bessel_native_broadcast_vjp(spherical, kind, derivative):
    order = np.arange(4)[:, None, None]
    z = np.array([[[1.2 + 0.1j], [1.4 - 0.2j]]])
    kwargs = {"spherical": spherical, "kind": kind, "derivative": derivative}
    value, context = diff.bessel(order, z, **kwargs)
    rng = np.random.default_rng(38)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradient = context.pullback(g)
    assert gradient.shape == z.shape
    direction = rng.normal(size=z.shape) * 0.1 + 0.03j
    h = 1e-5
    numerical = (
        diff.bessel(order, z + h * direction, **kwargs)[0]
        - diff.bessel(order, z - h * direction, **kwargs)[0]
    ) / (2 * h)
    assert_allclose(
        np.vdot(gradient, direction).real,
        np.vdot(g, numerical).real,
        rtol=2e-7,
        atol=2e-9,
    )
    _, context = diff.bessel(np.arange(4), np.array(1.3 + 0.2j), **kwargs)
    g = np.array([0.2 + 0.1j, 0.3, 0.4j, -0.1])
    scalar_gradient = context.pullback(g)
    assert scalar_gradient.shape == ()
    numerical = (
        diff.bessel(np.arange(4), 1.3 + 0.2j + h, **kwargs)[0]
        - diff.bessel(np.arange(4), 1.3 + 0.2j - h, **kwargs)[0]
    ) / (2 * h)
    assert_allclose(
        scalar_gradient.real, np.vdot(g, numerical).real, rtol=2e-7, atol=2e-9
    )


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


def test_bessel_ownership_strides_parallel_and_retry():
    order = np.tile(np.arange(4), 256)
    z = np.linspace(1.1, 2.3, 2048)[::2].astype(complex)
    value, context = diff.bessel(order, z)
    g = np.full_like(value, 0.2 + 0.1j)[::-1]
    expected = context.pullback(g)
    value, context = diff.bessel(order[::-1], z[::-1])
    order[:] = 1
    z[:] = 0
    with pytest.raises(ValueError, match="shape"):
        context.pullback(g[:1])
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full_like(g, np.nan))
    assert_allclose(context.pullback(g), expected[::-1])
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


def test_bessel_advect_broadcast_composition():
    order = np.arange(4)[:, None]
    z = np.array([1.2 + 0.1j, 1.3 - 0.2j])

    def objective(z):
        value = ad.bessel(z, order=order, kind="h1")
        return anp.sum(anp.abs(value) ** 2)

    gradient = advect.grad(objective)(z)
    direction = np.array([0.2 + 0.1j, -0.1j])
    h = 1e-5
    assert_allclose(
        np.vdot(gradient, direction).real,
        (objective(z + h * direction) - objective(z - h * direction)) / (2 * h),
        rtol=2e-7,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "name", ["hankel1", "hankel1_d", "jv", "spherical_jn", "spherical_yn"]
)
def test_bessel_ufunc_outputs_masks_aliasing_and_strides(name):
    function = getattr(special, name)
    order = np.arange(4)[:, None]
    arguments = np.linspace(0.8, 2.4, 257).astype(complex) + 0.2j
    expected = getattr(oracle, name)(order, arguments)
    buffer = np.full((257, 8), 17 + 3j)
    out = buffer[:, ::2].T[::-1]
    mask = np.indices(out.shape).sum(axis=0) % 3 == 0
    assert function(order, arguments, out=out, where=mask) is out
    assert_allclose(out[mask], expected[mask], rtol=3e-12, atol=1e-13)
    assert_allclose(out[~mask], 17 + 3j)
    assert_allclose(buffer[:, 1::2], 17 + 3j)
    shared = np.linspace(0.8, 2.4, 258).astype(complex) + 0.2j
    expected = getattr(oracle, name)(3, shared[:-1].copy())
    function(3, shared[:-1], out=shared[1:])
    assert_allclose(shared[1:], expected, rtol=3e-12, atol=1e-13)
    unaligned = np.ndarray(
        (257,), dtype=complex, buffer=bytearray(16 * 257 + 1), offset=1
    )
    unaligned[:] = arguments
    function(3, unaligned, out=unaligned)
    assert_allclose(
        unaligned, getattr(oracle, name)(3, arguments), rtol=3e-12, atol=1e-13
    )
    assert function([], 1.3).shape == (0,)


def test_bessel_ufunc_error_and_mask_contract():
    assert isinstance(special.jv, np.ufunc)
    assert (special.jv.nin, special.jv.nout) == (2, 1)
    for argument in (0j, complex(np.nan), np.zeros(256, complex)):
        with pytest.raises(ValueError):
            special.hankel1(1, argument)
    out = np.array([123, 0], dtype=complex)
    special.hankel1(1, [0, 1.3], out=out, where=[False, True])
    assert out[0] == 123
    assert_allclose(out[1], oracle.hankel1(1, 1.3), rtol=1e-13)


@given(order=st.integers(0, 10), size=st.integers(0, 160), backwards=st.booleans())
@settings(max_examples=30)
def test_bessel_ufunc_inplace_derivative_identity(order, size, backwards):
    z = np.linspace(0.7, 3.0, size).astype(complex) + 0.2j
    if backwards:
        z = z[::-1]
    expected = (special.jv(order - 1, z) - special.jv(order + 1, z)) / 2
    special.jv_d(order, z, out=z)
    assert_allclose(z, expected, rtol=2e-12, atol=1e-13)


def test_bessel_ufunc_concurrent_calls():
    from concurrent.futures import ThreadPoolExecutor

    z = np.linspace(0.6, 3, 256) + 0.2j
    expected = oracle.hankel1(np.arange(8)[:, None], z)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda order: special.hankel1(order, z), range(8)))
    assert_allclose(values, expected, rtol=2e-12, atol=1e-13)


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
    reference = oracle.hankel1 if kind == "h1" else oracle.hankel2
    derivative = special.hankel1_d if kind == "h1" else special.hankel2_d
    value, context = diff.bessel(order, z, kind=kind, derivative=True)
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

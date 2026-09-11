"""Vector-wave component conventions, gufunc contracts and owned analytic VJPs."""

import numpy as np
import pytest
import treams.special as oracle
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import diff
from treams_rs import special as sp

pytestmark = pytest.mark.filterwarnings(
    "ignore:`scipy.special.sph_harm` is deprecated:DeprecationWarning"
)

NAMES = ["vsh_X", "vsh_Y", "vsh_Z"] + [
    f"{family}_{radial}{pol}"
    for family in ["vsw", "vcw", "vpw"]
    for radial in (["", "r"] if family != "vpw" else [""])
    for pol in ["M", "N", "A"]
]


def arguments(name, theta=0.8, n=1):
    phase = np.linspace(-0.4, 0.7, n) if n != 1 else 0.3
    if name.startswith("vsh"):
        return [theta, phase]
    if name.startswith("vsw"):
        return [1.2 + 0.1j, theta, phase]
    if name.startswith("vcw"):
        result = [0.2, 1.2 + 0.1j, phase, 0.4]
        return result if name.endswith("M") else [*result, 1.3 + 0.1j]
    return [0.3 + 0.03j, 0.4, 1.2 + 0.04j, phase, 0.5, 0.6]


def public_args(name, args, degree=3, order=-1, polarization=1):
    if name.startswith(("vsh", "vsw")):
        result = [degree, order, *args]
    elif name.startswith("vcw"):
        result = [args[0], order, *args[1:]]
    else:
        result = list(args)
    return [*result, polarization] if name.endswith("A") else result


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("theta", [0.0, 0.8, np.pi, -0.7, 3.7, 0.7 + 0.3j, -0.7 + 0.3j])
def test_public_wave_reference(name, theta):
    args = public_args(name, arguments(name, theta))
    actual = getattr(sp, name)(*args)
    expected = getattr(oracle, name)(*args)
    assert actual.shape == (3,)
    assert actual.dtype == np.complex128
    assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)


@pytest.mark.parametrize("name", ["vpw_M", "vpw_N", "vpw_A"])
def test_scalar_plane_wave_fast_path_and_output(name):
    args = public_args(name, arguments(name))
    value = getattr(sp, name)(*args)
    out = np.empty(3, complex)
    assert getattr(sp, name)(*args, out) is out
    assert_allclose(out, value, rtol=2e-15)
    if name == "vpw_A":
        for pol in (-1, 2):
            with pytest.raises(ValueError, match="polarization"):
                sp.vpw_A(*args[:-1], pol)


@pytest.mark.parametrize("name", NAMES)
def test_vector_wave_gufunc_shapes_strides_and_parallel(name):
    args = public_args(name, arguments(name, n=2048))
    expected = getattr(oracle, name)(*args)
    func = getattr(sp, name)
    actual = func(*args)
    assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)
    out = np.empty((2048, 6), complex)[:, ::2]
    assert func(*args, out=out) is out
    assert_allclose(out, expected, rtol=3e-12, atol=3e-13)
    out = np.empty((2048, 3), complex)[:, ::-1]
    func(*args, out=out)
    assert_allclose(out, expected, rtol=3e-12, atol=3e-13)
    transposed = func(*args, axes=[()] * len(args) + [(0,)])
    assert_allclose(transposed, expected.T, rtol=3e-12, atol=3e-13)
    unaligned = np.ndarray(
        (2048, 3), complex, buffer=bytearray(2048 * 3 * 16 + 1), offset=1
    )
    func(*args, out=unaligned)
    assert_allclose(unaligned, expected, rtol=3e-12, atol=3e-13)
    empty = public_args(name, arguments(name, n=0))
    assert func(*empty).shape == (0, 3)


@pytest.mark.parametrize("name", NAMES)
def test_wave_pullbacks_all_arguments_ownership_broadcast_and_advect(name):
    import advect
    import advect.numpy as anp

    from treams_rs import advect as ad

    args = [np.asarray(v, complex) for v in arguments(name, n=3)]
    args[0] = np.broadcast_to(args[0], (2, 1)).copy()
    rng = np.random.default_rng(43)
    directions = [
        rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape) for v in args
    ]
    # The recorded boundary analytically continues every argument to complex.
    value, ctx = diff.vector_wave(*args, kind=name, degree=3, order=-1, polarization=1)
    assert_allclose(
        value,
        getattr(sp, name)(*public_args(name, arguments(name, n=3)))[None]
        + np.zeros((2, 1, 1)),
        rtol=3e-12,
        atol=3e-13,
    )
    cot = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    with pytest.raises(ValueError, match="shape"):
        ctx.pullback(np.ones((1,), complex))
    with pytest.raises(ValueError, match="finite"):
        ctx.pullback(np.full(value.shape, np.nan, complex))
    saved = [a.copy() for a in args]
    for a in args:
        a[...] += 0.1
    grads = ctx.pullback(cot)
    for a, g in zip(saved, grads, strict=True):
        assert g.shape == a.shape
    with pytest.raises(ValueError, match="consumed"):
        ctx.pullback(cot)
    step = 2e-6
    for i, direction in enumerate(directions):
        plus, minus = list(saved), list(saved)
        plus[i] = saved[i] + step * direction
        minus[i] = saved[i] - step * direction
        fd = (
            diff.vector_wave(*plus, kind=name, degree=3, order=-1, polarization=1)[0]
            - diff.vector_wave(*minus, kind=name, degree=3, order=-1, polarization=1)[0]
        ) / (2 * step)
        assert_allclose(
            np.vdot(grads[i], direction).real,
            np.vdot(cot, fd).real,
            rtol=3e-7,
            atol=3e-9,
        )

    def objective(*values):
        v = ad.vector_wave(*values, kind=name, degree=3, order=-1, polarization=1)
        return anp.real(anp.sum(anp.conj(v) * v))

    actual = advect.grad(objective, argnums=tuple(range(len(args))))(*saved)
    value, context = diff.vector_wave(
        *saved, kind=name, degree=3, order=-1, polarization=1
    )
    expected = context.pullback(2 * value)
    for a, b in zip(actual, expected, strict=True):
        assert_allclose(a, b, rtol=2e-12, atol=2e-12)


@given(
    degree=st.integers(1, 12),
    seed=st.integers(0, 24),
    theta=st.floats(0.1, 3),
    phi=st.floats(-3, 3),
)
@settings(max_examples=35)
def test_harmonic_addition_theorem_and_wave_helicity(degree, seed, theta, phi):
    orders = np.arange(-degree, degree + 1)
    y = sp.sph_harm(orders, degree, phi, theta)
    assert_allclose(np.sum(np.abs(y) ** 2), (2 * degree + 1) / (4 * np.pi), rtol=2e-12)
    assert_allclose(
        y, oracle.sph_harm(orders, degree, phi, theta), rtol=2e-12, atol=2e-13
    )
    for component in ["X", "Y", "Z"]:
        h = getattr(sp, "vsh_" + component)(degree, orders, theta, phi)
        assert_allclose(
            np.sum(np.abs(h) ** 2), (2 * degree + 1) / (4 * np.pi), rtol=2e-12
        )
    order = seed % (2 * degree + 1) - degree
    for family in ["vsw_", "vsw_r", "vcw_", "vcw_r", "vpw_"]:
        args = arguments(family + "A", theta)
        margs = args[:4] if family.startswith("vcw") else args
        m = getattr(sp, family + "M")(*public_args(family + "M", margs, degree, order))
        n = getattr(sp, family + "N")(*public_args(family + "N", args, degree, order))
        for pol in [0, 1]:
            a = getattr(sp, family + "A")(
                *public_args(family + "A", args, degree, order, pol)
            )
            assert_allclose(
                a, (n + (2 * pol - 1) * m) / np.sqrt(2), rtol=2e-12, atol=2e-13
            )


@pytest.mark.parametrize("degree", [30, 60, 128])
def test_high_degree_normalized_harmonics_remain_finite(degree):
    orders = np.arange(-degree, degree + 1)
    for theta in [0.0, 1e-10, 0.4, 1.4, np.pi]:
        for component in ["X", "Y", "Z"]:
            values = getattr(sp, "vsh_" + component)(degree, orders, theta, 0.3)
            assert_allclose(
                np.sum(np.abs(values) ** 2), (2 * degree + 1) / (4 * np.pi), rtol=2e-11
            )
        y = sp.sph_harm(orders, degree, 0.3, theta)
        # SciPy's normalized implementation is independent of treams' unscaled
        # vector-harmonic formula, which can overflow factorial intermediates.
        from scipy.special import sph_harm_y

        assert_allclose(
            y, sph_harm_y(degree, orders, theta, 0.3), rtol=3e-10, atol=3e-12
        )


def test_scalar_harmonic_advect_broadcast_and_empty_wave_pullbacks():
    import advect
    import advect.numpy as anp

    from treams_rs import advect as ad

    theta = np.array([[0.7 + 0.1j], [1.3 - 0.2j]])
    phi = np.array([0.2, 0.4, 0.9])
    value, context = diff.sph_harm(theta, phi, degree=5, order=-2)
    assert_allclose(value, sp.sph_harm(-2, 5, phi, theta), rtol=3e-12, atol=3e-13)
    expected = context.pullback(2 * value)

    def objective(t, p):
        v = ad.sph_harm(t, p, degree=5, order=-2)
        return anp.real(anp.sum(anp.conj(v) * v))

    actual = advect.grad(objective, argnums=(0, 1))(theta, phi)
    assert_allclose(actual[0], expected[0], rtol=2e-12, atol=2e-12)
    assert_allclose(actual[1], expected[1].real, rtol=2e-12, atol=2e-12)
    for name in ["sph_harm", "vsh_X", "vsw_rA", "vcw_rA", "vpw_A"]:
        args = [0.2, np.empty(0)] if name == "sph_harm" else arguments(name, n=0)
        value, context = diff.vector_wave(
            *args, kind=name, degree=3, order=-1, polarization=1
        )
        assert value.shape == ((0,) if name == "sph_harm" else (0, 3))
        for gradient, arg in zip(context.pullback(value), args, strict=True):
            assert gradient.shape == np.shape(arg)
            assert_allclose(gradient, 0)


@pytest.mark.parametrize("name", ["vpw_M", "vpw_N", "vpw_A"])
def test_plane_varying_direction_and_polarization_strides(name):
    for complex_k in [False, True]:
        x = np.linspace(0.2, 0.7, 2048)
        kx = np.linspace(0.1, 0.4, 2048)
        if complex_k:
            kx = kx + 0.03j
        args = [kx, 0.3, 1.2, x, 0.4, 0.7]
        if name.endswith("A"):
            args.append(np.arange(2048) % 2)
        actual = getattr(sp, name)(*args)
        assert_allclose(actual, getattr(oracle, name)(*args), rtol=3e-13, atol=3e-14)
        args = [a[::-2] if isinstance(a, np.ndarray) else a for a in args]
        assert_allclose(
            getattr(sp, name)(*args),
            getattr(oracle, name)(*args),
            rtol=3e-13,
            atol=3e-14,
        )


@pytest.mark.parametrize("size", [128, 2048])
def test_constant_direction_recorded_plane_matches_per_point_pullbacks(size):
    args = arguments("vpw_A", n=size)
    value, context = diff.vector_wave(*args, kind="vpw_A", polarization=1)
    rng = np.random.default_rng(713)
    cot = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradient = context.pullback(cot)
    # Equal-valued arrays force independent per-point polarization evaluation.
    arrays = np.broadcast_arrays(*[np.asarray(v, complex) for v in args])
    expected_value, context = diff.vector_wave(*arrays, kind="vpw_A", polarization=1)
    expected = context.pullback(cot)
    assert_allclose(value, expected_value, rtol=3e-14, atol=3e-14)
    for actual, reference, arg in zip(gradient, expected, args, strict=True):
        assert_allclose(
            actual,
            reference if np.ndim(arg) else reference.sum(),
            rtol=2e-12,
            atol=2e-12,
        )
    # A zero cotangent needs no direction derivative on the axial gauge.
    v, context = diff.vector_wave(
        0, 0, 1.2, np.linspace(0.2, 0.9, size), 0.3, 0.4, kind="vpw_A", polarization=1
    )
    for g in context.pullback(np.zeros_like(v)):
        assert_allclose(g, 0)

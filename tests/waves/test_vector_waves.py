"""Vector-wave component conventions, gufunc contracts and owned analytic VJPs."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams.special as oracle
from hypothesis import example, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import sph_harm_y

from treams_rs import advect as ad
from treams_rs import diff, pw, special
from treams_rs.testing import check_pullback

from _support import (
    assert_reusable_context,
    complex_normal,
    degree_order,
    selecting,
    sum_to,
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


@pytest.mark.reference
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("theta", [0.0, 0.8, np.pi, -0.7, 3.7, 0.7 + 0.3j, -0.7 + 0.3j])
def test_public_wave_reference(name, theta):
    args = public_args(name, arguments(name, theta))
    actual = getattr(special, name)(*args)
    expected = getattr(oracle, name)(*args)
    assert actual.shape == (3,)
    assert actual.dtype == np.complex128
    assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)


@pytest.mark.interface
@pytest.mark.parametrize("name", ["vpw_M", "vpw_N", "vpw_A"])
def test_scalar_plane_wave_fast_path_and_output(name):
    args = public_args(name, arguments(name))
    value = getattr(special, name)(*args)
    out = np.empty(3, complex)
    assert getattr(special, name)(*args, out) is out
    assert_allclose(out, value, rtol=2e-15)
    if name == "vpw_A":
        for pol in (-1, 2):
            with pytest.raises(ValueError, match="polarization"):
                special.vpw_A(*args[:-1], pol)


@pytest.mark.interface
@pytest.mark.reference
@pytest.mark.parametrize("name", NAMES)
def test_vector_wave_gufunc_reference_in_parallel(name):
    args = public_args(name, arguments(name, n=2048))
    actual = getattr(special, name)(*args)
    assert_allclose(actual, getattr(oracle, name)(*args), rtol=3e-12, atol=3e-13)
    empty = public_args(name, arguments(name, n=0))
    assert getattr(special, name)(*empty).shape == (0, 3)


def _phase_factors(name, args, order=-1):
    """Map argument positions entering as exp(i c x) to their factors i c."""
    if name.startswith("vsh"):
        return {1: 1j * order}
    if name.startswith("vsw"):
        return {2: 1j * order}
    if name.startswith("vcw"):
        return {2: 1j * order, 3: 1j * args[0]}
    return {3 + axis: 1j * args[axis] for axis in range(3)}


@pytest.mark.gradients
@pytest.mark.parametrize("name", NAMES)
def test_wave_pullbacks_all_arguments_ownership_broadcast_and_advect(name):
    args = [np.asarray(v, complex) for v in arguments(name, n=3)]
    args[0] = np.broadcast_to(args[0], (2, 1)).copy()
    rng = np.random.default_rng(43)
    directions = [complex_normal(rng, v.shape) for v in args]

    def record(*values):
        return diff.vector_wave(
            *values, function=name, degree=3, order=-1, polarization=1
        )

    # The recorded boundary analytically continues every argument to complex.
    value, ctx = record(*args)
    assert_allclose(
        value,
        getattr(special, name)(*public_args(name, arguments(name, n=3)))[None]
        + np.zeros((2, 1, 1)),
        rtol=3e-12,
        atol=3e-13,
    )
    cot = complex_normal(rng, value.shape)
    saved = [a.copy() for a in args]
    for a in args:
        a[...] += 0.1
    # The context owns its inputs and survives the rejected cotangents: it
    # pulls back what a fresh record at the saved inputs, checked below, does.
    grads = assert_reusable_context(
        ctx, cot, record(*saved)[1].pullback(cot), rtol=1e-13, atol=1e-15
    )
    assert isinstance(grads, tuple)
    for a, g in zip(saved, grads, strict=True):
        assert g.shape == a.shape
    # Azimuths and positions enter only through exp(i m phi), exp(i kz z) or
    # exp(i k.r): their gradients are exact.
    for i, factor in _phase_factors(name, saved).items():
        derivative = np.asarray(factor)[..., None] * value
        expected = sum_to(np.sum(cot * derivative.conj(), axis=-1), saved[i].shape)
        assert_allclose(grads[i], expected, rtol=1e-13, atol=1e-15)
    # The context returns a list; selecting every gradient makes it a tuple.
    check_pullback(
        selecting(record, *range(len(saved))),
        *saved,
        directions=tuple(directions),
        cotangents=cot,
        step=2e-6,
        rtol=3e-7,
        atol=3e-9,
    )

    def objective(*values):
        v = ad.vector_wave(*values, function=name, degree=3, order=-1, polarization=1)
        return anp.real(anp.sum(anp.conj(v) * v))

    actual = advect.grad(objective, argnums=tuple(range(len(args))))(*saved)
    value, context = record(*saved)
    expected = context.pullback(2 * value)
    for a, b in zip(actual, expected, strict=True):
        assert_allclose(a, b, rtol=2e-12, atol=2e-12)


@pytest.mark.gradients
@pytest.mark.parametrize("name", ["sph_harm", "vsh_Y", "vsw_rA", "vcw_N", "vpw_M"])
def test_wave_pullbacks_follow_the_public_tuple_contract(name):
    # A tuple holds one gradient per argument; the public checker, jax.wrap and
    # torch.wrap all read a list as one gradient.
    args = [0.8, np.array([0.2, 0.5])] if name == "sph_harm" else arguments(name, n=2)
    check_pullback(
        lambda *values: diff.vector_wave(
            *values, function=name, degree=3, order=-1, polarization=1
        ),
        *args,
    )


def _recorded(call, cotangent):
    """Value and gradients of a recorded call, or its ValueError message."""
    try:
        value, context = call()
        return value, context.pullback(np.full(value.shape, cotangent))
    except ValueError as error:
        return str(error)


NUMBER = st.one_of(
    st.floats(0.2, 1.5),
    st.builds(complex, st.floats(0.2, 1.5), st.floats(-0.3, 0.3)),
    st.integers(1, 2),
)


@pytest.mark.interface
@settings(max_examples=60)
@given(
    name=st.sampled_from([*NAMES, "sph_harm"]),
    numbers=st.lists(NUMBER, min_size=6, max_size=6),
    degree=st.sampled_from([3, np.int64(2), np.array([2, 3]), 3.0]),
    polarization=st.integers(0, 1),
)
def test_python_number_waves_equal_0d_array_waves(name, numbers, degree, polarization):
    # Python-number arguments share one NumPy conversion; values, gradients
    # (reduced to 0-d) and errors equal those of the same values as 0-d arrays.
    args = numbers[: 2 if name == "sph_harm" else len(arguments(name))]
    labels = {"degree": degree, "order": 1, "polarization": polarization}
    actual = _recorded(lambda: diff.vector_wave(*args, function=name, **labels), 0.3j)
    expected = _recorded(
        lambda: diff.vector_wave(*map(np.asarray, args), function=name, **labels), 0.3j
    )
    if isinstance(expected, str):
        assert actual == expected
        return
    assert_array_equal(actual[0], expected[0])
    assert [g.shape for g in actual[1]] == [()] * len(args)
    for gradient, reference in zip(actual[1], expected[1], strict=True):
        assert_array_equal(gradient, reference)


@pytest.mark.interface
@pytest.mark.parametrize("wrap", [int, np.asarray, lambda v: np.array([1, v])])
def test_label_bounds_do_not_depend_on_the_label_container(wrap):
    # Python ints, 0-d and broadcast arrays take the same validated path.
    calls = [
        (lambda v: diff.vector_wave(1.0, 0.2, 0.3, function="vsw_rA", degree=v), 129),
        (lambda v: diff.wignerd(v, 0, 0, 0.1, 0.2, 0.3), 261),
        (
            lambda v: diff.spherical_translation(
                1.0, 0.2, 0.3, destination=(v, 0, 0), source=(1, 0, 0)
            ),
            129,
        ),
        (lambda v: diff.cylindrical_translation(1.0, 0.2, 0.3, 0.4, order=v), 257),
        (lambda v: diff.intkambe(v, 0.5, 0.3), 261),
    ]
    for call, invalid in calls:
        with pytest.raises(ValueError, match=r"must be integers in \[-"):
            call(wrap(invalid))
    # Rust bounds only the Wigner degree; an order beyond it is an exact zero.
    for label in (200, -260):
        for labels in ((wrap(label), 0), (1, wrap(label))):
            value, _ = diff.wignerd(3, *labels, 0.1, 0.2, 0.3)
            assert np.ravel(value)[-1] == 0


@pytest.mark.physics
@pytest.mark.reference
@given(
    label=degree_order(1, 12),
    theta=st.floats(0.1, 3),
    phi=st.floats(-3, 3),
    other=st.tuples(st.floats(0, np.pi), st.floats(-3, 3)),
)
@example(label=(5, -2), theta=0.0, phi=0.3, other=(0.4, 2.0))
@example(label=(12, 12), theta=np.pi, phi=-1.0, other=(np.pi, 1.0))
@example(label=(1, 1), theta=1e-12, phi=2.0, other=(0.0, 0.0))
@settings(max_examples=35)
def test_harmonic_addition_theorem_and_wave_helicity(label, theta, phi, other):
    degree, order = label
    orders = np.arange(-degree, degree + 1)
    y = special.sph_harm(orders, degree, phi, theta)
    assert_allclose(np.sum(np.abs(y) ** 2), (2 * degree + 1) / (4 * np.pi), rtol=2e-12)
    assert_allclose(
        y, oracle.sph_harm(orders, degree, phi, theta), rtol=2e-12, atol=2e-13
    )
    # Conjugation symmetry, and the addition theorem for two directions at an
    # angle gamma: sum_m Y_m(1) conj(Y_m(2)) = (2l + 1) / (4 pi) P_l(cos gamma).
    assert_allclose(y[::-1], (-1.0) ** orders * y.conj(), rtol=2e-12, atol=2e-13)
    second = special.sph_harm(orders, degree, other[1], other[0])
    cosine = np.cos(theta) * np.cos(other[0]) + np.sin(theta) * np.sin(
        other[0]
    ) * np.cos(phi - other[1])
    assert_allclose(
        np.vdot(second, y),
        (2 * degree + 1)
        / (4 * np.pi)
        * special.lpmv(0, degree, np.clip(cosine, -1, 1)),
        rtol=0,
        atol=2e-13 * (2 * degree + 1),
    )
    for component in ["X", "Y", "Z"]:
        h = getattr(special, "vsh_" + component)(degree, orders, theta, phi)
        assert_allclose(
            np.sum(np.abs(h) ** 2), (2 * degree + 1) / (4 * np.pi), rtol=2e-12
        )
    for family in ["vsw_", "vsw_r", "vcw_", "vcw_r", "vpw_"]:
        args = arguments(family + "A", theta)
        margs = args[:4] if family.startswith("vcw") else args
        m = getattr(special, family + "M")(
            *public_args(family + "M", margs, degree, order)
        )
        n = getattr(special, family + "N")(
            *public_args(family + "N", args, degree, order)
        )
        for pol in [0, 1]:
            a = getattr(special, family + "A")(
                *public_args(family + "A", args, degree, order, pol)
            )
            assert_allclose(
                a, (n + (2 * pol - 1) * m) / np.sqrt(2), rtol=2e-12, atol=2e-13
            )


@pytest.mark.physics
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(
    direction=st.tuples(st.floats(0, np.pi), st.floats(-np.pi, np.pi)),
    point=st.tuples(st.floats(0, 2), st.floats(0, np.pi), st.floats(-np.pi, np.pi)),
    k=st.floats(0.5, 1.5),
    pol=st.integers(0, 1),
)
@settings(max_examples=20)
def test_plane_wave_is_the_sum_of_its_spherical_expansion(
    poltype, direction, point, k, pol
):
    # A plane wave equals the regular spherical waves weighted by pw.to_sw,
    # which ties the plane, spherical and coordinate ufuncs together. Up to
    # degree 20 the remainder is below 1e-17 for k |r| <= 2.
    vector = special.sph2car([k, *direction])
    position = special.sph2car([point[0] / k, *point[1:]])
    radius, theta, phi = spherical = special.car2sph(position)
    degrees = np.concatenate([np.full(2 * n + 1, n) for n in range(1, 21)])
    orders = np.concatenate([np.arange(-n, n + 1) for n in range(1, 21)])
    args = (degrees, orders, k * radius, theta, phi)
    if poltype == "helicity":
        waves = [special.vsw_rA(*args, q) for q in (0, 1)]
        expected = special.vpw_A(*vector, *position, pol)
    else:
        waves = [special.vsw_rM(*args), special.vsw_rN(*args)]
        expected = (special.vpw_N if pol else special.vpw_M)(*vector, *position)
    total = sum(
        pw.to_sw(degrees, orders, q, *vector, pol, poltype=poltype) @ wave
        for q, wave in enumerate(waves)
    )
    assert_allclose(special.vsph2car(total, spherical), expected, rtol=0, atol=1e-13)


@pytest.mark.reference
@pytest.mark.parametrize("degree", [30, 60, 128])
def test_high_degree_normalized_harmonics_remain_finite(degree):
    orders = np.arange(-degree, degree + 1)
    for theta in [0.0, 1e-10, 0.4, 1.4, np.pi]:
        y = special.sph_harm(orders, degree, 0.3, theta)
        # SciPy's normalized implementation is independent of treams' unscaled
        # vector-harmonic formula, which can overflow factorial intermediates.
        assert_allclose(
            y, sph_harm_y(degree, orders, theta, 0.3), rtol=3e-10, atol=3e-12
        )


@pytest.mark.gradients
@pytest.mark.interface
def test_scalar_harmonic_advect_broadcast_and_empty_wave_pullbacks():
    theta = np.array([[0.7 + 0.1j], [1.3 - 0.2j]])
    phi = np.array([0.2, 0.4, 0.9])
    value, context = diff.sph_harm(theta, phi, degree=5, order=-2)
    assert_allclose(value, special.sph_harm(-2, 5, phi, theta), rtol=3e-12, atol=3e-13)
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
            *args, function=name, degree=3, order=-1, polarization=1
        )
        assert value.shape == ((0,) if name == "sph_harm" else (0, 3))
        for gradient, arg in zip(context.pullback(value), args, strict=True):
            assert gradient.shape == np.shape(arg)
            assert_allclose(gradient, 0)


@pytest.mark.reference
@pytest.mark.parametrize("name", ["vpw_M", "vpw_N", "vpw_A"])
def test_plane_varying_direction_and_polarization_reference(name):
    for complex_k in [False, True]:
        x = np.linspace(0.2, 0.7, 2048)
        kx = np.linspace(0.1, 0.4, 2048)
        if complex_k:
            kx = kx + 0.03j
        args = [kx, 0.3, 1.2, x, 0.4, 0.7]
        if name.endswith("A"):
            args.append(np.arange(2048) % 2)
        actual = getattr(special, name)(*args)
        assert_allclose(actual, getattr(oracle, name)(*args), rtol=3e-13, atol=3e-14)


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize("size", [128, 2048])
def test_constant_direction_recorded_plane_matches_per_point_pullbacks(size):
    args = arguments("vpw_A", n=size)
    value, context = diff.vector_wave(*args, function="vpw_A", polarization=1)
    rng = np.random.default_rng(713)
    cot = complex_normal(rng, value.shape)
    gradient = context.pullback(cot)
    # Equal-valued arrays force independent per-point polarization evaluation.
    arrays = np.broadcast_arrays(*[np.asarray(v, complex) for v in args])
    expected_value, context = diff.vector_wave(
        *arrays, function="vpw_A", polarization=1
    )
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
        0,
        0,
        1.2,
        np.linspace(0.2, 0.9, size),
        0.3,
        0.4,
        function="vpw_A",
        polarization=1,
    )
    for g in context.pullback(np.zeros_like(v)):
        assert_allclose(g, 0)

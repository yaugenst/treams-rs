"""Direct translation coefficients, independent matrix kernels and angular VJPs.

Checks the special-namespace coefficients (tl_vsw_*, tl_vcw) and their diff records;
sw.translate is checked in test_wave_namespaces, and
_native.cartesian_translation_jet and the operators in test_translation.
"""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams.special as upstream_special
import treams.sw as upstream_sw
from hypothesis import example, given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import advect as ad
from treams_rs import diff, special
from treams_rs.testing import check_pullback

from _support import assert_one_use_context, complex_normal, degree_order, sum_to


@pytest.mark.reference
@pytest.mark.parametrize("name", ["tl_vsw_A", "tl_vsw_B", "tl_vsw_rA", "tl_vsw_rB"])
@pytest.mark.parametrize("theta", [0.0, 0.7, np.pi, -0.7, 0.7 + 0.2j])
def test_spherical_translation_reference(name, theta):
    labels = np.array(
        [
            (degree, order)
            for degree in range(1, 6)
            for order in range(-degree, degree + 1)
        ]
    )
    ell, m = labels.T
    args = (ell[:, None], m[:, None], ell, m, 1.4 + 0.2j, theta, 0.4)
    expected = getattr(upstream_special, name)(*args)
    actual = getattr(special, name)(*args)
    assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)


@pytest.mark.physics
@given(label=degree_order(1, 6), theta=st.floats(0.1, 3), phi=st.floats(-3, 3))
@example(label=(3, 2), theta=0.0, phi=-0.4)
@example(label=(6, 6), theta=np.pi, phi=0.3)
def test_regular_translation_origin_identity_and_azimuth(label, theta, phi):
    degree, order = label
    assert_allclose(
        special.tl_vsw_rA(degree, order, degree, order, 0, theta, phi), 1, rtol=2e-12
    )
    assert_allclose(
        special.tl_vsw_rB(degree, order, degree, order, 0, theta, phi), 0, atol=1e-14
    )
    for name in ["tl_vsw_A", "tl_vsw_B", "tl_vsw_rA", "tl_vsw_rB"]:
        a = getattr(special, name)(degree, order, 3, -1, 1.3 + 0.2j, theta, phi)
        b = getattr(special, name)(degree, order, 3, -1, 1.3 + 0.2j, theta, phi + 0.2)
        assert_allclose(b, a * np.exp(1j * (-1 - order) * 0.2), rtol=3e-12, atol=3e-12)


@pytest.mark.physics
@pytest.mark.reference
def test_tiny_angle_translation_against_closed_dipole_form():
    z = 1.4 + 0.2j
    h1 = -np.exp(1j * z) * (z + 1j) / z**2
    h2 = np.exp(1j * z) * (1j / z - 3 / z**2 - 3j / z**3)
    j1 = np.sin(z) / z**2 - np.cos(z) / z
    j2 = (3 / z**3 - 1 / z) * np.sin(z) - 3 * np.cos(z) / z**2
    for theta in [1e-8, 1e-10, 1e-100]:
        phase = np.exp(0.4j) * np.sin(theta)
        for regular, r1, r2 in [(False, h1, h2), (True, j1, j2)]:
            prefix = "tl_vsw_r" if regular else "tl_vsw_"
            expected_a = -3 / (2 * np.sqrt(2)) * r2 * np.cos(theta) * phase
            expected_b = 3j / (2 * np.sqrt(2)) * r1 * phase
            assert_allclose(
                getattr(special, prefix + "A")(1, 0, 1, 1, z, theta, 0.4),
                expected_a,
                rtol=3e-13,
                atol=0,
            )
            assert_allclose(
                getattr(special, prefix + "B")(1, 0, 1, 1, z, theta, 0.4),
                expected_b,
                rtol=3e-13,
                atol=0,
            )
    # Upstream forms sqrt(1-cos(theta)**2), losing this nonzero coupling.
    assert upstream_special.tl_vsw_A(1, 0, 1, 1, z, 1e-10, 0.4) == 0
    assert upstream_special.tl_vsw_B(1, 0, 1, 1, z, 1e-10, 0.4) == 0


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_polar_translation_owned_broadcast_pullback_and_advect(singular, poltype):
    destination = (2, np.array([-1, 0, 1])[:, None], np.array([0, 1, 0])[:, None])
    source = (3, np.array([-2, 0, 2]), np.array([1, 0, 1]))
    kwargs = dict(
        destination=destination, source=source, poltype=poltype, singular=singular
    )

    def record(*values):
        return diff.spherical_translation(*values, **kwargs)

    args = [
        np.array(1.4 + 0.2j),
        np.array([[0.4], [0.7], [1.2]]),
        np.array([0.2, 0.3, 0.5]),
    ]
    saved = [a.copy() for a in args]
    value, context = record(*args)
    assert_allclose(
        value,
        upstream_sw.translate(
            *destination, *source, *args, poltype=poltype, singular=singular
        ),
        rtol=3e-11,
        atol=3e-12,
    )
    rng = np.random.default_rng(223)
    g = complex_normal(rng, value.shape)
    complex_saved = [a.astype(complex) for a in saved]
    for a in args:
        a[...] += 0.1
    # The context owns its real inputs and survives the rejected cotangents: it
    # pulls back what the fresh complex record, checked below, does.
    gradients = assert_one_use_context(
        context,
        np.asfortranarray(g),
        record(*complex_saved)[1].pullback(g),
        rtol=1e-13,
        atol=1e-15,
    )
    # The azimuth enters only through exp(i (m_source - m_destination) phi).
    derivative = 1j * (source[1] - destination[1]) * value
    expected = sum_to(g * derivative.conj(), saved[2].shape)
    assert_allclose(
        gradients[2], expected, rtol=1e-13, atol=1e-14 * abs(expected).max()
    )
    for gradient, argument in zip(gradients, saved, strict=True):
        assert gradient.shape == argument.shape
    # Complex directions probe the analytic continuation of the real angles.
    check_pullback(
        record,
        *complex_saved,
        directions=tuple(complex_normal(rng, a.shape) for a in saved),
        cotangents=g,
        step=2e-6,
        rtol=4e-7,
        atol=3e-9,
    )

    def objective(*values):
        v = ad.spherical_translation(*values, **kwargs)
        return anp.real(anp.sum(anp.conj(v) * v))

    actual = advect.grad(objective, argnums=(0, 1, 2))(*saved)
    v, ctx = record(*saved)
    expected = ctx.pullback(2 * v)
    for a, b, original in zip(actual, expected, saved, strict=True):
        assert_allclose(
            a, b if np.iscomplexobj(original) else b.real, rtol=3e-12, atol=3e-12
        )


@pytest.mark.gradients
@pytest.mark.interface
def test_polar_fixed_plan_parallel_batch_and_empty_reduction():
    kwargs = dict(
        destination=(3, -1, 0), source=(4, 1, 1), poltype="parity", singular=False
    )
    theta = np.linspace(0.2, 1.2, 2048)
    value, context = diff.spherical_translation(1.3 + 0.1j, theta, 0.3, **kwargs)
    assert_allclose(
        value,
        special.tl_vsw_rB(3, -1, 4, 1, 1.3 + 0.1j, theta, 0.3),
        rtol=3e-12,
        atol=3e-13,
    )
    gradient = context.pullback(np.ones_like(value))
    split = [
        diff.spherical_translation(1.3 + 0.1j, t, 0.3, **kwargs)
        for t in np.array_split(theta, 4)
    ]
    expected = [ctx.pullback(np.ones_like(v)) for v, ctx in split]
    for i in [0, 2]:
        assert_allclose(
            gradient[i], sum(g[i] for g in expected), rtol=3e-12, atol=3e-12
        )
    assert_allclose(
        gradient[1], np.concatenate([g[1] for g in expected]), rtol=3e-12, atol=3e-12
    )
    value, context = diff.spherical_translation(1.3, [], 0.3, **kwargs)
    assert value.shape == (0,)
    for g in context.pullback(value):
        assert_allclose(g, 0)


@pytest.mark.reference
@pytest.mark.parametrize("regular", [False, True])
@pytest.mark.parametrize("argument", [1.4, 1.4 + 0.2j, -1.4 + 0.2j])
def test_cylindrical_translation_reference(regular, argument):
    name = "tl_vcw_r" if regular else "tl_vcw"
    mu = np.arange(-6, 7)[:, None]
    m = np.arange(-5, 6)
    kz = np.where(mu % 2, 0.2, -0.3)
    qz = np.where(m % 2, 0.2, -0.3)
    args = (kz, mu, qz, m, argument, 0.4, 0.3)
    value = getattr(special, name)(*args)
    assert_allclose(
        value, getattr(upstream_special, name)(*args), rtol=4e-12, atol=4e-12
    )
    assert np.all(value[np.broadcast_to(kz != qz, value.shape)] == 0)


@pytest.mark.physics
@given(
    m=st.integers(-4, 4),
    mu=st.integers(-4, 4),
    a=st.floats(0.1, 1.0),
    b=st.floats(0.1, 1.0),
    phi=st.tuples(st.floats(-3, 3), st.floats(-3, 3)),
    z=st.tuples(st.floats(-0.5, 0.5), st.floats(-0.5, 0.5)),
)
@example(m=2, mu=-1, a=0.4, b=0.7, phi=(0.3, 0.3), z=(0.3, -0.1))
def test_cylindrical_translation_group_composition(m, mu, a, b, phi, z):
    # Graf's addition theorem for two translations in independent directions.
    # A collinear pair only checks that the azimuthal phases add; this one ties
    # them to the geometric angle of the combined displacement.
    q = np.arange(-24, 25)
    left = special.tl_vcw_r(0.2, mu, 0.2, q, a, phi[0], z[0])
    right = special.tl_vcw_r(0.2, q, 0.2, m, b, phi[1], z[1])
    total = a * np.exp(1j * phi[0]) + b * np.exp(1j * phi[1])
    expected = special.tl_vcw_r(0.2, mu, 0.2, m, abs(total), np.angle(total), sum(z))
    assert_allclose(np.sum(left * right), expected, rtol=5e-12, atol=2e-14)


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize("singular", [False, True])
def test_cylindrical_polar_ownership_broadcast_adjoints_and_advect(singular):
    order = np.array([-2, 0, 1])[:, None]
    args = [
        np.array(1.4 + 0.2j),
        np.array([[0.3], [0.7], [1.1]]),
        np.linspace(0.1, 0.5, 4),
        np.array(0.2),
    ]
    saved = [a.copy() for a in args]
    kw = dict(order=order, singular=singular)

    def record(*values):
        return diff.cylindrical_translation(*values, **kw)

    value, context = record(*args)
    name = "tl_vcw" if singular else "tl_vcw_r"
    assert_allclose(
        value,
        getattr(upstream_special, name)(args[3], 0, args[3], order, *args[:3]),
        rtol=4e-12,
        atol=4e-12,
    )
    rng = np.random.default_rng(290)
    g = complex_normal(rng, value.shape)
    complex_saved = [a.astype(complex) for a in saved]
    for a in args:
        a[...] += 0.3
    # The context owns its real inputs and survives the rejected cotangents: it
    # pulls back what the fresh complex record, checked below, does.
    gradients = assert_one_use_context(
        context, g, record(*complex_saved)[1].pullback(g), rtol=1e-13, atol=1e-15
    )
    # The azimuth and the axial offset enter only through exp(i m phi + i kz z).
    for i, factor in [(1, 1j * order), (2, 1j * saved[3])]:
        expected = sum_to(g * (factor * value).conj(), saved[i].shape)
        assert_allclose(gradients[i], expected, rtol=1e-13, atol=1e-15)
    for gradient, argument in zip(gradients, saved, strict=True):
        assert gradient.shape == argument.shape
    check_pullback(
        record,
        *complex_saved,
        directions=tuple(complex_normal(rng, a.shape) for a in saved),
        cotangents=g,
        step=2e-6,
        rtol=5e-7,
        atol=5e-9,
    )

    def objective(*args):
        value = ad.cylindrical_translation(*args, **kw)
        return anp.real(anp.sum(anp.conj(value) * value))

    actual = advect.grad(objective, argnums=(0, 1, 2, 3))(*saved)
    value, context = record(*saved)
    expected = context.pullback(2 * value)
    for a, b, original in zip(actual, expected, saved, strict=True):
        assert_allclose(
            a, b if np.iscomplexobj(original) else b.real, rtol=3e-12, atol=3e-12
        )


@pytest.mark.gradients
def test_cylindrical_polar_parallel_and_origin_gradients():
    kr = np.linspace(0.5, 2, 2048) + 0.1j
    value, context = diff.cylindrical_translation(
        kr, 0.3, 0.4, 0.2, order=-2, singular=False
    )
    assert_allclose(
        value, special.tl_vcw_r(0.2, 0, 0.2, -2, kr, 0.3, 0.4), rtol=3e-12, atol=3e-12
    )
    g = context.pullback(np.ones_like(value))
    split = [
        diff.cylindrical_translation(k, 0.3, 0.4, 0.2, order=-2, singular=False)
        for k in np.array_split(kr, 4)
    ]
    gradients = [ctx.pullback(np.ones_like(v)) for v, ctx in split]
    assert_allclose(
        g[0], np.concatenate([v[0] for v in gradients]), rtol=3e-12, atol=3e-12
    )
    for i in [1, 2, 3]:
        assert_allclose(g[i], sum(v[i] for v in gradients), rtol=3e-12, atol=3e-12)
    for order in [-1, 0, 1, 2]:
        value, context = diff.cylindrical_translation(
            0, 0, 0, 0.2, order=order, singular=False
        )
        assert_allclose(value, int(order == 0))
        assert_allclose(
            context.pullback(np.array(1 + 0j))[0], 0.5 * order if abs(order) == 1 else 0
        )
    value, context = diff.cylindrical_translation([], 0.3, 0.4, 0.2, order=0)
    assert value.size == 0
    for g in context.pullback(value):
        assert_allclose(g, 0)

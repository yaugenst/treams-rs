"""Direct translation coefficients, independent matrix kernels and angular VJPs."""

import numpy as np
import pytest
import treams.special as oracle
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import special as sp


@pytest.mark.parametrize("name", ["tl_vsw_A", "tl_vsw_B", "tl_vsw_rA", "tl_vsw_rB"])
@pytest.mark.parametrize("theta", [0.0, 0.7, np.pi, -0.7, 0.7 + 0.2j])
def test_spherical_translation_reference_and_strides(name, theta):
    labels = np.array(
        [
            (degree, order)
            for degree in range(1, 6)
            for order in range(-degree, degree + 1)
        ]
    )
    ell, m = labels.T
    args = (ell[:, None], m[:, None], ell, m, 1.4 + 0.2j, theta, 0.4)
    expected = getattr(oracle, name)(*args)
    actual = getattr(sp, name)(*args)
    assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
    out = np.empty_like(actual).T
    assert getattr(sp, name)(*args, out=out) is out
    assert_allclose(out, expected, rtol=2e-10, atol=2e-11)
    out = np.full_like(actual, 7)
    mask = np.tril(np.ones(actual.shape, bool))
    getattr(sp, name)(*args, out=out, where=mask)
    assert_allclose(out[mask], expected[mask], rtol=2e-10, atol=2e-11)
    assert np.all(out[~mask] == 7)


@given(
    degree=st.integers(1, 6),
    seed=st.integers(0, 12),
    theta=st.floats(0.1, 3),
    phi=st.floats(-3, 3),
)
@settings(max_examples=30)
def test_regular_translation_origin_identity_and_azimuth(degree, seed, theta, phi):
    order = seed % (2 * degree + 1) - degree
    assert_allclose(
        sp.tl_vsw_rA(degree, order, degree, order, 0, theta, phi), 1, rtol=2e-12
    )
    assert_allclose(
        sp.tl_vsw_rB(degree, order, degree, order, 0, theta, phi), 0, atol=1e-14
    )
    for name in ["tl_vsw_A", "tl_vsw_B", "tl_vsw_rA", "tl_vsw_rB"]:
        a = getattr(sp, name)(degree, order, 3, -1, 1.3 + 0.2j, theta, phi)
        b = getattr(sp, name)(degree, order, 3, -1, 1.3 + 0.2j, theta, phi + 0.2)
        assert_allclose(b, a * np.exp(1j * (-1 - order) * 0.2), rtol=3e-12, atol=3e-12)


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
                getattr(sp, prefix + "A")(1, 0, 1, 1, z, theta, 0.4),
                expected_a,
                rtol=3e-13,
                atol=0,
            )
            assert_allclose(
                getattr(sp, prefix + "B")(1, 0, 1, 1, z, theta, 0.4),
                expected_b,
                rtol=3e-13,
                atol=0,
            )
    # Upstream forms sqrt(1-cos(theta)**2), losing this nonzero coupling.
    assert oracle.tl_vsw_A(1, 0, 1, 1, z, 1e-10, 0.4) == 0
    assert oracle.tl_vsw_B(1, 0, 1, 1, z, 1e-10, 0.4) == 0


@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_polar_translation_owned_broadcast_pullback_and_advect(singular, poltype):
    import advect
    import advect.numpy as anp
    import treams.sw as upstream

    from treams_rs import advect as ad
    from treams_rs import diff

    destination = (2, np.array([-1, 0, 1])[:, None], np.array([0, 1, 0])[:, None])
    source = (3, np.array([-2, 0, 2]), np.array([1, 0, 1]))
    kwargs = dict(
        destination=destination, source=source, poltype=poltype, singular=singular
    )
    args = [
        np.array(1.4 + 0.2j),
        np.array([[0.4], [0.7], [1.2]]),
        np.array([0.2, 0.3, 0.5]),
    ]
    saved = [a.copy() for a in args]
    value, context = diff.spherical_translation(*args, **kwargs)
    assert_allclose(
        value,
        upstream.translate(
            *destination, *source, *args, poltype=poltype, singular=singular
        ),
        rtol=3e-11,
        atol=3e-12,
    )
    rng = np.random.default_rng(223)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    with pytest.raises(ValueError, match="shape"):
        context.pullback(g[:-1])
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full(value.shape, np.nan, complex))
    for a in args:
        a[...] += 0.1
    gradients = context.pullback(np.asfortranarray(g))
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)
    step = 2e-6
    for i, gradient in enumerate(gradients):
        assert gradient.shape == saved[i].shape
        direction = rng.normal(size=saved[i].shape) + 1j * rng.normal(
            size=saved[i].shape
        )
        plus, minus = list(saved), list(saved)
        plus[i] = saved[i] + step * direction
        minus[i] = saved[i] - step * direction
        fd = (
            diff.spherical_translation(*plus, **kwargs)[0]
            - diff.spherical_translation(*minus, **kwargs)[0]
        ) / (2 * step)
        assert_allclose(
            np.vdot(gradient, direction).real, np.vdot(g, fd).real, rtol=4e-7, atol=3e-9
        )

    def objective(*values):
        v = ad.spherical_translation(*values, **kwargs)
        return anp.real(anp.sum(anp.conj(v) * v))

    actual = advect.grad(objective, argnums=(0, 1, 2))(*saved)
    v, ctx = diff.spherical_translation(*saved, **kwargs)
    expected = ctx.pullback(2 * v)
    for a, b, original in zip(actual, expected, saved, strict=True):
        assert_allclose(
            a, b if np.iscomplexobj(original) else b.real, rtol=3e-12, atol=3e-12
        )


def test_polar_fixed_plan_parallel_batch_and_empty_reduction():
    from treams_rs import diff

    kwargs = dict(
        destination=(3, -1, 0), source=(4, 1, 1), poltype="parity", singular=False
    )
    theta = np.linspace(0.2, 1.2, 2048)
    value, context = diff.spherical_translation(1.3 + 0.1j, theta, 0.3, **kwargs)
    assert_allclose(
        value, sp.tl_vsw_rB(3, -1, 4, 1, 1.3 + 0.1j, theta, 0.3), rtol=3e-12, atol=3e-13
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


@pytest.mark.parametrize("regular", [False, True])
@pytest.mark.parametrize("argument", [1.4, 1.4 + 0.2j, -1.4 + 0.2j])
def test_cylindrical_translation_reference_and_strided_outputs(regular, argument):
    name = "tl_vcw_r" if regular else "tl_vcw"
    mu = np.arange(-6, 7)[:, None]
    m = np.arange(-5, 6)
    kz = np.where(mu % 2, 0.2, -0.3)
    qz = np.where(m % 2, 0.2, -0.3)
    args = (kz, mu, qz, m, argument, 0.4, 0.3)
    value = getattr(sp, name)(*args)
    assert_allclose(value, getattr(oracle, name)(*args), rtol=4e-12, atol=4e-12)
    assert np.all(value[np.broadcast_to(kz != qz, value.shape)] == 0)
    storage = bytearray(value.nbytes + 1)
    out = np.ndarray(value.shape, dtype=complex, buffer=storage, offset=1)[::-1]
    assert getattr(sp, name)(*args, out=out) is out
    assert_allclose(out, value, rtol=2e-13)
    mask = np.indices(value.shape).sum(axis=0) % 2 == 0
    out[:] = 7
    getattr(sp, name)(*args, out=out, where=mask)
    assert_allclose(out[mask], value[mask], rtol=2e-13)
    assert np.all(out[~mask] == 7)


@given(
    m=st.integers(-4, 4),
    mu=st.integers(-4, 4),
    a=st.floats(0.1, 1.0),
    b=st.floats(0.1, 1.0),
    phi=st.floats(-3, 3),
)
@settings(max_examples=30)
def test_cylindrical_translation_group_composition(m, mu, a, b, phi):
    q = np.arange(-24, 25)
    left = sp.tl_vcw_r(0.2, mu, 0.2, q, a, phi, 0.3)
    right = sp.tl_vcw_r(0.2, q, 0.2, m, b, phi, -0.1)
    expected = sp.tl_vcw_r(0.2, mu, 0.2, m, a + b, phi, 0.2)
    assert_allclose(np.sum(left * right), expected, rtol=5e-12, atol=2e-14)


@pytest.mark.parametrize("singular", [False, True])
def test_cylindrical_polar_ownership_broadcast_adjoints_and_advect(singular):
    import advect
    import advect.numpy as anp

    from treams_rs import advect as ad
    from treams_rs import diff

    order = np.array([-2, 0, 1])[:, None]
    args = [
        np.array(1.4 + 0.2j),
        np.array([[0.3], [0.7], [1.1]]),
        np.linspace(0.1, 0.5, 4),
        np.array(0.2),
    ]
    saved = [a.copy() for a in args]
    kw = dict(order=order, singular=singular)
    value, context = diff.cylindrical_translation(*args, **kw)
    name = "tl_vcw" if singular else "tl_vcw_r"
    assert_allclose(
        value,
        getattr(oracle, name)(args[3], 0, args[3], order, *args[:3]),
        rtol=4e-12,
        atol=4e-12,
    )
    rng = np.random.default_rng(290)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    with pytest.raises(ValueError, match="shape"):
        context.pullback(g[:-1])
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full_like(g, np.nan))
    for a in args:
        a[...] += 0.3
    gradients = context.pullback(g)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)
    step = 2e-6
    for i, gradient in enumerate(gradients):
        assert gradient.shape == saved[i].shape
        d = rng.normal(size=saved[i].shape) + 1j * rng.normal(size=saved[i].shape)
        plus, minus = list(saved), list(saved)
        plus[i] = saved[i] + step * d
        minus[i] = saved[i] - step * d
        fd = (
            diff.cylindrical_translation(*plus, **kw)[0]
            - diff.cylindrical_translation(*minus, **kw)[0]
        ) / (2 * step)
        assert_allclose(
            np.vdot(gradient, d).real, np.vdot(g, fd).real, rtol=5e-7, atol=5e-9
        )

    def objective(*args):
        value = ad.cylindrical_translation(*args, **kw)
        return anp.real(anp.sum(anp.conj(value) * value))

    actual = advect.grad(objective, argnums=(0, 1, 2, 3))(*saved)
    value, context = diff.cylindrical_translation(*saved, **kw)
    expected = context.pullback(2 * value)
    for a, b, original in zip(actual, expected, saved, strict=True):
        assert_allclose(
            a, b if np.iscomplexobj(original) else b.real, rtol=3e-12, atol=3e-12
        )


def test_cylindrical_polar_parallel_and_origin_gradients():
    from treams_rs import diff

    kr = np.linspace(0.5, 2, 2048) + 0.1j
    value, context = diff.cylindrical_translation(
        kr, 0.3, 0.4, 0.2, order=-2, singular=False
    )
    assert_allclose(
        value, sp.tl_vcw_r(0.2, 0, 0.2, -2, kr, 0.3, 0.4), rtol=3e-12, atol=3e-12
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

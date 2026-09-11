"""Coordinate charts, vector frames and native NumPy generalized-ufunc semantics."""

import numpy as np
import pytest
import treams.special as oracle
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import special as sp

PAIRS = [
    ("car2cyl", "cyl2car"),
    ("car2sph", "sph2car"),
    ("cyl2sph", "sph2cyl"),
    ("car2pol", "pol2car"),
]
NAMES = [name for pair in PAIRS for name in pair]


@pytest.mark.parametrize("name", NAMES)
def test_coordinate_and_vector_reference_and_output_semantics(name):
    dim = 2 if "pol" in name else 3
    points = np.array([[0.4, -0.3, 0.2], [1.3, 0.7, -0.5], [0, 0, 1]])[:, :dim]
    expected = getattr(oracle, name)(points)
    actual = getattr(sp, name)(points)
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
    assert actual.dtype == np.float64
    # Core axis relocation, reversed component strides, and in-place output.
    assert_allclose(
        getattr(sp, name)(points.T, axis=0), expected.T, rtol=1e-14, atol=1e-15
    )
    assert_allclose(
        getattr(sp, name)(points[:, ::-1]),
        getattr(oracle, name)(points[:, ::-1]),
        rtol=1e-14,
        atol=1e-15,
    )
    inplace = points.copy()
    assert getattr(sp, name)(inplace, out=inplace) is inplace
    assert_allclose(inplace, expected, rtol=1e-14, atol=1e-15)
    unaligned = np.ndarray(
        points.shape, dtype=float, buffer=bytearray(points.nbytes + 1), offset=1
    )
    unaligned[:] = points
    getattr(sp, name)(unaligned, out=unaligned)
    assert_allclose(unaligned, expected, rtol=1e-14, atol=1e-15)
    for complex_values in [False, True]:
        vector = np.array([0.2, 0.3, -0.4])[:dim]
        if complex_values:
            vector = vector + 0.3j * vector[::-1]
        expected = getattr(oracle, "v" + name)(vector, points)
        actual = getattr(sp, "v" + name)(vector, points)
        assert actual.dtype == expected.dtype
        assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
        v = np.broadcast_to(vector, points.shape).copy()
        assert getattr(sp, "v" + name)(v, points, out=v) is v
        assert_allclose(v, expected, rtol=1e-14, atol=1e-15)
        v = np.broadcast_to(vector, points.shape).copy().T
        assert_allclose(
            getattr(sp, "v" + name)(v, points.T, axis=0),
            expected.T,
            rtol=1e-14,
            atol=1e-15,
        )
    assert getattr(sp, name)(np.empty((0, dim))).shape == (0, dim)
    with pytest.raises(ValueError):
        getattr(sp, name)([1] * (dim + 1))


@given(x=st.floats(0.2, 2), y=st.floats(-1, 1), z=st.floats(-1, 1))
@settings(max_examples=40)
def test_coordinate_roundtrip_vector_norm_and_frame_composition(x, y, z):
    cart = np.array([x, y, z])
    vector = np.array([0.2 + 0.3j, -0.4 + 0.1j, 0.7 - 0.2j])
    cylindrical = sp.car2cyl(cart)
    spherical = sp.car2sph(cart)
    assert_allclose(sp.cyl2car(cylindrical), cart, rtol=1e-13, atol=1e-15)
    assert_allclose(sp.sph2car(spherical), cart, rtol=1e-13, atol=1e-15)
    assert_allclose(sp.cyl2sph(cylindrical), spherical, rtol=1e-13, atol=1e-15)
    assert_allclose(sp.sph2cyl(spherical), cylindrical, rtol=1e-13, atol=1e-15)
    assert_allclose(sp.pol2car(sp.car2pol(cart[:2])), cart[:2], rtol=1e-13, atol=1e-15)
    vc = sp.vcar2cyl(vector, cart)
    vs = sp.vcar2sph(vector, cart)
    assert_allclose(sp.vcyl2car(vc, cylindrical), vector, rtol=1e-13, atol=1e-15)
    assert_allclose(sp.vsph2car(vs, spherical), vector, rtol=1e-13, atol=1e-15)
    assert_allclose(sp.vcyl2sph(vc, cylindrical), vs, rtol=1e-13, atol=1e-15)
    assert_allclose(sp.vsph2cyl(vs, spherical), vc, rtol=1e-13, atol=1e-15)
    assert_allclose(np.vdot(vc, vc), np.vdot(vector, vector), rtol=1e-13, atol=1e-15)
    assert_allclose(np.vdot(vs, vs), np.vdot(vector, vector), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("name", NAMES)
def test_coordinate_and_vector_pullbacks_broadcast_ownership_and_advect(name):
    import advect
    import advect.numpy as anp

    from treams_rs import advect as ad
    from treams_rs import diff

    dim = 2 if "pol" in name else 3
    points = np.array([[0.4, 0.7, 0.2], [1.3, 0.5, -0.5], [0.2, 0.3, 0.4]])[
        :, :dim
    ].copy()
    rng = np.random.default_rng(17)
    direction = rng.uniform(-0.2, 0.2, points.shape)
    point_g = rng.normal(size=points.shape)
    value, context = diff.coordinates(points, kind=name)
    assert_allclose(value, getattr(sp, name)(points), rtol=1e-14, atol=1e-15)
    expected = context.pullback(point_g)
    h = 1e-6
    difference = (
        getattr(sp, name)(points + h * direction)
        - getattr(sp, name)(points - h * direction)
    ) / (2 * h)
    assert_allclose(
        np.sum(expected * direction), np.sum(point_g * difference), rtol=2e-7, atol=1e-9
    )

    def objective(p):
        return anp.sum(ad.coordinates(p, kind=name) * point_g)

    assert_allclose(advect.grad(objective)(points), expected, rtol=1e-13, atol=1e-13)
    context = diff.coordinates(points, kind=name)[1]
    points_saved = points.copy()
    points[:] = 0
    with pytest.raises(ValueError, match="shape"):
        context.pullback(point_g[:-1])
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full_like(point_g, np.nan))
    assert_allclose(
        context.pullback(np.asfortranarray(point_g)), expected, rtol=1e-13, atol=1e-13
    )
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(point_g)
    points[:] = points_saved

    vectors = rng.normal(size=(2, 1, dim)) + 1j * rng.normal(size=(2, 1, dim))
    value, context = diff.vector_coordinates(vectors, points, kind=name)
    assert_allclose(
        value, getattr(sp, "v" + name)(vectors, points), rtol=1e-14, atol=1e-15
    )
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradient = context.pullback(g)
    assert gradient[0].shape == vectors.shape
    assert gradient[1].shape == points.shape
    vdir = rng.normal(size=vectors.shape) + 1j * rng.normal(size=vectors.shape)
    difference = (
        getattr(sp, "v" + name)(vectors + h * vdir, points + h * direction)
        - getattr(sp, "v" + name)(vectors - h * vdir, points - h * direction)
    ) / (2 * h)
    assert_allclose(
        np.vdot(gradient[0], vdir).real + np.sum(gradient[1] * direction),
        np.vdot(g, difference).real,
        rtol=2e-7,
        atol=1e-8,
    )

    def loss(v, p):
        return anp.real(anp.sum(anp.conj(g) * ad.vector_coordinates(v, p, kind=name)))

    for actual, expected in zip(
        advect.grad(loss, argnums=(0, 1))(vectors, points), gradient, strict=True
    ):
        assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    context = diff.vector_coordinates(vectors, points, kind=name)[1]
    vectors[:] = 0
    points[:] = 0
    with pytest.raises(ValueError, match="shape"):
        context.pullback(g[:-1])
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full_like(g, np.nan))
    for actual, expected in zip(
        context.pullback(np.asfortranarray(g)), gradient, strict=True
    ):
        assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


def test_coordinate_empty_and_axis_contexts():
    from treams_rs import diff

    for dim, kind in [(2, "car2pol"), (3, "car2sph")]:
        value, context = diff.coordinates(np.empty((0, dim)), kind=kind)
        assert context.pullback(value).shape == (0, dim)
        for v, p in [
            (np.ones(dim, complex), np.empty((0, dim))),
            (np.empty((0, dim), complex), np.ones(dim)),
        ]:
            value, context = diff.vector_coordinates(v, p, kind=kind)
            gv, gp = context.pullback(value)
            assert gv.shape == v.shape and gp.shape == p.shape
            assert_allclose(gv, 0)
            assert_allclose(gp, 0)
    _, context = diff.coordinates([[0, 0, -1]], kind="car2sph")
    assert_allclose(context.pullback(np.array([[1, 0, 0]], float)), [[0, 0, -1]])
    _, context = diff.coordinates([[0, 0, -1]], kind="car2sph")
    with pytest.raises(ValueError, match="undefined"):
        context.pullback(np.array([[0, 0, 1]], float))
    with pytest.raises(ValueError, match="dimension"):
        diff.vector_coordinates([1, 2], [1, 2, 3], kind="car2sph")

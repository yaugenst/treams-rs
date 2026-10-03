"""Coordinate charts, vector frames and native NumPy generalized-ufunc semantics."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams.special as oracle
from numpy.testing import assert_allclose

from treams_rs import advect as ad
from treams_rs import diff, special
from treams_rs.testing import check_pullback

from _support import assert_one_use_context, complex_normal, selecting

PAIRS = [
    ("car2cyl", "cyl2car"),
    ("car2sph", "sph2car"),
    ("cyl2sph", "sph2cyl"),
    ("car2pol", "pol2car"),
]
NAMES = [name for pair in PAIRS for name in pair]


@pytest.mark.interface
@pytest.mark.reference
@pytest.mark.parametrize("name", NAMES)
def test_coordinate_and_vector_reference_and_core_axes(name):
    dim = 2 if "pol" in name else 3
    points = np.array([[0.4, -0.3, 0.2], [1.3, 0.7, -0.5], [0, 0, 1]])[:, :dim]
    expected = getattr(oracle, name)(points)
    actual = getattr(special, name)(points)
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
    assert actual.dtype == np.float64
    # Core axis relocation and reversed component strides.
    assert_allclose(
        getattr(special, name)(points.T, axis=0), expected.T, rtol=1e-14, atol=1e-15
    )
    assert_allclose(
        getattr(special, name)(points[:, ::-1]),
        getattr(oracle, name)(points[:, ::-1]),
        rtol=1e-14,
        atol=1e-15,
    )
    for complex_values in [False, True]:
        vector = np.array([0.2, 0.3, -0.4])[:dim]
        if complex_values:
            vector = vector + 0.3j * vector[::-1]
        expected = getattr(oracle, "v" + name)(vector, points)
        actual = getattr(special, "v" + name)(vector, points)
        assert actual.dtype == expected.dtype
        assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
        v = np.broadcast_to(vector, points.shape).copy().T
        assert_allclose(
            getattr(special, "v" + name)(v, points.T, axis=0),
            expected.T,
            rtol=1e-14,
            atol=1e-15,
        )
    assert getattr(special, name)(np.empty((0, dim))).shape == (0, dim)
    with pytest.raises(ValueError):
        getattr(special, name)([1] * (dim + 1))


@pytest.mark.interface
@pytest.mark.reference
@pytest.mark.parametrize("name", NAMES)
def test_single_coordinate_fast_path_preserves_gufunc_semantics(name):
    dim = 2 if "pol" in name else 3
    points = np.array([0.4, 0.7, -0.3])[:dim]
    vector = np.array([0.2, -0.4, 0.8])[:dim]
    point_function, vector_function = (
        getattr(special, name),
        getattr(special, "v" + name),
    )
    expected = getattr(oracle, name)(points)
    for convert in (np.asarray, list, tuple):
        assert_allclose(
            point_function(convert(points)), expected, rtol=1e-14, atol=1e-15
        )
        for values in (vector, vector + 0.3j * vector[::-1]):
            result = vector_function(convert(values), convert(points))
            oracle_result = getattr(oracle, "v" + name)(values, points)
            assert result.dtype == oracle_result.dtype
            assert_allclose(result, oracle_result, rtol=1e-14, atol=1e-15)
            output = np.empty_like(result)
            assert vector_function(values, points, output) is output
            assert_allclose(output, result, rtol=1e-14, atol=1e-15)
    output = points.copy()
    assert point_function(output, output) is output
    assert_allclose(output, expected, rtol=1e-14, atol=1e-15)

    class Intercept(np.ndarray):
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            return "intercepted"

    subclass = points.view(Intercept)
    assert point_function(subclass) == "intercepted"
    assert vector_function(subclass, points) == "intercepted"
    assert vector_function(vector, subclass) == "intercepted"


@pytest.mark.physics
def test_coordinate_roundtrip_vector_norm_and_frame_composition():
    # Native properties check inversion, composition and orthogonality over random
    # and axis points at extreme scales; this checks the ufunc wiring on a batch
    # that includes the polar axis, the origin and both sides of the branch cut.
    cart = np.array(
        [
            [0.4, -0.3, 0.2],
            [0.0, 0.0, -1.3],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.5, 0.0, 0.3],
            [-1.5, -0.0, 0.3],
            [-1.5, 1e-300, -0.7],
        ]
    )
    vector = np.array([[0.2 + 0.3j, -0.4 + 0.1j, 0.7 - 0.2j]] * len(cart))
    cylindrical = special.car2cyl(cart)
    spherical = special.car2sph(cart)
    assert_allclose(special.cyl2car(cylindrical), cart, rtol=1e-13, atol=1e-15)
    assert_allclose(special.sph2car(spherical), cart, rtol=1e-13, atol=1e-15)
    assert_allclose(special.cyl2sph(cylindrical), spherical, rtol=1e-13, atol=1e-15)
    assert_allclose(special.sph2cyl(spherical), cylindrical, rtol=1e-13, atol=1e-15)
    assert_allclose(
        special.pol2car(special.car2pol(cart[:, :2])),
        cart[:, :2],
        rtol=1e-13,
        atol=1e-15,
    )
    vc = special.vcar2cyl(vector, cart)
    vs = special.vcar2sph(vector, cart)
    assert_allclose(special.vcyl2car(vc, cylindrical), vector, rtol=1e-13, atol=1e-15)
    assert_allclose(special.vsph2car(vs, spherical), vector, rtol=1e-13, atol=1e-15)
    assert_allclose(special.vcyl2sph(vc, cylindrical), vs, rtol=1e-13, atol=1e-15)
    assert_allclose(special.vsph2cyl(vs, spherical), vc, rtol=1e-13, atol=1e-15)
    norms = np.sum(np.abs(vector) ** 2, axis=-1)
    assert_allclose(np.sum(np.abs(vc) ** 2, axis=-1), norms, rtol=1e-13)
    assert_allclose(np.sum(np.abs(vs) ** 2, axis=-1), norms, rtol=1e-13)


@pytest.mark.gradients
@pytest.mark.parametrize("name", NAMES)
def test_coordinate_and_vector_pullbacks_broadcast_ownership_and_advect(name):
    dim = 2 if "pol" in name else 3
    points = np.array([[0.4, 0.7, 0.2], [1.3, 0.5, -0.5], [0.2, 0.3, 0.4]])[
        :, :dim
    ].copy()
    rng = np.random.default_rng(17)
    direction = rng.uniform(-0.2, 0.2, points.shape)
    point_g = rng.normal(size=points.shape)
    value, context = diff.coordinates(points, kind=name)
    assert_allclose(value, getattr(special, name)(points), rtol=1e-14, atol=1e-15)
    expected = context.pullback(point_g)
    check_pullback(
        lambda p: diff.coordinates(p, kind=name),
        points,
        directions=(direction,),
        cotangents=point_g,
        step=1e-6,
        rtol=2e-7,
        atol=1e-9,
    )

    def objective(p):
        return anp.sum(ad.coordinates(p, kind=name) * point_g)

    assert_allclose(advect.grad(objective)(points), expected, rtol=1e-13, atol=1e-13)
    context = diff.coordinates(points, kind=name)[1]
    points_saved = points.copy()
    points[:] = 0
    assert_one_use_context(
        context, np.asfortranarray(point_g), expected, rtol=1e-13, atol=1e-13
    )
    points[:] = points_saved

    vectors = complex_normal(rng, (2, 1, dim))
    value, context = diff.vector_coordinates(vectors, points, kind=name)
    assert_allclose(
        value, getattr(special, "v" + name)(vectors, points), rtol=1e-14, atol=1e-15
    )
    g = complex_normal(rng, value.shape)
    gradient = context.pullback(g)
    assert gradient[0].shape == vectors.shape
    assert gradient[1].shape == points.shape
    check_pullback(
        selecting(lambda v, p: diff.vector_coordinates(v, p, kind=name), 0, 1),
        vectors,
        points,
        directions=(complex_normal(rng, vectors.shape), direction),
        cotangents=g,
        step=1e-6,
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
    assert_one_use_context(
        context, np.asfortranarray(g), gradient, rtol=1e-13, atol=1e-13
    )


@pytest.mark.gradients
@pytest.mark.interface
def test_coordinate_empty_and_axis_contexts():
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


@pytest.mark.interface
@pytest.mark.reference
def test_overflowing_spherical_radius_is_infinite_with_exact_angles():
    # As in treams: the radius overflows, the angles stay exact.
    points = np.array([[1.7e308, 1.7e308, 0.0], [-1.7e308, -1.7e308, 1.0]])
    expected = [[np.inf, np.pi / 2, np.pi / 4], [np.inf, np.pi / 2, -3 * np.pi / 4]]
    with np.errstate(over="ignore"):
        assert_allclose(special.car2sph(points), expected, rtol=1e-15)
        assert_allclose(oracle.car2sph(points), expected, rtol=1e-15)
        assert_allclose(special.car2sph(np.tile(points, (600, 1)))[-2:], expected)


@pytest.mark.interface
def test_failed_loops_raise_their_error_with_warnings_ignored():
    # A failed loop's floating-point flags must not turn its error into NumPy's
    # overflow warning (a SystemError while warnings are ignored).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        special.car2cyl([[1.7e308, 1.7e308, 0.0]])
        for rows in (2, 2000):
            points = np.zeros((rows, 3))
            points[0] = [1.7e308, 1.7e308, 0.0]
            points[-1] = [np.nan, 0.0, 0.0]
            with pytest.raises(ValueError, match="finite"):
                special.car2cyl(points)

"""Coordinate JVPs preserve input shapes, broadcasting and recorded ownership."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from treams_rs import diff, special

from _support import arrange, complex_normal

pytestmark = pytest.mark.gradients

NAMES = (
    "car2cyl",
    "car2sph",
    "cyl2car",
    "cyl2sph",
    "sph2car",
    "sph2cyl",
    "car2pol",
    "pol2car",
)


@pytest.mark.parametrize("kind", NAMES)
@pytest.mark.parametrize("layout", ["C", "F", "reversed"])
def test_point_and_vector_pushforwards(kind, layout):
    dim = 2 if "pol" in kind else 3
    points = np.array([[0.4, 0.7, 0.2], [1.3, 0.5, -0.5], [0.2, 0.3, 0.4]])[
        :, :dim
    ].copy()
    rng = np.random.default_rng(27)
    dp = rng.uniform(-0.2, 0.2, points.shape)
    vectors = complex_normal(rng, (2, 1, dim))
    dv = complex_normal(rng, vectors.shape)
    h = 1e-6

    point_tangent = diff.coordinates(points, kind=kind)[1].pushforward(
        arrange(dp, layout)
    )
    point_function = getattr(special, kind)
    assert_allclose(
        point_tangent,
        (point_function(points + h * dp) - point_function(points - h * dp)) / (2 * h),
        rtol=2e-7,
        atol=1e-9,
    )

    vector_function = getattr(special, "v" + kind)
    expected = (
        vector_function(vectors + h * dv, points + h * dp)
        - vector_function(vectors - h * dv, points - h * dp)
    ) / (2 * h)
    value, context = diff.vector_coordinates(vectors, points, kind=kind)
    g = complex_normal(rng, value.shape)
    gv, gp = diff.vector_coordinates(vectors, points, kind=kind)[1].pullback(g)
    vectors[:] = 0
    points[:] = 0
    tangent = context.pushforward(arrange(dv, layout), arrange(dp, layout))
    assert tangent.shape == value.shape
    assert_allclose(tangent, expected, rtol=2e-7, atol=2e-9)
    assert_allclose(
        np.vdot(g, tangent).real,
        np.vdot(gv, dv).real + np.vdot(gp, dp).real,
        rtol=2e-13,
        atol=2e-13,
    )
    for actual, reference in zip(context.pullback(g), (gv, gp), strict=True):
        assert_allclose(actual, reference)
    assert_allclose(context.pushforward(-dv, -dp), -tangent)
    for actual, reference in zip(context.pullback(-g), (gv, gp), strict=True):
        assert_allclose(actual, -reference)
    assert_allclose(context.pushforward(dv, dp), tangent)


@pytest.mark.interface
def test_coordinate_tangent_validation_preserves_context():
    points = np.array([[0.4, 0.7, 0.2], [1.3, 0.5, -0.5]])
    _, point_context = diff.coordinates(points, kind="car2sph")
    for bad in (np.zeros(3), np.full_like(points, np.inf), points + 1j):
        with pytest.raises(ValueError, match="tangent"):
            point_context.pushforward(bad)
    assert_allclose(point_context.pushforward(np.zeros_like(points)), 0)

    vectors = np.array([0.2j, 0.3, 0.7])
    _, context = diff.vector_coordinates(vectors, points, kind="car2sph")
    for dv, dp in (
        (np.zeros_like(points), np.zeros_like(points)),
        (np.zeros_like(vectors), np.zeros(3)),
        (np.full_like(vectors, np.nan), np.zeros_like(points)),
        (np.zeros_like(vectors), np.full_like(points, np.inf)),
        (np.zeros_like(vectors), points + 1j),
    ):
        with pytest.raises(ValueError, match="tangent"):
            context.pushforward(dv, dp)
    assert_allclose(
        context.pushforward(np.zeros_like(vectors), np.zeros_like(points)), 0
    )


@pytest.mark.interface
@pytest.mark.parametrize("dim,kind", [(2, "car2pol"), (3, "car2sph")])
def test_coordinate_pushforwards_accept_empty_inputs(dim, kind):
    points = np.empty((0, dim))
    _, context = diff.coordinates(points, kind=kind)
    assert context.pushforward(points).shape == points.shape
    for vectors, points in (
        (np.ones(dim, complex), np.empty((0, dim))),
        (np.empty((0, dim), complex), np.ones(dim)),
    ):
        _, context = diff.vector_coordinates(vectors, points, kind=kind)
        assert context.pushforward(
            np.zeros_like(vectors), np.zeros_like(points)
        ).shape == (0, dim)


@pytest.mark.physics
def test_axial_point_directions_and_fixed_vector_frames():
    points = np.array([[0.0, 0.0, -1.0]])
    dp = np.array([[0.0, 0.0, 0.7]])
    _, context = diff.coordinates(points, kind="car2sph")
    assert_allclose(context.pushforward(dp), [[-0.7, 0.0, 0.0]])
    vectors = np.array([[0.1 + 0.2j, 0.3, 0.7]])
    dv = vectors * 0.3j
    _, context = diff.vector_coordinates(vectors, points, kind="car2sph")
    assert_allclose(context.pushforward(dv, dp), special.vcar2sph(dv, points))
    _, context = diff.coordinates(points, kind="car2sph")
    with pytest.raises(ValueError, match="undefined"):
        context.pushforward(np.array([[0.7, 0.0, 0.0]]))

"""Plane-wave JVPs preserve complex directions, static labels and operator layouts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import treams_rs as tr

from _support import LAYOUTS, arrange, complex_normal

pytestmark = pytest.mark.gradients


def _check_direction(record, inputs, directions):
    """Check a whole output direction against central differences and its adjoint."""
    value, context = record(*inputs)
    tangent = context.pushforward(*directions)
    h = 1e-5
    plus = record(*(x + h * dx for x, dx in zip(inputs, directions, strict=True)))[0]
    minus = record(*(x - h * dx for x, dx in zip(inputs, directions, strict=True)))[0]
    assert tangent.shape == value.shape
    assert_allclose(tangent, (plus - minus) / (2 * h), rtol=2e-8, atol=2e-9)
    cotangent = complex_normal(np.random.default_rng(231), value.shape)
    gradients = context.pullback(cotangent)
    if not isinstance(gradients, tuple):
        gradients = (gradients,)
    pairing = sum(
        np.vdot(g, dx).real for g, dx in zip(gradients, directions, strict=True)
    )
    assert_allclose(np.vdot(cotangent, tangent).real, pairing, rtol=2e-12, atol=2e-12)
    assert_allclose(context.pushforward(*(-d for d in directions)), -tangent)
    repeated = context.pullback(-cotangent)
    if not isinstance(repeated, tuple):
        repeated = (repeated,)
    for actual, expected in zip(repeated, gradients, strict=True):
        assert_allclose(actual, -expected)
    assert_allclose(context.pushforward(*directions), tangent)
    return tangent


def _plane_inputs():
    rng = np.random.default_rng(186)
    points = rng.normal(size=(4, 3)) * 0.2
    vectors = np.array(
        [[0.3, 0.2, 1.1 + 0.1j], [1.2, -0.1, 0.3j], [0.3, 0.2, 1.1 + 0.1j]]
    )
    return points, vectors, rng


@pytest.mark.parametrize("layout", LAYOUTS)
def test_plane_phases_pushforward(layout):
    points, vectors, rng = _plane_inputs()
    directions = (
        arrange(rng.normal(size=points.shape) * 0.1, layout),
        arrange(complex_normal(rng, vectors.shape) * 0.1, layout),
    )
    _check_direction(tr.diff.plane_phases, (points, vectors), directions)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("operator", [False, True])
@pytest.mark.parametrize("fixed_vectors", [False, True])
def test_plane_field_pushforward(poltype, operator, fixed_vectors):
    points, vectors, rng = _plane_inputs()
    coefficients = np.empty(0, complex) if operator else complex_normal(rng, 3)
    directions = (
        complex_normal(rng, coefficients.shape) * 0.1,
        rng.normal(size=points.shape) * 0.1,
        np.zeros_like(vectors)
        if fixed_vectors
        else complex_normal(rng, vectors.shape) * 0.1,
    )

    def record(coefficients, points, vectors):
        return tr.diff.plane_field(
            None if operator else coefficients,
            points,
            vectors,
            [0, 1, 1],
            poltype=poltype,
            fixed_vectors=fixed_vectors,
        )

    expected = _check_direction(record, (coefficients, points, vectors), directions)
    if fixed_vectors:
        # A frozen input contributes zero for any tangent, matching the VJP contract.
        actual = record(coefficients, points, vectors)[1].pushforward(
            *directions[:2], np.ones_like(vectors)
        )
        assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("fixed_vectors", [False, True])
def test_plane_expansion_pushforward(poltype, cylindrical, fixed_vectors):
    points, vectors, rng = _plane_inputs()
    points = points[:2]
    if cylindrical:
        vectors[:, 2] = 0.4
        basis = tr.CylindricalBasis.default([0.4], 2, 2, points)
    else:
        basis = tr.SphericalBasis.default(2, 2, points)
    dv = complex_normal(rng, vectors.shape) * 0.1
    if cylindrical:
        dv[:, 2] = 0
    if fixed_vectors:
        dv[:] = 0
    directions = (rng.normal(size=points.shape) * 0.1, dv)

    def record(points, vectors):
        return tr.diff.plane_expansion(
            type(basis)(basis.modes, points),
            vectors,
            [0, 1, 1],
            poltype=poltype,
            fixed_vectors=fixed_vectors,
        )

    expected = _check_direction(record, (points, vectors), directions)
    if cylindrical or fixed_vectors:
        ignored = np.ones_like(vectors)
        if not fixed_vectors:
            ignored[:, :2] = 0
        actual = record(points, vectors)[1].pushforward(directions[0], dv + ignored)
        assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("turns", [0, 1, 2])
def test_plane_permutation_pushforward(poltype, turns):
    _, vectors, rng = _plane_inputs()
    # Adjacent equal vectors carry distinct tangents, exercising shared local jets.
    vectors[1] = vectors[0]

    def record(vectors):
        return tr.diff.plane_permutation(vectors, [0, 1, 1], turns, poltype=poltype)

    _check_direction(record, (vectors,), (complex_normal(rng, vectors.shape) * 0.1,))


@pytest.mark.physics
def test_plane_pushforward_preserves_length_scaling():
    points, vectors, _ = _plane_inputs()
    phase = tr.diff.plane_phases(points, vectors)[1].pushforward(points, -vectors)
    field = tr.diff.plane_field(None, points, vectors, [0, 1, 1])[1].pushforward(
        np.empty(0, complex), points, -vectors
    )
    basis = tr.SphericalBasis.default(2, 1, points[:1])
    expansion = tr.diff.plane_expansion(basis, vectors, [0, 1, 1])[1].pushforward(
        points[:1], -vectors
    )
    permutation = tr.diff.plane_permutation(vectors, [0, 1, 1])[1].pushforward(vectors)
    for tangent in (phase, field, expansion, permutation):
        assert_allclose(tangent, 0, atol=3e-14)


@pytest.mark.interface
def test_plane_pushforward_rejected_tangents_preserve_context():
    points, vectors, _ = _plane_inputs()
    _, context = tr.diff.plane_phases(points, vectors)
    for invalid in (np.empty((0, 3)), np.full_like(points, np.nan), points + 1j):
        with pytest.raises(ValueError):
            context.pushforward(invalid, vectors)
    assert_allclose(context.pushforward(points, -vectors), 0, atol=3e-14)
    expected = tr.diff.plane_phases(points, vectors)[1].pushforward(points, vectors)
    assert_allclose(context.pushforward(points, vectors), expected)
    assert_allclose(context.pushforward(-points, -vectors), -expected)


@pytest.mark.interface
@pytest.mark.parametrize("operator", [False, True])
def test_plane_pushforward_empty_and_constant_axis_vectors(operator):
    vectors = np.array([[0, 0, 1.2]], complex)
    coefficients = np.empty(0, complex) if operator else np.ones(1, complex)
    for points in (np.empty((0, 3)), np.array([[0.1, 0.2, 0.3]])):
        value, context = tr.diff.plane_field(
            None if operator else coefficients, points, vectors, [1]
        )
        tangent = context.pushforward(
            np.zeros_like(coefficients), points, np.zeros_like(vectors)
        )
        assert tangent.shape == value.shape
        factor = 1j * points[:, 2] * vectors[0, 2]
        expected = value * factor.reshape((-1,) + (1,) * (value.ndim - 1))
        assert_allclose(tangent, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.interface
@pytest.mark.parametrize("cylindrical", [False, True])
def test_plane_expansion_pushforward_with_constant_axis_direction(cylindrical):
    positions = np.array([[0.1, 0.2, 0.3]])
    vectors = np.array([[0, 0, 1.2]], complex)
    if cylindrical:
        basis = tr.CylindricalBasis.default([1.2], 2, positions=positions)
    else:
        basis = tr.SphericalBasis.default(2, positions=positions)
    value, context = tr.diff.plane_expansion(basis, vectors, [1])
    dv = np.zeros_like(vectors)
    if cylindrical:
        # Axial wavevector components are fixed labels even for nonzero tangents.
        dv[:, 2] = 0.7
    tangent = context.pushforward(positions, dv)
    assert_allclose(
        tangent, value * (1j * 1.2 * positions[0, 2]), rtol=2e-14, atol=2e-14
    )


@pytest.mark.interface
def test_plane_permutation_zero_direction_preserves_axis_gauge():
    vectors = np.array([[0, 0, 1.2]], complex)
    _, context = tr.diff.plane_permutation(vectors, [1])
    assert_allclose(context.pushforward(np.zeros_like(vectors)), 0, atol=0)

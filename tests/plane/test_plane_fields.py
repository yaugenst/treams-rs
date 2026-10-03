"""Plane-wave field records (diff.plane_field): pullbacks, scale invariance, fixed-axis
vectors and cotangent memory layouts."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad
from treams_rs.testing import check_pullback

from _support import (
    LAYOUTS,
    arrange,
    assert_tree_allclose,
    complex_normal,
    selecting,
    strided_copies,
)


@pytest.mark.gradients
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("operator", [True, False])
def test_plane_pullback(poltype, operator):
    rng = np.random.default_rng(84)
    vectors = np.array(
        [[0.2, 0.3, 1.1 + 0.1j], [1.3, -0.2, 0.3j], [-0.3 + 0.1j, 0.1, -0.9 + 0.2j]]
    )
    points = rng.normal(size=(4, 3)) * 0.3
    coefficients = complex_normal(rng, 3)
    values = [coefficients, points, vectors]
    directions = [rng.normal(size=x.shape) * 0.1 for x in values]
    directions[0] = directions[0] + 0.13j
    directions[2] = directions[2] + 0.17j

    def forward(values):
        return tr.diff.plane_field(
            None if operator else values[0],
            values[1],
            values[2],
            [0, 1, 1],
            poltype=poltype,
        )

    value, context = forward(values)
    g = complex_normal(rng, value.shape)
    gradients = context.pullback(g)
    if not operator:
        # The field is linear in the coefficients: their gradient is the adjoint.
        matrix = tr.diff.plane_field(None, points, vectors, [0, 1, 1], poltype=poltype)
        adjoint = np.einsum("pcm,pc->m", matrix[0].conj(), g)
        assert_allclose(gradients[0], adjoint, rtol=1e-12, atol=1e-12)
    check_pullback(
        selecting(lambda p, v: forward([coefficients, p, v]), 1, 2),
        points,
        vectors,
        directions=tuple(directions[1:]),
        cotangents=g,
        step=1e-5,
        rtol=1e-8,
        atol=1e-9,
    )


@pytest.mark.physics
@pytest.mark.gradients
@given(kx=st.floats(0.1, 0.5), scale=st.floats(0.5, 2))
def test_plane_field_scale_invariance_and_advect(kx, scale):
    points = np.array([[0.2, 0.3, -0.1], [0.3, -0.2, 0.1]])
    vectors = np.array([[kx, 0.2, 1.3 + 0.1j], [1.5, -0.1, 0.2j]])
    coefficients = np.array([0.7 + 0.1j, -0.3 + 0.2j])

    def objective(points, vectors, coefficients):
        weighted = ad.plane_field(coefficients, points, vectors, polarizations=[0, 1])
        operator = ad.plane_field(None, points, vectors, polarizations=[0, 1])
        return anp.sum(anp.real(weighted * anp.conj(operator @ coefficients)))

    value = objective(points, vectors, coefficients)
    assert_allclose(
        objective(points * scale, vectors / scale, coefficients), value, rtol=1e-12
    )
    gp, gv, gc = advect.grad(objective, argnums=(0, 1, 2))(
        points, vectors, coefficients
    )
    assert_allclose(np.vdot(gp, points).real - np.vdot(gv, vectors).real, 0, atol=1e-11)
    assert_allclose(np.vdot(gc, coefficients).real, 2 * value, rtol=1e-12)


@pytest.mark.gradients
@pytest.mark.interface
@pytest.mark.reference
@pytest.mark.parametrize("z", [1.3, -1.3, 1.3 + 0.1j, -1.3 - 0.1j])
def test_axis_fixed_vectors_and_weighted_wave(z):
    vectors = [[0, 0, z]]
    points = np.array([[0.2, 0.3, 0.4]])
    _, context = tr.diff.plane_field([1], points, vectors, [1])
    with pytest.raises(ValueError, match="direction derivative"):
        context.pullback(np.ones((1, 3), complex))

    def objective(point):
        field = ad.plane_field(
            [1], point, vectors, polarizations=[1], fixed_vectors=True
        )
        return anp.sum(anp.real(field * anp.conj(field)))

    actual = advect.grad(objective)(points)
    assert_allclose(actual[0], [0, 0, -2 * np.imag(z) * objective(points)], atol=1e-12)
    wave = tr.plane_wave([0.2, 0.3, 0.9], [0.2 + 0.1j, 0.8], k0=1.3)
    oracle = treams.plane_wave([0.2, 0.3, 0.9], [0.2 + 0.1j, 0.8], k0=1.3)
    assert_allclose(wave.efield(points), oracle.efield(points), rtol=1e-12, atol=1e-12)


@pytest.mark.gradients
@pytest.mark.interface
@pytest.mark.parametrize("operator", [True, False])
def test_empty_plane_field_and_invalid_cotangent(operator):
    value, context = tr.diff.plane_field(
        None if operator else [1], np.empty((0, 3)), [[0, 0, 1]], [1]
    )
    assert value.shape == ((0, 3, 1) if operator else (0, 3))
    with pytest.raises(ValueError, match="shape"):
        context.pullback(np.ones((2, 3), complex))
    gc, gp, gk = context.pullback(value)
    assert gp.shape == (0, 3)
    assert_allclose(gc, 0)
    assert_allclose(gk, 0)


def _operator_field():
    points = [[0.1, 0.2, 0.3], [-0.1, 0.3, 0.2]]
    vectors = [[0.2, 0.3, 1.3], [1.5, -0.1, 0.2j]]
    return tr.diff.plane_field(None, points, vectors, [0, 1])


def _operator_cotangent_and_pullback():
    """A random cotangent in the layout of the output, and its pullback."""
    value, context = _operator_field()
    cotangent = np.empty_like(value)
    cotangent[:] = complex_normal(np.random.default_rng(97), value.shape)
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full_like(value, np.nan))
    return cotangent, context.pullback(cotangent)


@pytest.mark.gradients
@pytest.mark.interface
def test_operator_cotangent_layouts_preserve_pullback():
    cotangent, reference = _operator_cotangent_and_pullback()
    for layout in LAYOUTS:
        _, context = _operator_field()
        assert_tree_allclose(
            context.pullback(arrange(cotangent, layout)),
            reference,
            rtol=1e-13,
            atol=1e-13,
        )


@pytest.mark.gradients
@pytest.mark.interface
@given(st.data())
def test_operator_cotangent_in_composed_layouts_preserves_pullback(data):
    cotangent, reference = _operator_cotangent_and_pullback()
    alternative = data.draw(strided_copies(cotangent), label="cotangent")
    _, context = _operator_field()
    assert_tree_allclose(
        context.pullback(alternative), reference, rtol=1e-13, atol=1e-13
    )

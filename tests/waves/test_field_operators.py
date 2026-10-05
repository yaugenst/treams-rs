"""Field operators of spherical and cylindrical bases (operators.efield ... bfield)
against upstream, and the diff.field_operator pullbacks against weighted fields."""

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
from treams_rs.testing import check_gradient

from _support import complex_normal, to_oracle


@pytest.mark.reference
@pytest.mark.parametrize("cylindrical", [True, False])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("singular", [True, False])
@pytest.mark.parametrize("kind", ["efield", "hfield", "dfield", "bfield"])
def test_field_operators_reference(cylindrical, poltype, singular, kind):
    positions = [[0.1, 0.2, 0.3], [-0.2, 0.1, 0.4]]
    basis = (
        tr.CylindricalBasis.default([0.2, -0.3], 2, 2, positions)
        if cylindrical
        else tr.SphericalBasis.default(2, 2, positions)
    )
    oracle = to_oracle(basis)
    medium = (2.3 + 0.1j, 1.2 + 0.05j, 0.04 if poltype == "helicity" else 0)
    points = np.array(
        [[[0.7, 0.8, 0.2], [-0.4, 0.7, 0.1]], [[0.7, -0.4, 0.3], [0.5, 0.4, 0.1]]]
    )
    kwargs = dict(
        k0=1.3,
        material=medium,
        poltype=poltype,
        modetype="singular" if singular else "regular",
    )
    actual = getattr(tr.operators, kind)(points, basis=basis, **kwargs)
    expected = getattr(treams, kind)(points, basis=oracle, **kwargs)
    assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert_allclose(
        getattr(tr.operators, kind)(points[0, 0], basis=basis, **kwargs),
        actual[0, 0],
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.gradients
@pytest.mark.parametrize("cylindrical", [True, False])
@given(x=st.floats(0.7, 1.3), k=st.floats(0.8, 1.7))
def test_field_operator_pullback_and_weighted_equivalence(cylindrical, x, k):
    basis = (
        tr.CylindricalBasis.default([0.2], 2)
        if cylindrical
        else tr.SphericalBasis.default(2)
    )
    points = np.array([[x, 0.5, 0.3], [0.8, 0.2, 0.4]])
    ks = np.array([k + 0.1j, k + 0.2 + 0.1j])
    value, context = tr.diff.field_operator(points, basis, ks, singular=True)
    rng = np.random.default_rng(9)
    coefficients = complex_normal(rng, len(basis))
    field, weighted_context = tr.diff.field(
        coefficients, points, basis, ks, singular=True
    )
    assert_allclose(value @ coefficients, field, rtol=1e-12, atol=1e-12)
    g = complex_normal(rng, field.shape)
    gradients = context.pullback(g[:, :, None] * coefficients.conj())
    weighted = weighted_context.pullback(g)
    for a, b in zip(gradients, weighted[1:], strict=True):
        assert_allclose(a, b, rtol=1e-11, atol=1e-11)
    # The field is linear in the coefficients: their gradient is the adjoint.
    adjoint = np.einsum("pcm,pc->m", value.conj(), g)
    assert_allclose(weighted[0], adjoint, rtol=1e-11, atol=1e-11)
    h = 1e-5
    full_g = complex_normal(rng, value.shape)
    _, context = tr.diff.field_operator(points, basis, ks, singular=True)
    gradients = context.pullback(full_g)
    directions = [
        np.full_like(points, 0.07),
        np.full_like(basis.positions, -0.04),
        np.array([0.1 + 0.03j, -0.04 + 0.05j]),
    ]

    def shifted(step):
        b = type(basis)(basis.modes, basis.positions + step * directions[1])
        return tr.diff.field_operator(
            points + step * directions[0], b, ks + step * directions[2], singular=True
        )[0]

    numeric = np.vdot(full_g, (shifted(h) - shifted(-h)) / (2 * h)).real
    assert_allclose(
        sum(np.vdot(g, d).real for g, d in zip(gradients, directions, strict=True)),
        numeric,
        rtol=1e-7,
        atol=1e-7,
    )


@pytest.mark.gradients
@pytest.mark.parametrize("cylindrical", [True, False])
def test_operator_and_magnetic_advect(cylindrical):
    basis = (
        tr.CylindricalBasis.default([0.2], 1)
        if cylindrical
        else tr.SphericalBasis.default(1)
    )
    coefficients = np.arange(len(basis)) + 0.1j
    points = np.array([[0.8, 0.5, 0.3]])
    ks = np.array([1.2 + 0.1j, 1.3 + 0.1j])
    for poltype in ["helicity", "parity"]:
        if poltype == "parity":
            ks[:] = ks[0]

        def objective(points, origins, ks, impedance, poltype=poltype):
            magnetic = ad.hfield(
                coefficients,
                points,
                origins,
                ks,
                impedance,
                basis=basis,
                poltype=poltype,
            )
            electric = (
                ad.field_operator(points, origins, ks, basis=basis, poltype=poltype)
                @ coefficients
            )
            return anp.sum(anp.real(electric * anp.conj(magnetic)))

        check_gradient(
            objective,
            advect.grad(objective, argnums=(0, 1, 2, 3)),
            points,
            basis.positions,
            ks,
            np.array(0.8 + 0.1j),
            directions=(
                np.full_like(points, 0.1),
                np.full_like(basis.positions, -0.07),
                np.full_like(ks, 0.03 + 0.02j),
                np.array(0.04 + 0.03j),
            ),
            step=1e-5,
            rtol=1e-7,
            atol=1e-8,
        )


@pytest.mark.gradients
@pytest.mark.interface
def test_empty_operator_and_reusable_context():
    basis = tr.SphericalBasis.default(1)
    value, context = tr.diff.field_operator(np.empty((0, 3)), basis, [1, 1])
    assert value.shape == (0, 3, len(basis))
    points, origins, ks = context.pullback(value)
    assert points.shape == (0, 3)
    assert_allclose(origins, 0)
    assert_allclose(ks, 0)
    assert_allclose(context.pushforward(points, origins, ks), value)
    for actual, expected in zip(
        context.pullback(value), (points, origins, ks), strict=True
    ):
        assert_allclose(actual, expected)
    assert_allclose(context.pushforward(points, origins, ks), value)

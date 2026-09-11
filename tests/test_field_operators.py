import numpy as np
import pytest
import treams
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*scipy.special.sph_harm.*:DeprecationWarning"
)


@pytest.mark.parametrize("cylindrical", [True, False])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("singular", [True, False])
@pytest.mark.parametrize("kind", ["efield", "hfield", "dfield", "bfield"])
def test_field_operators_reference(cylindrical, poltype, singular, kind):
    positions = [[0.1, 0.2, 0.3], [-0.2, 0.1, 0.4]]
    basis = (
        tr.CylindricalWaveBasis.default([0.2, -0.3], 2, 2, positions)
        if cylindrical
        else tr.SphericalWaveBasis.default(2, 2, positions)
    )
    oracle = (
        treams.CylindricalWaveBasis if cylindrical else treams.SphericalWaveBasis
    )(basis.modes, positions)
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
    actual = getattr(tr, kind)(points, basis=basis, **kwargs)
    expected = getattr(treams, kind)(points, basis=oracle, **kwargs)
    assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert_allclose(
        getattr(tr, kind)(points[0, 0], basis=basis, **kwargs),
        actual[0, 0],
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize("cylindrical", [True, False])
@given(x=st.floats(0.7, 1.3), k=st.floats(0.8, 1.7))
def test_field_operator_pullback_and_weighted_equivalence(cylindrical, x, k):
    basis = (
        tr.CylindricalWaveBasis.default([0.2], 2)
        if cylindrical
        else tr.SphericalWaveBasis.default(2)
    )
    points = np.array([[x, 0.5, 0.3], [0.8, 0.2, 0.4]])
    ks = np.array([k + 0.1j, k + 0.2 + 0.1j])
    value, context = tr.diff.field_operator(points, basis, ks, singular=True)
    rng = np.random.default_rng(9)
    coefficients = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    field, weighted_context = tr.diff.field(
        coefficients, points, basis, ks, singular=True
    )
    assert_allclose(value @ coefficients, field, rtol=1e-12, atol=1e-12)
    g = rng.normal(size=field.shape) + 1j * rng.normal(size=field.shape)
    gradients = context.pullback(g[:, :, None] * coefficients.conj())
    weighted = weighted_context.pullback(g)
    for a, b in zip(gradients, weighted[1:], strict=True):
        assert_allclose(a, b, rtol=1e-11, atol=1e-11)
    h = 1e-5
    full_g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
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


@pytest.mark.parametrize("cylindrical", [True, False])
def test_operator_and_magnetic_advect(cylindrical):
    import advect
    import advect.numpy as anp

    from treams_rs import advect as ad

    basis = (
        tr.CylindricalWaveBasis.default([0.2], 1)
        if cylindrical
        else tr.SphericalWaveBasis.default(1)
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

        values = [points, basis.positions, ks, np.array(0.8 + 0.1j)]
        directions = [
            np.full_like(points, 0.1),
            np.full_like(basis.positions, -0.07),
            np.full_like(ks, 0.03 + 0.02j),
            0.04 + 0.03j,
        ]
        gradients = advect.grad(objective, argnums=(0, 1, 2, 3))(*values)
        h = 1e-5
        plus = [v + h * d for v, d in zip(values, directions, strict=True)]
        minus = [v - h * d for v, d in zip(values, directions, strict=True)]
        assert_allclose(
            sum(np.vdot(g, d).real for g, d in zip(gradients, directions, strict=True)),
            (objective(*plus) - objective(*minus)) / (2 * h),
            rtol=1e-7,
            atol=1e-8,
        )


def test_empty_operator_and_one_use_context():
    basis = tr.SphericalWaveBasis.default(1)
    value, context = tr.diff.field_operator(np.empty((0, 3)), basis, [1, 1])
    assert value.shape == (0, 3, len(basis))
    points, origins, ks = context.pullback(value)
    assert points.shape == (0, 3)
    assert_allclose(origins, 0)
    assert_allclose(ks, 0)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(value)

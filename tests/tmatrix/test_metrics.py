"""Global T-matrix metrics (cd, db and chi): upstream references, scale, phase and
helicity-swap invariants, orientation averages, and native and Advect pullbacks."""

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
from treams_rs.testing import check_gradient, check_pullback

from _support import complex_normal


def _matrix():
    rng = np.random.default_rng(91)
    return -0.15 * np.eye(6) + 0.02 * complex_normal(rng, (6, 6))


@pytest.mark.reference
@pytest.mark.parametrize("kind", ["cd", "db", "chi"])
@pytest.mark.parametrize("sphere", [False, True])
@pytest.mark.parametrize("kappa", [0, 0.07])
def test_global_metrics_reference(kind, sphere, kappa):
    materials = [(3.0 + 0.2j, 1.4 + 0.1j, 0.12), (1.3, 1.1, kappa)]
    if sphere:
        actual = tr.TMatrix.sphere(3, 1.3, 0.3, materials)
        expected = treams.TMatrix.sphere(3, 1.3, 0.3, materials)
    else:
        actual = tr.TMatrix(_matrix(), k0=1.3, material=materials[1])
        expected = treams.TMatrix(
            _matrix(), k0=1.3, material=treams.Material(materials[1])
        )
    assert_allclose(getattr(actual, kind), getattr(expected, kind), atol=1e-12)


@pytest.mark.gradients
@pytest.mark.parametrize("kind", ["cd", "db", "chi"])
def test_metric_native_matrix_and_wavenumber_vjps(kind):
    pol = tr.SphericalBasis.default(1).pol

    def record(a, ks):
        return tr.diff.tmatrix_metric(a, ks, polarizations=pol, kind=kind)

    values = (_matrix(), np.array([1.2, 1.4]))
    # Random unit directions are too large for this step; keep small ones.
    directions = (
        np.random.default_rng(4).normal(size=values[0].shape) * 0.01 + 0.013j,
        np.array([0.2, -0.1]),
    )
    check_pullback(
        record,
        *values,
        directions=directions,
        cotangents=np.asarray(0.7),
        step=1e-5,
        rtol=2e-8,
        atol=1e-9,
    )


@pytest.mark.physics
@pytest.mark.gradients
@pytest.mark.parametrize("kind", ["db", "chi"])
@given(scale=st.floats(0.1, 4), phase=st.floats(-3, 3))
def test_metric_scale_phase_and_helicity_swap_invariants(kind, scale, phase):
    a = _matrix()
    pol = tr.SphericalBasis.default(1).pol
    value, context = tr.diff.tmatrix_metric(a, polarizations=pol, kind=kind)
    gradient, gks = context.pullback(1.0)
    assert 0 <= value <= 1
    for factor in (scale * np.exp(1j * phase), 1e-200, 1e200):
        scaled, context = tr.diff.tmatrix_metric(
            a * factor, polarizations=1 - pol, kind=kind
        )
        gs, _ = context.pullback(1.0)
        assert_allclose(scaled, value, atol=1e-12)
        assert_allclose(gs * np.conj(factor), gradient, atol=1e-11)
    assert_allclose(np.vdot(gradient, a), 0, atol=1e-12)
    assert_allclose(gks, 0)


@pytest.mark.physics
@pytest.mark.gradients
@pytest.mark.parametrize("kind", ["cd", "db", "chi"])
@given(angles=st.tuples(st.floats(-3, 3), st.floats(0, 3.1), st.floats(-3, 3)))
def test_metrics_are_orientation_averages(kind, angles):
    # A rotation D keeps helicity, so metric(D a D^H) = metric(a) and the
    # gradient rotates with the matrix; the wavenumber gradient is unchanged.
    basis = tr.SphericalBasis.default(1)
    rotation = tr.operators.rotate(*angles, basis=basis)
    a = _matrix()
    ks = np.array([1.2, 1.4])
    value, context = tr.diff.tmatrix_metric(a, ks, polarizations=basis.pol, kind=kind)
    rotated, rotated_context = tr.diff.tmatrix_metric(
        rotation @ a @ rotation.conj().T, ks, polarizations=basis.pol, kind=kind
    )
    assert_allclose(rotated, value, rtol=0, atol=1e-13)
    gradient, gks = context.pullback(1.0)
    rotated_gradient, rotated_gks = rotated_context.pullback(1.0)
    assert_allclose(
        rotated_gradient, rotation @ gradient @ rotation.conj().T, rtol=0, atol=1e-12
    )
    assert_allclose(rotated_gks, gks, rtol=0, atol=1e-12)


@pytest.mark.physics
@pytest.mark.gradients
@given(scale=st.floats(0.2, 4))
def test_absorption_cd_common_wave_scale_and_swap(scale):
    a = _matrix()
    pol = tr.SphericalBasis.default(1).pol
    ks = np.array([1.2, 1.4])
    value, context = tr.diff.tmatrix_metric(a, ks, polarizations=pol, kind="cd")
    _, gradient = context.pullback(1.0)
    assert -1 <= value <= 1
    swapped, _ = tr.diff.tmatrix_metric(
        a, ks[::-1] * scale, polarizations=1 - pol, kind="cd"
    )
    assert_allclose(swapped, -value, atol=1e-12)
    assert_allclose(np.dot(gradient, ks), 0, atol=1e-12)


@pytest.mark.gradients
@pytest.mark.parametrize("kind", ["cd", "db", "chi"])
def test_complete_chiral_sphere_metric_advect(kind):
    basis = tr.SphericalBasis.default(2)

    def objective(radius, epsilon, kappa):
        matrix = ad.sphere(
            2,
            1.3,
            radius.reshape(1),
            anp.stack([epsilon, 1.0]),
            [1.4 + 0.1j, 1.0],
            anp.stack([kappa, 0.0]),
        )
        return ad.tmatrix_metric(matrix, [1.3, 1.3], polarizations=basis.pol, kind=kind)

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2)),
        np.array(0.3),
        np.array(3.0 + 0.2j),
        np.array(0.12 + 0.02j),
        directions=(np.array(0.03), np.array(0.2 + 0.1j), np.array(-0.05 + 0.01j)),
        step=1e-5,
        rtol=2e-7,
        atol=1e-9,
    )


@pytest.mark.gradients
@pytest.mark.interface
def test_undefined_normalizations_and_chirality_derivative():
    basis = tr.SphericalBasis.default(1)
    for kind in ("cd", "db", "chi"):
        with pytest.raises(ValueError, match="nonzero"):
            tr.diff.tmatrix_metric(np.zeros((6, 6)), polarizations=basis.pol, kind=kind)
    matrix = -0.1 * np.eye(6)
    value, context = tr.diff.tmatrix_metric(matrix, polarizations=basis.pol, kind="chi")
    assert value == 0
    with pytest.raises(ValueError, match="zero contrast"):
        context.pullback(1.0)
    assert_allclose(
        advect.grad(
            lambda a: ad.tmatrix_metric(a, polarizations=basis.pol, kind="chi") ** 2
        )(matrix),
        0,
    )
    value = tr.TMatrix(matrix, k0=1).changepoltype("parity")
    with pytest.raises(NotImplementedError, match="global helicity"):
        _ = value.chi

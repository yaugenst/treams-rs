"""Matrix-free numerical and ownership contracts through the native boundary."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import _native


@settings(max_examples=12, deadline=None)
@given(
    radius=st.floats(0.1, 0.35),
    epsilon=st.floats(1.2, 4.0),
    scale=st.floats(0.7, 1.4),
    lmax=st.integers(1, 2),
    columns=st.integers(1, 3),
)
def test_requested_illuminations_dense_adjoint_and_scale_invariance(
    radius: float, epsilon: float, scale: float, lmax: int, columns: int
) -> None:
    radii = np.array([radius, radius * 1.1])
    eps = np.array([epsilon, 2.2 + 0.03j], dtype=complex)
    positions = np.array([[0.0, 0.0, 0.0], [1.3, 0.2, -0.1]])
    op = _native.NativeSphereCluster(lmax, 1.2, radii, eps, positions)
    rng = np.random.default_rng(621)
    incident = np.asarray(
        rng.normal(size=(op.dimension, columns))
        + 1j * rng.normal(size=(op.dimension, columns)),
        dtype=np.complex128,
    )
    cotangent = np.asarray(
        rng.normal(size=incident.shape) + 1j * rng.normal(size=incident.shape),
        dtype=np.complex128,
    )
    value, context, reports = op.solve_with_pullback(incident, rtol=2e-12)
    dense, dense_context = _native.cluster(lmax, 1.2, radii, eps, positions)
    assert_allclose(value, dense @ incident, rtol=2e-9, atol=1e-13)
    assert len(reports) == columns
    assert all(residual <= 2e-12 * rhs for _, residual, rhs in reports)
    gr, gp, ge, gk, gincident, adjoint_reports = context.pullback(cotangent)
    physical = (gr, gp, ge, gk)
    expected = dense_context.pullback(cotangent @ incident.conj().T)
    for actual, reference in zip(physical, expected, strict=True):
        assert_allclose(actual, reference, rtol=2e-8, atol=2e-11)
    assert_allclose(gincident, dense.conj().T @ cotangent, rtol=2e-9, atol=1e-13)
    assert all(residual <= 2e-12 * rhs for _, residual, rhs in adjoint_reports)
    scaled = _native.NativeSphereCluster(
        lmax, 1.2 / scale, radii * scale, eps, positions * scale
    )
    assert_allclose(
        scaled.solve(incident, rtol=2e-12)[0], value, rtol=2e-10, atol=1e-13
    )
    assert_allclose(physical[1].sum(axis=0), 0, atol=1e-12)
    ward = (
        np.vdot(physical[0], radii).real
        + np.vdot(physical[1], positions).real
        - 1.2 * physical[3]
    )
    assert_allclose(ward, 0, atol=2e-10)
    single = [op.solve(incident[:, i : i + 1], rtol=2e-12)[0] for i in range(columns)]
    assert_allclose(np.column_stack(single), value, rtol=2e-11, atol=1e-13)


def test_native_residual_owns_inputs_and_rejects_invalid_cotangents_before_consuming():
    radii = np.array([0.2, 0.25])
    epsilon = np.array([2.0, 2.8 + 0.02j])
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.1, -0.2]])
    op = _native.NativeSphereCluster(1, 1.0, radii, epsilon, positions)
    incident = np.full((12, 2), 0.2 + 0.1j)
    original = incident.copy()
    value, context, _ = op.solve_with_pullback(incident)
    reference, _ = op.solve(incident)
    radii[:] = 50
    epsilon[:] = 8
    positions[:] = 0
    incident[:] = 0
    value[:] = 9
    assert_allclose(op.solve(original)[0], reference, rtol=1e-13)
    with pytest.raises(ValueError, match="shape"):
        context.pullback(np.zeros((2, 2), complex))
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full((12, 2), complex(np.nan, 0)))
    g = np.full((12, 2), 0.1 - 0.3j)
    result = context.pullback(g)
    other = op.solve_with_pullback(original)[1].pullback(g)
    for actual, expected in zip(result[:-1], other[:-1], strict=True):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


def test_zero_incident_and_nonconvergence_are_explicit():
    op = _native.NativeSphereCluster(
        2,
        1.7,
        np.array([0.3, 0.35]),
        np.array([3.0, 4.0], complex),
        np.array([[0.0, 0.0, 0.0], [0.9, 0.1, 0.2]]),
    )
    zero = np.zeros((op.dimension, 1), complex)
    value, context, reports = op.solve_with_pullback(zero)
    assert_allclose(value, 0, atol=0)
    assert reports == [(0, 0.0, 0.0)]
    physical = context.pullback(np.ones_like(zero))[:4]
    for gradient in physical:
        assert_allclose(gradient, 0, atol=0)
    with pytest.raises(ValueError, match="did not converge"):
        op.solve(np.ones_like(zero), rtol=1e-13, restart=1, max_iterations=1)
    with pytest.raises(ValueError, match="GMRES"):
        op.solve(zero, rtol=-1.0)


def test_public_vector_batch_results_and_pullback_shape_contract() -> None:
    from treams_rs.iterative import SphereCluster

    solver = SphereCluster(
        1, 1.2, [0.2, 0.25], [2.2, 2.7 + 0.02j], [[0, 0, 0], [1.1, 0.2, 0]]
    )
    incident = np.arange(solver.dimension) * (0.01 + 0.02j)
    vector, context = solver.solve_with_pullback(incident)
    batch = solver.solve(incident[:, None])
    assert vector.coefficients.shape == (solver.dimension,)
    assert batch.coefficients.shape == (solver.dimension, 1)
    assert_allclose(vector.coefficients, batch.coefficients[:, 0], rtol=1e-13)
    report = vector.convergence[0]
    assert report.iterations > 0
    assert report.residual_norm <= 1e-10 * report.rhs_norm
    with pytest.raises(ValueError, match="shape"):
        context.pullback(np.ones((solver.dimension, 1), complex))
    gradient = context.pullback(2 * vector.coefficients)
    assert gradient.incident.shape == incident.shape
    assert_allclose(
        np.vdot(gradient.incident, incident).real,
        2 * np.vdot(vector.coefficients, vector.coefficients).real,
        rtol=2e-10,
    )
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(vector.coefficients)

"""Matrix-free numerical and ownership contracts through the native boundary."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import treams_rs as tr

# The bindings of the matrix-free and dense sphere clusters are the layer under test.
from treams_rs import _native
from treams_rs.iterative import SphereCluster

from _support import assert_one_use_context, complex_normal


@pytest.mark.gradients
@pytest.mark.interface
@pytest.mark.parametrize("columns", [1, 3])
def test_requested_illuminations_match_the_dense_binding(columns):
    # The Rust iterative requested_columns_match_dense_and_obey_symmetries
    # property owns the physics (scale, translation and Ward identities); this
    # checks the binding: a Fortran-ordered, strided incident, the order of the
    # returned gradients against the dense _native.sphere_cluster context, the
    # convergence reports and that single columns stack to the batch.
    radii = np.array([0.2, 0.22])
    eps = np.array([2.5, 2.2 + 0.03j])
    positions = np.array([[0.0, 0.0, 0.0], [1.3, 0.2, -0.1]])
    op = _native.IterativeSphereCluster(2, 1.2, radii, eps, positions)
    rng = np.random.default_rng(621)
    storage = np.asfortranarray(complex_normal(rng, (op.dimension, 2 * columns)))
    incident = storage[:, ::2]
    cotangent = complex_normal(rng, incident.shape)
    value, context, convergence = op.record(incident, rtol=2e-12)
    dense, dense_context = _native.sphere_cluster(2, 1.2, radii, eps, positions)
    assert_allclose(value, dense @ incident, rtol=2e-9, atol=1e-13)
    assert len(convergence) == columns
    assert all(residual <= 2e-12 * rhs for _, residual, rhs in convergence)
    *physical, gincident, adjoint_convergence = context.pullback(cotangent)
    expected = dense_context.pullback(cotangent @ incident.conj().T)
    for actual, reference in zip(physical, expected, strict=True):
        assert_allclose(actual, reference, rtol=2e-8, atol=2e-11)
    assert_allclose(gincident, dense.conj().T @ cotangent, rtol=2e-9, atol=1e-13)
    assert len(adjoint_convergence) == columns
    assert all(residual <= 2e-12 * rhs for _, residual, rhs in adjoint_convergence)
    single = [op.solve(incident[:, i : i + 1], rtol=2e-12)[0] for i in range(columns)]
    assert_allclose(np.column_stack(single), value, rtol=2e-11, atol=1e-13)


@pytest.mark.interface
def test_native_operator_owns_its_inputs():
    """Overwriting the construction arrays leaves a built operator unchanged.

    tests/bindings/test_native_contexts.py checks that the context of record is
    one-use and owns its inputs.
    """
    radii = np.array([0.2, 0.25])
    epsilon = np.array([2.0, 2.8 + 0.02j])
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.1, -0.2]])
    op = _native.IterativeSphereCluster(1, 1.0, radii, epsilon, positions)
    incident = np.full((12, 2), 0.2 + 0.1j)
    reference, _ = op.solve(incident)
    radii[:] = 50
    epsilon[:] = 8
    positions[:] = 0
    assert_allclose(op.solve(incident)[0], reference, rtol=0, atol=0)


@pytest.mark.gradients
@pytest.mark.interface
def test_zero_incident_and_nonconvergence_are_explicit():
    op = _native.IterativeSphereCluster(
        2,
        1.7,
        np.array([0.3, 0.35]),
        np.array([3.0, 4.0], complex),
        np.array([[0.0, 0.0, 0.0], [0.9, 0.1, 0.2]]),
    )
    zero = np.zeros((op.dimension, 1), complex)
    value, context, convergence = op.record(zero)
    assert_allclose(value, 0, atol=0)
    assert convergence == [(0, 0.0, 0.0)]
    physical = context.pullback(np.ones_like(zero))[:4]
    for gradient in physical:
        assert_allclose(gradient, 0, atol=0)
    with pytest.raises(ValueError, match="did not converge"):
        op.solve(np.ones_like(zero), rtol=1e-13, restart=1, max_iterations=1)
    with pytest.raises(ValueError, match="GMRES"):
        op.solve(zero, rtol=-1.0)


@pytest.mark.gradients
@pytest.mark.interface
def test_public_vector_batch_results_and_pullback_shape_contract() -> None:
    solver = SphereCluster(
        1, 1.2, [0.2, 0.25], [2.2, 2.7 + 0.02j], [[0, 0, 0], [1.1, 0.2, 0]]
    )
    incident = np.arange(solver.dimension) * (0.01 + 0.02j)
    vector, context = solver.record(incident)
    batch = solver.solve(incident[:, None])
    assert vector.coefficients.shape == (solver.dimension,)
    assert batch.coefficients.shape == (solver.dimension, 1)
    assert_allclose(vector.coefficients, batch.coefficients[:, 0], rtol=1e-13)
    report = vector.convergence[0]
    assert report.iterations > 0
    assert report.residual_norm <= 1e-10 * report.rhs_norm
    gradient = assert_one_use_context(
        context,
        2 * vector.coefficients,
        wrong_shape=np.ones((solver.dimension, 1), complex),
    )
    assert gradient._fields == (
        "k0",
        "radii",
        "epsilon",
        "positions",
        "incident",
        "convergence",
    )
    assert gradient.incident.shape == incident.shape
    assert_allclose(
        np.vdot(gradient.incident, incident).real,
        2 * np.vdot(vector.coefficients, vector.coefficients).real,
        rtol=2e-10,
    )


@pytest.mark.interface
@pytest.mark.parametrize(
    "incident,match",
    [
        (tr.plane_wave([0, 0, 1], [1, 0], k0=1.3), "frequency and vacuum medium"),
        (
            tr.plane_wave([0, 0, 1], [1, 0], k0=1.2, material=2),
            "frequency and vacuum medium",
        ),
        (
            tr.plane_wave([0, 0, 1], [1, 0], k0=1.2, poltype="parity"),
            "requires helicity polarization",
        ),
    ],
)
def test_physical_scatter_requires_a_matching_vacuum_helicity_source(incident, match):
    solver = SphereCluster(1, 1.2, [0.2], [2.2], [[0, 0, 0]])
    with pytest.raises(ValueError, match=match):
        solver.scatter(incident)

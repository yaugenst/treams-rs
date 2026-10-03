"""Requested RHS solves, factor ownership, and complete native adjoints."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import treams_rs as tr
from treams_rs.testing import check_pullback

from _support import (
    assert_one_use_context,
    assert_tree_allclose,
    complex_normal,
    selecting,
)


def matrices(n=6, p=2):
    rng = np.random.default_rng(923)
    return [
        complex_normal(rng, shape) * scale
        for shape, scale in [((n, n), 0.08), ((n, n), 0.09), ((n, p), 1.0)]
    ]


@pytest.mark.physics
@given(st.integers(2, 9), st.integers(1, 4), st.floats(-2, 2))
@settings(max_examples=24)
def test_requested_columns_and_factor_linearity(n, p, scale):
    t, c, a = matrices(n, p)
    factor = tr.diff.factor_interaction(t, c)
    expected = np.linalg.solve(np.eye(n) - t @ c, t @ a)
    np.testing.assert_allclose(factor.solve(a), expected, rtol=2e-13, atol=1e-14)
    np.testing.assert_allclose(
        factor.solve(a * scale), expected * scale, rtol=2e-13, atol=1e-14
    )
    np.testing.assert_allclose(
        tr.diff.interaction(t, c)[0] @ a, expected, rtol=2e-13, atol=1e-14
    )


@pytest.mark.gradients
def test_shared_factor_snapshots_and_one_use_context():
    t, c, a = matrices()
    factor = tr.diff.factor_interaction(t, c)
    baseline, context = factor.record(a)
    second, second_context = factor.record(a * 2)
    expected_gradient = tr.diff.illuminate(t, c, a)[1].pullback(np.ones_like(a))
    expected_second = tr.diff.illuminate(t, c, 2 * a)[1].pullback(second.copy())
    t[:] = 100
    c[:] = 200
    a[:] = 300
    baseline[:] = -999
    del factor
    assert_one_use_context(
        context, np.ones((6, 2), complex), expected_gradient, atol=1e-14
    )
    # The second record keeps its own snapshot of the shared factor.
    assert_tree_allclose(
        second_context.pullback(second), expected_second, rtol=0, atol=1e-14
    )


@pytest.mark.gradients
@pytest.mark.interface
@given(st.tuples(st.booleans(), st.booleans(), st.booleans(), st.booleans()))
@settings(max_examples=16)
def test_fortran_views_preserve_logical_columns(reversed_columns):
    inputs = [*matrices(), np.full((6, 2), 0.3 - 0.2j)]
    views = [
        np.asfortranarray(value)[:, :: -1 if reverse else 1]
        for value, reverse in zip(inputs, reversed_columns, strict=True)
    ]
    value, context = tr.diff.illuminate(*views[:3])
    expected, reference = tr.diff.illuminate(*(x.copy() for x in views[:3]))
    np.testing.assert_allclose(value, expected, atol=1e-14)
    for actual, wanted in zip(
        context.pullback(views[3]), reference.pullback(views[3].copy()), strict=True
    ):
        np.testing.assert_allclose(actual, wanted, atol=1e-14)


@pytest.mark.gradients
def test_complete_complex_directional_pullback():
    # The incident illumination has its exact adjoint test below.
    t, c, a = matrices()
    check_pullback(
        selecting(lambda t, c: tr.diff.illuminate(t, c, a), 0, 1),
        t,
        c,
        cotangents=complex_normal(np.random.default_rng(37), a.shape),
        step=1e-5,
        rtol=2e-8,
        atol=1e-9,
        seed=37,
    )


@pytest.mark.gradients
def test_incident_pullback_is_the_exact_adjoint():
    # X = (I - T C)^-1 T a is linear in a, so its gradient is the adjoint map
    # and pairs with a to the real part of <g, X>.
    t, c, a = matrices()
    value, context = tr.diff.illuminate(t, c, a)
    g = complex_normal(np.random.default_rng(37), a.shape)
    gradient = context.pullback(g)[2]
    system = np.eye(len(t)) - t @ c
    adjoint = t.conj().T @ np.linalg.solve(system.conj().T, g)
    np.testing.assert_allclose(gradient, adjoint, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(
        np.vdot(gradient, a).real, np.vdot(g, value).real, rtol=1e-13
    )


@pytest.mark.gradients
def test_local_blocks_keep_values_and_gradient_blocks():
    t, c, a = matrices()
    t[:2, 2:] = 0
    t[2:, :2] = 0
    blocks = [t[:2, :2], t[2:, 2:]]
    factor = tr.diff.factor_interaction_blocks(blocks, c)
    value, context = factor.record(a)
    reference, full_context = tr.diff.illuminate(t, c, a)
    np.testing.assert_allclose(value, reference, atol=1e-14)
    g = np.full(a.shape, 0.3 - 0.2j)
    expected = full_context.pullback(g)
    with pytest.raises(ValueError, match="pullback_blocks"):
        context.pullback(g)
    actual = context.pullback_blocks(g)
    np.testing.assert_allclose(actual[0][0], expected[0][:2, :2], atol=1e-14)
    np.testing.assert_allclose(actual[0][1], expected[0][2:, 2:], atol=1e-14)
    for lhs, rhs in zip(actual[1:], expected[1:], strict=True):
        np.testing.assert_allclose(lhs, rhs, atol=1e-14)


@pytest.mark.workflows
def test_physical_cluster_convenience_and_field_reconstruction():
    spheres = [
        tr.TMatrix.sphere(2, 1.3, r, [eps, 1])
        for r, eps in [(0.2, 3 + 0.1j), (0.25, 4)]
    ]
    cluster = tr.Cluster(spheres, positions=[[0, 0, 0], [0.7, 0.2, 0.1]])
    ks = cluster.medium.ks(cluster.k0)
    wave = tr.plane_wave([0.2, 0.1, 1], 1, k0=1.3)
    a = wave.expand(cluster.basis)
    expected = cluster.solve().array @ a
    np.testing.assert_allclose(
        cluster.scatter(wave).coefficients, expected, rtol=2e-12, atol=1e-14
    )
    batch = np.column_stack([a, a * (0.3 + 0.2j)])
    coupling = tr.diff.expansion(cluster.basis, cluster.basis, ks, singular=True)[0]
    factor = tr.diff.factor_interaction_blocks([s.array for s in spheres], coupling)
    np.testing.assert_allclose(
        factor.solve(batch), cluster.scatter(batch).coefficients, atol=1e-14
    )
    native = tr.diff.sphere_cluster_factor(
        2, 1.3, [0.2, 0.25], [3 + 0.1j, 4], [[0, 0, 0], [0.7, 0.2, 0.1]]
    )
    np.testing.assert_allclose(native.solve(batch), factor.solve(batch), atol=1e-14)
    g = np.full(batch.shape, 0.2 - 0.4j)
    gradients = native.record(batch)[1].pullback_blocks(g)
    expected_gradients = factor.record(batch)[1].pullback_blocks(g)
    for actual, expected_local in zip(gradients[0], expected_gradients[0], strict=True):
        np.testing.assert_allclose(actual, expected_local, rtol=2e-12, atol=1e-14)
    for actual, expected_gradient in zip(
        gradients[1:], expected_gradients[1:], strict=True
    ):
        np.testing.assert_allclose(actual, expected_gradient, rtol=2e-12, atol=1e-14)
    points = [[1.5, 0.4, 0.2], [-0.2, 1.0, 0.4]]
    fields = tr.diff.field(expected, points, cluster.basis, ks)[0]
    actual = tr.diff.field(
        cluster.scatter(wave).coefficients, points, cluster.basis, ks
    )[0]
    np.testing.assert_allclose(actual, fields, rtol=2e-12, atol=1e-14)


@pytest.mark.workflows
def test_periodic_requested_incidence():
    sphere = tr.TMatrix.sphere(2, 1.3, 0.2, [3.1, 1])
    cell = [[1.7, 0], [0.1, 1.9]]
    kpar = [0.2, -0.1]
    incident = tr.plane_wave([0.2, -0.1, np.sqrt(1.3**2 - 0.05)], 0, k0=1.3).expand(
        sphere.basis
    )
    expected = sphere.latticeinteraction.solve(cell, kpar) @ incident
    np.testing.assert_allclose(
        sphere.latticeinteraction.illuminate(incident, cell, kpar),
        expected,
        rtol=2e-12,
        atol=1e-14,
    )

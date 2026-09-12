"""Requested RHS solves, factor ownership, and complete native adjoints."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import treams_rs as tr


def matrices(n=6, p=2):
    rng = np.random.default_rng(923)
    return [
        (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * scale
        for shape, scale in [((n, n), 0.08), ((n, n), 0.09), ((n, p), 1.0)]
    ]


@given(st.integers(2, 9), st.integers(1, 4), st.floats(-2, 2))
@settings(max_examples=24, deadline=None)
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


def test_shared_factor_snapshots_and_one_use_context():
    t, c, a = matrices()
    factor = tr.diff.factor_interaction(t, c)
    baseline, context = factor.record(a)
    second, second_context = factor.record(a * 2)
    expected_gradient = tr.diff.illuminate(t, c, a)[1].pullback(np.ones_like(a))
    t[:] = 100
    c[:] = 200
    a[:] = 300
    baseline[:] = -999
    del factor
    with pytest.raises(ValueError, match="shape"):
        context.pullback(np.zeros((2, 2), complex))
    for actual, expected in zip(
        context.pullback(np.ones((6, 2), complex)), expected_gradient, strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(np.ones((6, 2), complex))
    assert all(np.isfinite(x).all() for x in second_context.pullback(second))


@given(st.tuples(st.booleans(), st.booleans(), st.booleans(), st.booleans()))
@settings(max_examples=16, deadline=None)
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


@pytest.mark.parametrize("parameter", range(3))
def test_complete_complex_directional_pullback(parameter):
    values = matrices()
    rng = np.random.default_rng(37)
    weights = rng.normal(size=values[2].shape) + 1j * rng.normal(size=values[2].shape)
    direction = rng.normal(size=values[parameter].shape) + 1j * rng.normal(
        size=values[parameter].shape
    )
    gradient = tr.diff.illuminate(*values)[1].pullback(weights)[parameter]

    def loss(step):
        inputs = [x.copy() for x in values]
        inputs[parameter] += step * direction
        return np.vdot(weights, tr.diff.illuminate(*inputs)[0]).real

    numerical = (loss(1e-5) - loss(-1e-5)) / 2e-5
    np.testing.assert_allclose(
        np.vdot(gradient, direction).real, numerical, rtol=2e-8, atol=1e-9
    )


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


def test_physical_cluster_convenience_and_field_reconstruction():
    spheres = [
        tr.TMatrix.sphere(2, 1.3, r, [eps, 1])
        for r, eps in [(0.2, 3 + 0.1j), (0.25, 4)]
    ]
    cluster = tr.TMatrix.cluster(spheres, [[0, 0, 0], [0.7, 0.2, 0.1]])
    wave = tr.plane_wave([0.2, 0.1, 1], 1, k0=1.3)
    a = wave.expand(cluster.basis)
    expected = cluster.interaction.solve().array @ a
    np.testing.assert_allclose(
        cluster.interaction.illuminate(wave), expected, rtol=2e-12, atol=1e-14
    )
    batch = np.column_stack([a, a * (0.3 + 0.2j)])
    factor = cluster.interaction.factor()
    np.testing.assert_allclose(
        factor.solve(batch), cluster.interaction.illuminate(batch), atol=1e-14
    )
    native = tr.diff.cluster_factor(
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
    fields = tr.diff.field(expected, points, cluster.basis, cluster.ks)[0]
    actual = tr.diff.field(
        cluster.interaction.illuminate(wave), points, cluster.basis, cluster.ks
    )[0]
    np.testing.assert_allclose(actual, fields, rtol=2e-12, atol=1e-14)


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


def test_advect_requested_illumination():
    ad = pytest.importorskip("advect")
    from treams_rs import advect as tr_ad

    t, c, a = matrices()

    def loss(values):
        return ad.numpy.sum(ad.numpy.abs(tr_ad.illuminate(t, c, values)) ** 2)

    value, context = tr.diff.illuminate(t, c, a)
    expected = context.pullback(2 * value)[2]
    np.testing.assert_allclose(ad.grad(loss)(a), expected, rtol=2e-12, atol=1e-14)

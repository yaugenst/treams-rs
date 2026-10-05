"""Native dense-linear-algebra pushforwards and reusable contexts."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import linear_sum_assignment

from treams_rs import diff

from _support import assert_reusable_context, assert_tree_allclose, complex_normal

pytestmark = pytest.mark.gradients


def _operator():
    # Distinct eigenvalues, with unique phase pivots for the right eigenvectors.
    return np.array(
        [[1 + 0.1j, 0.2 + 0.1j, 0.1], [-0.05j, 2 - 0.2j, 0.3], [0.1, -0.1j, 3 + 0.15j]]
    )


def _matched_eig(operator, reference):
    (values, vectors), _ = diff.eig(operator)
    _, order = linear_sum_assignment(abs(reference[:, None] - values))
    return values[order], vectors[:, order]


def test_solve_pushforward_matches_finite_difference_and_adjoint():
    rng = np.random.default_rng(205)
    a = _operator()
    b = complex_normal(rng, (3, 2))
    da, db = complex_normal(rng, a.shape), complex_normal(rng, b.shape)
    x, context = diff.solve(a, b)
    dx = context.pushforward(da, db)
    assert_allclose(a @ dx + da @ x, db, atol=2e-14)
    h = 1e-5
    numeric = (
        np.linalg.solve(a + h * da, b + h * db)
        - np.linalg.solve(a - h * da, b - h * db)
    ) / (2 * h)
    # The centered difference has O(h**2) truncation error.
    assert_allclose(dx, numeric, rtol=1e-8, atol=1e-9)
    cotangent = complex_normal(rng, x.shape)
    _, context = diff.solve(a, b)
    ga, gb = context.pullback(cotangent)
    assert_allclose(
        np.vdot(cotangent, dx).real,
        np.vdot(ga, da).real + np.vdot(gb, db).real,
        atol=2e-14,
    )


@pytest.mark.parametrize(
    ("scale", "row_scale", "column_scale"),
    [
        (1e-200, np.ones(3), np.ones(3)),
        (1e200, np.ones(3), np.ones(3)),
        (1.0, np.array([1e-80, 1, 1e80]), np.array([1e80, 1, 1e-80])),
    ],
    ids=["tiny", "huge", "equilibrated"],
)
def test_solve_pushforward_respects_scaling_and_equilibration(
    scale, row_scale, column_scale
):
    rng = np.random.default_rng(206)
    a, da = _operator(), complex_normal(rng, (3, 3))
    b, db = complex_normal(rng, (3, 2)), complex_normal(rng, (3, 2))
    x = np.linalg.solve(a, b)
    expected = np.linalg.solve(a, db - da @ x)
    _, context = diff.solve(
        (row_scale[:, None] * a * column_scale) * scale,
        (row_scale[:, None] * b) * scale,
    )
    tangent = context.pushforward(
        (row_scale[:, None] * da * column_scale) * scale,
        (row_scale[:, None] * db) * scale,
    )
    assert_allclose(tangent * column_scale[:, None], expected, atol=2e-14)


def test_eig_pushforward_matches_finite_difference_phase_and_adjoint():
    rng = np.random.default_rng(207)
    a = _operator()
    da = complex_normal(rng, a.shape)
    (w, v), context = diff.eig(a)
    dw, dv = context.pushforward(da)
    assert_allclose(da @ v + a @ dv, dv * w + v * dw, atol=2e-14)
    assert_allclose(np.sum(v.conj() * dv, axis=0).real, 0, atol=2e-14)
    pivots = np.argmax(abs(v), axis=0)
    assert_allclose(dv[pivots, np.arange(len(w))].imag, 0, atol=2e-14)
    h = 1e-5
    wp, vp = _matched_eig(a + h * da, w)
    wm, vm = _matched_eig(a - h * da, w)
    # Matching modes removes ordering changes; the native phase convention is fixed.
    assert_allclose(dw, (wp - wm) / (2 * h), rtol=1e-8, atol=1e-9)
    assert_allclose(dv, (vp - vm) / (2 * h), rtol=1e-8, atol=1e-9)
    gw, gv = complex_normal(rng, w.shape), complex_normal(rng, v.shape)
    _, context = diff.eig(a)
    gradient = context.pullback(gw, gv)
    assert_allclose(
        np.vdot(gw, dw).real + np.vdot(gv, dv).real,
        np.vdot(gradient, da).real,
        atol=2e-14,
    )


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_eig_pushforward_respects_extreme_scaling(scale):
    a = _operator()
    da = complex_normal(np.random.default_rng(208), a.shape)
    (w, _), context = diff.eig(a)
    dw, dv = context.pushforward(da)
    (ws, _), context = diff.eig(a * scale)
    dws, dvs = context.pushforward(da * scale)
    _, order = linear_sum_assignment(abs(w[:, None] - ws / scale))
    assert_allclose(dws[order] / scale, dw, atol=2e-13)
    assert_allclose(dvs[:, order], dv, atol=2e-13)


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (5, 3), (3, 5)])
def test_svdvals_pushforward_matches_finite_difference_energy_and_adjoint(shape):
    rng = np.random.default_rng(209)
    a, da = complex_normal(rng, shape), complex_normal(rng, shape)
    values, context = diff.svdvals(a)
    tangent = context.pushforward(da)
    h = 1e-5
    numeric = (
        np.linalg.svd(a + h * da, compute_uv=False)
        - np.linalg.svd(a - h * da, compute_uv=False)
    ) / (2 * h)
    assert_allclose(tangent, numeric, rtol=1e-8, atol=1e-9)
    assert_allclose(np.dot(values, tangent), np.vdot(a, da).real, atol=2e-14)
    cotangent = np.linspace(0.2, 1.1, len(values))
    _, context = diff.svdvals(a)
    gradient = context.pullback(cotangent)
    assert_allclose(np.dot(cotangent, tangent), np.vdot(gradient, da).real, atol=2e-14)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_svdvals_pushforward_respects_extreme_scaling(scale):
    a = _operator()
    da = complex_normal(np.random.default_rng(210), a.shape)
    _, context = diff.svdvals(a)
    tangent = context.pushforward(da)
    _, context = diff.svdvals(a * scale)
    scaled = context.pushforward(da * scale)
    assert_allclose(scaled / scale, tangent, atol=2e-14)


@pytest.mark.parametrize(
    ("record", "operator", "message"),
    [
        (diff.eig, np.eye(3), "repeated eigenvalues"),
        (diff.eig, np.array([[1, 0.3j], [-0.3j, 1]]), "unique largest component"),
        (diff.svdvals, np.diag([2, 2, 1]), "repeated singular values"),
        (diff.svdvals, np.diag([2, 1, 0]), "at zero"),
    ],
    ids=[
        "repeated-eigenvalue",
        "phase-tie",
        "repeated-singular-value",
        "zero-singular-value",
    ],
)
def test_spectral_pushforwards_reject_nondifferentiable_outputs(
    record, operator, message
):
    _, context = record(operator)
    with pytest.raises(ValueError, match=message):
        context.pushforward(np.ones_like(operator, dtype=complex))


def _context_case(name):
    a = _operator()
    da = complex_normal(np.random.default_rng(211), a.shape)
    if name == "solve":
        b = np.ones((3, 2), dtype=complex)
        return diff.solve, (a, b), (da, b * (0.2 + 0.1j)), (b,)
    if name == "eig":
        return diff.eig, (a,), (da,), (np.ones(3, dtype=complex), np.ones_like(a))
    if name == "eigvals":
        return diff.eigvals, (a,), (da,), (np.ones(3, dtype=complex),)
    return diff.svdvals, (a,), (da,), (np.ones(3),)


@pytest.mark.interface
@pytest.mark.parametrize("name", ["solve", "eig", "eigvals", "svdvals"])
def test_pushforward_rejects_invalid_tangents_and_owns_inputs(name):
    record, inputs, tangents, cotangents = _context_case(name)
    cotangents = tuple(np.array(c, copy=True) for c in cotangents)
    _, reference = record(*inputs)
    expected = reference.pushforward(*tangents)
    expected_pullback = reference.pullback(*cotangents)
    _, context = record(*inputs)
    for value in inputs:
        value[:] = 0
    # Apply the same shape/nonfinite and reuse contract as native pullbacks.
    assert_reusable_context(
        SimpleNamespace(pullback=context.pushforward), tangents, expected
    )
    assert_tree_allclose(context.pullback(*cotangents), expected_pullback)
    assert_tree_allclose(context.pushforward(*tangents), expected)

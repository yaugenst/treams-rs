"""Spectral derivatives separate eigenvalue phases from unresolved singular values."""

import contextlib
import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import linear_sum_assignment

from treams_rs import diff
from treams_rs.testing import check_pushforward

from _support import jax_x64

pytestmark = pytest.mark.gradients


def _phase_tie():
    return np.array([[2.0, 1.0], [1.0, 2.0]], dtype=complex)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_eigenvalue_tangents_ignore_eigenvector_phase_ties(scale):
    operator = _phase_tie() * scale
    direction = np.array([[0.2 + 0.1j, 0.3 - 0.2j], [-0.1 + 0.4j, 0.5 - 0.1j]]) * scale
    values, context = diff.eigvals(operator)
    tangent = context.pushforward(direction)
    assert_allclose(np.sort(values.real) / scale, [1.0, 3.0], atol=2e-14)
    assert_allclose(np.sum(tangent) / scale, np.trace(direction) / scale, atol=2e-14)
    weights = np.array([0.2 + 0.1j, -0.3 + 0.2j])
    assert_allclose(
        np.vdot(weights, tangent / scale).real,
        np.vdot(context.pullback(weights), direction / scale).real,
        atol=2e-14,
    )
    shifted = []
    step = 1e-5
    for sign in (-1, 1):
        perturbed = np.linalg.eigvals((operator + sign * step * direction) / scale)
        _, order = linear_sum_assignment(abs(values[:, None] / scale - perturbed))
        shifted.append(perturbed[order])
    assert_allclose(tangent / scale, (shifted[1] - shifted[0]) / (2 * step), atol=1e-9)
    restored = type(context)._from_state(context._state())
    assert_allclose(restored.pushforward(direction), tangent, rtol=0, atol=0)
    assert_allclose(
        restored.pullback(weights), context.pullback(weights), rtol=0, atol=0
    )
    _, eigenpairs = diff.eig(operator)
    with pytest.raises(ValueError, match="unique largest component"):
        eigenpairs.pushforward(direction)


def test_eigenvalue_record_rejects_repeated_modes_but_allows_trace_pullback():
    values, context = diff.eigvals(np.eye(2))
    assert_allclose(values, 1)
    assert_allclose(context.pullback(np.ones(2, dtype=complex)), np.eye(2))
    with pytest.raises(ValueError, match="repeated eigenvalues"):
        context.pushforward(np.eye(2, dtype=complex))
    with pytest.raises(ValueError, match="repeated eigenvalues"):
        context.pullback(np.array([1.0, 0.0], dtype=complex))


@pytest.mark.parametrize("t", [-1e-5, 0.0, 1e-5])
def test_eigenvalue_order_and_derivative_weights_stay_aligned(t):
    direction = np.array([[0.2, 0.3], [-0.1, 0.5]], dtype=complex)
    operator = _phase_tie() + t * direction
    values, context = diff.eigvals(operator)
    assert np.all(np.diff(values.real) > 0)
    # Sorting values must also reorder the vectors used for both derivatives.
    raw, pairs = diff.eig(operator)
    order = np.lexsort((raw[0].imag, raw[0].real))
    assert_allclose(values, raw[0][order], atol=2e-14)
    weights = np.array([0.2 + 0.1j, -0.3 + 0.2j])
    raw_weights = np.empty_like(weights)
    raw_weights[order] = weights
    assert_allclose(
        context.pullback(weights),
        pairs.pullback(raw_weights, np.zeros_like(operator)),
        atol=2e-14,
    )


def test_eigenvalue_real_part_ordering_tie_requires_equal_loss_weights():
    operator = np.diag([1.0 + 1j, 1.0 - 1j])
    values, context = diff.eigvals(operator)
    assert_allclose(values, [1 - 1j, 1 + 1j])
    with pytest.raises(ValueError, match="distinct real parts"):
        context.pushforward(np.diag([1.0, -1.0]).astype(complex))
    with pytest.raises(ValueError, match="equal loss weights"):
        context.pullback(np.array([1.0, 0.0], dtype=complex))
    assert_allclose(context.pullback(np.ones(2, dtype=complex)), np.eye(2), atol=2e-14)


def test_eigenvalue_record_gradient_checker():
    check_pushforward(
        diff.eigvals,
        _phase_tie(),
        directions=(np.array([[0.2, 0.3], [-0.1, 0.5]], dtype=complex),),
        cotangents=np.array([0.2 + 0.1j, -0.3 + 0.2j]),
        step=1e-5,
        rtol=1e-8,
        atol=1e-9,
    )


@pytest.mark.parametrize("framework", ["jax", "torch", "autograd", "advect"])
def test_framework_eigenvalue_forward_and_reverse_at_phase_tie(framework):
    engine = pytest.importorskip(framework)
    adapter = importlib.import_module(f"treams_rs.{framework}")
    operator = _phase_tie()
    direction = np.array([[0.2, 0.3], [-0.1, 0.5]], dtype=complex)
    expected = 2 * np.trace(operator @ direction).real
    scope = jax_x64() if framework == "jax" else contextlib.nullcontext()
    with scope:
        if framework == "torch":
            operation = adapter.wrap(diff.eigvals)
            matrix = engine.tensor(operator)
            delta = engine.tensor(direction)

            def objective(x):
                return engine.real(engine.sum(operation(matrix + x * delta) ** 2))

            x = engine.tensor(0.0, dtype=engine.float64)
            _, tangent = engine.func.jvp(objective, (x,), (engine.ones_like(x),))
            x.requires_grad_(True)
            (gradient,) = engine.autograd.grad(objective(x), x)
            tangent, gradient = tangent.numpy(), gradient.numpy()
        else:
            xp = importlib.import_module(f"{framework}.numpy")
            operation = (
                adapter.eigvals
                if framework == "advect"
                else adapter.wrap(diff.eigvals, operator)
                if framework == "jax"
                else adapter.wrap(diff.eigvals)
            )

            def objective(x):
                return xp.real(xp.sum(operation(operator + x * direction) ** 2))

            x = xp.asarray(0.0)
            if framework == "jax":
                tangent = engine.jit(engine.jacfwd(objective))(x)
                gradient = engine.jit(engine.grad(objective))(x)
            elif framework == "autograd":
                _, tangent = engine.make_jvp(objective)(x)(np.asarray(1.0))
                gradient = engine.grad(objective)(x)
            else:
                _, tangent = engine.jvp(objective)(x, tangents=np.asarray(1.0))
                gradient = engine.grad(objective)(x)
        assert_allclose(tangent, expected, atol=2e-14)
        assert_allclose(gradient, expected, atol=2e-14)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
@pytest.mark.parametrize("size", [2, 3])
def test_rank_deficient_singular_values_use_relative_numerical_resolution(scale, size):
    vector = np.arange(1.0, size + 1)
    operator = np.outer(vector, vector) * scale
    values, context = diff.svdvals(operator)
    # The mathematically zero second value can be a small positive rounding error.
    assert values[-1] <= 64 * np.finfo(float).eps * values[0]
    with pytest.raises(ValueError, match="at zero or below numerical resolution"):
        context.pushforward(np.eye(size, dtype=complex) * scale)
    with pytest.raises(ValueError, match="at zero or below numerical resolution"):
        context.pullback(np.ones(size))
    # The squared Frobenius norm is smooth even at rank deficiency.
    assert_allclose(
        context.pullback(2 * values) / scale, 2 * operator / scale, atol=2e-14
    )


def test_tiny_positive_singular_value_is_numerically_unresolved_not_nondifferentiable():
    operator = np.diag([1.0, 1e-16])
    _, context = diff.svdvals(operator)
    with pytest.raises(ValueError, match="below numerical resolution"):
        context.pushforward(np.eye(2, dtype=complex))
    # A distinct positive value has a derivative mathematically. This rejection
    # is a numerical rank policy, invariant under common scaling of the matrix.
    step = 1e-18

    def small(x):
        return np.linalg.svd(np.diag([1.0, 1e-16 + x]), compute_uv=False)[1]

    assert_allclose((small(step) - small(-step)) / (2 * step), 1.0, atol=1e-14)


def test_unitary_individual_singular_values_lack_linear_pushforward():
    operator = np.eye(2, dtype=complex)
    _, context = diff.svdvals(operator)
    direction = np.diag([1.0, -1.0]).astype(complex)
    with pytest.raises(ValueError, match="repeated singular values"):
        context.pushforward(direction)
    step = 1e-5
    # Both signs of the perturbation increase the largest ordered value: the
    # map is 1 + abs(t), so no linear JVP of the individual values exists.
    for sign in (-1, 1):
        assert_allclose(
            np.linalg.svd(operator + sign * step * direction, compute_uv=False),
            [1 + step, 1 - step],
            atol=2e-14,
        )
    # Their sum is smooth: its valid pullback is independent of the SVD basis.
    assert_allclose(context.pullback(np.ones(2)), operator, atol=2e-14)

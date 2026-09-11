import advect
import advect.numpy as anp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import linear_sum_assignment

from treams_rs import _native, diff
from treams_rs import advect as ad


@pytest.mark.parametrize("size", [3, 65, 257])
def test_native_solve_strided_inputs_and_owned_residual(size):
    rng = np.random.default_rng(82)
    storage = rng.normal(size=(2 * size, 2 * size)).astype(complex)
    a = storage[::2, ::-2]
    a[np.diag_indices(size)] += 3 * size
    x = rng.normal(size=(size, size)) + 0.1j
    b = np.asfortranarray(a @ x)
    value, context = _native.linear_solve(a, b)
    assert_allclose(value, x, atol=1e-12)
    cotangent = rng.normal(size=(2 * size, 2 * size)).astype(complex)[::-2, ::2]
    gb = np.linalg.solve(a.conj().T, cotangent)
    ga = -gb @ x.conj().T
    a[:] = 0
    b[:] = 0
    gradients = context.pullback(cotangent)
    assert_allclose(gradients[0], ga, atol=1e-12)
    assert_allclose(gradients[1], gb, atol=1e-12)


@pytest.mark.parametrize("size", [1, 4, 7])
@pytest.mark.parametrize("complex_inputs", [False, True])
def test_linear_solve_reference_and_pullbacks(size, complex_inputs):
    rng = np.random.default_rng(125)
    a = np.eye(size) * 3 + rng.normal(size=(size, size)) * 0.1
    b = rng.normal(size=(size, 3))
    if complex_inputs:
        a = a + 0.1j * rng.normal(size=a.shape)
        b = b + 1j * rng.normal(size=b.shape)
    value, context = diff.solve(a, b)
    assert_allclose(value, np.linalg.solve(a, b), atol=1e-12)
    g = rng.normal(size=b.shape) + 1j * rng.normal(size=b.shape)
    gradients = context.pullback(g)
    directions = [rng.normal(size=v.shape) + 0.1j for v in (a, b)]
    h = 1e-5
    for i in range(2):
        plus, minus = [a, b], [a, b]
        plus[i] = plus[i] + h * directions[i]
        minus[i] = minus[i] - h * directions[i]
        numeric = np.vdot(
            g, (np.linalg.solve(*plus) - np.linalg.solve(*minus)) / (2 * h)
        ).real
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real, numeric, rtol=1e-8, atol=1e-10
        )
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


@given(scale=st.floats(0.1, 4), x=st.floats(-0.3, 0.3))
def test_linear_solve_common_scale_invariant(scale, x):
    a = np.array([[2, x + 0.1j], [0.2j, 1.4]])
    b = np.array([[0.2 + 0.1j], [0.5]])
    value, context = diff.solve(a, b)
    ga, gb = context.pullback(np.ones_like(value))
    assert_allclose(np.vdot(ga, a).real + np.vdot(gb, b).real, 0, atol=1e-12)
    assert_allclose(diff.solve(a * scale, b * scale)[0], value, atol=1e-12)


def _matrix():
    rng = np.random.default_rng(81)
    return np.diag(np.arange(1, 5)) + 0.15 * (
        rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    )


def _matched(operator, reference):
    (values, vectors), _ = diff.eig(operator)
    _, order = linear_sum_assignment(abs(reference[:, None] - values))
    return values[order], vectors[:, order]


@pytest.mark.parametrize("target", ["values", "vectors", "both"])
def test_general_complex_eigensystem_all_pullbacks(target):
    a = _matrix()
    (w, v), context = diff.eig(a)
    assert_allclose(a @ v, v * w, atol=1e-12)
    assert_allclose(np.linalg.norm(v, axis=0), 1, atol=1e-12)
    pivot = v[np.argmax(abs(v), axis=0), np.arange(len(w))]
    assert_allclose(pivot.imag, 0, atol=1e-12)
    assert np.all(pivot.real > 0)
    oracle = np.linalg.eigvals(a)
    _, order = linear_sum_assignment(abs(w[:, None] - oracle))
    assert_allclose(w, oracle[order], atol=1e-12)
    rng = np.random.default_rng(83)
    gw = (
        np.zeros_like(w)
        if target == "vectors"
        else rng.normal(size=w.shape) + 1j * rng.normal(size=w.shape)
    )
    gv = (
        np.zeros_like(v)
        if target == "values"
        else rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)
    )
    gradient = context.pullback(gw, gv)
    direction = rng.normal(size=a.shape) + 1j * rng.normal(size=a.shape)
    h = 1e-5
    wp, vp = _matched(a + h * direction, w)
    wm, vm = _matched(a - h * direction, w)
    numeric = (np.vdot(gw, wp - wm).real + np.vdot(gv, vp - vm).real) / (2 * h)
    assert_allclose(np.vdot(gradient, direction).real, numeric, rtol=2e-8, atol=2e-9)
    assert_allclose(np.vdot(gradient, a).real, np.vdot(gw, w).real, atol=2e-12)


@pytest.mark.parametrize("scale", [1e-200, 1e-100, 1e100, 1e200])
def test_eigen_extreme_common_scale(scale):
    a = _matrix()
    (w, v), _ = diff.eig(a)
    ws, vs = _matched(a * scale, w * scale)
    assert_allclose(ws / scale, w, atol=1e-12)
    assert_allclose(vs, v, atol=1e-12)
    gv = np.full_like(v, 0.2 + 0.1j)
    (_, _), context = diff.eig(a)
    gradient = context.pullback(np.zeros_like(w), gv)
    (ws, vs), context = diff.eig(a * scale)
    _, order = linear_sum_assignment(abs(w[:, None] * scale - ws))
    gs = np.empty_like(gv)
    gs[:, order] = gv
    scaled_gradient = context.pullback(np.zeros_like(w), gs)
    assert_allclose(scaled_gradient * scale, gradient, atol=1e-12)


@given(delta=st.floats(0.2, 2), real=st.floats(-0.5, 0.5), imag=st.floats(-0.5, 0.5))
def test_hermitian_spectrum_and_trace_square_gradient(delta, real, imag):
    a = np.array([[1 + delta, real + 1j * imag], [real - 1j * imag, 1 - delta]])
    (w, v), context = diff.eig(a)
    assert_allclose(
        np.sort(w.real),
        1 + np.array([-1, 1]) * np.sqrt(delta**2 + real**2 + imag**2),
        atol=1e-12,
    )
    assert_allclose(w.imag, 0, atol=1e-12)
    gradient = context.pullback(2 * w.conj(), np.zeros_like(v))
    assert_allclose(gradient, 2 * a.conj().T, atol=1e-12)


def test_repeated_modes_support_spectral_sums_not_individual_modes():
    a = np.eye(3, dtype=complex)
    (w, v), context = diff.eig(a)
    assert_allclose(context.pullback(np.ones_like(w), np.zeros_like(v)), a, atol=1e-12)
    (_, _), context = diff.eig(a)
    with pytest.raises(ValueError, match="repeated eigenvalues"):
        context.pullback(np.arange(3, dtype=complex), np.zeros_like(v))
    (_, _), context = diff.eig(a)
    with pytest.raises(ValueError, match="repeated eigenvalues"):
        context.pullback(np.zeros_like(w), np.ones_like(v))


def test_eigenvector_phase_pivot_tie():
    a = np.array([[1, 0.3j], [-0.3j, 1]])
    (w, v), context = diff.eig(a)
    assert_allclose(context.pullback(np.zeros_like(w), v), 0, atol=1e-12)
    (_, _), context = diff.eig(a)
    with pytest.raises(ValueError, match="unique largest component"):
        context.pullback(np.zeros_like(w), 1j * v)


def test_defective_eigensystem_trace_gradient():
    a = np.array([[1.0, 1.0], [0.0, 1.0]])
    (w, v), context = diff.eig(a)
    assert_allclose(a @ v, v * w, atol=1e-12)
    assert_allclose(
        context.pullback(np.full_like(w, 0.2 + 0.1j), np.zeros_like(v)),
        (0.2 + 0.1j) * np.eye(2),
        atol=1e-12,
    )


def test_eigen_bad_cotangent_does_not_consume_context():
    (w, v), context = diff.eig(_matrix())
    with pytest.raises(ValueError, match="shapes"):
        context.pullback(w[:1], v)
    with pytest.raises(ValueError, match="finite"):
        context.pullback(np.full_like(w, np.nan), v)
    context.pullback(np.ones_like(w), np.zeros_like(v))
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(w, v)


def test_advect_solve_and_eigenvectors():
    a = _matrix()
    b = np.ones((4, 2))

    def objective(operator):
        solution = ad.solve(operator, b)
        values, vectors = ad.eig(operator)
        return (
            anp.sum(anp.real(solution * anp.conj(solution)))
            + anp.sum(anp.real(values**2))
            + anp.sum(anp.real(vectors) * np.arange(16).reshape(4, 4))
        )

    gradient = advect.grad(objective)(a)
    direction = np.full_like(a, 0.03 + 0.01j)
    h = 1e-5
    assert_allclose(
        np.vdot(gradient, direction).real,
        (objective(a + h * direction) - objective(a - h * direction)) / (2 * h),
        rtol=1e-7,
        atol=1e-8,
    )

"""Dense linear algebra records (diff.solve, eig, svdvals and the packed LU): NumPy
references, spectral identities and pullbacks, at the diff and _native layers."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import linear_sum_assignment

# _native: one test checks the binding's solve with strided inputs directly.
from treams_rs import _native, diff
from treams_rs import advect as ad
from treams_rs.testing import check_pullback

from _support import complex_normal

pytestmark = pytest.mark.gradients


@pytest.mark.interface
@pytest.mark.parametrize("size", [3, 65, 257])
def test_native_solve_strided_inputs_and_owned_residual(size):
    rng = np.random.default_rng(82)
    storage = rng.normal(size=(2 * size, 2 * size)).astype(complex)
    a = storage[::2, ::-2]
    a[np.diag_indices(size)] += 3 * size
    x = rng.normal(size=(size, size)) + 0.1j
    b = np.asfortranarray(a @ x)
    value, context = _native.solve(a, b)
    assert_allclose(value, x, atol=1e-13)
    cotangent = rng.normal(size=(2 * size, 2 * size)).astype(complex)[::-2, ::2]
    gb = np.linalg.solve(a.conj().T, cotangent)
    ga = -gb @ x.conj().T
    a[:] = 0
    b[:] = 0
    gradients = context.pullback(cotangent)
    assert_allclose(gradients[0], ga, atol=1e-13)
    assert_allclose(gradients[1], gb, atol=1e-13)


@pytest.mark.reference
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
    assert_allclose(value, np.linalg.solve(a, b), atol=1e-13)
    g = complex_normal(rng, b.shape)
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


def _matrix():
    rng = np.random.default_rng(81)
    return np.diag(np.arange(1, 5)) + 0.15 * complex_normal(rng, (4, 4))


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
    gw = np.zeros_like(w) if target == "vectors" else complex_normal(rng, w.shape)
    gv = np.zeros_like(v) if target == "values" else complex_normal(rng, v.shape)
    gradient = context.pullback(gw, gv)
    direction = complex_normal(rng, a.shape)
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


@pytest.mark.reference
@pytest.mark.parametrize("shape", [(1, 1), (4, 4), (6, 3), (3, 6)])
def test_singular_values_reference_and_vjp(shape):
    rng = np.random.default_rng(82)
    a = complex_normal(rng, shape)
    s, context = diff.svdvals(a)
    assert_allclose(s, np.linalg.svd(a, compute_uv=False), atol=1e-12)
    weights = np.linspace(0.2, 1.1, len(s))
    gradient = context.pullback(weights)
    check_pullback(
        diff.svdvals,
        a,
        directions=(rng.normal(size=shape) + 0.13j,),
        cotangents=weights,
        step=1e-5,
        rtol=1e-8,
        atol=1e-10,
    )
    assert_allclose(np.vdot(gradient, a).real, np.dot(weights, s), atol=1e-12)
    assert_allclose(np.vdot(gradient, 1j * a).real, 0, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-200, 1e-100, 1e100, 1e200])
def test_singular_values_extreme_scale(scale):
    a = _matrix()
    s, context = diff.svdvals(a * scale)
    assert_allclose(s / scale, np.linalg.svd(a, compute_uv=False), atol=1e-12)
    gradient = context.pullback(2 * s)
    assert_allclose(gradient / scale, 2 * a, atol=1e-11)


@pytest.mark.parametrize(("x", "y"), [(-0.3, 0.2), (0.25, -0.1)])
def test_singular_value_energy_advect(x, y):
    a = np.array([[1.2, x + 0.2j], [y, 2.1j], [0.3, -0.1j]])

    def energy(a):
        s = ad.svdvals(a)
        return anp.sum(s**2)

    assert_allclose(energy(a), np.vdot(a, a).real, atol=1e-12)
    assert_allclose(advect.grad(energy)(a), 2 * a, atol=1e-12)


def test_singular_degenerate_and_rank_deficient_spectra():
    a = np.diag([2, 2, 0]).astype(complex)
    s, context = diff.svdvals(a)
    assert_allclose(context.pullback(2 * s), 2 * a, atol=1e-12)
    _, context = diff.svdvals(a)
    with pytest.raises(ValueError, match="repeated singular values"):
        context.pullback(np.array([1.0, 0.0, 0.0]))
    _, context = diff.svdvals(a)
    with pytest.raises(ValueError, match="at zero"):
        context.pullback(np.ones(3))
    s, context = diff.svdvals(np.zeros((3, 2)))
    assert_allclose(s, 0)
    assert_allclose(context.pullback(np.zeros(2)), np.zeros((3, 2)))


@given(
    permutation=st.permutations([0, 1, 2, 3]),
    scale=st.sampled_from([1e-200, 1.0, 1e200]),
)
def test_packed_lu_pivoting_and_adjoint(permutation, scale):
    a = np.array([[1, 0.2j, 0, 0], [0.1, 2, 0.3, 0], [0, 0.2, 3, 0.1j], [0, 0, 0.1, 4]])
    a = a[list(permutation)]
    b = np.arange(8).reshape(4, 2) + 0.2j
    value, residual = diff.solve(a * scale, b * scale)
    assert_allclose(value, np.linalg.solve(a, b), atol=2e-14)
    g = np.array([[0.3, 0.2j], [0.1j, 0.2], [0.7, 0.1], [0.5, 0.4j]])
    ga, gb = residual.pullback(g)
    expected = np.linalg.solve(a.conj().T, g)
    assert_allclose(gb * scale, expected, atol=2e-14)
    assert_allclose(ga * scale, -expected @ value.conj().T, atol=2e-14)

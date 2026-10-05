"""PyTorch's two forward-mode entry points compose native records and physics."""

import weakref

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.gradients

torch = pytest.importorskip("torch")

import treams_rs as tr  # noqa: E402
from treams_rs import diff  # noqa: E402
from treams_rs import torch as ad  # noqa: E402
from treams_rs._records import DerivativeContext  # noqa: E402
from treams_rs.testing import check_gradient  # noqa: E402


def _jvp(function, primals, tangents, interface):
    if interface == "func":
        return torch.func.jvp(function, primals, tangents)
    with torch.autograd.forward_ad.dual_level():
        inputs = tuple(
            torch.autograd.forward_ad.make_dual(primal, tangent)
            for primal, tangent in zip(primals, tangents, strict=True)
        )
        output = function(*inputs)
        if isinstance(output, tuple):
            pairs = tuple(torch.autograd.forward_ad.unpack_dual(v) for v in output)
            return tuple(v.primal for v in pairs), tuple(v.tangent for v in pairs)
        return torch.autograd.forward_ad.unpack_dual(output)


@pytest.mark.reference
@pytest.mark.parametrize("interface", ["func", "dual"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_solve_jvp_matches_torch_with_complex_and_strided_inputs(interface, dtype):
    generator = torch.Generator().manual_seed(7)
    matrix = torch.eye(3, dtype=dtype) * 3
    matrix += torch.randn((3, 3), dtype=dtype, generator=generator) * 0.1
    rhs = torch.randn((3, 4), dtype=dtype, generator=generator)
    primals = (matrix.T.conj(), rhs[:, ::2].conj())
    tangents = tuple(
        torch.randn(value.shape, dtype=dtype, generator=generator) for value in primals
    )
    actual = _jvp(ad.solve, primals, tangents, interface)
    expected = torch.func.jvp(
        torch.linalg.solve,
        tuple(value.to(torch.complex128) for value in primals),
        tuple(value.to(torch.complex128) for value in tangents),
    )
    for result, reference in zip(actual, expected, strict=True):
        assert result.dtype == torch.complex128
        assert_allclose(result.numpy(), reference.resolve_conj().numpy(), rtol=2e-13)


@pytest.mark.parametrize("interface", ["func", "dual"])
def test_forward_and_first_backward_share_context_then_replay_owned_inputs(interface):
    calls = []

    def record(matrix, rhs):
        calls.append(1)
        return diff.solve(matrix, rhs)

    matrix = np.array([[2.0, 0.1], [0.2, 1.5]])
    original = matrix.copy()
    rhs = torch.tensor([[0.4], [0.7]], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([[0.2], [-0.1]], dtype=torch.float64)
    output, tangent = _jvp(
        lambda b: ad.wrap(record)(matrix, b), (rhs,), (direction,), interface
    )
    assert len(calls) == 1
    assert_allclose(tangent.detach().numpy(), np.linalg.solve(original, direction))
    matrix[:] = 7.0
    expected = np.linalg.solve(original.T, np.ones((2, 1)))
    for index, retain in enumerate((True, False), start=1):
        (gradient,) = torch.autograd.grad(output.real.sum(), rhs, retain_graph=retain)
        assert_allclose(gradient.numpy(), expected)
        assert len(calls) == index


@pytest.mark.parametrize("interface", ["func", "dual"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_context_release_follows_primal_reverse_graph(interface, requires_grad):
    contexts = []

    def record(matrix, rhs):
        value, context = diff.solve(matrix, rhs)
        # Native residuals are held by these bound methods. The wrapper makes
        # their ownership observable without native implementation hooks.
        tracked = DerivativeContext(context.pullback, context.pushforward)
        contexts.append(weakref.ref(tracked))
        return value, tracked

    matrix = np.eye(2) * 2
    rhs = torch.ones((2, 1), dtype=torch.float64, requires_grad=requires_grad)
    output, tangent = _jvp(
        lambda b: ad.wrap(record)(matrix, b), (rhs,), (torch.ones_like(rhs),), interface
    )
    assert len(contexts) == 1
    assert (contexts[0]() is not None) == requires_grad
    if requires_grad:
        torch.autograd.grad(output.real.sum(), rhs, retain_graph=True)
        assert contexts[0]() is None
        torch.autograd.grad(output.real.sum(), rhs)
        assert len(contexts) == 2
        assert contexts[1]() is None
    # Outputs remain usable after native residuals are released.
    assert_allclose(output.detach().numpy(), np.full((2, 1), 0.5))
    assert_allclose(tangent.detach().numpy(), np.full((2, 1), 0.5))


@pytest.mark.parametrize("interface", ["func", "dual"])
def test_forward_primal_preserves_input_version_checks(interface):
    matrix = torch.eye(2, dtype=torch.float64) * 2
    rhs = torch.ones((2, 1), dtype=torch.float64, requires_grad=True)
    output, _ = _jvp(
        lambda b: ad.solve(matrix, b), (rhs,), (torch.ones_like(rhs),), interface
    )
    with torch.no_grad():
        rhs.add_(0.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(output.real.sum(), rhs)


@pytest.mark.parametrize("interface", ["func", "dual"])
def test_tuple_output_eigen_jvp_satisfies_differentiated_eigen_equation(interface):
    matrix = torch.tensor([[2.0 + 0.1j, 0.2], [0.1j, 3.0]], dtype=torch.complex128)
    direction = torch.tensor([[0.1j, 0.3], [0.2, -0.1j]], dtype=torch.complex128)
    (values, vectors), (dvalues, dvectors) = _jvp(
        ad.wrap(diff.eig), (matrix,), (direction,), interface
    )
    assert_allclose((matrix @ vectors).numpy(), (vectors * values).numpy(), atol=1e-14)
    assert_allclose(
        (direction @ vectors + matrix @ dvectors).numpy(),
        (dvectors * values + vectors * dvalues).numpy(),
        atol=1e-13,
    )


@pytest.mark.workflows
@pytest.mark.parametrize("interface", ["func", "dual"])
def test_public_sphere_scattering_forward_gradient(interface):
    def objective(radius):
        sphere = tr.sphere_tmatrix(k0=1.2, lmax=2, radius=radius, material=3 + 0.1j)
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        return sphere.cross_sections(wave).scattering

    def gradient(radius):
        primal = torch.tensor(radius, dtype=torch.float64)
        value, tangent = _jvp(
            objective, (primal,), (torch.ones_like(primal),), interface
        )
        assert_allclose(value.numpy(), objective(radius), rtol=2e-13)
        return tangent.numpy()

    check_gradient(objective, gradient, np.asarray(0.3))


@pytest.mark.workflows
@pytest.mark.parametrize("interface", ["func", "dual"])
def test_public_plane_wave_angle_field_forward_gradient(interface):
    def objective(theta):
        wave = tr.plane_wave_angle(theta, 0.2, "positive_helicity", k0=1.2)
        field = wave.efield([[0.1, 0.2, 1.3], [0.2, -0.1, 0.7]])
        return field.real.sum() + 0.3 * field.imag.sum()

    def gradient(theta):
        primal = torch.tensor(theta, dtype=torch.float64)
        _, tangent = _jvp(objective, (primal,), (torch.ones_like(primal),), interface)
        return tangent.numpy()

    check_gradient(objective, gradient, np.asarray(0.4))


def test_derivative_of_forward_tangent_is_explicitly_unsupported():
    matrix = torch.eye(2, dtype=torch.float64) * 2
    rhs = torch.ones((2, 1), dtype=torch.float64, requires_grad=True)
    _, tangent = torch.func.jvp(
        lambda b: ad.solve(matrix, b), (rhs,), (torch.ones_like(rhs),)
    )
    with pytest.raises(NotImplementedError, match="first-order"):
        torch.autograd.grad(tangent.real.sum(), rhs)

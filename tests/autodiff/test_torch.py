"""Optional PyTorch composition of native first-order scattering pullbacks."""

import re
import weakref

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytestmark = pytest.mark.gradients

torch = pytest.importorskip("torch")

import treams_rs as tr  # noqa: E402
from treams_rs import SphericalBasis, diff  # noqa: E402
from treams_rs import torch as ad  # noqa: E402


def _tensors(values):
    return tuple(torch.tensor(value, requires_grad=True) for value in values)


@pytest.mark.reference
def test_noncontiguous_and_conjugate_tensor_views():
    rng = np.random.default_rng(3)
    a = _tensors((np.eye(4) * 3 + 0.1j * rng.normal(size=(4, 4)),))[0]
    b = _tensors((rng.normal(size=(4, 4)) + 0.2j,))[0]
    actual = ad.solve(a.T.conj(), b[:, ::2].conj())
    reference = torch.linalg.solve(a.T.conj(), b[:, ::2].conj())
    weight = torch.tensor(rng.normal(size=(4, 2)) + 0.7j)
    actual_grad = torch.autograd.grad(actual, (a, b), grad_outputs=weight)
    reference_grad = torch.autograd.grad(reference, (a, b), grad_outputs=weight)
    assert_allclose(actual.detach().numpy(), reference.detach().numpy(), atol=1e-13)
    for actual_value, reference_value in zip(actual_grad, reference_grad, strict=True):
        assert_allclose(
            actual_value.resolve_conj().numpy(),
            reference_value.resolve_conj().numpy(),
            atol=1e-12,
        )


def test_repeat_backward_recomputes_only_after_first_pullback():
    calls = []

    def record(a, b):
        calls.append(1)
        return diff.solve(a, b)

    inputs = _tensors((np.eye(2) * (2 + 0.1j), np.array([[0.2j], [0.3 + 0.1j]])))
    output = ad.wrap(record)(*inputs)
    assert len(calls) == 1
    first = torch.autograd.grad(output.real.sum(), inputs, retain_graph=True)
    assert len(calls) == 1
    second = torch.autograd.grad(output.imag.sum(), inputs)
    assert len(calls) == 2
    _, real_context = diff.solve(*(x.detach().numpy() for x in inputs))
    _, imag_context = diff.solve(*(x.detach().numpy() for x in inputs))
    shape = tuple(output.shape)
    for actual, expected in zip(
        first, real_context.pullback(np.ones(shape, complex)), strict=True
    ):
        assert_allclose(actual.numpy(), expected, atol=1e-13)
    for actual, expected in zip(
        second, imag_context.pullback(np.full(shape, 1j)), strict=True
    ):
        assert_allclose(actual.numpy(), expected, atol=1e-13)


def test_composed_physics_fields_repeat_backward():
    radius = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    tm = ad.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3 + 0.1j)
    wave = ad.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    value = (abs(tm.scatter(wave).efield([[0.2, 0.3, 1.5]])) ** 2).sum()
    first = torch.autograd.grad(value, radius, retain_graph=True)[0]
    second = torch.autograd.grad(value, radius)[0]
    assert_allclose(first, second, rtol=1e-13)


def test_numpy_constant_mutation_cannot_change_repeated_backward():
    operator = np.eye(2) * (2.0 + 0.1j)
    original = operator.copy()
    rhs = _tensors((np.array([[0.2j], [0.3 + 0.1j]]),))[0]
    output = ad.solve(operator, rhs)
    operator[:] = 4.0
    first = torch.autograd.grad(output.real.sum(), rhs, retain_graph=True)[0]
    operator[:] = 7.0
    second = torch.autograd.grad(output.real.sum(), rhs)[0]
    expected = np.linalg.solve(original.conj().T, np.ones((2, 1), complex))
    assert_allclose(first.numpy(), expected, atol=1e-13)
    assert_allclose(second.numpy(), expected, atol=1e-13)


def test_saved_input_mutation_and_higher_order_are_rejected():
    a, b = _tensors((np.eye(2) * 2, np.ones((2, 1))))
    output = ad.solve(a, b)
    with torch.no_grad():
        a.add_(0.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(output.real.sum(), (a, b))
    output = ad.solve(a, b)
    with pytest.raises(NotImplementedError, match="first-order"):
        torch.autograd.grad(output.real.sum(), (a, b), create_graph=True)


@pytest.mark.interface
def test_read_only_numpy_constants_are_accepted_without_warning():
    # The suite turns warnings into errors; Torch warns on non-writable memory
    # and rejects negative strides, as in a reversed view.
    operator = np.broadcast_to(np.array([[2.0, 0.3], [0.1, 1.5]]), (2, 2))
    rhs = np.broadcast_to(np.array([[0.4], [0.7]]), (2, 1))
    expected, _ = diff.solve(operator, rhs)
    assert_allclose(ad.solve(operator, rhs).numpy(), expected, rtol=1e-15)
    reversed_operator = np.array([[0.1, 1.5], [2.0, 0.3]])[::-1]
    expected, _ = diff.solve(reversed_operator, rhs[::-1])
    actual = ad.solve(reversed_operator, rhs[::-1]).numpy()
    assert_allclose(actual, expected, rtol=1e-15)
    local = np.broadcast_to(np.array([[0.2 + 0.1j, 0.0], [0.0, 0.3]]), (2, 2))
    coupling = np.broadcast_to(np.array([[0.0, 0.1j], [0.1j, 0.0]]), (2, 2))
    (incident,) = _tensors((np.array([[1.0 + 0.0j], [0.5j]]),))
    actual = ad.wrap(diff.illuminate)(local, coupling, incident)
    expected, context = diff.illuminate(local, coupling, incident.detach().numpy())
    assert_allclose(actual.detach().numpy(), expected, rtol=1e-15)
    weight = np.array([[0.3 - 0.2j], [0.1j]])
    loss = (torch.tensor(weight).conj() * actual).real.sum()
    (gradient,) = torch.autograd.grad(loss, incident)
    assert_allclose(gradient.numpy(), context.pullback(weight)[2], rtol=1e-14)
    # The physics layer converts NumPy constants on its own path.
    basis = SphericalBasis.default(1)
    coefficients = np.linspace(0.1, 1.0, len(basis))[::-1]
    points = np.array([[1.3, 0.2, 0.1]])[:, ::-1]
    field = ad.wave(coefficients, basis=basis, k0=1.3).efield(points)
    expected = ad.wave(coefficients.copy(), basis=basis, k0=1.3).efield(points.copy())
    assert_array_equal(field.numpy(), expected.numpy())
    radii = np.array([0.3, 0.2])[::-1]
    tmatrix = ad.multilayer_sphere_tmatrix(
        k0=1.3, lmax=1, radii=radii, materials=[3.0, 2.0]
    )
    expected = ad.multilayer_sphere_tmatrix(
        k0=1.3, lmax=1, radii=radii.copy(), materials=[3.0, 2.0]
    )
    assert_array_equal(tmatrix.array.numpy(), expected.array.numpy())


@pytest.mark.interface
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_rejects_unsupported_dynamic_dtypes(dtype):
    # The message of the NumPy-side check, which names the dtype and the remedy.
    message = re.escape(
        "require float32, float64, complex64 or complex128 parameters; "
        f"received {str(dtype).removeprefix('torch.')}. Cast the parameter explicitly"
    )
    with pytest.raises(TypeError, match=message):
        ad.solve(torch.eye(2, dtype=dtype), torch.ones((2, 1), dtype=dtype))
    # Physical constructors share the check, including tensors inside lists.
    with pytest.raises(TypeError, match=message):
        ad.multilayer_sphere_tmatrix(
            k0=1.2,
            lmax=1,
            radii=[
                torch.tensor(0.3, dtype=torch.float64),
                torch.tensor(0.2, dtype=dtype),
            ],
            materials=(3.0, 2.0),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_single_precision_native_inputs_and_repeated_pullbacks(dtype):
    seen = []

    def record(a, b):
        seen.append((a.dtype, b.dtype))
        return diff.solve(a, b)

    a = torch.tensor([[2.0, 0.1], [0.2, 1.5]], dtype=dtype, requires_grad=True)
    b = torch.tensor([[0.4], [0.7]], dtype=dtype, requires_grad=True)
    if dtype == torch.complex64:
        a = (a + 0.1j).detach().requires_grad_()
        b = (b + 0.3j).detach().requires_grad_()
    output = ad.wrap(record)(a, b)
    assert output.dtype == torch.complex128  # native solve is complex-valued
    weight = np.array([[0.3 + 0.2j], [-0.1j]])
    reference_a = a.detach().numpy().astype(np.complex128)
    reference_b = b.detach().numpy().astype(np.complex128)
    expected, context = diff.solve(reference_a, reference_b)
    expected_gradients = context.pullback(weight)
    assert_allclose(output.detach().numpy(), expected, rtol=1e-14)
    for repeat in (True, False):
        gradients = torch.autograd.grad(
            output, (a, b), torch.tensor(weight), retain_graph=repeat
        )
        for actual, expected_gradient in zip(
            gradients, expected_gradients, strict=True
        ):
            assert actual.dtype == dtype
            if dtype == torch.float32:
                expected_gradient = expected_gradient.real
            assert_allclose(actual.numpy(), expected_gradient, rtol=2e-7, atol=1e-9)
    native_dtype = np.float64 if dtype == torch.float32 else np.complex128
    assert seen == [(native_dtype, native_dtype)] * 2


def test_default_precision_root_sphere_gradient():
    def objective(radius):
        sphere = tr.sphere_tmatrix(k0=2.0, lmax=2, radius=radius, material=3.0)
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=2.0)
        return sphere.cross_sections(wave).scattering

    radius = torch.tensor(0.2, requires_grad=True)
    value = objective(radius)
    (gradient,) = torch.autograd.grad(value, radius)
    assert value.dtype == torch.float64
    assert gradient.dtype == radius.dtype
    step = 1e-5
    expected_gradient = (objective(0.2 + step) - objective(0.2 - step)) / (2 * step)
    assert_allclose(value.detach().numpy(), objective(0.2), rtol=2e-7)
    assert_allclose(gradient.numpy(), expected_gradient, rtol=2e-7)


@pytest.mark.parametrize("parameter", ["k0", "medium"])
def test_plane_wave_angle_with_float_angles_and_tensor_parameter(parameter):
    def objective(x):
        k0, medium = (x, 2.0) if parameter == "k0" else (1.2, tr.Material(x))
        wave = tr.plane_wave_angle(0.3, 0.2, "positive_helicity", k0=k0, medium=medium)
        return wave.efield([[0.1, 0.2, 1.3]]).real.sum()

    x = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    value = objective(x)
    (gradient,) = torch.autograd.grad(value, x)
    step = 1e-5
    expected_gradient = (objective(1.5 + step) - objective(1.5 - step)) / (2 * step)
    assert_allclose(value.detach().numpy(), objective(1.5), rtol=1e-12)
    assert_allclose(gradient.numpy(), expected_gradient, rtol=2e-7)


def test_plane_wave_angle_rejects_complex_angles():
    k0 = torch.tensor(1.5, dtype=torch.float64)
    with pytest.raises(ValueError, match="real direction"):
        tr.plane_wave_angle(0.3 + 0.1j, 0.2, "positive_helicity", k0=k0)


def test_requested_illumination_matches_full_response_and_all_gradients():
    local = np.array([[0.2 + 0.1j, 0.03j], [0.05, 0.3 - 0.1j]])
    coupling = np.array([[0.0, 0.1 + 0.2j], [-0.03j, 0.0]])
    incident = np.array([[0.4 + 0.1j], [-0.2j]])

    def full_response(t, c, a):
        return torch.linalg.solve(torch.eye(2) - t @ c, t) @ a

    results = []
    for function in (ad.illuminate, full_response):
        inputs = _tensors((local, coupling, incident))
        output = function(*inputs)
        loss = (abs(output) ** 2).sum() + 0.1 * output.real.sum()
        results.append((output.detach(), *torch.autograd.grad(loss, inputs)))
    for actual, expected in zip(*results, strict=True):
        assert_allclose(actual.numpy(), expected.numpy(), atol=2e-13, rtol=2e-13)


def test_owned_recompute_snapshots_release_with_graph():
    references = []

    def pack(value):
        references.append(weakref.ref(value))
        return value

    inputs = _tensors((np.eye(2) * 2, np.ones((2, 1))))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output = ad.solve(*inputs)
    assert len(references) == 4
    assert all(reference() is not None for reference in references)
    torch.autograd.grad(output.real.sum(), inputs)
    assert all(reference() is None for reference in references[2:])

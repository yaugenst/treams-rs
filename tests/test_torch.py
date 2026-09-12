"""Optional PyTorch composition of native first-order scattering pullbacks."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

torch = pytest.importorskip("torch")

from treams_rs import SphericalWaveBasis, diff  # noqa: E402
from treams_rs import torch as ad  # noqa: E402


def _tensors(values):
    return tuple(torch.tensor(value, requires_grad=True) for value in values)


def _directional(record, values):
    """Check every dynamic argument against a real directional derivative."""
    inputs = _tensors(values)
    output = ad.wrap(record)(*inputs)
    rng = np.random.default_rng(28)
    weight = rng.normal(size=output.shape) + 1j * rng.normal(size=output.shape)
    loss = (torch.tensor(weight).conj() * output).real.sum()
    gradients = torch.autograd.grad(loss, inputs)
    step = 2e-6
    for i, (value, gradient) in enumerate(zip(values, gradients, strict=True)):
        direction = rng.normal(size=np.shape(value)) * 0.13
        if np.iscomplexobj(value):
            direction = direction + 0.09j
        plus, minus = list(values), list(values)
        plus[i] = value + step * direction
        minus[i] = value - step * direction
        expected = np.vdot(
            weight, (record(*plus)[0] - record(*minus)[0]) / (2 * step)
        ).real
        assert_allclose(
            np.vdot(gradient.numpy(), direction).real,
            expected,
            rtol=8e-6,
            atol=3e-8,
        )
        assert gradient.shape == inputs[i].shape
        assert gradient.dtype == inputs[i].dtype


@pytest.mark.ad_contract
@pytest.mark.parametrize("complex_inputs", [False, True])
def test_solve_analytic_adjoint_and_native_values(complex_inputs):
    a = np.array([[2.3, 0.2], [-0.1, 1.7]])
    b = np.array([[0.3, 0.1, 0.2], [0.5, -0.2, 0.7]])
    if complex_inputs:
        a = a + np.array([[0.1j, 0.2j], [-0.3j, 0.05j]])
        b = b + 0.13j
    inputs = _tensors((a, b))
    output = ad.solve(*inputs)
    weight = torch.tensor([[0.2 + 0.3j, -0.1j, 0.5], [0.7j, -0.4 + 0.2j, 0.1j]])
    gradients = torch.autograd.grad(output, inputs, grad_outputs=weight)
    expected = np.linalg.solve(a, b)
    gb = np.linalg.solve(a.conj().T, weight.numpy())
    ga = -gb @ expected.conj().T
    assert_allclose(output.detach().numpy(), expected, atol=1e-13)
    for gradient, reference in zip(gradients, (ga, gb), strict=True):
        assert_allclose(
            gradient.numpy(),
            reference if complex_inputs else reference.real,
            atol=1e-13,
        )
    _directional(diff.solve, (a, b))


@pytest.mark.ad_contract
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


@pytest.mark.ad_contract
def test_interaction_value_and_all_matrix_gradients():
    local = np.array([[0.2 + 0.1j, 0.03j], [0.05, 0.3 - 0.1j]])
    coupling = np.array([[0.0, 0.1 + 0.2j], [-0.03j, 0.0]])
    result = ad.interaction(*_tensors((local, coupling)))
    assert_allclose(
        result.detach().numpy(),
        np.linalg.solve(np.eye(2) - local @ coupling, local),
        atol=1e-13,
    )
    _directional(diff.interaction, (local, coupling))


@pytest.mark.ad_contract
def test_sphere_all_parameters_and_optional_materials():
    values = (
        np.asarray(1.2),
        np.array([0.18, 0.3]),
        np.array([3.2 + 0.1j, 2.1 + 0.05j, 1.0 + 0j]),
        np.array([1.1 + 0.02j, 1.2 + 0.01j, 1.0 + 0j]),
        np.array([0.1 + 0.01j, 0.05 + 0.02j, 0.0 + 0j]),
    )

    def record(k, r, e, m, c):
        return diff.sphere(1, float(k), r, e, m, c)

    _directional(record, values)
    inputs = _tensors(values)
    actual = ad.sphere(1, *inputs)
    expected, context = record(*values)
    assert_allclose(actual.detach().numpy(), expected, atol=1e-13)
    gradients = torch.autograd.grad(actual.real.sum(), inputs)
    for gradient, native in zip(
        gradients, context.pullback(np.ones_like(expected)), strict=True
    ):
        assert_allclose(gradient.numpy(), native, atol=1e-12)
    radius = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    default = ad.sphere(1, 1.2, radius, np.array([3.0, 1.0]))
    grad = torch.autograd.grad(default.real.sum(), radius)[0]
    _, context = diff.sphere(1, 1.2, [0.2], [3.0, 1.0])
    assert_allclose(
        grad.numpy(), context.pullback(np.ones_like(default.detach().numpy()))[1]
    )


@pytest.mark.ad_contract
def test_cluster_reorders_native_gradients_explicitly():
    def record(k, radii, epsilon, positions):
        output, context = diff.cluster(1, float(k), radii, epsilon, positions)

        def pullback(g):
            gr, gp, ge, gk = context.pullback(g)
            return gk, gr, ge, gp

        return output, pullback

    _directional(
        record,
        (
            np.asarray(1.2),
            np.array([0.2, 0.25]),
            np.array([3.0 + 0.1j, 2.3 + 0.2j]),
            np.array([[0.0, 0.0, 0.0], [0.2, 0.1, 1.5]]),
        ),
    )


@pytest.mark.ad_contract
@pytest.mark.parametrize("target", ["values", "vectors", "both"])
def test_eigen_tuple_outputs_materialize_unused_cotangents(target):
    rng = np.random.default_rng(81)
    operator = np.diag(np.arange(1, 5)) + 0.15 * (
        rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    )
    tensor = _tensors((operator,))[0]
    values, vectors = ad.wrap(diff.eig)(tensor)
    (nw, nv), context = diff.eig(operator)
    gw = rng.normal(size=nw.shape) + 1j * rng.normal(size=nw.shape)
    gv = rng.normal(size=nv.shape) + 1j * rng.normal(size=nv.shape)
    if target == "values":
        loss = (torch.tensor(gw).conj() * values).real.sum()
        gv[:] = 0
    elif target == "vectors":
        loss = (torch.tensor(gv).conj() * vectors).real.sum()
        gw[:] = 0
    else:
        loss = (torch.tensor(gw).conj() * values).real.sum()
        loss = loss + (torch.tensor(gv).conj() * vectors).real.sum()
    gradient = torch.autograd.grad(loss, tensor)[0]
    assert_allclose(values.detach().numpy(), nw, atol=1e-13)
    assert_allclose(vectors.detach().numpy(), nv, atol=1e-13)
    assert_allclose(gradient.numpy(), context.pullback(gw, gv), atol=1e-12)


@pytest.mark.ad_contract
def test_multiple_internal_illuminations():
    rng = np.random.default_rng(7)
    stacks = [
        0.05 * (rng.normal(size=(2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2)))
        for _ in range(2)
    ]
    for stack in stacks:
        stack[0, 0] += np.eye(2)
        stack[1, 1] += np.eye(2)
    up = rng.normal(size=(2, 3)) + 0.2j
    down = rng.normal(size=(2, 3)) + 0.1j
    _directional(diff.smatrix_illuminate, (*stacks, up, down))


@pytest.mark.ad_contract
@pytest.mark.parametrize("operator", [False, True])
def test_plane_field_weighted_and_operator_gradients(operator):
    points = np.array([[0.2, 0.3, -0.1], [0.3, -0.2, 0.1]])
    vectors = np.array([[0.3, 0.2, 1.3 + 0.1j], [1.5, -0.1, 0.2j]])

    if operator:

        def record(p, k):
            output, context = diff.plane_field(None, p, k, [0, 1])
            return output, lambda g: context.pullback(g)[1:]

        values = (points, vectors)
    else:

        def record(c, p, k):
            return diff.plane_field(c, p, k, [0, 1])

        values = (np.array([0.7 + 0.1j, -0.3 + 0.2j]), points, vectors)
    _directional(record, values)


@pytest.mark.ad_contract
def test_lattice_broadcast_and_all_geometry_parameters():
    def record(k, q, a, r, eta):
        return diff.lattice_sum(1, [2, 3], -1, k, q, a, r, eta, part="real")

    _directional(
        record,
        (
            np.array([[2.1 + 0.2j], [2.3 + 0.1j]]),
            np.asarray(0.13),
            np.asarray(1.7),
            np.array([[0.19, 0.11, 0.07], [0.21, -0.09, 0.05]]),
            np.asarray(0.9 + 0.03j),
        ),
    )


@pytest.mark.ad_contract
def test_bessel_broadcast_and_rotation_list_gradient():
    arguments = np.array([[1.2 + 0.2j], [2.1 - 0.1j]])
    z = _tensors((arguments,))[0]
    actual = ad.bessel(z, order=np.array([0.0, 1.0, 2.0]))
    expected, context = diff.bessel(np.array([0.0, 1.0, 2.0]), arguments)
    gradient = torch.autograd.grad(actual.real.sum(), z)[0]
    assert_allclose(actual.detach().numpy(), expected, atol=1e-13)
    assert_allclose(
        gradient.numpy(), context.pullback(np.ones_like(expected)), atol=1e-13
    )
    basis = SphericalWaveBasis.default(1)
    _directional(
        lambda angles: diff.rotation(angles, basis), (np.array([0.2, 0.3, -0.1]),)
    )


@pytest.mark.ad_contract
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


@pytest.mark.ad_contract
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


@pytest.mark.ad_contract
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


@pytest.mark.ad_contract
def test_pytorch_gradcheck_complex_solve_and_sphere_radius():
    a, b = _tensors(
        (np.array([[2.0 + 0.2j, 0.1], [-0.1j, 1.7]]), np.array([[0.3j], [0.2 + 0.1j]]))
    )
    assert torch.autograd.gradcheck(ad.solve, (a, b), atol=2e-6, rtol=1e-5)
    radius = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda r: ad.sphere(1, 1.2, r, np.array([3.0, 1.0])),
        (radius,),
        atol=2e-6,
        rtol=1e-5,
    )


@pytest.mark.ad_contract
@settings(max_examples=10, deadline=None)
@given(scale=st.floats(0.2, 4.0), coupling=st.floats(-0.3, 0.3))
def test_solve_common_scale_has_zero_directional_derivative(scale, coupling):
    a, b = _tensors(
        (
            np.array([[2.0, coupling + 0.1j], [0.2j, 1.4]]),
            np.array([[0.2 + 0.1j], [0.5]]),
        )
    )
    value = ad.solve(a, b)
    weight = torch.tensor([[0.2 + 0.7j], [-0.3 + 0.1j]])
    ga, gb = torch.autograd.grad(value, (a, b), grad_outputs=weight)
    assert_allclose(
        (torch.vdot(ga.ravel(), a.ravel()) + torch.vdot(gb.ravel(), b.ravel()))
        .real.detach()
        .numpy(),
        0,
        atol=1e-12,
    )
    assert_allclose(
        ad.solve(a * scale, b * scale).detach().numpy(),
        value.detach().numpy(),
        atol=1e-12,
    )


@pytest.mark.python_contract
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.int64])
def test_rejects_unsupported_dynamic_dtypes(dtype):
    with pytest.raises(TypeError, match="float64 or complex128"):
        ad.solve(torch.eye(2, dtype=dtype), torch.ones((2, 1), dtype=dtype))


def test_requested_illumination_matches_full_response_and_all_gradients():
    local = np.array([[0.2 + 0.1j, 0.03j], [0.05, 0.3 - 0.1j]])
    coupling = np.array([[0.0, 0.1 + 0.2j], [-0.03j, 0.0]])
    incident = np.array([[0.4 + 0.1j], [-0.2j]])
    values = (local, coupling, incident)
    inputs = _tensors(values)
    actual = ad.illuminate(*inputs)
    expected = (
        torch.linalg.solve(torch.eye(2) - inputs[0] @ inputs[1], inputs[0]) @ inputs[2]
    )
    assert_allclose(actual.detach().numpy(), expected.detach().numpy(), atol=1e-13)
    _directional(diff.illuminate, values)


def test_owned_recompute_snapshots_release_with_graph():
    import weakref

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

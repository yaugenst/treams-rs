"""HIPS Autograd composition of native first-order scattering pullbacks."""

import gc
import weakref

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytestmark = pytest.mark.gradients

autograd = pytest.importorskip("autograd")
anp = pytest.importorskip("autograd.numpy")

from treams_rs import autograd as ad  # noqa: E402
from treams_rs import diff  # noqa: E402
from treams_rs.testing import check_gradient  # noqa: E402


@pytest.mark.reference
@pytest.mark.parametrize("complex_inputs", [False, True])
def test_solve_matches_autograd_numpy_for_all_arguments(complex_inputs):
    operator = np.array([[2.0, 0.2], [0.1, 1.5]])
    rhs = np.array([[0.3], [0.7]])
    if complex_inputs:
        operator = operator + np.array([[0.2j, -0.1j], [0.3j, 0.1j]])
        rhs = rhs + np.array([[0.1j], [-0.3j]])

    def objective(solve, a, b):
        value = solve(a, b)
        return anp.sum(anp.abs(value) ** 2) + 0.2 * anp.sum(anp.imag(value))

    def reference(a, b):
        return objective(anp.linalg.solve, a, b)

    def actual(a, b):
        return objective(ad.solve, a, b)

    assert_allclose(actual(operator, rhs), reference(operator, rhs), atol=1e-14)
    for observed, expected in zip(
        autograd.grad(actual, (0, 1))(operator, rhs),
        autograd.grad(reference, (0, 1))(operator, rhs),
        strict=True,
    ):
        assert_allclose(observed, expected, atol=2e-14)
    # Real inputs receive real gradients even when the native output is complex.
    if not complex_inputs:
        assert not np.iscomplexobj(autograd.grad(actual)(operator, rhs))


def test_repeated_pullback_reuses_original_forward_and_keeps_input_snapshots():
    calls = []

    def record(a, b):
        calls.append(1)
        return diff.solve(a, b)

    operator = np.array([[2.0 + 0.2j, 0.1], [0.2j, 1.5]])
    original = operator.copy()
    rhs = np.array([[0.3 + 0.1j], [0.7 - 0.2j]])
    operation = ad.wrap(record)
    pullback, output = autograd.make_vjp(operation, (0, 1))(operator, rhs)
    reference, expected_output = autograd.make_vjp(anp.linalg.solve, (0, 1))(
        original, rhs
    )
    assert_allclose(output, expected_output, atol=2e-14)
    assert len(calls) == 1
    operator[:] = 5.0
    first = np.ones_like(output)
    second = np.full_like(output, 0.2 + 0.3j)
    results = []
    for cotangent in (first, second, first):
        actual = pullback(cotangent)
        results.append(actual)
        assert len(calls) == 1
        for observed, gradient in zip(actual, reference(cotangent), strict=True):
            assert_allclose(observed, gradient, atol=2e-14)
    for first_gradient, repeated_gradient in zip(results[0], results[2], strict=True):
        assert_array_equal(first_gradient, repeated_gradient)


def test_wrap_preserves_tuple_outputs_and_complex_cotangents():
    matrix = np.array([[1.3 + 0.1j, 0.2j], [0.1, 2.1 - 0.2j]])
    pullback, output = autograd.make_vjp(ad.wrap(diff.eig))(matrix)
    assert isinstance(output, tuple) and len(output) == 2
    cotangent = (np.array([0.2 + 0.1j, -0.3j]), np.full((2, 2), 0.1 - 0.2j))
    expected, context = diff.eig(matrix)
    for actual, value in zip(output, expected, strict=True):
        assert_array_equal(actual, value)
    gradient = context.pullback(*(g.conj() for g in cotangent)).conj()
    assert_allclose(pullback(cotangent), gradient, atol=1e-13)

    def eigenvalue_loss(a):
        return anp.sum(anp.real(ad.wrap(diff.eig)(a)[0]))

    assert_allclose(autograd.grad(eigenvalue_loss)(matrix), np.eye(2), atol=1e-13)


@pytest.mark.workflows
@pytest.mark.parametrize("epsilon", [3.0, 3.0 + 0.1j])
def test_sphere_radius_material_and_field_gradients(epsilon):
    def objective(radius, epsilon):
        sphere = ad.sphere_tmatrix(
            k0=1.2, lmax=1, radius=radius, material=ad.Material(epsilon)
        )
        wave = ad.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        field = sphere.scatter(wave).efield([[0.3, 0.2, 1.5]])
        return anp.sum(anp.abs(field) ** 2)

    check_gradient(
        objective,
        lambda r, e: tuple(g.conj() for g in autograd.grad(objective, (0, 1))(r, e)),
        np.asarray(0.2),
        np.asarray(epsilon),
        step=1e-5,
        rtol=2e-5,
        atol=1e-10,
    )


@pytest.mark.workflows
def test_nested_parameter_lists_preserve_sphere_gradients():
    def objective(radius):
        sphere = ad.multilayer_sphere_tmatrix(
            k0=1.2, lmax=1, radii=[radius, 0.3], materials=[3.0, 2.0]
        )
        wave = ad.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        return sphere.cross_sections(wave).scattering

    check_gradient(
        objective,
        autograd.grad(objective),
        np.asarray(0.2),
        step=1e-5,
        rtol=2e-5,
        atol=1e-10,
    )


@pytest.mark.workflows
def test_nested_real_and_complex_wave_coefficients_preserve_real_gradient():
    def objective(amplitude):
        wave = ad.wave(
            [amplitude, 0.2j, 0.1, 0.3, 0.0, 0.0],
            basis=ad.SphericalBasis.default(1),
            k0=1.2,
        )
        return anp.sum(anp.abs(wave.efield([[0.3, 0.2, 1.5]])) ** 2)

    check_gradient(objective, autograd.grad(objective), np.asarray(0.4))


@pytest.mark.workflows
def test_boxed_sequence_parameters_preserve_each_sphere_gradient():
    def objective(radii):
        sphere = ad.multilayer_sphere_tmatrix(
            k0=1.2, lmax=1, radii=radii, materials=[3.0, 2.0]
        )
        return anp.sum(anp.abs(sphere.array) ** 2)

    radii = np.array([0.2, 0.3])
    check_gradient(objective, autograd.grad(objective), radii)
    assert_allclose(
        autograd.grad(objective)(radii.tolist()),
        autograd.grad(objective)(radii),
        rtol=1e-13,
    )


@pytest.mark.interface
def test_higher_order_differentiation_is_rejected():
    def objective(value):
        return anp.real(ad.bessel(value, order=1))

    with pytest.raises(NotImplementedError, match="first-order"):
        autograd.grad(autograd.grad(objective))(0.3)
    with pytest.raises(NotImplementedError, match="first-order"):
        autograd.grad(lambda x: autograd.make_jvp(objective)(x)(1.0)[1])(0.3)
    with pytest.raises(NotImplementedError, match="first-order"):
        autograd.make_jvp(autograd.grad(objective))(0.3)(1.0)


@pytest.mark.parametrize("complex_inputs", [False, True])
@pytest.mark.parametrize("argnums", [(0,), (1,), (0, 1)])
def test_forward_solve_matches_linearized_equation_and_reuses_forward(
    complex_inputs, argnums
):
    operator = np.array([[2.0, 0.2], [0.1, 1.5]])
    rhs = np.array([[0.3, 0.1], [0.7, -0.2]])
    directions = (np.full_like(operator, 0.1), np.full_like(rhs, 0.2))
    if complex_inputs:
        operator = operator + 0.1j * np.eye(2)
        rhs = rhs + 0.2j
        directions = tuple(direction + 0.1j for direction in directions)
    calls = []

    def record(a, b):
        calls.append(1)
        return diff.solve(a, b)

    value, tangent = autograd.make_jvp(ad.wrap(record), argnums)(operator, rhs)(
        tuple(directions[i] for i in argnums)
    )
    da = directions[0] if 0 in argnums else np.zeros_like(operator)
    db = directions[1] if 1 in argnums else np.zeros_like(rhs)
    expected = np.linalg.solve(operator, rhs)
    assert_allclose(value, expected, atol=1e-14)
    assert_allclose(tangent, np.linalg.solve(operator, db - da @ expected), atol=1e-14)
    assert len(calls) == 1


def test_forward_wrap_preserves_multiple_outputs():
    matrix = np.array([[1.3 + 0.1j, 0.2j], [0.1, 2.1 - 0.2j]])
    direction = np.array([[0.2j, 0.1], [-0.1, 0.3j]])
    output, tangent = autograd.make_jvp(ad.wrap(diff.eig))(matrix)(direction)
    expected, context = diff.eig(matrix)
    expected_tangent = context.pushforward(direction)
    for value, wanted, derivative, wanted_derivative in zip(
        output, expected, tangent, expected_tangent, strict=True
    ):
        assert_allclose(value, wanted, atol=1e-14)
        assert_allclose(derivative, wanted_derivative, atol=1e-14)


@pytest.mark.workflows
def test_public_sphere_forward_sensitivity_matches_reverse_pairing():
    import treams_rs as tr

    def objective(radius, epsilon):
        sphere = tr.sphere_tmatrix(
            k0=1.2, lmax=1, radius=radius, material=tr.Material(epsilon)
        )
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        field = sphere.scatter(wave).efield([[0.3, 0.2, 1.5]])
        return anp.sum(anp.abs(field) ** 2)

    inputs = (np.asarray(0.2), np.asarray(3.0 + 0.1j))
    tangents = (np.asarray(0.05), np.asarray(0.2 - 0.1j))
    value, tangent = autograd.make_jvp(objective, (0, 1))(*inputs)(tangents)
    gradients = autograd.grad(objective, (0, 1))(*inputs)
    expected = sum(np.real(g * t) for g, t in zip(gradients, tangents, strict=True))
    assert_allclose(value, objective(*inputs), rtol=1e-13)
    assert_allclose(tangent, expected, rtol=1e-12, atol=1e-14)


def test_public_numerical_forward_mode_preserves_broadcasting():
    from treams_rs import special

    orders = np.array([0, 1, 2])
    z = np.asarray(0.8 + 0.2j)
    direction = np.asarray(0.2 - 0.1j)
    value, tangent = autograd.make_jvp(lambda x: special.jv(orders, x))(z)(direction)
    expected = special.jv(orders, z)
    derivative = (special.jv(orders - 1, z) - special.jv(orders + 1, z)) / 2
    assert_allclose(value, expected, atol=1e-14)
    assert_allclose(tangent, derivative * direction, atol=1e-14)


@pytest.mark.interface
@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.int64])
def test_wrap_rejects_unsupported_dynamic_dtypes(dtype):
    with pytest.raises(TypeError, match="require float64 or complex128"):
        ad.solve(np.eye(2, dtype=dtype), np.ones((2, 1), dtype=dtype))


@pytest.mark.interface
def test_wrap_registration_and_unused_contexts_do_not_accumulate():
    from autograd.core import primitive_jvps, primitive_vjps

    references = []

    class Pullback:
        def __call__(self, g):
            return g

        def pushforward(self, direction):
            return direction

    def record(value):
        context = Pullback()
        references.append(weakref.ref(context))
        return value, context

    count = len(primitive_vjps)
    jvp_count = len(primitive_jvps)
    for _ in range(3):
        pullback, _ = autograd.make_vjp(ad.wrap(record))(np.array([0.2, 0.3]))
        assert references[-1]() is not None
        del pullback
        autograd.make_jvp(ad.wrap(record))(np.array([0.2, 0.3]))(np.ones(2))
    gc.collect()
    assert all(reference() is None for reference in references)
    assert len(primitive_vjps) == count
    assert len(primitive_jvps) == jvp_count

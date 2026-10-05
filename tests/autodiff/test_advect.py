"""Native black-box composition through a real external autodiff framework."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import advect
import advect.numpy as anp
import numpy as np
import pytest
from advect import grad, vjp
from hypothesis import given
from hypothesis import strategies as st

from treams_rs import SphericalBasis, diff
from treams_rs import advect as ad
from treams_rs.testing import check_gradient

pytestmark = pytest.mark.gradients


def objective(radius, epsilon):
    value = ad.sphere_cluster(
        1, 1.2, radius, epsilon, anp.array([[0, 0, 0], [0, 0, 1.5]])
    )
    return anp.sum(anp.real(value * anp.conj(value))) + 0.03 * anp.sum(anp.real(value))


@given(radius=st.floats(0.1, 0.35), epsilon=st.floats(1.1, 6.0))
def test_advect_cluster_objective(radius, epsilon):
    check_gradient(
        objective,
        grad(objective, argnums=(0, 1)),
        np.array([radius, 0.25]),
        np.array([epsilon + 0.1j, 3 + 0.2j]),
        directions=(np.array([0.2, -0.3]), np.array([0.2 + 0.3j, -0.1 + 0.2j])),
        rtol=2e-6,
        atol=1e-9,
    )


def test_branched_objective_and_independent_thread_residuals():
    def evaluate(radius):
        return grad(lambda x: objective(anp.array([x, 0.25]), anp.array([3.0, 2.0])))(
            radius
        )

    inputs = [0.1, 0.2, 0.3, 0.35]
    expected = list(map(evaluate, inputs))
    with ThreadPoolExecutor(max_workers=4) as pool:
        np.testing.assert_allclose(
            list(pool.map(evaluate, inputs)), expected, rtol=1e-12
        )


def test_explicit_single_use_contract():
    def function(r):
        return anp.real(ad.sphere(1, 1.2, anp.array([r]), anp.array([3.0, 1.0]))).sum()

    _, pullback = vjp(function)(0.2)
    pullback(1.0)
    with pytest.raises(RuntimeError, match="consumed"):
        pullback(1.0)


@pytest.mark.interface
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_complex_cotangent_and_input_dtype(dtype):
    local = np.array([[0.2 + 0.1j, 0.03j], [0.05, 0.3 - 0.1j]])
    if np.dtype(dtype).kind != "c":
        local = local.real
    local = local.astype(dtype)
    coupling = np.array([[0.0, 0.1j], [-0.2j, 0.0]])
    weight = np.array([[0.4 + 0.3j, 0.2j], [0.7 - 0.2j, -0.1j]])
    # Ordinary NumPy objective, with an arbitrary complex output cotangent.
    _, pullback = vjp(lambda value: ad.interaction(value, coupling))(local)
    gradient = pullback(weight)
    assert gradient.dtype == local.dtype
    assert gradient.shape == local.shape
    direction = np.array([[0.2, -0.1], [0.3, 0.1]], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        direction += 0.2j
    step = 2e-3 if np.dtype(dtype).itemsize <= 8 else 1e-6
    numerical = np.real(
        np.vdot(
            weight,
            (
                ad.interaction(local + step * direction, coupling)
                - ad.interaction(local - step * direction, coupling)
            )
            / (2 * step),
        )
    )
    np.testing.assert_allclose(
        np.real(np.vdot(gradient, direction)), numerical, rtol=5e-5, atol=1e-7
    )


@pytest.mark.parametrize(
    "path", ["direct", "grad", "jvp", "consume", "close", "drop", "exception"]
)
def test_native_residual_released_on_every_exit(monkeypatch, path):
    original = diff.sphere
    contexts = []

    class ObservedContext:
        def __init__(self, native):
            self.native = native

        def pullback(self, cotangent):
            return self.native.pullback(cotangent)

        def pushforward(self, *tangents):
            return self.native.pushforward(*tangents)

    def forward(*args, **kwargs):
        value, native = original(*args, **kwargs)
        context = ObservedContext(native)
        contexts.append(weakref.ref(context))
        return value, context

    monkeypatch.setattr(diff, "sphere", forward)

    def loss(radius):
        value = ad.sphere(1, 1.2, anp.array([radius]), np.array([3.0, 1.0]))
        if path == "exception":
            raise RuntimeError("objective failed after forward")
        return np.real(value).sum()

    if path == "direct":
        loss(0.2)
    elif path == "grad":
        grad(loss)(0.2)
    elif path == "jvp":
        advect.jvp(loss)(0.2, tangents=1.0)
    elif path == "exception":
        with pytest.raises(RuntimeError, match="objective failed") as caught:
            grad(loss)(0.2)
        assert caught.value.__traceback__ is not None
    else:
        _, pullback = vjp(loss)(0.2)
        assert contexts[0]() is not None
        if path == "consume":
            pullback(1.0)
        elif path == "close":
            pullback.close()
        else:
            del pullback
            gc.collect()
    assert len(contexts) == 1
    assert all(context() is None for context in contexts)


@pytest.mark.interface
@pytest.mark.parametrize(
    "transform,error,match",
    [
        (
            lambda f, x: advect.grad(advect.grad(f))(x),
            advect.TracingError,
            "first-order",
        ),
        (lambda f, x: advect.stage(f, x), advect.TracingError, "abstract"),
        (
            lambda f, x: advect.grad(advect.checkpoint(f))(x),
            advect.TracingError,
            "residual primitive",
        ),
    ],
)
def test_unsupported_transforms_fail_before_native_execution(
    monkeypatch, transform, error, match
):
    def unexpected_forward(*args, **kwargs):
        pytest.fail("unsupported transform invoked the native solver")

    monkeypatch.setattr(diff, "sphere", unexpected_forward)

    def loss(radius):
        return np.real(
            ad.sphere(1, 1.2, anp.array([radius]), np.array([3.0, 1.0]))
        ).sum()

    with pytest.raises(error, match=match):
        transform(loss, 0.2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_forward_solve_matches_linearized_equation(dtype):
    operator = np.array([[2.0, 0.2], [0.1, 1.5]], dtype=dtype)
    rhs = np.array([[0.3, 0.1], [0.7, -0.2]], dtype=dtype)
    da, db = np.full_like(operator, 0.1), np.full_like(rhs, 0.2)
    if np.dtype(dtype).kind == "c":
        operator += 0.1j * np.eye(2)
        rhs += 0.2j
        da += 0.05j
        db += 0.1j
    value, tangent = advect.jvp(ad.solve, argnums=(0, 1))(
        operator, rhs, tangents=(da, db)
    )
    expected = np.linalg.solve(operator.astype(complex), rhs.astype(complex))
    np.testing.assert_allclose(value, expected, atol=1e-14)
    np.testing.assert_allclose(
        tangent,
        np.linalg.solve(operator.astype(complex), db - da @ expected),
        atol=1e-14,
    )


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
    value, tangent = advect.jvp(objective, argnums=(0, 1))(*inputs, tangents=tangents)
    gradients = advect.grad(objective, argnums=(0, 1))(*inputs)
    expected = sum(np.vdot(g, t).real for g, t in zip(gradients, tangents, strict=True))
    np.testing.assert_allclose(value, objective(*inputs), rtol=1e-13)
    np.testing.assert_allclose(tangent, expected, rtol=1e-12, atol=1e-14)


def test_public_numerical_forward_mode_preserves_broadcasting():
    from treams_rs import special

    orders = np.array([0, 1, 2])
    z = np.asarray(0.8 + 0.2j)
    direction = np.asarray(0.2 - 0.1j)
    value, tangent = advect.jvp(lambda x: special.jv(orders, x))(z, tangents=direction)
    derivative = (special.jv(orders - 1, z) - special.jv(orders + 1, z)) / 2
    np.testing.assert_allclose(value, special.jv(orders, z), atol=1e-14)
    np.testing.assert_allclose(tangent, derivative * direction, atol=1e-14)


def test_forward_eigenvalue_and_vector_outputs_preserve_packing():
    matrix = np.array([[1.3 + 0.1j, 0.2j], [0.1, 2.1 - 0.2j]])
    direction = np.array([[0.2j, 0.1], [-0.1, 0.3j]])
    output, tangent = advect.jvp(ad.eig)(matrix, tangents=direction)
    expected, context = diff.eig(matrix)
    expected_tangent = context.pushforward(direction)
    for value, wanted, derivative, wanted_derivative in zip(
        output, expected, tangent, expected_tangent, strict=True
    ):
        np.testing.assert_allclose(value, wanted, atol=1e-14)
        np.testing.assert_allclose(derivative, wanted_derivative, atol=1e-14)


def test_linearize_reuses_the_native_context_for_independent_directions(monkeypatch):
    from treams_rs import special

    calls = []
    original = diff.bessel

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(diff, "bessel", record)
    z = np.array([0.4 + 0.2j, 0.7 - 0.1j])
    value, linear = advect.linearize(lambda x: ad.bessel(x, order=1), z)
    derivative = (special.jv(0, z) - special.jv(2, z)) / 2
    directions = (np.ones_like(z), np.array([0.2j, 0.3 - 0.1j]))
    try:
        np.testing.assert_allclose(value, special.jv(1, z), atol=1e-14)
        for direction in directions:
            np.testing.assert_allclose(
                linear(direction), derivative * direction, atol=1e-14
            )
        for actual, direction in zip(
            linear.apply_many(directions), directions, strict=True
        ):
            np.testing.assert_allclose(actual, derivative * direction, atol=1e-14)
        for cotangent in directions:
            np.testing.assert_allclose(
                linear.pullback(cotangent),
                derivative.conj() * cotangent,
                atol=1e-14,
            )
        assert len(calls) == 1
    finally:
        linear.close()


def test_forward_expansion_preserves_axial_group_input_order():
    from treams_rs import CylindricalBasis

    basis = CylindricalBasis.default([-0.2, 0.1, 0.3], 1)
    groups = np.array([0.3, -0.2, 0.1])
    direction = np.array([0.1, -0.2, 0.3])

    def objective(kzs):
        value = ad.expansion(
            [[0.3, 0.2, -0.1]],
            [[-0.1, -0.2, 0.3]],
            [1.3 + 0.05j, 1.5 + 0.07j],
            destination=basis,
            source=basis,
            kzs=kzs,
        )
        return anp.real(value).sum() + 0.2 * anp.imag(value).sum()

    _, tangent = advect.jvp(objective)(groups, tangents=direction)
    np.testing.assert_allclose(
        tangent, np.vdot(advect.grad(objective)(groups), direction).real, atol=1e-13
    )
    check_gradient(
        objective,
        advect.grad(objective),
        groups,
        directions=(direction,),
        step=1e-6,
        rtol=1e-6,
        atol=2e-9,
    )


@pytest.mark.parametrize("fixed_vectors", [False, True])
def test_plane_field_operator_forward_without_amplitudes(fixed_vectors):
    points = np.array([[0.1, 0.2, 0.3]])
    vectors = np.array([[0.4, 0.5, 0.8]], dtype=complex)
    direction = np.full_like(points, 0.1)

    def operation(p):
        return ad.plane_field(
            None, p, vectors, polarizations=[1], fixed_vectors=fixed_vectors
        )

    value, tangent = advect.jvp(operation)(points, tangents=direction)
    expected, context = diff.plane_field(
        None, points, vectors, [1], fixed_vectors=fixed_vectors
    )
    np.testing.assert_allclose(value, expected, atol=1e-14)
    np.testing.assert_allclose(
        tangent,
        context.pushforward(
            np.empty(0, dtype=complex), direction, np.zeros_like(vectors)
        ),
        atol=1e-14,
    )


@given(z=st.floats(0.8, 2.0), imaginary=st.floats(0.01, 0.2))
def test_expansion_position_and_complex_wavenumber_gradients(z, imaginary):
    basis = SphericalBasis.default(2)
    positions = np.array([[0.1, 0.2, z]])
    ks = np.array([1.2 + imaginary * 1j, 1.3 + imaginary * 1j])

    def loss(position, wavenumbers):
        matrix = ad.expansion(
            position,
            [[0, 0, 0]],
            wavenumbers,
            destination=basis,
            source=basis,
            singular=True,
        )
        return np.sum(np.sin(np.real(matrix)) + np.imag(matrix) * 0.03)

    check_gradient(
        loss,
        grad(loss, argnums=(0, 1)),
        positions,
        ks,
        directions=(np.array([[0.1, -0.2, 0.3]]), np.array([0.2 + 0.1j, -0.1 + 0.3j])),
        rtol=2e-6,
        atol=1e-5,
    )


@pytest.mark.gradients
@pytest.mark.parametrize("fixed_q", [False, True])
def test_layer_stack_scalar_thickness_gradient_keeps_primal_shape(fixed_q):
    from treams_rs import diff

    # The native thickness gradient has shape (1,); Advect restores the scalar.
    ks = 1.2 * np.array([[1.0, 1.0], [1.5, 1.6], [1.0, 1.0]], dtype=complex)
    zs = np.array([1.0, 0.65, 1.0], dtype=complex)
    q = np.array([[0.1, 0.05]])
    rng = np.random.default_rng(7)
    weight = rng.normal(size=(1, 2, 2, 2, 2)) + 1j * rng.normal(size=(1, 2, 2, 2, 2))

    def loss(thickness):
        value = ad.layer_stack(ks, zs, q, thickness, fixed_q=fixed_q)
        return anp.sum(anp.real(anp.conj(weight) * value))

    gradient = grad(loss)(np.asarray(0.3))
    assert np.shape(gradient) == ()
    _, context = diff.layer_stack(ks, zs, q, np.asarray(0.3), fixed_q=fixed_q)
    native_gradient = context.pullback(weight)[3]
    assert native_gradient.shape == ()
    np.testing.assert_allclose(gradient, native_gradient.real, rtol=1e-13)
    direction = np.asarray(0.1)
    _, tangent = advect.jvp(loss)(np.asarray(0.3), tangents=direction)
    np.testing.assert_allclose(tangent, gradient * direction, rtol=1e-13)


@pytest.mark.parametrize("fixed_q", [False, True])
def test_layer_stack_scalar_direction_preserves_native_validation(fixed_q):
    ks = np.array([[1.2, 1.2], [2.0, 2.0], [1.3, 1.3]], dtype=complex)
    zs = np.array([1.0, 0.7, 1.0], dtype=complex)
    q = np.array([[0.2, 0.3]])
    _, context = diff.layer_stack(ks, zs, q, 0.3, fixed_q=fixed_q)
    zeros = (np.zeros_like(ks), np.zeros_like(zs), np.zeros_like(q))
    for invalid in (np.array(np.nan), np.array(np.inf)):
        with pytest.raises(ValueError, match=r"tangent.*finite"):
            context.pushforward(*zeros, invalid)
    with pytest.raises(ValueError, match=r"tangent.*shape"):
        context.pushforward(*zeros, np.zeros((1, 1)))
    scalar = context.pushforward(*zeros, np.asarray(0.1))
    vector = context.pushforward(*zeros, np.array([0.1]))
    np.testing.assert_allclose(scalar, vector, rtol=1e-14)

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
    "path", ["direct", "grad", "consume", "close", "drop", "exception"]
)
def test_native_residual_released_on_every_exit(monkeypatch, path):
    original = diff.sphere
    contexts = []

    class ObservedContext:
        def __init__(self, native):
            self.native = native

        def pullback(self, cotangent):
            return self.native.pullback(cotangent)

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
    assert contexts[0]() is None


@pytest.mark.interface
@pytest.mark.parametrize(
    "transform,error,match",
    [
        (lambda f, x: advect.jvp(f)(x, tangents=1.0), advect.NoJVPError, "no JVP rule"),
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
    np.testing.assert_allclose(
        gradient, context.pullback(weight)[3].real[0], rtol=1e-13
    )

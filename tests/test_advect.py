"""Native black-box composition through a real external autodiff framework."""

from concurrent.futures import ThreadPoolExecutor

import advect as ag
import advect.numpy as anp
import numpy as np
import pytest
from advect import grad, vjp
from hypothesis import given
from hypothesis import strategies as st

from treams_rs import advect as ad


def objective(radius, epsilon):
    value = ad.cluster(1, 1.2, radius, epsilon, anp.array([[0, 0, 0], [0, 0, 1.5]]))
    return anp.sum(anp.real(value * anp.conj(value))) + 0.03 * anp.sum(anp.real(value))


@pytest.mark.ad_contract
@given(radius=st.floats(0.1, 0.35), epsilon=st.floats(1.1, 6.0))
def test_advect_cluster_objective(radius, epsilon):
    radii = anp.array([radius, 0.25])
    eps = anp.array([epsilon + 0.1j, 3 + 0.2j])
    dr, de = grad(objective, argnums=(0, 1))(radii, eps)
    h = 1e-6
    for parameter, gradient, direction in [
        (0, dr, np.array([0.2, -0.3])),
        (1, de, np.array([0.2 + 0.3j, -0.1 + 0.2j])),
    ]:
        plus, minus = [radii, eps], [radii, eps]
        plus[parameter] = plus[parameter] + h * direction
        minus[parameter] = minus[parameter] - h * direction
        numerical = (objective(*plus) - objective(*minus)) / (2 * h)
        np.testing.assert_allclose(
            np.real(np.vdot(gradient, direction)), numerical, rtol=2e-6, atol=1e-9
        )


@pytest.mark.ad_contract
@pytest.mark.parametrize("kind", ["sphere", "mie", "cylinder", "interaction"])
def test_advect_other_native_boundaries(kind):
    def loss(x):
        if kind == "sphere":
            matrix = ad.sphere(2, 1.2, anp.array([x]), anp.array([3.0, 1.0]))
        elif kind == "mie":
            matrix = ad.mie(
                2, anp.array([x]), anp.array([3, 1]), anp.ones(2), anp.zeros(2)
            )
        elif kind == "cylinder":
            matrix = ad.mie_cyl(
                0.2,
                -1,
                1.1,
                anp.array([x]),
                anp.array([3, 1]),
                anp.ones(2),
                anp.zeros(2),
            )
        else:
            matrix = ad.interaction(anp.eye(3) * x * (1 + 0.1j), anp.ones((3, 3)) * 0.1)
        return anp.sum(anp.sin(anp.real(matrix)) + 0.2 * anp.imag(matrix) ** 2)

    x, h = 0.3, 1e-6
    np.testing.assert_allclose(
        grad(loss)(x), (loss(x + h) - loss(x - h)) / (2 * h), rtol=1e-6, atol=1e-9
    )


@pytest.mark.ad_contract
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


@pytest.mark.ad_contract
def test_explicit_first_order_single_use_contract():
    def function(r):
        return anp.real(ad.sphere(1, 1.2, anp.array([r]), anp.array([3.0, 1.0]))).sum()

    with pytest.raises(ag.TracingError, match="first-order"):
        grad(grad(function))(0.2)
    _, pullback = vjp(function)(0.2)
    pullback(1.0)
    with pytest.raises(RuntimeError, match="consumed"):
        pullback(1.0)


@pytest.mark.ad_contract
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


@pytest.mark.ad_contract
@pytest.mark.parametrize(
    "path", ["direct", "grad", "consume", "close", "drop", "exception"]
)
def test_native_residual_released_on_every_exit(monkeypatch, path):
    import gc
    import weakref

    from treams_rs import diff

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


@pytest.mark.ad_contract
@pytest.mark.parametrize(
    "transform,error,match",
    [
        (lambda f, x: ag.jvp(f)(x, tangents=1.0), ag.NoJVPError, "no JVP rule"),
        (lambda f, x: ag.grad(ag.grad(f))(x), ag.TracingError, "first-order"),
        (lambda f, x: ag.stage(f, x), ag.TracingError, "abstract"),
        (
            lambda f, x: ag.grad(ag.checkpoint(f))(x),
            ag.TracingError,
            "residual primitive",
        ),
    ],
)
def test_unsupported_transforms_fail_before_native_execution(
    monkeypatch, transform, error, match
):
    from treams_rs import diff

    def unexpected_forward(*args, **kwargs):
        pytest.fail("unsupported transform invoked the native solver")

    monkeypatch.setattr(diff, "sphere", unexpected_forward)

    def loss(radius):
        return np.real(
            ad.sphere(1, 1.2, anp.array([radius]), np.array([3.0, 1.0]))
        ).sum()

    with pytest.raises(error, match=match):
        transform(loss, 0.2)


@pytest.mark.ad_contract
@given(z=st.floats(0.8, 2.0), imaginary=st.floats(0.01, 0.2))
def test_expansion_joint_position_and_complex_wavenumber_gradient(z, imaginary):
    from treams_rs import SphericalBasis as SphericalWaveBasis

    basis = SphericalWaveBasis.default(2)
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

    gp, gk = grad(loss, argnums=(0, 1))(positions, ks)
    dp = np.array([[0.1, -0.2, 0.3]])
    dk = np.array([0.2 + 0.1j, -0.1 + 0.3j])
    h = 1e-6
    np.testing.assert_allclose(
        np.real(np.vdot(gp, dp) + np.vdot(gk, dk)),
        (loss(positions + h * dp, ks + h * dk) - loss(positions - h * dp, ks - h * dk))
        / (2 * h),
        rtol=2e-6,
        atol=1e-5,
    )

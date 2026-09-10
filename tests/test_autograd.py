"""Native black-box composition through a real external autodiff framework."""

from concurrent.futures import ThreadPoolExecutor

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import grad, make_vjp
from hypothesis import given
from hypothesis import strategies as st

from treams_rs import autograd as ad


def objective(radius, epsilon):
    value = ad.cluster(1, 1.2, radius, epsilon, anp.array([[0, 0, 0], [0, 0, 1.5]]))
    return anp.sum(anp.real(value * anp.conj(value))) + 0.03 * anp.sum(anp.real(value))


@pytest.mark.ad_contract
@given(radius=st.floats(0.1, 0.35), epsilon=st.floats(1.1, 6.0))
def test_autograd_cluster_objective(radius, epsilon):
    radii = anp.array([radius, 0.25])
    eps = anp.array([epsilon + 0.1j, 3 + 0.2j])
    dr, de = grad(objective, (0, 1))(radii, eps)
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
            anp.real(anp.sum(gradient * direction)), numerical, rtol=2e-6, atol=1e-9
        )


@pytest.mark.ad_contract
@pytest.mark.parametrize("kind", ["sphere", "mie", "cylinder", "interaction"])
def test_autograd_other_native_boundaries(kind):
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

    with pytest.raises(NotImplementedError, match="first-order"):
        grad(grad(function))(0.2)
    pullback, _ = make_vjp(function)(0.2)
    pullback(1.0)
    with pytest.raises(ValueError, match="consumed"):
        pullback(1.0)

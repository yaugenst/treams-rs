"""Native callbacks under JAX transformations, with analytic and physical checks."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from treams_rs import diff  # noqa: E402
from treams_rs import jax as tj  # noqa: E402


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64(True):
        yield


def test_complex_solve_jit_vjp_matches_framework_and_native():
    operator = np.array([[2 + 0.2j, 0.1j], [0.2, 3 - 0.1j]])
    rhs = np.array([[0.2 + 0.3j, 0.5], [1j, -0.2j]])
    weight = jnp.array([[0.4 + 0.3j, 0.2j], [0.7 - 0.2j, -0.1j]])

    def objective(solve, a, b):
        value = solve(a, b)
        return jnp.real(jnp.vdot(weight, value)) + jnp.sum(jnp.abs(value) ** 2)

    actual = jax.jit(
        jax.value_and_grad(lambda a, b: objective(tj.solve, a, b), argnums=(0, 1))
    )(operator, rhs)
    expected = jax.value_and_grad(
        lambda a, b: objective(jnp.linalg.solve, a, b), argnums=(0, 1)
    )(operator, rhs)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_allclose(a, b, rtol=2e-13, atol=2e-13)


def test_jit_vmap_and_repeated_pullbacks():
    z = jnp.array([0.4 + 0.1j, 0.7 - 0.2j, 1.1 + 0.3j])
    function = jax.jit(
        jax.vmap(jax.checkpoint(lambda value: tj.bessel(value, order=2)))
    )
    value, pullback = jax.vjp(function, z)
    expected, context = diff.bessel(2, np.asarray(z))
    weight = np.array([0.1 + 0.2j, 0.3 - 0.1j, 0.2j])
    native = context.pullback(np.conj(weight)).conj()
    assert_allclose(value, expected, atol=1e-14)
    for _ in range(2):
        assert_allclose(pullback(weight)[0], native, atol=1e-14)


@settings(max_examples=12, deadline=None)
@given(radius=st.floats(0.1, 0.4), epsilon=st.floats(1.2, 5.0))
def test_sphere_physical_scale_invariant_and_radius_gradient(radius, epsilon):
    def loss(k, r):
        matrix = tj.sphere(1, k, jnp.reshape(r, (1,)), jnp.array([epsilon + 0.1j, 1.0]))
        return jnp.sum(jnp.abs(matrix) ** 2)

    k = 1.2
    grad_k, grad_r = jax.grad(loss, argnums=(0, 1))(k, radius)
    # Maxwell scaling: k -> k/s and r -> r*s leaves the T-matrix unchanged.
    assert_allclose(-k * grad_k + radius * grad_r, 0, atol=1e-11)
    h = 1e-6
    numeric = (loss(k, radius + h) - loss(k, radius - h)) / (2 * h)
    assert_allclose(grad_r, numeric, rtol=2e-7, atol=1e-10)


def test_wrap_cluster_all_parameter_directions_and_static_arguments():
    # The native cluster pullback is ordered radii, positions, epsilon, k0.
    def record(r, p, e, k):
        return diff.cluster(1, float(k), r, e, p)

    values = (
        np.array([0.2, 0.25]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]),
        np.array([3.0 + 0.1j, 2.5 + 0.2j]),
        np.array(1.2),
    )
    operation = tj.wrap(record, *values)

    def loss(*args):
        matrix = operation(*args)
        return jnp.sum(jnp.abs(matrix) ** 2) + 0.03 * jnp.real(matrix).sum()

    gradient = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(*values)
    for i, g in enumerate(gradient):
        direction = np.full(values[i].shape, 0.1)
        if np.iscomplexobj(values[i]):
            direction = direction + 0.07j
        plus, minus = list(values), list(values)
        plus[i] = values[i] + 1e-6 * direction
        minus[i] = values[i] - 1e-6 * direction
        numeric = (loss(*plus) - loss(*minus)) / 2e-6
        assert_allclose(np.sum(g * direction).real, numeric, rtol=3e-6, atol=1e-9)


@pytest.mark.parametrize("vectors", [False, True])
def test_multiple_outputs_and_unused_cotangents(vectors):
    matrix = np.array([[1.0 + 0.2j, 0.3j], [0.1, 2.0 - 0.1j]])
    operation = tj.wrap(diff.eig, matrix)

    def loss(value):
        eigenvalues, eigenvectors = operation(value)
        return (
            jnp.real(eigenvectors).sum()
            if vectors
            else jnp.sum(jnp.abs(eigenvalues) ** 2)
        )

    gradient = jax.jit(jax.grad(loss))(matrix)
    direction = np.array([[0.1 + 0.2j, 0.3j], [-0.1j, 0.2]])
    h = 1e-6
    assert_allclose(
        np.sum(gradient * direction).real,
        (loss(matrix + h * direction) - loss(matrix - h * direction)) / (2 * h),
        rtol=3e-7,
        atol=1e-9,
    )


def test_lattice_broadcast_native_pullbacks():
    def record(k, q, a, r, eta):
        return diff.lattice_sum(2, [2, 3], -1, k, q, a, r, eta)

    values = (
        np.array([[2.1 + 0.2j], [2.3 + 0.1j]]),
        np.array([0.1, 0.2]),
        np.eye(2) * 1.5,
        np.array([[0.19, 0.11, 0.07], [0.21, -0.09, 0.05]]),
        np.array(0.9 + 0.02j),
    )
    function = tj.wrap(record, *values)
    expected, context = record(*values)
    weight = np.array([[0.1 + 0.2j, -0.3j], [0.4, 0.2j]])
    native = context.pullback(weight.conj())
    result, pullback = jax.vjp(jax.jit(function), *values)
    assert_allclose(result, expected, atol=1e-13)
    for actual, oracle, primal in zip(pullback(weight), native, values, strict=True):
        assert actual.shape == primal.shape
        assert_allclose(
            actual,
            oracle.conj() if np.iscomplexobj(primal) else oracle.real,
            atol=1e-11,
        )


def test_requested_illumination_jit_gradient_matches_full_response():
    local = np.array([[0.2 + 0.1j, 0.03j], [0.05, 0.3 - 0.1j]])
    coupling = np.array([[0.0, 0.1 + 0.2j], [-0.03j, 0.0]])
    incident = np.array([[0.4 + 0.1j], [-0.2j]])

    def loss(function, t, c, a):
        result = function(t, c, a)
        return jnp.sum(jnp.abs(result) ** 2) + 0.1 * jnp.real(result).sum()

    def full_response(t, c, a):
        return jnp.linalg.solve(jnp.eye(2) - t @ c, t) @ a

    actual = jax.jit(
        jax.value_and_grad(
            lambda t, c, a: loss(tj.illuminate, t, c, a), argnums=(0, 1, 2)
        )
    )(local, coupling, incident)
    expected = jax.value_and_grad(
        lambda t, c, a: loss(full_response, t, c, a), argnums=(0, 1, 2)
    )(local, coupling, incident)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_allclose(a, b, atol=2e-13, rtol=2e-13)


def test_adapter_contract_precision_shape_higher_derivatives():
    with jax.enable_x64(False), pytest.raises(ValueError, match="jax_enable_x64"):
        tj.bessel(0.3, order=1)
    with pytest.raises(TypeError, match="float64"):
        tj.bessel(jnp.array(0.3, dtype=jnp.float32), order=1)
    operation = tj.wrap(lambda z: diff.bessel(1, z), np.array([0.3]))
    with pytest.raises(ValueError, match="shapes and dtypes"):
        operation(jnp.array([0.2, 0.3]))
    with pytest.raises(ValueError, match="shapes and dtypes"):
        operation(jnp.array([0.3]), jnp.array([0.4]))
    with pytest.raises((ValueError, TypeError), match=r"JVP|jvp|differentiat"):
        jax.grad(jax.grad(lambda z: tj.bessel(z, order=1).real))(0.3)


def test_pullback_arity_and_shape_fail_explicitly():
    from treams_rs._adapters import gradients

    primal = (np.array([0.3]),)
    with pytest.raises(ValueError, match="one gradient"):
        gradients(lambda g: (g, g), primal, primal, conjugate=True)
    with pytest.raises(ValueError, match="gradient shape"):
        gradients(lambda g: np.array(1.0), primal, primal, conjugate=True)

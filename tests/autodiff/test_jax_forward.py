"""JAX native JVPs compose with transformations and preserve the native VJP."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.gradients

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import treams_rs as tr  # noqa: E402
from treams_rs import diff, special  # noqa: E402
from treams_rs import jax as tj  # noqa: E402
from treams_rs._records import DerivativeContext  # noqa: E402
from treams_rs.testing import check_gradient  # noqa: E402

from _support import jax_x64  # noqa: E402


@pytest.fixture(autouse=True)
def double_precision():
    with jax_x64():
        yield


@pytest.mark.parametrize("mode", ["primal", "jvp", "grad"])
@pytest.mark.parametrize("saved", [False, True])
def test_eager_wrap_reuses_compiled_callbacks(caplog, mode, saved):
    matrix = jnp.array([[2.0, 0.2], [0.1, 3.0]])
    rhs = jnp.array([[1.0], [2.0]])
    record = diff.solve if saved else lambda a, b: diff.solve(a, b)
    wrapped = tj.wrap(record, matrix, rhs)

    def function(rhs):
        return wrapped(matrix, rhs).real.sum()

    direction = jnp.ones_like(rhs)
    run = {
        "primal": function,
        "jvp": lambda rhs: jax.jvp(function, (rhs,), (direction,)),
        "grad": jax.grad(function),
    }[mode]
    values = (rhs, rhs * 2, rhs * 3)
    with jax.log_compiles(True):
        jax.block_until_ready(run(rhs))
        caplog.clear()
        for value in values:
            jax.block_until_ready(run(value))
    assert not [
        record.message
        for record in caplog.records
        if record.message.startswith("Compiling ")
    ]


@pytest.mark.parametrize("compiled", [False, True])
def test_opaque_custom_records_replay_for_each_derivative(compiled):
    calls = []

    def record(x):
        calls.append("record")
        value, context = diff.bessel(2, x, spherical=True)

        def pushforward(dx):
            calls.append("pushforward")
            return context.pushforward(dx)

        def pullback(g):
            calls.append("pullback")
            return context.pullback(g)

        return value, DerivativeContext(pullback, pushforward)

    x = jnp.array([0.4 + 0.1j, 0.7 - 0.2j])
    function = tj.wrap(record, x)

    def run(x):
        return jax.jvp(function, (x,), (jnp.ones_like(x),))

    if compiled:
        run = jax.jit(run)
    calls.clear()
    value, tangent = run(x)
    tangent.block_until_ready()
    assert calls == ["record", "record", "pushforward"]
    assert_allclose(value, special.spherical_jn(2, np.asarray(x)), atol=1e-14)
    assert_allclose(
        tangent, special.spherical_jn(2, np.asarray(x), derivative=True), atol=1e-14
    )

    calls.clear()
    _, pullback = jax.vjp(function, x)
    pullback(jnp.ones_like(value))[0].block_until_ready()
    assert calls == ["record", "record", "pullback"]


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("complex_input", [False, True])
def test_public_numerical_forward_precision_and_sequential_vmap(x64, complex_input):
    with jax_x64(x64):
        dtype = jnp.complex64 if complex_input else jnp.float32
        x = jnp.array([0.4, 0.7, 1.1], dtype=dtype)
        dx = jnp.array([0.1, -0.2, 0.3], dtype=dtype)
        if complex_input:
            x = x + jnp.array([0.1j, -0.2j, 0.3j], dtype=dtype)
            dx = dx + jnp.array([0.03j, 0.1j, -0.2j], dtype=dtype)
        function = jax.jit(
            jax.vmap(jax.checkpoint(lambda z: special.spherical_jn(2, z)))
        )
        value, tangent = jax.jvp(function, (x,), (dx,))
        expected = special.spherical_jn(2, np.asarray(x))
        expected_tangent = special.spherical_jn(
            2, np.asarray(x), derivative=True
        ) * np.asarray(dx)
        assert (
            tangent.dtype == value.dtype == (jnp.complex128 if x64 else jnp.complex64)
        )
        # Input and output rounding follows the caller's precision.
        assert_allclose(value, expected, rtol=2e-7, atol=1e-9)
        assert_allclose(tangent, expected_tangent, rtol=2e-7, atol=1e-9)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_jacfwd_jacrev_and_repeated_linearization(checkpoint):
    def function(x):
        return tj.bessel(x, order=2, spherical=True)

    if checkpoint:
        function = jax.checkpoint(function)
    x = jnp.array([0.4 + 0.1j, 0.7 - 0.2j])
    derivative = special.spherical_jn(2, np.asarray(x), derivative=True)
    expected = np.diag(derivative)
    for transform in (jax.jacfwd, jax.jacrev):
        for transformed in (
            jax.jit(transform(function, holomorphic=True)),
            transform(jax.jit(function), holomorphic=True),
        ):
            assert_allclose(transformed(x), expected, atol=1e-14)
    value, pushforward = jax.linearize(function, x)
    assert_allclose(value, special.spherical_jn(2, np.asarray(x)), atol=1e-14)
    for direction in (jnp.ones_like(x), jnp.array([0.2j, -0.3 + 0.1j])):
        assert_allclose(pushforward(direction), derivative * direction, atol=1e-14)


def test_solve_jvp_uses_complex_directions_for_both_inputs():
    matrix = jnp.array([[2.0 + 0.2j, 0.1 - 0.3j], [0.2j, 1.5 - 0.1j]])
    rhs = jnp.array([[0.4 + 0.1j, 0.2j], [0.7 - 0.2j, 0.3]])
    dmatrix = jnp.array([[0.1j, -0.2], [0.3, 0.2 - 0.1j]])
    drhs = jnp.array([[0.2, 0.1j], [-0.3j, 0.1]])
    value, tangent = jax.jit(lambda a, b, da, db: jax.jvp(tj.solve, (a, b), (da, db)))(
        matrix, rhs, dmatrix, drhs
    )
    expected = np.linalg.solve(matrix, rhs)
    expected_tangent = np.linalg.solve(matrix, drhs - dmatrix @ expected)
    assert_allclose(value, expected, atol=1e-14)
    assert_allclose(tangent, expected_tangent, atol=1e-14)


def test_multi_output_jvp_and_jax_complex_adjoint_convention():
    matrix = jnp.array([[2.0 + 0.2j, 0.1 - 0.3j], [0.2j, 1.5 - 0.1j]])
    direction = jnp.array([[0.1j, -0.2], [0.3, 0.2 - 0.1j]])
    function = tj.wrap(diff.eig, matrix)
    values, tangents = jax.jvp(jax.jit(function), (matrix,), (direction,))
    weights = tuple(jnp.full_like(value, 0.2 - 0.3j) for value in values)
    _, pullback = jax.vjp(jax.jit(function), matrix)
    (gradient,) = pullback(weights)
    forward_pairing = sum(
        jnp.real(jnp.sum(g * dy)) for g, dy in zip(weights, tangents, strict=True)
    )
    reverse_pairing = jnp.real(jnp.sum(gradient * direction))
    assert_allclose(forward_pairing, reverse_pairing, atol=1e-13)


@pytest.mark.workflows
def test_cluster_radius_position_and_material_forward_gradient():
    def objective(parameters):
        radius, distance, epsilon = parameters
        first = tr.sphere_tmatrix(
            k0=1.2, lmax=1, radius=radius, material=epsilon + 0.1j
        )
        second = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.15, material=2.0)
        zero = 0 * distance
        cluster = tr.Cluster(
            [first, second],
            positions=[[zero, zero, zero], [0.3 + zero, zero, distance]],
        )
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        field = cluster.solve().scatter(wave).efield([[0.2, 0.4, 2.5]])
        return (abs(field) ** 2).sum()

    parameters = np.array([0.2, 1.0, 3.0])
    forward = jax.jit(jax.jacfwd(objective))
    reverse = jax.jit(jax.grad(objective))
    assert_allclose(forward(parameters), reverse(parameters), rtol=1e-11, atol=1e-13)
    check_gradient(objective, forward, parameters)


@pytest.mark.workflows
def test_radius_jvp_returns_the_whole_scattered_field_map():
    def field(radius):
        sphere = tr.sphere_tmatrix(k0=1.2, lmax=2, radius=radius, material=3 + 0.1j)
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        return sphere.scatter(wave).efield([[0.2, 0.4, 1.5], [0.7, 0.3, 2.0]])

    radius = jnp.array(0.2)
    value, tangent = jax.jvp(jax.jit(field), (radius,), (jnp.ones_like(radius),))
    assert value.shape == tangent.shape == (2, 3)
    assert_allclose(value, field(0.2), rtol=1e-13, atol=1e-14)
    weights = np.array([[0.2j, 0.3, 0.1 - 0.4j], [0.3 - 0.1j, 0.5, 0.2j]])

    def objective(r):
        return np.vdot(weights, field(r)).real

    check_gradient(objective, lambda _: np.vdot(weights, tangent).real, 0.2)


@pytest.mark.interface
def test_nested_forward_derivatives_remain_unsupported():
    with pytest.raises(ValueError, match="first-order differentiation"):
        jax.jacfwd(jax.jacfwd(lambda x: tj.bessel(x, order=1).real))(0.3)


@pytest.mark.parametrize("compiled", [False, True])
def test_transpose_of_linearized_map_applies_gauss_newton(compiled):
    def function(x):
        return tj.bessel(x, order=2, spherical=True).real

    x = jnp.array([0.4, 0.7])
    direction = jnp.array([0.2, -0.3])
    _, linear = jax.linearize(function, x)

    def gauss_newton(direction):
        tangent, transpose = jax.vjp(linear, direction)
        return transpose(tangent)[0]

    actual = (jax.jit(gauss_newton) if compiled else gauss_newton)(direction)
    derivative = special.spherical_jn(2, np.asarray(x), derivative=True).real
    assert_allclose(actual, derivative**2 * direction, atol=1e-14)


@pytest.mark.interface
def test_jacfwd_preserves_empty_array_shape():
    result = jax.jit(jax.jacfwd(lambda x: tj.bessel(x, order=1)))(jnp.empty(0))
    assert result.shape == (0, 0)

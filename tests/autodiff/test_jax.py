"""Native callbacks under JAX transformations, with analytic and physical checks."""

import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytestmark = pytest.mark.gradients

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import treams_rs as tr  # noqa: E402
from treams_rs import diff, special  # noqa: E402

# `tj` is the treams_rs JAX adapter; `jax` is the JAX library.
from treams_rs import jax as tj  # noqa: E402

from _support import assert_unitary_ports  # noqa: E402


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64(True):
        yield


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


@pytest.mark.interface
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


# XLA flushes subnormals to zero (FTZ/DAZ) on the thread that runs a computation,
# also while a pure_callback runs, even on the Python main thread. The complex
# reciprocal of faer, which the native LU applies to each pivot z, scales z by
# sqrt(1 / f64::MIN_POSITIVE) when max(|Re z|, |Im z|) <= 1; for |z| > 1 the
# reciprocal of the scaled |z|**2 is subnormal. Native entry points keep
# subnormals for their own work and restore XLA's mode on return; flushed, a
# scalar solve would return zero and larger solves would be wrong.


@pytest.mark.interface
def test_callbacks_keep_subnormals_and_restore_the_xla_mode():
    tiny, two = sys.float_info.min, 2.0
    seen = []

    def callback(x):
        before = tiny / two
        value = special.spherical_jn(10, np.asarray(x))
        seen.append((before, tiny / two))
        return value

    shape = jax.ShapeDtypeStruct((2,), jnp.complex128)
    x = np.array([1e-31, 3e-31])
    actual = jax.jit(lambda x: jax.pure_callback(callback, shape, x))(x)
    # XLA flushes in the callback, before and after the native call.
    assert seen == [(0.0, 0.0)]
    expected = special.spherical_jn(10, x)
    assert np.all((abs(expected) > 0) & (abs(expected) < tiny))
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "pivot",
    [
        0.6 + 0.6j,
        1.01 + 0.5j,
        1e-3 + 0j,
        # Inside the unit square, outside the unit disc.
        0.9 + 0.6j,
        1 + 0.1j,
        -0.3 - 0.99j,
    ],
)
def test_scalar_solve_is_the_reciprocal(pivot):
    solution = tj.solve(jnp.array([[pivot]]), jnp.array([[1.0 + 0j]]))
    assert_allclose(solution, [[1 / pivot]], rtol=1e-15)


@pytest.mark.parametrize(
    "thickness,k0,epsilon,kappa",
    [(0.234375, 1.4375, 5.0, 0.0), (0.5, 0.8125, 3.75, 0.0), (0.3, 1.5, 3.0, 0.125)],
)
def test_slab_cascade_with_unit_scale_pivot(thickness, k0, epsilon, kappa):
    # The cascade pivot 1 - r**2 exp(2i kz thickness) lies in the band that
    # XLA's mode flushes.
    ports = tr.PlaneWavePorts.default([[0.1, 0.05]])
    options = dict(k0=k0, basis=ports, thickness=thickness)
    actual = tj.slab(**options, material=tj.Material(epsilon, 1.0, kappa))
    expected = tr.slab(**options, material=tr.Material(epsilon, 1.0, kappa))
    assert_allclose(actual.array, expected.array, atol=1e-13)


# The first native call that uses Rayon, such as a solve, creates the global pool.
# Made through treams_rs.jax, it starts every worker inside a callback, where the
# workers inherit XLA's mode; they must keep subnormals for every later parallel
# native computation, in NumPy, Advect and PyTorch too. This needs a fresh process.
JAX_FIRST_SLAB = """
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import treams_rs as tr
from treams_rs import jax as tj

identity = jnp.eye(2, dtype=complex)
tj.solve(identity, identity).block_until_ready()
ports = tr.PlaneWavePorts.default([[0.1 + 0.01 * i, 0.05] for i in range(16)])
material = tr.Material(5.0, 1.0, 0.0)
slab = tr.slab(k0=1.4375, basis=ports, thickness=0.234375, material=material)
np.save(sys.argv[1], slab.array)
"""


@pytest.mark.physics
def test_numpy_slab_after_a_first_native_call_through_jax_is_unitary(tmp_path):
    path = tmp_path / "slab.npy"
    subprocess.run(
        [sys.executable, "-c", JAX_FIRST_SLAB, str(path)], check=True, timeout=600
    )
    # Lossless, in vacuum, every order propagating: the S-matrix is unitary. The 16
    # channels run on the pool's workers, and all of them meet a pivot in the
    # band XLA's mode flushes; the first is the first case of the single-channel
    # test above.
    assert_unitary_ports(np.load(path), atol=1e-13)

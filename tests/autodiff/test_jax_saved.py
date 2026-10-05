"""Array-owned residuals avoid recording again across JAX derivative calls."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytestmark = pytest.mark.gradients

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from treams_rs import jax as tj  # noqa: E402
from treams_rs._saved import (  # noqa: E402
    ArraySpec,
    SavedRecord,
    pack_state,
    unpack_state,
)

from _support import jax_x64  # noqa: E402


@dataclass(frozen=True)
class SinContext:
    derivative: np.ndarray
    calls: Counter

    def pullback(self, cotangent):
        self.calls["pullback"] += 1
        return self.derivative.conj() * cotangent

    def pushforward(self, tangent):
        self.calls["pushforward"] += 1
        return self.derivative * tangent


def saved_sine(calls):
    # Integer and boolean state exercise exact transport as well as derivatives.
    integer = np.asarray(2**63 + 7, dtype=np.uint64)
    boolean = np.asarray(True)

    def record(x):
        calls["record"] += 1
        return np.sin(x), SinContext(np.cos(x), calls)

    def schema(inputs):
        return (
            inputs[0],
            ArraySpec((), integer.dtype),
            ArraySpec((), boolean.dtype),
        )

    def save(context):
        return context.derivative, integer, boolean

    def restore(state, x):
        derivative, index, active = state
        # The saved native derivative stays double precision even with x64 off.
        assert derivative.dtype == np.cos(x).dtype
        assert_array_equal(derivative, np.cos(x))
        assert_array_equal(index, integer)
        assert_array_equal(active, boolean)
        return SinContext(derivative, calls)

    return SavedRecord(record, schema, save, restore)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("complex_input", [False, True])
def test_saved_vjp_linearize_and_jacfwd_record_once(x64, complex_input):
    with jax_x64(x64):
        calls = Counter()
        x = jnp.array([0.23, 0.47], dtype=jnp.float32)
        if complex_input:
            x = x + jnp.array([0.1j, -0.2j], dtype=jnp.complex64)
        function = tj.wrap(saved_sine(calls), x)
        native_x = np.asarray(x, dtype=np.complex128 if complex_input else np.float64)
        expected = np.cos(native_x)
        # The inputs are single precision under both output configurations.
        tolerance = 3e-7

        for wrapped in (function, jax.jit(function)):
            calls.clear()
            _, tangent = jax.jvp(wrapped, (x,), (jnp.ones_like(x),))
            assert_allclose(tangent, expected, rtol=tolerance)
            assert calls == {"record": 1, "pushforward": 1}

            calls.clear()
            value, pullback = jax.vjp(wrapped, x)
            assert_allclose(value, np.sin(native_x), rtol=tolerance)
            for weight in (1, 2):
                (gradient,) = pullback(jnp.ones_like(value) * weight)
                assert gradient.dtype == x.dtype
                assert_allclose(gradient, expected * weight, rtol=tolerance)
            assert calls == {"record": 1, "pullback": 2}

            calls.clear()
            _, pushforward = jax.linearize(wrapped, x)
            for direction in (jnp.ones_like(x), x):
                assert_allclose(
                    pushforward(direction), expected * direction, rtol=tolerance
                )
            assert calls == {"record": 1, "pushforward": 2}

            calls.clear()
            actual = jax.jacfwd(wrapped, holomorphic=complex_input)(x)
            assert_allclose(actual, np.diag(expected), rtol=tolerance)
            assert calls == {"record": 1, "pushforward": 2}


@pytest.mark.parametrize("x64", [False, True])
def test_checkpoint_recomputes_saved_state_when_requested(x64):
    with jax_x64(x64):
        calls = Counter()
        x = jnp.array([0.2, 0.4])
        function = tj.wrap(saved_sine(calls), x)
        wrapped = jax.jit(jax.checkpoint(function))
        calls.clear()
        value, pullback = jax.vjp(wrapped, x)
        for _ in range(2):
            assert_allclose(pullback(jnp.ones_like(value))[0], np.cos(x), rtol=3e-7)
        assert calls == {"record": 3, "pullback": 2}


@pytest.mark.parametrize("x64", [False, True])
def test_saved_state_is_independent_across_concurrent_jit_calls(x64):
    with jax_x64(x64):
        calls = Counter()
        x = jnp.array([0.2, 0.4])
        function = tj.wrap(saved_sine(calls), x)
        objective = jax.jit(jax.value_and_grad(lambda x: jnp.sum(function(x))))
        jax.block_until_ready(objective(x))
        calls.clear()
        values = [x * factor for factor in (1, 2, 3, 4)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(objective, values))
        for input_value, (value, gradient) in zip(values, results, strict=True):
            assert_allclose(value, np.sin(input_value).sum(), rtol=3e-7)
            assert_allclose(gradient, np.cos(input_value), rtol=3e-7)
        assert calls == {"record": 4, "pullback": 4}


@pytest.mark.parametrize("x64", [False, True])
def test_saved_multi_output_complex_adjoint_and_sequential_batching(x64):
    with jax_x64(x64):
        calls = Counter()

        class Context:
            def __init__(self, derivative, x):
                self.derivative, self.x = derivative, x

            def pullback(self, wave, power):
                return self.derivative.conj() * wave + 2 * self.x * power

            def pushforward(self, tangent):
                return (
                    self.derivative * tangent,
                    2 * np.real(self.x.conj() * tangent),
                )

        def record(x):
            calls["record"] += 1
            return (np.sin(x), np.abs(x) ** 2), Context(np.cos(x), x)

        saved = SavedRecord(
            record,
            lambda inputs: inputs,
            lambda context: (context.derivative,),
            lambda state, x: Context(state[0], x),
        )
        x = jnp.array([0.2 + 0.1j, 0.4 - 0.2j], dtype=jnp.complex64)
        direction = jnp.array([0.1 - 0.2j, -0.3 + 0.1j], dtype=x.dtype)
        function = tj.wrap(saved, x[0])
        batched = jax.jit(jax.vmap(function))
        values, tangents = jax.jvp(batched, (x,), (direction,))
        weights = (jnp.full_like(values[0], 0.3 + 0.2j), jnp.full_like(values[1], 0.7))
        _, pullback = jax.vjp(batched, x)
        (gradient,) = pullback(weights)
        actual = sum(
            np.real(np.sum(g * dy)) for g, dy in zip(weights, tangents, strict=True)
        )
        expected = np.real(np.sum(gradient * direction))
        assert_allclose(actual, expected, rtol=3e-7, atol=1e-8)


@pytest.mark.interface
def test_saved_higher_derivatives_remain_unsupported():
    with jax_x64():
        function = tj.wrap(saved_sine(Counter()), np.array(0.2))
        with pytest.raises(ValueError, match="first-order differentiation"):
            jax.jacfwd(jax.jacfwd(function))(0.2)


@pytest.mark.interface
def test_unused_saved_record_has_no_lifetime_side_effect():
    with jax_x64():
        calls = Counter()
        x = jnp.array([0.2, 0.4])
        function = tj.wrap(saved_sine(calls), x)
        calls.clear()
        assert_array_equal(jax.jit(lambda x: (function(x), x)[1])(x), x)
        assert not calls


@pytest.mark.interface
def test_saved_state_preserves_noncontiguous_arrays_and_rejects_wrong_schema():
    value = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
    specs = (ArraySpec(value.shape, value.dtype),)
    packed = pack_state((value,), specs)
    assert_array_equal(unpack_state(packed, specs)[0], value)
    with pytest.raises(ValueError, match="one array per state_spec"):
        pack_state((), specs)
    with pytest.raises(ValueError, match="must match state_spec"):
        pack_state((value.astype(np.float32),), specs)
    with pytest.raises(TypeError, match="numeric or boolean"):
        pack_state((np.array([object()]),), (ArraySpec((1,), np.dtype(object)),))

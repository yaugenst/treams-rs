"""Repeated slab derivatives reuse the same recorded layer and power solves."""

from collections import Counter

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.gradients

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import treams_rs as tr  # noqa: E402
from treams_rs import _native  # noqa: E402

from _support import jax_x64  # noqa: E402


def test_jit_stack_linearization_and_pullback_reuse_native_state(monkeypatch):
    ports = tr.PlaneWavePorts.default([[0.0, 0.0]])
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)

    def transmission(thickness):
        layers = [
            tr.slab(
                basis=ports,
                k0=1.2,
                thickness=thickness[index],
                material=epsilon,
            )
            for index, epsilon in enumerate((2.2 + 0.05j, 3.1 + 0.07j))
        ]
        return tr.stack(layers).power(incident).transmission

    initial = np.array([0.23, 0.31])
    expected_value = transmission(initial)
    calls = Counter()

    def count_native(name):
        original = getattr(_native, name)

        def counted(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(_native, name, counted)

    for name in ("layer_stack", "smatrix_add", "smatrix_tr"):
        count_native(name)

    with jax_x64():
        function = jax.jit(transmission)
        parameters = jnp.asarray(initial)
        value, pushforward = jax.linearize(function, parameters)
        assert_allclose(value, expected_value, rtol=1e-13)
        one_forward = Counter(layer_stack=2, smatrix_add=1, smatrix_tr=1)
        assert calls == one_forward

        directions = (jnp.array([1.0, 0.0]), jnp.array([0.3, -0.7]))
        projected = [np.asarray(pushforward(direction)) for direction in directions]
        assert calls == one_forward

        reverse_value, pullback = jax.vjp(function, parameters)
        assert_allclose(reverse_value, value, rtol=1e-13)
        assert calls == one_forward + one_forward

        for scale in (1.0, -0.4):
            gradient = np.asarray(pullback(jnp.asarray(scale))[0])
            for direction, derivative in zip(directions, projected, strict=True):
                assert_allclose(
                    np.dot(gradient, direction), scale * derivative, rtol=1e-12
                )
        assert calls == one_forward + one_forward


@pytest.mark.parametrize("operation", ["cascade", "power"])
def test_mapped_smatrix_contexts_keep_state_without_dense_primals(operation):
    from treams_rs import jax as tj

    ports = tr.PlaneWavePorts.default([[0.1, 0.05]])
    initial = tr.slab(basis=ports, k0=1.2, thickness=0.2, material=2.2 + 0.05j).array

    def function(*matrices):
        system = tj.smatrix(matrices[0], basis=ports, k0=1.2)
        if operation == "cascade":
            upper = tj.smatrix(matrices[1], basis=ports, k0=1.2)
            return system.cascade(upper).array
        return system.power([1.0, 0.3j], side="positive").transmission

    with jax_x64():
        inputs = (jnp.asarray(initial),) * (2 if operation == "cascade" else 1)
        directions = tuple(jnp.full_like(value, 0.03 + 0.02j) for value in inputs)
        value, linear = jax.linearize(function, *inputs)
        retained = jax.make_jaxpr(linear)(*directions).consts
        state = [array for array in retained if array.dtype == np.uint8]
        expected_bytes = (
            _native.SMatrixAddContext._state_spec(2)
            if operation == "cascade"
            else _native.SMatrixTrContext._state_spec(2, 1, 1)
        )
        assert len(state) == 1
        assert state[0].nbytes == expected_bytes
        # Fixed metadata needs zero tangents, but its values and the dense
        # input matrices must not survive alongside the recorded state.
        for array in retained:
            if array.dtype != np.uint8:
                assert_allclose(array, 0, rtol=0, atol=0)
        assert sum(array.nbytes for array in retained) < expected_bytes + sum(
            array.nbytes for array in inputs
        )

        first = linear(*directions)
        assert_allclose(linear(*(2 * d for d in directions)), 2 * first, rtol=1e-12)
        _, reverse = jax.vjp(function, *inputs)
        weights = jnp.ones_like(value)
        gradients = reverse(weights)
        assert_allclose(
            jnp.real(
                sum(jnp.sum(g * d) for g, d in zip(gradients, directions, strict=True))
            ),
            jnp.real(jnp.sum(first)),
            rtol=1e-12,
        )
        for actual, expected in zip(reverse(weights), gradients, strict=True):
            assert_allclose(actual, expected, rtol=0, atol=0)

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

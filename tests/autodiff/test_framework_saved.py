"""Ordinary physics workflows retain their original native state in JAX."""

from collections import Counter
from functools import wraps

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.gradients

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import treams_rs as tr  # noqa: E402
from treams_rs import _native, diff  # noqa: E402

from _support import jax_x64  # noqa: E402


def _field_energy(radius, workflow):
    sphere = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3 + 0.1j)
    target = sphere
    if workflow != "sphere":
        other = tr.sphere_tmatrix(
            k0=1.2, lmax=1, radius=0.8 * radius, material=2 + 0.05j
        )
        target = tr.Cluster([sphere, other], positions=[[0, 0, 0], [0, 0, 1]])
        if workflow == "cluster_solve":
            target = target.solve()
    wave = tr.plane_wave([0, 0, 1], 1, k0=1.2)
    field = target.scatter(wave).efield([[0.3, 0.4, 1.7]])
    return (abs(field) ** 2).sum()


@pytest.mark.parametrize("workflow", ["sphere", "cluster_solve", "cluster_scatter"])
@pytest.mark.parametrize("transform", ["linearize", "vjp", "value_and_grad"])
def test_jax_physics_derivatives_do_not_repeat_native_forwards(
    monkeypatch, workflow, transform
):
    radius = 0.2
    step = 1e-6
    reference = _field_energy(radius, workflow)
    derivative = (
        _field_energy(radius + step, workflow) - _field_energy(radius - step, workflow)
    ) / (2 * step)
    calls = Counter()

    def watch(module, name):
        original = getattr(module, name)

        @wraps(original)
        def counted(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    for name in ("sphere", "particle_cluster", "expansion", "plane_expansion", "field"):
        watch(_native, name)
    watch(diff, "factor_interaction_blocks")

    def objective(value):
        return _field_energy(value, workflow)

    with jax_x64():
        if transform == "value_and_grad":
            value, gradient = jax.jit(jax.value_and_grad(objective))(
                jnp.asarray(radius)
            )
            assert_allclose(gradient, derivative, rtol=2e-8, atol=1e-12)
        else:
            value, linear = (
                jax.linearize(objective, jnp.asarray(radius))
                if transform == "linearize"
                else jax.vjp(objective, jnp.asarray(radius))
            )
            value.block_until_ready()
            forward_calls = calls.copy()
            for direction in (1.0, -0.3, 1.0):
                actual = jax.jit(linear)(jnp.asarray(direction))
                actual = actual[0] if transform == "vjp" else actual
                assert_allclose(actual, derivative * direction, rtol=2e-8, atol=1e-12)
                assert calls == forward_calls
        assert_allclose(value, reference, rtol=1e-13)
    assert calls["sphere"] == (1 if workflow == "sphere" else 2)
    assert calls["particle_cluster"] == (workflow == "cluster_solve")
    assert calls["factor_interaction_blocks"] == (workflow == "cluster_scatter")
    assert calls["field"] == 1


def test_jax_guard_restores_identity_without_rechecking_inputs():
    from treams_rs import jax as tj

    checks = []

    def require_positive(value, floor):
        checks.append(1)
        if np.any(value <= floor):
            raise ValueError("value must exceed floor")

    def objective(value):
        guarded = tj._backend.guard(require_positive, value, jnp.asarray(0.0))
        return (guarded**2).sum()

    with jax_x64():
        x = jnp.array([1.0, 2.0])
        value, linear = jax.linearize(objective, x)
        assert_allclose(value, 5.0)
        for direction in (jnp.array([0.2, -0.3]), jnp.ones(2)):
            assert_allclose(jax.jit(linear)(direction), jnp.vdot(2 * x, direction))
        assert checks == [1]

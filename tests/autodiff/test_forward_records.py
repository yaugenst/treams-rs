"""The shared forward boundary preserves domains, shapes and native residuals."""

import numpy as np
import pytest

from treams_rs._records import (
    DerivativeContext,
    apply_pushforward,
    record_outputs,
    run_jvp,
)

pytestmark = [pytest.mark.gradients, pytest.mark.interface]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_forward_record_preserves_real_linear_complex_derivative(dtype):
    calls = []

    def record(x, z):
        calls.append("forward")

        def pushforward(dx, dz):
            calls.append("pushforward")
            assert dx.dtype == np.float64
            assert dz.dtype == np.complex128
            return (2 * x * dx, np.conj(dz) + 1j * dx)

        return (x * x, np.conj(z) + 1j * x), DerivativeContext(None, pushforward)

    x, dx = np.array([0.5, 1.5], dtype=dtype), np.array([2.0, -0.5], dtype=dtype)
    z, dz = np.array([1 + 2j, 3 - 4j]), np.array([2 - 3j, 4 + 5j])
    outputs, tangents, multiple = run_jvp(record, (x, z), (dx, dz))
    assert multiple
    assert calls == ["forward", "pushforward"]
    np.testing.assert_array_equal(outputs[0], x * x)
    np.testing.assert_array_equal(tangents[0], 2 * x * dx)
    np.testing.assert_array_equal(tangents[1], np.conj(dz) + 1j * dx)


def test_saved_context_pushforward_reuses_original_record():
    def record(x):
        return x * x, DerivativeContext(None, lambda dx: 2 * x * dx)

    primal = np.array([1.0, 2.0])
    outputs, context, multiple = record_outputs(record, (primal,))
    assert not multiple
    (tangent,) = apply_pushforward(context, (np.ones(2),), (primal,), outputs)
    np.testing.assert_array_equal(tangent, [2.0, 4.0])


def test_inactive_input_has_zero_tangent_in_its_own_shape_and_domain():
    primal = np.array([1 + 2j, 3 - 4j])
    context = DerivativeContext(None, lambda dz: dz)
    (tangent,) = apply_pushforward(context, (None,), (primal,), (primal,))
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))


def test_complex_tangent_to_real_input_is_rejected_before_calling_context():
    calls = []
    context = DerivativeContext(None, lambda tangent: calls.append(tangent))
    with pytest.raises(TypeError, match="real dynamic parameter"):
        apply_pushforward(
            context, (np.array(1 + 2j),), (np.array(1.0),), (np.array(1.0),)
        )
    assert not calls


def test_tangent_shape_mismatch_is_rejected_before_calling_context():
    calls = []
    context = DerivativeContext(None, lambda tangent: calls.append(tangent))
    with pytest.raises(ValueError, match=r"parameter\[0\].*\(2,\).+\(1,\)"):
        apply_pushforward(context, (np.ones(1),), (np.ones(2),), (np.ones(2),))
    assert not calls


def test_pullback_only_custom_record_has_explicit_forward_boundary():
    with pytest.raises(NotImplementedError, match="context with pushforward"):
        run_jvp(lambda x: (x, lambda g: g), (np.array(1.0),), (np.array(2.0),))

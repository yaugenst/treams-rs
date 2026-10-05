"""Native context reconstruction preserves input shapes and rejects invalid layouts."""

import numpy as np
import pytest

from treams_rs import _native, diff

pytestmark = [pytest.mark.interface, pytest.mark.gradients]


@pytest.mark.parametrize("shape", [[2], [2**63, 2]], ids=["mismatch", "overflow"])
@pytest.mark.parametrize(
    "construct",
    [
        lambda a, shape: _native.bessel_context(
            np.array([1.0]), a, "j", False, 0, shape, []
        ),
        lambda a, shape: _native.angular_context(
            np.array([1.0]), np.array([0.0]), a, "legendre", shape, []
        ),
        lambda a, shape: _native.wignerd_context(
            [[1, 0, 0]], a, a, a, shape, [[], [], []]
        ),
        lambda a, shape: _native.incgamma_context(np.array([0.5]), a, shape, []),
        lambda a, shape: _native.intkambe_context(
            np.array([-2], dtype=np.int32), a, a, shape, [[], []]
        ),
        lambda a, shape: _native.spherical_translation_context(
            [[(1, 0, 0), (1, 0, 0)]], [a, a, a], True, False, shape, [[], [], []]
        ),
        lambda a, shape: _native.cylindrical_translation_context(
            [0], [a, a, a, a], False, shape, [[], [], [], []]
        ),
        lambda a, shape: _native.vector_wave_context(
            "vsh_Z", [(1, 0, 0)], [a, a], shape, [[], []]
        ),
    ],
    ids=[
        "bessel",
        "angular",
        "wignerd",
        "incgamma",
        "intkambe",
        "spherical",
        "cylindrical",
        "vector-wave",
    ],
)
def test_input_context_rejects_wrong_or_overflowing_output_shape(construct, shape):
    with pytest.raises(ValueError, match=r"shape|state"):
        construct(np.array([0.7 + 0.1j]), shape)


@pytest.mark.parametrize("thickness", [0.3, [0.3]], ids=["scalar", "vector"])
def test_layer_stack_native_pullback_preserves_thickness_shape_after_restore(thickness):
    value, original = diff.layer_stack(
        [[1.2, 1.2], [1.8 + 0.08j, 1.92 + 0.08j], [1.2, 1.2]],
        [1.0, 0.65 + 0.02j, 1.0],
        [[0.1, 0.05]],
        thickness,
    )
    restored = type(original)._from_state(original._state())
    expected = original.pullback(np.ones_like(value))[-1]
    actual = restored.pullback(np.ones_like(value))[-1]
    assert expected.shape == actual.shape == np.shape(thickness)
    np.testing.assert_array_equal(actual, expected)

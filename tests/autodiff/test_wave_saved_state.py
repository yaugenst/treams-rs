"""Wave coefficient wrappers restore both derivatives without another forward."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from treams_rs import _wave_functions as wave
from treams_rs._framework_backend import Backend
from treams_rs._records import apply_pullback, apply_pushforward
from treams_rs._saved import ArraySpec, saved_record

pytestmark = pytest.mark.gradients


def _capture(function):
    calls = []

    def operation(record, *values, shape, real=False):
        calls.append((saved_record(record), values, shape))
        return np.zeros(shape, dtype=np.float64 if real else np.complex128)

    function(Backend(np, operation))
    assert len(calls) == 1
    record, primals, shape = calls[0]
    assert record is not None
    return record, primals, shape


CASES = [
    pytest.param(
        lambda backend: wave.sw_translate(
            backend,
            [1, 2],
            [0, 1],
            1,
            1,
            0,
            1,
            np.array([[0.6], [0.8]]),
            0.8,
            [0.2, 0.4],
            singular=False,
        ),
        "spherical_translation",
        (),
        id="spherical-translation-broadcast",
    ),
    pytest.param(
        lambda backend: wave.sw_rotate(
            backend, np.array([[1], [2]]), 1, 1, [1, 2], 0, 1, 0.6, 0.8, 1.1
        ),
        "wignerd",
        (),
        id="spherical-rotation-mask",
    ),
    pytest.param(
        lambda backend: wave.cw_translate(
            backend, 0.2, [1, 2], 1, 0.2, 0, 1, 0.7, 0.8, 1.1, singular=False
        ),
        "cylindrical_translation",
        (),
        id="cylindrical-translation",
    ),
    pytest.param(
        lambda backend: wave.cw_rotate(
            backend, 0.2, np.array([[1], [2]]), 1, 0.2, [1, 2], 1, 0.6
        ),
        "wignerd",
        (1, 2),
        id="cylindrical-rotation-mask",
    ),
    pytest.param(
        lambda backend: wave.pw_translate(backend, 0.6, 0.8, 1.1, 0.2, 0.3, 0.4),
        "plane_phases",
        (),
        id="plane-translation-scalar",
    ),
    pytest.param(
        lambda backend: wave.pw_translate(
            backend, np.array([[0.6], [0.8]]), 0.3, 1.1, [0.2, 0.5], 0.3, 0.4
        ),
        "plane_phases",
        (),
        id="plane-translation-broadcast",
    ),
    pytest.param(
        lambda backend: wave.pw_translate(
            backend, np.empty((0, 1)), 0.3, 1.1, [0.2, 0.5], 0.3, 0.4
        ),
        "plane_phases",
        (),
        id="plane-translation-empty",
    ),
    pytest.param(
        lambda backend: wave.pw_to_sw(backend, 2, 1, 1, 0.6, 0.8, 1.1, 1),
        "plane_expansion",
        (),
        id="plane-to-spherical-scalar",
    ),
    pytest.param(
        lambda backend: wave.pw_to_cw(backend, 0.2, 1, 1, 0.6, 0.8, 0.2, 1),
        "plane_expansion",
        (0, 3),
        id="plane-to-cylindrical-active",
    ),
    pytest.param(
        lambda backend: wave.pw_to_cw(backend, 0.4, 1, 1, 0.6, 0.8, 0.2, 1),
        "plane_expansion",
        (0, 3),
        id="plane-to-cylindrical-unmatched",
    ),
    pytest.param(
        lambda backend: wave.pw_to_cw(backend, 0.4, 1, 1, 0.0, 0.0, 0.2, 1),
        "plane_expansion",
        (0, 3),
        id="plane-to-cylindrical-unmatched-axial",
    ),
    pytest.param(
        lambda backend: wave.pw_to_cw(
            backend,
            0.2,
            [0, 1, 2],
            [1, 1, 0],
            np.array([[0.6], [0.8]]),
            0.7,
            [0.2, 0.4, 0.2],
            1,
        ),
        "plane_expansion",
        (0, 3),
        id="plane-to-cylindrical-mixed-broadcast",
    ),
    pytest.param(
        lambda backend: wave.pw_to_cw(
            backend, 0.2, [0, 1], 1, np.empty((0, 1)), 0.7, 0.2, 1
        ),
        "plane_expansion",
        (0, 3),
        id="plane-to-cylindrical-empty",
    ),
    pytest.param(
        lambda backend: wave.pw_permute_xyz(
            backend, 0.6, 0.8, 1.1, 1, 0, poltype="parity"
        ),
        "plane_permutation",
        (),
        id="plane-permutation-scalar",
    ),
    pytest.param(
        lambda backend: wave.sw_periodic_to_cw(
            backend, 0.2, [0, 1], 1, [1, 2], [0, 1], 1, 1.3, 2.1
        ),
        "periodic_to_cw",
        (),
        id="periodic-spherical-to-cylindrical",
    ),
    pytest.param(
        lambda backend: wave.cw_to_sw(
            backend, [1, 2], [0, 1], 1, 0.2, [0, 1], 1, np.array([[1.3], [1.4]])
        ),
        "expansion",
        (),
        id="cylindrical-to-spherical-broadcast",
    ),
    pytest.param(
        lambda backend: wave.sw_translate_periodic(
            backend,
            1.3 + 0.1j,
            0.06,
            2.5,
            [0, 0, 0],
            ([1, 1], [0, 1], [1, 1]),
            rsin=[0.2, 0.1, 0.41],
            eta=0.7,
        ),
        "lattice_expansion",
        (),
        id="spherical-periodic-translation",
    ),
    pytest.param(
        lambda backend: wave.cw_translate_periodic(
            backend,
            1.3 + 0.1j,
            0.06,
            2.5,
            [0, 0, 0],
            ([0.2, 0.2], [0, 1], [1, 1]),
            rsin=[0.2, 0.1, 0.41],
            eta=0.7,
        ),
        "lattice_expansion",
        (),
        id="cylindrical-periodic-translation",
    ),
]


@pytest.mark.parametrize("function,forward,fixed", CASES)
def test_saved_wave_wrapper_preserves_derivatives_without_forward(
    monkeypatch, function, forward, fixed
):
    record, primals, shape = _capture(function)
    specs = record.state_spec(tuple(ArraySpec(x.shape, x.dtype) for x in primals))
    value, context = record(*primals)
    value = np.asarray(value)
    assert value.shape == shape
    state = record.save(context)
    for array, spec in zip(state, specs, strict=True):
        assert array.shape == spec.shape
        assert array.dtype == spec.dtype
    saved = tuple(array.copy() for array in state)

    directions = []
    for index, primal in enumerate(primals):
        direction = np.linspace(0.03, 0.08, primal.size).reshape(primal.shape)
        if np.iscomplexobj(primal):
            direction = direction + 0.02j
        directions.append(
            np.zeros_like(primal) if index in fixed else direction * (index + 1)
        )
    if forward == "cylindrical_translation":
        # These two arguments are one shared axial label. Their exact equality
        # must be preserved when taking a finite difference of the public value.
        directions[3] = directions[4].copy()
    directions = tuple(directions)
    cotangent = np.full(shape, 0.7 - 0.2j)
    (tangent,) = apply_pushforward(context, directions, primals, (value,))
    gradients = apply_pullback(context.pullback, (cotangent,), primals, conjugate=False)

    step = 1e-5
    plus, _ = record(
        *(x + step * dx for x, dx in zip(primals, directions, strict=True))
    )
    minus, _ = record(
        *(x - step * dx for x, dx in zip(primals, directions, strict=True))
    )
    assert_allclose(tangent, (plus - minus) / (2 * step), rtol=2e-6, atol=2e-8)
    assert_allclose(
        np.vdot(cotangent, tangent).real,
        sum(
            np.vdot(gradient, direction).real
            for gradient, direction in zip(gradients, directions, strict=True)
        ),
        rtol=2e-12,
        atol=2e-12,
    )

    def forbidden_forward(*_args, **_kwargs):
        pytest.fail("restoring or differentiating saved state ran a native forward")

    monkeypatch.setattr(wave.diff, forward, forbidden_forward)
    for _ in range(2):
        restored = record.restore(state, *primals)
        for scale in (1.0, -0.5):
            (actual,) = apply_pushforward(
                restored,
                tuple(scale * direction for direction in directions),
                primals,
                (value,),
            )
            assert_allclose(actual, scale * tangent, rtol=2e-13, atol=2e-13)
            actual_gradients = apply_pullback(
                restored.pullback, (scale * cotangent,), primals, conjugate=False
            )
            for actual, expected in zip(actual_gradients, gradients, strict=True):
                assert_allclose(actual, scale * expected, rtol=2e-13, atol=2e-13)
    for actual, expected in zip(state, saved, strict=True):
        assert_array_equal(actual, expected)


@pytest.mark.parametrize("x64", [False, True])
def test_jax_reuses_active_and_zero_coefficient_states(monkeypatch, x64):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from treams_rs import pw

    calls = []
    original = wave.diff.plane_expansion

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(wave.diff, "plane_expansion", counted)
    with jax.enable_x64(x64):
        function = jax.jit(
            lambda x: (
                pw.to_cw(
                    0.2,
                    np.array([1, 1, 2]),
                    np.array([1, 1, 0]),
                    x[0],
                    x[1],
                    np.array([0.2, 0.4, 0.2]),
                    1,
                ).real
            )
        )
        parameters = jnp.array([0.6, 0.8])
        value, pullback = jax.vjp(function, parameters)
        value.block_until_ready()
        assert len(calls) == 1
        for _ in range(2):
            jax.block_until_ready(pullback(jnp.ones_like(value)))
        assert len(calls) == 1

        value, pushforward = jax.linearize(function, parameters)
        value.block_until_ready()
        assert len(calls) == 2
        for _ in range(2):
            jax.block_until_ready(pushforward(jnp.ones_like(parameters)))
        assert len(calls) == 2

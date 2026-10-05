"""Numerical residuals survive the array-only framework callback boundary."""

import inspect
from collections import Counter
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from test_native_contexts import CASES, _cotangents, _split, case

from treams_rs import _native, diff
from treams_rs._saved import saved_record

from _support import assert_tree_allclose, jax_x64

# Object records share the authoritative native-contract fixtures. Broadcast
# records retain either computed state or their original inputs.
SAVED_CASES = {
    name: factory for name, factory in CASES.items() if name != "lattice_sum_record"
}

INPUT_CASES = {
    "coordinates": case(
        partial(diff.coordinates, kind="car2sph"),
        [[0.4, 0.7, 0.2], [1.3, 0.5, -0.5]],
    ),
    "vector_coordinates": case(
        partial(diff.vector_coordinates, kind="car2sph"),
        [[[0.2 + 0.1j, 0.3, 0.7]], [[-0.1j, 0.8, 0.5]]],
        [[0.4, 0.7, 0.2], [1.3, 0.5, -0.5]],
    ),
    "lattice_sum": case(
        partial(diff.lattice_sum, 2, [2, 3], -1),
        [2.1 + 0.2j, 2.3 + 0.1j],
        [0.1, 0.2],
        [[1.5, 0.0], [0.2, 1.4]],
        [[0.19, 0.11, 0.07], [0.21, -0.09, 0.05]],
        0.9 + 0.02j,
    ),
    "bessel": case(
        partial(diff.bessel, np.array([1, 2])[:, None]),
        [[0.7 + 0.2j, 1.1 + 0.1j]],
    ),
    "angular": case(
        partial(diff.angular, np.array([2, 3])[:, None], 1),
        [[0.2 + 0.1j, 0.4 - 0.1j]],
    ),
    "incgamma": case(
        partial(diff.incgamma, np.array([0.5, 1.5])[:, None]),
        [[0.7 + 0.2j, 1.1 + 0.1j]],
    ),
    "intkambe": case(
        partial(diff.intkambe, np.array([-3, -2])[:, None]),
        [[0.7 + 0.2j, 1.1 + 0.1j]],
        0.9 + 0.1j,
    ),
    "wignerd": case(
        partial(diff.wignerd, np.array([2, 3])[:, None], 1, -1),
        [[0.2 + 0.1j, 0.4 - 0.1j]],
        0.6 + 0.1j,
        [[0.1], [0.3]],
    ),
    "spherical_translation": case(
        partial(
            diff.spherical_translation,
            destination=(2, 1, 1),
            source=(1, 0, 1),
            singular=False,
        ),
        [[0.7 + 0.2j, 1.1 + 0.1j]],
        [[0.4], [0.7]],
        0.3,
    ),
    "cylindrical_translation": case(
        partial(
            diff.cylindrical_translation,
            order=np.array([1, 2])[:, None],
            singular=False,
        ),
        [[0.7 + 0.2j, 1.1 + 0.1j]],
        [[0.4], [0.7]],
        0.3,
        0.2,
    ),
    "vector_wave": case(
        partial(
            diff.vector_wave,
            function="vsw_rA",
            degree=np.array([2, 3])[:, None],
            order=1,
            pol=1,
        ),
        [[0.7 + 0.2j, 1.1 + 0.1j]],
        [[0.4], [0.7]],
        0.3,
    ),
}

ALL_CASES = SAVED_CASES | INPUT_CASES


def _directions(pushforward, gradients):
    """Use valid gradient shapes as seeds, excluding iterative solve reports."""
    directions = gradients if isinstance(gradients, tuple) else (gradients,)
    parameters = tuple(inspect.signature(pushforward).parameters.values())
    if any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    ):
        return directions
    return directions[: len(parameters)]


def _negative(value):
    if isinstance(value, tuple | list):
        return type(value)(_negative(item) for item in value)
    return -value


@pytest.mark.gradients
@pytest.mark.parametrize("name", ALL_CASES)
def test_saved_context_restores_repeated_pushforwards_and_pullbacks(name):
    fixture = ALL_CASES[name]()
    values, original = _split(fixture.record(*fixture.arrays))
    cotangents = _cotangents(values)
    pullback = getattr(original, fixture.method)
    pushforward_name = fixture.method.replace("pullback", "pushforward")
    pushforward = getattr(original, pushforward_name)
    gradients = pullback(*cotangents)
    directions = _directions(pushforward, gradients)
    tangent = pushforward(*directions)
    negative_directions, negative_cotangents = (
        _negative(directions),
        _negative(cotangents),
    )
    negative_tangent = pushforward(*negative_directions)
    negative_gradients = pullback(*negative_cotangents)
    probes = (
        (directions, cotangents, tangent, gradients),
        (
            negative_directions,
            negative_cotangents,
            negative_tangent,
            negative_gradients,
        ),
        (directions, cotangents, tangent, gradients),
    )

    if name in INPUT_CASES:
        record = saved_record(fixture.record)
        assert record is not None and record.needs_primals
        assert record.state_spec(()) == record.save(original) == ()
        restored = record.restore((), *fixture.arrays)
    else:
        state = original._state()
        assert state.dtype == np.uint8
        assert state.ndim == 1
        assert state.flags.c_contiguous
        saved = state.copy()
        restored = type(original)._from_state(state)
        assert_array_equal(restored._state(), saved)
        state[:] = 0

    # Neither the original input arrays, returned values nor transport buffer
    # may remain borrowed by an original or reconstructed native residual.
    for array in (*fixture.arrays, *values):
        if isinstance(array, np.ndarray) and array.dtype.kind in "fc":
            array[...] = np.nan
    for context in (original, restored):
        for seeds, covectors, expected_tangent, expected_gradients in probes:
            assert_tree_allclose(
                getattr(context, pushforward_name)(*seeds),
                expected_tangent,
                rtol=0,
                atol=0,
            )
            assert_tree_allclose(
                getattr(context, fixture.method)(*covectors),
                expected_gradients,
                rtol=0,
                atol=0,
            )
        if name not in INPUT_CASES:
            assert_array_equal(context._state(), saved)


@pytest.mark.interface
@pytest.mark.parametrize("name", SAVED_CASES)
def test_saved_context_rejects_truncated_or_trailing_state(name):
    fixture = SAVED_CASES[name]()
    _, context = _split(fixture.record(*fixture.arrays))
    state = context._state()
    for invalid in (state[:-1], np.append(state, np.uint8(0))):
        with pytest.raises(ValueError):
            type(context)._from_state(invalid)
    assert_array_equal(type(context)._from_state(state)._state(), state)


@pytest.mark.interface
@pytest.mark.parametrize(
    "name", [name for name, factory in SAVED_CASES.items() if factory().arrays]
)
def test_saved_state_shape_depends_on_shapes_not_physical_values(name):
    fixture = SAVED_CASES[name]()
    _, first = _split(fixture.record(*fixture.arrays))
    changed = list(fixture.arrays)
    changed[0] = changed[0] * 1.001
    _, second = _split(fixture.record(*changed))
    assert first._state().shape == second._state().shape


@pytest.mark.interface
def test_saved_context_fixtures_cover_every_native_derivative_class():
    covered = {
        type(_split(fixture.record(*fixture.arrays))[1]).__name__
        for factory in ALL_CASES.values()
        for fixture in (factory(),)
    }
    native = {
        name
        for name in dir(_native)
        if isinstance(cls := getattr(_native, name), type) and hasattr(cls, "pullback")
    }
    assert covered == native


@pytest.mark.gradients
@pytest.mark.parametrize(
    "name",
    [
        name
        for name in INPUT_CASES
        if name not in ("coordinates", "vector_coordinates", "lattice_sum")
    ],
)
@pytest.mark.parametrize("shape", ["scalar", "empty"])
def test_input_context_reconstruction_preserves_broadcast_shapes(name, shape):
    fixture = INPUT_CASES[name]()
    inputs = tuple(
        np.asarray(value.reshape(-1)[0])
        if shape == "scalar"
        else value[..., :0]
        if value.ndim and value.shape[-1] > 1
        else value
        for value in fixture.arrays
    )
    value, original = fixture.record(*inputs)
    prepared = saved_record(fixture.record)
    restored = prepared.restore((), *inputs)
    directions = tuple(np.ones_like(value) for value in inputs)
    assert_tree_allclose(
        restored.pushforward(*directions),
        original.pushforward(*directions),
        rtol=0,
        atol=0,
    )
    assert_tree_allclose(
        restored.pullback(np.ones_like(value)),
        original.pullback(np.ones_like(value)),
        rtol=0,
        atol=0,
    )


@pytest.mark.gradients
@pytest.mark.parametrize("name", INPUT_CASES)
@pytest.mark.parametrize("x64", [False, True])
def test_input_context_jax_derivatives_do_not_replay_values(name, x64, monkeypatch):
    jax = pytest.importorskip("jax")
    from treams_rs import jax as tj

    calls = Counter()
    for suffix in ("_record", "_record_scalar"):
        binding_name = name + suffix
        if not hasattr(_native, binding_name):
            continue
        evaluate = getattr(_native, binding_name)

        def watched(*args, _evaluate=evaluate):
            calls["forward"] += 1
            return _evaluate(*args)

        monkeypatch.setattr(_native, binding_name, watched)

    fixture = INPUT_CASES[name]()
    with jax_x64(x64):
        inputs = tuple(jax.numpy.asarray(value) for value in fixture.arrays)
        directions = tuple(jax.numpy.ones_like(value) * 0.1 for value in inputs)
        function = jax.jit(tj.wrap(fixture.record, *inputs))
        calls.clear()
        value, forward = jax.linearize(function, *inputs)
        jax.block_until_ready(value)
        first = jax.block_until_ready(forward(*directions))
        second = jax.block_until_ready(forward(*(2 * d for d in directions)))
        assert_allclose(second, 2 * first, rtol=3e-6, atol=1e-7)
        assert calls == {"forward": 1}

        calls.clear()
        value, reverse = jax.vjp(function, *inputs)
        jax.block_until_ready(value)
        weights = jax.numpy.ones_like(value)
        first = jax.block_until_ready(reverse(weights))
        second = jax.block_until_ready(reverse(weights))
        assert_tree_allclose(first, second, rtol=0, atol=0)
        assert calls == {"forward": 1}

        calls.clear()
        jax.block_until_ready(
            jax.jit(jax.jacfwd(function, holomorphic=np.iscomplexobj(inputs[0])))(
                *inputs
            )
        )
        # With only derivatives requested, no computed state depends on the
        # values, so JAX can omit their evaluation altogether.
        assert not calls


@pytest.mark.gradients
def test_saved_zero_contrast_metric_keeps_its_derivative_error():
    matrix = -0.1 * np.eye(6)
    polarizations = [1, 0, 1, 0, 1, 0]
    _, original = diff.tmatrix_metric(matrix, polarizations=polarizations, kind="chi")
    state = original._state()
    restored = type(original)._from_state(state)
    for context in (original, restored):
        for _ in range(2):
            with pytest.raises(ValueError, match="zero contrast"):
                context.pullback(1.0)
            with pytest.raises(ValueError, match="zero contrast"):
                context.pushforward(np.ones_like(matrix), np.zeros(2))
            assert_allclose(context.pushforward(np.zeros_like(matrix), np.zeros(2)), 0)
            assert_tree_allclose(
                context.pullback(0.0), (np.zeros_like(matrix), np.zeros(2))
            )
        assert_array_equal(context._state(), state)

    # An optional derivative error cannot change the static transport shape.
    regular = matrix.copy()
    regular[0, 0] *= 1.1
    _, regular_context = diff.tmatrix_metric(
        regular, polarizations=polarizations, kind="chi"
    )
    assert regular_context._state().shape == state.shape

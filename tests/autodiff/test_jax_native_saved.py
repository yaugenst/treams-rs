"""Native factorizations survive repeated JAX derivative applications."""

import gc
from collections import Counter
from dataclasses import replace
from functools import partial, wraps
from weakref import ref

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytestmark = pytest.mark.gradients

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from treams_rs import (  # noqa: E402
    _native,
    _saved,
    diff,
)
from treams_rs import jax as tj  # noqa: E402
from treams_rs._records import DerivativeContext  # noqa: E402
from treams_rs._saved import (  # noqa: E402
    native_record,
    native_state,
    preserve_state,
    primal_record,
    saved_record,
)

from _support import jax_x64  # noqa: E402


@pytest.mark.interface
def test_all_direct_diff_records_declare_saved_state():
    factories = {
        "factor_interaction",
        "factor_interaction_blocks",
        "sphere_cluster_factor",
    }
    for name in diff.__all__:
        if name not in factories:
            assert saved_record(getattr(diff, name)) is not None, name


@pytest.mark.interface
@pytest.mark.parametrize("kind", ["native_state", "native_record", "primal_record"])
@pytest.mark.parametrize("prepare", [False, True])
def test_record_metadata_does_not_require_cyclic_gc(kind, prepare):
    enabled = gc.isenabled()
    gc.disable()
    try:

        def record(matrix, rhs):
            return diff.solve(matrix, rhs)

        if kind == "primal_record":
            record = primal_record(lambda *_arguments: None)(record)
        else:
            decorate = native_record if kind == "native_record" else native_state
            record = decorate(_native.SolveContext, lambda _inputs: (2, 1))(record)
        owner = ref(record)
        prepared = saved_record(record) if prepare else None
        del record
        if prepare:
            # A live derivative operation intentionally owns its evaluation.
            assert owner() is not None
            del prepared
        assert owner() is None
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("offset", [None, -0.25])
def test_primal_record_binds_defaults_and_partials_without_bypassing_wrappers(offset):
    calls = Counter()

    def context(scale, x, offset):
        derivative = 2 * scale * (x + offset)
        return DerivativeContext(
            lambda gradient: derivative * gradient,
            lambda tangent: derivative * tangent,
        )

    @primal_record(context)
    def record(scale, x, *, offset=0.5):
        calls["forward"] += 1
        return scale * (x + offset) ** 2, context(scale, x, offset)

    @wraps(record)
    def doubled(*args, **kwargs):
        value, original = record(*args, **kwargs)
        return 2 * value, DerivativeContext(
            lambda gradient: original.pullback(2 * gradient),
            lambda tangent: 2 * original.pushforward(tangent),
        )

    options = {} if offset is None else {"offset": offset}
    bound = partial(record, 3.0, **options)
    copied = partial(doubled, 3.0, **options)
    assert saved_record(bound) is not None
    assert saved_record(doubled) is saved_record(copied) is None
    expected = 0.2 + (0.5 if offset is None else offset)
    with jax_x64():
        for candidate, scale, forwards in ((bound, 1, 1), (copied, 2, 2)):
            function = jax.jit(tj.wrap(candidate, np.array(0.2)))
            calls.clear()
            value, gradient = jax.value_and_grad(function)(0.2)
            assert_allclose(value, scale * 3 * expected**2)
            assert_allclose(gradient, scale * 6 * expected)
            assert calls == {"forward": forwards}
            calls.clear()
            _, linear = jax.linearize(function, 0.2)
            assert_allclose(linear(0.4), scale * 6 * expected * 0.4)
            assert_allclose(linear(-0.3), scale * 6 * expected * -0.3)
            assert calls == {"forward": 1 if candidate is bound else 3}


@pytest.mark.interface
@pytest.mark.parametrize("native", [False, True])
def test_output_check_resolves_saved_state_only_when_requested(monkeypatch, native):
    def source(a, b):
        return diff.solve(a, b)

    if native:
        source = native_state(_native.SolveContext, lambda _inputs: (1, 1))(source)
    calls = []
    lookup = _saved.saved_record

    def watched(record):
        if record is source:
            calls.append("lookup")
        return lookup(record)

    monkeypatch.setattr(_saved, "saved_record", watched)

    def checked(*values):
        result, context = source(*values)
        assert result.shape == (1, 1)
        return result, context

    assert preserve_state(source, checked) is checked
    assert_allclose(checked(np.array([[2.0]]), np.array([[1.0]]))[0], [[0.5]])
    assert calls == []
    prepared = _saved.saved_record(checked)
    assert calls == ["lookup"]
    if native:
        assert prepared is not None
        assert prepared.evaluate is checked
        assert not prepared.needs_primals
    else:
        assert prepared is None

    @wraps(checked)
    def copied(*values):
        return checked(*values)

    assert _saved.saved_record(copied) is None


@pytest.mark.interface
@pytest.mark.parametrize("prepare", [False, True])
def test_output_check_state_does_not_require_cyclic_gc(prepare):
    def make_checked():
        @native_state(_native.SolveContext, lambda _inputs: (1, 1))
        def source(a, b):
            return diff.solve(a, b)

        def checked(*values):
            return source(*values)

        return ref(source), preserve_state(source, checked)

    enabled = gc.isenabled()
    gc.disable()
    try:
        source_owner, checked = make_checked()
        checked_owner = ref(checked)
        prepared = saved_record(checked) if prepare else None
        del checked
        if prepare:
            assert checked_owner() is not None
            assert source_owner() is not None
            del prepared
        assert checked_owner() is None
        assert source_owner() is None
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("name", ["solve", "interaction"])
def test_copied_state_metadata_cannot_bypass_a_record_wrapper(name):
    original = getattr(diff, name)
    calls = Counter()

    @wraps(original)
    def doubled(*values):
        calls["forward"] += 1
        value, context = original(*values)
        return 2 * value, DerivativeContext(
            lambda gradient: context.pullback(2 * gradient),
            lambda *tangents: 2 * context.pushforward(*tangents),
        )

    # solve uses native_state; interaction uses the named native_record factory.
    assert saved_record(doubled) is None
    assert saved_record(partial(doubled)) is None
    with jax_x64():
        matrix = jnp.array([[2.0, 0.1], [0.2, 1.5]])
        rhs = jnp.array([[0.4, 0.1], [0.1, 0.3]])
        function = jax.jit(tj.wrap(doubled, matrix, rhs))
        reference = jax.jit(tj.wrap(original, matrix, rhs))
        calls.clear()
        assert_allclose(function(matrix, rhs), 2 * reference(matrix, rhs))
        assert calls == {"forward": 1}
        directions = (jnp.full_like(matrix, 0.03), jnp.full_like(rhs, -0.02))
        calls.clear()
        value, tangent = jax.jvp(function, (matrix, rhs), directions)
        _, expected = jax.jvp(reference, (matrix, rhs), directions)
        assert_allclose(tangent, 2 * expected)
        assert calls == {"forward": 2}
        calls.clear()
        _, pullback = jax.vjp(function, matrix, rhs)
        _, reference_pullback = jax.vjp(reference, matrix, rhs)
        actual = pullback(jnp.ones_like(value))
        expected = reference_pullback(jnp.ones_like(value))
        for result, single in zip(actual, expected, strict=True):
            assert_allclose(result, 2 * single)
        assert calls == {"forward": 2}


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("name", ["solve", "eig", "svdvals", "bessel"])
def test_native_saved_record_reuses_forward_for_vjp_and_linearize(name, x64):
    with jax_x64(x64):
        matrix = jnp.array([[2 + 0.2j, 0.1 - 0.3j], [0.2j, 1.5 - 0.1j]])
        rhs = jnp.array([[0.4 + 0.1j], [0.7 - 0.2j]])
        if name == "solve":
            record, inputs = diff.solve, (matrix, rhs)
        elif name == "bessel":
            record, inputs = partial(diff.bessel, 2, spherical=True), (rhs,)
        else:
            record, inputs = getattr(diff, name), (matrix,)
        prepared = saved_record(record)
        assert prepared is not None
        calls = Counter()

        def evaluate(*values):
            calls["forward"] += 1
            return record(*values)

        function = jax.jit(tj.wrap(replace(prepared, evaluate=evaluate), *inputs))
        directions = tuple(jnp.full_like(value, 0.03 + 0.02j) for value in inputs)
        calls.clear()
        value, pullback = jax.vjp(function, *inputs)
        weights = jax.tree.map(jnp.ones_like, value)
        first = jax.block_until_ready(pullback(weights))
        second = jax.block_until_ready(pullback(weights))
        for left, right in zip(first, second, strict=True):
            assert_allclose(left, right)
        assert calls == {"forward": 1}

        calls.clear()
        _, pushforward = jax.linearize(function, *inputs)
        tangent = jax.block_until_ready(pushforward(*directions))
        twice = jax.block_until_ready(pushforward(*(2 * value for value in directions)))
        for left, right in zip(
            jax.tree.leaves(tangent), jax.tree.leaves(twice), strict=True
        ):
            assert_allclose(2 * left, right, rtol=3e-6, atol=1e-7)
        assert calls == {"forward": 1}

        # Compare both native directions using JAX's bilinear cotangent pairing.
        forward_pairing = sum(
            np.real(np.sum(g * dy))
            for g, dy in zip(
                jax.tree.leaves(weights), jax.tree.leaves(tangent), strict=True
            )
        )
        reverse_pairing = sum(
            np.real(np.sum(g * dx)) for g, dx in zip(first, directions, strict=True)
        )
        assert_allclose(forward_pairing, reverse_pairing, rtol=4e-6, atol=1e-7)


def test_native_solve_jacfwd_factors_once_for_all_directions():
    with jax_x64():
        matrix = jnp.array([[2.0, 0.1], [0.2, 1.5]])
        rhs = jnp.array([[0.4], [0.7]])
        calls = Counter()
        prepared = saved_record(diff.solve)
        assert prepared is not None

        def evaluate(a, b):
            calls["forward"] += 1
            return diff.solve(a, b)

        function = tj.wrap(replace(prepared, evaluate=evaluate), matrix, rhs)
        calls.clear()
        actual = jax.jit(jax.jacfwd(function, argnums=1))(matrix, rhs)
        assert_allclose(
            np.asarray(actual).reshape(2, 2), np.linalg.inv(matrix), atol=1e-14
        )
        assert calls == {"forward": 1}


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("checked", [False, True])
def test_native_solve_retains_only_state_and_input_metadata(x64, checked):
    with jax_x64(x64):
        matrix = jnp.array([[2.0, 0.1], [0.2, 1.5]], dtype=jnp.float32)
        rhs = jnp.array([[0.4], [0.7]], dtype=jnp.float32)

        def function(a, b):
            return (
                tj._backend.apply(diff.solve, b.shape, a, b)
                if checked
                else tj.solve(a, b)
            )

        _, pushforward = jax.linearize(function, matrix, rhs)
        directions = (jnp.zeros_like(matrix), jnp.ones_like(rhs))
        residuals = jax.make_jaxpr(pushforward)(*directions).consts
        assert residuals
        assert all(value.dtype == np.uint8 for value in residuals)
        assert_allclose(
            pushforward(*directions),
            np.linalg.solve(matrix, np.ones_like(rhs)),
            rtol=3e-7,
        )
        _, pullback = jax.vjp(function, matrix, rhs)
        for _ in range(2):
            gradient = pullback(
                jnp.ones((2, 1), dtype=jnp.complex128 if x64 else jnp.complex64)
            )
            assert all(value.dtype == jnp.float32 for value in gradient)
            assert_allclose(
                gradient[1],
                np.linalg.solve(np.asarray(matrix).T, np.ones_like(rhs)),
                rtol=3e-7,
            )


@pytest.mark.parametrize("x64", [False, True])
def test_value_dependent_context_mapping_retains_its_primals(x64):
    def map_context(context, matrix, _rhs):
        sign = np.sign(matrix)

        def pullback(gradient):
            matrix_gradient, rhs_gradient = context.pullback(gradient)
            return sign * matrix_gradient, rhs_gradient

        def pushforward(matrix_tangent, rhs_tangent):
            return context.pushforward(sign * matrix_tangent, rhs_tangent)

        return DerivativeContext(pullback, pushforward, context)

    @native_state(
        _native.SolveContext,
        lambda inputs: (inputs[0].shape[0], inputs[1].shape[1]),
        map_context=map_context,
    )
    def record(matrix, rhs):
        result, context = diff.solve(np.abs(matrix), rhs)
        return result, map_context(context, matrix, rhs)

    with jax_x64(x64):
        matrix = jnp.array([[-2.0, 0.1], [0.2, -1.5]], dtype=jnp.float32)
        rhs = jnp.array([[0.4], [0.7]], dtype=jnp.float32)
        function = jax.jit(tj.wrap(record, matrix, rhs))

        def reference(matrix, rhs):
            return jnp.linalg.solve(jnp.abs(matrix), rhs)

        directions = (jnp.full_like(matrix, 0.1), jnp.full_like(rhs, -0.3))
        _, linear = jax.linearize(function, matrix, rhs)
        _, expected_linear = jax.linearize(reference, matrix, rhs)
        for scale in (1.0, -0.4):
            scaled = tuple(scale * value for value in directions)
            assert_allclose(linear(*scaled), expected_linear(*scaled), rtol=5e-7)
        value, pullback = jax.vjp(function, matrix, rhs)
        expected_value, expected_pullback = jax.vjp(reference, matrix, rhs)
        for actual, expected in zip(
            pullback(jnp.ones_like(value)),
            expected_pullback(jnp.ones_like(expected_value)),
            strict=True,
        ):
            assert actual.dtype == jnp.float32
            assert_allclose(actual, expected, rtol=5e-7)


@pytest.mark.parametrize(
    ("matrix", "expected", "message"),
    [
        (np.eye(2) * 0.5, 0.0, "zero contrast"),
        (np.diag([0.0, 0.5]), 1.0, "below numerical resolution"),
        (
            np.array([[0.3, 0.0], [0.1, 0.2]]),
            np.sqrt(1 / 7),
            "below numerical resolution",
        ),
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_chirality_keeps_primal_and_deferred_error_in_saved_maps(
    matrix, expected, message, compiled
):
    with jax_x64(True):
        matrix = jnp.asarray(matrix)

        ks = jnp.ones(2)
        wrapped = tj.wrap(
            partial(diff.tmatrix_metric, polarizations=[0, 1], metric="chi"), matrix, ks
        )

        def metric(operator):
            return wrapped(operator, ks)

        function = jax.jit(metric) if compiled else metric
        value, pullback = jax.vjp(function, matrix)
        assert_allclose(value, expected, rtol=1e-14, atol=1e-15)
        # Recording/saving succeeds; only a nonzero derivative request fails.
        assert_array_equal(pullback(jnp.zeros_like(value))[0], np.zeros_like(matrix))
        with pytest.raises((ValueError, jax.errors.JaxRuntimeError), match=message):
            jax.block_until_ready(pullback(jnp.ones_like(value)))
        value, pushforward = jax.linearize(function, matrix)
        assert_allclose(value, expected, rtol=1e-14, atol=1e-15)
        assert_array_equal(pushforward(jnp.zeros_like(matrix)), 0.0)
        with pytest.raises((ValueError, jax.errors.JaxRuntimeError), match=message):
            jax.block_until_ready(pushforward(jnp.ones_like(matrix)))

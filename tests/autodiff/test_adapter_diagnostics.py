"""Malformed custom pullbacks report which parameter needs repair."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from treams_rs._records import apply_pullback, input_array, native_array, run_record

pytestmark = [pytest.mark.gradients, pytest.mark.interface]


@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.int64])
def test_parameter_dtype_reports_actual_precision_and_remedy(dtype):
    with pytest.raises(TypeError) as error:
        input_array(np.ones(2, dtype=dtype))
    message = str(error.value)
    assert "float64 or complex128" in message
    assert f"received {np.dtype(dtype)}" in message
    assert "Cast the parameter explicitly" in message


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_native_input_promotion_preserves_values(dtype):
    value = np.array([0.25, 0.5], dtype=dtype)
    actual = native_array(value)
    assert actual.dtype == (np.complex128 if np.iscomplexobj(value) else np.float64)
    np.testing.assert_array_equal(actual, value)


@pytest.mark.parametrize("dtype", [np.float16, np.int32, np.bool_])
def test_native_input_promotion_rejects_unsupported_dtypes(dtype):
    with pytest.raises(TypeError, match=f"received {np.dtype(dtype)}"):
        native_array(np.ones(2, dtype=dtype))


def test_empty_output_tuple_explains_missing_result():
    with pytest.raises(ValueError, match="received an empty output tuple"):
        run_record(lambda: ((), lambda: ()), ())


@pytest.mark.parametrize("extra", [-1, 1])
@pytest.mark.parametrize("expected", range(1, 6))
def test_pullback_arity_reports_both_counts(expected, extra):
    primals = (np.asarray(1.0),) * expected
    actual = expected + extra
    with pytest.raises(ValueError) as error:
        apply_pullback(
            lambda: (np.asarray(1.0),) * actual, (), primals, conjugate=False
        )
    assert f"expected {expected} gradients, received {actual}" in str(error.value)
    assert "dynamic parameter order" in str(error.value)


@given(shape=st.lists(st.integers(1, 3), max_size=3).map(tuple))
def test_gradient_shape_reports_parameter_index_and_both_shapes(shape):
    wrong_shape = (*shape, 2)
    primals = (np.asarray(1.0), np.ones(shape))
    with pytest.raises(ValueError) as error:
        apply_pullback(
            lambda: (np.asarray(1.0), np.ones(wrong_shape)),
            (),
            primals,
            conjugate=False,
        )
    assert f"parameter[1] expected shape {shape}, received {wrong_shape}" in str(
        error.value
    )


@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_scalar_gradient_accepts_native_singleton_shape(conjugate, dtype):
    primal = np.asarray(0.5, dtype=dtype)
    (gradient,) = apply_pullback(
        lambda: np.array([2.0 + 1.0j]), (), (primal,), conjugate=conjugate
    )
    assert gradient.shape == ()
    assert gradient.dtype == primal.dtype
    expected = 2.0 - 1.0j if conjugate else 2.0 + 1.0j
    np.testing.assert_array_equal(
        gradient, expected if np.iscomplexobj(primal) else expected.real
    )


@pytest.mark.parametrize("conjugate", [False, True])
def test_arity_and_shape_are_checked_in_both_conventions(conjugate):
    # A bare array is one gradient; JAX's conjugating path checks it the same way.
    primal = (np.array([0.3]),)
    with pytest.raises(ValueError, match="one gradient"):
        apply_pullback(lambda g: (g, g), primal, primal, conjugate=conjugate)
    with pytest.raises(ValueError, match="gradient shape"):
        apply_pullback(lambda g: np.array(1.0), primal, primal, conjugate=conjugate)


@given(shape=st.lists(st.integers(1, 3), max_size=3).map(tuple))
def test_reshape_accepts_only_gradients_of_the_primal_size(shape):
    primal = np.zeros(shape, dtype=np.float32)
    flat = np.arange(primal.size) + 0.5j
    (gradient,) = apply_pullback(
        lambda: flat, (), (primal,), conjugate=False, reshape=True
    )
    assert gradient.shape == primal.shape
    assert gradient.dtype == primal.dtype
    np.testing.assert_array_equal(gradient.ravel(), flat.real)
    with pytest.raises(ValueError, match="gradient shape"):
        apply_pullback(
            lambda: np.append(flat, 1.0), (), (primal,), conjugate=False, reshape=True
        )


def _advect_gradient_error(pullback, *primals):
    # Advect raises the transpose rule's ValueError as the cause of a RuntimeError.
    with pytest.raises(RuntimeError) as error:
        _advect_gradient(pullback, *primals)
    assert isinstance(error.value.__cause__, ValueError)
    return str(error.value.__cause__)


def _advect_gradient(pullback, *primals):
    advect = pytest.importorskip("advect")
    from treams_rs import advect as tr

    def record(*values):
        return np.asarray(values[0], dtype=complex), pullback

    def loss(*values):
        return advect.numpy.sum(advect.numpy.real(tr._operation(record, *values)))

    return advect.grad(loss, argnums=tuple(range(len(primals))))(*primals)


@pytest.mark.parametrize("extra", [-1, 1])
def test_advect_pullback_arity_reports_both_counts(extra):
    primals = (np.ones(2), np.ones(3))
    actual = len(primals) + extra
    message = _advect_gradient_error(lambda g: (np.ones(2),) * actual, *primals)
    assert f"expected 2 gradients, received {actual}" in message
    assert "dynamic parameter order" in message


def test_advect_gradient_size_reports_parameter_index_and_both_shapes():
    message = _advect_gradient_error(
        lambda g: (g.real, np.ones(4)), np.ones(2), np.ones(3)
    )
    assert "parameter[1] expected shape (3,), received (4,)" in message


def test_advect_reshapes_a_gradient_of_the_same_size():
    # A native gradient may store a scalar parameter as shape (1,).
    primals = (np.ones(2), np.asarray(0.5, dtype=np.float32))
    _, gradient = _advect_gradient(lambda g: (g.real, np.array([2.0 + 1.0j])), *primals)
    assert gradient.shape == ()
    assert gradient.dtype == np.float32
    assert gradient == 2.0


def test_advect_empty_output_tuple_explains_missing_result():
    advect = pytest.importorskip("advect")
    from treams_rs import advect as tr

    with pytest.raises(ValueError, match="received an empty output tuple"):
        tr._operation(lambda x: ((), lambda g: g), advect.numpy.asarray(1.0))

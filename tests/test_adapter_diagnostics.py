"""Malformed custom pullbacks identify the parameter contract that needs repair."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from treams_rs._adapters import execute, gradients, input_array


@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.int64])
def test_parameter_dtype_reports_actual_precision_and_remedy(dtype):
    with pytest.raises(TypeError) as error:
        input_array(np.ones(2, dtype=dtype))
    message = str(error.value)
    assert "float64 or complex128" in message
    assert f"received {np.dtype(dtype)}" in message
    assert "Cast the parameter explicitly" in message


def test_empty_output_tuple_explains_missing_result():
    with pytest.raises(ValueError, match="received an empty output tuple"):
        execute(lambda: ((), lambda: ()), ())


@given(expected=st.integers(1, 5), extra=st.sampled_from([-1, 1]))
def test_pullback_arity_reports_both_counts(expected, extra):
    primals = (np.asarray(1.0),) * expected
    actual = expected + extra
    with pytest.raises(ValueError) as error:
        gradients(lambda: (np.asarray(1.0),) * actual, (), primals, conjugate=False)
    assert f"expected {expected} gradients, received {actual}" in str(error.value)
    assert "dynamic parameter order" in str(error.value)


@given(shape=st.lists(st.integers(1, 3), max_size=3).map(tuple))
def test_gradient_shape_reports_parameter_index_and_both_shapes(shape):
    wrong_shape = (*shape, 1)
    primals = (np.asarray(1.0), np.ones(shape))
    with pytest.raises(ValueError) as error:
        gradients(
            lambda: (np.asarray(1.0), np.ones(wrong_shape)),
            (),
            primals,
            conjugate=False,
        )
    assert f"parameter[1] expected shape {shape}, received {wrong_shape}" in str(
        error.value
    )

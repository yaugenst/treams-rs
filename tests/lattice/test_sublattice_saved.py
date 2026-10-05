"""Sublattice records retain row topology without reconstructing a lattice."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from treams_rs import _lattice
from treams_rs._saved import ArraySpec, SavedRecord

pytestmark = pytest.mark.gradients


class RecordingBackend:
    array = staticmethod(np.asarray)

    def apply(self, record, shape, array, *, real):
        assert isinstance(record, SavedRecord)
        assert real
        self.record = record
        value, self.context = record(array)
        assert value.shape == shape
        return value


@pytest.mark.parametrize(
    ("array", "kpar", "spherical", "rows", "columns"),
    [
        ([[0, 0, 4], [2, 0.3, 0], [0.2, 3, 0]], [0, 0], True, [1, 2], [0, 1]),
        ([[0, 0, 4], [2, 0.3, 0], [0.2, 3, 0]], [0], True, [0], [2]),
        ([[0, 3, 0], [0, 0, 4], [2, 0, 0]], [0], False, [2], [0]),
        ([[0, 3], [2, 0]], [0], False, [1], [0]),
    ],
)
def test_saved_sublattice_restores_both_derivatives_without_forward(
    monkeypatch, array, kpar, spherical, rows, columns
):
    # Reciprocal values play no part in row selection: keep this test in Python.
    monkeypatch.setattr(_lattice._native, "cell_reciprocal", np.zeros_like)
    array = np.asarray(array, dtype=np.float64)
    backend = RecordingBackend()
    value, _ = _lattice.framework_cell(backend, array, kpar, spherical)
    selection = np.ix_(rows, columns)
    assert_array_equal(value, array[selection])

    record = backend.record
    schema = record.state_spec((ArraySpec(array.shape, array.dtype),))
    state = record.save(backend.context)
    assert schema == (ArraySpec((len(rows),), np.dtype(np.intp)),)
    assert len(state) == 1
    assert_array_equal(state[0], rows)
    assert state[0].dtype == schema[0].dtype
    state[0].flags.writeable = False
    del backend

    def no_reconstruction(*_args, **_kwargs):
        pytest.fail("restoring derivative state must not reconstruct the lattice")

    monkeypatch.setattr(_lattice.Lattice, "__init__", no_reconstruction)
    tangent = np.arange(array.size, dtype=np.float64).reshape(array.shape)
    cotangent = np.arange(value.size, dtype=np.float64).reshape(value.shape) + 1
    expected = np.zeros_like(array)
    expected[selection] = cotangent
    for _ in range(2):
        context = record.restore(state, array)
        for _ in range(2):
            assert_array_equal(context.pushforward(tangent), tangent[selection])
            (gradient,) = context.pullback(cotangent)
            assert_array_equal(gradient, expected)
    assert_array_equal(state[0], rows)

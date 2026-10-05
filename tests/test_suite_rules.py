"""Guards for the shared test infrastructure in conftest.py and _support.py."""

import os

import numpy as np
import pytest
from hypothesis import find, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from _support import (
    LAYOUTS,
    arrange,
    assert_reusable_context,
    assert_tree_allclose,
    complex_arrays,
    layouts,
    strided_copies,
)

pytestmark = pytest.mark.interface


def test_every_collected_test_has_a_registered_category(
    pytestconfig, category_markers, uncategorized_tests
):
    registered = {line.split(":")[0] for line in pytestconfig.getini("markers")}
    assert set(category_markers) <= registered
    assert uncategorized_tests == []


def test_hypothesis_profile_is_owned_by_conftest(pytestconfig):
    # Runs after collection, so a module that loads a profile on import fails it.
    expected = pytestconfig.getoption("hypothesis_profile") or os.environ.get(
        "HYPOTHESIS_PROFILE", "dev"
    )
    assert settings.get_current_profile_name() == expected
    assert settings.default.deadline is None
    assert settings.default.suppress_health_check == ()


# Storage order and stride signs of the layouts that read contiguous storage
# backwards; NumPy reports none of them as contiguous.
FLIPPED = {
    "flipped": ("C", (-1, -1, -1)),
    "F-flipped": ("F", (-1, -1, -1)),
    "C-flip-first": ("C", (-1, 1, 1)),
    "C-flip-last": ("C", (1, 1, -1)),
    "F-flip-first": ("F", (-1, 1, 1)),
    "F-flip-last": ("F", (1, 1, -1)),
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_layouts_hold_equal_values_in_fresh_memory(layout):
    a = np.arange(24).reshape(2, 3, 4) * (0.5 - 0.25j)
    view = arrange(a, layout)
    assert view.dtype == a.dtype
    assert np.array_equal(view, a)
    assert not np.shares_memory(view, a)
    strides = np.array(view.strides)
    contiguous = view.flags.c_contiguous or view.flags.f_contiguous
    if layout in FLIPPED:
        order, signs = FLIPPED[layout]
        storage = np.empty_like(a, order=order).strides
        assert not contiguous, (layout, view.flags)
        assert tuple(np.sign(strides)) == signs, (layout, view.strides)
        assert tuple(np.abs(strides)) == storage, (layout, view.strides)
        return
    expected = {
        "C": (view.flags.c_contiguous, (strides > 0).all()),
        "F": (view.flags.f_contiguous, (strides > 0).all()),
        "block-F": (not contiguous, abs(strides[-2]) < abs(strides[-1])),
        "strided": (not contiguous, strides[-1] == 2 * a.itemsize),
        "reversed": (not contiguous, strides[-1] == -2 * a.itemsize),
    }[layout]
    assert all(expected), (layout, view.strides)


@given(st.data(), complex_arrays(hnp.array_shapes(max_dims=3, max_side=4)))
def test_every_layout_and_strided_copy_is_an_equal_fresh_array(data, values):
    for view in (
        arrange(values, data.draw(layouts())),
        data.draw(strided_copies(values)),
    ):
        assert view.dtype == values.dtype
        assert view.shape == values.shape
        assert np.array_equal(view, values)
        assert not np.shares_memory(view, values)


def test_strided_copies_reach_mixed_signs_and_other_axis_orders():
    a = np.zeros((2, 3, 4))
    # Neither C nor F order: the middle axis varies fastest in memory.
    find(strided_copies(a), lambda v: np.argmin(np.abs(v.strides)) == 1)
    find(strided_copies(a), lambda v: len(set(np.sign(v.strides))) == 2)


class _Context:
    """Fake reusable context for outputs of ``shapes``, with a switchable defect."""

    def __init__(self, shapes, defect=None):
        self.shapes = shapes
        self.defect = defect
        self.consumed = False

    def _validate(self, cotangent, shape):
        if self.defect == "broadcasts":
            try:
                cotangent = np.broadcast_to(cotangent, shape)
            except ValueError:
                raise ValueError("cotangent shape does not match the output") from None
        elif self.defect == "checks only the size":
            if cotangent.size != np.prod(shape):
                raise ValueError("cotangent shape does not match the output")
            cotangent = cotangent.reshape(shape)
        elif cotangent.shape != shape:
            raise ValueError("cotangent shape does not match the output")
        entries = (
            cotangent.flat[:1] if self.defect == "checks the first entry" else cotangent
        )
        if self.defect != "accepts nan" and not np.isfinite(entries).all():
            raise ValueError("cotangent must be finite")
        return cotangent

    def pullback(self, *cotangents):
        if self.defect == "consumes invalid":
            self.consumed = True
        if self.consumed and self.defect in {"consumes invalid", "one use"}:
            raise ValueError("pullback residual has already been consumed")
        unchecked = 1 if self.defect == "checks only the last output" else 0
        cotangents = [
            c if i < unchecked else self._validate(c, shape)
            for i, (c, shape) in enumerate(zip(cotangents, self.shapes, strict=True))
        ]
        self.consumed = True
        scale = 3 if self.defect == "wrong gradient" else 2
        gradients = tuple(scale * c for c in cotangents)
        return gradients if len(gradients) > 1 else gradients[0]


def _outputs(structure):
    """Valid cotangents, output shapes and gradients of the fake context."""
    matrix = np.arange(6).reshape(2, 3) * (1 + 0.5j)
    if structure == "one":
        return matrix, (matrix.shape,), 2 * matrix
    vector = np.array([1, 2j, 3])
    return (vector, matrix), (vector.shape, matrix.shape), (2 * vector, 2 * matrix)


@pytest.mark.parametrize("structure", ["one", "pair"])
def test_reusable_checker_accepts_a_conforming_context(structure):
    cotangent, shapes, expected = _outputs(structure)
    gradients = assert_reusable_context(_Context(shapes), cotangent)
    assert_tree_allclose(gradients, expected, rtol=0)


DEFECTS = (
    "consumes invalid",
    "accepts nan",
    "one use",
    "wrong gradient",
    "broadcasts",
    "checks only the size",
    "checks the first entry",
)


@pytest.mark.parametrize(
    ("structure", "defect"),
    [
        *((structure, defect) for structure in ("one", "pair") for defect in DEFECTS),
        ("pair", "checks only the last output"),
    ],
)
def test_reusable_checker_detects_each_contract_violation(structure, defect):
    cotangent, shapes, expected = _outputs(structure)
    with pytest.raises((AssertionError, ValueError, pytest.fail.Exception)):
        assert_reusable_context(_Context(shapes, defect), cotangent, expected)


def test_site_probe_catches_a_size_only_check_of_a_vector():
    # No wrong shape of the same dimension keeps the size of a 1-d cotangent.
    g = np.array([1, 2j, 3])
    assert_reusable_context(_Context((g.shape,), "checks only the size"), g)
    with pytest.raises(pytest.fail.Exception):
        assert_reusable_context(
            _Context((g.shape,), "checks only the size"),
            g,
            wrong_shape=np.ones((3, 1), complex),
        )


@given(complex_arrays((3, 2), max_magnitude=2.0))
def test_complex_arrays_are_bounded_finite_complex128(values):
    assert values.dtype == np.complex128
    assert values.shape == (3, 2)
    assert np.isfinite(values).all()
    # Hypothesis bounds the magnitude up to a relative rounding of about epsilon.
    assert (np.abs(values) <= 2.0 * (1 + 4 * np.finfo(float).eps)).all()

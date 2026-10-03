"""The accuracy reporter must preserve the underlying numerical assertion."""

from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from _scripts import load

pytestmark = pytest.mark.reference

upstream = load("qualify_upstream")
bench = load("benchmark_cluster")
PARAMETERS = {"workload": "sentinel"}


def gate(actual, expected):
    """The benchmark comparison, resolved at call time as the worker does."""
    return lambda: bench._assert_allclose(actual, expected)


def whole_array_residuals(actual, expected, *, rtol, atol):
    """The pre-streaming implementation, kept as an oracle for the chunked one."""
    actual, expected = np.broadcast_arrays(np.asarray(actual), np.asarray(expected))
    finite = np.isfinite(actual) & np.isfinite(expected)
    a, b = actual[finite], expected[finite]
    if a.dtype == np.bool_ and b.dtype == np.bool_:
        a, b = a.astype(np.int8), b.astype(np.int8)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        differences = np.abs(a - b)
        maximum = float(np.max(differences, initial=0))
        scaled = differences / (atol + rtol * np.abs(b))
        max_scaled = float(np.max(scaled, initial=0))
        scale = max(
            float(np.max(np.abs(a), initial=0)), float(np.max(np.abs(b), initial=0))
        )
        if scale == 0:
            relative = 0.0
        else:
            reference_norm = np.linalg.norm(b / scale)
            relative = float(np.linalg.norm(a / scale - b / scale) / reference_norm)
    coordinates = np.argwhere(finite)
    worst = int(np.argmax(scaled)) if scaled.size else None
    details = {
        "actual_shape": list(actual.shape),
        "element_count": actual.size,
        "finite_count": int(np.count_nonzero(finite)),
        "nonfinite_count": int(np.count_nonzero(~finite)),
        "atol": atol,
        "rtol": rtol,
        "worst_coordinates": coordinates[worst].tolist() if worst is not None else None,
        "worst_actual": [float(a[worst].real), float(a[worst].imag)]
        if worst is not None
        else None,
        "worst_reference": [float(b[worst].real), float(b[worst].imag)]
        if worst is not None
        else None,
    }
    values = {
        "max_abs_error": maximum,
        "relative_l2_error": relative,
        "max_scaled_error": max_scaled,
    }
    return {
        key: value if np.isfinite(value) else None for key, value in values.items()
    }, details


def test_residual_capture_preserves_failures():
    expected = np.array([0, 1, 1j])
    for perturbation, passes in ((0.0, True), (1e-3, False)):
        actual = expected + perturbation
        rows, error = upstream.collect(gate(actual, expected), PARAMETERS)
        assert len(rows) == 3
        assert (error is None) == passes
        assert all(row["status"] == ("passed" if passes else "failed") for row in rows)
        scaled = next(row for row in rows if row["metric"] == "max_scaled_error")
        assert (scaled["error"] <= 1) == passes
        assert scaled["nonfinite_count"] == 0
        assert scaled["element_count"] == 3
    metrics, _ = upstream.residuals(
        np.array([1e300]), np.array([1e300 * (1 + 1e-12)]), rtol=2e-9, atol=1e-12
    )
    assert 5e-13 < metrics["relative_l2_error"] < 2e-12
    rows, failure = upstream.collect(gate([np.nan], [np.nan]), PARAMETERS)
    assert (
        failure is None
    )  # NumPy permits equal NaNs; the report does not qualify them.
    assert all(row["error"] is None and row["status"] == "error" for row in rows)
    assert all(row["nonfinite_count"] == 1 for row in rows)
    rows, failure = upstream.collect(gate(np.ones(2), np.ones(3)), PARAMETERS)
    assert rows == [] and "shape mismatch" in failure["message"]

    for actual, expected, passes in (
        (np.array([False, True]), np.array([False, True]), True),
        (np.array([False, False]), np.array([False, False]), True),
        (np.array([False, True]), np.array([True, True]), False),
    ):
        saved_actual, saved_expected = actual.copy(), expected.copy()
        with patch.object(
            bench, "_assert_allclose", wraps=bench._assert_allclose
        ) as original:
            rows, failure = upstream.collect(gate(actual, expected), PARAMETERS)
        original.assert_called_once()
        assert original.call_args.args[0] is actual
        assert original.call_args.args[1] is expected
        assert actual.dtype == np.bool_ and expected.dtype == np.bool_
        assert len(rows) == 3
        assert (failure is None) == passes
        assert all(row["status"] == ("passed" if passes else "failed") for row in rows)
        metrics = {row["metric"]: row["error"] for row in rows}
        assert metrics["max_abs_error"] == (0 if passes else 1)
        assert (metrics["max_scaled_error"] <= 1) == passes
        assert metrics["relative_l2_error"] == (0 if passes else 1 / np.sqrt(2))
        assert np.array_equal(actual, saved_actual)
        assert np.array_equal(expected, saved_expected)


@pytest.mark.parametrize("perturb", [False, True])
def test_large_output_is_one_observation_per_metric(perturb):
    """Outputs above one gate chunk are summarized as a whole, not per slice."""
    expected = np.arange(131076, dtype=np.float64).reshape(3, -1) * (1 + 0.5j)
    actual = expected.copy()
    if perturb:
        actual[2, -1] += 1
    rows, failure = upstream.collect(gate(actual, expected), PARAMETERS)
    assert (failure is None) != perturb
    assert len(rows) == 3
    assert len({row["id"] for row in rows}) == 3
    for row in rows:
        assert row["element_count"] == 131076
        assert row["actual_shape"] == [3, 43692]
        assert row["status"] == ("failed" if perturb else "passed")
    metrics = {row["metric"]: row["error"] for row in rows}
    if perturb:
        assert rows[0]["worst_coordinates"] == [2, 43691]
        assert metrics["max_abs_error"] == 1
        # The whole-array norm ratio; one slice's ratio differs by 10**2.
        oracle, _ = whole_array_residuals(actual, expected, rtol=2e-9, atol=1e-12)
        np.testing.assert_allclose(
            metrics["relative_l2_error"], oracle["relative_l2_error"], rtol=1e-13
        )
        np.testing.assert_allclose(
            metrics["relative_l2_error"],
            np.linalg.norm(actual - expected) / np.linalg.norm(expected),
            rtol=1e-10,
        )
    else:
        assert metrics == dict.fromkeys(metrics, 0.0)


FINITE = st.integers(-(2**20), 2**20).map(lambda n: n / 2**10)
SPECIAL = st.sampled_from([np.nan, np.inf, -np.inf])


@st.composite
def compared_outputs(draw, *, nonfinite=True):
    """Pairs of outputs with ties, exact matches and gate-scale perturbations."""
    shape = draw(hnp.array_shapes(min_dims=0, max_dims=3, min_side=1, max_side=6))
    kind = draw(st.sampled_from(["complex", "real", "bool"]))
    if kind == "bool":
        return (
            draw(hnp.arrays(np.bool_, shape)),
            draw(hnp.arrays(np.bool_, shape)),
        )
    elements = st.one_of(FINITE, SPECIAL) if nonfinite else FINITE
    expected = draw(hnp.arrays(np.float64, shape, elements=elements))
    factors = draw(
        hnp.arrays(
            np.float64,
            shape,
            elements=st.sampled_from([0.0, 1e-10, 1.9e-9, 2.1e-9, 1e-8, 0.5]),
        )
    )
    actual = expected * (1 + factors) + draw(
        hnp.arrays(np.float64, shape, elements=st.sampled_from([0.0, 5e-13, 2e-12]))
    )
    if kind == "complex":
        expected = expected + 1j * draw(hnp.arrays(np.float64, shape, elements=FINITE))
        actual = actual + 1j * expected.imag
    if nonfinite and draw(st.booleans()):
        actual.flat[draw(st.integers(0, actual.size - 1))] = draw(SPECIAL)
    return actual, expected


@settings(max_examples=200)
@given(outputs=compared_outputs(), chunk=st.integers(1, 9))
def test_streamed_residuals_equal_whole_array_summaries(outputs, chunk):
    actual, expected = outputs
    metrics, details = upstream.residuals(
        actual, expected, rtol=2e-9, atol=1e-12, chunk=chunk
    )
    oracle, oracle_details = whole_array_residuals(
        actual, expected, rtol=2e-9, atol=1e-12
    )
    assert details == oracle_details
    assert metrics["max_abs_error"] == oracle["max_abs_error"]
    assert metrics["max_scaled_error"] == oracle["max_scaled_error"]
    if oracle["relative_l2_error"] is None:
        assert metrics["relative_l2_error"] is None
    else:
        # Only the summation order differs: at most 216 nonnegative terms.
        np.testing.assert_allclose(
            metrics["relative_l2_error"], oracle["relative_l2_error"], rtol=1e-13
        )


@settings(max_examples=100)
@given(outputs=compared_outputs(nonfinite=False), power=st.integers(-60, 60))
def test_residual_gate_matches_allclose_and_relative_error_is_scale_free(
    outputs, power
):
    actual, expected = outputs
    ratio = np.abs(actual.astype(complex) - expected) / (
        1e-12 + 2e-9 * np.abs(expected)
    )
    assume(not np.any(np.abs(ratio - 1) < 1e-9))
    metrics, _ = upstream.residuals(actual, expected, rtol=2e-9, atol=1e-12, chunk=5)
    passed = bool(np.allclose(actual, expected, rtol=2e-9, atol=1e-12))
    assert (metrics["max_scaled_error"] <= 1) == passed
    same, _ = upstream.residuals(expected, expected, rtol=2e-9, atol=1e-12)
    assert same == dict.fromkeys(same, 0.0)
    if actual.dtype != np.bool_:
        scaled, _ = upstream.residuals(
            actual * 2.0**power, expected * 2.0**power, rtol=2e-9, atol=1e-12
        )
        if metrics["relative_l2_error"] is None:  # zero reference, nonzero output
            assert scaled["relative_l2_error"] is None
        else:
            np.testing.assert_allclose(
                scaled["relative_l2_error"], metrics["relative_l2_error"], rtol=1e-15
            )

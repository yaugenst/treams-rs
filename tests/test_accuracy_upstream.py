"""The accuracy reporter must preserve the underlying numerical assertion."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np


def test_residual_capture_preserves_failures(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "qualify_upstream", directory / "qualify_upstream.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parameters = {"workload": "sentinel"}
    expected = np.array([0, 1, 1j])
    for perturbation, passes in ((0.0, True), (1e-3, False)):
        actual = expected + perturbation
        rows, error = module.collect(
            lambda actual=actual: np.testing.assert_allclose(
                actual, expected, rtol=2e-9, atol=1e-12
            ),
            parameters,
        )
        assert len(rows) == 3
        assert (error is None) == passes
        assert all(row["status"] == ("passed" if passes else "failed") for row in rows)
        scaled = next(row for row in rows if row["metric"] == "max_scaled_error")
        assert (scaled["error"] <= 1) == passes
        assert scaled["nonfinite_count"] == 0
        assert scaled["element_count"] == 3
    metrics, _ = module.residuals(
        np.array([1e300]), np.array([1e300 * (1 + 1e-12)]), rtol=2e-9, atol=1e-12
    )
    assert 5e-13 < metrics["relative_l2_error"] < 2e-12
    rows, failure = module.collect(
        lambda: np.testing.assert_allclose([np.nan], [np.nan], rtol=2e-9, atol=1e-12),
        parameters,
    )
    assert (
        failure is None
    )  # NumPy permits equal NaNs; the report does not qualify them.
    assert all(row["error"] is None and row["status"] == "error" for row in rows)
    assert all(row["nonfinite_count"] == 1 for row in rows)

    for actual, expected, passes in (
        (np.array([False, True]), np.array([False, True]), True),
        (np.array([False, False]), np.array([False, False]), True),
        (np.array([False, True]), np.array([True, True]), False),
    ):
        saved_actual, saved_expected = actual.copy(), expected.copy()
        with patch.object(
            np.testing, "assert_allclose", wraps=np.testing.assert_allclose
        ) as original:
            rows, failure = module.collect(
                lambda actual=actual, expected=expected: np.testing.assert_allclose(
                    actual, expected, rtol=2e-9, atol=1e-12
                ),
                parameters,
            )
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

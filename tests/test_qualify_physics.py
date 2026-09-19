"""Qualification must preserve failures and exact residuals in portable JSON."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

SCRIPT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/qualify_physics.py")
)


def test_collector_preserves_failures_nonfinite_and_exceptions():
    collector = SCRIPT["Collector"]()
    with collector.case("residuals", "test", {"order": 2}):
        collector.add("exact_zero", 0, 1e-12)
        collector.add("failed", 1e-3, 1e-12)
        collector.add("nonfinite", np.nan, 1e-12)
        collector.add("convergence", 0.1, None)
    with collector.case("broken", "test", {}):
        raise ValueError("deliberate evaluation failure")
    rows = json.loads(json.dumps(collector.observations, allow_nan=False))
    assert [row["status"] for row in rows] == [
        "passed",
        "failed",
        "error",
        "diagnostic",
        "error",
    ]
    assert rows[0]["error"] == 0
    assert rows[1]["error"] == 1e-3
    assert rows[2]["error"] is None
    assert "deliberate evaluation failure" in rows[-1]["message"]


@given(
    error=st.floats(min_value=0, max_value=1, allow_nan=False),
    tolerance=st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_collector_acceptance_matches_declared_tolerance(error, tolerance):
    collector = SCRIPT["Collector"]()
    with collector.case("threshold", "test", {}):
        collector.add("absolute", error, tolerance)
    row = collector.observations[0]
    assert row["error"] == error
    assert (row["status"] == "passed") == (error <= tolerance)


@pytest.mark.physics
def test_fresh_coefficient_qualification_has_physics_and_provenance():
    report = SCRIPT["qualify"](1234, ("coefficients",))
    assert report["complete"] and report["passed"]
    assert report["source_unchanged"]
    assert all(
        len(report["source"][key]) == 64
        for key in ("native_sha256", "python_source_sha256", "script_sha256")
    )
    assert len(report["observations"]) == 24 * 4 + 18 * 3
    assert {row["family"] for row in report["observations"]} == {
        "sphere coefficients",
        "cylinder coefficients",
    }
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_nested_details_cannot_pass_or_break_json(nonfinite):
    collector = SCRIPT["Collector"]()
    with collector.case("passivity", "test", {}):
        # A one-sided max can hide NaN in the residual; observables must expose it.
        collector.add(
            "violation",
            max(0, nonfinite),
            1e-12,
            observables={"power": nonfinite},
            conditioning={"samples": [1.0, nonfinite]},
        )
    row = json.loads(json.dumps(collector.observations, allow_nan=False))[0]
    assert row["status"] == "error"
    assert row["error"] is None
    assert row["unqualified_residual"] == (None if nonfinite == float("inf") else 0)
    assert row["observables"]["power"] is None
    assert row["conditioning"]["samples"] == [1.0, None]
    assert {entry["path"] for entry in row["nonfinite_details"]} == {
        "/observables/power",
        "/conditioning/samples/1",
    }
    assert {entry["value"] for entry in row["nonfinite_details"]} == {str(nonfinite)}

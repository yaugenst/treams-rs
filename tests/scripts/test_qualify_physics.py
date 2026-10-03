"""Qualification must preserve failures and exact residuals in portable JSON."""

import json
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from _scripts import load

physics = load("qualify_physics")


@pytest.mark.interface
def test_collector_preserves_failures_nonfinite_and_exceptions():
    collector = physics.Collector()
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


@pytest.mark.interface
@given(
    error=st.floats(min_value=0, max_value=1, allow_nan=False),
    tolerance=st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_collector_acceptance_matches_declared_tolerance(error, tolerance):
    collector = physics.Collector()
    with collector.case("threshold", "test", {}):
        collector.add("absolute", error, tolerance)
    row = collector.observations[0]
    assert row["error"] == error
    assert (row["status"] == "passed") == (error <= tolerance)


@pytest.mark.interface
def test_every_declared_family_has_its_collector_in_seed_order():
    # FAMILIES order fixes each family's independent random stream.
    assert tuple(physics.FAMILY_FUNCTIONS) == physics.FAMILIES
    assert all(f.__name__ == name for name, f in physics.FAMILY_FUNCTIONS.items())


@pytest.mark.physics
def test_fresh_coefficient_qualification_has_physics_and_provenance():
    report = physics.qualify(1234, ("coefficients",))
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


@pytest.mark.interface
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_nested_details_cannot_pass_or_break_json(nonfinite):
    collector = physics.Collector()
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


LEAVES = st.one_of(
    st.floats(allow_nan=False, allow_infinity=False),
    st.sampled_from([float("nan"), float("inf"), -float("inf")]),
    st.integers(-5, 5),
    st.booleans(),
    st.none(),
    st.text("ab", max_size=2),
)
TREES = st.recursive(
    LEAVES,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(st.text("xyz", min_size=1, max_size=2), children, max_size=3),
    ),
    max_leaves=12,
)


def nonfinite_paths(node, path):
    if isinstance(node, float) and not math.isfinite(node):
        yield path
    elif isinstance(node, dict):
        for key, child in node.items():
            yield from nonfinite_paths(child, f"{path}/{key}")
    elif isinstance(node, list):
        for index, child in enumerate(node):
            yield from nonfinite_paths(child, f"{path}/{index}")


def finite_part(node):
    if isinstance(node, float) and not math.isfinite(node):
        return None
    if isinstance(node, dict):
        return {key: finite_part(child) for key, child in node.items()}
    if isinstance(node, list):
        return [finite_part(child) for child in node]
    return node


@given(
    tree=TREES,
    error=st.floats(0, 1),
    tolerance=st.one_of(st.none(), st.floats(0, 1)),
)
@pytest.mark.interface
def test_nonfinite_details_anywhere_are_reported_by_path(tree, error, tolerance):
    """Arbitrary nested details keep strict JSON; any NaN/inf leaf is an error."""
    collector = physics.Collector()
    with collector.case("tree", "test", {}):
        collector.add("residual", error, tolerance, observables=tree)
    (row,) = json.loads(json.dumps(collector.observations, allow_nan=False))
    injected = list(nonfinite_paths(tree, "/observables"))
    assert row["observables"] == finite_part(tree)
    if injected:
        assert row["status"] == "error" and row["error"] is None
        assert row["unqualified_residual"] == error
        assert [entry["path"] for entry in row["nonfinite_details"]] == injected
    else:
        assert row["error"] == error and "nonfinite_details" not in row
        assert row["status"] == (
            "diagnostic"
            if tolerance is None
            else "passed"
            if error <= tolerance
            else "failed"
        )

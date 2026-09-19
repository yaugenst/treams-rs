"""The diagnostic distinguishes equation residuals from entrywise agreement."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

SCRIPT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/qualify_cluster_conditioning.py")
)


@given(
    scale=st.floats(
        min_value=1e-8, max_value=1e8, allow_nan=False, allow_infinity=False
    )
)
def test_backward_errors_are_scale_invariant_and_detect_perturbations(scale):
    system = np.array([[2, 0.3j], [-0.1j, 3]], dtype=complex)
    solution = np.array([[1 + 0.2j, 0.4], [0.1, -0.5j]])
    rhs = system @ solution
    exact = SCRIPT["backward_errors"](system, solution, rhs)
    assert max(exact.values()) < 1e-15
    perturbed = solution.copy()
    perturbed[0, 0] += 0.01
    original = SCRIPT["backward_errors"](system, perturbed, rhs)
    scaled = SCRIPT["backward_errors"](scale * system, perturbed, scale * rhs)
    for metric in original:
        assert original[metric] > 1e-4
        np.testing.assert_allclose(
            scaled[metric], original[metric], rtol=1e-11, atol=1e-14
        )


def test_lapack_condition_estimate_matches_diagonal_system():
    diagonal = np.diag(np.array([1, 1e-3, 2e-7], dtype=complex))
    result = SCRIPT["condition_estimate"](diagonal)
    np.testing.assert_allclose(
        result["system_reciprocal_condition_1_estimate"], 2e-7, rtol=1e-14
    )
    np.testing.assert_allclose(result["system_condition_1_estimate"], 5e6, rtol=1e-14)


@pytest.mark.physics
def test_small_chain_retains_both_backends_and_cutoff_metadata(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    result = SCRIPT["qualify"](2, (1, 2))
    assert result["complete"] and result["passed"]
    assert result["source_unchanged"]
    rows = result["observations"]
    assert len({row["id"] for row in rows}) == len(rows)
    residuals = [
        row
        for row in rows
        if row["metric"] == "illuminated_componentwise_backward_error"
    ]
    assert {row["backend"] for row in residuals} == {"treams", "treams-rs"}
    assert all(row["error"] < 1e-12 for row in residuals)
    cutoff = [row for row in rows if row["reference_kind"] == "self_convergence"]
    assert cutoff and all(row["parameters"]["reference_lmax"] == 2 for row in cutoff)
    assert all(row["parameters"]["lmax"] < 2 for row in cutoff)
    json.dumps(result, allow_nan=False)

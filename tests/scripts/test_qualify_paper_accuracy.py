"""Saved-spectrum normalization must retain provenance, failures and caveats."""

import json

import pytest

from _scripts import ROOT, load

pytestmark = pytest.mark.reference


def test_saved_paper_accuracy():
    module = load("qualify_paper_accuracy")
    raw = json.loads((ROOT / "benchmarks/papers/qualification.json").read_text())
    curves = module.normalize_curves(raw, module.references())
    assert len(curves) == 8
    assert sum(len(curve["grid"]) for curve in curves) == 853
    rows = module.observe(curves)
    assert all(row["status"] in ("passed", "diagnostic") for row in rows)
    corrected = next(
        curve for curve in curves if curve["id"] == "ebeam-cylinder-author-notebook"
    )
    assert corrected["references"][0]["reference_kind"] == "corrected_author_notebook"
    assert corrected["source_correction"]
    array = next(
        curve for curve in curves if curve["id"] == "cpc-sphere_array_above_slab"
    )
    assert array["excluded_author_samples"][0]["index"] == 99
    assert any(
        row["reference_kind"] == "truncation" and row["tolerance"] is None
        for row in rows
    )
    curves[0]["native"][0][0] += 1e-3
    assert any(row["status"] == "failed" for row in module.observe(curves))
    curves[0]["native"][0][0] = None
    assert any(row["status"] == "error" for row in module.observe(curves))

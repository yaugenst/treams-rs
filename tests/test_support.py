"""Discoverability claims checked against the installed public/native boundary."""

import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import treams_rs as tr
from treams_rs.support import _markdown


def test_catalog_resolves_and_documents_native_records():
    catalog = tr.support_catalog()
    assert catalog["schema_version"] == 1
    assert json.loads(json.dumps(catalog)) == catalog
    paths = [row["path"] for row in catalog["api"]]
    assert len(paths) == len(set(paths))
    for row in catalog["api"]:
        parts = row["path"].split(".")
        if len(parts) == 2:
            value = getattr(tr, parts[1])
        else:
            if parts[1] in {"advect", "jax", "torch", "io"}:
                continue  # Optional APIs are source-derived, not imported by discovery.
            value = getattr(importlib.import_module(".".join(parts[:-1])), parts[-1])
        assert callable(value) or row["kind"] == "attribute"
        if row["kind"] == "class":
            for member in row["members"]:
                name = member["path"].split(".")[-1]
                if member["kind"] == "attribute":
                    assert any(
                        name in getattr(base, "__annotations__", {})
                        or name in vars(base)
                        for base in value.__mro__
                    )
                else:
                    assert inspect.getattr_static(value, name) is not None
    solve = next(row for row in catalog["api"] if row["path"] == "treams_rs.diff.solve")
    assert solve["pullbacks"][0]["context"] == "SolveContext"
    assert "A_bar, B_bar" in solve["doc"]
    tmatrix = next(row for row in catalog["api"] if row["path"] == "treams_rs.TMatrix")
    assert {member["path"] for member in tmatrix["members"]} >= {
        "treams_rs.TMatrix.cluster",
        "treams_rs.TMatrix.efield",
        "treams_rs.TMatrix.interaction",
        "treams_rs.TMatrix.sphere",
    }
    solution = next(
        row for row in catalog["api"] if row["path"] == "treams_rs.iterative.Solution"
    )
    assert {row["path"] for row in solution["members"]} >= {
        "treams_rs.iterative.Solution.coefficients",
        "treams_rs.iterative.Solution.convergence",
    }
    area = next(
        row for row in catalog["api"] if row["path"] == "treams_rs.lattice.area"
    )
    assert area["alias_of"] == "treams_rs.lattice.volume"
    interaction = next(
        row for row in catalog["returned_python_types"] if row["path"] == "_Interaction"
    )
    assert "_Interaction.illuminate" in {row["path"] for row in interaction["members"]}
    assert "not a Python backend" in catalog["backends"]["wasm"]["distribution"]


def test_offline_discovery_does_not_load_optional_frameworks():
    code = """
import json, sys, pydoc
import treams_rs as tr
before = set(sys.modules)
catalog = tr.support_catalog()
assert "support_catalog" in pydoc.render_doc(tr)
for dependency in ('advect', 'jax', 'torch', 'h5py'):
    assert not any(name == dependency or name.startswith(dependency + '.') for name in set(sys.modules) - before)
assert any(row['path'] == 'treams_rs.jax.wrap' for row in catalog['api'])
print(json.dumps(catalog['backends']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    result = subprocess.run(
        [sys.executable, "-m", "treams_rs"], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout)["schema_version"] == 1
    result = subprocess.run(
        [sys.executable, "-m", "treams_rs", "--format", "markdown"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "treams_rs.diff.solve" in result.stdout
    assert "**python_execution**: CPU" in result.stdout


@given(
    st.floats(min_value=0.05, max_value=0.8, allow_nan=False),
    st.floats(min_value=0.5, max_value=4, allow_nan=False),
)
def test_documented_sphere_and_recorded_solve(radius, k0):
    sphere = tr.TMatrix.sphere(2, k0, radius, [2.0, 1.0])
    np.testing.assert_allclose(
        sphere.xs_ext_avg, sphere.xs_sca_avg, rtol=1e-11, atol=1e-14
    )
    matrix = np.eye(2, dtype=complex) * (2 + radius)
    rhs = np.array([[1 + 0.3j], [k0]])
    solution, context = tr.diff.solve(matrix, rhs)
    np.testing.assert_allclose(matrix @ solution, rhs, rtol=1e-14)
    weights = np.array([[0.2j], [0.7]])
    matrix_bar, rhs_bar = context.pullback(weights)
    np.testing.assert_allclose(matrix.conj().T @ rhs_bar, weights, rtol=1e-14)
    np.testing.assert_allclose(matrix_bar, -rhs_bar @ solution.conj().T, rtol=1e-14)


def test_generated_reference_is_source_current():
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/api.md").read_text() == _markdown(tr.support_catalog())

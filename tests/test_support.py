"""Discoverability claims checked against the installed public/native boundary."""

import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
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
        "treams_rs.TMatrix.scatter",
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
    assert set(catalog["backends"]) == {"cpu"}
    assert not any(row["path"].startswith("treams_rs.cuda") for row in catalog["api"])


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
        [sys.executable, "-m", "treams_rs", "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
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


def test_framework_physics_contracts_are_discovered_without_importing_backends():
    catalog = tr.support_catalog()
    entries = {row["path"]: row for row in catalog["api"]}
    for engine in ("advect", "jax", "torch"):
        prefix = f"treams_rs.{engine}"
        assert f"{prefix}.sphere_tmatrix" in entries
        wave = entries[f"{prefix}.Wave"]
        assert {
            f"{prefix}.Wave.{name}"
            for name in ("efield", "in_basis", "with_polarization")
        } <= {member["path"] for member in wave["members"]}
    assert "treams_rs.operators.efield" in entries
    assert "treams_rs.efield" not in entries
    assert "treams_rs.SMatrix" in entries
    assert "treams_rs.SMatrices" not in entries


def test_generated_reference_is_source_current():
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/api.md").read_text() == _markdown(tr.support_catalog())


def test_focused_installed_help_and_executable_quickstarts():
    import re
    import textwrap

    for topic, expected in (
        (None, "A complete particle calculation"),
        ("sphere_tmatrix", "response.cross_sections"),
        ("advect", "ad.value_and_grad"),
        ("TMatrix.cross_sections", "Named scattering"),
    ):
        command = [sys.executable, "-m", "treams_rs"]
        result = subprocess.run(
            command + ([topic] if topic else []),
            check=True,
            capture_output=True,
            text=True,
        )
        assert expected in result.stdout
        assert len(result.stdout) < 20000
    result = subprocess.run(
        [sys.executable, "-m", "treams_rs", "--search", "cross_sections"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "treams_rs.TMatrix.cross_sections" in result.stdout
    for module in (tr, tr.advect):
        namespace = {}
        for block in re.findall(r"::\n\n((?:    [^\n]*\n|\n)+)", module.__doc__):
            exec(textwrap.dedent(block), namespace)
        assert np.isfinite(namespace["derivative" if module is tr else "gradient"])


def test_complete_workflow_examples_and_search_alternatives():
    import re
    import textwrap

    for value in (tr.Cluster, tr.solve_periodic, tr.slab, tr.advect.field):
        for block in re.findall(r"::\n\n((?:    [^\n]*\n|\n)+)", inspect.getdoc(value)):
            exec(textwrap.dedent(block), {})
    result = subprocess.run(
        [sys.executable, "-m", "treams_rs", "--search", r"solve_periodic\|slab"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "treams_rs.solve_periodic" in result.stdout
    assert "treams_rs.slab" in result.stdout


def test_renamed_basis_errors_point_to_the_public_name():
    with pytest.raises(AttributeError, match=r"exported as treams_rs\.PlaneWavePorts"):
        _ = tr.PlaneWaveBasisByComp
    entries = {row["path"] for row in tr.support_catalog()["api"]}
    for engine in ("advect", "jax", "torch"):
        assert f"treams_rs.{engine}.PlaneWavePorts" in entries
        assert f"treams_rs.{engine}.Lattice" in entries

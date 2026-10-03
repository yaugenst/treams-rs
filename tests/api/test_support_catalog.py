"""Discoverability claims checked against the installed public and native modules."""

import ast
import collections
import importlib
import inspect
import json
import pkgutil
import re
import runpy
import subprocess
import sys
from pathlib import Path
from typing import TypeAliasType

import numpy as np
import pytest

import treams_rs as tr
from treams_rs import _catalog
from treams_rs._catalog import render_markdown

from _support import ROOT
from _support import catalog as installed_catalog

pytestmark = pytest.mark.interface

OPTIONAL = tr._OPTIONAL_MODULES


@pytest.fixture(scope="module")
def catalog():
    return installed_catalog()


def _entries(rows):
    """Catalog rows followed by their members, depth first."""
    return [entry for row in rows for entry in (row, *_entries(row.get("members", [])))]


def _namespace(path):
    parts = path.split(".")
    return parts[1] if len(parts) > 2 and parts[1] in OPTIONAL else "core"


def _assert_resolves(row):
    module, name = row["path"].rsplit(".", 1)
    value = getattr(importlib.import_module(module), name)
    assert callable(value) or row["kind"] in {"attribute", "type_alias"}, row["path"]
    if row["kind"] == "type_alias":
        assert isinstance(value, TypeAliasType), row["path"]
        assert value.__name__ == name
        assert row["signature"].startswith(" = ")
    if row["kind"] == "ufunc":
        # The signature field carries the call form once, under the public name.
        assert not row["doc"].startswith(value.__name__ + "("), row["path"]
    if row["kind"] != "class":
        return
    for member in row["members"]:
        name = member["path"].rsplit(".", 1)[1]
        if member["kind"] == "attribute":
            assert any(
                name in getattr(base, "__annotations__", {}) or name in vars(base)
                for base in value.__mro__
            ), member["path"]
        else:
            assert inspect.getattr_static(value, name) is not None, member["path"]


@pytest.mark.parametrize("namespace", ["core", *sorted(OPTIONAL)])
def test_catalog_entries_resolve_at_runtime(catalog, namespace):
    # Discovery reads source only; the optional namespaces must still import
    # and define every name it advertises when their dependency is installed.
    if namespace != "core":
        pytest.importorskip(OPTIONAL[namespace])
    rows = [row for row in catalog["api"] if _namespace(row["path"]) == namespace]
    assert rows
    for row in rows:
        _assert_resolves(row)


def test_catalog_schema_and_documented_contracts(catalog):
    assert catalog["schema_version"] == 1
    assert json.loads(json.dumps(catalog)) == catalog
    paths = [row["path"] for row in catalog["api"]]
    assert len(paths) == len(set(paths))
    entries = {row["path"]: row for row in catalog["api"]}
    solve = entries["treams_rs.diff.solve"]
    assert solve["pullbacks"][0]["context"] == "SolveContext"
    assert "Dynamic inputs:\n    operator:" in solve["doc"]
    members = {
        path: {member["path"] for member in entries[path]["members"]}
        for path in ("treams_rs.TMatrix", "treams_rs.iterative.Solution")
    }
    assert members["treams_rs.TMatrix"] >= {
        "treams_rs.TMatrix.scatter",
        "treams_rs.TMatrix.interaction",
        "treams_rs.TMatrix.sphere",
    }
    assert members["treams_rs.iterative.Solution"] >= {
        "treams_rs.iterative.Solution.coefficients",
        "treams_rs.iterative.Solution.convergence",
    }
    assert entries["treams_rs.lattice.area"]["alias_of"] == "treams_rs.lattice.volume"
    # A native ufunc alias without an attribute docstring shows the ufunc doc.
    assert entries["treams_rs.lattice.realsumcw2d"]["doc"].startswith(
        "The real-space part of the Ewald sum"
    )
    # Public PEP 695 aliases named in signatures have their own rows.
    for alias in ("ebcm.Modes", "io.MatrixSet", "lattice.SumResult"):
        assert entries[f"treams_rs.{alias}"]["kind"] == "type_alias"
    interaction = next(
        row for row in catalog["returned_python_types"] if row["path"] == "_Interaction"
    )
    assert "_Interaction.illuminate" in {row["path"] for row in interaction["members"]}
    # A Python class with a pullback method is a context like the native ones.
    solver = {
        member["path"]: member
        for member in entries["treams_rs.iterative.SphereCluster"]["members"]
    }
    assert [
        (pullback["context"], pullback["method"])
        for pullback in solver["treams_rs.iterative.SphereCluster.record"]["pullbacks"]
    ] == [("IterativeContext", "pullback")]
    # Adapter summaries are the opening paragraphs of the adapter docstrings.
    for name, adapter in catalog["adapters"].items():
        doc = catalog["modules"][f"treams_rs.{name}"]
        assert adapter["summary"] == " ".join(doc.partition("\n\n")[0].split())
        assert adapter["summary"]
        assert adapter["extra"] == name
    assert set(catalog["adapters"]) == {"advect", "jax", "torch"}
    assert set(catalog["optional_dependencies"]) == set(OPTIONAL.values())
    assert set(catalog["backends"]) == {"cpu"}
    assert not any(path.startswith("treams_rs.cuda") for path in paths)


def test_catalog_lists_every_public_definition(catalog):
    # Source discovery must not miss a definition that the modules expose, for
    # example one built by a factory rather than a def.
    paths = {row["path"] for row in catalog["api"]}
    names = [
        info.name
        for info in pkgutil.iter_modules(tr.__path__)
        if not info.name.startswith("_") and info.name not in OPTIONAL
    ]
    missing = set()
    for name in ["treams_rs", *(f"treams_rs.{name}" for name in names)]:
        module = importlib.import_module(name)
        if name != "treams_rs":
            assert name in catalog["modules"]
        exported = set(getattr(module, "__all__", ()))
        for attribute, value in vars(module).items():
            if attribute.startswith("_") or inspect.ismodule(value):
                continue
            defined = getattr(value, "__module__", None) in (name, "treams_rs._native")
            public = attribute in exported or defined or isinstance(value, np.ufunc)
            if public and f"{name}.{attribute}" not in paths:
                missing.add(f"{name}.{attribute}")
    assert not missing


def test_framework_namespaces_share_one_vocabulary(catalog):
    names = {
        engine: {
            row["path"].split(".", 2)[2]
            for row in catalog["api"]
            if row["path"].startswith(f"treams_rs.{engine}.")
        }
        for engine in ("advect", "jax", "torch")
    }
    assert names["jax"] == names["torch"]
    # Advect composes Python functions directly and needs no wrap.
    assert names["jax"] - {"wrap"} <= names["advect"]
    entries = {row["path"]: row for row in catalog["api"]}
    for engine in names:
        prefix = f"treams_rs.{engine}"
        # Shared constructors are bound methods; every export must stay documented.
        tree = ast.parse(
            (Path(tr.__file__).parent / f"{engine}.py").read_text(encoding="utf-8")
        )
        exports = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "__all__" for t in node.targets)
        )
        assert {f"{prefix}.{name}" for name in exports} <= set(entries)
        assert entries[f"{prefix}.slab"]["signature"].startswith("(*, basis:")
        assert {f"{prefix}.{name}" for name in ("PlaneWavePorts", "Lattice")} <= set(
            entries
        )
        assert {
            f"{prefix}.Wave.{name}"
            for name in ("efield", "in_basis", "with_polarization")
        } <= {member["path"] for member in entries[f"{prefix}.Wave"]["members"]}
        # Framework classes are listed without __init__, so the internal
        # Backend stays out of the reference.
        rows = _entries(
            [row for row in catalog["api"] if row["path"].startswith(prefix)]
        )
        assert f"{prefix}.TMatrix.__init__" not in {row["path"] for row in rows}
        assert not [
            row["path"] for row in rows if "Backend" in row.get("signature", "")
        ]


def _assert_alias_resolves(cls, row, members):
    """An alias row repeats its member's signature and doc and is that member."""
    original = members[row["alias_of"]]
    assert original["kind"] != "alias"
    assert (row["signature"], row["doc"]) == (original["signature"], original["doc"])
    name, target = (path.rsplit(".", 1)[1] for path in (row["path"], original["path"]))
    assert inspect.getattr_static(cls, name) is inspect.getattr_static(cls, target)


def test_class_aliases_resolve_to_their_members(catalog):
    for row in catalog["api"]:
        members = {member["path"]: member for member in row.get("members", [])}
        aliases = [member for member in members.values() if member["kind"] == "alias"]
        if aliases:
            module, name = row["path"].rsplit(".", 1)
            cls = getattr(importlib.import_module(module), name)
            for alias in aliases:
                _assert_alias_resolves(cls, alias, members)
    # The same rule on a class whose aliases chain.
    source = '''
class Wave:
    def in_basis(self, basis: str) -> "Wave":
        """Expand in another basis."""
        return self

    expand = in_basis
    expand_again = expand
'''
    tree = ast.parse(source)
    entry = _catalog._class_entry("example.Wave", tree.body[0], "example.py", {}, {})
    members = {member["path"]: member for member in entry["members"]}
    assert [(member["path"], member["kind"]) for member in members.values()] == [
        ("example.Wave.in_basis", "function"),
        ("example.Wave.expand", "alias"),
        ("example.Wave.expand_again", "alias"),
    ]
    namespace = {}
    exec(compile(tree, "example.py", "exec"), namespace)
    for member in members.values():
        if member["kind"] == "alias":
            assert member["alias_of"] == "example.Wave.in_basis"
            _assert_alias_resolves(namespace["Wave"], member, members)


def test_public_names_and_renamed_classes(catalog):
    entries = {row["path"]: row for row in catalog["api"]}
    assert "treams_rs.operators.efield" in entries
    assert "treams_rs.efield" not in entries
    assert "treams_rs.SMatrix" in entries
    assert "treams_rs.SMatrices" not in entries
    network = {
        member["path"]: member for member in entries["treams_rs.SMatrix"]["members"]
    }
    assert network["treams_rs.SMatrix.block"]["signature"].endswith(
        "-> ScatteringBlock"
    )
    assert network["treams_rs.SMatrix.cascade"]["signature"].endswith("-> SMatrix")
    with pytest.raises(
        AttributeError,
        match=r"treams\.PlaneWaveBasisByComp corresponds to treams_rs\.PlaneWavePorts",
    ):
        _ = tr.PlaneWaveBasisByComp


def test_generated_documentation_is_source_current(monkeypatch):
    # The same checks as `just docs-check`: every docs page is in the nav with a
    # description, and every generated file (reference pages, generated
    # regions, llms.txt) matches its source.
    monkeypatch.setattr(_catalog, "support_catalog", installed_catalog)
    generator = runpy.run_path(str(ROOT / "scripts/generate_docs.py"))
    assert generator["index_errors"]() == []
    files = generator["generated_files"]()
    for path, content in files.items():
        assert path.read_text(encoding="utf-8") == content, (
            f"{path.name} is stale; run just docs"
        )
    assert generator["orphan_pages"](files) == []


def test_agent_index_uses_new_reference_pages_before_they_are_written(
    tmp_path, monkeypatch
):
    generator = runpy.run_path(str(ROOT / "scripts/generate_docs.py"))
    generate = generator["generated_files"]
    docs = tmp_path / "docs"
    docs.mkdir()
    page = docs / "reference/python/new.md"
    content = "---\ndescription: A new module.\n---\n# New module\n"
    (tmp_path / "mkdocs.yml").write_text(
        "site_url: https://example.test/\nnav:\n  - reference/python/new.md\n",
        encoding="utf-8",
    )
    for name, value in {
        "ROOT": tmp_path,
        "DOCS": docs,
        "support_catalog": lambda: {"version": "0.1.0"},
        "reference_pages": lambda catalog: {page: content},
    }.items():
        monkeypatch.setitem(generate.__globals__, name, value)
    files = generate()
    assert (
        "[New module](docs/reference/python/new.md): A new module."
        in files[tmp_path / "llms.txt"]
    )
    assert not page.exists()


def test_offline_discovery_does_not_load_optional_frameworks():
    code = """
import json, sys, pydoc
import treams_rs as tr
before = set(sys.modules)
catalog = tr.support_catalog()
assert "support_catalog" in pydoc.render_doc(tr)
for dependency in tr._OPTIONAL_MODULES.values():
    loaded = set(sys.modules) - before
    assert not any(n == dependency or n.startswith(dependency + '.') for n in loaded)
assert any(row['path'] == 'treams_rs.jax.wrap' for row in catalog['api'])
"""
    subprocess.run([sys.executable, "-c", code], check=True)
    # The installed entry point itself.
    result = subprocess.run(
        [sys.executable, "-m", "treams_rs", "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["schema_version"] == 1


def run_cli(monkeypatch, capsys, *arguments):
    """Run ``python -m treams_rs`` in-process against the cached catalog."""
    monkeypatch.setattr(sys, "argv", ["treams_rs", *arguments])
    monkeypatch.setattr(_catalog, "support_catalog", installed_catalog)
    runpy.run_module("treams_rs", run_name="__main__")
    return capsys.readouterr().out


@pytest.mark.workflows
@pytest.mark.parametrize(
    "arguments,expected",
    [
        ((), ["A complete particle calculation"]),
        (("sphere_tmatrix",), ["response.cross_sections"]),
        (("advect",), ["ad.value_and_grad"]),
        (("TMatrix.cross_sections",), ["Named scattering"]),
        (("--search", "cross_sections"), ["treams_rs.TMatrix.cross_sections"]),
        (
            ("--search", "solve_periodic|slab"),
            ["treams_rs.solve_periodic", "treams_rs.slab"],
        ),
    ],
)
def test_installed_help(monkeypatch, capsys, arguments, expected):
    output = run_cli(monkeypatch, capsys, *arguments)
    for text in expected:
        assert text in output
    assert len(output) < 20000


def test_installed_help_exports_the_reference(monkeypatch, capsys, catalog):
    output = run_cli(monkeypatch, capsys, "--format", "markdown")
    assert output == render_markdown(catalog) + "\n"
    assert "**python_execution**: CPU" in output


def test_module_page_renders_exactly_that_module(catalog):
    page = render_markdown(catalog, module="treams_rs.sw")
    rows = [row for row in catalog["api"] if row["path"].startswith("treams_rs.sw.")]
    assert rows
    module_doc = _catalog.docstring_markdown(catalog["modules"]["treams_rs.sw"])
    assert page.startswith(module_doc + "\n")
    # Every heading is a treams_rs.sw row, named relative to the module, and
    # every treams_rs.sw row has one.
    headings = re.findall(r"^#+ `(\S+)`$", page, flags=re.M)
    assert sorted(headings) == sorted(
        row["path"].removeprefix("treams_rs.sw.") for row in _entries(rows)
    )
    assert "```python\ntranslate(" in page
    assert "```text" not in page
    with pytest.raises(ValueError, match=r"unknown module 'treams_rs\.nope'"):
        render_markdown(catalog, module="treams_rs.nope")


def test_docstring_sections_become_markdown():
    doc = """Sphere T-matrix.

Args:
    radius (float): Radius in the
        length unit of k0.
    material: Relative permittivity.

Returns:
    The response. Its fields are:

    * extinction

Raises:
    ValueError: lmax is below 1.

Examples
--------
Text under a numpy-style header."""
    assert (
        _catalog.docstring_markdown(doc)
        == """Sphere T-matrix.

**Args**

- `radius` (float): Radius in the length unit of k0.
- `material`: Relative permittivity.

**Returns**

The response. Its fields are:

* extinction

**Raises**

- `ValueError`: lmax is below 1.

**Examples**

Text under a numpy-style header."""
    )


def test_docstring_examples_become_fences():
    doc = """A sphere::

    import treams_rs as tr
    particle = tr.sphere_tmatrix(k0=2.0, lmax=1, radius=0.1, material=2.0)

The same in a doctest:

>>> 1 + 1
2

::

    standalone = True
Prose with `code` stays as written.

```python
fenced = "unchanged::"
```"""
    assert (
        _catalog.docstring_markdown(doc)
        == """A sphere:

```python
import treams_rs as tr
particle = tr.sphere_tmatrix(k0=2.0, lmax=1, radius=0.1, material=2.0)
```

The same in a doctest:

```pycon
>>> 1 + 1
2
```

```python
standalone = True
```

Prose with `code` stays as written.

```python
fenced = "unchanged::"
```"""
    )


def test_generated_pages_render_every_name_once(monkeypatch):
    monkeypatch.setattr(_catalog, "support_catalog", installed_catalog)
    generator = runpy.run_path(str(ROOT / "scripts/generate_docs.py"))
    catalog = installed_catalog()
    pages = generator["reference_pages"](catalog)
    modules = {
        path.stem: text
        for path, text in pages.items()
        if path.stem not in {"index", "native-types"}
    }
    assert set(modules) == {"treams_rs"} | {
        module.removeprefix("treams_rs.") for module in catalog["modules"]
    }
    for path, text in pages.items():
        assert "```text" not in text, path.name
        assert text.startswith("---\ndescription: "), path.name
    # Headings name entries relative to the page's module; resolved, they cover
    # every catalog path exactly once.
    rendered = collections.Counter(
        f"{'treams_rs' if stem == 'treams_rs' else f'treams_rs.{stem}'}.{heading}"
        for stem, text in modules.items()
        for heading in re.findall(r"^#{2,} `(\S+)`$", text, flags=re.M)
    )
    assert rendered == collections.Counter(
        row["path"] for row in _entries(catalog["api"])
    )


def test_name_map_region_lists_every_treams_name(catalog):
    text = (ROOT / "docs/coming-from-treams/names.md").read_text(encoding="utf-8")
    region = text.partition("<!-- generated: upstream-names -->")[2]
    region = region.partition("<!-- end generated -->")[0]
    assert set(catalog["upstream"]["names"]) == set(tr._upstream.UPSTREAM_NAMES)
    for name in catalog["upstream"]["names"]:
        assert f"| `treams.{name}` |" in region, name
    for owner, members in catalog["upstream"]["members"].items():
        for name in members:
            assert f"| `{owner}` | `{name}` |" in region, (owner, name)


def test_generated_regions_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(_catalog, "support_catalog", installed_catalog)
    generator = runpy.run_path(str(ROOT / "scripts/generate_docs.py"))
    regions = {"names": lambda: "| a | b |\n| --- | --- |\n| 1 | 2 |\n"}
    page = "# Page\n\nIntro.\n\n<!-- generated: names -->\n<!-- end generated -->\n"
    filled = generator["fill_regions"](page, regions)
    assert filled == (
        "# Page\n\nIntro.\n\n<!-- generated: names -->\n\n"
        "| a | b |\n| --- | --- |\n| 1 | 2 |\n\n<!-- end generated -->\n"
    )
    assert generator["fill_regions"](filled, regions) == filled
    # --check notices a hand edit inside the region.
    path = tmp_path / "page.md"
    path.write_text(filled.replace("| 1 | 2 |", "| 1 | 3 |"), encoding="utf-8")
    assert generator["stale_files"]({path: filled}) == [path]
    path.write_text(filled, encoding="utf-8")
    assert generator["stale_files"]({path: filled}) == []
    with pytest.raises(ValueError, match="unknown generated region 'nope'"):
        generator["fill_regions"](
            "<!-- generated: nope -->\n<!-- end generated -->", {}
        )


def test_rust_crosswalk_is_read_from_the_crate_docs():
    generator = runpy.run_path(str(ROOT / "scripts/generate_docs.py"))
    source = """//! Crate summary.
//!
//! <!-- crosswalk:start -->
//! | Rust module | treams_rs namespace | treams | Contents |
//! |---|---|---|---|
//! | [`sw`] | `treams_rs.sw` | `treams.sw` | [`sw::translate`][crate::sw::translate] and [`x`](https://example.org) |
//! <!-- crosswalk:end -->
//!
//! More docs.
"""
    assert generator["rust_crosswalk"](source) == (
        "| Rust module | treams_rs namespace | treams | Contents |\n"
        "|---|---|---|---|\n"
        "| `sw` | `treams_rs.sw` | `treams.sw` | `sw::translate` and"
        " [`x`](https://example.org) |\n"
    )
    with pytest.raises(ValueError, match="crosswalk"):
        generator["rust_crosswalk"]("//! No table.\n")


@pytest.mark.parametrize(
    "arguments,message",
    [
        (("--search", "zzz"), "no names match"),
        (("sphere_tmatrix", "--search", "slab"), "choose one topic"),
        (("nope",), "unknown API path"),
    ],
)
def test_installed_help_rejects_invalid_requests(
    monkeypatch, capsys, arguments, message
):
    with pytest.raises(SystemExit) as exit_:
        run_cli(monkeypatch, capsys, *arguments)
    assert exit_.value.code == 2
    assert message in capsys.readouterr().err

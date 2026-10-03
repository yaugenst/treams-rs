"""Export rules of the public namespaces.

Each namespace below declares a literal, sorted ``__all__`` that the API catalog
reads from source, and keeps every other import private, so that ``dir()`` and
tab completion show the API and nothing else. The upstream-mirroring namespaces
export exactly the treams names plus the treams-rs extensions listed here.

Every catalog row and public member has its own docstring: mirrored functions
name their treams function, Python functions list their Args, and every
``diff`` record lists Returns, Dynamic inputs and Static configuration.
"""

import __future__

import ast
import importlib
import importlib.util
import inspect
import json
import pkgutil
import re
import subprocess
import sys
import textwrap
import typing
from types import ModuleType

import pytest

import treams_rs as tr
from treams_rs._upstream import UPSTREAM_MEMBERS, UPSTREAM_NAMES

from _support import catalog as installed_catalog

pytestmark = pytest.mark.interface

UPSTREAM_MIRRORS = (
    "coeffs",
    "cw",
    "ebcm",
    "io",
    "lattice",
    "misc",
    "pw",
    "special",
    "sw",
)
NAMESPACES = (
    "treams_rs",
    *(f"treams_rs.{name}" for name in (*UPSTREAM_MIRRORS, "iterative")),
    "treams_rs.operators",
    "treams_rs.testing",
)
#: treams-rs names exported by an upstream-mirroring namespace besides the treams ones.
EXTENSIONS = {
    "ebcm": {"Modes"},
    "io": {"MatrixSet"},
    "lattice": {"SumResult"},
}
_MISSING = object()


def _import(name):
    """Import a namespace, skipping an optional one whose dependency is missing."""
    dependency = tr._OPTIONAL_MODULES.get(name.removeprefix("treams_rs."))
    if dependency is not None:
        pytest.importorskip(dependency)
    return importlib.import_module(name)


def _bound_names(node):
    """Names a top-level statement assigns to."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        targets = [node.target]
    else:
        targets = []
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _literal_all(name):
    """The ``__all__`` list as written in the module source."""
    with open(importlib.util.find_spec(name).origin, encoding="utf-8") as source:
        tree = ast.parse(source.read())
    bindings = [node for node in tree.body if "__all__" in _bound_names(node)]
    assert len(bindings) == 1, f"{name} must bind __all__ exactly once"
    assert isinstance(bindings[0], ast.Assign), f"{name}.__all__ must be assigned"
    exports = ast.literal_eval(bindings[0].value)
    assert isinstance(exports, list), f"{name}.__all__ must be a list literal"
    assert all(isinstance(export, str) for export in exports)
    return exports


def _ruff_order(name):
    """Sort key of ruff's RUF022: constants, classes, then the rest, digits by value."""
    kind = 0 if name.isupper() else 1 if name[0].isupper() else 2
    return kind, [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)
    ]


def _incidental(name, value):
    """Imports that may stay public: modules, the annotations feature, typing names."""
    return (
        isinstance(value, ModuleType)
        or value is __future__.annotations
        or getattr(typing, name, _MISSING) is value
    )


def _upstream_names(module):
    """Public names of a treams namespace, without modules and standard-library imports."""
    return {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and not isinstance(value, ModuleType)
        and (getattr(value, "__module__", None) or "").partition(".")[0]
        not in sys.stdlib_module_names
    }


@pytest.mark.parametrize("name", NAMESPACES)
def test_namespace_exports_a_sorted_literal_all(name):
    module = _import(name)
    exports = _literal_all(name)
    assert module.__all__ == exports
    assert len(set(exports)) == len(exports)
    assert exports == sorted(exports, key=_ruff_order)
    assert [export for export in exports if not hasattr(module, export)] == []


@pytest.mark.parametrize("name", NAMESPACES)
def test_namespace_keeps_its_imports_private(name):
    module = _import(name)
    public = {attribute for attribute in dir(module) if not attribute.startswith("_")}
    leaked = {
        attribute
        for attribute in public - set(module.__all__)
        if not _incidental(attribute, getattr(module, attribute))
    }
    assert leaked == set()


@pytest.mark.parametrize("namespace", UPSTREAM_MIRRORS)
def test_upstream_mirror_exports_the_treams_names(namespace):
    pytest.importorskip("treams")
    module = _import(f"treams_rs.{namespace}")
    upstream = importlib.import_module(f"treams.{namespace}")
    names = _upstream_names(upstream)
    assert set(module.__all__) == names | EXTENSIONS.get(namespace, set())
    # Every treams function is a callable of the same name (constants are data).
    assert [
        name
        for name in names
        if callable(getattr(upstream, name)) and not callable(getattr(module, name))
    ] == []


ADAPTERS = ("treams_rs.advect", "treams_rs.jax", "treams_rs.torch")


@pytest.mark.parametrize("name", NAMESPACES + ADAPTERS)
def test_exported_classes_carry_their_export_name(name):
    module = _import(name)
    renamed = {
        export: value.__name__
        for export in module.__all__
        if isinstance(value := getattr(module, export), type)
        and value.__name__ != export
    }
    assert renamed == {}


def test_objects_report_their_export_name():
    k0 = 2.0
    sphere = tr.sphere_tmatrix(k0=k0, lmax=1, radius=0.2, material=3.0)
    cylinder = tr.cylinder_tmatrix(k0=k0, kz=0.0, mmax=1, radius=0.2, material=3.0)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
    objects = {
        "CylindricalBasis": cylinder.basis,
        "CylindricalTMatrix": cylinder,
        "PlaneWave": incident,
        "PlaneWaveBasis": tr.PlaneWaveBasis.default([[0, 0, 1]]),
        "PlaneWavePorts": tr.PlaneWavePorts.default([0, 0]),
        "SphericalBasis": sphere.basis,
        "TMatrix": sphere,
        "Wave": sphere.scatter(incident),
    }
    assert {name: type(value).__name__ for name, value in objects.items()} == {
        name: name for name in objects
    }
    assert all(getattr(tr, name) is type(value) for name, value in objects.items())


@pytest.mark.parametrize("name", sorted(UPSTREAM_NAMES))
def test_treams_names_raise_an_error_naming_the_target(name):
    target, note = UPSTREAM_NAMES[name]
    with pytest.raises(AttributeError) as error:
        getattr(tr, name)
    message = str(error.value)
    assert message.startswith(f"treams_rs has no {name!r}")
    assert (f"treams_rs.{target}" if target is not None else note) in message
    if target is not None:
        # The target exists under exactly the name the message gives.
        value = tr
        for part in target.split("."):
            value = getattr(value, part)


def test_unknown_names_raise_the_default_error():
    with pytest.raises(
        AttributeError, match=r"^module treams_rs has no attribute 'foo'$"
    ):
        _ = tr.foo


def test_every_treams_top_level_name_resolves_or_is_mapped():
    treams = pytest.importorskip("treams")
    public = {name for name in dir(treams) if not name.startswith("_")}
    unmapped = {name for name in public if not hasattr(tr, name)} - set(UPSTREAM_NAMES)
    assert unmapped == set()
    # A mapped name never shadows a treams-rs export.
    assert set(UPSTREAM_NAMES) & set(tr.__all__) == set()


def _physics_objects():
    """One instance of each physics class whose treams members raise helpful errors."""
    k0 = 1.3
    sphere = tr.sphere_tmatrix(k0=k0, lmax=1, radius=0.2, material=3.0)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
    ports = tr.PlaneWavePorts.default([0, 0])
    return {
        "TMatrix": sphere,
        "CylindricalTMatrix": tr.cylinder_tmatrix(
            k0=k0, kz=0.0, mmax=1, radius=0.2, material=3.0
        ),
        "SMatrix": tr.slab(basis=ports, k0=k0, thickness=0.2, material=2.0),
        "Wave": sphere.scatter(incident),
        "PlaneWave": incident,
    }


_PHYSICS = _physics_objects()
_TREAMS_MEMBERS = [
    (owner, name) for owner in _PHYSICS for name in UPSTREAM_MEMBERS[owner]
]
_MARKER = "# treams-compatible names (see treams_rs._upstream.UPSTREAM_MEMBERS)"


@pytest.mark.parametrize(("owner", "name"), _TREAMS_MEMBERS)
def test_treams_members_resolve_or_raise_an_error_naming_the_target(owner, name):
    value = _PHYSICS[owner]
    target, note = UPSTREAM_MEMBERS[owner][name]
    if target is not None:
        # The treams-rs member that the table names exists.
        assert hasattr(value, target), f"{owner}.{target}"
    if hasattr(type(value), name):
        if target is not None and not note and not callable(getattr(value, name)):
            # A treams attribute reads the same value as its treams-rs name.
            assert getattr(value, name) == getattr(value, target)
        return
    for holder in (type(value), value):
        with pytest.raises(AttributeError) as error:
            getattr(holder, name)
        message = str(error.value)
        assert message.startswith(f"{owner} has no {name!r}")
        assert (f"{owner}.{target}" if target is not None else note) in message


@pytest.mark.parametrize("owner", sorted(_PHYSICS))
def test_physics_objects_raise_the_default_error_for_unknown_names(owner):
    value = _PHYSICS[owner]
    with pytest.raises(
        AttributeError, match=rf"^type object '{owner}' has no attribute 'foo'$"
    ):
        _ = type(value).foo
    with pytest.raises(
        AttributeError, match=rf"^'{owner}' object has no attribute 'foo'$"
    ):
        _ = value.foo


def _class_members(cls):
    """Line of each member defined in the body of ``cls``, and of the marker comment."""
    lines, _ = inspect.getsourcelines(cls)
    body = ast.parse(textwrap.dedent("".join(lines))).body[0].body
    members = {}
    for statement in body:
        if isinstance(statement, ast.FunctionDef):
            members[statement.name] = statement.lineno
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
            if isinstance(statement, ast.AnnAssign)
            else []
        )
        members.update(
            {t.id: statement.lineno for t in targets if isinstance(t, ast.Name)}
        )
    markers = [i + 1 for i, line in enumerate(lines) if line.strip() == _MARKER]
    return members, markers


@pytest.mark.parametrize("owner", sorted(_PHYSICS))
def test_treams_members_follow_one_marker_at_the_end_of_each_class(owner):
    treams_names = set(UPSTREAM_MEMBERS[owner])
    for cls in type(_PHYSICS[owner]).__mro__:
        if not cls.__module__.startswith("treams_rs.") or cls.__name__ in (
            "WaveFields",
            "UpstreamMembers",
        ):
            continue
        members, markers = _class_members(cls)
        assert len(markers) == 1, cls
        below = {name for name, line in members.items() if line > markers[0]}
        # Below the marker only treams names; above it none of them.
        assert below <= treams_names, (cls, below - treams_names)
        assert members.keys() & treams_names <= below, cls


def test_optional_namespace_names_its_install_command():
    code = """
import sys
sys.modules["h5py"] = None  # makes `import h5py` fail as if it were missing
import treams_rs as tr
try:
    tr.io
except ModuleNotFoundError as error:
    print(error)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == (
        "treams_rs.io needs h5py; install it with pip install 'treams-rs[io]'"
    )


def _declared_type_parameters():
    """Names of the type parameters declared anywhere in the package."""
    names = set()
    for info in pkgutil.iter_modules(tr.__path__):
        if info.name == "__main__":  # importing it would break runpy in later tests
            continue
        try:
            module = importlib.import_module(f"treams_rs.{info.name}")
        except ModuleNotFoundError:  # an optional framework that is not installed
            continue
        for value in vars(module).values():
            if getattr(value, "__module__", None) == module.__name__:
                names |= {p.__name__ for p in getattr(value, "__type_params__", ())}
    return names


def _own_type_parameters(path):
    """Type parameters of the class a catalog path names, e.g. T of CrossSections."""
    module, _, name = path.rpartition(".")
    try:
        value = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return set()
    return {p.__name__ for p in getattr(value, "__type_params__", ())}


def test_catalog_signatures_name_importable_or_listed_types():
    catalog = tr.support_catalog()
    returned = {entry["path"] for entry in catalog["returned_python_types"]}
    native = {f"_native.{entry['path']}" for entry in catalog["native_types"]}
    variables = _declared_type_parameters() | {"Self"}
    assert {"B", "M"} <= variables
    unresolved = []

    def check(entry, declared):
        for token in re.findall(r"[A-Za-z_][\w.]*", entry.get("signature", "")):
            private = token.startswith("_") and token not in returned | native
            if private or (token in variables and token not in declared):
                unresolved.append((entry["path"], token))
        for member in entry.get("members", []):
            check(member, declared)

    for entry in catalog["api"]:
        check(entry, _own_type_parameters(entry["path"]))
    for entry in catalog["returned_python_types"]:
        check(entry, set())
    assert unresolved == []
    members = {
        member["path"]: member["signature"]
        for entry in catalog["api"]
        for member in entry.get("members", [])
        if "signature" in member
    }
    assert members["treams_rs.TMatrix.in_basis"] == "(basis: SphericalBasis) -> TMatrix"
    assert members["treams_rs.CylindricalTMatrix.in_basis"] == (
        "(basis: CylindricalBasis) -> CylindricalTMatrix"
    )
    assert members["treams_rs.TMatrix.interaction"] == "() -> _Interaction[TMatrix]"
    solve = {
        member["path"]: member["signature"]
        for entry in catalog["returned_python_types"]
        for member in entry["members"]
    }["_Interaction.solve"]
    assert solve == "() -> TMatrix | CylindricalTMatrix"


def test_catalog_lists_the_treams_name_map():
    upstream = tr.support_catalog()["upstream"]
    assert json.loads(json.dumps(upstream)) == upstream
    assert upstream["names"]["TMatrixC"] == {"target": "CylindricalTMatrix", "note": ""}
    assert upstream["names"].keys() == UPSTREAM_NAMES.keys()
    assert upstream["members"]["Material"]["from_n"]["target"] == (
        "from_refractive_index"
    )


# Docstrings ------------------------------------------------------------------------

#: The placeholder docstring that a native ufunc without its own would show.
_GENERIC_UFUNC_DOC = "Rust special function with NumPy broadcasting"
#: Python wrappers of one lattice dimension: one paragraph names the sum, the
#: argument shapes and ``eta``; the dimension dispatchers (``lsumsw`` ...) carry the
#: full Args section.
_SHORT_LATTICE_WRAPPERS = {
    f"treams_rs.lattice.{name}"
    for name in (
        "dsumcw1d",
        "dsumcw1d_shift",
        "dsumcw2d",
        "lsumcw1d",
        "lsumcw1d_shift",
        "lsumcw2d",
        "lsumsw1d",
        "lsumsw1d_shift",
        "lsumsw2d",
        "lsumsw2d_shift",
        "lsumsw3d",
    )
}


def _catalog_entries(rows):
    """Catalog rows followed by their members, depth first."""
    return [
        entry
        for row in rows
        for entry in (row, *_catalog_entries(row.get("members", [])))
    ]


def _mirror_rows(namespace):
    prefix = f"treams_rs.{namespace}."
    return [row for row in installed_catalog()["api"] if row["path"].startswith(prefix)]


def _sections(doc):
    """Headings of the Google-style sections of a docstring, in order."""
    return [
        line[:-1]
        for line in inspect.cleandoc(doc).splitlines()
        if re.fullmatch(r"[A-Z][a-z]+(?: [a-z]+)*:", line)
    ]


@pytest.mark.parametrize("namespace", UPSTREAM_MIRRORS)
def test_upstream_mirror_rows_have_their_own_docstrings(namespace):
    rows = _mirror_rows(namespace)
    assert rows
    assert [row["path"] for row in rows if not row["doc"].strip()] == []
    assert [row["path"] for row in rows if _GENERIC_UFUNC_DOC in row["doc"]] == []
    # A ufunc row shows the treams argument names, not NumPy's x1, x2, ...
    assert [
        row["path"]
        for row in rows
        if row["kind"] == "ufunc" and re.search(r"\(x1, x2", row["signature"])
    ] == []


@pytest.mark.parametrize("namespace", UPSTREAM_MIRRORS)
def test_upstream_mirror_rows_name_their_treams_function(namespace):
    extensions = {
        f"treams_rs.{namespace}.{name}" for name in EXTENSIONS.get(namespace, ())
    }
    unnamed = []
    for row in _mirror_rows(namespace):
        if row["path"] in extensions:
            continue
        # An alias such as lattice.area names the function it shares.
        target = row.get("alias_of", row["path"]).replace("treams_rs.", "treams.", 1)
        if f"Mirrors ``{target}``" not in row["doc"]:
            unnamed.append(row["path"])
    assert unnamed == []


@pytest.mark.parametrize("namespace", UPSTREAM_MIRRORS)
def test_python_functions_of_upstream_mirrors_have_an_args_section(namespace):
    module = _import(f"treams_rs.{namespace}")
    without_args = {
        row["path"]
        for row in _mirror_rows(namespace)
        if row["kind"] == "function"
        and inspect.isfunction(getattr(module, row["path"].rpartition(".")[2]))
        and "Args" not in _sections(row["doc"])
    }
    # Native functions follow the ufunc docstring form; the Python ones list Args.
    assert without_args == {
        path
        for path in _SHORT_LATTICE_WRAPPERS
        if path.startswith(f"treams_rs.{namespace}.")
    }


def test_short_docstring_lists_name_python_functions():
    for path in _SHORT_LATTICE_WRAPPERS:
        module, _, name = path.rpartition(".")
        assert inspect.isfunction(getattr(importlib.import_module(module), name)), path


def test_every_catalog_definition_and_public_member_has_a_docstring():
    catalog = installed_catalog()
    undocumented = [
        entry["path"]
        for entry in _catalog_entries(catalog["api"] + catalog["returned_python_types"])
        # Field annotations of named tuples and dataclasses carry no docstring.
        if entry["kind"] != "attribute"
        and not entry["path"].rpartition(".")[2].startswith("_")
        and not entry["doc"].strip()
    ]
    assert undocumented == []
    assert [name for name, doc in catalog["modules"].items() if not doc.strip()] == []


def test_diff_records_follow_one_docstring_template():
    rows = [
        row
        for row in installed_catalog()["api"]
        if row["path"].startswith("treams_rs.diff.") and row["kind"] == "function"
    ]
    assert len(rows) == len(tr.diff.__all__)
    template = ["Returns", "Dynamic inputs", "Static configuration"]
    assert {
        row["path"]: _sections(row["doc"])
        for row in rows
        if [s for s in _sections(row["doc"]) if s in template] != template
    } == {}

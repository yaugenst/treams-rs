"""Source-derived API catalog behind support_catalog(), python -m treams_rs and the generated reference.

Signatures, docstrings and pullback methods come from the package source and
the native type stub, parsed rather than imported, so discovery never loads an
optional framework and works from an installed wheel without a checkout.

Discovery is complete only for source that follows these constraints:

* Only top-level statements of the modules directly inside the package count.
* ``__all__`` is a literal list of names.
* A public function built by a module-level instance is bound as
  ``name = _instance.method`` and documented by that method.
* A class-level alias is an assignment ``alias = member`` in the class body;
  it is listed with kind ``"alias"`` and the member's signature and docstring.
* Re-exports are ``from .module import name`` statements listed in ``__all__``.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import re
import textwrap
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from . import _native, _upstream

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = [
    "docstring_markdown",
    "render_markdown",
    "render_overview",
    "render_returned_types",
    "support_catalog",
]


def _version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class _Substitute(ast.NodeTransformer):
    """Replace type-parameter names in an annotation by concrete types."""

    def __init__(self, types: Mapping[str, ast.expr]):
        self.types = types

    @override
    def visit_Name(self, node: ast.Name) -> ast.expr:
        replacement = self.types.get(node.id)
        return node if replacement is None else copy.deepcopy(replacement)


def _annotation(node: ast.expr, types: Mapping[str, ast.expr]) -> str:
    """Source text of an annotation with ``types`` substituted."""
    if types:
        node = _Substitute(types).visit(copy.deepcopy(node))
    return ast.unparse(node)


def _signature(
    node: ast.FunctionDef, types: Mapping[str, ast.expr] | None = None
) -> str:
    args = copy.copy(node.args)
    positional = args.posonlyargs + args.args
    if positional and positional[0].arg in {"self", "cls"}:
        if args.posonlyargs:
            args.posonlyargs = args.posonlyargs[1:]
        else:
            args.args = args.args[1:]
    if types:
        args = _Substitute(types).visit(copy.deepcopy(args))
    result = f"({ast.unparse(args)})"
    if node.returns is not None:
        result += f" -> {_annotation(node.returns, types or {})}"
    return result


def _definitions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.ClassDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def _exports(tree: ast.Module) -> list[str] | None:
    """The module's literal ``__all__``, when it defines one."""
    return next(
        (
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        ),
        None,
    )


def _following_docstring(body: list[ast.stmt], index: int) -> str:
    """The string literal right after ``body[index]`` (an attribute docstring)."""
    following = body[index + 1] if index + 1 < len(body) else None
    if (
        isinstance(following, ast.Expr)
        and isinstance(following.value, ast.Constant)
        and isinstance(following.value.value, str)
    ):
        return following.value.value
    return ""


def _row(path: str, kind: str, signature: str, doc: str, source: str) -> dict[str, Any]:
    """A catalog entry without members or pullbacks."""
    return {
        "path": path,
        "kind": kind,
        "signature": signature,
        "doc": doc,
        "source": source,
        "pullbacks": [],
    }


def _type_alias_row(
    path: str, body: list[ast.stmt], index: int, stem: str
) -> dict[str, Any]:
    """The entry of the ``type`` statement ``body[index]`` of module ``stem``."""
    node = body[index]
    assert isinstance(node, ast.TypeAlias)
    return _row(
        path,
        "type_alias",
        f" = {ast.unparse(node.value)}",
        _following_docstring(body, index),
        f"{stem}.py:{node.lineno}",
    )


def _pullback_methods(context: ast.ClassDef) -> list[ast.FunctionDef]:
    """The methods of a context class whose names start with ``pullback``."""
    return [
        method
        for method in context.body
        if isinstance(method, ast.FunctionDef) and method.name.startswith("pullback")
    ]


def _entry(
    path: str,
    node: ast.FunctionDef | ast.ClassDef,
    source: str,
    contexts: dict[str, ast.ClassDef],
    classes: dict[str, tuple[ast.ClassDef, str]],
) -> dict[str, Any]:
    if isinstance(node, ast.ClassDef):
        return _class_entry(path, node, source, contexts, classes)
    return _function_entry(path, node, source, contexts)


def _function_entry(
    path: str,
    node: ast.FunctionDef,
    source: str,
    contexts: dict[str, ast.ClassDef],
    types: Mapping[str, ast.expr] | None = None,
) -> dict[str, Any]:
    """Signature, docstring and the pullbacks of an annotated returned context.

    ``types`` replaces type-parameter names in the signature, for example
    ``B`` by ``SphericalBasis`` in members that TMatrix inherits.
    """
    property_ = any(
        isinstance(d, ast.Name) and d.id in {"property", "cached_property"}
        for d in node.decorator_list
    )
    pullbacks: list[dict[str, str]] = []
    for annotation in ast.walk(node.returns) if node.returns is not None else ():
        name = (
            annotation.attr
            if isinstance(annotation, ast.Attribute)
            else (annotation.id if isinstance(annotation, ast.Name) else "")
        )
        context = contexts.get(name)
        if context is not None:
            pullbacks.extend(
                {
                    "context": name,
                    "method": method.name,
                    "signature": _signature(method),
                    "doc": ast.get_docstring(method) or "",
                }
                for method in _pullback_methods(context)
            )
    return {
        "path": path,
        "kind": "property" if property_ else "function",
        "doc": ast.get_docstring(node) or "",
        "source": f"{source}:{node.lineno}",
        "signature": _signature(node, types),
        "pullbacks": pullbacks,
    }


def _type_arguments(base: ast.expr) -> list[ast.expr]:
    """The subscript arguments of a base class, e.g. ``[B]`` for ``_TMatrix[B]``."""
    if not isinstance(base, ast.Subscript):
        return []
    index = base.slice
    return list(index.elts) if isinstance(index, ast.Tuple) else [index]


def _union(types: list[ast.expr]) -> ast.expr:
    result = types[0]
    for item in types[1:]:
        result = ast.BinOp(result, ast.BitOr(), item)
    return result


def _type_parameters(node: ast.ClassDef) -> list[ast.TypeVar]:
    """The type variables a class declares, e.g. ``B`` of ``class _TMatrix[B: ...]``."""
    return [
        parameter
        for parameter in node.type_params
        if isinstance(parameter, ast.TypeVar)
    ]


def _bound(
    parameter: ast.TypeVar, classes: dict[str, tuple[ast.ClassDef, str]]
) -> ast.expr | None:
    """The types a type parameter stands for when no subclass fixes it.

    Constraints become their union, and a private generic bound becomes the
    union of its public subclasses: ``M: _TMatrix[Any]`` reads
    ``TMatrix | CylindricalTMatrix``. None for an unbounded parameter.
    """
    bound = parameter.bound
    if bound is None:
        return None
    if isinstance(bound, ast.Tuple):
        return _union(list(bound.elts))
    name = ast.unparse(bound.value if isinstance(bound, ast.Subscript) else bound)
    if not name.startswith("_"):
        return bound
    subclasses = {
        node.name: None
        for node, _ in classes.values()
        if not node.name.startswith("_")
        and any(
            ast.unparse(base.value if isinstance(base, ast.Subscript) else base) == name
            for base in node.bases
        )
    }
    return (
        _union([ast.Name(subclass) for subclass in subclasses]) if subclasses else bound
    )


def _class_entry(
    path: str,
    node: ast.ClassDef,
    source: str,
    contexts: dict[str, ast.ClassDef],
    classes: dict[str, tuple[ast.ClassDef, str]],
    types: Mapping[str, ast.expr] | None = None,
    *,
    constructor: bool | None = None,
) -> dict[str, Any]:
    """Class docstring and members, including inherited treams-rs members.

    Signatures name concrete types: ``Self`` becomes the listed class, and the
    type parameters of a generic base become the arguments the subclass passes,
    so TMatrix(_TMatrix[SphericalBasis]) lists ``in_basis(basis:
    SphericalBasis)``. ``types`` carries those arguments into the base.

    Classes of the ``_framework*`` modules are listed without ``__init__``, which
    keeps internal arguments such as the Backend out of the reference; the
    lowercase constructors and the solvers build most of these objects.
    ``constructor`` overrides that choice.
    """
    own: dict[str, ast.expr] = {"Self": ast.Name(path.rsplit(".", 1)[-1])}
    own |= types or {}
    for parameter in _type_parameters(node):
        bound = None if parameter.name in own else _bound(parameter, classes)
        if bound is not None:
            own[parameter.name] = bound
    members: dict[str, dict[str, Any]] = {}
    for base in node.bases:
        arguments = _type_arguments(base)
        base = base.value if isinstance(base, ast.Subscript) else base
        base_name = ast.unparse(base)
        parent_class = classes.get(
            f"{Path(source).stem}.{base_name}", classes.get(base_name)
        )
        if parent_class is not None:
            parent, parent_source = parent_class
            inherited_types = {"Self": own["Self"]} | {
                parameter.name: _Substitute(own).visit(copy.deepcopy(argument))
                for parameter, argument in zip(
                    _type_parameters(parent), arguments, strict=False
                )
            }
            inherited = _class_entry(
                path,
                parent,
                parent_source,
                contexts,
                classes,
                inherited_types,
                constructor=constructor,
            )
            members.update({member["path"]: member for member in inherited["members"]})
    if constructor is None:
        constructor = not Path(source).stem.startswith("_framework")
    members.update(
        {
            f"{path}.{member.name}": _function_entry(
                f"{path}.{member.name}", member, source, contexts, own
            )
            for member in node.body
            if isinstance(member, ast.FunctionDef)
            and (
                not member.name.startswith("_")
                or (member.name == "__init__" and constructor)
            )
        }
    )
    for index, attribute in enumerate(node.body):
        if isinstance(attribute, ast.AnnAssign):
            targets = [attribute.target]
            annotation = f": {_annotation(attribute.annotation, own)}"
        elif isinstance(attribute, ast.Assign):
            targets, annotation = attribute.targets, ""
        else:
            continue
        value = attribute.value
        # `alias = member` documents the member it names (`expand = in_basis`).
        original = (
            members.get(f"{path}.{value.id}")
            if isinstance(attribute, ast.Assign) and isinstance(value, ast.Name)
            else None
        )
        if original is not None:
            for target in targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    members[f"{path}.{target.id}"] = original | {
                        "path": f"{path}.{target.id}",
                        "kind": "alias",
                        "source": f"{source}:{attribute.lineno}",
                        "alias_of": original.get("alias_of", original["path"]),
                    }
            continue
        signature = annotation + (f" = {ast.unparse(value)}" if value else "")
        kind, description = "attribute", _following_docstring(node.body, index)
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "OperatorAttribute"
        ):
            kind = "descriptor"
            operator, _ = classes[ast.unparse(value.args[0]).split(".")[-1]]
            description = (
                (ast.get_docstring(operator) or "")
                + " Bound OperatorAttribute: call applies to the owning object; .eval returns the operator matrix."
            )
        for target in targets:
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                members[f"{path}.{target.id}"] = _row(
                    f"{path}.{target.id}",
                    kind,
                    signature,
                    description,
                    f"{source}:{attribute.lineno}",
                )
    return {
        "path": path,
        "kind": "class",
        "doc": ast.get_docstring(node) or "",
        "source": f"{source}:{node.lineno}",
        "bases": [ast.unparse(base) for base in node.bases],
        "members": list(members.values()),
    }


def _optional_dependencies(root: ast.Module) -> list[str]:
    """Distributions of the optional namespaces in the root's ``_OPTIONAL_MODULES``.

    Read from source: the package root imports this module before it defines
    the table.
    """
    table = next(
        node.value
        for node in root.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_OPTIONAL_MODULES"
            for t in node.targets
        )
    )
    return list(ast.literal_eval(table).values())


def _native_parts(native: Any) -> tuple[str, str]:
    """Signature and docstring of a native function or ufunc.

    A ufunc docstring opens with NumPy's call, ``jv(x1, x2, /, out=None, ...)``,
    and then with its own, ``jv(v, z, /, out=None, *, where=True)``, which names
    the arguments as treams does. The signature is the last of these calls, and
    the docstring is the text after them.
    """
    doc = inspect.getdoc(native) or ""
    calls = []
    while doc.startswith(f"{native.__name__}("):
        call, _, doc = doc.partition("\n")
        calls.append(call.removeprefix(native.__name__))
        doc = doc.lstrip("\n")
    if hasattr(native, "nin") and len(calls) > 1:
        return calls[-1], doc
    try:
        return str(inspect.signature(native)), doc
    except (TypeError, ValueError):
        return (calls[0] if calls else ""), doc


def _module_entries(
    stem: str,
    tree: ast.Module,
    definitions: dict[str, ast.FunctionDef | ast.ClassDef],
    contexts: dict[str, ast.ClassDef],
    classes: dict[str, tuple[ast.ClassDef, str]],
) -> list[dict[str, Any]]:
    exports = _exports(tree)

    def public(name: str) -> bool:
        return not name.startswith("_") and (exports is None or name in exports)

    entries = [
        _entry(f"treams_rs.{stem}.{name}", node, f"{stem}.py", contexts, classes)
        for name, node in definitions.items()
        if public(name)
    ]
    # Bound methods of a module-level instance document the method itself, e.g.
    # ``slab = _api.slab`` after ``_api = _framework.Constructors(_backend, __name__)``.
    instances = {
        target.id: classes.get(f"{stem}.{ast.unparse(node.value.func)}")
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.TypeAlias) and public(node.name.id):
            entries.append(
                _type_alias_row(
                    f"treams_rs.{stem}.{node.name.id}", tree.body, index, stem
                )
            )
        # Literal constants such as iterative.DEFAULT_RTOL, with their docstrings.
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            entries.extend(
                _row(
                    f"treams_rs.{stem}.{target.id}",
                    "attribute",
                    f" = {ast.unparse(node.value)}",
                    _following_docstring(tree.body, index),
                    f"{stem}.py:{node.lineno}",
                )
                for target in node.targets
                if isinstance(target, ast.Name) and public(target.id)
            )
            continue
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
        ):
            continue
        targets = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name) and not target.id.startswith("_")
        ]
        if node.value.value.id == "_native":
            native = getattr(_native, node.value.attr)
            signature, native_doc = _native_parts(native)
            # An attribute docstring after the alias documents its arguments.
            doc = "\n\n".join(
                filter(None, (_following_docstring(tree.body, index), native_doc))
            )
            entries.extend(
                _row(
                    f"treams_rs.{stem}.{target}",
                    "ufunc" if hasattr(native, "nin") else "function",
                    signature,
                    doc,
                    f"{stem}.py:{node.lineno}",
                )
                for target in targets
            )
            continue
        owner = instances.get(node.value.value.id)
        method = (
            None
            if owner is None
            else next(
                (
                    member
                    for member in owner[0].body
                    if isinstance(member, ast.FunctionDef)
                    and member.name == node.value.attr
                ),
                None,
            )
        )
        if owner is not None and method is not None:
            entries.extend(
                _function_entry(
                    f"treams_rs.{stem}.{target}", method, owner[1], contexts
                )
                for target in targets
                if exports is None or target in exports
            )
    # Same-module aliases reuse their target's entry (for example lattice.area).
    by_name = {entry["path"].rsplit(".", 1)[-1]: entry for entry in entries}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            original = by_name.get(node.value.id)
            if original is None:
                continue
            entries.extend(
                original
                | {
                    "path": f"treams_rs.{stem}.{target.id}",
                    "source": f"{stem}.py:{node.lineno}",
                    "alias_of": original["path"],
                }
                for target in node.targets
                if isinstance(target, ast.Name) and not target.id.startswith("_")
            )
    return entries


def support_catalog() -> dict[str, Any]:
    """Return a JSON-serializable, version-matched public API and backend catalog.

    ``api`` contains public definitions, root re-exports, class members (including treams-rs inheritance),
    declared attributes/descriptors, and callable aliases. Inherited NumPy methods and incidental imports are
    excluded. ``pullbacks`` lists the methods on an annotated returned context;
    an empty list does not promise automatic differentiation through that API.
    Static parameters and derivative ordering are stated in each source docstring
    and signature. Use ``diff`` or a framework adapter to request derivatives.

    ``backends`` describes execution support. Optional dependency versions only
    establish installation, not compatible devices or a configured framework.

    ``returned_python_types`` lists the private classes that public members
    return, such as the solver of ``TMatrix.interaction``, with their methods.
    ``upstream`` maps the treams names that treams-rs spells differently:
    ``names`` covers the treams top level and ``members`` the members of treams
    objects, each as ``{"target": name or None, "note": text}``.

    No optional framework is imported, and this function also works from a wheel
    without a checkout or network. Reading source is diagnostic work, never part
    of the numerical hot path.

    Examples
    --------
    >>> import treams_rs as tr
    >>> catalog = tr.support_catalog()
    >>> catalog["schema_version"]
    1
    >>> catalog["backends"]["cpu"]["compiled"]
    True
    """
    package = Path(__file__).parent
    sources = {
        file.name: file.read_bytes()
        for file in sorted(package.glob("*.py*"))
        if file.suffix in {".py", ".pyi"}
    }
    trees = {
        Path(name).stem: ast.parse(source)
        for name, source in sources.items()
        if name.endswith(".py")
    }
    definitions = {stem: _definitions(tree) for stem, tree in trees.items()}
    native = _definitions(ast.parse(sources["_native.pyi"]))
    # A function annotated to return one of these classes lists its pullback
    # methods: every native stub class, and every package class that defines a
    # pullback method itself. A package class shadows a native class of the same
    # name (iterative.IterativeContext wraps _native.IterativeContext), so the
    # native types resolve their contexts among the stub classes alone.
    native_contexts = {
        name: node for name, node in native.items() if isinstance(node, ast.ClassDef)
    }
    contexts = native_contexts | {
        name: node
        for members in definitions.values()
        for name, node in members.items()
        if isinstance(node, ast.ClassDef) and _pullback_methods(node)
    }
    classes = {
        name: (node, f"{stem}.py")
        for stem, members in definitions.items()
        for name, node in members.items()
        if isinstance(node, ast.ClassDef)
    }
    # Resolve inherited methods in the defining module, including framework
    # facade subclasses such as jax.PlaneWave(_framework.PlaneWave).
    for stem, members in definitions.items():
        for name, node in members.items():
            if isinstance(node, ast.ClassDef):
                classes[f"{stem}.{name}"] = (node, f"{stem}.py")
    for stem, tree in trees.items():
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                if node.module in trees:
                    target = classes.get(f"{node.module}.{alias.name}")
                    if target is not None:
                        classes[f"{stem}.{local}"] = target
                elif node.module is None and alias.name in trees:
                    for name, definition in definitions[alias.name].items():
                        if isinstance(definition, ast.ClassDef):
                            classes[f"{stem}.{local}.{name}"] = (
                                definition,
                                f"{alias.name}.py",
                            )
    entries = [
        entry
        for stem, tree in sorted(trees.items())
        if not stem.startswith("_")
        for entry in _module_entries(stem, tree, definitions[stem], contexts, classes)
    ]

    # Explicit re-exports share the entry of their definition, including optional
    # framework classes. Resolve source only: discovery must not import them.
    def resolve(
        stem: str, name: str
    ) -> tuple[ast.FunctionDef | ast.ClassDef | ast.TypeAlias, str] | None:
        definition = definitions[stem].get(name)
        if definition is not None:
            return definition, stem
        for node in trees[stem].body:
            if isinstance(node, ast.TypeAlias) and node.name.id == name:
                return node, stem
            if isinstance(node, ast.ImportFrom) and node.module in trees:
                for alias in node.names:
                    if (alias.asname or alias.name) == name:
                        return resolve(node.module, alias.name)
        return None

    paths = {entry["path"] for entry in entries}
    for stem, tree in trees.items():
        if stem.startswith("_") and stem != "__init__":
            continue
        exports = _exports(tree) or []
        prefix = "treams_rs" if stem == "__init__" else f"treams_rs.{stem}"
        for name in exports:
            path = f"{prefix}.{name}"
            resolved = resolve(stem, name)
            if path not in paths and resolved is not None:
                node, source = resolved
                if isinstance(node, ast.TypeAlias):
                    body = trees[source].body
                    entries.append(
                        _type_alias_row(path, body, body.index(node), source)
                    )
                else:
                    entries.append(
                        _entry(path, node, f"{source}.py", contexts, classes)
                    )
                paths.add(path)
    # Names used in any function's return annotation, collected in one pass.
    returned = {
        annotation.id
        for tree in trees.values()
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef) and function.returns is not None
        for annotation in ast.walk(function.returns)
        if isinstance(annotation, ast.Name)
    }

    optional = {
        name: _version(name) for name in _optional_dependencies(trees["__init__"])
    }
    return {
        "schema_version": 1,
        "version": _version("treams-rs"),
        "python_source_sha256": hashlib.sha256(
            b"".join(name.encode() + b"\0" + data for name, data in sources.items())
        ).hexdigest(),
        "rules": {
            "numeric_precision": "float64 / complex128; metadata also uses integer and boolean arrays",
            "native_pairing": "dL = Re(sum(conj(g) * dx))",
            "derivative_order": 1,
            "static_parameters": "mode counts, integer labels, topology; see each operation's docstring",
            "context_lifetime": "Treat native recorded contexts as single-use; record again for another pullback.",
            "python_execution": "CPU",
        },
        "backends": {
            "cpu": {
                "compiled": True,
                "runtime": "loaded",
                "precision": ["float64", "complex128"],
            },
        },
        "optional_dependencies": optional,
        # Each adapter module's docstring opens with its one-paragraph summary.
        "adapters": {
            name: {
                "device": "cpu",
                "derivative_order": 1,
                "extra": name,
                "summary": " ".join(
                    (ast.get_docstring(trees[name]) or "").partition("\n\n")[0].split()
                ),
            }
            for name in ("advect", "jax", "torch")
        },
        "modules": {
            f"treams_rs.{stem}": ast.get_docstring(tree) or ""
            for stem, tree in sorted(trees.items())
            if not stem.startswith("_")
        },
        "returned_python_types": [
            _class_entry(name, node, source, contexts, classes, constructor=False)
            for name, (node, source) in classes.items()
            if name.startswith("_") and "." not in name and name in returned
        ],
        "native_types": [
            _entry(name, node, "_native.pyi", native_contexts, {})
            for name, node in native.items()
            if isinstance(node, ast.ClassDef)
        ],
        # treams names that treams-rs spells differently, for the docs name map.
        "upstream": {
            "names": {
                name: renamed._asdict()
                for name, renamed in _upstream.UPSTREAM_NAMES.items()
            },
            "members": {
                owner: {name: renamed._asdict() for name, renamed in members.items()}
                for owner, members in _upstream.UPSTREAM_MEMBERS.items()
            },
        },
        "api": sorted(entries, key=itemgetter("path")),
    }


_SECTION = re.compile(
    r"^(Args|Arguments|Parameters|Keyword Args|Keyword Arguments|Attributes|Returns"
    r"|Return|Yields|Raises|Note|Notes|Warning|Example|Examples|See Also"
    r"|Differences from treams|Dynamic inputs|Static configuration):$"
)
# Sections whose lines are ``name (type): text`` entries.
_ENTRY_SECTIONS = frozenset(
    {
        "Args",
        "Arguments",
        "Parameters",
        "Keyword Args",
        "Keyword Arguments",
        "Attributes",
        "Raises",
        "Dynamic inputs",
        "Static configuration",
    }
)
_SECTION_ENTRY = re.compile(
    r"^(?P<name>\*{0,2}[\w.]+)(?: \((?P<type>.+?)\))?:(?:\s+(?P<text>.*))?$"
)
_UNDERLINE = re.compile(r"^-{3,}$")


def _block(lines: list[str], start: int, indented: Callable[[str], bool]) -> int:
    """End of the block from ``start`` whose non-blank lines are ``indented``.

    Trailing blank lines stay outside the block.
    """
    end = start
    for index in range(start, len(lines)):
        if lines[index].strip():
            if not indented(lines[index]):
                break
            end = index + 1
    return end


def _section_entries(lines: list[str]) -> list[str]:
    """Bullets ``- `name` (type): text`` from the dedented body of an entry section.

    A deeper-indented line continues the entry above it.
    """
    out: list[str] = []
    for line in lines:
        match = _SECTION_ENTRY.match(line)
        if line[:1].isspace() and out and out[-1].startswith("- "):
            out[-1] += " " + line.strip()
        elif match:
            entry = f"- `{match['name']}`"
            if match["type"]:
                entry += f" ({match['type']})"
            out.append(entry + (f": {match['text']}" if match["text"] else ":"))
        else:
            out.append(line)
    return out


def _fence(out: list[str], language: str, code: list[str], following: str) -> None:
    """Append ``code`` as a fenced block, separated from the text around it."""
    if out and out[-1].strip():
        out.append("")
    out.extend([f"```{language}", *textwrap.dedent("\n".join(code)).splitlines()])
    out.append("```")
    if following.strip():
        out.append("")


def docstring_markdown(doc: str) -> str:
    """Markdown for a Google- or reST-style docstring.

    Section headers such as ``Args:`` become bold labels, and the
    ``name (type): text`` entries of argument, attribute and exception
    sections become bullets. A paragraph ending in ``::`` introduces a Python
    fence, ``>>>`` examples become ``pycon`` fences, and a numpy-style header
    underlined with dashes becomes a bold label. Other text stays as written.
    """
    lines = [*doc.splitlines(), ""]
    out: list[str] = []
    index = 0
    while index < len(lines) - 1:
        line, following = lines[index], lines[index + 1]
        section = _SECTION.match(line)
        if line.lstrip().startswith("```"):
            # An existing fence passes through unchanged.
            end = next(
                (
                    position + 1
                    for position in range(index + 1, len(lines) - 1)
                    if lines[position].lstrip().startswith("```")
                ),
                len(lines) - 1,
            )
            out.extend(lines[index:end])
            index = end
        elif section:
            end = _block(lines, index + 1, lambda text: text[:1].isspace())
            body = textwrap.dedent("\n".join(lines[index + 1 : end])).splitlines()
            out.extend([f"**{section[1]}**", ""])
            if section[1] in _ENTRY_SECTIONS:
                out.extend(_section_entries(body))
            else:
                out.extend(docstring_markdown("\n".join(body)).splitlines())
            index = end
        elif line.strip() and not line[:1].isspace() and _UNDERLINE.match(following):
            out.extend([f"**{line.strip()}**", ""])
            index += 3 if not lines[index + 2].strip() else 2
        elif line.lstrip().startswith(">>> "):
            # A doctest runs up to the next blank line.
            end = next(
                position
                for position in range(index, len(lines))
                if not lines[position].strip()
            )
            _fence(out, "pycon", lines[index:end], lines[end])
            index = end
        elif (
            line.rstrip().endswith("::")
            and not following.strip()
            and index + 2 < len(lines)
            and lines[index + 2].startswith("    ")
        ):
            text = line.rstrip()[:-2].rstrip()
            if text:
                out.append(text + ":")
            end = _block(lines, index + 2, lambda text: text.startswith("    "))
            _fence(out, "python", lines[index + 2 : end], lines[end])
            index = end
        else:
            out.append(line)
            index += 1
    return "\n".join(out)


def _signature_line(entry: dict[str, Any], name: str) -> str:
    """The call form of an entry: ``f(x) -> y``, ``Class(x)`` or ``prop: type``."""
    signature = entry["signature"]
    if entry["kind"] == "property":
        returns = signature.partition(" -> ")[2]
        return f"{name}: {returns}" if returns else name
    if name.endswith(".__init__"):
        return name.removesuffix(".__init__") + signature
    return name + signature


def _entry_markdown(entry: dict[str, Any], level: int, module: str) -> list[str]:
    """Markdown of one catalog entry and its members, headed at ``level``.

    Headings and signatures name the entry relative to ``module``, for example
    ``TMatrix.scatter`` on the page of the package root.
    """

    def name(path: str) -> str:
        return path.removeprefix(f"{module}.") if module else path

    lines = [f"{'#' * level} `{name(entry['path'])}`", ""]
    if "signature" in entry:
        lines.extend(
            ["```python", _signature_line(entry, name(entry["path"])), "```", ""]
        )
    if "alias_of" in entry:
        lines.extend([f"Same as `{name(entry['alias_of'])}`.", ""])
    if entry["doc"]:
        lines.extend([docstring_markdown(entry["doc"]), ""])
    for pullback in entry.get("pullbacks", []):
        call = f"{pullback['context']}.{pullback['method']}{pullback['signature']}"
        lines.extend([f"**Pullback** `{call}`", ""])
        if pullback["doc"]:
            lines.extend([docstring_markdown(pullback["doc"]), ""])
    for member in entry.get("members", []):
        lines.extend(_entry_markdown(member, level + 1, module))
    return lines


def _module_markdown(catalog: dict[str, Any], module: str, level: int) -> list[str]:
    """The module docstring and the entries defined directly in ``module``."""
    rows = [
        entry for entry in catalog["api"] if entry["path"].rpartition(".")[0] == module
    ]
    if not rows and module not in catalog["modules"]:
        raise ValueError(f"unknown module {module!r}; the catalog has no rows in it")
    doc = catalog["modules"].get(module, "")
    lines = [docstring_markdown(doc), ""] if doc else []
    for entry in rows:
        lines.extend(_entry_markdown(entry, level, module))
    return lines


def render_overview(catalog: dict[str, Any]) -> str:
    """The execution rules and backends of a catalog, under a level-2 heading."""
    lines = ["## How native calls run", ""]
    lines.extend(f"- **{name}**: {rule}" for name, rule in catalog["rules"].items())
    lines.append("")
    for name, backend in catalog["backends"].items():
        details = "; ".join(
            f"{key} {', '.join(value) if isinstance(value, list) else value}"
            for key, value in backend.items()
            if key not in {"compiled", "runtime"}
        )
        lines.extend([f"Backend `{name}`: {details}.", ""])
    lines.extend(
        [
            "`support_catalog()` and `python -m treams_rs --format json` report the",
            "installed optional frameworks. An installed framework does not mean",
            "that treams-rs supports its devices: every adapter runs on the CPU.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_returned_types(catalog: dict[str, Any], *, level: int = 2) -> str:
    """The classes that public calls return but users never construct."""
    heading = "#" * level
    lines = [
        f"{heading} Python helper objects",
        "",
        "Private classes that public methods return, such as the solver of",
        "`TMatrix.interaction`.",
        "",
    ]
    for entry in catalog["returned_python_types"]:
        lines.extend(_entry_markdown(entry, level + 1, ""))
    lines.extend(
        [
            f"{heading} Native contexts and factors",
            "",
            "Objects of the compiled extension that records and factorizations",
            "return. Use them through the public calls that return them; their",
            "constructors belong to the private `treams_rs._native` module.",
            "",
        ]
    )
    for entry in catalog["native_types"]:
        lines.extend(_entry_markdown(entry, level + 1, ""))
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(catalog: dict[str, Any], *, module: str | None = None) -> str:
    """Render a catalog as the complete API reference or as one module's page.

    A module path such as ``"treams_rs.sw"`` renders that module's docstring and
    the ``api`` rows defined directly in it: top-level names under level-2
    headings relative to the module, their members under level-3 headings.
    With ``module=None`` the result holds the execution rules, every module
    (the package root first), and the returned Python and native types.
    Docstrings become Markdown through ``docstring_markdown``, and signatures
    sit in ``python`` fences.

    Pages leave out installed-machine availability fields and depend only on
    the documented API, not on source bytes, so private refactors leave them
    current (the JSON keeps the source hash).

    Raises:
        ValueError: ``module`` is neither a catalogued module nor the parent of
            an ``api`` row.
    """
    if module is not None:
        return "\n".join(_module_markdown(catalog, module, 2)).rstrip() + "\n"
    lines = [
        "# Python API reference",
        "",
        f"treams-rs {catalog['version']}, generated from the source and the native"
        " type stubs.",
        "",
        render_overview(catalog),
    ]
    for name in ["treams_rs", *catalog["modules"]]:
        lines.extend([f"## `{name}`", "", *_module_markdown(catalog, name, 3)])
    lines.append(render_returned_types(catalog))
    return "\n".join(lines).rstrip() + "\n"

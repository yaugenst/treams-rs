"""Offline API discovery from the installed source, without importing optional backends.

Signatures and pullback contracts come from Python source and native type stubs.
Backend availability reports compiled symbols and installed distributions; it
never initializes CUDA, imports an autodiff framework, or claims a working GPU.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from operator import itemgetter
from pathlib import Path
from typing import Any

from . import _native

__all__ = ["support_catalog"]


def _version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _signature(node: ast.FunctionDef) -> str:
    args = copy.copy(node.args)
    positional = args.posonlyargs + args.args
    if positional and positional[0].arg in {"self", "cls"}:
        if args.posonlyargs:
            args.posonlyargs = args.posonlyargs[1:]
        else:
            args.args = args.args[1:]
    result = f"({ast.unparse(args)})"
    if node.returns is not None:
        result += f" -> {ast.unparse(node.returns)}"
    return result


def _definitions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.ClassDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def _entry(
    path: str,
    node: ast.FunctionDef | ast.ClassDef,
    source: str,
    contexts: dict[str, ast.FunctionDef | ast.ClassDef],
    classes: dict[str, tuple[ast.ClassDef, str]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path,
        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
        "doc": ast.get_docstring(node) or "",
        "source": f"{source}:{node.lineno}",
    }
    if isinstance(node, ast.ClassDef):
        row["bases"] = [ast.unparse(base) for base in node.bases]
        members: dict[str, dict[str, Any]] = {}
        for base in node.bases:
            base = base.value if isinstance(base, ast.Subscript) else base
            if isinstance(base, ast.Name) and base.id in classes:
                parent, parent_source = classes[base.id]
                members.update(
                    {
                        member["path"]: member
                        for member in _entry(
                            path, parent, parent_source, contexts, classes
                        )["members"]
                    }
                )
        members.update(
            {
                f"{path}.{member.name}": _entry(
                    f"{path}.{member.name}", member, source, contexts, classes
                )
                for member in node.body
                if isinstance(member, ast.FunctionDef)
                and (not member.name.startswith("_") or member.name == "__init__")
            }
        )
        for index, attribute in enumerate(node.body):
            if isinstance(attribute, ast.AnnAssign):
                targets = [attribute.target]
                value = attribute.value
                annotation = ast.unparse(attribute.annotation)
            elif isinstance(attribute, ast.Assign):
                targets = attribute.targets
                value = attribute.value
                annotation = ""
            else:
                continue
            for target in targets:
                if not isinstance(target, ast.Name) or target.id.startswith("_"):
                    continue
                kind = "attribute"
                description = ""
                signature = f": {annotation}" if annotation else ""
                if value is not None:
                    signature += f" = {ast.unparse(value)}"
                if index + 1 < len(node.body):
                    following = node.body[index + 1]
                    if (
                        isinstance(following, ast.Expr)
                        and isinstance(following.value, ast.Constant)
                        and isinstance(following.value.value, str)
                    ):
                        description = following.value.value
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "OperatorAttribute"
                ):
                    kind = "descriptor"
                    operator_name = ast.unparse(value.args[0]).split(".")[-1]
                    operator, _ = classes[operator_name]
                    description = (
                        (ast.get_docstring(operator) or "")
                        + " Bound OperatorAttribute: call applies to the owning object; .eval returns the operator matrix."
                    )
                members[f"{path}.{target.id}"] = {
                    "path": f"{path}.{target.id}",
                    "kind": kind,
                    "signature": signature,
                    "doc": description,
                    "source": f"{source}:{attribute.lineno}",
                    "pullbacks": [],
                }
        row["members"] = list(members.values())
        return row
    row["signature"] = _signature(node)
    if any(isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list):
        row["kind"] = "property"
    pullbacks: list[dict[str, str]] = []
    row["pullbacks"] = pullbacks
    if node.returns is not None:
        for annotation in ast.walk(node.returns):
            name = (
                annotation.attr
                if isinstance(annotation, ast.Attribute)
                else (annotation.id if isinstance(annotation, ast.Name) else "")
            )
            context = contexts.get(name)
            if isinstance(context, ast.ClassDef):
                row["pullbacks"].extend(
                    {
                        "context": name,
                        "method": method.name,
                        "signature": _signature(method),
                        "doc": ast.get_docstring(method) or "",
                    }
                    for method in context.body
                    if isinstance(method, ast.FunctionDef)
                    and method.name.startswith("pullback")
                )
    return row


def _module_entries(
    stem: str,
    tree: ast.Module,
    contexts: dict[str, ast.FunctionDef | ast.ClassDef],
    classes: dict[str, tuple[ast.ClassDef, str]],
) -> list[dict[str, Any]]:
    entries = [
        _entry(f"treams_rs.{stem}.{name}", node, f"{stem}.py", contexts, classes)
        for name, node in _definitions(tree).items()
        if not name.startswith("_")
    ]
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "_native"
        ):
            continue
        native = getattr(_native, node.value.attr)
        try:
            signature = str(inspect.signature(native))
        except (TypeError, ValueError):
            signature = (
                (inspect.getdoc(native) or "")
                .split("\n")[0]
                .removeprefix(native.__name__)
            )
        entries.extend(
            {
                "path": f"treams_rs.{stem}.{target.id}",
                "kind": "ufunc" if hasattr(native, "nin") else "function",
                "signature": signature,
                "doc": inspect.getdoc(native) or "",
                "source": f"{stem}.py:{node.lineno}",
                "pullbacks": [],
            }
            for target in node.targets
            if isinstance(target, ast.Name) and not target.id.startswith("_")
        )
    # Same-module aliases reuse their target's contract (for example lattice.area).
    by_name = {entry["path"].rsplit(".", 1)[-1]: entry for entry in entries}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Name)
            and node.value.id in by_name
        ):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    entry = dict(by_name[node.value.id])
                    entry.update(
                        path=f"treams_rs.{stem}.{target.id}",
                        source=f"{stem}.py:{node.lineno}",
                        alias_of=by_name[node.value.id]["path"],
                    )
                    entries.append(entry)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and not node.target.id.startswith("_")
        ):
            entries.append(
                {
                    "path": f"treams_rs.{stem}.{node.target.id}",
                    "kind": "attribute",
                    "signature": f": {ast.unparse(node.annotation)}"
                    + (
                        f" = {ast.unparse(node.value)}"
                        if node.value is not None
                        else ""
                    ),
                    "doc": "",
                    "source": f"{stem}.py:{node.lineno}",
                    "pullbacks": [],
                }
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

    ``backends`` distinguishes build support from runtime availability. CUDA
    runtime availability is deliberately unprobed; construct ``cuda.Device`` to
    test the driver, libraries and device. Optional dependency versions only
    establish installation, not compatible devices or a configured framework.

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
    native = _definitions(ast.parse(sources["_native.pyi"]))
    contexts = native | _definitions(trees["iterative"])
    classes = {
        name: (node, f"{stem}.py")
        for stem, tree in trees.items()
        for name, node in _definitions(tree).items()
        if isinstance(node, ast.ClassDef)
    }
    entries = [
        entry
        for stem, tree in sorted(trees.items())
        if not stem.startswith("_")
        for entry in _module_entries(stem, tree, contexts, classes)
    ]
    root = trees["__init__"]
    exports = next(
        ast.literal_eval(node.value)
        for node in root.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    )
    for node in root.body:
        if isinstance(node, ast.ImportFrom) and node.module in trees:
            definitions = _definitions(trees[node.module])
            for alias in node.names:
                name = alias.asname or alias.name
                if name in exports and alias.name in definitions:
                    entries.append(
                        _entry(
                            f"treams_rs.{name}",
                            definitions[alias.name],
                            f"{node.module}.py",
                            contexts,
                            classes,
                        )
                    )
    from . import _OPTIONAL_MODULES

    optional = {
        name: _version(name) for name in _OPTIONAL_MODULES.values() if name is not None
    }
    return {
        "schema_version": 1,
        "version": _version("treams-rs"),
        "python_source_sha256": hashlib.sha256(
            b"".join(name.encode() + b"\0" + data for name, data in sources.items())
        ).hexdigest(),
        "contract": {
            "numeric_precision": "float64 / complex128; metadata also uses integer and boolean arrays",
            "native_pairing": "dL = Re(sum(conj(g) * dx))",
            "derivative_order": 1,
            "static_parameters": "mode counts, integer labels, topology; see each operation's docstring",
            "context_lifetime": "Treat native recorded contexts as single-use; record again for another pullback.",
            "python_execution": "CPU unless explicitly constructing treams_rs.cuda objects",
        },
        "backends": {
            "cpu": {
                "compiled": True,
                "runtime": "loaded",
                "precision": ["float64", "complex128"],
            },
            "cuda": {
                "compiled": hasattr(_native, "CudaDevice"),
                "runtime": "unprobed",
                "precision": ["complex128"],
                "scope": "explicit device matrices, products, reusable LU and solve pullbacks",
                "enable": "maturin develop --release --features cuda; then cuda.Device()",
            },
            "cuda_tile": {
                "compiled": hasattr(_native, "CudaPlaneWaves"),
                "runtime": "unprobed",
                "precision": ["complex128"],
                "scope": "weighted plane-wave electric fields at real points; no pullback",
                "enable": "Linux CUDA 13.3 toolkit; maturin develop --release --features cuda-tile",
            },
            "wasm": {
                "distribution": "separate JavaScript/TypeScript build, not a Python backend",
                "runtime": "not applicable to this Python installation",
                "scope": "layered/chiral spheres, interacting or independent finite clusters, illumination, direct incident and exterior scattered electric fields, fixed-target total-intensity radius/position gradient; bounded square sphere arrays with diffraction powers; lossless normal-incidence 1D crystal spectra, Bloch bands and internal fields",
                "limits": "serial; fixed-target geometry objective only, no generic pullbacks, arbitrary periodic systems or particle-interior fields; array diffraction thresholds excluded; no WebGPU; regular cluster expansion is not a total plane-wave field",
                "enable": "just wasm-check; see docs/wasm.md",
            },
        },
        "optional_dependencies": optional,
        "adapters": {
            "advect": {
                "device": "cpu",
                "derivative_order": 1,
                "extra": "advect",
                "contract": "native first-order VJPs; static labels captured by each wrapper",
            },
            "jax": {
                "device": "cpu",
                "derivative_order": 1,
                "extra": "jax",
                "contract": "jax_enable_x64; jit and sequential vmap; callbacks recompute reverse residuals; no JVP/higher AD",
            },
            "torch": {
                "device": "cpu",
                "derivative_order": 1,
                "extra": "torch",
                "contract": "float64/complex128; repeated backward recomputes residuals; no torch.func/compile/higher AD",
            },
        },
        "modules": {
            f"treams_rs.{stem}": ast.get_docstring(tree) or ""
            for stem, tree in sorted(trees.items())
            if not stem.startswith("_")
        },
        "returned_python_types": [
            _entry(name, node, source, contexts, classes)
            for name, (node, source) in classes.items()
            if name.startswith("_")
            and any(
                isinstance(function, ast.FunctionDef)
                and function.returns is not None
                and any(
                    isinstance(annotation, ast.Name) and annotation.id == name
                    for annotation in ast.walk(function.returns)
                )
                for tree in trees.values()
                for function in ast.walk(tree)
            )
        ],
        "native_types": [
            _entry(name, node, "_native.pyi", contexts, {})
            for name, node in native.items()
            if isinstance(node, ast.ClassDef)
        ],
        "api": sorted(entries, key=itemgetter("path")),
    }


def _markdown(catalog: dict[str, Any]) -> str:
    """Render the same catalog without installed-machine availability fields."""
    lines = [
        "# Python API reference",
        "",
        f"Generated from treams-rs {catalog['version']} source and native type stubs.",
        f"Python source/stub SHA-256: `{catalog['python_source_sha256']}`.",
        "Run `just docs` after changing the public API; do not edit this file.",
        "",
        "Native pullbacks use `Re(vdot(cotangent, tangent))` and are first order.",
        "An empty pullback list does not imply autodiff support. Capture static",
        "labels in a closure and use `diff` or a framework adapter for gradients.",
        "Inherited treams-rs members are included; inherited NumPy methods are omitted.",
        "",
    ]
    lines.extend(["## Execution contracts", ""])
    for name, description in catalog["contract"].items():
        lines.extend([f"- **{name}**: {description}"])
    lines.append("")
    for name, backend in catalog["backends"].items():
        lines.extend([f"### {name}", ""])
        for key, value in backend.items():
            if key not in {"compiled", "runtime"}:
                lines.append(f"- **{key}**: {value}")
        lines.append("")
    lines.extend(
        [
            "Installed dependency versions and compiled flags vary by wheel; inspect",
            "`support_catalog()` or JSON CLI output for this installation. Runtime GPU",
            "availability is unprobed. An installed framework does not establish device support.",
            "",
        ]
    )
    for name, doc in catalog["modules"].items():
        lines += [f"## {name}", "", doc, ""]

    def render(entry: dict[str, Any], level: int) -> None:
        lines.extend([f"{'#' * level} {entry['path']}", ""])
        if "signature" in entry:
            lines.extend(["```text", f"{entry['path']}{entry['signature']}", "```", ""])
        if entry["doc"]:
            lines.extend([entry["doc"], ""])
        for context in entry.get("pullbacks", []):
            lines.extend(
                [
                    "```text",
                    f"{context['context']}.{context['method']}{context['signature']}",
                    "```",
                    "",
                ]
            )
            if context["doc"]:
                lines.extend([context["doc"], ""])
        for member in entry.get("members", []):
            render(member, level + 1)

    for entry in catalog["api"]:
        render(entry, 2)
    for entry in catalog["returned_python_types"]:
        render(entry, 2)
    lines.extend(
        [
            "# Returned native types",
            "",
            "These opaque types are returned by public recording/factor APIs.",
            "Use their methods through those objects, rather than constructing `_native` internals.",
            "",
        ]
    )
    for entry in catalog["native_types"]:
        render(entry, 2)
    return "\n".join(lines).rstrip() + "\n"

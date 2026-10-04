"""Suite-wide pytest configuration.

pytest imports this initial conftest before any test module, so the Hypothesis
profile below applies to every property no matter which files are selected or in
which order they are collected. Hypothesis binds its settings when ``@given``
decorates a test, which is why the profile must be loaded here and never as a
side effect of importing a test module.

Every test also carries at least one category marker. ``CATEGORIES`` is the
only list of their names and descriptions: ``pytest_configure`` registers them,
and tests/test_suite_rules.py enforces the rule for everything collected.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import platform
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hypothesis import settings
from packaging.requirements import Requirement

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

# Every profile derives from the settings active at import: Hypothesis's own "ci"
# profile (derandomized, no example database) when it detects CI, otherwise its
# stock default. Keep every health check enabled, including CI's normally
# suppressed slow-generation check. Debug-build timings vary too much for a
# per-example deadline.
# Hosted CI selects "ci", which replaces Hypothesis's profile of that name and
# keeps the stock budget of 100 examples.
settings.register_profile(
    "dev",
    max_examples=30,
    deadline=None,
    print_blob=True,
    suppress_health_check=(),
)
settings.register_profile("ci", settings.get_profile("dev"), max_examples=100)
settings.register_profile("thorough", settings.get_profile("dev"), max_examples=300)
# `--hypothesis-profile=<name>` on the command line takes precedence.
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

#: Category marker names and descriptions; docs/development/testing.md explains them.
CATEGORIES = {
    "physics": "analytic and conservation-law validation",
    "gradients": "native pullback correctness",
    "interface": "Python interface behavior",
    "workflows": "complete scattering workflows",
    "reference": "comparison to independent numerical references",
}
_UNCATEGORIZED = pytest.StashKey[list[str]]()


def pytest_configure(config):
    """Register ``CATEGORIES`` before collection, which ``--strict-markers`` checks."""
    unavailable = _unavailable_requirements()
    if unavailable:
        config.stash[_POLICY] = _Policy(unavailable)
    for name, description in CATEGORIES.items():
        config.addinivalue_line("markers", f"{name}: {description}")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Record collected tests without a category before `-m` deselects any."""
    config.stash[_UNCATEGORIZED] = [
        item.nodeid
        for item in items
        if not any(item.get_closest_marker(name) for name in CATEGORIES)
    ]


@pytest.fixture
def category_markers():
    """Names of the category markers every test must carry at least one of."""
    return tuple(CATEGORIES)


@pytest.fixture
def uncategorized_tests(pytestconfig):
    """Node ids of collected tests that carry none of ``CATEGORIES``."""
    return pytestconfig.stash[_UNCATEGORIZED]


# Skip oracle tests only when pyproject.toml excludes the unavailable dependency
# on this interpreter. Missing supported oracles are errors. Imports in function
# bodies are deferred so shared helpers do not exclude independent tests.
ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_LOADERS = frozenset({"run_path", "spec_from_file_location"})
_DEFERRED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_IMPORT_ERRORS = frozenset(
    {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
)


@dataclass
class _Policy:
    """Interpreter-excluded dev requirements and the tests they removed."""

    unavailable: dict[str, str]
    ignored: dict[str, tuple[str, ...]] = field(default_factory=dict)
    skipped: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


_POLICY = pytest.StashKey[_Policy]()


def _import_name(requirement: Requirement) -> str:
    return requirement.name.lower().replace("-", "_").replace(".", "_")


def _dev_requirements() -> list[Requirement]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return [
        Requirement(entry)
        for entry in project["dependency-groups"]["dev"]
        if isinstance(entry, str)
    ]


def _unavailable_requirements() -> dict[str, str]:
    """Return import name -> reason for marker-excluded, absent dev packages.

    Raises ``pytest.UsageError`` when a marker-conditional dev package that this
    interpreter is required to have is missing.
    """
    applies: dict[str, bool] = {}
    excluded_by: dict[str, list[str]] = defaultdict(list)
    conditional: set[str] = set()
    for requirement in _dev_requirements():
        name = _import_name(requirement)
        if requirement.marker is None:
            applies[name] = True
            continue
        conditional.add(name)
        if requirement.marker.evaluate():
            applies[name] = True
        else:
            applies.setdefault(name, False)
            excluded_by[name].append(str(requirement))
    python = f"Python {platform.python_version()}"
    missing = sorted(
        name
        for name in conditional
        if applies[name] and importlib.util.find_spec(name) is None
    )
    if missing:
        raise pytest.UsageError(
            f"Test dependencies required on {python} are not installed: "
            f"{', '.join(missing)}. Install the dev group, e.g. "
            "`uv sync --locked --group dev`."
        )
    return {
        name: f"not installable on {python} ({'; '.join(excluded_by[name])})"
        for name in sorted(conditional)
        if not applies[name] and importlib.util.find_spec(name) is None
    }


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _guards_imports(node: ast.Try | ast.TryStar) -> bool:
    """Whether a ``try`` statement handles a failed import in its body."""
    for handler in node.handlers:
        if handler.type is None:
            return True
        types = (
            handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        )
        if any(
            isinstance(kind, ast.Name) and kind.id in _IMPORT_ERRORS for kind in types
        ):
            return True
    return False


def _executed_at_import(nodes: Iterable[ast.AST]) -> Generator[ast.AST]:
    """Yield nodes that run when a module body executes.

    Function bodies, ``if TYPE_CHECKING:`` blocks and the bodies of ``try``
    statements that handle an import failure are left out.
    """
    stack = [node for node in nodes if not isinstance(node, _DEFERRED)]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            children: Iterable[ast.AST] = node.orelse
        elif isinstance(node, ast.Try | ast.TryStar) and _guards_imports(node):
            children = [*node.handlers, *node.orelse, *node.finalbody]
        else:
            children = ast.iter_child_nodes(node)
        stack.extend(child for child in children if not isinstance(child, _DEFERRED))


def _script_paths(call: ast.Call, origin: Path) -> Generator[Path]:
    for node in ast.walk(call):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.endswith(".py")
        ):
            for base in (ROOT, ROOT / "scripts", origin.parent):
                if (candidate := (base / node.value).resolve()).is_file():
                    yield candidate
                    break


@cache
def _direct_imports(path: Path) -> tuple[frozenset[str], tuple[Path, ...]]:
    """Modules ``path`` imports at load time, and scripts it loads at that time."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    script_loaders = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "_scripts"
        for alias in node.names
        if alias.name == "load"
    }
    imported: set[str] = set()
    followed: list[Path] = []
    for node in _executed_at_import(tree.body):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.partition(".")[0])
        elif isinstance(node, ast.Call):
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else None
            if isinstance(function, ast.Name):
                name = function.id
            if name in _SCRIPT_LOADERS:
                followed.extend(_script_paths(node, path))
            elif name in script_loaders and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str
                ):
                    followed.append(ROOT / "scripts" / f"{argument.value}.py")
    # Tests and scripts may import helpers beside them or in the tests root.
    followed.extend(
        sibling
        for name in imported
        for base in (path.parent, ROOT / "tests")
        if (sibling := base / f"{name}.py").is_file()
    )
    return frozenset(imported), tuple(followed)


def _imports_at_load(path: Path) -> frozenset[str]:
    """Top-level module names imported while ``path`` is loaded as a module."""
    imported: set[str] = set()
    pending, seen = [path], set()
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            direct, followed = _direct_imports(current)
            imported |= direct
            pending.extend(followed)
    return frozenset(imported)


def _policy(config: pytest.Config) -> _Policy | None:
    return config.stash.get(_POLICY, None)


def pytest_report_header(config: pytest.Config) -> list[str]:
    policy = _policy(config)
    if policy is None:
        return []
    return [
        f"unavailable test dependency {name}: {reason}"
        for name, reason in policy.unavailable.items()
    ]


class _UnavailableModule(pytest.Module):
    """A test module whose load-time imports this interpreter cannot provide."""

    blocked: tuple[str, ...] = ()

    def collect(self) -> list[pytest.Item | pytest.Collector]:
        policy = _policy(self.config)
        assert policy is not None
        shown = self.nodeid or str(self.path)
        policy.ignored[shown] = self.blocked
        pytest.skip(
            f"{shown} not collected: "
            + "; ".join(
                f"imports {name}, {policy.unavailable[name]}" for name in self.blocked
            ),
            allow_module_level=True,
        )


def pytest_pycollect_makemodule(
    module_path: Path, parent: pytest.Collector
) -> pytest.Module | None:
    """Skip, without importing, modules that need an unavailable dependency.

    Only called for files matching ``python_files``; the module is reported when
    the session actually collects it.
    """
    policy = _policy(parent.config)
    if policy is None:
        return None
    blocked = sorted(
        _imports_at_load(module_path.resolve()) & policy.unavailable.keys()
    )
    if not blocked:
        return None
    module = _UnavailableModule.from_parent(parent, path=module_path)
    module.blocked = tuple(blocked)
    return module


def _unavailable_reason(item: pytest.Item, error: ModuleNotFoundError) -> str | None:
    policy = _policy(item.config)
    name = (error.name or "").partition(".")[0]
    if policy is None or name not in policy.unavailable:
        return None
    policy.skipped[name].append(item.nodeid)
    return f"{item.nodeid} requires {name}, {policy.unavailable[name]}"


# The skip is raised after the handler so that it carries no exception context:
# on Python 3.15, formatting the chained import failure emits a linecache
# DeprecationWarning that ``filterwarnings = error`` turns into an internal error.
@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None]:
    try:
        return (yield)
    except ModuleNotFoundError as error:
        if (reason := _unavailable_reason(item, error)) is None:
            raise
    pytest.skip(reason)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None]:
    try:
        return (yield)
    except ModuleNotFoundError as error:
        if (reason := _unavailable_reason(item, error)) is None:
            raise
    pytest.skip(reason)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    """Under pytest-xdist, merge what each worker reported into the controller."""
    output = getattr(node, "workeroutput", {})
    if output.get("rayon_global_pool_started"):
        _GLOBAL_POOL_UNUSED[:] = [False]
    policy = _policy(node.config)
    if policy is not None and "unavailable" in output:
        ignored, skipped = output["unavailable"]
        policy.ignored.update(ignored)
        for name, tests in skipped.items():
            policy.skipped[name].extend(tests)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    policy = _policy(terminalreporter.config)
    if policy is None or not (policy.ignored or policy.skipped):
        return
    python = f"{sys.implementation.name} {platform.python_version()}"
    terminalreporter.write_sep("=", f"test dependencies unavailable on {python}")
    for name, reason in policy.unavailable.items():
        modules = [path for path, names in policy.ignored.items() if name in names]
        tests = policy.skipped.get(name, [])
        terminalreporter.write_line(
            f"{name}: {reason}; {len(modules)} module(s) not collected, "
            f"{len(tests)} test(s) skipped"
        )
        for path in sorted(modules):
            terminalreporter.write_line(f"  not collected: {path}")
        for nodeid in tests:
            terminalreporter.write_line(f"  skipped: {nodeid}")


_GLOBAL_POOL_UNUSED: list[bool] = []


def pytest_sessionfinish(session, exitstatus):
    """Fail the session if native code started Rayon's global pool.

    Every parallel region must run on the treams-rs pool
    (``treams_core::threads``): the global pool cannot be resized, and a child
    forked after it started hangs. The probe starts that pool, so it runs once
    per interpreter, and only when the extension was loaded.
    """
    native = sys.modules.get("treams_rs._native")
    if not _GLOBAL_POOL_UNUSED and hasattr(native, "rayon_global_pool_unused"):
        _GLOBAL_POOL_UNUSED.append(native.rayon_global_pool_unused())
    # A pytest-xdist worker hands its results to the controller, which reports.
    if (output := getattr(session.config, "workeroutput", None)) is not None:
        output["rayon_global_pool_started"] = _GLOBAL_POOL_UNUSED == [False]
        if (policy := _policy(session.config)) is not None:
            output["unavailable"] = (policy.ignored, dict(policy.skipped))
        return
    if _GLOBAL_POOL_UNUSED == [False]:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "FAILED: native code started Rayon's global pool; run parallel "
                "work inside treams_core::threads::install",
                red=True,
            )

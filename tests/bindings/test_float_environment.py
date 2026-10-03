"""Native work keeps IEEE subnormals whatever the caller's floating-point mode.

Every native entry point runs its whole body in ``treams_core::fpenv::ieee``,
which restores the caller's mode on return; the workers of the treams-rs pool
clear the flushing modes as they start, so they keep subnormals too. scripts/float_environment.py
compares native results on a thread that flushes as XLA does with those of IEEE
callers; tests/autodiff/test_jax.py checks JAX callbacks.
"""

import ast
import math
import multiprocessing
import operator
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import treams_rs.special as sp
from treams_rs import _native

from _scripts import load
from _support import ROOT, assert_unitary_ports

environment = load("float_environment")

BINDINGS = ROOT / "crates" / "treams-py" / "src"
FN = re.compile(
    r'^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:(?:const|async|unsafe|extern\s+"[^"]*")\s+)*'
    r"fn\s+([\w$]+)"
)
C_LOOP = re.compile(r'^(\s*)(?:pub(?:\([^)]*\))?\s+)?unsafe extern "C" fn ([\w$]+)')
# Names of fns written out in the source, unlike the metavariables of macro
# templates (`fn $point`).
DEFINED = re.compile(r"\bfn\s+(\w+)")
# The words of the PyO3 attributes that export to Python, in any spelling of the
# attribute (`#[pymethods]`, `#[pyo3::pyfunction(name = ...)]`, inside
# `cfg_attr`). `_unscanned` reports any other use in code, such as an import
# that renames one.
EXPORT = re.compile(r"\b(pyfunction|pymethods)\b")
# Comments, string and character literals, which may hold brackets.
LITERALS = re.compile(r'//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\\\'])\'')
GUARDED = re.compile(r"ieee\((?:move\s+)?\|\|")


def _code(text):
    """``text`` without comments, and with empty string and character literals."""
    return LITERALS.sub(lambda m: "" if m[0].startswith("/") else m[0][0] * 2, text)


def _body(lines, start):
    """The non-blank lines of the body of the function declared at ``start``."""
    indent = len(lines[start]) - len(lines[start].lstrip())
    depth = 0
    for i in range(start, len(lines)):
        code = _code(lines[i])
        depth += code.count("(") - code.count(")")
        if depth == 0 and "{" in code:
            if code.rstrip().endswith("{"):
                end = lines.index(" " * indent + "}", i + 1)
                return [line.strip() for line in lines[i + 1 : end] if line.strip()]
            # An empty or one-line body stays on the line of its opening brace.
            return [code[code.index("{") + 1 : code.rindex("}")].strip()]
    raise AssertionError(f"no body after line {start + 1}")


def _guarded(body):
    """Whether ``body`` is one ``ieee(|| ...)`` call, which runs all of its work."""
    code = _code("\n".join(body)).strip().removesuffix(";")
    if not GUARDED.match(code):
        return False
    depth = 0
    for position, char in enumerate(code[4:], start=4):
        depth += (char == "(") - (char == ")")
        if depth == 0:
            return position == len(code) - 1
    return False


def _sources():
    """Lines of every binding source file, by path relative to ``BINDINGS``."""
    return {
        path.relative_to(BINDINGS).as_posix(): path.read_text().splitlines()
        for path in sorted(BINDINGS.rglob("*.rs"))
    }


class Entry(NamedTuple):
    """A fn that Python calls: a ``#[pyfunction]`` or a ``#[pymethods]`` fn."""

    kind: str  # "pyfunction" or "pymethods"
    name: str  # the Rust name; a metavariable such as `$point` in macro templates
    attribute: str  # "path:line" of the attribute that exports it
    location: str  # "path:line", for failure messages
    body: list[str]


def _entry_points(sources):
    """Every ``#[pyfunction]`` and every fn of a ``#[pymethods]`` block."""
    for path, lines in sources.items():
        function, methods, attribute = None, None, 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            code = _code(stripped)
            if attribute > 0:
                # The continuation of an attribute that rustfmt wrapped.
                attribute += code.count("[") - code.count("]")
            elif stripped.startswith("#["):
                attribute = code.count("[") - code.count("]")
                if export := EXPORT.search(code):
                    origin = f"{path}:{i + 1}"
                    if export[1] == "pyfunction":
                        function = origin
                    else:
                        methods = (len(line) - len(line.lstrip()), origin)
            elif methods is not None and line == " " * methods[0] + "}":
                methods = None
            elif match := FN.match(line):
                kind, origin = None, None
                if function is not None:
                    kind, origin = "pyfunction", function
                elif methods is not None and len(match[1]) == methods[0] + 4:
                    kind, origin = "pymethods", methods[1]
                if kind is not None:
                    location = f"{path}:{i + 1}"
                    yield Entry(kind, match[2], origin, location, _body(lines, i))
                function = None
            elif function is not None and stripped and not stripped.startswith("//"):
                # Checking a later fn instead could pass an unguarded binding.
                raise AssertionError(f"{path}:{i + 1}: no fn after #[pyfunction]")


def _unscanned(sources, entries):
    """Uses of ``pyfunction`` or ``pymethods`` in code that open none of ``entries``.

    Reported as "path:line: kind"; a use on a line that opens entries counts once.
    """
    mentions = Counter(
        f"{path}:{i + 1}: {kind}"
        for path, lines in sources.items()
        for i, line in enumerate(lines)
        for kind in EXPORT.findall(_code(line))
    )
    opened = Counter({f"{entry.attribute}: {entry.kind}" for entry in entries})
    return sorted((mentions - opened).elements())


def _declared():
    """Names of the functions and of the methods that the native stub declares.

    Read from the stub as ``float_environment.scalar_bindings`` reads it, without
    names that start with an underscore: test hooks and constructors bind under
    other Rust names.
    """
    stub = Path(_native.__file__).with_name("_native.pyi")
    tree = ast.parse(stub.read_text())

    def public(nodes):
        return {
            node.name
            for node in nodes
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        }

    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    return public(tree.body), set().union(*(public(node.body) for node in classes))


@pytest.mark.interface
def test_every_native_entry_point_runs_in_the_guard():
    sources = _sources()
    entries = list(_entry_points(sources))
    # Every exporting attribute, in any spelling, leads the scan to its entry
    # points: the fn after a #[pyfunction], the fns of a #[pymethods] block.
    assert _unscanned(sources, entries) == []
    # The scan finds every function and method of the stub that a binding
    # defines under that name, in whichever file. Macro templates define the
    # other functions under metavariables, and the scan finds those templates.
    defined = {
        name
        for lines in sources.values()
        for line in lines
        for name in DEFINED.findall(_code(line))
    }
    for kind, declared in zip(("pyfunction", "pymethods"), _declared(), strict=True):
        found = {entry.name for entry in entries if entry.kind == kind}
        assert sorted((declared & defined) - found) == [], kind
    templates = {"$point", "$vector", "$name"}
    assert templates <= {entry.name for entry in entries if entry.kind == "pyfunction"}
    # The body is one `ieee` call, so argument checks and all numerical work keep
    # subnormals. PyO3 converts arguments before it, in the caller's mode, and
    # by-value results after it: float64 and complex128 values as bit copies,
    # while float32 subnormals read as zero on flushing threads.
    unguarded = [entry for entry in entries if not _guarded(entry.body)]
    assert [f"{entry.location}: {entry.name}" for entry in unguarded] == []
    # NumPy calls the ufunc loops directly; each runs its kernel in `guard`.
    loops = [
        (f"{path}:{i + 1}: {match[2]}", _body(lines, i))
        for path, lines in sources.items()
        for i, line in enumerate(lines)
        if (match := C_LOOP.match(line))
    ]
    assert len(loops) >= 8
    assert [name for name, body in loops if "guard(" not in str(body)] == []
    [guard] = [
        _body(lines, i)
        for lines in sources.values()
        for i, line in enumerate(lines)
        if (match := FN.match(line)) and match[2] == "guard"
    ]
    assert _guarded(guard)
    # PyO3 would generate field accessors without a body to guard.
    text = "\n".join("\n".join(lines) for lines in sources.values())
    assert not re.search(r"#\[pyo3\((get|set)\b|\b(get|set)_all\b", text)


SCANNED = """\
#[pyfunction]
#[pyo3(signature = (
    a,
    b = "])",
))]
pub(super) fn wrapped(a: f64, b: &str) -> f64 {
    ieee(|| a * f64::from(u8::try_from(b.len()).unwrap_or(0)))
}

#[pyfunction]
fn before(a: f64) -> f64 {
    let b = 2.0 * a;
    ieee(|| b)
}

#[pyfunction]
fn after(a: f64) -> f64 {
    let b = ieee(|| a);
    2.0 * b
}

#[pymethods]
impl Context {
    fn moved(&mut self, a: f64) -> f64 {
        ieee(move || {
            // ) ieee(|| a)
            let text = ")";
            self.value * a + f64::from(u8::try_from(text.len()).unwrap_or(0))
        })
    }
}

#[pyo3::pyfunction(name = "spelled")]
fn qualified(a: f64) -> f64 {
    ieee(|| a)
}

#[pyo3::pymethods]
impl Record {
    fn unguarded(&self) -> f64 {
        self.value
    }
}
"""


@pytest.mark.interface
def test_the_entry_point_scan_accepts_only_bodies_that_run_in_the_guard():
    # rustfmt wraps long attributes over several lines; comments and literals
    # may hold brackets; attributes may name their path; work before or after
    # the `ieee` call does not count.
    sources = {"scanned.rs": SCANNED.splitlines()}
    entries = list(_entry_points(sources))
    guarded = {entry.name: _guarded(entry.body) for entry in entries}
    assert guarded == {
        "wrapped": True,
        "before": False,
        "after": False,
        "moved": True,
        "qualified": True,
        "unguarded": False,
    }
    assert _unscanned(sources, entries) == []
    with pytest.raises(AssertionError, match="no fn after"):
        list(_entry_points({"split.rs": ["#[pyfunction]", "pub", "fn split() {}"]}))
    # An attribute under another name, or a block without methods, hides its
    # entry points from the scan.
    sources = {
        "renamed.rs": ["use pyo3::pyfunction as export;", "#[export]", "fn f() {}"],
        "empty.rs": ["#[pymethods]", "impl Empty {}"],
    }
    assert _unscanned(sources, list(_entry_points(sources))) == [
        "empty.rs:1: pymethods",
        "renamed.rs:1: pyfunction",
    ]


@pytest.fixture
def flushing():
    """Skip where the test hook cannot flush subnormals (only x86-64 and AArch64)."""
    if not environment.supported():
        pytest.skip("this platform cannot flush subnormals to zero")


@pytest.mark.interface
def test_flush_sensitivity_sees_subnormal_results_and_other_values_from_zeros():
    def sensitive(function, *operands):
        return environment.sensitive(environment.Call(function, *operands))

    assert sensitive(operator.mul, 3e-310, 1.0)  # a subnormal result
    assert sensitive(operator.eq, 3e-310, 0.0)
    assert sensitive(np.multiply, np.array([3e-310j]), np.array([1e10]))
    assert not sensitive(operator.add, 1.0, 3e-310)  # rounded away
    assert not sensitive(math.copysign, 1.0, -3e-310)  # zeros keep their sign
    assert not sensitive(operator.truediv, 1.0, 3e-310)  # zero raises instead
    # Results flush too: zeros give a subnormal result, which flushes to 0.
    assert not sensitive(lambda x: 3e-310 if x == 0.0 else 0.0, 3e-310)


@pytest.mark.interface
@pytest.mark.usefixtures("flushing")
def test_every_scalar_binding_gets_the_results_of_ieee_callers():
    # Bindings that take and return register values, enumerated from the stub,
    # with subnormal, tiny and ordinary arguments; each returns values for some
    # calls and, with float operands, has calls that flushing is predicted to
    # change.
    bindings = environment.scalar_bindings()
    assert {
        "incgamma_scalar",
        "intkambe_scalar",
        "hankel_scalar",
        "lpmv_real_scalar",
    } <= set(bindings)
    environment.check_drawn(environment.scalar_cases(0, 48))


@pytest.mark.interface
@pytest.mark.usefixtures("flushing")
def test_every_ufunc_loop_gets_the_results_of_ieee_callers():
    loops = environment.ufunc_cases(0, 48)
    assert {"spherical_jn[dD->D]", "lsumsw3d[ddDdddD->D]", "vsw_A[llDDdl->D]"} <= set(
        loops
    )
    environment.check_drawn(loops)
    # The witness of the exempt loop passes through the subnormal square of
    # k = 1e-160, which a kernel that flushes would return as 0.
    loop = "wave_vector_z[ddd->D]"
    kx, ky, k = witness = environment.INSENSITIVE[loop]
    assert kx == ky == 0 and 0 < k * k < sys.float_info.min
    assert_array_equal(loops[loop][f"{loop}{witness!r}"](), [math.sqrt(k * k)])


def _subnormal(parts):
    parts = np.abs(np.ravel(parts))
    return (parts > 0) & (parts < sys.float_info.min)


def _all_subnormal(values):
    """Every value has a nonzero subnormal magnitude."""
    for value in values:
        assert np.all(_subnormal(np.abs(value))), value


def _some_subnormal(values):
    """Every value has a nonzero subnormal real or imaginary part."""
    for value in values:
        parts = np.concatenate([np.ravel(np.real(value)), np.ravel(np.imag(value))])
        assert np.any(_subnormal(parts)), value


def _check_scalars(values):
    *plain, bessel, gamma = values
    _all_subnormal([*plain, bessel[0], gamma[0]])


def _check_solves(values):
    *scalars, (value, _) = values
    assert_allclose([x[0][0, 0] for x in scalars] * environment.PIVOTS, 1, rtol=1e-15)
    assert_allclose(environment.OPERATOR @ value, np.eye(3), atol=1e-15)


def _check_slab(values):
    assert_unitary_ports(values[0], atol=1e-13)


# Checks of the values of IEEE callers, which flushing callers get bit for bit.
FAMILIES = {
    "bessel": _all_subnormal,
    "scalars": _check_scalars,
    "wrappers": _some_subnormal,
    "records": lambda values: _all_subnormal(value[0] for value in values[:2]),
    "solve": _check_solves,
    "slab": _check_slab,
}


@pytest.mark.interface
@pytest.mark.usefixtures("flushing")
@pytest.mark.parametrize("family", FAMILIES)
def test_flushing_callers_get_the_native_results_of_ieee_callers(family):
    families = environment.family_cases()
    assert set(families) == set(FAMILIES)
    environment.check_family({family: families[family]})
    FAMILIES[family]([call() for call in families[family].values()])


FLUSHING_FIRST = """
import sys

import numpy as np
import torch

assert torch.set_flush_denormal(True)
import treams_rs as core
from treams_rs import special

ports = core.PlaneWavePorts.default([[0.1 + 0.01 * i, 0.05] for i in range(16)])
material = core.Material(5.0, 1.0, 0.0)
slab = core.slab(k0=1.4375, basis=ports, thickness=0.234375, material=material)
bessel = special.spherical_jn(10, np.array([1e-31]))
tiny, two = sys.float_info.min, 2.0
flushing = tiny / two == 0.0
torch.set_flush_denormal(False)
np.savez(sys.argv[1], slab=slab.array, bessel=bessel, flushing=flushing)
"""


@pytest.mark.interface
def test_first_native_call_on_a_flushing_thread_keeps_subnormals(tmp_path):
    # The treams-rs pool of a fresh process starts inside the first native call
    # that uses it, here on a thread that flushes. Workers that inherited the
    # mode would flush the LU pivots of every channel and break unitarity.
    pytest.importorskip("torch")
    path = tmp_path / "results.npz"
    subprocess.run(
        [sys.executable, "-c", FLUSHING_FIRST, str(path)], check=True, timeout=600
    )
    results = np.load(path)
    assert results["flushing"], "native calls restore the caller's mode"
    assert_unitary_ports(results["slab"], atol=1e-13)
    assert_array_equal(results["bessel"], sp.spherical_jn(10, np.array([1e-31])))
    assert results["bessel"][0] != 0


FORK_AFTER_NATIVE_CALLS = """
import multiprocessing
import sys

import numpy as np
import treams_rs as core
from treams_rs import special

core.Lattice.square(1.0)
special.wigner3j(1, 1, 2, 0, 0, 0)
special.incgamma(0.5, 1.0)
parent = special.spherical_jn(3, np.linspace(0.1, 10, 200_000))


def child():
    values = special.spherical_jn(3, np.linspace(0.1, 10, 200_000))
    sys.exit(0 if np.array_equal(values, parent) else 1)


process = multiprocessing.get_context("fork").Process(target=child)
process.start()
process.join(timeout=60)
if process.is_alive():
    process.kill()
    sys.exit("the forked child hung in its first parallel native call")
sys.exit(process.exitcode)
"""


@pytest.mark.interface
def test_forked_children_compute_in_parallel_after_native_calls():
    # Pool threads do not survive a fork. A child forked after serial and
    # parallel native calls builds a pool of its own instead of waiting on the
    # parent's workers, and its workers keep subnormals.
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("this platform cannot fork")
    environment = dict(os.environ, TREAMS_RS_NUM_THREADS="2")
    subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore::DeprecationWarning",
            "-c",
            FORK_AFTER_NATIVE_CALLS,
        ],
        check=True,
        timeout=600,
        env=environment,
    )

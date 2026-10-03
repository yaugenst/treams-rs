# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.1,<3"]
# ///
"""Compare two treams_rs package trees call by call, for speed and agreement.

Run after `just build-ext-release` on an otherwise idle host, normally through
`just bench-compare`, or as

    uv run --no-sync python scripts/compare_builds.py --baseline-ref main

The baseline is either a directory that holds a built ``treams_rs`` package
(``--baseline``), or the ``python/treams_rs`` sources of a git ref
(``--baseline-ref``), which run on the candidate's native library; the latter
needs the native sources (``crates/``, ``Cargo.lock``) of the ref and of the
working tree to agree. The candidate is ``python/`` of this checkout unless
``--candidate`` names another directory.

Each round starts one worker process per tree, with the same pinned thread
counts, and the two workers then time each selected public call in turn, in
ABBA order from call to call, so that the two measurements of a call are
seconds apart. A measurement repeats the call until it lasts at least
``--min-sample`` seconds and keeps the fastest of ``--samples`` such samples.
The first round also records each call's values, and the trees must agree to
``--rtol`` before any timing counts. The ratio of a call is the median over
rounds of candidate/baseline; it may not exceed 1.05 for calls of at least one
millisecond, nor 1.10 below. ``--memory`` adds one process per tree and call
that reports the call's growth of the peak resident memory; calls that grow it
by at least 16 MiB in the baseline may not grow it by more than 5%.

Calls whose optional framework (advect, jax, torch) is missing are skipped. The
report goes to benchmarks/results/local/, which git ignores; the exit status is
nonzero when any call disagrees or misses its limit.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import ExitStack, contextmanager, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Any

from _harness import file_sha256, peak_rss_mib, pinned_threads, python_source_sha256

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks/results/local/compare-builds.json"
NATIVE_SOURCES = ("crates", "Cargo.lock", "Cargo.toml")
#: Ratio limits of the median over rounds, for calls below and above one millisecond.
SLOW_LIMIT, FAST_LIMIT, FAST_SECONDS = 1.05, 1.10, 1e-3
#: Peak-memory growth limit, applied from this baseline growth up.
MEMORY_LIMIT, MEMORY_FLOOR_MIB = 1.05, 16.0

# Calls ------------------------------------------------------------------------------
# A call is a setup function of the treams_rs module (or of an optional framework
# namespace); it returns the timed zero-argument function. Setup is not timed.

CALLS: dict[str, Callable[[Any], Callable[[], Any]]] = {}
#: Optional framework namespace each call needs, if any.
FRAMEWORK: dict[str, str] = {}


def call(name: str, framework: str | None = None):
    def register(setup: Callable[[Any], Callable[[], Any]]):
        CALLS[name] = setup
        if framework is not None:
            FRAMEWORK[name] = framework
        return setup

    return register


def _plane(tr: Any) -> Any:
    return tr.plane_wave([0.1, 0.2, 1.0], "positive_helicity", k0=1.3)


def _sphere(tr: Any, lmax: int) -> Any:
    return tr.sphere_tmatrix(k0=1.3, lmax=lmax, radius=0.25, material=4 + 0.1j)


def _cluster(tr: Any, count: int) -> Any:
    particles = [
        tr.sphere_tmatrix(
            k0=1.3, lmax=3, radius=0.15 + 0.1 * i / max(count - 1, 1), material=4 + 0.1j
        )
        for i in range(count)
    ]
    positions = [[0.0, 0.0, 0.8 * i] for i in range(count)]
    return tr.Cluster(particles, positions=positions)


def _ports(tr: Any, modes: int) -> Any:
    groups = max(modes // 2, 1)
    q = [[0.1 + 0.7 * i / groups, 0.2] for i in range(groups)]
    return tr.PlaneWavePorts.default(q)


def _amplitudes(modes: int) -> list[complex]:
    return [1.0 if i == 0 else 0.0 for i in range(modes)]


FIELD_POINTS = {
    1: [[0.5, 0.6, 0.7]],
    4096: [
        [0.5 + 1.5 * t, 0.6 + 0.9 * t, 0.7 + 2.3 * t]
        for t in (i / 4095 for i in range(4096))
    ],
}

for _lmax in (1, 3, 10, 20):

    @call(f"sphere-scatter-l{_lmax}")
    def _scatter(tr, lmax=_lmax):
        tm, wave = _sphere(tr, lmax), _plane(tr)
        return lambda: tm.scatter(wave).coefficients

    @call(f"sphere-cross-sections-l{_lmax}")
    def _cross_sections(tr, lmax=_lmax):
        tm, wave = _sphere(tr, lmax), _plane(tr)
        return lambda: tuple(tm.cross_sections(wave))


for _lmax, _count in ((3, 1), (3, 4096), (10, 4096)):

    @call(f"sphere-efield-l{_lmax}-p{_count}")
    def _efield(tr, lmax=_lmax, count=_count):
        scattered = _sphere(tr, lmax).scatter(_plane(tr))
        return lambda: scattered.efield(FIELD_POINTS[count])


for _mmax in (1, 3, 12):

    @call(f"cylinder-cross-widths-m{_mmax}")
    def _cross_widths(tr, mmax=_mmax):
        tm = tr.cylinder_tmatrix(k0=1.3, kz=0.2, mmax=mmax, radius=0.2, material=3.0)
        wave = tr.plane_wave([1.0, 0.0, 0.2], "positive_helicity", k0=1.3)
        return lambda: tuple(tm.cross_widths(wave))


for _count in (2, 4, 16, 32, 64):

    @call(f"cluster-solve-n{_count}")
    def _solve(tr, count=_count):
        cluster = _cluster(tr, count)
        return lambda: cluster.solve().array

    @call(f"cluster-scatter-n{_count}")
    def _cluster_scatter(tr, count=_count):
        cluster, wave = _cluster(tr, count), _plane(tr)
        return lambda: cluster.scatter(wave).coefficients


for _count in (4, 16):

    @call(f"cluster-cross-sections-n{_count}")
    def _cluster_xs(tr, count=_count):
        solved, wave = _cluster(tr, count).solve(), _plane(tr)
        return lambda: tuple(solved.cross_sections(wave))


for _modes in (2, 16, 64, 256):
    for _polarization in ("helicity", "parity"):
        _suffix = f"{_polarization[0]}{_modes}"

        @call(f"interface-{_suffix}")
        def _interface(tr, modes=_modes, polarization=_polarization):
            ports = _ports(tr, modes)
            return lambda: (
                tr.interface(
                    k0=1.3,
                    basis=ports,
                    negative_medium=1.0,
                    positive_medium=2.3 + 0.1j,
                    polarization=polarization,
                ).array
            )

        @call(f"slab-{_suffix}")
        def _slab(tr, modes=_modes, polarization=_polarization):
            ports = _ports(tr, modes)
            return lambda: (
                tr.slab(
                    k0=1.3,
                    basis=ports,
                    thickness=0.3,
                    material=2.3 + 0.1j,
                    polarization=polarization,
                ).array
            )

        @call(f"propagation-{_suffix}")
        def _propagation(tr, modes=_modes, polarization=_polarization):
            ports = _ports(tr, modes)
            return lambda: (
                tr.propagation(
                    distance=0.3, basis=ports, k0=1.3, polarization=polarization
                ).array
            )

    @call(f"multilayer-slab-h{_modes}")
    def _multilayer(tr, modes=_modes):
        ports = _ports(tr, modes)
        return lambda: (
            tr.multilayer_slab(
                k0=1.3,
                basis=ports,
                thicknesses=[0.1, 0.2, 0.3, 0.4],
                materials=[2.3 + 0.1j, 1.7 + 0.05j, 2.3 + 0.1j, 1.7 + 0.05j],
            ).array
        )

    @call(f"stack-power-h{_modes}")
    def _stack_power(tr, modes=_modes):
        ports = _ports(tr, modes)
        lower = tr.slab(k0=1.3, basis=ports, thickness=0.3, material=2.3 + 0.1j)
        gap = tr.propagation(distance=0.2, basis=ports, k0=1.3)
        amplitudes = _amplitudes(modes)
        return lambda: tuple(tr.stack([lower, gap, lower]).power(amplitudes))


for _modes in (2, 16, 64):

    @call(f"bands-h{_modes}")
    def _bands(tr, modes=_modes):
        layer = tr.slab(k0=1.3, basis=_ports(tr, modes), thickness=0.3, material=2.0)
        return lambda: tuple(layer.bands(period=1.2))


@call("periodic-to-smatrix-l3")
def _periodic(tr):
    tm = _sphere(tr, 3)
    ports = tr.PlaneWavePorts.default([[0.1, 0.05]])

    def run():
        response = tr.solve_periodic(
            tm, lattice=[[1.5, 0.0], [0.0, 1.5]], kpar=[0.1, 0.05]
        )
        return response.to_smatrix(ports).array

    return run


# Framework calls: value and gradient of one real objective in each namespace.


def _gradient(framework: str, namespace: Any, objective: Callable, x0: float):
    """Zero-argument value-and-gradient of ``objective(namespace, x)`` at ``x0``."""
    if framework == "jax":
        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", True)
        compiled = jax.jit(jax.value_and_grad(functools.partial(objective, namespace)))
        x = jax.numpy.asarray(x0)
        return lambda: tuple(float(v) for v in compiled(x))
    if framework == "torch":
        torch = importlib.import_module("torch")

        def run():
            x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
            value = objective(namespace, x)
            (gradient,) = torch.autograd.grad(value, x)
            return float(value.detach()), float(gradient)

        return run
    advect = importlib.import_module("advect")
    function = advect.value_and_grad(functools.partial(objective, namespace))
    return lambda: tuple(float(v) for v in function(x0))


def _objective_cluster(count: int, solve: bool) -> Callable:
    def objective(tr, radius):
        particles = [
            tr.sphere_tmatrix(
                k0=1.3, lmax=3, radius=radius if i == 0 else 0.2, material=4 + 0.1j
            )
            for i in range(count)
        ]
        cluster = tr.Cluster(
            particles, positions=[[0, 0, 0.8 * i] for i in range(count)]
        )
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
        scattered = (cluster.solve() if solve else cluster).scatter(wave)
        return (abs(scattered.coefficients) ** 2).sum()

    return objective


def _objective_cross_sections(lmax: int) -> Callable:
    def objective(tr, radius):
        tm = tr.sphere_tmatrix(k0=1.3, lmax=lmax, radius=radius, material=4 + 0.1j)
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
        return tm.cross_sections(wave).scattering

    return objective


def _objective_layers(kind: str, modes: int, polarization: str) -> Callable:
    def objective(tr, thickness):
        ports = _ports(importlib.import_module("treams_rs"), modes)
        if kind == "slab":
            network = tr.slab(
                k0=1.3,
                basis=ports,
                thickness=thickness,
                material=2.3 + 0.1j,
                polarization=polarization,
            )
        else:
            network = tr.propagation(
                distance=thickness, basis=ports, k0=1.3, polarization=polarization
            )
        return (abs(network.array) ** 2).sum()

    return objective


FRAMEWORK_OBJECTIVES = {
    "cluster-solve-n4": (_objective_cluster(4, True), 0.2),
    "cluster-solve-n16": (_objective_cluster(16, True), 0.2),
    "cluster-scatter-n16": (_objective_cluster(16, False), 0.2),
    "cross-sections-l3": (_objective_cross_sections(3), 0.25),
    "slab-h16": (_objective_layers("slab", 16, "helicity"), 0.3),
    "slab-p2": (_objective_layers("slab", 2, "parity"), 0.3),
    "slab-p64": (_objective_layers("slab", 64, "parity"), 0.3),
    "slab-p256": (_objective_layers("slab", 256, "parity"), 0.3),
    "propagation-h2": (_objective_layers("propagation", 2, "helicity"), 0.3),
    "propagation-p2": (_objective_layers("propagation", 2, "parity"), 0.3),
    "propagation-p64": (_objective_layers("propagation", 64, "parity"), 0.3),
}

for _framework in ("advect", "jax", "torch"):
    for _name, (_objective, _x0) in FRAMEWORK_OBJECTIVES.items():

        @call(f"{_framework}-grad-{_name}", framework=_framework)
        def _framework_call(tr, framework=_framework, objective=_objective, x0=_x0):
            namespace = importlib.import_module(f"treams_rs.{framework}")
            return _gradient(framework, namespace, objective, x0)


# Workers ----------------------------------------------------------------------------


def values(result: Any) -> list[complex]:
    """Flat complex values of a call result (arrays, tuples and scalars)."""
    import numpy as np

    if isinstance(result, tuple):
        return [v for item in result for v in values(item)]
    return [complex(v) for v in np.asarray(result, dtype=np.complex128).ravel()]


def available(name: str) -> bool:
    framework = FRAMEWORK.get(name)
    if framework is None:
        return True
    try:
        importlib.import_module(framework)
    except ImportError:
        return False
    return True


def time_call(function: Callable[[], Any], min_sample: float, samples: int) -> float:
    """Fastest seconds per call over ``samples`` samples of at least ``min_sample``."""
    start = time.perf_counter()
    function()
    single = time.perf_counter() - start
    number = max(1, math.ceil(min_sample / max(single, 1e-9)))
    seconds = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(number):
            function()
        seconds.append((time.perf_counter() - start) / number)
    return min(seconds)


def serve(tree: Path, min_sample: float, samples: int) -> None:
    """Answer one JSON request per input line: time (and record) the named call."""
    import gc

    with redirect_stdout(sys.stderr):
        tr = importlib.import_module("treams_rs")
    if not Path(tr.__file__).resolve().is_relative_to(tree.resolve()):
        raise RuntimeError(f"treams_rs resolved outside {tree}")
    functions: dict[str, Callable[[], Any]] = {}
    for line in sys.stdin:
        request = json.loads(line)
        name = request["name"]
        with redirect_stdout(sys.stderr):
            if name not in functions:
                functions[name] = CALLS[name](tr)
            function = functions[name]
            entry: dict[str, Any] = {}
            if request["record"]:
                entry["values"] = [[v.real, v.imag] for v in values(function())]
            gc.disable()
            try:
                entry["seconds"] = time_call(function, min_sample, samples)
            finally:
                gc.enable()
        print(json.dumps(entry), flush=True)


def run_memory_worker(name: str, tree: Path) -> dict:
    tr = importlib.import_module("treams_rs")
    if not Path(tr.__file__).resolve().is_relative_to(tree.resolve()):
        raise RuntimeError(f"treams_rs resolved outside {tree}")
    function = CALLS[name](tr)
    before = peak_rss_mib()
    function()
    return {"growth_mib": peak_rss_mib() - before}


# Driver -----------------------------------------------------------------------------


def abba(count: int) -> Iterator[tuple[str, str]]:
    """Tree order of each step: baseline first at even steps."""
    for index in range(count):
        yield ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")


def environment(tree: Path, threads: int) -> dict[str, str]:
    return {
        **os.environ,
        **pinned_threads(threads),
        "PYTHONPATH": str(tree),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


class Worker:
    """A serving worker process of one tree."""

    def __init__(self, tree: Path, arguments: argparse.Namespace):
        self.tree = tree
        self.process = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--serve",
                "--tree",
                str(tree),
                "--min-sample",
                str(arguments.min_sample),
                "--samples",
                str(arguments.samples),
            ],
            env=environment(tree, arguments.threads),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def measure(self, name: str, record: bool) -> dict:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps({"name": name, "record": record}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"worker for {self.tree} stopped at {name}")
        return json.loads(line)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        self.process.wait()


def run_round(
    trees: dict[str, Path], names: list[str], arguments: argparse.Namespace, index: int
) -> dict[str, dict]:
    """One worker per tree; each call measured by both, in alternating order."""
    measured: dict[str, dict] = {label: {} for label in trees}
    with ExitStack() as cleanup:
        workers = {}
        for label, tree in trees.items():
            worker = Worker(tree, arguments)
            cleanup.callback(worker.close)
            workers[label] = worker
        # The order alternates from call to call and flips from round to round.
        orders = list(abba(len(names) + index))[index:]
        for name, order in zip(names, orders, strict=True):
            for label in order:
                measured[label][name] = workers[label].measure(name, index == 0)
    return measured


def measure_memory(tree: Path, name: str, threads: int) -> float:
    completed = subprocess.run(
        [sys.executable, __file__, "--memory-call", name, "--tree", str(tree)],
        env=environment(tree, threads),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"worker for {tree} failed:\n{completed.stderr[-4000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])["growth_mib"]


def agreement(baseline: list, candidate: list, rtol: float) -> float | None:
    """Largest difference relative to the largest magnitude, or None on a shape change."""
    if len(baseline) != len(candidate):
        return None
    if not baseline:
        return 0.0
    scale = max(math.hypot(*v) for v in baseline) or 1.0
    return (
        max(
            math.hypot(a[0] - b[0], a[1] - b[1])
            for a, b in zip(baseline, candidate, strict=True)
        )
        / scale
    )


def verdict(baseline_seconds: float, ratio: float) -> bool:
    limit = SLOW_LIMIT if baseline_seconds >= FAST_SECONDS else FAST_LIMIT
    return ratio <= limit


def summarize(rounds: list[dict[str, dict]], names: list[str], rtol: float) -> dict:
    """Per-call medians, paired ratios, agreement and pass/fail."""
    calls = {}
    first = rounds[0]
    for name in names:
        baseline = [r["baseline"][name]["seconds"] for r in rounds]
        candidate = [r["candidate"][name]["seconds"] for r in rounds]
        ratios = [c / b for b, c in zip(baseline, candidate, strict=True)]
        ratio = statistics.median(ratios)
        difference = agreement(
            first["baseline"][name]["values"], first["candidate"][name]["values"], rtol
        )
        agrees = difference is not None and difference <= rtol
        median_baseline = statistics.median(baseline)
        calls[name] = {
            "baseline_seconds": median_baseline,
            "candidate_seconds": statistics.median(candidate),
            "ratio": ratio,
            "ratio_range": [min(ratios), max(ratios)],
            "relative_difference": difference,
            "agrees": agrees,
            "passed": agrees and verdict(median_baseline, ratio),
        }
    return calls


def memory_verdict(baseline_mib: float, candidate_mib: float) -> bool:
    if baseline_mib < MEMORY_FLOOR_MIB:
        return True
    return candidate_mib <= MEMORY_LIMIT * baseline_mib


def native_library(tree: Path) -> Path:
    found = sorted((tree / "treams_rs").glob("_native*"))
    found = [path for path in found if path.suffix in (".so", ".pyd", ".dylib")]
    if len(found) != 1:
        raise SystemExit(f"expected one built native library in {tree / 'treams_rs'}")
    return found[0]


@contextmanager
def baseline_from_ref(ref: str, candidate: Path) -> Iterator[Path]:
    """``python/treams_rs`` of ``ref`` with the candidate's native library."""
    changed = subprocess.run(
        ["git", "diff", "--quiet", ref, "--", *NATIVE_SOURCES], cwd=ROOT, check=False
    )
    if changed.returncode == 1:
        raise SystemExit(
            f"the native sources differ from {ref}; build that ref and pass --baseline"
        )
    if changed.returncode:
        raise SystemExit(f"git cannot compare with {ref}")
    with tempfile.TemporaryDirectory(prefix="compare-builds-") as directory:
        archive = Path(directory) / "baseline.tar"
        subprocess.run(
            ["git", "archive", "--output", str(archive), ref, "python/treams_rs"],
            cwd=ROOT,
            check=True,
        )
        with tarfile.open(archive) as stream:
            stream.extractall(directory, filter="data")
        tree = Path(directory) / "python"
        library = native_library(candidate)
        (tree / "treams_rs" / library.name).write_bytes(library.read_bytes())
        yield tree


def provenance(tree: Path) -> dict:
    return {
        "native_sha256": file_sha256(native_library(tree)),
        "python_sha256": python_source_sha256(tree / "treams_rs"),
    }


def compare(arguments: argparse.Namespace, baseline: Path, candidate: Path) -> dict:
    pattern = re.compile(arguments.match) if arguments.match else None
    names = [name for name in CALLS if pattern is None or pattern.search(name)]
    skipped = [name for name in names if not available(name)]
    names = [name for name in names if name not in skipped]
    if not names:
        raise SystemExit("no call selected")
    trees = {"baseline": baseline, "candidate": candidate}
    rounds = []
    for index in range(arguments.rounds):
        rounds.append(run_round(trees, names, arguments, index))
        print(f"round {index + 1}/{arguments.rounds} done", file=sys.stderr)
    calls = summarize(rounds, names, arguments.rtol)
    if arguments.memory:
        for name in names:
            growth = [
                measure_memory(trees[label], name, arguments.threads) for label in trees
            ]
            calls[name]["memory_mib"] = growth
            calls[name]["passed"] &= memory_verdict(*growth)
    return {
        "baseline": provenance(baseline),
        "candidate": provenance(candidate),
        "baseline_ref": arguments.baseline_ref,
        "host": platform.node(),
        "threads": arguments.threads,
        "rounds": arguments.rounds,
        "samples": arguments.samples,
        "min_sample_seconds": arguments.min_sample,
        "rtol": arguments.rtol,
        "skipped": skipped,
        "calls": calls,
    }


def report(result: dict) -> str:
    lines = [f"{'call':40s} {'baseline':>12s} {'candidate':>12s} {'ratio':>7s}  result"]
    for name, entry in result["calls"].items():
        outcome = "pass" if entry["passed"] else "FAIL"
        if not entry["agrees"]:
            outcome = f"FAIL (values differ: {entry['relative_difference']})"
        lines.append(
            f"{name:40s} {entry['baseline_seconds'] * 1e6:10.1f}us "
            f"{entry['candidate_seconds'] * 1e6:10.1f}us {entry['ratio']:7.3f}  {outcome}"
        )
    if result["skipped"]:
        lines.append(f"skipped (framework missing): {', '.join(result['skipped'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--baseline", type=Path, help="Directory holding treams_rs/")
    source.add_argument("--baseline-ref", help="Git ref of the baseline Python sources")
    parser.add_argument("--candidate", type=Path, default=ROOT / "python")
    parser.add_argument("--match", help="Regular expression on the call name")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--min-sample", type=float, default=0.02)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--list", action="store_true", help="Print the call names")
    # Worker modes, started by the driver.
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tree", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--memory-call", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)

    if arguments.list:
        print("\n".join(CALLS))
        return 0
    if arguments.serve:
        serve(arguments.tree, arguments.min_sample, arguments.samples)
        return 0
    if arguments.memory_call:
        print(json.dumps(run_memory_worker(arguments.memory_call, arguments.tree)))
        return 0
    if arguments.baseline is None and arguments.baseline_ref is None:
        parser.error("give --baseline or --baseline-ref")

    candidate = arguments.candidate
    if arguments.baseline is not None:
        result = compare(arguments, arguments.baseline, candidate)
    else:
        with baseline_from_ref(arguments.baseline_ref, candidate) as baseline:
            result = compare(arguments, baseline, candidate)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=1) + "\n")
    print(report(result))
    return 0 if all(entry["passed"] for entry in result["calls"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

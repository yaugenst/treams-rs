# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Compare forward and reverse mode on the same two physical calculations.

From a checkout with treams-rs and JAX installed in .venv:
    uv run --script scripts/benchmark_forward_reverse.py

The driver uses only the standard library and runs each example separately
with .venv/bin/python. Use --check-only to verify the derivatives without
recording times. Normal runs report tracing/compilation separately from warm
execution and write raw measurements under benchmarks/results/local/.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import runpy
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

from _harness import cpu_affinity, file_sha256, package_sha256, pinned_threads

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "field-map": "forward_field_sensitivity.py",
    "cluster": "reverse_cluster_gradient.py",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-points", type=int, default=5)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--min-sample", type=float, default=0.04)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--worker", choices=EXAMPLES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.grid_points, args.particles, args.threads, args.samples) < 1:
        parser.error("grid-points, particles, threads and samples must be positive")
    if args.min_sample <= 0 or args.max_seconds <= 0:
        parser.error("min-sample and max-seconds must be positive")
    return args


def problem(args):
    import jax.numpy as jnp

    example = ROOT / "docs/examples" / EXAMPLES[args.worker]
    make_problem = runpy.run_path(str(example))["make_problem"]
    if args.worker == "field-map":
        field_map, initial = make_problem(args.grid_points)

        def function(parameters):
            field = field_map(parameters)
            # Keep all complex field information. Reverse mode needs real
            # outputs for a function whose input parameters are real.
            return jnp.stack((field.real, field.imag), axis=-1)

        description = "Changes in the entire complex scattered field map for k0 and common sphere radius."
    else:
        function, initial = make_problem(args.particles)
        description = "Changes in one scattering cross section for every sphere radius and center coordinate."
    return function, initial, description, example


def provenance(example):
    import treams_rs as tr
    import treams_rs._native as native

    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    return {
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "jax", "jaxlib", "treams-rs")
        },
        "cpu_model": cpu,
        "cpu_affinity": cpu_affinity(),
        "thread_environment": {
            name: value
            for name, value in os.environ.items()
            if name.endswith("NUM_THREADS") or name in ("XLA_FLAGS", "JAX_PLATFORMS")
        },
        "hashes": {
            "script": file_sha256(__file__),
            "example": file_sha256(example),
            "native_library": file_sha256(native.__file__),
            "python_package": package_sha256(Path(tr.__file__).parent, (".py",)),
            "rust_sources": package_sha256(ROOT / "crates", (".rs", ".toml")),
            "cargo_lock": file_sha256(ROOT / "Cargo.lock"),
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }


def sample(function, minimum, repetitions):
    import jax

    while True:
        start = time.perf_counter()
        for _ in range(repetitions):
            jax.block_until_ready(function())
        elapsed = time.perf_counter() - start
        if elapsed >= minimum:
            return elapsed / repetitions, elapsed, repetitions
        repetitions *= 2


def worker(args):
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    function, initial, description, example = problem(args)
    parameters = jnp.asarray(initial)
    output = np.asarray(function(parameters))
    transforms = {
        "forward": jax.jacfwd(function),
        "reverse": jax.grad(function) if output.ndim == 0 else jax.jacrev(function),
    }
    compiled, compilation = {}, {}
    for name, transform in transforms.items():
        if args.check_only:
            compiled[name] = jax.jit(transform).lower(parameters).compile()
        else:
            start = time.perf_counter()
            compiled[name] = jax.jit(transform).lower(parameters).compile()
            compilation[name] = time.perf_counter() - start

    derivatives = {
        name: np.asarray(jax.block_until_ready(run(parameters)))
        for name, run in compiled.items()
    }
    forward, reverse = derivatives["forward"], derivatives["reverse"]
    np.testing.assert_allclose(forward, reverse, rtol=3e-10, atol=1e-10)

    # A separate finite difference checks the complete array in one joint
    # parameter direction. It is never part of the measured work.
    rng = np.random.default_rng(51)
    direction = (
        rng.normal(size=initial.shape) * np.maximum(np.abs(initial), 1e-3) * 0.01
    )
    step = 1e-4
    difference = (
        np.asarray(function(parameters + step * direction))
        - np.asarray(function(parameters - step * direction))
    ) / (2 * step)
    predicted = (
        forward.reshape(output.size, initial.size) @ direction.ravel()
    ).reshape(output.shape)
    np.testing.assert_allclose(predicted, difference, rtol=2e-6, atol=1e-9)
    scale = max(float(np.linalg.norm(forward)), 1e-30)
    validation = {
        "forward_reverse_relative_error": float(np.linalg.norm(forward - reverse))
        / scale,
        "finite_difference_relative_error": float(
            np.linalg.norm(predicted - difference)
        )
        / max(float(np.linalg.norm(predicted)), 1e-30),
        "finite_difference_step": step,
    }

    results = {}
    if not args.check_only:
        for run in compiled.values():
            jax.block_until_ready(run(parameters))
            jax.block_until_ready(run(parameters))
        samples = {name: [] for name in compiled}
        counts = {name: 1 for name in compiled}
        for round_number in range(args.samples):
            order = (
                ("forward", "reverse")
                if round_number % 2 == 0
                else ("reverse", "forward")
            )
            for name in order:
                run = compiled[name]
                per_call, elapsed, repetitions = sample(
                    partial(run, parameters), args.min_sample, counts[name]
                )
                counts[name] = repetitions
                samples[name].append(
                    {
                        "seconds_per_call": per_call,
                        "elapsed_seconds": elapsed,
                        "calls": repetitions,
                    }
                )
        results = {
            name: {
                "derivative_transform": "jax.jacfwd"
                if name == "forward"
                else "jax.grad"
                if output.ndim == 0
                else "jax.jacrev",
                "trace_and_compile_seconds": compilation[name],
                "median_seconds": statistics.median(
                    row["seconds_per_call"] for row in rows
                ),
                "samples": rows,
            }
            for name, rows in samples.items()
        }
    return {
        "case": args.worker,
        "description": description,
        "input_real_parameters": initial.size,
        "output_real_values": output.size,
        "parameter_shape": list(initial.shape),
        "derivative_shape": list(forward.shape),
        "validation": validation,
        "timings": results,
        "provenance": provenance(example),
    }


def main():
    args = arguments()
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = args.output or ROOT / "benchmarks/results/local/forward-reverse" / stamp
    output.mkdir(parents=True, exist_ok=False)
    environment = {
        **os.environ,
        **pinned_threads(args.threads),
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads="
        + str(args.threads),
    }
    deadline = time.monotonic() + args.max_seconds
    reports = []
    for name in EXAMPLES:
        command = [
            str(args.python),
            __file__,
            "--worker",
            name,
            "--grid-points",
            str(args.grid_points),
            "--particles",
            str(args.particles),
            "--threads",
            str(args.threads),
            "--samples",
            str(args.samples),
            "--min-sample",
            str(args.min_sample),
        ]
        if args.check_only:
            command.append("--check-only")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=max(1, deadline - time.monotonic()),
        )
        (output / f"{name}.stderr.txt").write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{name} failed; see {output / (name + '.stderr.txt')}")
        report = json.loads(completed.stdout)
        path = output / f"{name}.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        reports.append({"path": path.name, "sha256": file_sha256(path)})
        print(
            f"{name}: {report['input_real_parameters']} inputs, {report['output_real_values']} real outputs; derivatives agree"
        )
        if report["timings"]:
            for mode, data in report["timings"].items():
                print(
                    f"  {mode}: prepare {data['trace_and_compile_seconds']:.3f} s; warm median {data['median_seconds'] * 1e3:.3f} ms"
                )
            times = report["timings"]
            print(
                f"  warm reverse / forward: {times['reverse']['median_seconds'] / times['forward']['median_seconds']:.3f}"
            )
    (output / "benchmark.py").write_bytes(Path(__file__).read_bytes())
    for example in EXAMPLES.values():
        (output / example).write_bytes((ROOT / "docs/examples" / example).read_bytes())
    manifest = {
        "schema": "treams-forward-reverse-1",
        "started_utc": stamp,
        "command": sys.argv,
        "threads": args.threads,
        "samples": args.samples,
        "minimum_sample_seconds": args.min_sample,
        "check_only": args.check_only,
        "reports": reports,
        "notes": [
            "Each mode computes the same complete derivative array for the same inputs; neither comparison substitutes a directional derivative for a full Jacobian or gradient.",
            "The field comparison represents each complex component by its real and imaginary parts. It preserves all field information.",
            "Preparation includes JAX tracing and compilation. Warm measurements recompute the whole differentiated scattering calculation at the fixed chosen parameter values; no result arrays are reused across calls.",
            "Preparation is measured once per mode in forward-then-reverse order within each worker; framework caches can influence these setup times.",
            "Every timed result is synchronized; modes alternate within each case. Case workers run sequentially with the same thread caps and inherited CPU affinity.",
            "These times describe the complete JAX workflows, including framework and callback overhead, at the selected problem sizes.",
            "The field wavelength sensitivity holds material permittivity fixed; material dispersion is excluded.",
            "These are first derivatives at the fixed mode cutoff and grid. Agreement is not a mode-cutoff convergence study.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()

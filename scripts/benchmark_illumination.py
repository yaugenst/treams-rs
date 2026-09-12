# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "treams==0.4.5", "threadpoolctl"]
# ///
"""Qualify requested illumination runtime and isolated-process peak memory.

Build the release extension first. Each backend/forward-or-adjoint measurement
runs in a fresh process. Dense reference limits are explicit, never OOM probes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprints() -> dict[str, str]:
    from treams_rs import _native

    folder = Path(_native.__file__).parent
    source = hashlib.sha256()
    for path in sorted(folder.glob("*.py")):
        source.update(path.name.encode())
        source.update(bytes.fromhex(digest(path)))
    return {
        "native_sha256": digest(Path(_native.__file__)),
        "python_source_sha256": source.hexdigest(),
        "benchmark_sha256": digest(Path(__file__)),
    }


def peak_mib() -> float:
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def case(args):
    import numpy as np

    import treams_rs as tr

    n = args.particles
    side = int(np.ceil(n ** (1 / 3)))
    grid = np.column_stack(np.unravel_index(np.arange(n), (side, side, side)))
    positions = grid * (1.3 if args.strength == "weak" else 1.1)
    radii = (
        np.linspace(0.12, 0.18, n)
        if args.strength == "weak"
        else np.linspace(0.23, 0.29, n)
    )
    epsilon = np.full(n, 2.2 + 0.02j if args.strength == "weak" else 4.0 + 0.05j)
    basis = tr.SphericalWaveBasis.default(args.lmax, n, positions=positions)
    columns = []
    for p in range(args.columns):
        direction = np.array([0.13 + p * 0.07, -0.11 + p * 0.03, 1.0])
        direction /= np.linalg.norm(direction)
        columns.append(
            np.asarray(tr.plane_wave(direction, p % 2, k0=1.3).expand(basis))
        )
    incident = np.asarray(np.column_stack(columns), dtype=np.complex128)
    return radii, epsilon, positions, basis, incident


def worker(args) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    import treams_rs as tr
    from treams_rs.iterative import SphereCluster

    if tr._native.build_profile() != "release":
        raise RuntimeError("benchmark requires a release extension")
    before = fingerprints()
    threads = args.threads
    with threadpool_limits(limits=threads):
        start = time.perf_counter()
        radii, epsilon, positions, basis, incident = case(args)
        common_setup = time.perf_counter() - start
        baseline = peak_mib()
        options = dict(
            rtol=args.rtol, restart=args.restart, max_iterations=args.max_iterations
        )

        def prepare(record=False):
            if args.backend == "matrix-free":
                return SphereCluster(args.lmax, 1.3, radii, epsilon, positions)
            if args.backend == "full":
                value, context = tr.diff.cluster(
                    args.lmax, 1.3, radii, epsilon, positions
                )
                return (value, context) if record else value
            if not record:
                return tr.diff.cluster_factor(args.lmax, 1.3, radii, epsilon, positions)
            particles = [
                tr.diff.sphere(args.lmax, 1.3, [r], [eps, 1])
                for r, eps in zip(radii, epsilon, strict=True)
            ]
            coupling, expansion = tr.diff.expansion(
                basis, basis, [1.3, 1.3], singular=True
            )
            factor = tr.diff.factor_interaction_blocks(
                [p[0] for p in particles], coupling
            )
            return factor, particles, expansion

        latest_reports = []

        def solve(prepared):
            nonlocal latest_reports
            if args.backend == "matrix-free":
                solution = prepared.solve(incident, **options)
                latest_reports = [list(report) for report in solution.convergence]
                return solution.coefficients
            if args.backend == "full":
                return prepared @ incident
            return prepared.solve(incident)

        def record(prepared):
            nonlocal latest_reports
            if args.backend == "matrix-free":
                solution, context = prepared.solve_with_pullback(incident, **options)
                latest_reports = [list(report) for report in solution.convergence]
                return solution.coefficients, context
            if args.backend == "full":
                return prepared[0] @ incident, prepared
            value, context = prepared[0].record(incident)
            return value, (context, prepared[1], prepared[2])

        latest_adjoint_reports = []

        def backward(context, cotangent):
            nonlocal latest_adjoint_reports
            if args.backend == "matrix-free":
                g = context.pullback(cotangent)
                latest_adjoint_reports = [list(report) for report in g.convergence]
                return {
                    "radii": g.radii,
                    "positions": g.positions,
                    "epsilon": g.epsilon,
                    "k0": np.asarray(g.k0),
                    "incident": g.incident,
                }
            if args.backend == "full":
                value, context = context
                gincident = (value.T @ cotangent.conj()).conj()
                gr, gp, ge, gk = context.pullback(cotangent @ incident.conj().T)
            else:
                interaction, particles, expansion = context
                blocks, coupling, gincident = interaction.pullback_blocks(cotangent)
                to, source, ks = expansion.pullback(coupling)
                gp, gk = to + source, float(np.real(ks).sum())
                gr, ge = np.zeros(args.particles), np.zeros(args.particles, complex)
                for i, ((_, context), block) in enumerate(
                    zip(particles, blocks, strict=True)
                ):
                    k0, radius, eps, _, _ = context.pullback(block)
                    gr[i], ge[i] = radius[0], eps[0]
                    gk += k0
            return {
                "radii": gr,
                "positions": gp,
                "epsilon": ge,
                "k0": np.asarray(gk),
                "incident": gincident,
            }

        setup_times, fresh_times, reuse_times, backward_times = [], [], [], []
        if args.phase == "forward":
            for i in range(args.warmup + args.repeats):
                start = time.perf_counter()
                prepared = prepare()
                setup_end = time.perf_counter()
                value = solve(prepared)
                if i >= args.warmup:
                    setup_times.append(setup_end - start)
                    fresh_times.append(time.perf_counter() - start)
                del value, prepared
                gc.collect()
            prepared = prepare()
            for i in range(args.warmup + args.repeats):
                start = time.perf_counter()
                value = solve(prepared)
                if i >= args.warmup:
                    reuse_times.append(time.perf_counter() - start)
                del value
            value = solve(prepared)
            output = {"value": value}
        else:
            for i in range(args.warmup + args.repeats):
                start = time.perf_counter()
                prepared = prepare(record=True)
                setup_end = time.perf_counter()
                value, context = record(prepared)
                forward_end = time.perf_counter()
                g = 2 * value
                start_backward = time.perf_counter()
                gradients = backward(context, g)
                if i >= args.warmup:
                    setup_times.append(setup_end - start)
                    fresh_times.append(forward_end - start)
                    backward_times.append(time.perf_counter() - start_backward)
                del value, context, prepared, gradients
                gc.collect()
            prepared = prepare(record=True)
            value, context = record(prepared)
            output = {"value": value, **backward(context, 2 * value)}
        peak = peak_mib()
        np.savez(args.array_output, **output)
        numerical = {}
        if args.phase == "adjoint":
            # Geometry derivatives hold the supplied incident multipoles fixed.
            translation_ward = np.linalg.norm(output["positions"].sum(axis=0))
            scale_ward = float(
                np.vdot(output["radii"], radii).real
                + np.vdot(output["positions"], positions).real
                - 1.3 * output["k0"]
            )
            illumination_ward = float(
                np.vdot(output["incident"], incident).real
                - 2 * np.vdot(output["value"], output["value"]).real
            )
            ward_scale = max(
                1.0, float(np.linalg.norm(output["radii"]) * np.linalg.norm(radii))
            )
            assert translation_ward <= 1e-8 * ward_scale
            assert abs(scale_ward) <= 1e-8 * ward_scale
            assert abs(illumination_ward) <= 1e-8 * ward_scale
            numerical = {
                "translation_ward_absolute": float(translation_ward),
                "scale_ward_absolute": scale_ward,
                "illumination_ward_absolute": illumination_ward,
            }
            if args.backend == "matrix-free":
                direction = radii * np.cos(np.arange(args.particles) * 0.37)
                h = 1e-4
                losses = []
                for sign in [-1, 1]:
                    perturbed = SphereCluster(
                        args.lmax, 1.3, radii + sign * h * direction, epsilon, positions
                    )
                    result = perturbed.solve(incident, **options).coefficients
                    losses.append(float(np.vdot(result, result).real))
                numerical_fd = (losses[1] - losses[0]) / (2 * h)
                analytical = float(np.vdot(output["radii"], direction).real)
                error = abs(numerical_fd - analytical) / max(abs(analytical), 1e-12)
                assert error < 2e-6, (numerical_fd, analytical, error)
                numerical.update(
                    radius_direction_fd=numerical_fd,
                    radius_direction_adjoint=analytical,
                    radius_direction_relative_error=error,
                )
        # High-water RSS above excludes validation-only finite-difference solves.
        assert before == fingerprints(), (
            "sources or native binary changed during measurement"
        )
        print(
            json.dumps(
                {
                    **before,
                    "backend": args.backend,
                    "phase": args.phase,
                    "native_profile": tr._native.build_profile(),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "treams": importlib.metadata.version("treams"),
                    "affinity": sorted(os.sched_getaffinity(0))
                    if hasattr(os, "sched_getaffinity")
                    else None,
                    "threadpools": [
                        {**pool, "filepath": Path(pool["filepath"]).name}
                        for pool in threadpool_info()
                    ],
                    "threads": threads,
                    "common_input_setup_seconds": common_setup,
                    "setup_seconds": setup_times,
                    "fresh_seconds": fresh_times,
                    "reuse_seconds": reuse_times,
                    "backward_seconds": backward_times,
                    "setup_median_seconds": statistics.median(setup_times),
                    "fresh_median_seconds": statistics.median(fresh_times),
                    "reuse_median_seconds": statistics.median(reuse_times)
                    if reuse_times
                    else None,
                    "backward_median_seconds": statistics.median(backward_times)
                    if backward_times
                    else None,
                    "baseline_rss_mib": baseline,
                    "peak_rss_mib": peak,
                    "additional_peak_rss_mib": peak - baseline,
                    "forward_convergence": latest_reports,
                    "adjoint_convergence": latest_adjoint_reports,
                    "numerical": numerical,
                }
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--lmax", type=int, default=1)
    parser.add_argument("--columns", type=int, default=1)
    parser.add_argument("--strength", choices=["weak", "moderate"], default="weak")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--restart", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument(
        "--dense-limit",
        type=int,
        default=3072,
        help="Maximum dense-reference dimension; larger cases explicitly skip both dense paths",
    )
    parser.add_argument(
        "--oracle-limit",
        type=int,
        default=768,
        help="Maximum dimension checked against upstream treams",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=["full", "selected", "matrix-free"])
    parser.add_argument("--phase", choices=["forward", "adjoint"])
    parser.add_argument("--array-output", type=Path)
    args = parser.parse_args()
    if (
        min(args.particles, args.lmax, args.columns, args.threads, args.repeats) <= 0
        or args.warmup < 0
    ):
        parser.error("counts must be positive and warmup nonnegative")
    if args.worker:
        worker(args)
        return
    dimension = args.particles * 2 * args.lmax * (args.lmax + 2)
    env = dict(
        os.environ,
        RAYON_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        VECLIB_MAXIMUM_THREADS=str(args.threads),
    )
    backends = (
        ["full", "selected", "matrix-free"]
        if dimension <= args.dense_limit
        else ["matrix-free"]
    )
    measurements = []
    arrays = {}
    with tempfile.TemporaryDirectory(prefix="treams-illumination-") as folder:
        for backend in backends:
            for phase in ["forward", "adjoint"]:
                array_file = Path(folder) / f"{backend}-{phase}.npz"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--backend",
                    backend,
                    "--phase",
                    phase,
                    "--array-output",
                    str(array_file),
                ]
                for name in [
                    "particles",
                    "lmax",
                    "columns",
                    "strength",
                    "threads",
                    "repeats",
                    "warmup",
                    "rtol",
                    "restart",
                    "max_iterations",
                ]:
                    cmd.extend(
                        ["--" + name.replace("_", "-"), str(getattr(args, name))]
                    )
                print(
                    f"{backend} {phase}: N={args.particles}, lmax={args.lmax}, P={args.columns}",
                    file=sys.stderr,
                    flush=True,
                )
                result = subprocess.run(
                    cmd, env=env, capture_output=True, text=True, check=False
                )
                if result.returncode:
                    raise RuntimeError(f"{backend} {phase} failed:\n{result.stderr}")
                measurements.append(json.loads(result.stdout))
                import numpy as np

                with np.load(array_file) as payload:
                    arrays[(backend, phase)] = dict(payload)
        fingerprints_set = {
            tuple(
                item[key]
                for key in ["native_sha256", "python_source_sha256", "benchmark_sha256"]
            )
            for item in measurements
        }
        assert len(fingerprints_set) == 1, "all measurements must use the same code"
        reference = arrays[(backends[0], "adjoint")]
        errors = {}
        for (backend, phase), payload in arrays.items():
            for name, value in payload.items():
                expected = reference[name]
                relative = float(
                    np.linalg.norm(value - expected)
                    / max(np.linalg.norm(expected), 1e-15)
                )
                assert relative < (3e-8 if name != "value" else 3e-9), (
                    backend,
                    phase,
                    name,
                    relative,
                )
                errors[f"{backend}/{phase}/{name}"] = relative
        oracle_error = None
        if dimension <= args.oracle_limit:
            import treams
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=args.threads):
                radii, epsilon, positions, _, incident = case(args)
                spheres = [
                    treams.TMatrix.sphere(args.lmax, 1.3, r, [eps, 1])
                    for r, eps in zip(radii, epsilon, strict=True)
                ]
                expected = np.asarray(
                    treams.TMatrix.cluster(spheres, positions).interaction.solve()
                    @ incident
                )
                oracle_error = float(
                    np.linalg.norm(reference["value"] - expected)
                    / np.linalg.norm(expected)
                )
                assert oracle_error < 3e-9, oracle_error
        result = {
            "recorded_at": datetime.now(UTC).isoformat(),
            "case": {
                name: getattr(args, name)
                for name in [
                    "particles",
                    "lmax",
                    "columns",
                    "strength",
                    "threads",
                    "repeats",
                    "warmup",
                    "rtol",
                    "restart",
                    "max_iterations",
                    "dense_limit",
                    "oracle_limit",
                ]
            },
            "dimension": dimension,
            "dense_skipped_reason": None
            if "full" in backends
            else f"dimension {dimension} exceeds explicit dense limit {args.dense_limit}; no same-size dense comparison claimed",
            "oracle_relative_error": oracle_error,
            "comparison_reference": "strongest available native full-T cluster API"
            if "full" in backends
            else "recorded matrix-free forward; independent large-case gradient finite differences and invariants only",
            "comparison_relative_errors": errors,
            "measurements": measurements,
            "selected_assembly": "Forward/reuse use native diff.cluster_factor. Complete physical adjoints compose sphere/expansion contexts with block LU and include their generic assembly and retained memory.",
            "peak_rss_scope": "Each backend and forward/adjoint phase in separate fresh process. Includes imports, common inputs, warmup, repeated fresh and reused operations; excludes finite-difference validation after capture.",
            "timing_scope": "Fresh includes geometry/local coefficients/coupling/factor setup plus requested solve. Reuse keeps the full T, selected LU, or matrix-free geometry. Common plane-wave input construction excluded and reported separately. Python result destruction excluded.",
            "complete": True,
        }
        text = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            print(text, end="")


if __name__ == "__main__":
    main()

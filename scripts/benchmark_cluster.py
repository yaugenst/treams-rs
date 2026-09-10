# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "treams==0.4.5", "threadpoolctl"]
# ///
"""Isolated-process, matched-thread benchmarks of full cluster T-matrices.

Run after `just build-ext-release`: uv run --no-sync python scripts/benchmark_cluster.py.
The parent never imports either solver, keeping peak-RSS measurements separate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


def worker(
    backend: str, particles: int, order: int, repeats: int, workload: str, samples: int
) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    if backend in ("rust", "check"):
        from treams_rs import SphericalWaveBasis, _native, diff

        if _native.build_profile() != "release":
            raise RuntimeError("benchmark requires just build-ext-release")
    if backend in ("treams", "check"):
        import treams

    threads = int(os.environ["BENCH_THREADS"])
    with threadpool_limits(limits=threads):
        radii = np.linspace(0.15, 0.25, particles)
        epsilon = np.full(particles, 4 + 0.1j)
        positions = np.column_stack(
            [np.arange(particles) * 0.8, np.zeros((particles, 2))]
        )

        if workload == "field":
            points = np.column_stack(
                [
                    np.linspace(0.1, particles * 0.8 + 0.2, samples),
                    np.full(samples, 1.1),
                    np.full(samples, 0.4),
                ]
            )
            rng = np.random.default_rng(5)
            dimension = particles * 2 * order * (order + 2)
            amplitudes = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
            if backend in ("rust", "check"):
                basis = SphericalWaveBasis.default(order, particles, positions)
            if backend in ("treams", "check"):
                oracle_basis = treams.SphericalWaveBasis.default(
                    order, particles, positions
                )

        def rust():
            if workload == "field":
                return diff.field(amplitudes, points, basis, [1.3, 1.3], singular=True)
            return diff.cluster(order, 1.3, radii, epsilon, positions)

        def upstream():
            if workload == "field":
                return (
                    np.asarray(
                        treams.efield(
                            points,
                            basis=oracle_basis,
                            k0=1.3,
                            modetype="singular",
                            poltype="helicity",
                        )
                    )
                    @ amplitudes
                )
            spheres = [
                treams.TMatrix.sphere(order, 1.3, r, [e, 1])
                for r, e in zip(radii, epsilon, strict=True)
            ]
            return treams.TMatrix.cluster(spheres, positions).interaction.solve()

        if backend == "check":
            expected = upstream()
            actual, _ = rust()
            np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-12)
            print(json.dumps({"accuracy_check": "passed"}))
            return
        baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        function = rust if backend == "rust" else upstream
        function()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = function()
            times.append(time.perf_counter() - start)
            del result
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(
            json.dumps(
                {
                    "backend": backend,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy_version": np.__version__,
                    "treams_version": importlib.metadata.version("treams"),
                    "native_sha256": hashlib.sha256(
                        Path(_native.__file__).read_bytes()
                    ).hexdigest()
                    if backend == "rust"
                    else None,
                    "native_profile": _native.build_profile()
                    if backend == "rust"
                    else None,
                    "workload": workload,
                    "samples": samples if workload == "field" else None,
                    "particles": particles,
                    "lmax": order,
                    "dimension": particles * 2 * order * (order + 2),
                    "threads": threads,
                    "median_seconds": statistics.median(times),
                    "peak_rss_mib": peak,
                    "baseline_rss_mib": baseline,
                    "samples_seconds": times,
                    "blas": threadpool_info(),
                }
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=["cluster", "field"], default="cluster")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--worker", choices=["rust", "treams", "check"])
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.worker:
        worker(
            args.worker,
            args.particles,
            args.lmax,
            args.repeats,
            args.workload,
            args.samples,
        )
        return
    env = dict(
        os.environ,
        BENCH_THREADS=str(args.threads),
        RAYON_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
    )
    results = []
    for backend in ["check", "treams", "rust"]:
        command = [
            sys.executable,
            __file__,
            "--worker",
            backend,
            "--workload",
            args.workload,
            "--samples",
            str(args.samples),
            "--particles",
            str(args.particles),
            "--lmax",
            str(args.lmax),
            "--repeats",
            str(args.repeats),
        ]
        result = subprocess.run(
            command, env=env, check=True, capture_output=True, text=True
        )
        if backend != "check":
            results.append(json.loads(result.stdout))
    print(
        json.dumps(
            {
                "results": results,
                "speedup": results[0]["median_seconds"] / results[1]["median_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "treams==0.4.5", "threadpoolctl"]
# ///
"""Isolated-process, matched-thread benchmarks of scattering and fields.

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
        from treams_rs import (
            CylindricalWaveBasis,
            PlaneWaveBasisByComp,
            SMatrices,
            SphericalWaveBasis,
            _native,
            diff,
            lattice,
        )

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

        if workload == "internal-field":
            n = 2 * particles * order
            rng = np.random.default_rng(81)
            lower, upper = [
                0.1
                / np.sqrt(n)
                * (rng.normal(size=(2, 2, n, n)) + 1j * rng.normal(size=(2, 2, n, n)))
                for _ in range(2)
            ]
            for blocks in (lower, upper):
                blocks[0, 0] += np.eye(n)
                blocks[1, 1] += np.eye(n)
            up, down = [
                (rng.normal(size=(n, samples)) + 1j * rng.normal(size=(n, samples)))
                / np.sqrt(n)
                for _ in range(2)
            ]
            if backend in ("treams", "check"):
                q = np.column_stack(
                    [np.linspace(0.1, 0.8, n // 2), np.full(n // 2, 0.2)]
                )
                oracle_basis = treams.PlaneWaveBasisByComp.default(q)
                oracle_lower = treams.SMatrices(lower, basis=oracle_basis, k0=1.3)
                oracle_upper = treams.SMatrices(upper, basis=oracle_basis, k0=1.3)

        if workload == "ebcm":
            if particles != 1:
                raise ValueError("EBCM benchmark uses one surface")
            nodes, quadrature = np.polynomial.legendre.leggauss(samples)
            theta = (nodes + 1) * np.pi / 2
            quadrature *= np.pi / 2
            surface_radii = 0.3 * (1 + 0.23 * np.cos(theta) ** 2)
            surface_slopes = -0.138 * np.cos(theta) * np.sin(theta)
            index = np.sqrt((3.1 + 0.2j) * (1.2 + 0.1j))
            surface_ks = 1.3 * np.array([[index - 0.07, index + 0.07], [1, 1]])
            surface_zs = [np.sqrt((1.2 + 0.1j) / (3.1 + 0.2j)), 1.0]
            if backend in ("rust", "check"):
                basis = SphericalWaveBasis.default(order)
            if backend in ("treams", "check"):
                from treams.ebcm import qmat as reference_qmat

                oracle_basis = treams.SphericalWaveBasis.default(order)

        if workload == "slab":
            q = np.column_stack([np.linspace(0.1, 0.8, samples), np.full(samples, 0.2)])
            layer_eps = np.array(
                [
                    1,
                    *[
                        2.3 + 0.1j if i % 2 == 0 else 1.7 + 0.05j
                        for i in range(particles)
                    ],
                    1,
                ]
            )
            layer_ks = np.repeat((1.3 * np.sqrt(layer_eps))[:, None], 2, axis=1)
            layer_zs = 1 / np.sqrt(layer_eps)
            layer_thickness = np.linspace(0.1, 0.4, particles)
            if backend in ("rust", "check"):
                basis = PlaneWaveBasisByComp.default(q)
            if backend in ("treams", "check"):
                oracle_basis = treams.PlaneWaveBasisByComp.default(q)

        if workload in ("plane-field", "plane-operator"):
            q = np.column_stack(
                [
                    np.linspace(0.1, 1.7, particles * order),
                    np.full(particles * order, 0.2),
                ]
            )
            points = np.column_stack(
                [
                    np.linspace(0.1, 4, samples),
                    np.full(samples, 0.3),
                    np.full(samples, 0.4),
                ]
            )
            rng = np.random.default_rng(5)
            amplitudes = rng.normal(size=2 * particles * order) + 1j * rng.normal(
                size=2 * particles * order
            )
            if backend in ("rust", "check"):
                basis = PlaneWaveBasisByComp.default(q)
                vectors = np.column_stack(basis.kvecs(1.3))
            if backend in ("treams", "check"):
                oracle_basis = treams.PlaneWaveBasisByComp.default(q)

        if workload in ("conversion", "periodic-conversion"):
            if particles != 1:
                raise ValueError(
                    "the upstream conversion benchmark uses one common origin"
                )
            kz = (
                0.2 + 2 * np.pi / 200 * (np.arange(samples) - samples // 2)
                if workload == "periodic-conversion"
                else np.linspace(-0.7, 0.7, samples)
            )
            if backend in ("rust", "check"):
                basis = SphericalWaveBasis.default(order)
                source_basis = CylindricalWaveBasis.default(kz, order)
                if workload == "periodic-conversion":
                    basis, source_basis = source_basis, basis
            if backend in ("treams", "check"):
                oracle_basis = treams.SphericalWaveBasis.default(order)
                oracle_source = treams.CylindricalWaveBasis.default(kz, order)
                if workload == "periodic-conversion":
                    oracle_basis, oracle_source = oracle_source, oracle_basis

        if workload in ("rotation", "plane-expansion"):
            if backend in ("rust", "check"):
                basis = SphericalWaveBasis.default(order, particles, positions)
            if backend in ("treams", "check"):
                oracle_basis = treams.SphericalWaveBasis.default(
                    order, particles, positions
                )

        if workload == "plane-expansion":
            q = np.column_stack([np.linspace(0.1, 1.7, samples), np.full(samples, 0.2)])
            if backend in ("rust", "check"):
                source_basis = PlaneWaveBasisByComp.default(q)
                vectors = np.column_stack(source_basis.kvecs(1.3))
            if backend in ("treams", "check"):
                oracle_source = treams.PlaneWaveBasisByComp.default(q)

        if workload == "cylindrical-plane-expansion":
            q = np.column_stack([np.full(samples, 0.2), np.linspace(0.1, 1.7, samples)])
            if backend in ("rust", "check"):
                basis = CylindricalWaveBasis.default([0.2], order, particles, positions)
                source_basis = PlaneWaveBasisByComp.default(q, "zx")
                vectors = np.column_stack(source_basis.kvecs(1.3))
            if backend in ("treams", "check"):
                oracle_basis = treams.CylindricalWaveBasis.default(
                    [0.2], order, particles, positions
                )
                oracle_source = treams.PlaneWaveBasisByComp.default(q, "zx")

        if workload in ("periodic", "array"):
            width = int(np.ceil(np.sqrt(particles)))
            positions = (
                np.column_stack(
                    [
                        np.arange(particles) % width,
                        np.arange(particles) // width,
                        np.zeros(particles),
                    ]
                )
                * 0.8
            )
            vectors = np.diag([width * 0.8, width * 0.8])
            bloch = np.array([0.1, 0.15])
            orders = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
            q = bloch + orders @ (2 * np.pi * np.linalg.inv(vectors).T)
            if backend in ("rust", "check"):
                basis = SphericalWaveBasis.default(order, particles, positions)
                ports = PlaneWaveBasisByComp.default(q)

        if workload in ("field", "cylindrical-field", "field-operator"):
            points = np.column_stack(
                [
                    np.linspace(0.1, particles * 0.8 + 0.2, samples),
                    np.full(samples, 1.1),
                    np.full(samples, 0.4),
                ]
            )
            rng = np.random.default_rng(5)
            dimension = particles * (
                4 * (2 * order + 1)
                if workload == "cylindrical-field"
                else 2 * order * (order + 2)
            )
            amplitudes = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
            if backend in ("rust", "check"):
                basis = (
                    CylindricalWaveBasis.default(
                        [0.2, -0.3], order, particles, positions
                    )
                    if workload == "cylindrical-field"
                    else SphericalWaveBasis.default(order, particles, positions)
                )
            if backend in ("treams", "check"):
                oracle_basis = (
                    treams.CylindricalWaveBasis.default(
                        [0.2, -0.3], order, particles, positions
                    )
                    if workload == "cylindrical-field"
                    else treams.SphericalWaveBasis.default(order, particles, positions)
                )

        if workload == "cylindrical-array":
            vectors = np.array([[particles * 0.8]])
            bloch = np.array([0.1])
            q = np.column_stack(
                [
                    np.full(5, 0.2),
                    bloch[0] + np.array([0, 1, -1, 2, -2]) * 2 * np.pi / vectors[0, 0],
                ]
            )
            if backend in ("rust", "check"):
                basis = CylindricalWaveBasis.default([0.2], order, particles, positions)
                ports = PlaneWaveBasisByComp.default(q, "zx")
            if backend in ("treams", "check"):
                oracle_ports = treams.PlaneWaveBasisByComp.default(q, "zx")

        # The reference cylinder sum loses accuracy for larger cells at eta=0.
        # Use the same converged split for both implementations.
        eta = 0.7 if workload == "cylindrical-array" else 0

        def rust():
            if workload == "ebcm":
                return diff.ebcm_qmat(
                    surface_radii,
                    surface_slopes,
                    surface_ks,
                    surface_zs,
                    theta=theta,
                    weights=quadrature,
                    out=basis,
                    legacy=True,
                )
            if workload == "internal-field":
                return diff.smatrix_illuminate(lower, upper, up, down)
            if workload == "slab":
                return SMatrices.slab(
                    layer_thickness, basis, 1.3, list(layer_eps)
                ).array, None
            if workload in ("plane-expansion", "cylindrical-plane-expansion"):
                return diff.plane_expansion(basis, vectors, source_basis.pol)
            if workload in ("plane-field", "plane-operator"):
                return diff.plane_field(
                    None if workload == "plane-operator" else amplitudes,
                    points,
                    vectors,
                    basis.pol,
                )
            if workload == "conversion":
                return diff.expansion(basis, source_basis, [1.3, 1.3])
            if workload == "periodic-conversion":
                return diff.periodic_conversion(basis, source_basis, [1.3, 1.3], 200)
            if workload == "rotation":
                return diff.rotation([0.2, 0.7, -0.3], basis)
            if workload == "field-operator":
                return diff.field_operator(points, basis, [1.3, 1.3], singular=True)
            if workload in ("field", "cylindrical-field"):
                return diff.field(amplitudes, points, basis, [1.3, 1.3], singular=True)
            if workload in ("periodic", "array", "cylindrical-array"):
                dimension = len(basis)
                local = np.zeros((dimension, dimension), dtype=complex)
                contexts = []
                block = dimension // particles
                for index, (radius, eps) in enumerate(zip(radii, epsilon, strict=True)):
                    value, context = (
                        diff.cylinder([0.2], order, 1.3, [radius], [eps, 1])
                        if workload
                        in ("cylindrical-array", "cylindrical-plane-expansion")
                        else diff.sphere(order, 1.3, [radius], [eps, 1])
                    )
                    contexts.append(context)
                    local[
                        index * block : (index + 1) * block,
                        index * block : (index + 1) * block,
                    ] = value
                coupling, coupling_context = lattice.expansion_with_context(
                    basis, basis, [1.3, 1.3], vectors, bloch, eta=eta
                )
                value, context = diff.interaction(local, coupling)
                residual = (contexts, coupling_context, context)
                if workload in ("array", "cylindrical-array"):
                    if workload == "cylindrical-array":
                        channels, channel_context = diff.cylindrical_channels(
                            basis,
                            [1.3, 1.3],
                            ports.components,
                            ports.pol,
                            float(vectors[0, 0]),
                        )
                    else:
                        channels, channel_context = diff.spherical_channels(
                            basis,
                            [1.3, 1.3],
                            ports.components,
                            ports.pol,
                            float(abs(np.linalg.det(vectors))),
                        )
                    value, radiation_context = diff.smatrix_from_array(value, channels)
                    residual += (channel_context, radiation_context)
                return value, residual
            return diff.cluster(order, 1.3, radii, epsilon, positions)

        def upstream():
            if workload == "ebcm":
                return reference_qmat(
                    lambda t: 0.3 * (1 + 0.23 * np.cos(t) ** 2),
                    lambda t: -0.138 * np.cos(t) * np.sin(t),
                    surface_ks,
                    surface_zs,
                    (oracle_basis.l, oracle_basis.m, oracle_basis.pol),
                )
            if workload == "internal-field":
                return np.asarray(oracle_lower.illuminate(up, down, smat=oracle_upper))
            if workload == "slab":
                value = treams.SMatrices.slab(
                    layer_thickness, oracle_basis, 1.3, list(layer_eps)
                )
                return np.array(
                    [[np.asarray(value[i, j]) for j in range(2)] for i in range(2)]
                )
            if workload in ("plane-expansion", "cylindrical-plane-expansion"):
                return treams.expand((oracle_basis, oracle_source), k0=1.3)
            if workload in ("plane-field", "plane-operator"):
                operator = np.asarray(
                    treams.efield(points, basis=oracle_basis, k0=1.3, modetype="up")
                )
                return (
                    operator if workload == "plane-operator" else operator @ amplitudes
                )
            if workload == "conversion":
                return treams.expand((oracle_basis, oracle_source), k0=1.3)
            if workload == "periodic-conversion":
                return treams.expandlattice(
                    200, 0.2, basis=(oracle_basis, oracle_source), k0=1.3
                )
            if workload == "rotation":
                return treams.rotate(0.2, 0.7, -0.3, basis=oracle_basis)
            if workload in ("field", "cylindrical-field", "field-operator"):
                operator = np.asarray(
                    treams.efield(
                        points,
                        basis=oracle_basis,
                        k0=1.3,
                        modetype="singular",
                        poltype="helicity",
                    )
                )
                return (
                    operator if workload == "field-operator" else operator @ amplitudes
                )
            if workload == "cylindrical-array":
                cylinders = [
                    treams.TMatrixC.cylinder([0.2], order, 1.3, [r], [e, 1])
                    for r, e in zip(radii, epsilon, strict=True)
                ]
                cluster = treams.TMatrixC.cluster(cylinders, positions)
                response = np.asarray(
                    cluster.latticeinteraction.solve(vectors, bloch, eta=eta)
                )
                incoming = []
                outgoing = []
                for side in ("up", "down"):
                    incoming.append(
                        np.asarray(
                            treams.expand(
                                (cluster.basis, oracle_ports), ("regular", side), k0=1.3
                            )
                        )
                    )
                    outgoing.append(
                        np.asarray(
                            treams.expandlattice(
                                vectors,
                                bloch,
                                basis=(oracle_ports, cluster.basis),
                                modetype=side,
                                k0=1.3,
                            )
                        )
                    )
                return np.array(
                    [
                        [
                            outgoing[i] @ response @ incoming[j]
                            + (np.eye(len(oracle_ports)) if i == j else 0)
                            for j in range(2)
                        ]
                        for i in range(2)
                    ]
                )
            spheres = [
                treams.TMatrix.sphere(order, 1.3, r, [e, 1])
                for r, e in zip(radii, epsilon, strict=True)
            ]
            cluster = treams.TMatrix.cluster(spheres, positions)
            if workload in ("periodic", "array"):
                response = cluster.latticeinteraction.solve(vectors, bloch)
                if workload == "array":
                    scattering = treams.SMatrices.from_array(
                        response, treams.PlaneWaveBasisByComp.default(q)
                    )
                    return np.array(
                        [
                            [np.asarray(scattering[i, j]) for j in range(2)]
                            for i in range(2)
                        ]
                    )
                return response
            return cluster.interaction.solve()

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
        backward_times = []
        if backend == "rust":
            for iteration in range(repeats + 1):
                value, context = (
                    diff.layer_stack(layer_ks, layer_zs, q, layer_thickness)
                    if workload == "slab"
                    else rust()
                )
                cotangent = np.full_like(value, (1 + 0.3j) / value.size)
                start = time.perf_counter()
                if workload in ("periodic", "array", "cylindrical-array"):
                    particle_contexts, coupling_context, solve_context = context[:3]
                    if workload in ("array", "cylindrical-array"):
                        channel_context, radiation_context = context[3:]
                        cotangent, channels_gradient = radiation_context.pullback(
                            cotangent
                        )
                        channel_context.pullback(channels_gradient)
                    local_gradient, coupling_gradient = solve_context.pullback(
                        cotangent
                    )
                    coupling_result = coupling_context.pullback(coupling_gradient)
                    block = len(basis) // particles
                    particle_results = [
                        item.pullback(
                            local_gradient[
                                i * block : (i + 1) * block,
                                i * block : (i + 1) * block,
                            ].copy()
                        )
                        for i, item in enumerate(particle_contexts)
                    ]
                    del coupling_result, particle_results
                else:
                    context.pullback(cotangent)
                elapsed = time.perf_counter() - start
                if iteration:
                    backward_times.append(elapsed)
                del value, context, cotangent
        backward_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
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
                    "ebcm_legacy": True if workload == "ebcm" else None,
                    "ewald_eta": eta
                    if workload in ("periodic", "array", "cylindrical-array")
                    else None,
                    "samples": samples
                    if workload
                    in (
                        "field",
                        "internal-field",
                        "ebcm",
                        "cylindrical-field",
                        "field-operator",
                        "conversion",
                        "periodic-conversion",
                        "plane-field",
                        "plane-operator",
                        "plane-expansion",
                        "cylindrical-plane-expansion",
                    )
                    else None,
                    "particles": particles if workload != "slab" else None,
                    "layers": particles if workload == "slab" else None,
                    "channels": samples if workload == "slab" else None,
                    "lmax": order,
                    "dimension": 2 * samples
                    if workload == "slab"
                    else 2 * particles * order
                    if workload == "internal-field"
                    else 2 * samples * (2 * order + 1)
                    if workload == "periodic-conversion"
                    else particles
                    * (
                        2 * order
                        if workload in ("plane-field", "plane-operator")
                        else 4 * (2 * order + 1)
                        if workload == "cylindrical-field"
                        else 2 * (2 * order + 1)
                        if workload
                        in ("cylindrical-array", "cylindrical-plane-expansion")
                        else 2 * order * (order + 2)
                    ),
                    "threads": threads,
                    "median_seconds": statistics.median(times),
                    "peak_rss_mib": peak,
                    "baseline_rss_mib": baseline,
                    "samples_seconds": times,
                    "backward_median_seconds": statistics.median(backward_times)
                    if backward_times
                    else None,
                    "backward_samples_seconds": backward_times,
                    "forward_and_backward_peak_rss_mib": backward_peak
                    if backward_times
                    else None,
                    "blas": threadpool_info(),
                }
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=[
            "cluster",
            "slab",
            "field",
            "internal-field",
            "ebcm",
            "cylindrical-field",
            "field-operator",
            "periodic",
            "array",
            "rotation",
            "conversion",
            "periodic-conversion",
            "plane-field",
            "plane-operator",
            "plane-expansion",
            "cylindrical-plane-expansion",
            "cylindrical-array",
        ],
        default="cluster",
    )
    parser.add_argument("--samples", "--channels", type=int, default=2048)
    parser.add_argument("--worker", choices=["rust", "treams", "check"])
    parser.add_argument("--particles", "--layers", type=int, default=8)
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

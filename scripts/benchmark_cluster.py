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

    if backend in ("rust", "check", "compare"):
        from treams_rs import (
            CylindricalWaveBasis,
            PlaneWaveBasisByComp,
            SMatrices,
            SphericalWaveBasis,
            _native,
            diff,
            lattice,
            special,
        )

        if _native.build_profile() != "release":
            raise RuntimeError("benchmark requires just build-ext-release")
    if backend in ("treams", "check", "compare"):
        import treams

    # Forward-only APIs return their actual result, without a benchmark-created
    # dummy residual tuple that would burden only one side of tiny-kernel timings.
    forward_only = workload.endswith(("-forward", "-public")) or workload == "slab"
    threads = int(os.environ["BENCH_THREADS"])
    with threadpool_limits(limits=threads):
        radii = np.linspace(0.15, 0.25, particles)
        epsilon = np.full(particles, 4 + 0.1j)
        positions = np.column_stack(
            [np.arange(particles) * 0.8, np.zeros((particles, 2))]
        )

        if workload.startswith("wave-"):
            wave_name = workload.split("-")[1]
            varying = 0.8 if samples == 1 else np.linspace(0.4, 1.2, samples)
            wave_labels = {"degree": order, "order": 1, "polarization": 1}
            if wave_name == "sph_harm":
                wave_args = (1, order, 0.3, varying)
            elif wave_name.startswith("vsh"):
                wave_arguments = (varying, 0.3)
                wave_args = (order, 1, *wave_arguments)
            elif wave_name.startswith("vsw"):
                wave_arguments = (1.2 + 0.1j, varying, 0.3)
                wave_args = (order, 1, *wave_arguments)
            elif wave_name.startswith("vcw"):
                wave_arguments = (0.2, varying + 0.1j, 0.3, 0.4)
                if not wave_name.endswith("M"):
                    wave_arguments = (*wave_arguments, 1.3 + 0.1j)
                wave_args = (wave_arguments[0], 1, *wave_arguments[1:])
            else:
                wave_arguments = (0.3 + 0.03j, 0.4, 1.2 + 0.04j, varying, 0.5, 0.6)
                wave_args = wave_arguments
            if wave_name.endswith("A"):
                wave_args = (*wave_args, 1)

        if workload.startswith("coordinate-"):
            coordinate_name = workload.split("-")[1]
            dimension = 2 if "pol" in coordinate_name else 3
            coordinate_points = (
                np.array([0.4, 0.7, 0.3])[:dimension]
                if samples == 1
                else np.column_stack(
                    (
                        np.linspace(0.2, 1.5, samples),
                        np.full(samples, 0.7),
                        np.full(samples, 0.3),
                    )
                )[:, :dimension].copy()
            )
            coordinate_vector = np.array([0.2 + 0.3j, -0.1 + 0.2j, 0.4 - 0.1j])[
                :dimension
            ]

        if workload.startswith(("cylindrical-expansion", "cylindrical-periodic")):
            vectors = np.array([[particles * 0.8]])
            bloch = np.array([0.1])
            if backend in ("rust", "check", "compare"):
                basis = CylindricalWaveBasis.default(
                    [0.2, -0.3], order, particles, positions
                )
            if backend in ("treams", "check", "compare"):
                oracle_basis = treams.CylindricalWaveBasis.default(
                    [0.2, -0.3], order, particles, positions
                )

        if workload in ("internal-field", "internal-field-forward"):
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
            if backend in ("treams", "check", "compare"):
                q = np.column_stack(
                    [np.linspace(0.1, 0.8, n // 2), np.full(n // 2, 0.2)]
                )
                oracle_basis = treams.PlaneWaveBasisByComp.default(q)
                oracle_lower = treams.SMatrices(lower, basis=oracle_basis, k0=1.3)
                oracle_upper = treams.SMatrices(upper, basis=oracle_basis, k0=1.3)

        if workload.startswith(("wigner", "incgamma", "intkambe")):
            special_arguments = (
                0.7 + 0.1j if samples == 1 else np.linspace(0.3, 1.3, samples) + 0.1j
            )
            wigner_degrees = order if samples == 1 else np.full(samples, order)

        if workload.startswith("angular-"):
            angular_kind = workload.split("-")[1]
            angular_arguments = (
                0.3 + 0.1j if samples == 1 else np.linspace(-0.8, 0.8, samples) + 0.1j
            )
            angular_name = {"legendre": "lpmv", "pi": "pi_fun", "tau": "tau_fun"}[
                angular_kind
            ]
            angular_labels = (2, order) if angular_kind == "legendre" else (order, 2)

        if workload.startswith("bessel"):
            bessel_arguments = (
                1.3 + 0.2j if samples == 1 else np.linspace(0.6, 8.0, samples) + 0.2j
            )

        if "particle-cluster" in workload:
            degrees = [order + i % 2 for i in range(particles)]
            cylindrical_particles = workload.startswith("cylindrical-")
            particle_dimension = sum(
                4 * (2 * degree + 1)
                if cylindrical_particles
                else 2 * degree * (degree + 2)
                for degree in degrees
            )
            if backend in ("rust", "check", "compare"):
                from treams_rs import TMatrix, TMatrixC

                local_tmats = [
                    TMatrixC.cylinder([0.2, 0.4], degree, 1.3, radius, [eps, 1])
                    if cylindrical_particles
                    else TMatrix.sphere(degree, 1.3, radius, [eps, 1])
                    for degree, radius, eps in zip(degrees, radii, epsilon, strict=True)
                ]
                local_bases = [tm.basis for tm in local_tmats]
                local_arrays = [tm.array for tm in local_tmats]
            if backend in ("treams", "check", "compare"):
                oracle_tmats = [
                    treams.TMatrixC.cylinder([0.2, 0.4], degree, 1.3, radius, [eps, 1])
                    if cylindrical_particles
                    else treams.TMatrix.sphere(degree, 1.3, radius, [eps, 1])
                    for degree, radius, eps in zip(degrees, radii, epsilon, strict=True)
                ]

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
            if backend in ("rust", "check", "compare"):
                basis = SphericalWaveBasis.default(order)
            if backend in ("treams", "check", "compare"):
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
            if backend in ("rust", "check", "compare"):
                basis = PlaneWaveBasisByComp.default(q)
            if backend in ("treams", "check", "compare"):
                oracle_basis = treams.PlaneWaveBasisByComp.default(q)

        if workload in (
            "plane-field",
            "plane-operator",
            "plane-phases",
            "plane-permutation",
            "oriented-chirality",
        ):
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
            if backend in ("rust", "check", "compare"):
                basis = PlaneWaveBasisByComp.default(q)
                vectors = np.column_stack(basis.kvecs(1.3))
            if backend in ("treams", "check", "compare"):
                oracle_basis = treams.PlaneWaveBasisByComp.default(q)
                oracle_vectors = np.column_stack(oracle_basis.kvecs(1.3))

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
            if backend in ("rust", "check", "compare"):
                basis = SphericalWaveBasis.default(order)
                source_basis = CylindricalWaveBasis.default(kz, order)
                if workload == "periodic-conversion":
                    basis, source_basis = source_basis, basis
            if backend in ("treams", "check", "compare"):
                oracle_basis = treams.SphericalWaveBasis.default(order)
                oracle_source = treams.CylindricalWaveBasis.default(kz, order)
                if workload == "periodic-conversion":
                    oracle_basis, oracle_source = oracle_source, oracle_basis

        if workload in ("rotation", "plane-expansion"):
            if backend in ("rust", "check", "compare"):
                basis = SphericalWaveBasis.default(order, particles, positions)
            if backend in ("treams", "check", "compare"):
                oracle_basis = treams.SphericalWaveBasis.default(
                    order, particles, positions
                )

        if workload == "plane-expansion":
            q = np.column_stack([np.linspace(0.1, 1.7, samples), np.full(samples, 0.2)])
            if backend in ("rust", "check", "compare"):
                source_basis = PlaneWaveBasisByComp.default(q)
                vectors = np.column_stack(source_basis.kvecs(1.3))
            if backend in ("treams", "check", "compare"):
                oracle_source = treams.PlaneWaveBasisByComp.default(q)

        if workload == "cylindrical-plane-expansion":
            q = np.column_stack([np.full(samples, 0.2), np.linspace(0.1, 1.7, samples)])
            if backend in ("rust", "check", "compare"):
                basis = CylindricalWaveBasis.default([0.2], order, particles, positions)
                source_basis = PlaneWaveBasisByComp.default(q, "zx")
                vectors = np.column_stack(source_basis.kvecs(1.3))
            if backend in ("treams", "check", "compare"):
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
            if backend in ("rust", "check", "compare"):
                basis = SphericalWaveBasis.default(order, particles, positions)
                ports = PlaneWaveBasisByComp.default(q)

        if workload in (
            "field",
            "cylindrical-field",
            "cylindrical-field-axial",
            "field-operator",
        ):
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
                if workload.startswith("cylindrical-field")
                else 2 * order * (order + 2)
            )
            amplitudes = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
            if backend in ("rust", "check", "compare"):
                basis = (
                    CylindricalWaveBasis.default(
                        [0.2, -0.3], order, particles, positions
                    )
                    if workload.startswith("cylindrical-field")
                    else SphericalWaveBasis.default(order, particles, positions)
                )
            if backend in ("treams", "check", "compare"):
                oracle_basis = (
                    treams.CylindricalWaveBasis.default(
                        [0.2, -0.3], order, particles, positions
                    )
                    if workload.startswith("cylindrical-field")
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
            if backend in ("rust", "check", "compare"):
                basis = CylindricalWaveBasis.default([0.2], order, particles, positions)
                ports = PlaneWaveBasisByComp.default(q, "zx")
            if backend in ("treams", "check", "compare"):
                oracle_ports = treams.PlaneWaveBasisByComp.default(q, "zx")

        # The reference cylinder sum loses accuracy for larger cells at eta=0.
        # Use the same converged split for both implementations.
        eta = (
            0.7
            if workload == "cylindrical-array"
            or workload.startswith("cylindrical-periodic")
            else 0
        )

        def rust():
            if workload.startswith("wave-"):
                if forward_only:
                    return getattr(special, wave_name)(*wave_args)
                return diff.vector_wave(*wave_arguments, kind=wave_name, **wave_labels)
            if workload.startswith("coordinate-"):
                args = (
                    (coordinate_vector, coordinate_points)
                    if coordinate_name.startswith("v")
                    else (coordinate_points,)
                )
                return getattr(special, coordinate_name)(*args)
            if workload.startswith("cylindrical-expansion"):
                return diff.expansion(basis, basis, [1.3, 1.3], singular=True)
            if workload.startswith("cylindrical-periodic"):
                return lattice.expansion_with_context(
                    basis, basis, [1.3, 1.3], vectors, bloch, eta=eta
                )
            if workload == "wigner":
                return diff.wigner(order, 1, -2, 0.2, special_arguments, -0.1)
            if workload == "wigner-forward":
                return special.wignerd(order, 1, -2, 0.2, special_arguments, -0.1)
            if workload == "wigner-small-forward":
                return special.wignersmalld(order, 1, -2, special_arguments)
            if workload == "wigner3j-forward":
                return special.wigner3j(order, order, wigner_degrees, 1, -2, 1)
            if workload == "incgamma-forward":
                return special.incgamma(1.5, special_arguments)
            if workload == "intkambe-forward":
                return special.intkambe(-2, special_arguments, 0.7 + 0.1j)
            if workload.startswith("angular-"):
                if workload.endswith("-forward"):
                    return getattr(special, angular_name)(
                        *angular_labels, angular_arguments
                    )
                return diff.angular(order, 2, angular_arguments, kind=angular_kind)
            if workload == "bessel-forward":
                return special.hankel1(order, bessel_arguments)
            if workload == "bessel-derivative-forward":
                return special.hankel1_d(order, bessel_arguments)
            if workload.startswith("bessel"):
                return diff.bessel(
                    order,
                    bessel_arguments,
                    kind="h1",
                    derivative=workload == "bessel-derivative",
                )
            if "particle-cluster" in workload and workload.endswith("public"):
                return (
                    type(local_tmats[0])
                    .cluster(local_tmats, positions)
                    .interaction.solve()
                    .array
                )
            if "particle-cluster" in workload:
                return diff.particle_cluster(
                    local_arrays, positions, [1.3, 1.3], bases=local_bases
                )
            if workload == "oriented-chirality":
                return diff.oriented_chirality(
                    vectors[:, :2].real,
                    vectors[:, 2],
                    (-0.2, 0.7),
                    polarizations=basis.pol,
                    axis=0,
                )
            if workload == "plane-permutation":
                return diff.plane_permutation(vectors, basis.pol)
            if workload == "plane-phases":
                return diff.plane_phases(points, vectors)
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
            if workload in ("internal-field", "internal-field-forward"):
                if workload == "internal-field-forward":
                    return _native.smatrix_illuminate_forward(lower, upper, up, down)
                return diff.smatrix_illuminate(lower, upper, up, down)
            if workload == "slab":
                return SMatrices.slab(
                    layer_thickness, basis, 1.3, list(layer_eps)
                ).array
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
            if workload in ("field", "cylindrical-field", "cylindrical-field-axial"):
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
            if workload.startswith("wave-"):
                return getattr(treams.special, wave_name)(*wave_args)
            if workload.startswith("coordinate-"):
                args = (
                    (coordinate_vector, coordinate_points)
                    if coordinate_name.startswith("v")
                    else (coordinate_points,)
                )
                return getattr(treams.special, coordinate_name)(*args)
            if workload.startswith("cylindrical-expansion"):
                return treams.expand(
                    (oracle_basis, oracle_basis),
                    k0=1.3,
                    modetype=("regular", "singular"),
                )
            if workload.startswith("cylindrical-periodic"):
                return treams.expandlattice(
                    float(vectors[0, 0]),
                    float(bloch[0]),
                    basis=(oracle_basis, oracle_basis),
                    k0=1.3,
                    eta=eta,
                )
            if workload in ("wigner", "wigner-forward"):
                return treams.special.wignerd(
                    order, 1, -2, 0.2, special_arguments, -0.1
                )
            if workload == "wigner-small-forward":
                return treams.special.wignersmalld(order, 1, -2, special_arguments)
            if workload == "wigner3j-forward":
                return treams.special.wigner3j(order, order, wigner_degrees, 1, -2, 1)
            if workload == "incgamma-forward":
                return treams.special.incgamma(1.5, special_arguments)
            if workload == "intkambe-forward":
                return treams.special.intkambe(-2, special_arguments, 0.7 + 0.1j)
            if workload.startswith("angular-"):
                return getattr(treams.special, angular_name)(
                    *angular_labels, angular_arguments
                )
            if workload.startswith("bessel"):
                function = (
                    treams.special.hankel1_d
                    if "derivative" in workload
                    else treams.special.hankel1
                )
                return function(order, bessel_arguments)
            if "particle-cluster" in workload:
                return (
                    type(oracle_tmats[0])
                    .cluster(oracle_tmats, positions)
                    .interaction.solve()
                )
            if workload == "oriented-chirality":
                q0, q1, normal = oracle_vectors.T
                up_polarization = treams.special.vpw_A(
                    normal, q0, q1, 0, 0, 0, oracle_basis.pol
                )
                down_polarization = treams.special.vpw_A(
                    -normal, q0, q1, 0, 0, 0, oracle_basis.pol
                )
                sign = 2 * oracle_basis.pol - 1

                def mean(slope):
                    width = slope * 0.9
                    result = np.ones_like(width)
                    np.divide(np.expm1(width), width, out=result, where=width != 0)
                    return np.exp(-0.2 * slope) * result

                return np.array(
                    [
                        2
                        * sign
                        * np.sum(up_polarization.conj() * up_polarization, axis=-1)
                        * mean(-2 * normal.imag),
                        2
                        * sign
                        * np.sum(down_polarization.conj() * down_polarization, axis=-1)
                        * mean(2 * normal.imag),
                        4
                        * sign
                        * np.sum(down_polarization.conj() * up_polarization, axis=-1)
                        * mean(2j * normal.real),
                    ]
                )
            if workload == "plane-permutation":
                return treams.pw.permute_xyz(
                    *oracle_vectors.T, np.arange(2)[:, None], oracle_basis.pol[None, :]
                )
            if workload == "plane-phases":
                return treams.pw.translate(
                    *oracle_vectors.T,
                    points[:, None, 0],
                    points[:, None, 1],
                    points[:, None, 2],
                )
            if workload == "ebcm":
                return reference_qmat(
                    lambda t: 0.3 * (1 + 0.23 * np.cos(t) ** 2),
                    lambda t: -0.138 * np.cos(t) * np.sin(t),
                    surface_ks,
                    surface_zs,
                    (oracle_basis.l, oracle_basis.m, oracle_basis.pol),
                )
            if workload in ("internal-field", "internal-field-forward"):
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
            if workload in (
                "field",
                "cylindrical-field",
                "cylindrical-field-axial",
                "field-operator",
            ):
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

        if backend in ("check", "compare"):
            expected = upstream()
            actual = rust()
            if not forward_only:
                actual = actual[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-12)
            if backend == "check":
                print(json.dumps({"accuracy_check": "passed"}))
                return
            # Tiny kernels need both implementations in the same process, with
            # alternating order, to separate kernel cost from process/core drift.
            batch = 1
            while True:
                durations = []
                for function in (upstream, rust):
                    start = time.perf_counter()
                    for _ in range(batch):
                        function()
                    durations.append(time.perf_counter() - start)
                if min(durations) >= 0.02:
                    break
                batch *= 10
            pairs = []
            for index in range(2 * repeats):
                pair = {}
                functions = (("treams", upstream), ("rust", rust))
                for name, function in functions if index % 2 == 0 else functions[::-1]:
                    start = time.perf_counter()
                    for _ in range(batch):
                        function()
                    pair[name] = (time.perf_counter() - start) / batch
                pairs.append(pair)
            print(
                json.dumps(
                    {
                        "method": "paired_alternating_process",
                        "calls_per_sample": batch,
                        "pairs_seconds": pairs,
                        "speedup": statistics.median(
                            p["treams"] / p["rust"] for p in pairs
                        ),
                        "cpu_affinity": sorted(os.sched_getaffinity(0))
                        if hasattr(os, "sched_getaffinity")
                        else None,
                    }
                )
            )
            return
        baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        function = rust if backend == "rust" else upstream
        function()
        # Microsecond kernels need sustained samples, not seven individual calls
        # dominated by timer noise, cold caches and CPU frequency ramp-up.
        batch = 1
        while True:
            start = time.perf_counter()
            for _ in range(batch):
                function()
            if time.perf_counter() - start >= 0.02:
                break
            batch *= 10
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(batch):
                function()
            times.append((time.perf_counter() - start) / batch)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        backward_times = []
        if backend == "rust" and not workload.endswith(("-forward", "-public")):
            sample_total = 0.0
            for iteration in range((repeats + 1) * batch):
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
                elif workload.endswith("-axial"):
                    context.pullback_axial(cotangent)
                else:
                    context.pullback(cotangent)
                elapsed = time.perf_counter() - start
                sample_total += elapsed
                if (iteration + 1) % batch == 0:
                    if iteration >= batch:
                        backward_times.append(sample_total / batch)
                    sample_total = 0.0
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
                    "oracle": "Cartesian treams plane fields"
                    if workload == "oriented-chirality"
                    else "treams public operation",
                    "ebcm_legacy": True if workload == "ebcm" else None,
                    "ewald_eta": eta
                    if workload in ("periodic", "array", "cylindrical-array")
                    or workload.startswith("cylindrical-periodic")
                    else None,
                    "samples": samples
                    if workload
                    in (
                        "field",
                        "internal-field",
                        "internal-field-forward",
                        "ebcm",
                        "cylindrical-field",
                        "cylindrical-field-axial",
                        "field-operator",
                        "conversion",
                        "periodic-conversion",
                        "plane-field",
                        "plane-operator",
                        "plane-phases",
                        "plane-permutation",
                        "oriented-chirality",
                        "plane-expansion",
                        "cylindrical-plane-expansion",
                    )
                    else None,
                    "particles": particles if workload != "slab" else None,
                    "layers": particles if workload == "slab" else None,
                    "channels": samples if workload == "slab" else None,
                    "lmax": order,
                    "dimension": particles * 4 * (2 * order + 1)
                    if workload.startswith(
                        ("cylindrical-expansion", "cylindrical-periodic")
                    )
                    else samples
                    if workload.startswith(
                        (
                            "bessel",
                            "angular-",
                            "wigner",
                            "incgamma",
                            "intkambe",
                            "coordinate-",
                            "wave-",
                        )
                    )
                    else particle_dimension
                    if "particle-cluster" in workload
                    else 2 * samples
                    if workload == "slab"
                    else 2 * particles * order
                    if workload in ("internal-field", "internal-field-forward")
                    else 2 * samples * (2 * order + 1)
                    if workload == "periodic-conversion"
                    else particles
                    * (
                        2 * order
                        if workload
                        in (
                            "plane-field",
                            "plane-operator",
                            "plane-phases",
                            "plane-permutation",
                            "oriented-chirality",
                        )
                        else 4 * (2 * order + 1)
                        if workload.startswith("cylindrical-field")
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
                    "calls_per_timing_sample": batch,
                    "forward_includes_result_destruction": True,
                    "forward_records_adjoint": not forward_only,
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
            "wave-sph_harm-forward",
            "wave-vsh_X-forward",
            "wave-vsh_Y-forward",
            "wave-vsh_Z-forward",
            "wave-vsw_M-forward",
            "wave-vsw_N-forward",
            "wave-vsw_A-forward",
            "wave-vsw_rM-forward",
            "wave-vsw_rN-forward",
            "wave-vsw_rA-forward",
            "wave-vcw_M-forward",
            "wave-vcw_N-forward",
            "wave-vcw_A-forward",
            "wave-vcw_rM-forward",
            "wave-vcw_rN-forward",
            "wave-vcw_rA-forward",
            "wave-vpw_M-forward",
            "wave-vpw_N-forward",
            "wave-vpw_A-forward",
            "wave-vsw_rA",
            "wave-vcw_rA",
            "wave-vpw_A",
            "wigner",
            "wigner-forward",
            "wigner-small-forward",
            "wigner3j-forward",
            "incgamma-forward",
            "intkambe-forward",
            "angular-legendre",
            "angular-pi",
            "angular-tau",
            "angular-legendre-forward",
            "angular-pi-forward",
            "angular-tau-forward",
            "bessel-forward",
            "bessel-derivative-forward",
            "bessel",
            "bessel-derivative",
            "cluster",
            "particle-cluster",
            "cylindrical-particle-cluster",
            "particle-cluster-public",
            "cylindrical-particle-cluster-public",
            "slab",
            "field",
            "internal-field",
            "internal-field-forward",
            "ebcm",
            "coordinate-car2cyl-forward",
            "coordinate-car2sph-forward",
            "coordinate-cyl2car-forward",
            "coordinate-cyl2sph-forward",
            "coordinate-sph2car-forward",
            "coordinate-sph2cyl-forward",
            "coordinate-car2pol-forward",
            "coordinate-pol2car-forward",
            "coordinate-vcar2cyl-forward",
            "coordinate-vcar2sph-forward",
            "coordinate-vcyl2car-forward",
            "coordinate-vcyl2sph-forward",
            "coordinate-vsph2car-forward",
            "coordinate-vsph2cyl-forward",
            "coordinate-vcar2pol-forward",
            "coordinate-vpol2car-forward",
            "cylindrical-expansion",
            "cylindrical-expansion-axial",
            "cylindrical-periodic",
            "cylindrical-periodic-axial",
            "cylindrical-field",
            "cylindrical-field-axial",
            "field-operator",
            "periodic",
            "array",
            "rotation",
            "conversion",
            "periodic-conversion",
            "plane-field",
            "plane-operator",
            "plane-phases",
            "plane-permutation",
            "oriented-chirality",
            "plane-expansion",
            "cylindrical-plane-expansion",
            "cylindrical-array",
        ],
        default="cluster",
    )
    parser.add_argument("--samples", "--channels", type=int, default=2048)
    parser.add_argument("--worker", choices=["rust", "treams", "check", "compare"])
    parser.add_argument("--particles", "--layers", type=int, default=8)
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--require-speedup",
        type=float,
        default=0,
        help="Fail if upstream/Rust median runtime falls below this ratio",
    )
    parser.add_argument(
        "--require-rss-ratio",
        type=float,
        help="Fail if Rust/upstream peak RSS exceeds this ratio",
    )
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[: args.threads]))
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
            "--threads",
            str(args.threads),
        ]
        try:
            result = subprocess.run(
                command, env=env, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as error:
            raise SystemExit(error.stderr or str(error)) from error
        if backend != "check":
            results.append(json.loads(result.stdout))
    comparison = {
        "method": "isolated_process",
        "speedup": results[0]["median_seconds"] / results[1]["median_seconds"],
    }
    # Pair every sub-millisecond workload, not just failed or borderline results.
    # Separate processes above remain authoritative for peak RSS and adjoint costs.
    if max(r["median_seconds"] for r in results) < 0.001:
        command[command.index("--worker") + 1] = "compare"
        result = subprocess.run(
            command, env=env, check=True, capture_output=True, text=True
        )
        comparison = json.loads(result.stdout)
    print(
        json.dumps(
            {
                "results": results,
                "timing_comparison": comparison,
                "speedup": comparison["speedup"],
            },
            indent=2,
        )
    )
    if comparison["speedup"] < args.require_speedup:
        raise SystemExit(
            "Performance gate failed: Rust runtime exceeds the required ratio"
        )
    if (
        args.require_rss_ratio is not None
        and results[1]["peak_rss_mib"] / results[0]["peak_rss_mib"]
        > args.require_rss_ratio
    ):
        raise SystemExit(
            "Performance gate failed: Rust peak RSS exceeds the required ratio"
        )


if __name__ == "__main__":
    main()

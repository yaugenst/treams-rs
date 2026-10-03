# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.1", "scipy>=1.16,<1.17", "treams==0.4.7", "threadpoolctl>=3.7", "mpmath==1.3.0"]
# ///
"""High-cutoff conditioning collector for benchmark_cluster.py's eight-sphere chain.

For each cutoff it compares the native and upstream coupled, local and interaction
matrices under the benchmark's entrywise tolerance (atol 1e-12, rtol 2e-9) and
records backward errors, 1-norm condition estimates, passivity and the cutoff
convergence of three illuminated responses. From lmax 6 on, it also checks the
native response against an independently balanced upstream system with a
forward-error bound, under the same entrywise tolerance. It does not modify the
numerical solvers.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path


def backward_errors(system, solution, rhs, *, componentwise=True):
    """Normwise Frobenius and componentwise backward residuals of A X = B."""
    import numpy as np

    residual = system @ solution - rhs
    denominator = np.linalg.norm(system) * np.linalg.norm(solution) + np.linalg.norm(
        rhs
    )
    normwise = float(np.linalg.norm(residual) / max(float(denominator), 1e-300))
    result = {"normwise_backward_error": normwise}
    if componentwise:
        scale = abs(system) @ abs(solution) + abs(rhs)
        ratios = np.divide(
            abs(residual), scale, out=np.zeros_like(scale), where=scale > 0
        )
        ratios[(scale == 0) & (abs(residual) != 0)] = np.inf
        result["componentwise_backward_error"] = float(np.max(ratios, initial=0))
    return result


def condition_estimate(system):
    """LAPACK reciprocal 1-norm condition estimate; one LU, no dense SVD."""
    import numpy as np
    from scipy.linalg import lu_factor
    from scipy.linalg.lapack import get_lapack_funcs

    lu, _ = lu_factor(system)
    gecon = get_lapack_funcs("gecon", (lu,))
    anorm = float(np.linalg.norm(system, 1))
    reciprocal, info = gecon(lu, anorm, norm="1")
    if info:
        raise RuntimeError(f"LAPACK gecon failed with info={info}")
    return {
        "system_reciprocal_condition_1_estimate": float(reciprocal),
        "system_condition_1_estimate": float(1 / reciprocal)
        if reciprocal > 0
        else None,
        "system_norm_1": anorm,
        "maximum_abs_system_entry": float(np.max(abs(system))),
        "method": "SciPy complex LU plus LAPACK gecon, 1-norm estimate",
        "matrix_dimension": len(system),
    }


def compare(c, actual, expected, *, prefix, **details):
    """Use exactly the benchmark's atol/rtol; no failures are suppressed."""
    import numpy as np
    from qualify_upstream import residuals

    metrics, evidence = residuals(actual, expected, rtol=2e-9, atol=1e-12)
    evidence["failed_entry_count"] = int(
        np.count_nonzero(~np.isclose(actual, expected, rtol=2e-9, atol=1e-12))
    )
    for metric, error in metrics.items():
        c.add(
            f"{prefix}_{metric}",
            float("nan") if error is None or evidence["nonfinite_count"] else error,
            1 if metric == "max_scaled_error" else None,
            reference_kind="upstream",
            **evidence,
            **details,
        )


def _coefficient_scaled_reference(system, local):
    """Independent SciPy solve; scaling depends only on upstream local coefficients."""
    import numpy as np
    from scipy.linalg import solve

    scale = np.sqrt(np.max(abs(local), axis=1))
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError("Coefficient scaling requires finite nonzero local rows")
    balanced = system * scale[None, :] / scale[:, None]
    rhs = local / scale[:, None]
    solution = scale[:, None] * solve(balanced, rhs, assume_a="gen")
    return solution, scale, balanced


def _encoded_system_bound(system, local, solution, scale, balanced):
    """Enclose each solution column's error for the encoded original A X = T.

    Complex component 1-norms avoid square-root rounding in norm bounds. gamma_k
    envelopes scaling, reductions, and conventional complex BLAS dot products;
    a smallest-normal allowance also covers gradual underflow or FTZ. Overflow
    or a noncontractive scaled matrix leaves this certificate unsupported.
    """
    import math

    import numpy as np

    n = len(system)
    u = np.finfo(float).eps / 2
    tiny = np.finfo(float).tiny

    def gamma(k):
        return math.nextafter(k * u / (1 - k * u), math.inf)

    def upper(value):
        return np.nextafter(np.asarray(value) * (1 + 32 * u) + tiny, np.inf)

    def magnitude(value):
        return abs(value.real) + abs(value.imag)

    def row_norm(value):
        return float(
            upper(np.max(np.sum(magnitude(value), axis=1)) / (1 - gamma(n + 2)))
        )

    balanced_norm = row_norm(balanced)
    scaling_defect = float(
        upper(
            gamma(4) / (1 - gamma(4)) * balanced_norm
            + 8 * n * tiny * (1 + 1 / np.min(scale))
        )
    )
    q = float(
        upper(
            row_norm(np.eye(n) - balanced)
            + scaling_defect
            + gamma(1) * (1 + balanced_norm)
        )
    )
    if not math.isfinite(q) or q >= 1:
        raise ValueError(
            f"No Neumann certificate: scaled off-diagonal norm upper bound {q}"
        )
    weighted = solution / scale[:, None]
    rhs = local / scale[:, None]
    residual = rhs - balanced @ weighted
    ymax = upper(np.max(magnitude(weighted), axis=0))
    cmax = upper(np.max(magnitude(rhs), axis=0))
    rmax = upper(np.max(magnitude(residual), axis=0))
    # Includes complex products/accumulation, final subtraction, and abs/sum.
    arithmetic = upper(gamma(8 * n + 8) * (balanced_norm * ymax + cmax) + 16 * n * tiny)
    scaling = upper(
        gamma(3) / (1 - gamma(3)) * (cmax + (balanced_norm + scaling_defect) * ymax)
        + scaling_defect * ymax
        + 16 * n * tiny
    )
    residual_upper = upper(rmax + arithmetic + scaling)
    weighted_error = upper(residual_upper / (1 - q))
    error = upper(np.max(scale) * weighted_error)
    if not np.all(np.isfinite(error)):
        raise ValueError("Overflow in encoded-system error certificate")
    return error, {
        "norm": "maximum absolute real-plus-imaginary component per RHS column",
        "scaled_off_diagonal_norm_upper": q,
        "inverse_norm_upper": float(upper(1 / (1 - q))),
        "scaling_defect_norm_upper": scaling_defect,
        "maximum_weighted_residual": float(np.max(rmax)),
        "per_column_weighted_residual_upper_bound": residual_upper.tolist(),
        "maximum_rounding_envelope": float(np.max(arithmetic + scaling)),
        "maximum_absolute_forward_error_bound": float(np.max(error)),
        "per_column_absolute_forward_error_bound": error.tolist(),
        "scope": "Exact solution of the encoded upstream A X = T system; excludes matrix assembly, multipole truncation and physical-model error.",
        "arithmetic": "IEEE binary64 conventional complex BLAS operations; gamma_(8n+8) dot-product envelope, positive norm-reduction envelopes, scaling-rounding terms, outward scalar padding, and smallest-normal underflow allowances. Certificate refuses overflow or q>=1.",
    }


def _high_precision_residuals(system, local, solution, scale, columns):
    """Independent 80/120-digit checks of selected encoded-equation residuals."""
    import math

    import mpmath as mp
    import numpy as np

    output = []
    for column in columns:
        # Include all rows in the double residual, then independently evaluate
        # its worst scaled row with scalar multiprecision arithmetic.
        approximate = (local[:, column] - system @ solution[:, column]) / scale
        row = int(np.argmax(abs(approximate)))
        values = []
        for precision in (80, 120):
            with mp.workdps(precision):

                def exact(z):
                    return mp.mpc(float(z.real), float(z.imag))

                residual = (
                    exact(local[row, column])
                    - mp.fsum(
                        exact(a) * exact(x)
                        for a, x in zip(system[row], solution[:, column], strict=True)
                    )
                ) / mp.mpf(float(scale[row]))
                values.append(abs(residual.real) + abs(residual.imag))
        with mp.workdps(160):
            precision_change = float(abs(values[1] - values[0]))
        output.append(
            {
                "row": row,
                "column": int(column),
                "weighted_residual_80_digits": float(values[0]),
                "weighted_residual_80_digits_text": mp.nstr(values[0], 80),
                "weighted_residual_120_digits": float(values[1]),
                "weighted_residual_120_digits_upper": math.nextafter(
                    float(values[1]), math.inf
                ),
                "weighted_residual_120_digits_text": mp.nstr(values[1], 120),
                "precision_change": precision_change,
            }
        )
    return output


def certify_cluster_reference(
    actual, upstream, *, lmax, k0, radii, epsilon, positions, rtol=2e-9, atol=1e-12
):
    """Qualify native output against a stable, independently assembled reference.

    The disagreement with treams stays recorded. The independent reference must
    carry a full encoded-equation forward-error bound, and native/reference
    distance plus that uncertainty must meet the benchmark's elementwise tolerance.
    This function imports no Rust implementation or native solver.
    """
    import hashlib

    import numpy as np
    import treams
    from _harness import file_sha256
    from benchmark_illumination import fingerprints
    from qualify_upstream import residuals

    original, details = residuals(actual, upstream, rtol=rtol, atol=atol)
    source = {
        **fingerprints(upstream=True),
        "reference_collector_sha256": file_sha256(__file__),
    }
    result = {
        "passed": False,
        "reference_kind": "independently_balanced_upstream_system",
        "source": source,
        "original_disagreement": {**original, **details},
        "parameters": {
            "lmax": lmax,
            "k0": k0,
            "particles": len(radii),
            "rtol": rtol,
            "atol": atol,
        },
    }
    try:
        local = treams.TMatrix.cluster(
            [
                treams.TMatrix.sphere(lmax, k0, r, [e, 1])
                for r, e in zip(radii, epsilon, strict=True)
            ],
            positions,
        )
        system, rhs = np.asarray(local.interaction()), np.asarray(local)
        reference, scale, balanced = _coefficient_scaled_reference(system, rhs)
        bound, certificate = _encoded_system_bound(
            system, rhs, reference, scale, balanced
        )
        actual = np.asarray(actual)
        if actual.shape != system.shape or np.shape(upstream) != system.shape:
            raise ValueError("Response shape does not match the reconstructed system")
        if not np.all(np.isfinite(actual)) or atol <= 0 or rtol < 0:
            raise ValueError("Require finite responses and positive absolute tolerance")
        difference = abs(actual - reference)
        u = np.finfo(float).eps / 2
        tiny = np.finfo(float).tiny
        difference_upper = np.nextafter(
            difference * (1 + 8 * u)
            + 8 * u * (abs(actual) + abs(reference))
            + 8 * tiny,
            np.inf,
        )
        reference_magnitude_lower = np.nextafter(abs(reference) * (1 - 8 * u), 0)
        exact_reference_minimum = np.maximum(
            np.nextafter(reference_magnitude_lower - bound[None, :], -np.inf), 0
        )
        gate = np.nextafter(atol + np.nextafter(rtol * exact_reference_minimum, 0), 0)
        scaled = np.nextafter(
            np.nextafter(difference_upper + bound[None, :], np.inf) / gate, np.inf
        )
        failed = (~np.isfinite(scaled)) | (scaled > 1)
        metrics, native_details = residuals(actual, reference, rtol=rtol, atol=atol)
        selected = sorted(
            {
                0,
                len(reference) - 1,
                int(np.unravel_index(np.argmax(difference), difference.shape)[1]),
            }
        )
        precision_checks = _high_precision_residuals(
            system, rhs, reference, scale, selected
        )
        precision_stable = all(
            row["precision_change"] <= 1e-60 for row in precision_checks
        )
        for row in precision_checks:
            row["weighted_residual_enclosure"] = certificate[
                "per_column_weighted_residual_upper_bound"
            ][row["column"]]
            row["within_residual_enclosure"] = (
                row["weighted_residual_120_digits_upper"]
                <= row["weighted_residual_enclosure"]
            )
        residual_checks_passed = all(
            row["within_residual_enclosure"] for row in precision_checks
        )
        raw_delta = abs(np.asarray(upstream) - reference)
        raw_delta_lower = np.maximum(
            np.nextafter(
                raw_delta * (1 - 8 * u)
                - 8 * u * (abs(upstream) + abs(reference))
                - 8 * tiny,
                -np.inf,
            ),
            0,
        )
        raw_error_lower = np.maximum(
            np.nextafter(raw_delta_lower - bound[None, :], -np.inf), 0
        )
        result.update(
            {
                "upstream_absolute_error_lower_bound": float(np.max(raw_error_lower)),
                "upstream_vs_balanced_max_absolute_difference": float(
                    np.max(raw_delta)
                ),
                "passed": not bool(np.any(failed))
                and precision_stable
                and residual_checks_passed,
                "certificate": certificate,
                "native_comparison": {
                    "acceptance_arithmetic": "Upper enclosure of complex subtraction/magnitude and reference uncertainty; lower enclosure of atol + rtol times minimum exact-reference magnitude; outward ratio.",
                    **metrics,
                    **native_details,
                    "max_scaled_error_including_reference_bound": float(np.max(scaled)),
                    "failed_entry_count": int(np.count_nonzero(failed)),
                },
                "high_precision_residual_checks": precision_checks,
                "high_precision_stable": precision_stable,
                "high_precision_residual_checks_passed": residual_checks_passed,
                "encoded_system_sha256": hashlib.sha256(
                    np.ascontiguousarray(system).view(np.uint8)
                ).hexdigest(),
                "encoded_rhs_sha256": hashlib.sha256(
                    np.ascontiguousarray(rhs).view(np.uint8)
                ).hexdigest(),
                "method": "Upstream public A and T; D_i=sqrt(max_j|T_ij|); independent SciPy solve of D^-1 A D; Neumann inverse bound with arithmetic envelopes and selected 80/120-digit residual checks. Native errors plus reference uncertainty use unchanged atol/rtol.",
            }
        )
    except (ValueError, ArithmeticError) as error:
        result["failure"] = f"{type(error).__name__}: {error}"
    return result


def chain_geometry(particles):
    import numpy as np

    return (
        np.linspace(0.15, 0.25, particles),
        np.full(particles, 4 + 0.1j),
        np.column_stack([np.arange(particles) * 0.8, np.zeros((particles, 2))]),
    )


def evaluate_order(c, degree, particles, block_degree):
    import numpy as np
    import treams
    from scipy.linalg import block_diag

    import treams_rs as tr

    radii, epsilon, positions = chain_geometry(particles)
    k0 = 1.3
    native_particles = [
        tr.TMatrix.sphere(degree, k0, radius, [eps, 1])
        for radius, eps in zip(radii, epsilon, strict=True)
    ]
    native_local = tr.TMatrix(
        block_diag(*(particle.array for particle in native_particles)),
        basis=tr.SphericalBasis(
            [
                (index, *mode[1:])
                for index, particle in enumerate(native_particles)
                for mode in particle.basis
            ],
            positions,
        ),
        k0=k0,
    )
    upstream_local = treams.TMatrix.cluster(
        [
            treams.TMatrix.sphere(degree, k0, radius, [eps, 1])
            for radius, eps in zip(radii, epsilon, strict=True)
        ],
        positions,
    )
    native_modes = np.column_stack(
        (
            native_local.basis.pidx,
            native_local.basis.l,
            native_local.basis.m,
            native_local.basis.pol,
        )
    )
    upstream_modes = np.column_stack(
        (
            upstream_local.basis.pidx,
            upstream_local.basis.l,
            upstream_local.basis.m,
            upstream_local.basis.pol,
        )
    )
    if not np.array_equal(native_modes, upstream_modes):
        raise ValueError("native/upstream mode ordering differs")

    # These are the exact two full-T paths used by benchmark_cluster's cluster case.
    native, context = tr.diff.sphere_cluster(degree, k0, radii, epsilon, positions)
    del context
    upstream = upstream_local.interaction.solve()
    upstream_array = np.asarray(upstream)
    native_system, upstream_system = (
        native_local.interaction(),
        np.asarray(upstream_local.interaction()),
    )
    selector = native_local.basis.l <= block_degree
    block = np.ix_(selector, selector)
    x_axis = {"name": "lmax", "value": degree, "unit": "order"}
    compare(
        c,
        native,
        upstream_array,
        prefix="full_T",
        series="full response matrix",
        x=x_axis,
    )
    compare(
        c,
        native[block],
        upstream_array[block],
        prefix="low_order_T",
        series=f"l <= {block_degree} response block",
        x=x_axis,
    )
    compare(
        c,
        native_local.array,
        np.asarray(upstream_local),
        prefix="local_T",
        series="isolated-sphere coefficients",
        x=x_axis,
    )
    compare(
        c,
        native_system,
        upstream_system,
        prefix="system_A",
        series="interaction system assembly",
        x=x_axis,
    )

    # Numerically tiny entries are not declared analytically zero without a proof.
    tiny = abs(native) <= 1e-20
    tiny_disagreements = tiny & (abs(upstream_array) > 1e-12)
    c.add(
        "native_tiny_upstream_nonzero_entry_count",
        np.count_nonzero(tiny_disagreements),
        None,
        reference_kind="upstream",
        x=x_axis,
        native_tiny_threshold=1e-20,
        upstream_absolute_threshold=1e-12,
        interpretation="Numerical pattern only; not an analytic zero classification.",
    )

    directions = ([1, 0, 0], [0, 0, 1], [0, 0, 1])
    pols = ([0, 1, 0], [1, 0, 0], [0, 1, 0])
    waves = {
        "treams-rs": [
            tr.plane_wave(direction, pol, k0=k0)
            for direction, pol in zip(directions, pols, strict=True)
        ],
        "treams": [
            treams.plane_wave(direction, pol, k0=k0, material=1, poltype="helicity")
            for direction, pol in zip(directions, pols, strict=True)
        ],
    }
    results = {}
    systems = {"treams-rs": native_system, "treams": upstream_system}
    locals_ = {"treams-rs": native_local, "treams": upstream_local}
    matrices = {"treams-rs": native, "treams": upstream_array}
    for backend in ("treams-rs", "treams"):
        local, value, system = locals_[backend], matrices[backend], systems[backend]
        local_array = np.asarray(local)
        condition = condition_estimate(system)
        c.add(
            "reciprocal_condition_1_estimate",
            condition["system_reciprocal_condition_1_estimate"],
            None,
            backend=backend,
            id=f"{c.current['id']}/{backend}/condition",
            reference_kind="analytic",
            conditioning=condition,
            series="algebraically equivalent A = I - Tlocal C, assembled by each backend",
            x=x_axis,
        )
        for metric, error in backward_errors(
            system, value, local_array, componentwise=False
        ).items():
            c.add(
                f"full_T_{metric}",
                error,
                None,
                backend=backend,
                id=f"{c.current['id']}/{backend}/full_T_{metric}",
                reference_kind="analytic",
                conditioning=condition,
                x=x_axis,
            )
        incident = np.column_stack(
            [np.asarray(wave.expand(local.basis)) for wave in waves[backend]]
        )
        scattered = value @ incident
        rhs = local_array @ incident
        for metric, error in backward_errors(system, scattered, rhs).items():
            c.add(
                f"illuminated_{metric}",
                error,
                None,
                backend=backend,
                id=f"{c.current['id']}/{backend}/illuminated_{metric}",
                reference_kind="analytic",
                conditioning=condition,
                x=x_axis,
            )
        # Shared native A/T applied to the other response separates formulation
        # rounding from each backend's own linear-solve residual.
        if backend == "treams":
            for metric, error in backward_errors(
                native_system, scattered, native_local.array @ incident
            ).items():
                c.add(
                    f"upstream_in_native_system_{metric}",
                    error,
                    None,
                    backend=backend,
                    reference_kind="analytic",
                    conditioning={"system_assembly": "treams-rs"},
                    x=x_axis,
                )
        response = (
            tr.TMatrix(value, k0=k0, basis=local.basis)
            if backend == "treams-rs"
            else upstream
        )
        cross_sections = []
        for index, wave in enumerate(waves[backend]):
            sca, ext = map(float, response.xs(wave))
            absorption = ext - sca
            cross_sections.append([sca, ext, absorption])
            c.add(
                "passivity_violation_relative",
                max(0, -sca, -ext, -absorption) / max(abs(ext), abs(sca), 1e-30),
                2e-10,
                backend=backend,
                id=f"{c.current['id']}/{backend}/passivity_wave{index}",
                reference_kind="physical_invariant",
                series=f"wave {index}",
                x=x_axis,
                observables={
                    "scattering": sca,
                    "extinction": ext,
                    "absorption": absorption,
                },
                illumination={
                    "direction": directions[index],
                    "electric_polarization": pols[index],
                },
                conditioning=condition,
            )
        results[backend] = {
            "cross_sections": np.asarray(cross_sections),
            "low_order_block": value[block].copy(),
        }
    compare(
        c,
        results["treams-rs"]["cross_sections"],
        results["treams"]["cross_sections"],
        prefix="illuminated_cross_sections",
        series="three waves, scattering/extinction/absorption",
        x=x_axis,
    )
    if degree >= 6:
        proof = certify_cluster_reference(
            native,
            upstream_array,
            lmax=degree,
            k0=k0,
            radii=radii,
            epsilon=epsilon,
            positions=positions,
        )
        c.add(
            "certified_reference_max_scaled_error",
            proof.get("native_comparison", {}).get(
                "max_scaled_error_including_reference_bound", float("nan")
            ),
            1.0,
            reference_kind="independently_balanced_upstream_system",
            series="native output plus certified reference uncertainty",
            x=x_axis,
            certificate=proof,
        )
    return results


def source_metadata():
    from _harness import file_sha256
    from benchmark_illumination import fingerprints

    from treams_rs import _native

    folder = Path(__file__).parent
    source = {**fingerprints(), **fingerprints(upstream=True)}
    source.update(
        {
            "native_profile": _native.build_profile(),
            "script_sha256": file_sha256(__file__),
            "physics_collector_sha256": file_sha256(folder / "qualify_physics.py"),
            "residual_helper_sha256": file_sha256(folder / "qualify_upstream.py"),
            "benchmark_harness_sha256": file_sha256(folder / "benchmark_cluster.py"),
        }
    )
    return source


def qualify(particles=8, orders=(3, 4, 6, 8, 12)):
    import numpy as np
    from qualify_physics import Collector, relative

    source = source_metadata()
    c = Collector()
    block_degree = min(3, min(orders))
    results = {}
    for degree in orders:
        with c.case(
            f"chain-n{particles}-l{degree}",
            "cluster conditioning diagnostic",
            dict(
                particles=particles,
                lmax=degree,
                low_block_lmax=block_degree,
                k0=1.3,
                radius_range=[0.15, 0.25],
                epsilon=[4, 0.1],
                spacing=0.8,
                chain_axis="x",
            ),
        ):
            results[degree] = evaluate_order(c, degree, particles, block_degree)
    if results:
        reference_degree = max(results)
        for degree, result in results.items():
            if degree == reference_degree:
                continue  # A reference compared with itself is not evidence.
            for backend, values in result.items():
                with c.case(
                    f"chain-cutoff-{backend}-l{degree}",
                    "cluster cutoff convergence",
                    dict(
                        particles=particles,
                        lmax=degree,
                        reference_lmax=reference_degree,
                        low_block_lmax=block_degree,
                    ),
                ):
                    reference = results[reference_degree][backend]
                    for name in ("cross_sections", "low_order_block"):
                        c.add(
                            f"{name}_relative_l2",
                            relative(values[name], reference[name]),
                            None,
                            backend=backend,
                            reference_kind="self_convergence",
                            series=name,
                            x={"name": "lmax", "value": degree, "unit": "order"},
                            reference_note="Largest successfully evaluated cutoff of this same backend, not independent ground truth.",
                        )
                    for wave in range(3):
                        for component, name in enumerate(
                            ("scattering", "extinction", "absorption")
                        ):
                            actual = float(values["cross_sections"][wave, component])
                            expected = float(
                                reference["cross_sections"][wave, component]
                            )
                            c.add(
                                f"wave{wave}_{name}_relative",
                                abs(actual - expected) / max(abs(expected), 1e-30),
                                None,
                                backend=backend,
                                reference_kind="self_convergence",
                                series=f"wave {wave} {name}",
                                x={"name": "lmax", "value": degree, "unit": "order"},
                                observables={"actual": actual, "reference": expected},
                            )
    after = source_metadata()
    return {
        "kind": "accuracy",
        "suite": "targeted cluster conditioning investigation",
        "observations": c.observations,
        "source": source,
        "source_after": after,
        "source_unchanged": source == after,
        "environment": {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "treams": importlib.metadata.version("treams"),
        },
        "protocol": {
            "selection": "Targeted post-failure diagnostic: Linux eight-sphere benchmark agreement failures at lmax8 and12. Not a prespecified performance sample.",
            "geometry": "Exactly benchmark_cluster.py's cluster defaults: k0=1.3, radii=linspace(0.15,0.25,N), epsilon=4+0.1j, positions=(0.8*i,0,0), vacuum exterior.",
            "entrywise_gate": "Original atol=1e-12 and rtol=2e-9. max_scaled_error <= 1; every violating entry counted, no gate relaxation.",
            "equation": "A X = Tlocal, A = I - Tlocal C; X is the full coupled T matrix and C is singular spherical translation excluding self-coupling. Each backend assembles the same algebraic equation, with its own floating-point coefficients. A assembly differences and upstream illuminated residuals in native A are recorded separately.",
            "full_response_paths": "Native diff.sphere_cluster(...)[0] and upstream treams.TMatrix.cluster(...).interaction.solve(), exactly as in benchmark_cluster.",
            "normwise_backward_error": "||A X-B||_F / (||A||_F ||X||_F + ||B||_F). This may be small despite inaccurate entries when coordinates are badly scaled.",
            "componentwise_backward_error": "max |A X-B| / (|A||X|+|B|), evaluated for three physical illuminated columns; zero/zero=0, nonzero/zero=nonfinite error.",
            "condition": "LAPACK gecon estimates reciprocal condition in the induced matrix 1-norm from SciPy LU. This is the unbalanced multipole-coordinate equation, not a coordinate-independent physical condition number. No SVD.",
            "cross_sections": "Each backend's public xs(plane_wave) with Cartesian unit electric polarization and flux0.5. Lossy particles require scattering>=0 and absorption=extinction-scattering>=0.",
            "convergence": "Fixed physical problem; local lmax changes, with the same low-order input/output block selected by (particle,l,m,polarization). Largest successful same-backend cutoff is a self-convergence comparator, not proof of accuracy. Reference-self zeros are omitted.",
            "limitations": "Small backward error, passivity, and same-backend convergence are distinct necessary checks and do not certify high-order entries or establish either backend as ground truth.",
        },
        "complete": True,
        "native_reference_passed": all(
            row["status"] == "passed" and row["certificate"]["passed"]
            for row in c.observations
            if row["metric"] == "certified_reference_max_scaled_error"
        ),
        "evaluated_orders": sorted(results),
        "requested_orders": list(orders),
        "passed": source == after
        and len(results) == len(orders)
        and all(row["status"] not in ("failed", "error") for row in c.observations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--lmax", nargs="+", type=int, default=[3, 4, 6, 8, 12])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.particles < 1 or args.threads < 1 or min(args.lmax) < 1:
        parser.error("particles, threads, and lmax must be positive")
    from _harness import pinned_threads, threadpools

    os.environ.update(pinned_threads(args.threads))
    import numpy  # noqa: F401 -- Load BLAS before applying threadpool limits.
    import scipy.linalg  # noqa: F401
    from threadpoolctl import threadpool_limits

    start = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        report = qualify(args.particles, tuple(sorted(set(args.lmax))))
        report["environment"]["threadpools"] = threadpools()
    report["environment"]["requested_threads"] = args.threads
    report["elapsed_seconds"] = time.perf_counter() - start
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "treams==0.4.5", "threadpoolctl"]
# ///
"""Full real-coordinate gradients: upstream centered differences versus native VJPs.

Every timed finite-difference operation computes every selected coordinate, including
both parts of complex parameters. This is an algorithm comparison, not a claim that
upstream has reverse-mode AD. Field coefficients also get an exact upstream linear
adjoint baseline, including construction of its dense sampling matrix.

Build the optimized extension first, then use ``uv run --no-sync python``. Each
backend/phase runs in a fresh process. Validation finishes before timed processes.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from _harness import (
    cpu_affinity,
    file_sha256,
    package_sha256,
    peak_rss_mib,
    pinned_threads,
    threadpools,
)

FAMILIES = ("cluster", "sphere", "layers", "field", "cylindrical-field")
PARAMETERS = {
    "cluster": {
        "radii": ("radii",),
        "geometry": ("radii", "positions"),
        "physical": ("radii", "positions", "epsilon", "k0"),
    },
    "sphere": {
        "radii": ("radii",),
        "physical": ("radii", "epsilon", "mu", "kappa", "k0"),
    },
    "layers": {"thickness": ("thickness",), "physical": ("thickness", "epsilon", "k0")},
    "field": {
        "coefficients": ("coefficients",),
        "geometry": ("points", "positions"),
        "physical": ("coefficients", "points", "positions", "k0"),
    },
    "cylindrical-field": {
        "coefficients": ("coefficients",),
        "geometry": ("points", "positions"),
        "physical": ("coefficients", "points", "positions", "k0"),
    },
}


def objective(value, family):
    selected = value[:, 1, 0, :, 0] if family == "layers" else value
    return float(np.vdot(selected, selected).real / selected.size)


def seed(value, family):
    if family != "layers":
        return 2 * value / value.size
    result = np.zeros_like(value)
    result[:, 1, 0, :, 0] = 2 * value[:, 1, 0, :, 0] / value[:, 1, 0, :, 0].size
    return result


class Problem:
    """One deterministic physical problem, packed as independent real coordinates."""

    def __init__(self, args, backend):
        import importlib

        self.args, self.family, self.backend = args, args.family, backend
        self.solver = importlib.import_module(
            "treams_rs" if backend == "rust" else "treams"
        )
        n = args.particles
        axis = np.arange(n, dtype=float)
        side = int(np.ceil(n ** (1 / 3)))
        grid = np.column_stack(np.unravel_index(np.arange(n), (side, side, side)))
        self.data = {
            "radii": np.linspace(0.17, 0.24, n),
            "positions": grid.astype(float) * 0.9,
            "epsilon": 2.7 + 0.25 * np.cos(axis * 0.7) + 0.04j,
            "k0": np.asarray(1.3),
        }
        if self.family == "sphere":
            self.data.update(
                radii=np.linspace(0.12, 0.38, n),
                mu=1.1 + 0.03 * np.sin(axis) + 0.015j,
                kappa=0.03 + 0.01 * np.cos(axis) + 0.002j,
            )
        elif self.family == "layers":
            self.data["thickness"] = np.linspace(0.13, 0.27, n)
            self.q = np.column_stack(
                (np.linspace(0.05, 0.7, args.samples), np.full(args.samples, 0.11))
            )
            # Explicit polarization ordering matches diff.layer_stack's final axes.
            self.basis = (
                self.solver.PlaneWavePorts
                if self.backend == "rust"
                else self.solver.PlaneWaveBasisByComp
            )([(x, y, pol) for x, y in self.q for pol in (0, 1)])
        elif "field" in self.family:
            self.data["positions"] *= 0.4
            self.data["points"] = np.column_stack(
                (
                    np.linspace(-0.4, 0.7, args.samples),
                    0.3 * np.sin(np.linspace(0.2, 3.0, args.samples)),
                    np.linspace(1.7, 2.4, args.samples),
                )
            )
            count = len(self.field_basis(self.data))
            rng = np.random.default_rng(args.seed)
            self.data["coefficients"] = (
                rng.normal(size=count) + 1j * rng.normal(size=count)
            ) / np.sqrt(count)
        self.names = PARAMETERS[self.family][args.parameters]
        self.x = self.pack(self.data)
        # Relative coordinate steps: dimensions stay physical and complex imaginary
        # directions are tested independently rather than using complex-step on |E|².
        self.scales = np.maximum(np.abs(self.x), 1.0)

    def pack(self, values):
        result = []
        for name in self.names:
            value = np.asarray(values[name])
            result.append(value.real.ravel())
            if np.iscomplexobj(self.data[name]):
                result.append(value.imag.ravel())
        return np.concatenate(result)

    def unpack(self, x):
        values = dict(self.data)
        offset = 0
        for name in self.names:
            template = self.data[name]
            count = template.size
            value = x[offset : offset + count].reshape(template.shape)
            offset += count
            if np.iscomplexobj(template):
                value = value + 1j * x[offset : offset + count].reshape(template.shape)
                offset += count
            values[name] = value
        return values

    def field_basis(self, values):
        if self.family == "cylindrical-field":
            return (
                self.solver.CylindricalBasis
                if self.backend == "rust"
                else self.solver.CylindricalWaveBasis
            ).default([0.2], self.args.lmax, self.args.particles, values["positions"])
        return (
            self.solver.SphericalBasis
            if self.backend == "rust"
            else self.solver.SphericalWaveBasis
        ).default(self.args.lmax, self.args.particles, values["positions"])

    def field_matrix(self, values):
        return np.asarray(
            (self.solver.operators if self.backend == "rust" else self.solver).efield(
                values["points"],
                basis=self.field_basis(values),
                k0=float(values["k0"]),
                poltype="helicity",
                modetype="regular",
            )
        )

    def layer_inputs(self, values):
        epsilon = np.concatenate(([1 + 0j], values["epsilon"], [1 + 0j]))
        index = np.sqrt(epsilon)
        return (
            np.repeat((float(values["k0"]) * index)[:, None], 2, axis=1),
            1 / index,
            self.q,
            values["thickness"],
        )

    def forward(self, x):
        values = self.unpack(x)
        tr, lmax, k0 = self.solver, self.args.lmax, float(values["k0"])
        if self.family == "cluster":
            spheres = [
                tr.TMatrix.sphere(lmax, k0, r, [eps, 1], poltype="helicity")
                for r, eps in zip(values["radii"], values["epsilon"], strict=True)
            ]
            if self.backend == "rust":
                return np.asarray(
                    tr.Cluster(spheres, positions=values["positions"]).solve()
                )
            return np.asarray(
                tr.TMatrix.cluster(spheres, values["positions"]).interaction.solve()
            )
        if self.family == "sphere":
            materials = list(
                zip(values["epsilon"], values["mu"], values["kappa"], strict=True)
            )
            return np.asarray(
                tr.TMatrix.sphere(
                    lmax,
                    k0,
                    values["radii"],
                    [*materials, (1, 1, 0)],
                    poltype="helicity",
                )
            )
        if self.family == "layers":
            stack = (tr.SMatrix if self.backend == "rust" else tr.SMatrices).slab(
                values["thickness"],
                self.basis,
                k0,
                [1, *values["epsilon"], 1],
                poltype="helicity",
            )
            matrix = np.asarray(
                [[np.asarray(stack[i, j]) for j in range(2)] for i in range(2)]
            )
            return np.asarray(
                [
                    matrix[:, :, 2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
                    for i in range(self.args.samples)
                ]
            )
        wave = (tr.operators if self.backend == "rust" else tr).PhysicsArray(
            values["coefficients"],
            basis=self.field_basis(values),
            k0=k0,
            material=1,
            poltype="helicity",
            modetype="regular",
        )
        return np.asarray(wave.efield(values["points"]))

    def record(self, x):
        values = self.unpack(x)
        if self.backend == "treams":
            matrix = self.field_matrix(values)
            return matrix @ values["coefficients"], matrix
        diff, k0 = self.solver.diff, float(values["k0"])
        if self.family == "cluster":
            return diff.sphere_cluster(
                self.args.lmax,
                k0,
                values["radii"],
                values["epsilon"],
                values["positions"],
            )
        if self.family == "sphere":
            return diff.sphere(
                self.args.lmax,
                k0,
                values["radii"],
                np.r_[values["epsilon"], 1],
                np.r_[values["mu"], 1],
                np.r_[values["kappa"], 0],
            )
        if self.family == "layers":
            return diff.layer_stack(*self.layer_inputs(values), fixed_q=True)
        return diff.field(
            values["coefficients"],
            values["points"],
            self.field_basis(values),
            [k0, k0],
            poltype="helicity",
        )

    def reverse(self, context, cotangent, x):
        if self.backend == "treams":
            return self.pack(
                {"coefficients": np.einsum("pim,pi->m", context.conj(), cotangent)}
            )
        raw = context.pullback(cotangent)
        if self.family == "cluster":
            gradients = dict(
                zip(("k0", "radii", "epsilon", "positions"), raw, strict=True)
            )
        elif self.family == "sphere":
            gradients = dict(
                zip(("k0", "radii", "epsilon", "mu", "kappa"), raw, strict=True)
            )
            for name in ("epsilon", "mu", "kappa"):
                gradients[name] = gradients[name][:-1]
        elif self.family == "layers":
            values = self.unpack(x)
            epsilon = np.r_[1 + 0j, values["epsilon"], 1 + 0j]
            index = np.sqrt(epsilon)
            gk, gz, _, gd = raw
            geps = np.sum(gk, axis=1) * np.conj(
                float(values["k0"]) / (2 * index)
            ) + gz * np.conj(-0.5 / (epsilon * index))
            gradients = {
                "thickness": gd,
                "epsilon": geps[1:-1],
                "k0": np.asarray(np.sum(gk.conj() * index[:, None]).real),
            }
        else:
            gradients = dict(
                zip(("coefficients", "points", "positions", "ks"), raw, strict=True)
            )
            gradients["k0"] = np.asarray(np.sum(gradients["ks"]).real)
        return self.pack(gradients)


def finite_difference(problem, x, step):
    gradient = np.empty_like(x)
    for i, scale in enumerate(problem.scales):
        plus, minus = x.copy(), x.copy()
        h = step * scale
        plus[i] += h
        minus[i] -= h
        gradient[i] = (
            objective(problem.forward(plus), problem.family)
            - objective(problem.forward(minus), problem.family)
        ) / (2 * h)
    return gradient


def relative_error(actual, expected):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("benchmark values and gradients must be finite")
    return float(
        np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-14)
    )


def invariants(problem, gradient, value):
    # Check exact continuous symmetries in the selected real-coordinate space.
    result = {}
    gradients = problem.unpack(gradient)
    if "positions" in problem.names:
        ward = np.sum(gradients["positions"], axis=0)
        if "points" in problem.names:
            ward += np.sum(gradients["points"], axis=0)
        result["translation_relative_residual"] = float(
            np.linalg.norm(ward) / max(np.linalg.norm(gradient), 1e-14)
        )
    if problem.family in ("cluster", "sphere") and "k0" in problem.names:
        ward = float(
            np.vdot(gradients["radii"], problem.data["radii"]).real
            - gradients["k0"] * problem.data["k0"]
        )
        if "positions" in problem.names:
            ward += float(
                np.vdot(gradients["positions"], problem.data["positions"]).real
            )
        result["scale_relative_residual"] = abs(ward) / max(
            np.linalg.norm(gradient), 1e-14
        )
    if "coefficients" in problem.names:
        expected = 2 * objective(value, problem.family)
        actual = float(
            np.vdot(gradients["coefficients"], problem.data["coefficients"]).real
        )
        result["amplitude_quadratic_relative_residual"] = abs(actual - expected) / max(
            abs(expected), 1e-14
        )
    if any(error > 3e-8 for error in result.values()):
        raise AssertionError(f"gradient invariant failure: {result}")
    return result


def validate(args):
    native, oracle = Problem(args, "rust"), Problem(args, "treams")
    identities = {"rust": fingerprints(native), "treams": fingerprints(oracle)}
    value, context = native.record(native.x)
    gradient = native.reverse(context, seed(value, args.family), native.x)
    expected = oracle.forward(oracle.x)
    native_forward = native.forward(native.x)
    error = relative_error(value, expected)
    public_error = relative_error(native_forward, expected)
    if max(error, public_error) > 3e-9:
        raise AssertionError(
            f"forward mismatch: recorded={error}, public={public_error}"
        )
    steps = args.fd_steps
    full_sweep = native.x.size <= args.full_sweep_limit
    rng = np.random.default_rng(args.seed + 1)
    directions = rng.normal(size=(args.directions, native.x.size)) * native.scales
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    directional = []
    for step in steps:
        measured = np.asarray(
            [
                (
                    objective(oracle.forward(oracle.x + step * d), args.family)
                    - objective(oracle.forward(oracle.x - step * d), args.family)
                )
                / (2 * step)
                for d in directions
            ]
        )
        analytic = directions @ gradient
        directional.append(
            {
                "step": step,
                "numerical": measured.tolist(),
                "analytic": analytic.tolist(),
                "relative_error": relative_error(measured, analytic),
            }
        )
    sweep = []
    full_gradients = {}
    if full_sweep:
        for step in steps:
            numerical = finite_difference(oracle, oracle.x, step)
            full_gradients[step] = numerical
            sweep.append(
                {
                    "step": step,
                    "gradient_relative_error": relative_error(numerical, gradient),
                }
            )
        selected_step = min(sweep, key=lambda row: row["gradient_relative_error"])[
            "step"
        ]
    else:
        selected_step = min(directional, key=lambda row: row["relative_error"])["step"]
        full_gradients[selected_step] = finite_difference(
            oracle, oracle.x, selected_step
        )
        sweep.append(
            {
                "step": selected_step,
                "gradient_relative_error": relative_error(
                    full_gradients[selected_step], gradient
                ),
            }
        )
    gradient_error = relative_error(full_gradients[selected_step], gradient)
    # A large radius derivative must not hide an incorrect, smaller material
    # derivative. Check every physical block in addition to the aggregate norm.
    analytic_blocks = native.unpack(gradient)
    numeric_blocks = native.unpack(full_gradients[selected_step])
    block_errors = {
        name: relative_error(numeric_blocks[name], analytic_blocks[name])
        for name in native.names
    }
    if (
        max(gradient_error, *block_errors.values()) > args.gradient_rtol
        or min(row["relative_error"] for row in directional) > args.gradient_rtol
    ):
        raise AssertionError(f"finite-difference gradient mismatch: {gradient_error}")
    linear_error = None
    if args.parameters == "coefficients":
        linear_value, linear_context = oracle.record(oracle.x)
        exact = oracle.reverse(
            linear_context, seed(linear_value, args.family), oracle.x
        )
        linear_error = relative_error(exact, gradient)
        if linear_error > 3e-9:
            raise AssertionError(f"exact linear adjoint mismatch: {linear_error}")
    result = {
        "forward_relative_error": error,
        "public_forward_relative_error": public_error,
        "gradient_relative_error": gradient_error,
        "gradient_block_relative_errors": block_errors,
        "selected_fd_step": selected_step,
        "fd_step_sweep": sweep,
        "full_coordinate_step_sweep": full_sweep,
        "directional_checks": directional,
        "invariants": invariants(native, gradient, value),
        "linear_adjoint_relative_error": linear_error,
        "parameter_count": int(native.x.size),
        "parameter_blocks": list(native.names),
        "fingerprints": identities,
        "reference_objective": objective(expected, args.family),
        "passed": True,
    }
    np.savez(
        args.array_output,
        value=value,
        gradient=gradient,
        fd_gradient=full_gradients[selected_step],
        input_coordinates=native.x,
    )
    if identities != {"rust": fingerprints(native), "treams": fingerprints(oracle)}:
        raise RuntimeError("solver or benchmark changed during validation")
    return result


def fingerprints(problem):
    directory = Path(problem.solver.__file__).parent
    return {
        "package_sha256": package_sha256(directory, (".py", ".so", ".pyd", ".dylib")),
        "benchmark_sha256": file_sha256(__file__),
        "native_sha256": file_sha256(problem.solver._native.__file__)
        if problem.backend == "rust"
        else None,
    }


def measure(args):
    problem = Problem(args, args.backend)
    if args.backend == "rust" and problem.solver._native.build_profile() != "release":
        raise RuntimeError("benchmark requires just build-ext-release")
    before = fingerprints(problem)
    baseline = peak_rss_mib()
    total_times, record_times, reverse_times = [], [], []
    last = None

    def run():
        start = time.perf_counter()
        if args.phase == "finite_difference":
            value = problem.forward(problem.x)
            loss = objective(value, args.family)
            gradient = finite_difference(problem, problem.x, args.fd_step)
            return time.perf_counter() - start, 0.0, 0.0, value, gradient, loss
        if args.phase == "forward":
            value = problem.forward(problem.x)
            return time.perf_counter() - start, 0.0, 0.0, value, None, None
        value, context = problem.record(problem.x)
        record_end = time.perf_counter()
        loss, cotangent = objective(value, args.family), seed(value, args.family)
        reverse_start = time.perf_counter()
        gradient = problem.reverse(context, cotangent, problem.x)
        end = time.perf_counter()
        return (
            end - start,
            record_end - start,
            end - reverse_start,
            value,
            gradient,
            loss,
        )

    # Calibrate serial batches, including the complete recording needed by each
    # one-use residual. Never hold a batch of residuals simultaneously in memory.
    batch = 1
    while True:
        start = time.perf_counter()
        for _ in range(batch):
            run()
        if time.perf_counter() - start >= args.min_sample_seconds:
            break
        batch *= 2
    for repeat in range(args.warmup + args.repeats):
        totals = np.zeros(3)
        for _ in range(batch):
            last = run()
            totals += last[:3]
            # Drop each return before the next call, avoiding a retained previous
            # output inflating measured peak memory or allocator costs.
            del last
            last = None
        if repeat >= args.warmup:
            total_times.append(float(totals[0] / batch))
            record_times.append(float(totals[1] / batch))
            reverse_times.append(float(totals[2] / batch))
        gc.collect()
    peak = peak_rss_mib()
    # Validation arrays are saved only after the measured high-water capture.
    last = run()
    np.savez(
        args.array_output,
        value=last[3],
        **({"gradient": last[4]} if last[4] is not None else {}),
    )
    if before != fingerprints(problem):
        raise RuntimeError("solver or benchmark changed during measurement")
    methods = {
        "forward": "public_forward",
        "adjoint": "native_analytic",
        "finite_difference": "centered_finite_difference",
        "linear_adjoint": "dense_linear_adjoint",
    }
    return {
        **before,
        "backend": args.backend,
        "phase": args.phase,
        "method": methods[args.phase],
        "elapsed_seconds": total_times,
        "median_seconds": statistics.median(total_times),
        "record_seconds": record_times
        if args.phase in ("adjoint", "linear_adjoint")
        else [],
        "reverse_seconds": reverse_times
        if args.phase in ("adjoint", "linear_adjoint")
        else [],
        "record_median_seconds": statistics.median(record_times)
        if args.phase in ("adjoint", "linear_adjoint")
        else None,
        "reverse_median_seconds": statistics.median(reverse_times)
        if args.phase in ("adjoint", "linear_adjoint")
        else None,
        "calls_per_sample": batch,
        "objective_evaluations_per_gradient": 2 * problem.x.size + 1
        if args.phase == "finite_difference"
        else 1,
        "baseline_rss_mib": baseline,
        "peak_rss_mib": peak,
        "additional_peak_rss_mib": max(0.0, peak - baseline),
        "threads": args.threads,
        "native_profile": problem.solver._native.build_profile()
        if args.backend == "rust"
        else None,
        "python": platform.python_version(),
        "system": platform.system(),
        "architecture": platform.machine(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "package_version": importlib.metadata.version(
            "treams-rs" if args.backend == "rust" else "treams"
        ),
        "affinity": cpu_affinity(),
        "threadpools": threadpools(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, default="cluster")
    parser.add_argument("--parameters", default="physical")
    parser.add_argument(
        "--particles",
        type=int,
        default=2,
        help="sphere count, or number of concentric/planar layers",
    )
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="electric-field points or planar transverse channels",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--fd-steps", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5, 1e-6]
    )
    parser.add_argument("--full-sweep-limit", type=int, default=32)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--gradient-rtol", type=float, default=2e-5)
    parser.add_argument("--min-sample-seconds", type=float, default=0.02)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=["validate", "measure"])
    parser.add_argument("--backend", choices=["rust", "treams"])
    parser.add_argument(
        "--phase", choices=["forward", "adjoint", "finite_difference", "linear_adjoint"]
    )
    parser.add_argument("--array-output", type=Path)
    parser.add_argument("--fd-step", type=float, default=1e-5)
    args = parser.parse_args()
    if args.parameters not in PARAMETERS[args.family]:
        parser.error(
            f"{args.family} parameters must be one of {tuple(PARAMETERS[args.family])}"
        )
    if (
        min(
            args.particles,
            args.lmax,
            args.samples,
            args.threads,
            args.repeats,
            args.directions,
        )
        <= 0
        or args.warmup < 0
    ):
        parser.error("counts must be positive and warmup nonnegative")
    if (
        min(
            *args.fd_steps,
            args.fd_step,
            args.gradient_rtol,
            args.min_sample_seconds,
            args.timeout,
        )
        <= 0
        or args.full_sweep_limit < 0
    ):
        parser.error(
            "steps, tolerances and durations must be positive; sweep limit nonnegative"
        )
    if args.worker:
        from threadpoolctl import threadpool_limits

        if args.array_output is None:
            parser.error("worker requires --array-output")
        # NumPy 2 emits this upstream propagation warning once per call. Numerical
        # agreement is checked separately; timing diagnostic I/O would inflate
        # the reference cost. No other warnings are suppressed.
        warnings.filterwarnings(
            "ignore",
            message="'where' used without 'out'.*",
            category=UserWarning,
            module=r"treams\._operators",
        )
        with threadpool_limits(limits=args.threads):
            print(
                json.dumps(
                    validate(args) if args.worker == "validate" else measure(args)
                )
            )
        return
    env = {**os.environ, **pinned_threads(args.threads)}
    config_keys = [
        "family",
        "parameters",
        "particles",
        "lmax",
        "samples",
        "threads",
        "repeats",
        "warmup",
        "seed",
        "full_sweep_limit",
        "directions",
        "gradient_rtol",
        "min_sample_seconds",
    ]
    common = [sys.executable, str(Path(__file__).resolve())]
    for name in config_keys:
        common.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    common.extend(["--fd-steps", *map(str, args.fd_steps)])
    measurements = []
    with tempfile.TemporaryDirectory(prefix="treams-gradients-") as temporary:

        def child(worker, backend=None, phase=None, step=None):
            file = Path(temporary) / f"{worker}-{backend}-{phase}.npz"
            command = [*common, "--worker", worker, "--array-output", str(file)]
            if backend:
                command.extend(
                    ["--backend", backend, "--phase", phase, "--fd-step", str(step)]
                )
            print(
                f"{args.family}/{args.parameters}: {worker} {backend or ''} {phase or ''}",
                file=sys.stderr,
                flush=True,
            )
            completed = subprocess.run(
                command,
                env=env,
                text=True,
                capture_output=True,
                timeout=args.timeout,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"gradient {worker}/{backend}/{phase} failed:\n{completed.stderr}"
                )
            with np.load(file) as archive:
                arrays = dict(archive)
            return json.loads(completed.stdout), arrays

        validation, reference = child("validate")
        if args.array_output or args.output:
            arrays_path = args.array_output or args.output.with_suffix(".npz")
            arrays_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(arrays_path, **reference)
            validation["check_arrays"] = {
                "path": arrays_path.name,
                "sha256": file_sha256(arrays_path),
            }
        phases = [
            ("rust", "forward"),
            ("treams", "forward"),
            ("rust", "adjoint"),
            ("treams", "finite_difference"),
        ]
        if args.parameters == "coefficients":
            phases.append(("treams", "linear_adjoint"))
        for backend, phase in phases:
            row, arrays = child(
                "measure", backend, phase, validation["selected_fd_step"]
            )
            if any(
                row[key] != value
                for key, value in validation["fingerprints"][backend].items()
            ):
                raise RuntimeError("measured code differs from validated code")
            value_error = relative_error(arrays["value"], reference["value"])
            gradient_error = (
                relative_error(arrays["gradient"], reference["gradient"])
                if "gradient" in arrays
                else None
            )
            tolerance = args.gradient_rtol if phase == "finite_difference" else 3e-9
            if value_error > 3e-9 or (
                gradient_error is not None and gradient_error > tolerance
            ):
                raise AssertionError(
                    f"measured worker mismatch {backend}/{phase}: {value_error}, {gradient_error}"
                )
            row["check"] = {
                "value_relative_error": value_error,
                "gradient_relative_error": gradient_error,
                "passed": True,
            }
            measurements.append(row)
    result = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "case": {
            **{key: getattr(args, key) for key in config_keys},
            "parameter_count": validation["parameter_count"],
            "fd_steps": args.fd_steps,
        },
        "measurements": measurements,
        "validation": validation,
        "complete": True,
        "gradient_contract": "dL = Re(vdot(g, dx)); complex parameters packed as independent real and imaginary coordinates. Exterior media and discrete mode labels are held fixed.",
        "objective": "Mean squared reflected amplitude over outgoing helicities and transverse channels, with one incident helicity"
        if args.family == "layers"
        else "Mean squared magnitude of the complete T matrix or electric samples",
        "comparison": "Upstream centered finite differences compute the complete same scalar-objective gradient; native uses recorded forward and analytic reverse. This compares derivative algorithms, not two native AD engines. Exact upstream dense linear adjoint is also measured for coefficient-only fields.",
        "timing_scope": "Inputs and imports excluded. Forward uses each public API; adjoint includes packing, recording, scalar objective/cotangent, and complete selected gradient. One-use contexts are recreated per call. Reverse excludes recording and seeding; total is timed directly, not summed medians. Returned-array destruction is excluded.",
        "peak_rss_scope": "Fresh process per backend and phase; includes imports, inputs, calibration, warmup, repeats and allocator retention. Captured before output serialization; validation is a separate process. Additional RSS subtracts a high-water input baseline, not live allocation bytes.",
        "validation_scope": "Full selected-coordinate FD gradient always checked. Step sweep is full-coordinate at/below full_sweep_limit; larger cases select the step with independent random directions before a full coordinate check. No directional cost is presented as a full-gradient timing.",
        "suppressed_warnings": [
            "NumPy 2 'where used without out' UserWarning from upstream treams._operators; numerical agreement separately checked"
        ],
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

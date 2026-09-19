# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.1", "treams==0.4.5", "threadpoolctl"]
# ///
"""Record numerical residuals from the existing benchmark's reference check.

This instruments the check in a standalone process; it does not modify the
benchmark or relax its assertion. Physical and high-precision qualifications
provide independent evidence separately.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import os
import platform
from pathlib import Path
from unittest.mock import patch

import benchmark_cluster
import numpy as np
from benchmark_illumination import fingerprints


def residuals(actual, expected, *, rtol, atol):
    actual, expected = np.broadcast_arrays(np.asarray(actual), np.asarray(expected))
    finite = np.isfinite(actual) & np.isfinite(expected)
    a, b = actual[finite], expected[finite]
    if a.dtype == np.bool_ and b.dtype == np.bool_:
        # Metric arithmetic uses exact 0/1 indicators; the original assertion
        # still receives the untouched boolean arrays.
        a, b = a.astype(np.int8), b.astype(np.int8)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        differences = np.abs(a - b)
        maximum = float(np.max(differences, initial=0))
        scaled = differences / (atol + rtol * np.abs(b))
        max_scaled = float(np.max(scaled, initial=0))
        scale = max(
            float(np.max(np.abs(a), initial=0)), float(np.max(np.abs(b), initial=0))
        )
        if scale == 0:
            relative = 0.0
        else:
            reference_norm = np.linalg.norm(b / scale)
            relative = float(np.linalg.norm(a / scale - b / scale) / reference_norm)
    coordinates = np.argwhere(finite)
    worst = int(np.argmax(scaled)) if scaled.size else None
    details = {
        "actual_shape": list(actual.shape),
        "element_count": actual.size,
        "finite_count": int(np.count_nonzero(finite)),
        "nonfinite_count": int(np.count_nonzero(~finite)),
        "atol": atol,
        "rtol": rtol,
        "worst_coordinates": coordinates[worst].tolist() if worst is not None else None,
        "worst_actual": [float(a[worst].real), float(a[worst].imag)]
        if worst is not None
        else None,
        "worst_reference": [float(b[worst].real), float(b[worst].imag)]
        if worst is not None
        else None,
    }
    values = {
        "max_abs_error": maximum,
        "relative_l2_error": relative,
        "max_scaled_error": max_scaled,
    }
    return {
        key: value if np.isfinite(value) else None for key, value in values.items()
    }, details


def collect(check, parameters):
    """Keep the original assertion authoritative, including its failure state."""
    observations = []
    original = np.testing.assert_allclose

    def record(actual, expected, *args, **kwargs):
        metrics, details = residuals(
            actual, expected, rtol=kwargs["rtol"], atol=kwargs["atol"]
        )
        status = "error" if details["nonfinite_count"] else "passed"
        message = (
            "Nonfinite outputs are not qualified by finite residual summaries"
            if details["nonfinite_count"]
            else None
        )
        if not details["finite_count"]:
            metrics = dict.fromkeys(metrics)
        try:
            original(actual, expected, *args, **kwargs)
        except AssertionError as error:
            status, message = "failed", str(error)
            raise
        finally:
            for metric, value in metrics.items():
                observations.append(
                    {
                        "id": f"{parameters['workload']}-{len(observations)}",
                        "family": parameters["workload"],
                        "backend": "treams-rs",
                        "reference_kind": "upstream",
                        "metric": metric,
                        "error": value,
                        "tolerance": 1.0 if metric == "max_scaled_error" else None,
                        "status": status,
                        "message": message,
                        "parameters": parameters,
                        **details,
                    }
                )

    failure = None
    try:
        with (
            patch.object(np.testing, "assert_allclose", record),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            check()
    except (AssertionError, ValueError, RuntimeError) as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    return observations, failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    for key in (
        "BENCH_THREADS",
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(args.threads)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[: args.threads]))
    observations, failure = collect(
        lambda: benchmark_cluster.worker(
            "check", args.particles, args.lmax, 1, args.workload, args.samples
        ),
        vars(args),
    )
    from treams_rs import _native

    result = {
        "kind": "accuracy",
        "observations": observations,
        "passed": bool(observations)
        and failure is None
        and all(row["status"] == "passed" for row in observations),
        "complete": bool(observations),
        "failure": failure,
        "source": {
            **fingerprints(),
            **fingerprints(upstream=True),
            "native_profile": _native.build_profile(),
            "collector_sha256": benchmark_cluster._digest(Path(__file__)),
            "reference_harness_sha256": benchmark_cluster._digest(
                Path(benchmark_cluster.__file__)
            ),
        },
        "environment": {
            "os": platform.system(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "treams": importlib.metadata.version("treams"),
            "threads_requested": args.threads,
        },
        "protocol": {
            "reference": "treams 0.4.5 public calculation; agreement is not an independent proof of physical correctness",
            "assertion": "Original benchmark assert_allclose remains active; native and upstream outputs evaluated once",
            "relative_error": "Normwise error on finite entries with magnitude scaling; exact zero/zero is zero, undefined or overflowed values are null",
            "scaled_error": "max(abs(actual-reference)/(atol+rtol*abs(reference))); threshold 1",
            "nonfinite": "Nonfinite entries counted separately and excluded from finite residual summaries; original assertion still compares them",
            "ebcm_legacy": args.workload == "ebcm",
        },
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

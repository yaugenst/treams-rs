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
from unittest.mock import patch

import benchmark_cluster
import numpy as np
from _harness import file_sha256, pinned_threads
from benchmark_illumination import fingerprints


def _squared_norm(values):
    """Sum of squared moduli, accumulated as np.linalg.norm does."""
    if np.iscomplexobj(values):
        return values.real @ values.real + values.imag @ values.imag
    return values @ values


def residuals(actual, expected, *, rtol, atol, chunk=benchmark_cluster.CHUNK):
    """Summarize finite-entry residuals in memory independent of the output size.

    Entries where either input is nonfinite are counted and excluded. ``atol``
    must be positive, so scaled errors are never NaN. The worst entry is the
    first maximum of the scaled error in C order, as ``np.argmax`` reports it.
    """
    actual, expected = np.broadcast_arrays(np.asarray(actual), np.asarray(expected))
    # Metric arithmetic uses exact 0/1 indicators; the original assertion still
    # receives the untouched boolean arrays.
    indicators = actual.dtype == np.bool_ and expected.dtype == np.bool_

    def chunks():
        for start in range(0, actual.size, chunk):
            a = actual.flat[start : start + chunk]
            b = expected.flat[start : start + chunk]
            finite = np.isfinite(a) & np.isfinite(b)
            a, b = a[finite], b[finite]
            if indicators:
                a, b = a.astype(np.int8), b.astype(np.int8)
            yield start, finite, a, b

    finite_count, scale = 0, 0.0
    maximum, max_scaled, worst = 0.0, 0.0, None
    difference_norm = reference_norm = np.float64(0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _, finite, a, b in chunks():
            finite_count += int(np.count_nonzero(finite))
            scale = max(
                scale,
                float(np.max(np.abs(a), initial=0)),
                float(np.max(np.abs(b), initial=0)),
            )
        for start, finite, a, b in chunks():
            if not a.size:
                continue
            differences = np.abs(a - b)
            maximum = max(maximum, float(np.max(differences)))
            scaled = differences / (atol + rtol * np.abs(b))
            index = int(np.argmax(scaled))
            if worst is None or scaled[index] > max_scaled:
                max_scaled = float(scaled[index])
                flat = start + int(np.flatnonzero(finite)[index])
                worst = flat, a[index], b[index]
            if scale:
                difference_norm += _squared_norm(a / scale - b / scale)
                reference_norm += _squared_norm(b / scale)
        relative = (
            float(np.sqrt(difference_norm) / np.sqrt(reference_norm)) if scale else 0.0
        )
    details = {
        "actual_shape": list(actual.shape),
        "element_count": actual.size,
        "finite_count": finite_count,
        "nonfinite_count": actual.size - finite_count,
        "atol": atol,
        "rtol": rtol,
        "worst_coordinates": None,
        "worst_actual": None,
        "worst_reference": None,
    }
    if worst is not None:
        flat, a, b = worst
        details["worst_coordinates"] = [
            int(i) for i in np.unravel_index(flat, actual.shape)
        ]
        details["worst_actual"] = [float(a.real), float(a.imag)]
        details["worst_reference"] = [float(b.real), float(b.imag)]
    values = {
        "max_abs_error": maximum,
        "relative_l2_error": relative,
        "max_scaled_error": max_scaled,
    }
    return {
        key: value if np.isfinite(value) else None for key, value in values.items()
    }, details


def collect(check, parameters):
    """Record each whole compared output; the chunked check decides pass or fail.

    The benchmark's ``_assert_allclose`` is the single upstream comparison of a
    check. Each call adds one observation per metric for the complete output.
    """
    observations = []
    original = benchmark_cluster._assert_allclose

    def record(actual, expected):
        if np.shape(actual) != np.shape(expected):
            # Shape mismatches have no elementwise residuals; the check raises.
            original(actual, expected)
            return
        metrics, details = residuals(
            actual, expected, rtol=benchmark_cluster.RTOL, atol=benchmark_cluster.ATOL
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
            original(actual, expected)
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
            patch.object(benchmark_cluster, "_assert_allclose", record),
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
    os.environ.update(pinned_threads(args.threads), BENCH_THREADS=str(args.threads))
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
            "collector_sha256": file_sha256(__file__),
            "reference_harness_sha256": file_sha256(benchmark_cluster.__file__),
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
            "assertion": "Original chunked benchmark assertion remains active; native and upstream outputs evaluated once; one observation per metric and compared output",
            "relative_error": "Normwise error on finite entries with magnitude scaling; exact zero/zero is zero, undefined or overflowed values are null",
            "scaled_error": "max(abs(actual-reference)/(atol+rtol*abs(reference))); threshold 1",
            "nonfinite": "Nonfinite entries counted separately and excluded from finite residual summaries; original assertion still compares them",
            "ebcm_legacy": args.workload == "ebcm",
        },
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

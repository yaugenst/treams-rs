"""Independent BLAS sanity baseline for the cached physical sampling benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg.blas import zgemv
from threadpoolctl import threadpool_info, threadpool_limits

from treams_rs import diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points", type=int)
    parser.add_argument("modes", type=int)
    args = parser.parse_args()
    mode = np.arange(args.modes)
    azimuth = mode * 2.399963229728653
    z = 0.15 + 0.8 * (mode + 0.5) / args.modes
    radial = np.sqrt(1.0 - z * z)
    vectors = np.column_stack((radial * np.cos(azimuth), radial * np.sin(azimuth), z))
    point = np.arange(args.points) + 0.5
    points = np.column_stack(
        (
            6.4 * np.modf(point * 0.754877666)[0] - 3.2,
            12.8 * np.modf(point * 0.569840296)[0] - 6.4,
            2.0 * np.modf(point * 0.438579021)[0] - 1.0,
        )
    )
    start = time.perf_counter()
    operator, _ = diff.plane_field(None, points, vectors, mode % 2, poltype="helicity")
    assembly_seconds = time.perf_counter() - start
    operator = operator.reshape(3 * args.points, args.modes)
    column = np.asfortranarray(operator)
    row_transpose = np.ascontiguousarray(operator).T
    coefficients = [
        (np.sin(mode * 0.3 + i * 0.37) + 1j * np.cos(mode * 0.7 - i * 0.11))
        / args.modes
        for i in range(1, 8)
    ]
    expected = [
        diff.plane_field(c, points, vectors, mode % 2, poltype="helicity")[0].ravel()
        for c in coefficients
    ]
    results = []
    for threads in (1, 4, 8, 16, 32):
        with threadpool_limits(limits=threads, user_api="blas"):
            for layout, matrix, trans in (
                ("column-major", column, 0),
                ("row-major", row_transpose, 1),
            ):
                zgemv(1.0, matrix, coefficients[0], trans=trans)
                elapsed = []
                maximum_error = 0.0
                for c, reference in zip(coefficients, expected, strict=True):
                    start = time.perf_counter()
                    actual = zgemv(1.0, matrix, c, trans=trans)
                    elapsed.append(time.perf_counter() - start)
                    np.testing.assert_allclose(
                        actual, reference, rtol=2e-12, atol=2e-12
                    )
                    maximum_error = max(
                        maximum_error, float(np.max(np.abs(actual - reference)))
                    )
                seconds = statistics.median(elapsed)
                results.append(
                    {
                        "threads": threads,
                        "layout": layout,
                        "seconds": seconds,
                        "nominal_operator_read_gb_per_second": column.nbytes
                        / seconds
                        / 1e9,
                        "max_abs_error": maximum_error,
                    }
                )
    print(
        json.dumps(
            {
                "points": args.points,
                "modes": args.modes,
                "precision": "complex128",
                "point_geometry": "deterministic irregular three-dimensional cloud",
                "operator_bytes": column.nbytes,
                "assembly_seconds": assembly_seconds,
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "libraries": [
                    {**pool, "filepath": Path(pool["filepath"]).name}
                    for pool in threadpool_info()
                ],
                "results": results,
                "scope": "Cached physical sampling operator; separate prepacked CPU "
                "layouts; changing coefficient vectors; median of seven warm "
                "applications including output allocation, excluding assembly and "
                "packing.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.12"
# dependencies = ["mpmath>=1.3", "numpy>=2.1"]
# ///
"""Check the installed native real-degree API against 70-digit hypergeometric values.

Run: uv run --no-sync --with mpmath python scripts/qualify_legendre.py
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np

from treams_rs import _native, diff


def ferrers(degree, order, x):
    """Hypergeometric representation, independent of the native recurrences."""
    m = abs(order)
    factor = (1 - x * x) ** (mp.mpf(m) / 2) / (2**m * mp.factorial(m))
    value = factor * mp.hyp2f1(m - degree, m + degree + 1, m + 1, (1 - x) / 2)
    if order < 0:
        return value
    return (-1) ** m * mp.gamma(degree + m + 1) / mp.gamma(degree - m + 1) * value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/fractional-legendre-physical.json"),
    )
    args = parser.parse_args()
    rows = []
    with mp.workdps(70):
        for degree in (
            0.2,
            2.3,
            math.nextafter(10.0, 11.0),
            10.0 + 1e-10,
            10.2,
            32.3,
            65.2,
            127.2,
        ):
            orders = sorted(
                {
                    s * m
                    for s in (-1, 1)
                    for m in (0, 1, 3, 6, 10, 12, 25, 64, 126)
                    if m < degree
                }
            )
            for order in orders:
                for x in (-0.999999, -0.9999, -0.9, -0.7, -0.3501, 0.3, 0.999999):
                    v, z = mp.mpf(degree), mp.mpf(x)
                    value = ferrers(v, order, z)
                    derivative = (
                        -mp.sqrt(1 - z * z) * ferrers(v, order + 1, z)
                        - order * z * value
                    ) / (1 - z * z)
                    reference = (value, derivative)
                    overflow = any(not math.isfinite(float(r)) for r in reference)
                    try:
                        actual, context = diff.angular(degree, order, x)
                        gradient = context.pullback(np.asarray(1, complex))
                    except ValueError:
                        if not overflow:
                            raise
                        actual_pair, error = None, 0.0
                    else:
                        assert not overflow, (degree, order, x)
                        actual_pair = [float(actual.real), float(gradient.real)]
                        error = max(
                            float(abs(mp.mpf(a) - r) / max(abs(r), mp.mpf("5e-324")))
                            for a, r in zip(actual_pair, reference, strict=True)
                        )
                        assert error <= 1e-10, (degree, order, x, error)
                    rows.append(
                        dict(
                            degree=degree,
                            order=order,
                            argument=x,
                            actual=actual_pair,
                            reference=[str(r) for r in reference],
                            expected_overflow=overflow,
                            relative_error=error,
                        )
                    )
    with Path(_native.__file__).open("rb") as stream:
        native_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    result = dict(
        precision_digits=70,
        native_sha256=native_sha256,
        cases=len(rows),
        finite_cases=sum(not row["expected_overflow"] for row in rows),
        max_relative_error=max(row["relative_error"] for row in rows),
        results=rows,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"Passed {result['finite_cases']} finite cases and {result['cases'] - result['finite_cases']} overflow checks; maximum relative error {result['max_relative_error']:.3g}"
    )


if __name__ == "__main__":
    main()

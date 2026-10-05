# /// script
# requires-python = ">=3.12"
# dependencies = ["diff-cover==10.6.0"]
# ///
"""Enforce overall and changed-line coverage from saved coverage.py reports.

Run from the repository root:
    uv run --script scripts/check_coverage.py HEAD_XML BASE_XML --compare-branch SHA
"""

from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

MAX_DROP = Fraction(1, 10)  # Percentage points, not a relative percentage.
PATCH_MINIMUM = 90


def coverage_percent(path: Path) -> Fraction:
    """Read exact line coverage; XML's rounded line-rate is display-only."""
    counts = ET.parse(path).getroot().attrib
    covered = int(counts["lines-covered"])
    total = int(counts["lines-valid"])
    if not 0 <= covered <= total or total == 0:
        raise ValueError(f"{path}: invalid coverage counts {covered}/{total}")
    return Fraction(100 * covered, total)


def main(argv: list[str] | None = None) -> int:
    """Check both gates and return a failing status if either falls short."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("head_report", type=Path)
    parser.add_argument("base_report", type=Path)
    parser.add_argument("--compare-branch", required=True)
    args = parser.parse_args(argv)
    try:
        head = coverage_percent(args.head_report)
        base = coverage_percent(args.base_report)
    except (OSError, KeyError, ValueError, ET.ParseError) as error:
        parser.error(str(error))

    overall_passed = head >= base - MAX_DROP
    print(
        f"Overall coverage {'PASS' if overall_passed else 'FAIL'}: "
        f"head {float(head):.3f}%, base {float(base):.3f}% "
        f"({float(head - base):+.3f} percentage points; "
        f"maximum drop {float(MAX_DROP):.1f}).",
        flush=True,
    )
    patch = subprocess.run(
        [
            "diff-cover",
            str(args.head_report),
            f"--compare-branch={args.compare_branch}",
            f"--fail-under={PATCH_MINIMUM}",
            "--total-percent-float",
            "--ignore-staged",
            "--ignore-unstaged",
        ],
        check=False,
    )
    return int(not overall_passed or patch.returncode != 0)


if __name__ == "__main__":
    raise SystemExit(main())

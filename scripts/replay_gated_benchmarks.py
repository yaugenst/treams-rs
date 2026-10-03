"""Rerun the recorded reference benchmarks with their pass criteria.

Run after `just build-ext-release` on an otherwise idle host, normally through
`just bench`, or as uv run --no-sync python scripts/replay_gated_benchmarks.py.
Each verified entry of benchmarks/complete-qualification.json is rerun with its
recorded arguments, thread count and speed and peak-RSS criteria. Every result is
written under its manifest file name. A failed case does not stop later cases; the exit
status is nonzero when any selected case failed.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/complete-qualification.json"
HARNESS = "scripts/benchmark_cluster.py"
# Workload groups of the historical recipes; `performance` holds the remainder.
PREFIXES = {
    "geometry": ("geometry-",),
    "lattice": ("lattice-",),
    "api": ("operator-", "callback-", "angular-fractional"),
    "power": ("power-",),
}
GROUPS = (*PREFIXES, "performance", "all")


def workload(entry: dict) -> str:
    command = entry["command"]
    return command[command.index("--workload") + 1]


def group_of(name: str) -> str:
    for group, prefixes in PREFIXES.items():
        if name.startswith(prefixes):
            return group
    return "performance"


def select(entries: list[dict], group: str, match: str | None = None) -> list[dict]:
    return [
        entry
        for entry in entries
        if group in ("all", group_of(workload(entry)))
        and (match is None or re.search(match, workload(entry)))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, default="all")
    parser.add_argument("--match", help="Regular expression on the workload name")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "benchmarks/results/local"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    entries = select(
        json.loads(args.manifest.read_text())["verified"], args.group, args.match
    )
    if not entries:
        parser.error("no recorded cases selected")
    if args.dry_run:
        for entry in entries:
            print(shlex.join([HARNESS, *entry["command"]]))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for entry in entries:
        output = args.output_dir / Path(entry["output"]).name
        with output.open("w") as stream:
            result = subprocess.run(
                [sys.executable, HARNESS, *entry["command"]],
                cwd=ROOT,
                stdout=stream,
                check=False,
            )
        if result.returncode:
            failures.append(output.name)
            print(f"FAILED ({result.returncode}): {output}", file=sys.stderr)
    if failures:
        print(
            f"{len(failures)} of {len(entries)} gated cases failed: "
            + ", ".join(failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

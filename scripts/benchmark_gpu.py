# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Wrap existing CUDA release examples as one evidence-preserving JSON result.

Build the examples with the commands in benchmarks/gpu-comparison-plan.json first.
This script never builds, changes the Python extension, or labels a native CPU/GPU
comparison as upstream treams. The suite controller owns time and memory limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "sampling": ("treams-cuda", "benchmark_sampling"),
    "fields": ("treams-cuda-tile", "qualify_plane"),
    "field-phases": ("treams-cuda-tile", "probe_plane"),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def source_fingerprint(crate: str, example: str, binary: Path) -> dict:
    paths = [
        ROOT / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")
    ]
    for package in ("treams-core", crate):
        folder = ROOT / "crates" / package
        paths.extend([folder / "Cargo.toml", *sorted((folder / "src").rglob("*.rs"))])
    paths.extend(
        [ROOT / "crates" / crate / "examples" / f"{example}.rs", Path(__file__)]
    )
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    return {
        "commit": capture(["git", "rev-parse", "HEAD"]),
        "source_sha256": hashes,
        "source_tree_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "binary": str(binary.relative_to(ROOT)),
        "binary_sha256": digest(binary),
        "rustc": capture(["rustc", "--version"]),
        "cargo": capture(["cargo", "--version"]),
        "requested_profile": "release",
        "release_configuration": tomllib.loads((ROOT / "Cargo.toml").read_text())[
            "profile"
        ]["release"],
        "build_note": "The caller must build the current release examples before execution; hashes identify the measured executable and current source separately.",
    }


def normalize(raw: dict, workload: str) -> list[dict]:
    rows = []

    def add(backend: str, phase: str, path: str, key: str) -> None:
        samples = (
            [sample[key] for sample in raw["samples"]] if "samples" in raw else None
        )
        rows.append(
            {
                "backend": backend,
                "phase": phase,
                "path": path,
                "median_seconds": statistics.median(samples) if samples else raw[key],
                "samples_seconds": samples,
                "sample_count": len(samples) if samples else 7,
                "samples_retained": samples is not None,
            }
        )

    if workload == "sampling":
        for path, key in (
            ("native-fused", "cpu_fused_seconds"),
            ("prepared-fused", "cpu_prepared_fused_seconds"),
            ("cached-column-major", "cpu_column_major_seconds"),
            ("cached-row-major", "cpu_row_major_seconds"),
        ):
            add("treams-rs-cpu", "forward", path, key)
        for path in ("column_major", "row_major"):
            add(
                "treams-rs-cpu",
                "coefficient-adjoint",
                f"cached-{path.replace('_', '-')}",
                f"cpu_adjoint_{path}_seconds",
            )
        for phase, prefix in (
            ("forward", "gpu"),
            ("coefficient-adjoint", "gpu_adjoint"),
        ):
            for path in ("resident", "roundtrip"):
                add("treams-rs-gpu", phase, path, f"{prefix}_{path}_seconds")
        for phase in (
            "coefficient_upload",
            "field_download",
            "field_cotangent_upload",
            "coefficient_gradient_download",
        ):
            add(
                "treams-rs-gpu", "transfer", phase.replace("_", "-"), f"{phase}_seconds"
            )
    elif workload == "fields":
        add("treams-rs-cpu", "forward", "native-fused", "cpu_seconds")
        add("treams-rs-cpu", "forward", "prepared-fused", "cpu_preweighted_seconds")
        add("treams-rs-gpu", "forward", "roundtrip", "gpu_seconds")
    else:
        add(
            "treams-rs-cpu",
            "forward-diagnostic",
            "native-fused",
            "cpu_original_seconds",
        )
        add(
            "treams-rs-cpu",
            "forward-diagnostic",
            "prepared-fused",
            "cpu_preweighted_seconds",
        )
        add(
            "treams-rs-gpu",
            "forward-diagnostic",
            "resident-kernel",
            "warm_kernel_seconds",
        )
    return rows


def self_check() -> None:
    sample = {
        key: value
        for value, key in enumerate(
            (
                "cpu_fused_seconds",
                "cpu_prepared_fused_seconds",
                "cpu_column_major_seconds",
                "cpu_row_major_seconds",
                "cpu_adjoint_column_major_seconds",
                "cpu_adjoint_row_major_seconds",
                "gpu_resident_seconds",
                "gpu_roundtrip_seconds",
                "gpu_adjoint_resident_seconds",
                "gpu_adjoint_roundtrip_seconds",
                "coefficient_upload_seconds",
                "field_download_seconds",
                "field_cotangent_upload_seconds",
                "coefficient_gradient_download_seconds",
            ),
            1,
        )
    }
    rows = normalize(
        {
            "samples": [
                sample,
                {key: value * 3 for key, value in sample.items()},
                sample,
            ]
        },
        "sampling",
    )
    assert len(rows) == 14
    resident, roundtrip = rows[6:8]
    assert (resident["path"], roundtrip["path"]) == ("resident", "roundtrip")
    assert resident["median_seconds"] == 7 and resident["samples_seconds"] == [7, 21, 7]
    rows = normalize(
        {"cpu_seconds": 3, "cpu_preweighted_seconds": 2, "gpu_seconds": 1}, "fields"
    )
    assert len(rows) == 3 and all(row["samples_seconds"] is None for row in rows)
    assert all(row["backend"].startswith("treams-rs-") for row in rows)
    print("GPU wrapper normalization passed; no solver or device was executed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=EXAMPLES, default="sampling")
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--threads", type=int, choices=(4, 16), default=4)
    parser.add_argument(
        "--wave-type", choices=("lossless", "lossy"), default="lossless"
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if sys.platform != "linux":
        parser.error("This optional CUDA benchmark requires Linux")
    if min(args.points, args.modes, args.repeats) <= 0:
        parser.error("Points, modes and repeats must be positive")
    if args.workload == "sampling" and args.wave_type != "lossless":
        parser.error("The existing cached sampling example uses real wavevectors")
    if args.workload != "sampling" and args.repeats != 7:
        parser.error("The existing fused examples fix their repetition count at seven")
    crate, example = EXAMPLES[args.workload]
    binary = ROOT / "target" / "release" / "examples" / example
    source = source_fingerprint(crate, example, binary)
    device = capture(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if "RTX 4080" not in device:
        parser.error("This predeclared appendix targets the RTX 4080 only")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        parser.error(
            "CPU affinity contains fewer logical CPUs than the requested thread budget"
        )
    os.sched_setaffinity(0, affinity[: args.threads])
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    command = (
        [str(binary), str(args.points), str(args.modes), str(args.repeats)]
        if args.workload == "sampling"
        else [
            str(binary),
            str(args.modes),
            str(args.points),
            *(["3"] if args.workload == "field-phases" else []),
            args.wave_type,
        ]
    )
    process = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    if process.stderr:
        print(
            process.stderr.replace(str(ROOT), "<checkout>").replace(
                str(Path.home()), "<home>"
            ),
            file=sys.stderr,
            end="",
        )
    process.check_returncode()
    raw = json.loads(process.stdout)
    result = {
        "schema_version": 1,
        "kind": "gpu",
        "comparison": "treams-rs-native-cpu-vs-cuda",
        "precision": "complex128",
        "finished": datetime.now(UTC).isoformat(),
        "parameters": {
            key: value for key, value in vars(args).items() if key != "self_check"
        },
        "source": source,
        "environment": {
            "gpu_before": device,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_threads": args.threads,
        },
        "results": normalize(raw, args.workload),
        "raw": raw,
        "scope": "Native treams-rs CPU versus CUDA, not upstream treams. All existing native numerical checks must pass. Setup and allocation evidence remain in raw, outside warm results. Sampling retains per-application timings; fused examples retain seven-sample medians only. Memory counts are logical owned buffers, not peak VRAM. The field-phases example is a diagnostic mirror of the production FMA kernel, not the production API. GPU gradients cover fixed-operator coefficients only.",
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

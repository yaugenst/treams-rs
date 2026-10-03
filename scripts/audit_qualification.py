# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Read-only evidence audit. No solver imports, builds, numerical runs or plots.

Exit 1: integrity error; exit 2: intact evidence with unresolved case or native checks.
Recorded upstream disagreements and explicitly excluded/diagnostic data stay visible.
Run it on a finished benchmark run; it reads manifests and results and never writes to them.
"""

import argparse
import collections
import functools
import hashlib
import json
import math
import statistics
import subprocess
import zipfile
from pathlib import Path


@functools.cache
def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reject(value):
    raise ValueError("nonfinite JSON token " + value)


def read(path):
    value = json.loads(Path(path).read_text(), parse_constant=reject)

    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON number in " + str(path))
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    return value


def tree_digest(paths, base, ordered=False):
    result = hashlib.sha256()
    for path in paths if ordered else sorted(paths):
        result.update(path.relative_to(base).as_posix().encode())
        result.update(bytes.fromhex(sha(path)))
    return result.hexdigest()


def samples_ok(values, median, repeats, *, empty=False, zero=False):
    if empty and values == []:
        return median is None
    return (
        isinstance(values, list)
        and len(values) == repeats
        and all(
            isinstance(x, (int, float))
            and math.isfinite(x)
            and (x >= 0 if zero else x > 0)
            for x in values
        )
        and statistics.median(values) == median
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("host", choices=("mac", "linux"))
    parser.add_argument("phases", nargs="+")
    parser.add_argument(
        "--cohort", choices=("comparison", "corrected"), default="comparison"
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Directory containing the selected group manifests",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    errors, unresolved, reports, disagreements = [], [], [], []
    identities = collections.defaultdict(set)

    def check(ok, message):
        if not ok:
            errors.append(message)

    def quality(ok, message):
        if not ok:
            unresolved.append(message)

    package = root / "python/treams_rs"
    current = {
        "native_sha256": sha(next(package.glob("_native*.so"))),
        "python_source_sha256": tree_digest(package.glob("*.py"), package),
        "package_sha256": tree_digest(
            [
                p
                for p in package.rglob("*")
                if p.suffix in (".py", ".so", ".pyd", ".dylib")
            ],
            package,
        ),
    }
    tracked = subprocess.check_output(
        ["git", "ls-files", "crates", "python", "Cargo.toml", "Cargo.lock"],
        cwd=root,
        text=True,
    ).splitlines()
    current["numerical_source_sha256"] = tree_digest(
        [root / p for p in tracked if (root / p).is_file()], root, ordered=True
    )
    plan_path = args.plan or root / (
        "benchmarks/correction-plan.json"
        if args.cohort == "corrected"
        else "benchmarks/comparison-plan.json"
    )
    plan = read(plan_path)
    extras = {case["id"]: case for case in plan["cases"]}
    check(len(extras) == len(plan["cases"]), "plan: duplicate case IDs")
    broad = {}
    if args.cohort == "comparison" and "broad" in args.phases:
        for index, entry in enumerate(
            read(root / "benchmarks/complete-qualification.json")["verified"]
        ):
            command = []
            for flag, value in zip(
                entry["command"][::2], entry["command"][1::2], strict=True
            ):
                if flag not in (
                    "--require-speedup",
                    "--require-rss-ratio",
                    "--repeats",
                ):
                    command.extend((flag, value))
            old = read(root / entry["output"])
            repeats = (
                3 if max(row["median_seconds"] for row in old["results"]) >= 0.2 else 7
            )
            broad[f"broad-{index:03d}-{Path(entry['output']).stem}"] = {
                "command": [
                    "scripts/benchmark_cluster.py",
                    *command,
                    "--repeats",
                    str(repeats),
                ]
            }

    def source_check(source, tag, script=None):
        for name in (
            "native_sha256",
            "python_source_sha256",
            "numerical_source_sha256",
            "package_sha256",
        ):
            if source.get(name):
                check(
                    source[name] == current[name],
                    tag + ": current " + name + " differs",
                )
                identities[name].add(source[name])
        if "native_profile" in source and source.get("native_sha256"):
            check(source["native_profile"] == "release", tag + ": not release")
        auxiliary = {
            "physics_collector_sha256": "qualify_physics.py",
            "residual_helper_sha256": "qualify_upstream.py",
            "benchmark_harness_sha256": "benchmark_cluster.py",
            "ferrers_reference_script_sha256": "qualify_legendre.py",
        }
        if script:
            auxiliary.update(
                script_sha256=Path(script).name, collector_sha256=Path(script).name
            )
            auxiliary["reference_harness_sha256"] = (
                "qualify_papers.py"
                if script.endswith("qualify_paper_accuracy.py")
                else "benchmark_cluster.py"
            )
        for name, relative in auxiliary.items():
            if name in source:
                check(
                    sha(root / "scripts" / relative) == source[name],
                    tag + ": changed " + relative,
                )
        for relative, digest in source.get("fixture_sha256", {}).items():
            check(
                sha(root / "benchmarks/papers" / relative) == digest,
                tag + ": changed fixture " + relative,
            )
        if source.get("mpmath_source_sha256"):
            identities["mpmath_source_sha256"].add(source["mpmath_source_sha256"])
        for name in ("upstream_package_sha256", "treams_package_sha256"):
            if source.get(name):
                identities["upstream_package_sha256"].add(source[name])

    results_root = (
        args.results_root or root / "benchmarks/results" / args.cohort / args.host
    ).resolve()
    for phase in args.phases:
        folder = results_root / phase
        manifest_path = folder / "manifest.json"
        entries, observations = {}, collections.Counter()
        phase_errors, phase_unresolved = len(errors), len(unresolved)
        result_count = sidecar_count = 0
        try:
            manifest = read(manifest_path)
            expected = (
                broad
                if broad and phase == "broad"
                else {k: v for k, v in extras.items() if v["group"] == phase}
            )
            check(bool(expected), phase + ": no planned cases")
            check(
                {c["id"] for c in manifest["cases"]} == set(expected),
                phase + ": plan coverage differs",
            )
            check(
                len({c["id"] for c in manifest["cases"]}) == len(manifest["cases"]),
                phase + ": duplicate case IDs",
            )
            quality(bool(manifest.get("finished")), phase + ": manifest unfinished")
            source_check(manifest["source"], phase)
            check(
                sha(root / "scripts/run_benchmark_suite.py")
                == manifest["source"]["runner_sha256"],
                phase + ": controller source changed",
            )
            for script, digest in manifest["source"]["harness_sha256"].items():
                check(
                    sha(root / script) == digest,
                    phase + ": harness source changed " + script,
                )
            for recovery in manifest.get("controller_recoveries", []):
                for file_key, hash_key in (
                    ("previous_manifest", "previous_manifest_sha256"),
                    ("previous_runner_source", "previous_runner_sha256"),
                ):
                    check(
                        sha(folder / recovery[file_key]) == recovery[hash_key],
                        phase + ": recovery archive hash mismatch",
                    )
            for case in manifest["cases"]:
                tag = phase + "/" + case["id"]
                try:
                    want = expected.get(case["id"])
                    if want:
                        for key, value in want.items():
                            if key not in ("status", "result"):
                                check(
                                    case.get(key) == value,
                                    tag + ": plan metadata changed: " + key,
                                )
                    terminal = (
                        "success",
                        "error",
                        "timeout",
                        "memory_limit",
                        "invalid_output",
                        "unsupported_platform",
                    )
                    check(case["status"] in terminal, tag + ": nonterminal case state")
                    platforms = case.get(
                        "platforms", ["darwin" if args.host == "mac" else "linux"]
                    )
                    supported = (
                        "darwin" if args.host == "mac" else "linux"
                    ) in platforms
                    check(
                        (case["status"] != "unsupported_platform") == supported,
                        tag + ": unsupported platform classification wrong",
                    )
                    attempts = case.get("attempts", [])
                    successes = [a for a in attempts if a["status"] == "success"]
                    check(len(successes) <= 1, tag + ": multiple successful attempts")
                    for attempt in attempts:
                        if "stdout" in attempt:
                            side = folder / Path(attempt["stdout"]).with_suffix(".json")
                            check(
                                side.is_file(),
                                tag + ": missing durable attempt sidecar",
                            )
                            if side.is_file():
                                check(
                                    read(side) == attempt,
                                    tag + ": durable attempt differs",
                                )
                                entries[str(side.relative_to(folder))] = sha(side)
                        for key in ("stdout", "stderr"):
                            if key not in attempt:
                                continue
                            path = folder / attempt[key]
                            entries[attempt[key]] = sha(path)
                            data = path.read_bytes()
                            check(
                                b"/Users/" not in data and b"/home/" not in data,
                                tag + ": personal path in " + key,
                            )
                        if attempt["status"] in ("timeout", "memory_limit"):
                            check(
                                bool(attempt.get("limit_event")),
                                tag + ": missing durable limit event",
                            )
                    path = folder / case["result"]
                    if case["status"] != "success":
                        check(not path.exists(), tag + ": nonsuccess has result JSON")
                        quality(
                            case["status"] == "unsupported_platform",
                            tag + ": " + case["status"],
                        )
                        continue
                    check(len(successes) == 1, tag + ": missing successful attempt")
                    check(
                        successes[0].get("returncode") == 0,
                        tag + ": successful attempt has nonzero return code",
                    )
                    raw = read(path)
                    check(
                        raw == read(folder / successes[0]["stdout"]),
                        tag + ": result differs from stdout",
                    )
                    entries[case["result"]] = sha(path)
                    result_count += 1
                    command = case["command"]
                    options = dict(zip(command[1::2], command[2::2], strict=True))
                    threads = int(options["--threads"])
                    repeats = int(options.get("--repeats", 0))
                    rows = raw.get("results", raw.get("measurements", []))
                    kind = case["kind"]
                    if kind in ("cluster", "gradient", "illumination"):
                        for row in rows:
                            check(
                                row["threads"] == threads,
                                tag + ": measurement thread count differs",
                            )
                            native = row["backend"] == "rust" or (
                                kind == "illumination"
                                and not row["backend"].startswith("treams-")
                            )
                            if native:
                                source_check(row, tag)
                            elif row.get("package_sha256"):
                                identities["upstream_package_sha256"].add(
                                    row["package_sha256"]
                                )
                            elif row.get("treams_package_sha256"):
                                identities["upstream_package_sha256"].add(
                                    row["treams_package_sha256"]
                                )
                            check(
                                row["benchmark_sha256"]
                                == manifest["source"]["harness_sha256"][command[0]],
                                tag + ": measured harness differs",
                            )
                            check(
                                row["peak_rss_mib"] >= row["baseline_rss_mib"] > 0,
                                tag + ": invalid RSS",
                            )
                            if kind == "illumination":
                                for part in ("setup", "fresh", "reuse", "backward"):
                                    check(
                                        samples_ok(
                                            row[part + "_seconds"],
                                            row[part + "_median_seconds"],
                                            repeats,
                                            empty=part in ("reuse", "backward"),
                                            zero=part == "setup",
                                        ),
                                        tag + ": invalid " + part + " samples/median",
                                    )
                            else:
                                samples = row.get(
                                    "samples_seconds", row.get("elapsed_seconds")
                                )
                                check(
                                    samples_ok(samples, row["median_seconds"], repeats),
                                    tag + ": invalid samples/median",
                                )
                                parts = (
                                    (
                                        ("record_seconds", "record_median_seconds"),
                                        ("reverse_seconds", "reverse_median_seconds"),
                                    )
                                    if kind == "gradient"
                                    else (
                                        (
                                            "backward_samples_seconds",
                                            "backward_median_seconds",
                                        ),
                                    )
                                )
                                for samples_key, median_key in parts:
                                    if samples_key in row:
                                        check(
                                            samples_ok(
                                                row[samples_key],
                                                row[median_key],
                                                repeats,
                                                empty=True,
                                            ),
                                            tag + ": invalid " + samples_key,
                                        )
                                if (
                                    row.get("forward_and_backward_peak_rss_mib")
                                    is not None
                                ):
                                    check(
                                        row["forward_and_backward_peak_rss_mib"]
                                        >= row["peak_rss_mib"],
                                        tag + ": adjoint RSS below forward RSS",
                                    )
                    if kind == "cluster":
                        comparison = raw["timing_comparison"]
                        check(
                            raw["speedup"] == comparison["speedup"],
                            tag + ": inconsistent speedup",
                        )
                        if comparison["method"] == "paired_alternating_process":
                            pairs = comparison["pairs_seconds"]
                            check(
                                len(pairs) == 2 * repeats,
                                tag + ": paired count differs",
                            )
                            check(
                                statistics.median(
                                    p["treams"] / p["rust"] for p in pairs
                                )
                                == raw["speedup"],
                                tag + ": paired speedup differs",
                            )
                        else:
                            check(
                                rows[0]["median_seconds"] / rows[1]["median_seconds"]
                                == raw["speedup"],
                                tag + ": isolated speedup differs",
                            )
                        if args.cohort == "corrected":
                            validation = raw["validation"]
                            quality(
                                validation["accuracy_check"] == "passed",
                                tag + ": validation failed",
                            )
                            check(
                                (validation["atol"], validation["rtol"])
                                == (1e-12, 2e-9),
                                tag + ": accuracy tolerance changed",
                            )
                            if "independent_reference" in validation:
                                quality(
                                    validation["independent_reference"]["passed"],
                                    tag + ": independent reference failed",
                                )
                                disagreements.append(
                                    {
                                        "case": tag,
                                        "reference_kind": validation["reference_kind"],
                                        "proof": str(path.relative_to(root))
                                        + "#validation.independent_reference",
                                    }
                                )
                        # A certificate accepts a failed upstream comparison
                        # with code outside the hashed benchmark scripts; its
                        # scripts must match.
                        proof = raw.get("validation", {}).get("independent_reference")
                        source = proof.get("source", {}) if proof else {}
                        recorded = dict(source.get("certifier_sha256", {}))
                        if "reference_collector_sha256" in source:
                            collector = "qualify_cluster_conditioning.py"
                            digest = source["reference_collector_sha256"]
                            conflict = recorded.setdefault(collector, digest) != digest
                            check(
                                not conflict,
                                f"{tag}: conflicting digests for certifier {collector}",
                            )
                        for name, digest in recorded.items():
                            check(
                                Path(name).name == name
                                and sha(root / "scripts" / name) == digest,
                                tag + ": changed certifier " + name,
                            )
                    elif kind == "gradient":
                        validation = raw["validation"]
                        quality(
                            raw["complete"] and validation["passed"],
                            tag + ": gradient validation failed",
                        )
                        for row in rows:
                            for key, value in validation["fingerprints"][
                                row["backend"]
                            ].items():
                                check(
                                    row[key] == value,
                                    tag
                                    + ": validation/measurement fingerprint differs",
                                )
                            quality(
                                row["check"]["passed"],
                                tag + ": measured gradient check failed",
                            )
                        sidecar = validation["check_arrays"]
                        array = path.parent / sidecar["path"]
                        check(
                            sha(array) == sidecar["sha256"], tag + ": NPZ hash differs"
                        )
                        with zipfile.ZipFile(array) as archive:
                            check(
                                archive.testzip() is None, tag + ": corrupt NPZ member"
                            )
                            names = archive.namelist()
                            check(
                                bool(names)
                                and len(names) == len(set(names))
                                and all(n.endswith(".npy") for n in names),
                                tag + ": invalid NPZ entries",
                            )
                        entries[str(array.relative_to(folder))] = sha(array)
                        sidecar_count += 1
                    elif kind == "illumination":
                        quality(raw["complete"], tag + ": illumination incomplete")
                        for key, error in raw["comparison_relative_errors"].items():
                            quality(
                                0 <= error < (3e-9 if key.endswith("/value") else 3e-8),
                                tag + ": comparison failed " + key,
                            )
                        if raw["dimension"] <= raw["case"]["oracle_limit"]:
                            quality(
                                raw["oracle_relative_error"] is not None
                                and raw["oracle_relative_error"] < 3e-9,
                                tag + ": upstream oracle failed",
                            )
                        else:
                            check(
                                raw["oracle_relative_error"] is None
                                and bool(raw["upstream_skipped_reason"]),
                                tag + ": native-only oracle scope missing",
                            )
                        for row in rows:
                            for part in ("forward_convergence", "adjoint_convergence"):
                                for iterations, residual, rhs in row[part]:
                                    quality(
                                        0 <= iterations <= raw["case"]["max_iterations"]
                                        and 0 <= residual <= raw["case"]["rtol"] * rhs,
                                        tag + ": iterative residual failed",
                                    )
                            if "radius_direction_relative_error" in row["numerical"]:
                                quality(
                                    row["numerical"]["radius_direction_relative_error"]
                                    < 2e-6,
                                    tag + ": radius FD check failed",
                                )
                    elif kind == "accuracy":
                        source_check(raw["source"], tag, command[0])
                        if "benchmark_sha256" in raw["source"]:
                            check(
                                raw["source"]["benchmark_sha256"]
                                == sha(root / "scripts/benchmark_illumination.py"),
                                tag + ": shared fingerprint helper changed",
                            )
                        if "source_after" in raw:
                            check(
                                raw["source_after"] == raw["source"]
                                and raw["source_unchanged"],
                                tag + ": accuracy source changed during execution",
                            )
                        quality(raw["complete"], tag + ": accuracy incomplete")
                        obs = raw["observations"]
                        check(
                            bool(obs) and len({o["id"] for o in obs}) == len(obs),
                            tag + ": missing/duplicate observations",
                        )
                        for item in obs:
                            observations[
                                (
                                    item["backend"],
                                    item["reference_kind"],
                                    item["status"],
                                )
                            ] += 1
                            if item["status"] in ("failed", "error"):
                                disagreements.append(
                                    {
                                        "case": tag,
                                        "observation": item["id"],
                                        "backend": item["backend"],
                                        "reference_kind": item["reference_kind"],
                                        "status": item["status"],
                                    }
                                )
                        if "counts" in raw and "cases" in raw:
                            cases = raw["cases"]
                            check(
                                len(cases)
                                == raw["counts"]["cases"]
                                == raw["protocol"]["planned_cases"],
                                tag + ": high-precision case count differs",
                            )
                            check(
                                len({c["id"] for c in cases}) == len(cases),
                                tag + ": duplicate high-precision IDs",
                            )
                            passed = all(
                                c.get("reference_stable")
                                and (
                                    not c.get("float64_no_overflow", False)
                                    or c["backends"].get("treams-rs", {}).get("status")
                                    == "passed"
                                )
                                for c in cases
                            )
                            check(
                                passed == raw["passed"],
                                tag + ": high-precision pass summary differs",
                            )
                            for backend, name in (
                                ("treams-rs", "native_status"),
                                ("treams", "upstream_status"),
                            ):
                                check(
                                    dict(
                                        collections.Counter(
                                            c["backends"]
                                            .get(backend, {})
                                            .get("status", "error")
                                            for c in cases
                                        )
                                    )
                                    == raw["counts"][name],
                                    tag + ": high-precision backend count differs",
                                )
                        # Upstream disagreement is retained; independently certified native
                        # cluster accuracy decides the native accuracy check.
                        quality(
                            raw.get("native_reference_passed", raw["passed"]),
                            tag + ": native accuracy gate failed",
                        )
                    else:
                        check(False, tag + ": unrecognized case kind " + kind)
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    IndexError,
                    ZeroDivisionError,
                    zipfile.BadZipFile,
                ) as exc:
                    errors.append(
                        tag
                        + ": "
                        + type(exc).__name__
                        + ": "
                        + str(exc).replace(str(root), "<checkout>")
                    )
            counts = dict(collections.Counter(c["status"] for c in manifest["cases"]))
            reports.append(
                {
                    "phase": phase,
                    "finished": bool(manifest.get("finished")),
                    "case_statuses": counts,
                    "result_count": result_count,
                    "sidecar_count": sidecar_count,
                    "referenced_files": len(entries),
                    "manifest_sha256": sha(manifest_path),
                    "artifacts_inventory_sha256": hashlib.sha256(
                        json.dumps(
                            entries, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                    "observation_counts": [
                        {"backend": b, "reference_kind": r, "status": s, "count": n}
                        for (b, r, s), n in sorted(observations.items())
                    ],
                    "errors": len(errors) - phase_errors,
                    "unresolved": len(unresolved) - phase_unresolved,
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(
                phase
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc).replace(str(root), "<checkout>")
            )
    print(
        json.dumps(
            {
                "host": args.host,
                "cohort": args.cohort,
                "plan_sha256": sha(plan_path),
                "auditor_sha256": sha(Path(__file__)),
                "current_fingerprints": current,
                "results_root": str(results_root.relative_to(root)),
                "observed_fingerprints": {k: sorted(v) for k, v in identities.items()},
                "phases": reports,
                "integrity_errors": errors,
                "unresolved_cases_or_native_gates": unresolved,
                "recorded_disagreements": disagreements,
                "scope": "File/source/plan/sample/median/status/NPZ-ZIP integrity only. No solver execution or new numerical proof. No speedup gate. All case failures and upstream disagreements retained. Native-only illumination remains its declared scope. NPZ hash and CRC validated; gradient arrays not numerically recomputed.",
            },
            indent=2,
            allow_nan=False,
        )
    )
    raise SystemExit(1 if errors else 2 if unresolved else 0)


if __name__ == "__main__":
    main()

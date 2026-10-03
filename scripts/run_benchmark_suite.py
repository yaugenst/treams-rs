# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run a predeclared, sequential benchmark plan with resumable raw evidence.

Use the existing release environment: .venv/bin/python scripts/run_benchmark_suite.py
--phase broad --output benchmarks/results/comparison/mac/broad. The controller
never imports either solver; every benchmark owns its isolated workers.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from _harness import cpu_affinity, file_sha256, pinned_threads

ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def capture(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def neutral(text: str) -> str:
    return text.replace(str(ROOT), "<checkout>").replace(str(Path.home()), "<home>")


def numeric_options(command: list[str]) -> dict:
    options = dict(zip(command[::2], command[1::2], strict=True))
    return {
        key.removeprefix("--").replace("-", "_"): int(value)
        if value.isdecimal()
        else value
        for key, value in options.items()
    }


def broad_cases() -> list[dict]:
    historical = json.loads(
        (ROOT / "benchmarks/complete-qualification.json").read_text()
    )
    cases = []
    for index, entry in enumerate(historical["verified"]):
        original = entry["command"]
        command = []
        for flag, value in zip(original[::2], original[1::2], strict=True):
            if flag not in ("--require-speedup", "--require-rss-ratio", "--repeats"):
                command.extend((flag, value))
        raw = json.loads((ROOT / entry["output"]).read_text())
        maximum = max(row["median_seconds"] for row in raw["results"])
        repeats = 3 if maximum >= 0.2 else 7
        command += ["--repeats", str(repeats)]
        parameters = numeric_options(command)
        workload = parameters["workload"]
        dimension = raw["results"][0]["dimension"]
        tier = (
            "small"
            if dimension < 128
            else "medium"
            if dimension < 1024
            else "large"
            if dimension < 8192
            else "very-large"
        )
        cases.append(
            {
                "id": f"broad-{index:03d}-{Path(entry['output']).stem}",
                "group": "broad",
                "family": workload,
                "tier": tier,
                "kind": "cluster",
                "parameters": parameters,
                "series": workload,
                "x": {"name": "dimension", "value": dimension, "unit": "elements"},
                "command": ["scripts/benchmark_cluster.py", *command],
                "historical_max_median_seconds": maximum,
            }
        )
    return cases


def load_cases(args: argparse.Namespace) -> list[dict]:
    cases = broad_cases() if args.phase in ("broad", "all") else []
    if args.plan and args.phase in ("extra", "all"):
        plan = json.loads(args.plan.read_text())
        cases.extend(plan if isinstance(plan, list) else plan["cases"])
    if args.group:
        cases = [case for case in cases if case["group"] in args.group]
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
    if args.limit is not None:
        cases = cases[: args.limit]
    seen = set()
    for case in cases:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case["id"]):
            raise ValueError(f"Unsafe case id: {case['id']}")
        if case["id"] in seen:
            raise ValueError(f"Duplicate case id: {case['id']}")
        seen.add(case["id"])
        script = Path(case["command"][0])
        if script.is_absolute() or ".." in script.parts or script.suffix != ".py":
            raise ValueError(
                "Case commands must begin with a repository-relative script"
            )
        if any(value.startswith("--require-") for value in case["command"]):
            raise ValueError("Suite cases must not gate on favorable performance")
        case["status"] = (
            "pending"
            if sys.platform in case.get("platforms", [sys.platform])
            else "unsupported_platform"
        )
        case["result"] = f"results/{case['id']}.json"
    if not cases:
        raise ValueError("No cases selected")
    return cases


# One child preflight records the actual loaded release binary and package.
PREFLIGHT = """
import hashlib, json
from pathlib import Path
from treams_rs import _native
assert _native.build_profile() == 'release', 'Build with just build-ext-release first'
p=Path(_native.__file__)
h=hashlib.sha256()
for f in sorted(p.parent.glob('*.py')):
 h.update(f.name.encode()); h.update(hashlib.sha256(f.read_bytes()).digest())
print(json.dumps({'native_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'python_source_sha256':h.hexdigest(),'native_profile':_native.build_profile()}))
"""


def source_fingerprint(cases: list[dict]) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", PREFLIGHT],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(neutral(result.stderr))
    source = json.loads(result.stdout)
    source["commit"] = capture(["git", "rev-parse", "HEAD"])
    # Every benchmark script imports the shared helper, so its hash belongs to each.
    scripts = {case["command"][0] for case in cases} | {"scripts/_harness.py"}
    source["harness_sha256"] = {
        script: file_sha256(ROOT / script) for script in sorted(scripts)
    }
    source["runner_sha256"] = file_sha256(__file__)
    tree = hashlib.sha256()
    tracked = capture(
        ["git", "ls-files", "crates", "python", "Cargo.toml", "Cargo.lock"]
    ).splitlines()
    for relative in tracked:
        path = ROOT / relative
        if path.is_file():
            tree.update(relative.encode())
            tree.update(bytes.fromhex(file_sha256(path)))
    source["numerical_source_sha256"] = tree.hexdigest()
    return source


def environment() -> dict:
    result = {
        "os": platform.system(),
        "os_version": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "logical_cores": os.cpu_count(),
        "cpu_affinity": cpu_affinity(),
    }
    if sys.platform == "darwin":
        result.update(
            cpu_model=capture(["sysctl", "-n", "machdep.cpu.brand_string"]),
            physical_cores=int(capture(["sysctl", "-n", "hw.physicalcpu"])),
            ram_bytes=int(capture(["sysctl", "-n", "hw.memsize"])),
            power_mode=capture(["pmset", "-g", "custom"]),
        )
    elif sys.platform == "linux":
        cpuinfo = Path("/proc/cpuinfo").read_text()
        topology = capture(["lscpu", "-p=CORE,SOCKET"])
        result.update(
            cpu_model=next(
                line.split(":", 1)[1].strip()
                for line in cpuinfo.splitlines()
                if line.startswith("model name")
            ),
            physical_cores=len(
                {line for line in topology.splitlines() if not line.startswith("#")}
            ),
            ram_bytes=int(
                re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text())[1]
            )
            * 1024,
            power_mode={
                path.parent.name: path.read_text().strip()
                for path in Path("/sys/devices/system/cpu/cpufreq").glob(
                    "policy*/scaling_governor"
                )
            },
        )
    return result


def telemetry() -> dict:
    result = {"load_average": list(os.getloadavg())}
    if sys.platform == "linux":
        memory = Path("/proc/meminfo").read_text()
        result.update(
            {
                key: int(re.search(rf"^{label}:\s+(\d+)", memory, re.M)[1]) * 1024
                for key, label in (
                    ("available_memory_bytes", "MemAvailable"),
                    ("swap_total_bytes", "SwapTotal"),
                    ("swap_free_bytes", "SwapFree"),
                )
            }
        )
    elif sys.platform == "darwin":
        result["swap_usage"] = capture(["sysctl", "-n", "vm.swapusage"])
        result["vm_stat"] = capture(["vm_stat"])
    return result


def group_rss(process_group: int) -> int:
    rows = subprocess.check_output(["ps", "-axo", "pgid=,rss="], text=True).splitlines()
    return sum(
        int(parts[1]) * 1024
        for row in rows
        if len(parts := row.split()) == 2 and int(parts[0]) == process_group
    )


def wait_group_exit(process_group: int, permission_error=None) -> None:
    deadline = time.monotonic() + 5
    while True:
        rows = subprocess.check_output(
            ["ps", "-axo", "pgid=,stat="], text=True
        ).splitlines()
        live_states = [
            parts[1]
            for row in rows
            if len(parts := row.split()) == 2
            and int(parts[0]) == process_group
            and not parts[1].startswith("Z")
        ]
        if not live_states:
            return
        # Darwin may deny signals to ?E processes during exit teardown. Wait
        # for their disappearance; a denied ordinary live process is fatal.
        if permission_error is not None and any(
            "E" not in state for state in live_states
        ):
            permission_error.add_note(f"Process group {process_group}: {live_states}")
            raise permission_error
        if time.monotonic() >= deadline:
            message = f"Process group {process_group} did not exit: {live_states}"
            if permission_error is not None:
                permission_error.add_note(message)
                raise permission_error
            raise TimeoutError(message)
        time.sleep(0.05)


def signal_group(process_group: int, sig: int) -> None:
    try:
        os.killpg(process_group, sig)
    except ProcessLookupError:
        return
    except PermissionError as error:
        if error.errno != errno.EPERM:
            raise
        wait_group_exit(process_group, error)


def terminate_group(process: subprocess.Popen) -> None:
    signal_group(process.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1)
    # A dead parent does not imply its worker children exited.
    signal_group(process.pid, signal.SIGKILL)
    process.wait()
    wait_group_exit(process.pid)


def execute_case(
    case: dict,
    output: Path,
    timeout: float,
    memory_bytes: int,
    poll: float,
    checkpoint=None,
) -> dict:
    attempt_number = len(case.get("attempts", [])) + 1
    base = f"logs/{case['id']}.attempt-{attempt_number}"
    attempt = {
        "started": utc_now(),
        "stdout": f"{base}.stdout",
        "stderr": f"{base}.stderr",
        "status": "running",
    }
    start = time.monotonic()
    peak = 0
    before = telemetry()
    attempt["telemetry"] = {"before": before}

    def save_progress():
        attempt.update(
            elapsed_seconds=time.monotonic() - start,
            max_process_tree_rss_bytes=peak,
        )
        atomic_json(output / f"{base}.json", attempt)
        if checkpoint is not None:
            checkpoint(attempt)

    command = [
        value.replace("{output_dir}", str(output)).replace("{case_id}", case["id"])
        for value in case["command"]
    ]
    child_env = dict(os.environ)
    if "--threads" in case["command"]:
        threads = case["command"][case["command"].index("--threads") + 1]
        child_env.update(pinned_threads(threads), BENCH_THREADS=threads)
    with (
        (output / attempt["stdout"]).open("wb") as stdout,
        (output / attempt["stderr"]).open("wb") as stderr,
    ):
        process = subprocess.Popen(
            [sys.executable, *command],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env=child_env,
        )
        cleanup_started = False
        try:
            while process.poll() is None:
                peak = max(peak, group_rss(process.pid))
                if peak > memory_bytes or time.monotonic() - start > timeout:
                    attempt["status"] = (
                        "memory_limit" if peak > memory_bytes else "timeout"
                    )
                    attempt["limit_event"] = {
                        "recorded_at": utc_now(),
                        "memory_budget_bytes": memory_bytes,
                        "timeout_seconds": timeout,
                        "process_group": process.pid,
                        "elapsed_seconds": time.monotonic() - start,
                        "max_process_tree_rss_bytes": peak,
                    }
                    save_progress()  # Durable classification before any cleanup signal.
                    cleanup_started = True
                    terminate_group(process)
                    break
                time.sleep(poll)
        except BaseException as error:
            if not cleanup_started:
                try:
                    terminate_group(process)
                except Exception as cleanup_error:
                    attempt["cleanup_error"] = neutral(str(cleanup_error))
            else:
                attempt["cleanup_error"] = neutral(
                    "\n".join([str(error), *getattr(error, "__notes__", [])])
                )
            attempt["error"] = attempt.get("cleanup_error", neutral(str(error)))
            save_progress()
            raise
    for key in ("stdout", "stderr"):
        path = output / attempt[key]
        path.write_text(
            neutral(path.read_text(errors="surrogateescape")), errors="surrogateescape"
        )
    attempt.update(
        returncode=process.returncode,
        finished=utc_now(),
        elapsed_seconds=time.monotonic() - start,
        max_process_tree_rss_bytes=peak,
    )
    if attempt["status"] == "running":
        attempt["status"] = "success" if process.returncode == 0 else "error"
    if attempt["status"] == "success":
        try:
            result = json.loads((output / attempt["stdout"]).read_text())
            if not isinstance(result, dict):
                raise ValueError("Expected one complete JSON object")
            atomic_json(output / case["result"], result)
        except (ValueError, UnicodeError) as error:
            attempt["status"] = "invalid_output"
            attempt["error"] = str(error)
    if attempt["status"] != "success" and "error" not in attempt:
        attempt["error"] = neutral((output / attempt["stderr"]).read_text())
    attempt["telemetry"]["after"] = telemetry()
    save_progress()
    return attempt


def merge_resume(previous: dict, current: dict) -> dict:
    for key in ("schema_version", "suite_id", "source", "protocol"):
        if previous[key] != current[key]:
            raise ValueError(f"Resume refused: {key} changed")
    if len(previous["cases"]) != len(current["cases"]):
        raise ValueError("Resume refused: case selection changed")
    for old, new in zip(previous["cases"], current["cases"], strict=True):
        for key, value in new.items():
            if key not in ("status", "result") and old.get(key) != value:
                raise ValueError(f"Resume refused: {new['id']} changed {key}")
        if old["status"] == "running":
            old["status"] = "pending"
            old.setdefault("attempts", []).append(
                {
                    "status": "interrupted",
                    "note": "Controller stopped before result commit",
                }
            )
    return previous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("broad", "extra", "all"), default="all")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--group", action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite-id", default="treams-comparison-v1")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--memory-gib", type=float, default=8 if sys.platform == "darwin" else 16
    )
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--affinity", help="Linux CPU ids, e.g. 8,9,10,11")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.timeout, args.memory_gib, args.poll_seconds) <= 0:
        parser.error("Timeout, memory budget and polling period must be positive")
    if args.affinity:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("CPU affinity is only supported on Linux")
        os.sched_setaffinity(0, {int(cpu) for cpu in args.affinity.split(",")})
    cases = load_cases(args)
    if args.dry_run:
        print(json.dumps({"cases": cases}, indent=2))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.resume:
        parser.error("Manifest exists; use --resume to preserve completed evidence")
    manifest = {
        "schema_version": 1,
        "suite_id": args.suite_id,
        "phase": args.phase,
        "started": utc_now(),
        "source": source_fingerprint(cases),
        "environment": environment(),
        "protocol": {
            "repeats": "Broad: 3 if historical max median >= 0.2 s, otherwise 7; extras: predeclared in plan",
            "failure_policy": "Retain all results including regressions, errors, timeouts and memory limits; never retry completed cases",
            "memory_budget_bytes": int(args.memory_gib * 1024**3),
            "default_timeout_seconds": args.timeout,
            "watchdog": "ps process-group RSS sum; sum can double-count shared pages; polling is a guard, not reported benchmark peak RSS",
            "watchdog_poll_seconds": args.poll_seconds,
            "cpu_affinity": cpu_affinity(),
            "sampling": "Existing harness raw samples retained unchanged; benchmark_cluster uses predeclared alternating paired timing for all sub-ms cases",
        },
        "cases": cases,
    }
    if manifest_path.exists():
        manifest = merge_resume(json.loads(manifest_path.read_text()), manifest)
    atomic_json(manifest_path, manifest)
    for index, case in enumerate(manifest["cases"]):
        if case["status"] != "pending":
            continue
        case["status"] = "running"
        case["started"] = utc_now()
        atomic_json(manifest_path, manifest)
        print(f"[{index + 1}/{len(cases)}] {case['id']}", flush=True)
        previous_attempts = list(case.get("attempts", []))

        def checkpoint(attempt, case=case, previous_attempts=previous_attempts):
            case["attempts"] = [*previous_attempts, attempt]
            case["status"] = attempt["status"]
            case["telemetry"] = {
                **attempt["telemetry"],
                "elapsed_seconds": attempt["elapsed_seconds"],
                "max_process_tree_rss_bytes": attempt["max_process_tree_rss_bytes"],
            }
            if "error" in attempt:
                case["error"] = attempt["error"]
            atomic_json(manifest_path, manifest)

        attempt = execute_case(
            case,
            output,
            case.get("timeout_seconds", args.timeout),
            int(args.memory_gib * 1024**3),
            args.poll_seconds,
            checkpoint,
        )
        print(f"  {case['status']} ({attempt['elapsed_seconds']:.1f} s)", flush=True)
    manifest["finished"] = utc_now()
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()

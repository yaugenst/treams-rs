# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run fresh wheel-only attempts; preserve every launch and keep logs outside tools."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from runner import CODEX, VENV, launch_args

MODELS = {
    "luna": "gpt-5.6-luna",
    "sol": "gpt-5.6-sol",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
}


def run_one(spec):
    args, model, case, repeat = spec
    name = f"{model}/{case['id']}/repeat-{repeat:02}"
    log = args.output / name
    trial = Path("/tmp/treams-api-agent-trials") / args.output.name / name
    log.mkdir(parents=True, exist_ok=False)
    trial.mkdir(parents=True, exist_ok=False)
    (trial / ".tmp").mkdir()
    prompt = case["prompt"]
    (log / "prompt.txt").write_text(prompt)
    fds = ()
    if model in ("luna", "sol"):
        command, fds = launch_args(trial, MODELS[model])
        command += [
            "exec",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--ephemeral",
            "--json",
            "--color",
            "never",
            "-",
        ]
        input_data = prompt.encode()
        stdout = log / "events.jsonl"
    else:
        command = [
            "/usr/bin/python3",
            str(Path(__file__).with_name("claude_runner.py")),
            "run",
            "--trial-dir",
            str(trial),
            "--model",
            MODELS[model],
            "--prompt-file",
            str(log / "prompt.txt"),
            "--log-file",
            str(log / "events.jsonl"),
        ]
        input_data = None
        stdout = log / "launcher.stdout"
    record = {
        "model": MODELS[model],
        "effort": "medium",
        "case": case["id"],
        "repeat": repeat,
        "trial": str(trial),
        "venv": str(VENV),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_limit_seconds": args.seconds,
        "status": "running",
    }
    p = log / "run.json"
    p.write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    print(json.dumps({"event": "started", "trial": name}), flush=True)
    try:
        with stdout.open("xb") as out, (log / "stderr.log").open("xb") as err:
            process = subprocess.Popen(
                command,
                cwd=trial,
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=err,
                pass_fds=fds,
                start_new_session=True,
            )
            for fd in fds:
                os.close(fd)
            fds = ()
            try:
                process.communicate(input_data, timeout=args.seconds)
                record.update(status="finished", exit_code=process.returncode)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                record.update(status="timed_out", exit_code=process.returncode)
    finally:
        for fd in fds:
            os.close(fd)
        record["elapsed_seconds"] = time.monotonic() - started
        p.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "completed",
                "trial": name,
                "status": record["status"],
                "exit_code": record.get("exit_code"),
                "seconds": round(record["elapsed_seconds"], 1),
            }
        ),
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    args.output = args.output.resolve()
    cases = json.loads(args.suite.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "models": {k: MODELS[k] for k in args.models},
        "repeats": args.repeats,
        "workers": args.workers,
        "cases": [c["id"] for c in cases],
        "venv": str(VENV),
        "codex": str(CODEX),
        "seconds": args.seconds,
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    specs = [
        (args, model, case, repeat)
        for repeat in range(1, args.repeats + 1)
        for case in cases
        for model in args.models
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_one, specs))
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

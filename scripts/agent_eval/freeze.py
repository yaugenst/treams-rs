# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Freeze prompts and independently cross-checked references before model calls."""

import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from cases import TASKS
from reference import evaluate, jsonable, native, sphere_efficiency, upstream

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/agent-api"


def main():
    warnings.filterwarnings("ignore", message="'where' used without 'out'")
    tasks = TASKS
    for task in tasks:
        for index, config in enumerate(task["inputs"]):
            if task["id"] == "radius_inverse_design":
                config["target_efficiencies"] = [
                    sphere_efficiency(
                        [0.11, 0.105][index], config["epsilon"], w, config["lmax"]
                    )
                    for w in config["wavelengths"]
                ]
            first = evaluate(task["id"], config, upstream)
            second = evaluate(task["id"], config, native)
            for key in first:
                np.testing.assert_allclose(
                    first[key], second[key], rtol=2e-6, atol=2e-9
                )
        task["prompt"] = (
            task["prompt"].split("Example config:\n")[0]
            + "Example config:\n"
            + json.dumps(task["inputs"][0], indent=2)
        )
        task["expected"] = [jsonable(evaluate(task["id"], c)) for c in task["inputs"]]
    OUT.mkdir(exist_ok=True)
    files = {
        "reference.json": tasks,
        "tasks.json": [{k: v for k, v in t.items() if k != "expected"} for t in tasks],
        "main-prompts.json": [
            {"id": t["id"], "prompt": t["prompt"]} for t in tasks if not t["transfer"]
        ],
        "transfer-prompts.json": [
            {"id": t["id"], "prompt": t["prompt"]} for t in tasks if t["transfer"]
        ],
    }
    for name, value in files.items():
        with (OUT / name).open("x") as file:
            file.write(json.dumps(value, indent=2) + "\n")
    protocol = {
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_commit": "85986f1e157a78be35538fb34e4a9ec98be3ea93",
        "models": {
            "luna": "gpt-5.6-luna",
            "sol": "gpt-5.6-sol",
            "sonnet": "claude-sonnet-5",
            "opus": "claude-opus-5",
        },
        "effort": "medium",
        "seconds_per_attempt": 900,
        "workers": 4,
        "primary_design": "10 tasks x 4 models x 3 fresh attempts for baseline and final candidate; development screens may use one repeat, all preserved.",
        "transfer_design": "3 unseen task families x 4 models x 2 fresh attempts after candidate freezes, no fixes between transfer trials.",
        "success": "Final main and transfer cohorts each >=90% full passes; main failures at most half baseline. If baseline already >=90%, require >=40% reduction in aggregate agent wall time with no pass-rate loss. Report time and effective tokens regardless; no solver-speed claim.",
        "scoring": "Replay both supplied and hidden numeric input variants. Every required numerical result must match independent upstream/Airy references. Optimization must meet requested loss, initial/final replay and history checks; sensitivity/optimization must execute native analytic pullbacks. Parent reviews successful derivative source for finite-difference substitution and public API compliance. Missing any requested artifact prevents full pass.",
        "timeouts": "Retain all attempts, no favorable retries; artifacts present by deadline can be scored. Infrastructure failures are listed separately but included as nonpasses in primary denominator.",
        "isolation": "Installed wheel[advect] and declared dependencies only; no repository, upstream, scipy, grader, other trials, global instructions, skills, plugins, MCP or tool network. Identical dependency versions across wheels.",
        "interpretation": "Literature-inspired self-contained adaptations, not published-figure reproduction. Task families and model services are bounded; repeats are not broad ecosystem reliability proof.",
        "hashes": {
            name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            for name in files
        },
        "grader_sha256": hashlib.sha256(
            Path(__file__).with_name("grade.py").read_bytes()
        ).hexdigest(),
    }
    with (OUT / "protocol.json").open("x") as file:
        file.write(json.dumps(protocol, indent=2) + "\n")
    print("Frozen:", len(tasks), "tasks; all upstream/native controls agree.")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Grade finished attempts mechanically; print failures and cohort completion only."""

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path

import claude_runner
import grade

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("cohorts", type=Path, nargs="+")
args = parser.parse_args()
tasks = {t["id"]: t for t in json.loads(args.reference.read_text())}
pending = set(args.cohorts)
while pending:
    importlib.reload(grade)
    version = hashlib.sha256(Path(grade.__file__).read_bytes()).hexdigest()
    for cohort in list(pending):
        for file in sorted(cohort.glob("*/*/repeat-*/run.json")):
            record = json.loads(file.read_text())
            score = file.parent / "grade.json"
            if record["status"] == "running":
                continue
            if (
                score.exists()
                and json.loads(score.read_text()).get("grader_sha256") == version
            ):
                continue
            claude_runner.VENV = Path(record["venv"])
            result = grade.grade_run(file.parent, tasks)
            (file.parent / "grade.json").write_text(json.dumps(result, indent=2) + "\n")
            if result["outcome"] != "pass":
                print(
                    json.dumps(
                        {
                            "event": "nonpass",
                            "cohort": cohort.name,
                            "model": record["model"],
                            "case": record["case"],
                            "outcome": result["outcome"],
                            "private_imports": result.get("private_imports", []),
                            "reasons": [
                                c.get("reason", "")[:450] for c in result["checks"]
                            ],
                        }
                    ),
                    flush=True,
                )
        if (cohort / "summary.json").exists():
            scores = [
                json.loads(p.read_text())
                for p in sorted(cohort.glob("*/*/repeat-*/grade.json"))
            ]
            if len(scores) != len(json.loads((cohort / "summary.json").read_text())):
                continue  # A trial can finish after this iteration's directory scan.
            (cohort / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "event": "cohort_graded",
                        "cohort": cohort.name,
                        "outcomes": {
                            name: sum(r["outcome"] == name for r in scores)
                            for name in ["pass", "partial", "fail", "incomplete"]
                        },
                    }
                ),
                flush=True,
            )
            pending.remove(cohort)
    if pending:
        time.sleep(10)

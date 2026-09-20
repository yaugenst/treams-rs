# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Export observable tool actions, results and usage; omit private reasoning."""

import argparse
import hashlib
import json
from pathlib import Path


def trace(path):
    actions = []
    usage = {}
    provider_errors = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # A killed process can leave a partial final line.
        if event["type"] == "item.completed":
            item = event["item"]
            if item["type"] == "command_execution":
                actions.append(
                    {
                        k: item.get(k)
                        for k in ("command", "aggregated_output", "exit_code")
                    }
                )
        elif event["type"] == "turn.completed":
            usage = event.get("usage", {})
        elif event["type"] == "assistant":
            message = event.get("message", {})
            actions.extend(
                {
                    "id": block["id"],
                    "command": block.get("input", {}).get("command"),
                    "tool": block["name"],
                }
                for block in message.get("content", [])
                if block.get("type") == "tool_use"
            )
        elif event["type"] == "user":
            actions.extend(
                {
                    "result_for": block["tool_use_id"],
                    "output": block.get("content"),
                    "is_error": block.get("is_error", False),
                }
                for block in event.get("message", {}).get("content", [])
                if block.get("type") == "tool_result"
            )
        elif event["type"] == "result":
            usage = event.get("usage", {})
        elif event["type"] in ("error", "turn.failed"):
            provider_errors.append(
                event.get("message")
                or event.get("error", {}).get("message", "Unknown provider error")
            )
    # Use provider totals. Claude's per-message stream fragments do not carry
    # the completed output count; an interrupted run may have no token total.
    effective = None
    if usage:
        effective = (
            usage.get("input_tokens", 0)
            - usage.get("cached_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("output_tokens", 0)
        )
    return {
        "actions": actions,
        "usage": usage,
        "effective_tokens": effective,
        "provider_errors": list(dict.fromkeys(provider_errors)),
        "tool_calls": sum("command" in a for a in actions),
        "tool_errors": sum(
            bool(a.get("exit_code")) or bool(a.get("is_error")) for a in actions
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohorts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-audit", type=Path)
    args = parser.parse_args()
    audits = (
        {row["trial"]: row for row in json.loads(args.source_audit.read_text())}
        if args.source_audit
        else {}
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for cohort in args.cohorts:
        for grade in sorted(cohort.glob("*/*/repeat-*/grade.json")):
            log = grade.parent
            record = json.loads(grade.read_text())
            trial = Path(record["trial"])
            for name in ("solution.py", "REPORT.md", "results.json"):
                if (trial / name).is_symlink():
                    raise ValueError(f"Refusing artifact symlink: {trial / name}")
            key = str(Path(cohort.name) / log.relative_to(cohort))
            if key in audits:
                review = audits[key]
                assert (
                    review["solution_sha256"]
                    == hashlib.sha256((trial / "solution.py").read_bytes()).hexdigest()
                ), f"Source changed since review: {key}"
                record["source_audit"] = review
                record["automated_outcome"] = record["outcome"]
                if record["outcome"] == "pass" and review.get("outcome_override"):
                    record["outcome"] = review["outcome_override"]
            public = args.output / cohort.name / log.relative_to(cohort)
            public.mkdir(parents=True, exist_ok=True)
            for name in ("solution.py", "REPORT.md", "results.json"):
                if (trial / name).is_file():
                    (public / name).write_bytes((trial / name).read_bytes())
            evidence = trace(log / "events.jsonl")
            (public / "actions.json").write_text(json.dumps(evidence, indent=2) + "\n")
            for source in [grade, *sorted(log.glob("replay-*.json"))]:
                (public / source.name).write_bytes(source.read_bytes())
            row = {
                **record,
                **{k: v for k, v in evidence.items() if k != "actions"},
                "cohort": cohort.name,
                "evidence": str(public),
            }
            summary.append(row)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Exported", len(summary), "graded attempts; private reasoning omitted.")


if __name__ == "__main__":
    main()

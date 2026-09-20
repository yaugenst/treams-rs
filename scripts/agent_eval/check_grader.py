# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Check valid controls and reject corrupt results before accepting agent scores."""

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from grade import check
from reference import coating_reflection, derivative, sphere_efficiency
from summarize import trace


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        trial = root / "trial"
        trial.mkdir()
        outside = root / "outside.txt"
        outside.write_text("MUST_NOT_EXPORT")
        (trial / "solution.py").symlink_to(outside)
        cohort = root / "cohort"
        log = cohort / "luna/case/repeat-01"
        log.mkdir(parents=True)
        (log / "grade.json").write_text(json.dumps({"trial": str(trial)}))
        rejected = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("summarize.py")),
                str(cohort),
                "--output",
                str(root / "export"),
            ],
            capture_output=True,
            text=True,
        )
        assert (
            rejected.returncode != 0 and "Refusing artifact symlink" in rejected.stderr
        )
        assert not list((root / "export").rglob("solution.py"))
        path = Path(directory) / "events.jsonl"
        events = [
            {
                "type": "assistant",
                "message": {
                    "usage": {"output_tokens": 3},
                    "content": [
                        {"type": "thinking", "thinking": "PRIVATE_SENTINEL"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Bash",
                            "input": {"command": "python solution.py"},
                        },
                    ],
                },
            },
            {
                "type": "result",
                "usage": {
                    "input_tokens": 1,
                    "cache_creation_input_tokens": 2,
                    "cache_read_input_tokens": 500,
                    "output_tokens": 101,
                },
            },
        ]
        path.write_text("\n".join(json.dumps(event) for event in events))
        result = trace(path)
        assert result["effective_tokens"] == 104
        assert result["tool_calls"] == 1
        assert "PRIVATE_SENTINEL" not in json.dumps(result)
        path.write_text(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 75,
                        "output_tokens": 5,
                    },
                }
            )
        )
        assert trace(path)["effective_tokens"] == 30
        path.write_text(
            json.dumps({"type": "error", "message": "Selected model is at capacity."})
        )
        assert trace(path)["effective_tokens"] is None
        assert trace(path)["provider_errors"] == ["Selected model is at capacity."]
    root = Path(__file__).resolve().parents[2] / "benchmarks/agent-api"
    tasks = [
        task
        for name in ("reference.json", "polarization-reference.json")
        for task in json.loads((root / name).read_text())
    ]
    for task in tasks:
        for config, expected in zip(task["inputs"], task["expected"], strict=True):
            output = copy.deepcopy(expected)
            if task["category"] == "optimization":
                if task["id"] == "radius_inverse_design":
                    # Recover this synthetic control from the original hidden-data construction.
                    radius = 0.11 if config["initial_radius"] == 0.075 else 0.105

                    def objective(r, config=config):
                        q = [
                            sphere_efficiency(r, config["epsilon"], w, config["lmax"])
                            for w in config["wavelengths"]
                        ]
                        return sum(
                            ((a - b) / b) ** 2
                            for a, b in zip(
                                q, config["target_efficiencies"], strict=True
                            )
                        ) / len(q)

                    output = {
                        "radius": radius,
                        "loss": objective(radius),
                        "initial_loss": expected["initial_loss"],
                        "gradient": derivative(objective, radius),
                        "fitted_efficiencies": config["target_efficiencies"],
                        "loss_history": [expected["initial_loss"], objective(radius)],
                    }
                else:
                    thickness = task["id"] == "antireflection_design"
                    key = "thickness" if thickness else "wavelength"
                    x = expected["optimum"]

                    def objective(value, config=config, thickness=thickness):
                        return coating_reflection(
                            value if thickness else config["thickness"],
                            config["wavelength"] if thickness else value,
                            config["epsilon"],
                            config["substrate_epsilon"],
                        )

                    output = {
                        key: x,
                        "reflection": objective(x),
                        "initial_reflection": expected["initial_reflection"],
                        "gradient": derivative(objective, x),
                        "loss_history": [expected["initial_reflection"], objective(x)],
                    }
            check(task, config, expected, output, {"SphereContext": 2})
            if task["id"] == "coated_sphere_sensitivity_transfer":
                for name in (
                    "d_absorption_d_outer_radius",
                    "dabsorption_douter_radius",
                    "gradient_outer_radius",
                    "d_absorption_cross_section_d_outer_radius",
                ):
                    alias = {
                        "absorption_cross_section": output["value"],
                        name: output["gradient"],
                    }
                    check(task, config, expected, alias, {"SphereContext": 1})
            bad = copy.deepcopy(output)
            key = (
                next(iter(expected))
                if task["category"] != "optimization"
                else ("loss" if task["id"] == "radius_inverse_design" else "reflection")
            )
            bad[key] = 12345.0
            try:
                check(task, config, expected, bad, {"SphereContext": 2})
            except AssertionError:
                pass
            else:
                raise AssertionError(task["id"] + " accepted a corrupt result")
            if task["category"] != "forward":
                try:
                    check(task, config, expected, output, {})
                except AssertionError:
                    pass
                else:
                    raise AssertionError(
                        task["id"] + " accepted missing native pullbacks"
                    )
            if task["id"] in ("coated_sphere", "cylinder_width", "dimer_field"):
                alias = {
                    k + "_cross_section"
                    if k in ("scattering", "extinction", "absorption")
                    else k: v
                    for k, v in output.items()
                }
                check(task, config, expected, alias, {})
    print(
        f"PASS: {sum(len(task['inputs']) for task in tasks)} independent controls; corrupted outputs and missing native gradients rejected; equivalent quantity names accepted."
    )


if __name__ == "__main__":
    main()

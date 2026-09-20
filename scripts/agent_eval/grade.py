# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Replay in isolation, then compare exports outside the candidate process."""

import argparse
import ast
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np
from claude_runner import tool_command
from reference import coating_reflection, derivative, sphere_efficiency

REPLAY = """import contextlib, json, sys
import solution
counts = {}
def profile(frame, event, arg):
    if event == 'c_call' and getattr(arg, '__name__', '') == 'pullback':
        name = type(getattr(arg, '__self__', None)).__name__
        if name.endswith('Context'):
            counts[name] = counts.get(name, 0) + 1
config = json.load(sys.stdin)
sys.setprofile(profile)
with contextlib.redirect_stdout(sys.stderr):
    result = solution.run(config)
sys.setprofile(None)
print(json.dumps({'result': result, 'native_pullbacks': counts}, allow_nan=False))
"""


def close(actual, expected, *, gradient=False):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, f"shape {a.shape}, expected {b.shape}"
    assert np.isfinite(a).all(), "non-finite output"
    np.testing.assert_allclose(
        a, b, rtol=3e-4 if gradient else 2e-6, atol=1e-7 if gradient else 1e-8
    )


def check(task, config, expected, output, pullbacks):
    for key, names in {
        "transmission": ("T", "transmittance"),
        "reflection": ("R", "reflectance"),
        "absorption": ("A", "absorptance"),
    }.items():
        for name in names:
            if key not in output and name in output:
                output = {**output, key: output[name]}
    if task["id"] in ("coated_sphere", "cylinder_width", "dimer_field"):
        # These prompts name quantities, but do not mandate JSON key spelling.
        output = dict(output)
        for key in ("scattering", "extinction", "absorption"):
            for prefix in ("xs_", "xw_"):
                if key not in output and prefix + key in output:
                    output[key] = output[prefix + key]
            for suffix in (
                "_cross_section",
                "_cross_sections",
                "_cross_width",
                "_cross_widths",
                "_width",
                "_widths",
            ):
                if key not in output and key + suffix in output:
                    output[key] = output[key + suffix]
    value_names = {
        "sphere_sensitivity": ("Q", "Q_scattering", "Q_sca", "scattering_efficiency"),
        "dimer_sensitivity": ("intensity", "field_intensity"),
        "coated_sphere_sensitivity_transfer": (
            "absorption",
            "absorption_cross_section",
        ),
    }
    for name in value_names.get(task["id"], ()):
        if "value" not in output and name in output:
            output = {**output, "value": output[name]}
    if task["id"] == "coated_sphere_sensitivity_transfer":
        for name in output:
            if "gradient" not in output and name.replace("_", "").casefold() in (
                "dabsorptiondouterradius",
                "dabsorptioncrosssectiondouterradius",
                "gradientouterradius",
            ):
                output = {**output, "gradient": output[name]}
    if task["category"] != "forward":
        assert sum(pullbacks.values()) > 0, "no native analytic pullback executed"
    if task["category"] != "optimization":
        for key, value in expected.items():
            close(output[key], value, gradient=key == "gradient")
        return
    case_id = task["id"]
    if case_id == "radius_inverse_design":
        x = output["radius"]

        def objective(r):
            values = np.array(
                [
                    sphere_efficiency(r, config["epsilon"], w, config["lmax"])
                    for w in config["wavelengths"]
                ]
            )
            return np.mean(
                (
                    (values - config["target_efficiencies"])
                    / config["target_efficiencies"]
                )
                ** 2
            )

        initial, loss_key = config["initial_radius"], "loss"
        close(
            output["fitted_efficiencies"],
            [
                sphere_efficiency(x, config["epsilon"], w, config["lmax"])
                for w in config["wavelengths"]
            ],
        )
    else:
        thickness_design = case_id == "antireflection_design"
        key = "thickness" if thickness_design else "wavelength"
        x, initial, loss_key = output[key], config["initial_" + key], "reflection"

        def objective(value):
            thickness = value if thickness_design else config["thickness"]
            wavelength = config["wavelength"] if thickness_design else value
            return coating_reflection(
                thickness, wavelength, config["epsilon"], config["substrate_epsilon"]
            )

    assert config["bounds"][0] <= x <= config["bounds"][1], "outside bounds"
    loss = objective(x)
    assert loss < 1e-7, f"objective {loss} >= 1e-7"
    close(output[loss_key], loss)
    close(output["initial_" + loss_key], objective(initial))
    close(output["gradient"], derivative(objective, x), gradient=True)
    history = np.asarray(output["loss_history"], dtype=float)
    assert history.ndim == 1 and len(history) >= 2 and np.isfinite(history).all(), (
        "missing finite optimization history"
    )
    close(history[0], objective(initial))
    close(history[-1], loss)
    assert history[-1] < history[0], "no improvement from initial design"
    assert sum(pullbacks.values()) >= 2, (
        "optimization needs repeated analytic gradients"
    )


def grade_run(log, tasks):
    record = json.loads((log / "run.json").read_text())
    record["grader_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    task = tasks[record["case"]]
    trial = Path(record["trial"])
    checks = []
    artifacts = all(
        (trial / name).is_file()
        for name in ("solution.py", "results.json", "REPORT.md")
    )
    if not (trial / "solution.py").is_file():
        return {
            **record,
            "outcome": "incomplete",
            "checks": [],
            "reason": "solution.py absent",
        }
    for index, config in enumerate(task["inputs"]):
        result = {"input": index, "passed": False}
        try:
            process = subprocess.run(
                tool_command(trial, "python -c " + shlex.quote(REPLAY)),
                input=json.dumps(config),
                text=True,
                capture_output=True,
                timeout=60,
            )
            (log / f"replay-{index}.stderr").write_text(process.stderr)
            (log / f"replay-{index}.json").write_text(process.stdout)
            assert process.returncode == 0, (
                f"replay exit {process.returncode}: {process.stderr[-1200:]}"
            )
            export = json.loads(process.stdout)
            result["native_pullbacks"] = export["native_pullbacks"]
            check(
                task,
                config,
                task["expected"][index],
                export["result"],
                export["native_pullbacks"],
            )
            result["passed"] = True
        except (
            AssertionError,
            KeyError,
            ValueError,
            TypeError,
            subprocess.TimeoutExpired,
        ) as error:
            result["reason"] = str(error)[-1600:]
        checks.append(result)
    private_imports = []
    if any(c["passed"] for c in checks):
        for node in ast.walk(ast.parse((trial / "solution.py").read_text())):
            if isinstance(node, ast.ImportFrom) and (
                (node.module or "").startswith("treams_rs._")
                or (
                    node.module == "treams_rs"
                    and any(alias.name.startswith("_") for alias in node.names)
                )
            ):
                private_imports.append(ast.unparse(node))
            if isinstance(node, ast.Import) and any(
                alias.name.startswith("treams_rs._") for alias in node.names
            ):
                private_imports.append(ast.unparse(node))
        cli_check = {"input": "cli_example", "passed": False}
        try:
            process = subprocess.run(
                tool_command(trial, "python solution.py"),
                input=json.dumps(task["inputs"][0]),
                text=True,
                capture_output=True,
                timeout=60,
            )
            assert process.returncode == 0, (
                f"CLI exit {process.returncode}: {process.stderr[-1000:]}"
            )
            check(
                task,
                task["inputs"][0],
                task["expected"][0],
                json.loads(process.stdout),
                checks[0].get("native_pullbacks", {}),
            )
            cli_check["passed"] = True
        except (
            AssertionError,
            KeyError,
            ValueError,
            TypeError,
            subprocess.TimeoutExpired,
        ) as error:
            cli_check["reason"] = str(error)[-1600:]
        checks.append(cli_check)
    passed = sum(c["passed"] for c in checks)
    outcome = (
        "pass"
        if passed == len(checks) and artifacts and not private_imports
        else "partial"
        if passed
        else "fail"
    )
    return {
        **record,
        "outcome": outcome,
        "artifacts_complete": artifacts,
        "private_imports": private_imports,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    tasks = {t["id"]: t for t in json.loads(args.reference.read_text())}
    scores = []
    for path in sorted(args.cohort.glob("*/*/repeat-*/run.json")):
        result = grade_run(path.parent, tasks)
        (path.parent / "grade.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            result["model"],
            result["case"],
            result["repeat"],
            result["outcome"],
            flush=True,
        )
        scores.append(result)
    (args.cohort / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()

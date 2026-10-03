"""The read-only evidence audit and the evidence producers agree on identities."""

import contextlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import treams

import treams_rs

# The audit fingerprints the compiled extension and reads its build profile.
from treams_rs import _native

from _scripts import ROOT, load

pytestmark = pytest.mark.interface

audit = load("audit_qualification")
PACKAGE = Path(treams_rs.__file__).parent
BINARY_SUFFIXES = (".py", ".so", ".pyd", ".dylib")


def package_digest(package):
    folder = Path(package.__file__).parent
    paths = [p for p in folder.rglob("*") if p.suffix in BINARY_SUFFIXES]
    return audit.tree_digest(paths, folder)


def test_producer_fingerprints_equal_the_auditors_independent_digests(monkeypatch):
    """Drift would otherwise surface only when a finished recorded run is audited."""
    monkeypatch.setattr(_native, "build_profile", lambda: "release")
    harness = load("_harness")
    native = audit.sha(_native.__file__)
    python_source = audit.tree_digest(PACKAGE.glob("*.py"), PACKAGE)
    illumination = load("benchmark_illumination").fingerprints()
    gradients = load("benchmark_gradients").fingerprints(
        SimpleNamespace(solver=treams_rs, backend="rust")
    )
    physics = load("qualify_physics").source_metadata()
    references = load("qualify_references").fingerprints()
    preflight = io.StringIO()
    with contextlib.redirect_stdout(preflight):
        exec(load("run_benchmark_suite").PREFLIGHT, {})
    suite = json.loads(preflight.getvalue())
    assert {
        harness.file_sha256(_native.__file__),
        illumination["native_sha256"],
        gradients["native_sha256"],
        physics["native_sha256"],
        references["native_sha256"],
        suite["native_sha256"],
    } == {native}
    assert {
        harness.python_source_sha256(PACKAGE),
        illumination["python_source_sha256"],
        physics["python_source_sha256"],
        references["python_source_sha256"],
        suite["python_source_sha256"],
    } == {python_source}
    assert harness.package_sha256(PACKAGE, BINARY_SUFFIXES) == package_digest(treams_rs)
    assert gradients["package_sha256"] == package_digest(treams_rs)
    oracle = load("benchmark_gradients").fingerprints(
        SimpleNamespace(solver=treams, backend="treams")
    )
    assert {
        load("benchmark_illumination").fingerprints(upstream=True)[
            "treams_package_sha256"
        ],
        oracle["package_sha256"],
        references["upstream_package_sha256"],
    } == {package_digest(treams)}


HARNESSES = (
    "_harness.py",
    "run_benchmark_suite.py",
    "benchmark_cluster.py",
    "qualify_upstream.py",
    "benchmark_illumination.py",
    "qualify_cluster_conditioning.py",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def run_audit(monkeypatch, capsys, *arguments):
    """Audit exit code and report; digests are recomputed for every run."""
    audit.sha.cache_clear()
    argv = ["audit_qualification.py", *map(str, arguments)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as stop:
        audit.main()
    return stop.value.code, json.loads(capsys.readouterr().out)


class Evidence:
    """A minimal recorded checkout with one cluster and one accuracy case."""

    def __init__(self, root):
        self.root = root
        package = root / "python/treams_rs"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VERSION = 1\n")
        (package / "_native.abi3.so").write_bytes(b"native binary")
        (root / "scripts").mkdir()
        for name in HARNESSES:
            shutil.copy(ROOT / "scripts" / name, root / "scripts" / name)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "python"], cwd=root, check=True)
        self.folder = root / "results/checks"
        self.current = {
            "native_sha256": audit.sha(package / "_native.abi3.so"),
            "python_source_sha256": audit.tree_digest(package.glob("*.py"), package),
            "numerical_source_sha256": audit.tree_digest(
                [package / "__init__.py", package / "_native.abi3.so"],
                root,
                ordered=True,
            ),
            "native_profile": "release",
        }
        self.plan = [
            {
                "id": "cluster-case",
                "group": "checks",
                "kind": "cluster",
                "command": [
                    "scripts/benchmark_cluster.py",
                    *("--threads", "1", "--repeats", "3"),
                ],
            },
            {
                "id": "accuracy-case",
                "group": "checks",
                "kind": "accuracy",
                "command": ["scripts/qualify_upstream.py", "--threads", "1"],
            },
        ]
        rows = [
            {"backend": "treams", "median_seconds": 2, "samples_seconds": [2, 2, 2]},
            {
                "backend": "rust",
                "median_seconds": 1,
                "samples_seconds": [1, 1, 1],
                **self.current,
            },
        ]
        for row in rows:
            row.update(
                threads=1,
                benchmark_sha256=self.digest("benchmark_cluster.py"),
                peak_rss_mib=60,
                baseline_rss_mib=50,
            )
        self.results = {
            "cluster-case": {
                "results": rows,
                "speedup": 2.0,
                "timing_comparison": {"method": "isolated_process", "speedup": 2.0},
                "validation": {
                    "accuracy_check": "passed",
                    "reference_kind": "upstream",
                    "rtol": 2e-9,
                    "atol": 1e-12,
                },
            },
            "accuracy-case": {
                "kind": "accuracy",
                "passed": True,
                "complete": True,
                "observations": [
                    {
                        "id": f"cluster-{index}",
                        "backend": "treams-rs",
                        "reference_kind": "upstream",
                        "status": "passed",
                    }
                    for index in range(3)
                ],
                "source": {
                    **self.current,
                    "collector_sha256": self.digest("qualify_upstream.py"),
                    "reference_harness_sha256": self.digest("benchmark_cluster.py"),
                    "benchmark_sha256": self.digest("benchmark_illumination.py"),
                },
            },
        }
        self.statuses = dict.fromkeys(self.results, "success")

    def digest(self, name):
        return audit.sha(self.root / "scripts" / name)

    def write(self):
        cases = []
        for case in self.plan:
            identifier = case["id"]
            attempt = {
                "status": self.statuses[identifier],
                "returncode": 0,
                "stdout": f"attempts/{identifier}-1.stdout.txt",
                "stderr": f"attempts/{identifier}-1.stderr.txt",
            }
            write_json(self.folder / attempt["stdout"], self.results[identifier])
            (self.folder / attempt["stderr"]).write_text("")
            write_json(self.folder / f"attempts/{identifier}-1.stdout.json", attempt)
            result = f"results/{identifier}.json"
            write_json(self.folder / result, self.results[identifier])
            status = self.statuses[identifier]
            cases.append(
                {**case, "status": status, "result": result, "attempts": [attempt]}
            )
        write_json(self.root / "plan.json", {"cases": self.plan})
        write_json(
            self.folder / "manifest.json",
            {
                "finished": True,
                "source": {
                    **self.current,
                    "runner_sha256": self.digest("run_benchmark_suite.py"),
                    "harness_sha256": {
                        f"scripts/{name}": self.digest(name)
                        for name in ("benchmark_cluster.py", "qualify_upstream.py")
                    },
                },
                "cases": cases,
            },
        )
        return self

    def audit(self, monkeypatch, capsys):
        return run_audit(
            monkeypatch,
            capsys,
            self.root,
            "linux",
            "checks",
            "--cohort",
            "corrected",
            "--plan",
            self.root / "plan.json",
            "--results-root",
            self.root / "results",
        )


@pytest.fixture
def evidence(tmp_path):
    return Evidence(tmp_path)


def test_intact_evidence_passes_the_audit(evidence, monkeypatch, capsys):
    code, report = evidence.write().audit(monkeypatch, capsys)
    assert report["integrity_errors"] == []
    assert report["unresolved_cases_or_native_gates"] == []
    assert code == 0
    (phase,) = report["phases"]
    assert phase["result_count"] == 2 and phase["case_statuses"] == {"success": 2}
    assert (
        report["current_fingerprints"]["native_sha256"]
        == (evidence.current["native_sha256"])
    )


@pytest.mark.parametrize(
    "tamper,message",
    [
        (
            lambda e: (e.root / "scripts/benchmark_cluster.py").write_text("# edit\n"),
            "harness source changed scripts/benchmark_cluster.py",
        ),
        (
            lambda e: (e.root / "scripts/qualify_upstream.py").write_text("# edit\n"),
            "changed qualify_upstream.py",
        ),
        (
            lambda e: (e.folder / "results/cluster-case.json").write_text(
                '{"speedup": NaN}'
            ),
            "nonfinite JSON token NaN",
        ),
        (
            lambda e: (e.folder / "attempts/accuracy-case-1.stdout.json").unlink(),
            "missing durable attempt sidecar",
        ),
        (
            lambda e: (e.root / "python/treams_rs/__init__.py").write_text("V = 2\n"),
            "current python_source_sha256 differs",
        ),
    ],
)
def test_changed_sources_and_damaged_files_are_integrity_errors(
    evidence, monkeypatch, capsys, tamper, message
):
    tamper(evidence.write())
    code, report = evidence.audit(monkeypatch, capsys)
    assert code == 1
    assert any(message in error for error in report["integrity_errors"])


def test_failed_case_must_not_leave_a_result(evidence, monkeypatch, capsys):
    evidence.statuses["cluster-case"] = "error"
    code, report = evidence.write().audit(monkeypatch, capsys)
    assert code == 1
    assert report["integrity_errors"] == [
        "checks/cluster-case: nonsuccess has result JSON"
    ]
    assert "checks/cluster-case: error" in report["unresolved_cases_or_native_gates"]


def test_duplicate_observations_are_integrity_errors(evidence, monkeypatch, capsys):
    observations = evidence.results["accuracy-case"]["observations"]
    observations[1]["id"] = observations[0]["id"]
    code, report = evidence.write().audit(monkeypatch, capsys)
    assert code == 1
    assert report["integrity_errors"] == [
        "checks/accuracy-case: missing/duplicate observations"
    ]


def test_failed_native_accuracy_is_unresolved_and_retained(
    evidence, monkeypatch, capsys
):
    result = evidence.results["accuracy-case"]
    result["passed"] = False
    result["observations"][2]["status"] = "failed"
    code, report = evidence.write().audit(monkeypatch, capsys)
    assert code == 2
    assert report["integrity_errors"] == []
    assert report["unresolved_cases_or_native_gates"] == [
        "checks/accuracy-case: native accuracy gate failed"
    ]
    (disagreement,) = report["recorded_disagreements"]
    assert disagreement["observation"] == "cluster-2"
    assert disagreement["status"] == "failed"


@pytest.mark.parametrize("change", [None, "edited", "conflicting"])
def test_disagreement_certificates_bind_their_scripts(
    evidence, monkeypatch, capsys, change
):
    """Accepting a failed upstream comparison needs the recorded certifier code."""
    certifiers = load("benchmark_cluster").CERTIFIERS["cluster"]
    collector = certifiers[0]
    assert collector == "qualify_cluster_conditioning.py"
    proof = {
        "passed": True,
        "source": {
            # The collector records itself; the harness records every certifier.
            "reference_collector_sha256": "0" * 64
            if change == "conflicting"
            else evidence.digest(collector),
            "certifier_sha256": {name: evidence.digest(name) for name in certifiers},
        },
    }
    validation = evidence.results["cluster-case"]["validation"]
    validation["reference_kind"] = "upstream_with_independent_disagreement_checks"
    validation["independent_reference"] = proof
    evidence.write()
    if change == "edited":
        (evidence.root / "scripts" / collector).write_text("# edited\n")
    code, report = evidence.audit(monkeypatch, capsys)
    assert code == (0 if change is None else 1)
    errors = {
        None: [],
        "edited": [f"changed certifier {collector}"],
        "conflicting": [f"conflicting digests for certifier {collector}"],
    }[change]
    assert report["integrity_errors"] == [f"checks/cluster-case: {e}" for e in errors]
    assert report["recorded_disagreements"][0]["reference_kind"] == (
        "upstream_with_independent_disagreement_checks"
    )


def test_broad_plan_reconstruction_matches_the_benchmark_runner(
    tmp_path, monkeypatch, capsys
):
    """The auditor and run_benchmark_suite derive the same 527 broad commands."""
    evidence = Evidence(tmp_path)
    (tmp_path / "benchmarks/results").mkdir(parents=True)
    for relative in (
        "benchmarks/complete-qualification.json",
        "benchmarks/results/final",
    ):
        (tmp_path / relative).symlink_to(ROOT / relative)
    cases = [
        {**case, "status": "error", "result": f"results/{case['id']}.json"}
        for case in load("run_benchmark_suite").broad_cases()
    ]
    write_json(
        tmp_path / "results/broad/manifest.json",
        {
            "finished": True,
            "source": {
                "runner_sha256": evidence.digest("run_benchmark_suite.py"),
                "harness_sha256": {
                    "scripts/benchmark_cluster.py": evidence.digest(
                        "benchmark_cluster.py"
                    )
                },
            },
            "cases": cases,
        },
    )
    write_json(tmp_path / "plan.json", {"cases": []})
    code, report = run_audit(
        monkeypatch,
        capsys,
        *(tmp_path, "linux", "broad", "--plan", tmp_path / "plan.json"),
        *("--results-root", tmp_path / "results"),
    )
    assert report["integrity_errors"] == []
    assert len(report["unresolved_cases_or_native_gates"]) == len(cases) == 527
    assert code == 2

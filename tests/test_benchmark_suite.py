"""The suite preserves failures and stops complete worker groups at its limits."""

import argparse
import copy
import errno
import importlib.util
import json
import signal
from pathlib import Path
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_suite",
    Path(__file__).resolve().parents[1] / "scripts/run_benchmark_suite.py",
)
assert SPEC is not None and SPEC.loader is not None
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


def test_broad_plan_keeps_every_case_without_performance_gates():
    cases = suite.broad_cases()
    assert len(cases) == 527
    assert len({case["id"] for case in cases}) == 527
    for case in cases:
        assert not any(arg.startswith("--require-") for arg in case["command"])
        expected = 3 if case["historical_max_median_seconds"] >= 0.2 else 7
        assert case["parameters"]["repeats"] == expected


@pytest.mark.parametrize(
    "code,expected,timeout,memory",
    [
        ("print('{\"samples_seconds\": [0.1, 0.2]}')", "success", 3, 2**30),
        (
            'print("partial {"); raise RuntimeError("deliberate failure")',
            "error",
            3,
            2**30,
        ),
        ("print('{\"value\": 1} trailing')", "invalid_output", 3, 2**30),
        ("import time; time.sleep(20)", "timeout", 0.1, 2**30),
        ("import time; time.sleep(20)", "memory_limit", 3, 1),
    ],
)
def test_case_evidence_and_resource_limits(
    tmp_path, monkeypatch, code, expected, timeout, memory
):
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "telemetry", lambda: {"load_average": [0, 0, 0]})
    (tmp_path / "worker.py").write_text(code)
    (tmp_path / "logs").mkdir()
    (tmp_path / "results").mkdir()
    case = {"id": "sample", "command": ["worker.py"], "result": "results/sample.json"}
    result = suite.execute_case(case, tmp_path, timeout, memory, 0.01)
    assert result["status"] == expected
    assert (tmp_path / result["stdout"]).exists()
    assert (tmp_path / result["stderr"]).exists()
    assert str(tmp_path) not in (tmp_path / result["stdout"]).read_text()
    assert str(tmp_path) not in (tmp_path / result["stderr"]).read_text()
    assert (tmp_path / case["result"]).exists() == (expected == "success")
    if expected == "success":
        assert json.loads((tmp_path / case["result"]).read_text())[
            "samples_seconds"
        ] == [0.1, 0.2]
    if expected == "error":
        assert "deliberate failure" in result["error"]
    if expected in ("timeout", "memory_limit"):
        assert result["returncode"] < 0


def test_timeout_kills_grandchild(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "telemetry", lambda: {})
    (tmp_path / "logs").mkdir()
    (tmp_path / "results").mkdir()
    # The grandchild ignores TERM; the controller must kill the process group,
    # including children that survive their immediate parent's termination.
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)"
    (tmp_path / "worker.py").write_text(
        "import subprocess,sys,time\n"
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}])\n"
        "open('child.pid','w').write(str(p.pid))\ntime.sleep(20)\n"
    )
    case = {"id": "tree", "command": ["worker.py"], "result": "results/tree.json"}
    result = suite.execute_case(case, tmp_path, 0.3, 2**30, 0.01)
    assert result["status"] == "timeout"
    pid = int((tmp_path / "child.pid").read_text())
    import subprocess

    state = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="], text=True, capture_output=True
    )
    assert not state.stdout.strip() or state.stdout.strip().startswith("Z")


def test_resume_rejects_changed_evidence_and_preserves_failures():
    current = {
        "schema_version": 1,
        "suite_id": "test",
        "source": {"native_sha256": "native", "harness_sha256": "harness"},
        "protocol": {"memory_budget_bytes": 1024},
        "cases": [
            {
                "id": "one",
                "command": ["worker.py"],
                "status": "pending",
                "result": "one.json",
            }
        ],
    }
    previous = copy.deepcopy(current)
    previous["cases"][0]["status"] = "timeout"
    assert suite.merge_resume(previous, current)["cases"][0]["status"] == "timeout"
    changed = copy.deepcopy(current)
    changed["source"]["native_sha256"] = "new-native"
    with pytest.raises(ValueError, match="source changed"):
        suite.merge_resume(previous, changed)
    changed = copy.deepcopy(current)
    changed["cases"][0]["command"].append("--different")
    with pytest.raises(ValueError, match="changed command"):
        suite.merge_resume(previous, changed)


def test_platform_plan_keeps_unsupported_case_explicit(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "gpu",
                        "group": "gpu",
                        "command": ["scripts/gpu.py"],
                        "platforms": ["other-os"],
                    }
                ]
            }
        )
    )
    cases = suite.load_cases(
        argparse.Namespace(phase="extra", plan=plan, group=None, case=None, limit=None)
    )
    assert cases[0]["status"] == "unsupported_platform"


def test_sidecar_placeholder_is_resolved_without_recording_host_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "telemetry", lambda: {})
    (tmp_path / "logs").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "worker.py").write_text(
        "from pathlib import Path\nimport json,sys\n"
        "p=Path(sys.argv[1]); p.write_bytes(b'array-data')\n"
        "print(json.dumps({'array':p.name}))\n"
    )
    case = {
        "id": "gradient",
        "command": ["worker.py", "{output_dir}/results/{case_id}.npz"],
        "result": "results/gradient.json",
    }
    result = suite.execute_case(case, tmp_path, 3, 2**30, 0.01)
    assert result["status"] == "success"
    assert (tmp_path / "results/gradient.npz").read_bytes() == b"array-data"
    assert case["command"][1] == "{output_dir}/results/{case_id}.npz"


@pytest.mark.parametrize("denied_signal", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.parametrize(
    "snapshot,live",
    [
        ("", False),
        ("999 S\n", False),
        ("321 Z\n321 Z+\n999 R\n", False),
        ("321 Z\n321 S\n", True),
    ],
)
def test_cleanup_permission_error_requires_no_live_group_members(
    monkeypatch, denied_signal, snapshot, live
):
    process = Mock(pid=321)
    sent = []

    def killpg(group, sig):
        assert group == process.pid
        sent.append(sig)
        if sig == denied_signal:
            raise PermissionError(errno.EPERM, "Operation not permitted")

    def inspect_group(command, *, text):
        assert command == ["ps", "-axo", "pgid=,stat="] and text
        return snapshot

    monkeypatch.setattr(suite.os, "killpg", killpg)
    monkeypatch.setattr(suite.subprocess, "check_output", inspect_group)
    if live:
        with pytest.raises(PermissionError) as error:
            suite.terminate_group(process)
        assert error.value.errno == errno.EPERM
    else:
        suite.terminate_group(process)
        assert sent == [signal.SIGTERM, signal.SIGKILL]
        assert process.wait.call_args_list[-1].args == ()
        assert process.wait.call_args_list[-1].kwargs == {}


def test_cleanup_does_not_hide_other_permission_or_inspection_errors(monkeypatch):
    def denied(group, sig):
        raise PermissionError(errno.EACCES, "Access denied")

    monkeypatch.setattr(suite.os, "killpg", denied)
    inspect = Mock(side_effect=RuntimeError("ps unavailable"))
    monkeypatch.setattr(suite.subprocess, "check_output", inspect)
    with pytest.raises(PermissionError) as error:
        suite.signal_group(321, signal.SIGTERM)
    assert error.value.errno == errno.EACCES
    inspect.assert_not_called()

    def eperm(group, sig):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(suite.os, "killpg", eperm)
    with pytest.raises(RuntimeError, match="ps unavailable"):
        suite.signal_group(321, signal.SIGKILL)


def test_darwin_exiting_group_is_observed_until_gone(monkeypatch):
    def denied(group, sig):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(suite.os, "killpg", denied)
    inspect = Mock(side_effect=["321 ?E\n", "321 ?E\n", ""])
    monkeypatch.setattr(suite.subprocess, "check_output", inspect)
    sleep = Mock()
    monkeypatch.setattr(suite.time, "sleep", sleep)
    suite.signal_group(321, signal.SIGKILL)
    assert inspect.call_count == 3
    assert sleep.call_count == 2


def test_stuck_exiting_group_is_fatal_with_observed_state(monkeypatch):
    def denied(group, sig):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(suite.os, "killpg", denied)
    monkeypatch.setattr(
        suite.subprocess, "check_output", lambda *args, **kwargs: "321 ?E\n"
    )
    monkeypatch.setattr(suite.time, "monotonic", Mock(side_effect=[0, 6]))
    with pytest.raises(PermissionError) as error:
        suite.signal_group(321, signal.SIGKILL)
    assert "did not exit: ['?E']" in error.value.__notes__[0]


def test_successful_kill_also_waits_for_worker_exit(monkeypatch):
    process = Mock(pid=321)
    monkeypatch.setattr(suite.os, "killpg", Mock())
    inspect = Mock(side_effect=["321 R\n", "321 ?E\n", ""])
    monkeypatch.setattr(suite.subprocess, "check_output", inspect)
    monkeypatch.setattr(suite.time, "sleep", Mock())
    suite.terminate_group(process)
    assert inspect.call_count == 3


def test_memory_limit_is_durable_before_cleanup_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "telemetry", lambda: {"load_average": [0, 0, 0]})
    (tmp_path / "logs").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "worker.py").write_text("import time; time.sleep(20)")
    original = suite.terminate_group
    calls = []

    def cleanup(process):
        calls.append(process.pid)
        persisted = json.loads((tmp_path / "manifest.json").read_text())
        assert persisted["status"] == "memory_limit"
        assert persisted["limit_event"]["max_process_tree_rss_bytes"] > 1
        assert persisted["limit_event"]["process_group"] == process.pid
        original(process)  # Leave no real worker behind in this failure simulation.
        raise PermissionError(errno.EPERM, "simulated cleanup denial")

    monkeypatch.setattr(suite, "terminate_group", cleanup)
    case = {"id": "limit", "command": ["worker.py"], "result": "results/limit.json"}
    with pytest.raises(PermissionError, match="simulated cleanup denial"):
        suite.execute_case(
            case,
            tmp_path,
            3,
            1,
            0.01,
            checkpoint=lambda attempt: suite.atomic_json(
                tmp_path / "manifest.json", attempt
            ),
        )
    assert len(calls) == 1
    persisted = json.loads((tmp_path / "manifest.json").read_text())
    assert persisted["status"] == "memory_limit"
    assert "simulated cleanup denial" in persisted["cleanup_error"]
    assert json.loads((tmp_path / "logs/limit.attempt-1.json").read_text()) == persisted

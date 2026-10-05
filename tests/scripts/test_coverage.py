"""The CI coverage gate preserves its thresholds and fails on missing evidence."""

import subprocess
from fractions import Fraction

import pytest

from _scripts import load

pytestmark = pytest.mark.interface

coverage_gate = load("check_coverage")


def _report(path, covered, total):
    # A rounded line-rate must not decide whether a boundary passes.
    path.write_text(
        f'<coverage lines-covered="{covered}" lines-valid="{total}" line-rate="1.0" />',
        encoding="utf-8",
    )
    return path


def test_overall_coverage_uses_exact_counts(tmp_path):
    report = _report(tmp_path / "coverage.xml", 1, 3)
    assert coverage_gate.coverage_percent(report) == Fraction(100, 3)


@pytest.mark.parametrize(
    ("head_covered", "patch_status", "expected"),
    [
        (94900, 0, 0),  # Exactly 0.1 percentage points below 95% passes.
        (94899, 0, 1),  # A 0.101-point drop fails without rounding it away.
        (95000, 1, 1),  # Passing overall coverage cannot hide a patch failure.
    ],
)
def test_both_gates_control_the_exit_status(
    tmp_path, monkeypatch, capsys, head_covered, patch_status, expected
):
    head = _report(tmp_path / "head.xml", head_covered, 100000)
    base = _report(tmp_path / "base.xml", 95000, 100000)
    calls = []

    def run(command, *, check):
        calls.append(command)
        assert not check
        return subprocess.CompletedProcess(command, patch_status)

    monkeypatch.setattr(coverage_gate.subprocess, "run", run)
    status = coverage_gate.main([str(head), str(base), "--compare-branch", "a" * 40])
    assert status == expected
    assert len(calls) == 1  # Still show patch coverage when overall coverage fails.
    assert calls[0][:2] == ["diff-cover", str(head)]
    assert "--compare-branch=" + "a" * 40 in calls[0]
    assert "--fail-under=90" in calls[0]
    assert "base 95.000%" in capsys.readouterr().out


@pytest.mark.parametrize(
    "contents",
    [
        None,  # A missing report cannot silently disable the gate.
        "<coverage",
        "<coverage />",
        '<coverage lines-covered="0" lines-valid="0" />',
        '<coverage lines-covered="2" lines-valid="1" />',
    ],
)
def test_invalid_report_fails_the_command(tmp_path, contents):
    head = tmp_path / "head.xml"
    if contents is not None:
        head.write_text(contents, encoding="utf-8")
    base = _report(tmp_path / "base.xml", 1, 1)
    with pytest.raises(SystemExit) as error:
        coverage_gate.main([str(head), str(base), "--compare-branch", "HEAD"])
    assert error.value.code == 2

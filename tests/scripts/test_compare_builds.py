"""The build comparison pairs rounds, checks agreement first and applies its limits."""

import argparse
import io
import json
import subprocess

import numpy as np
import pytest

from _scripts import load
from _support import ROOT

pytestmark = pytest.mark.interface

compare = load("compare_builds")


def test_steps_alternate_which_tree_runs_first():
    assert list(compare.abba(4)) == [
        ("baseline", "candidate"),
        ("candidate", "baseline"),
        ("baseline", "candidate"),
        ("candidate", "baseline"),
    ]


@pytest.mark.parametrize(
    "seconds,ratio,passed",
    [
        (2e-3, 1.05, True),
        (2e-3, 1.06, False),
        (5e-4, 1.10, True),
        (5e-4, 1.11, False),
    ],
)
def test_sub_millisecond_calls_get_the_wider_limit(seconds, ratio, passed):
    assert compare.verdict(seconds, ratio) is passed


def test_memory_limit_applies_only_to_large_growth():
    assert compare.memory_verdict(4.0, 40.0)
    assert compare.memory_verdict(100.0, 105.0)
    assert not compare.memory_verdict(100.0, 106.0)


def test_values_flatten_tuples_arrays_and_scalars():
    result = (np.array([[1, 2j]]), 3.0, (4 + 1j,))
    assert compare.values(result) == [1, 2j, 3, 4 + 1j]


def test_agreement_is_relative_to_the_largest_magnitude():
    baseline = [[3.0, 4.0], [0.0, 0.0]]
    assert compare.agreement(baseline, [[3.0, 4.0], [0.0, 5e-12]], 1e-12) == 1e-12
    assert compare.agreement(baseline, baseline[:1], 1e-12) is None


def test_summary_uses_paired_ratios_and_fails_on_disagreement():
    def entry(seconds, values=None):
        return {"seconds": seconds, **({"values": values} if values else {})}

    rounds = [
        {
            "baseline": {"a": entry(1.0, [[1, 0]]), "b": entry(1.0, [[1, 0]])},
            "candidate": {"a": entry(1.2, [[1, 0]]), "b": entry(1.0, [[2, 0]])},
        },
        {
            "baseline": {"a": entry(2.0), "b": entry(1.0)},
            "candidate": {"a": entry(2.0), "b": entry(1.0)},
        },
        {
            "baseline": {"a": entry(1.0), "b": entry(1.0)},
            "candidate": {"a": entry(1.0), "b": entry(1.0)},
        },
    ]
    calls = compare.summarize(rounds, ["a", "b"], 1e-12)
    assert calls["a"]["ratio"] == 1.0
    assert calls["a"]["ratio_range"] == [1.0, 1.2]
    assert calls["a"]["passed"]
    assert not calls["b"]["agrees"]
    assert not calls["b"]["passed"]


def test_ref_baseline_requires_the_same_native_sources(monkeypatch):
    monkeypatch.setattr(
        compare.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1),
    )
    with (
        pytest.raises(SystemExit, match="native sources differ"),
        compare.baseline_from_ref("HEAD", ROOT / "python"),
    ):
        pass


def test_every_call_name_is_unique_and_framework_calls_are_marked():
    assert len(compare.CALLS) == len(set(compare.CALLS))
    for name in compare.CALLS:
        framework = name.split("-")[0]
        if framework in ("advect", "jax", "torch"):
            assert compare.FRAMEWORK[name] == framework


def test_worker_failure_reaps_both_processes(monkeypatch):
    workers = []
    original = compare.Worker

    def start(*args):
        worker = original(*args)
        workers.append(worker)
        return worker

    monkeypatch.setattr(compare, "Worker", start)
    arguments = argparse.Namespace(min_sample=0.001, samples=1, threads=2)
    with pytest.raises(RuntimeError, match="stopped at missing-call"):
        compare.run_round(
            {"baseline": ROOT / "python", "candidate": ROOT / "python"},
            ["missing-call"],
            arguments,
            0,
        )
    assert len(workers) == 2
    assert all(worker.process.poll() is not None for worker in workers)


def test_worker_keeps_library_prints_out_of_its_protocol(monkeypatch, capsys):
    def setup(_):
        print("setup message")

        def call():
            print("call message")
            return 2.0

        return call

    monkeypatch.setitem(compare.CALLS, "noisy", setup)
    monkeypatch.setattr(
        compare.sys, "stdin", io.StringIO('{"name":"noisy","record":true}\n')
    )
    compare.serve(ROOT / "python", 0.0001, 1)
    captured = capsys.readouterr()
    assert json.loads(captured.out)["values"] == [[2.0, 0.0]]
    assert "setup message" in captured.err and "call message" in captured.err


@pytest.mark.workflows
def test_identical_trees_agree_and_pass(tmp_path):
    output = tmp_path / "result.json"
    status = compare.main(
        [
            "--baseline",
            str(ROOT / "python"),
            "--match",
            "^sphere-scatter-l1$",
            "--rounds",
            "2",
            "--samples",
            "1",
            "--min-sample",
            "0.001",
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_text())
    entry = result["calls"]["sphere-scatter-l1"]
    assert entry["agrees"]
    assert entry["relative_difference"] == 0.0
    assert result["baseline"] == result["candidate"]
    # Timing of one tiny call against itself may still exceed the limit.
    assert status in (0, 1)

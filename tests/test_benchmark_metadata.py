"""Benchmark metadata stays portable without changing measured-build fingerprints."""

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import pytest
import threadpoolctl

from treams_rs import _native

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script,args,key",
    [
        ("benchmark_sampling_blas.py", ["16", "4"], "libraries"),
        (
            "benchmark_cluster.py",
            ["--worker", "rust", "--workload", "bessel-forward", "--samples", "4"],
            "blas",
        ),
        (
            "benchmark_illumination.py",
            [
                "--worker",
                "--backend",
                "selected",
                "--phase",
                "forward",
                "--particles",
                "2",
                "--lmax",
                "1",
                "--warmup",
                "0",
            ],
            "threadpools",
        ),
    ],
)
def test_library_metadata(script, args, key, monkeypatch, capsys, tmp_path):
    pool = {
        "filepath": str(tmp_path / "libblas.so"),
        "version": "1.2",
        "num_threads": 1,
    }
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [pool])
    # Exercise serialization in ordinary debug-build CI; these are not timing gates.
    monkeypatch.setattr(_native, "build_profile", lambda: "release")
    if hasattr(os, "sched_setaffinity"):
        monkeypatch.setattr(os, "sched_setaffinity", lambda _pid, _cpus: None)
    monkeypatch.setenv("BENCH_THREADS", "1")
    if script != "benchmark_sampling_blas.py":
        args = [*args, "--repeats", "1"]
    if script == "benchmark_illumination.py":
        args = [*args, "--array-output", str(tmp_path / "result.npz")]
    path = ROOT / "scripts" / script
    monkeypatch.setattr(sys, "argv", [str(path), *args])
    runpy.run_path(str(path), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result[key] == [{**pool, "filepath": "libblas.so"}]


def test_gpu_raw_artifact_checksums():
    evidence = json.loads(
        (ROOT / "benchmarks/gpu-fields-qualification.json").read_text()
    )
    for filename, expected in evidence["raw_sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, (
            filename
        )

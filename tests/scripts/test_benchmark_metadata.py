"""Benchmark metadata stays portable without changing measured-build fingerprints."""

import json
import os
import runpy
import sys
from pathlib import Path

import pytest
import threadpoolctl

# The metadata reads the extension's build profile, which the tests patch.
from treams_rs import _native

from _scripts import ROOT, load

pytestmark = pytest.mark.interface


@pytest.mark.parametrize(
    "script,args,key",
    [
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
    args = [*args, "--repeats", "1"]
    if script == "benchmark_illumination.py":
        args = [*args, "--array-output", str(tmp_path / "result.npz")]
    path = ROOT / "scripts" / script
    monkeypatch.setattr(sys, "argv", [str(path), *args])
    runpy.run_path(str(path), run_name="__main__")
    text = capsys.readouterr().out
    for local in (tmp_path, ROOT, Path.home()):
        assert str(local) not in text
    result = json.loads(text)
    assert result[key] == [{**pool, "filepath": "libblas.so"}]


def test_illumination_upstream_workers_are_isolated(monkeypatch, capsys, tmp_path):
    """Actual upstream full/selected solves agree without importing the Rust package."""
    import argparse
    import builtins

    import numpy as np

    script = load("benchmark_illumination")
    args = argparse.Namespace(
        particles=2,
        lmax=1,
        columns=2,
        strength="moderate",
        threads=1,
        rtol=1e-10,
        restart=30,
        max_iterations=300,
        repeats=1,
        warmup=0,
        phase="forward",
    )
    radii, epsilon, positions, _, incident = script.case(args)
    from treams_rs import diff

    expected = diff.sphere_cluster_factor(1, 1.3, radii, epsilon, positions).solve(
        incident
    )
    original_import = builtins.__import__

    def without_rust(name, *args, **kwargs):
        assert not name.startswith("treams_rs"), "upstream RSS includes Rust import"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_rust)
    for backend in ("treams-full", "treams-selected"):
        args.backend = backend
        args.array_output = tmp_path / f"{backend}.npz"
        script.worker(args)
        result = json.loads(capsys.readouterr().out)
        assert result["backend"] == backend
        assert result["native_profile"] is None
        assert result["backward_seconds"] == []
        assert result["backward_median_seconds"] is None
        assert len(result["fresh_seconds"]) == len(result["reuse_seconds"]) == 1
        assert result["peak_rss_mib"] >= result["baseline_rss_mib"]
        assert "treams_package_sha256" in result
        with np.load(args.array_output) as arrays:
            np.testing.assert_allclose(arrays["value"], expected, rtol=2e-9, atol=1e-12)

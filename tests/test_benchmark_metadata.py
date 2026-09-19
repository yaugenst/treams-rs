"""Benchmark metadata stays portable without changing measured-build fingerprints."""

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


def test_illumination_upstream_workers_are_isolated(monkeypatch, capsys, tmp_path):
    """Actual upstream full/selected solves agree without importing the Rust package."""
    import argparse
    import builtins

    import numpy as np

    script = runpy.run_path(str(ROOT / "scripts/benchmark_illumination.py"))
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
    radii, epsilon, positions, _, incident = script["case"](args)
    from treams_rs import diff

    expected = diff.cluster_factor(1, 1.3, radii, epsilon, positions).solve(incident)
    original_import = builtins.__import__

    def without_rust(name, *args, **kwargs):
        assert not name.startswith("treams_rs"), "upstream RSS includes Rust import"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_rust)
    for backend in ("treams-full", "treams-selected"):
        args.backend = backend
        args.array_output = tmp_path / f"{backend}.npz"
        script["worker"](args)
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


@pytest.mark.parametrize("transpose", [False, True])
def test_large_parity_check_is_bounded_and_checks_last_element(transpose, monkeypatch):
    import numpy as np

    check = runpy.run_path(str(ROOT / "scripts/benchmark_cluster.py"))[
        "_assert_allclose"
    ]
    expected = np.arange(131076, dtype=np.float64).reshape(3, -1).astype(complex)
    actual = expected.copy()
    if transpose:
        actual, expected = actual.T[::-1], expected.T[::-1]
    original = np.testing.assert_allclose
    sizes = []

    def bounded(a, b, **kwargs):
        sizes.append(a.size)
        assert a.size <= 65536
        original(a, b, **kwargs)

    monkeypatch.setattr(np.testing, "assert_allclose", bounded)
    check(actual, expected)
    assert sum(sizes) == actual.size
    actual.flat[-1] += 1
    with pytest.raises(AssertionError, match="flat indices 131072:131076"):
        check(actual, expected)


def test_bounded_parity_preserves_nonfinite_and_shape_checks():
    import numpy as np

    check = runpy.run_path(str(ROOT / "scripts/benchmark_cluster.py"))[
        "_assert_allclose"
    ]
    check([np.nan, np.inf, -np.inf], [np.nan, np.inf, -np.inf])
    with pytest.raises(AssertionError):
        check([np.inf], [-np.inf])
    with pytest.raises(AssertionError, match="shape mismatch"):
        check(np.ones((2, 1)), np.ones((1, 2)))

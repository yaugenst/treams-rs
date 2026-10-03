"""Provenance digests and thread pinning shared by the benchmark scripts.

Scripts import this module as a sibling (``from _harness import ...``), as they
import each other. It loads neither treams_rs nor threadpoolctl at import time:
upstream-only benchmark workers must not load the native package, and tests
replace threadpoolctl.threadpool_info. audit_qualification.py deliberately keeps
an independent implementation of the digests; tests/scripts/test_audit_qualification.py
requires both to agree.
"""

from __future__ import annotations

import hashlib
import os
import resource
import sys
from pathlib import Path

# Every native, BLAS and OpenMP pool that a benchmark script or collector may start.
THREAD_VARIABLES = (
    "TREAMS_RS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def python_source_sha256(directory: str | Path) -> str:
    """Digest of the names and contents of a package's top-level Python files."""
    digest = hashlib.sha256()
    for path in sorted(Path(directory).glob("*.py")):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def package_sha256(directory: str | Path, suffixes: tuple[str, ...]) -> str:
    """Digest of relative paths and contents of every file with ``suffixes``."""
    directory = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.suffix in suffixes:
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def pinned_threads(threads: int | str) -> dict[str, str]:
    """Environment entries limiting every thread pool to ``threads``."""
    return dict.fromkeys(THREAD_VARIABLES, str(threads))


def cpu_affinity() -> list[int] | None:
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def threadpools() -> list[dict]:
    """Loaded BLAS/OpenMP pools, naming library files without host paths."""
    from threadpoolctl import threadpool_info

    return [
        {**pool, "filepath": Path(pool["filepath"]).name} for pool in threadpool_info()
    ]


def peak_rss_mib() -> float:
    """Peak resident memory of this process so far."""
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor

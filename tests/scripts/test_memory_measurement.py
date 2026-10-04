"""Memory workers retain native peaks without inheriting the driver's high water."""

import json
import subprocess
import sys

import pytest

from _support import ROOT

pytestmark = pytest.mark.interface


@pytest.mark.skipif(sys.platform != "linux", reason="Linux exec memory accounting")
def test_memory_peak_excludes_parent_and_keeps_freed_native_allocation():
    # The child must measure its own smaller peak even while the driver holds
    # large comparison results. Touching every mmap page commits the allocation;
    # closing it before sampling checks that short-lived native buffers count.
    retained = bytearray(96 * 1024 * 1024)
    program = """
import json
import mmap
import sys
sys.path.insert(0, sys.argv[1])
from compare_builds import memory_peak_mib

before = memory_peak_mib()
with mmap.mmap(-1, 32 * 1024 * 1024) as allocation:
    for offset in range(0, len(allocation), mmap.PAGESIZE):
        allocation[offset] = 1
print(json.dumps([before, memory_peak_mib()]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(ROOT / "scripts")],
        capture_output=True,
        text=True,
        check=True,
    )
    before, peak = json.loads(completed.stdout)
    assert len(retained) == 96 * 1024 * 1024
    assert before < 64
    # Allow for page accounting and import overhead without accepting a flat
    # inherited high-water mark or a current-RSS sample after the unmap.
    assert 24 < peak - before < 40

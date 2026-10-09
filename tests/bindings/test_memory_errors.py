"""Native requests beyond the memory the system grants raise MemoryError.

Rust aborts the process when an allocation fails, which takes the interpreter and a
Jupyter kernel with it. treams-core reserves its large outputs and workspaces
fallibly, and the bindings raise the refusal (``Error::OutOfMemory``) as
``MemoryError``. Each request runs in a subprocess whose address space
``RLIMIT_AS`` caps, so the system refuses large requests or stops growing outputs
within a small memory allowance. tests/lattice checks the
same error for ``lattice.cube`` without a limit.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.interface

#: Additional address space after imports; also bounds incrementally grown outputs.
HEADROOM = 64 << 20

#: Calls whose dense output or workspace exceeds the allowance, by native function.
REQUESTS = {
    # 100000 points and 2000 plane waves: a 9.6 GB complex field operator.
    "plane_field": """
        vectors = np.tile([0.0, 0.0, 1.0], (2000, 1))
        diff.plane_field(None, np.zeros((100_000, 3)), vectors, [0] * 2000)
    """,
    # The 19200 modes of lmax 30 at 10 positions from 42000 cylindrical modes: a
    # 12.9 GB expansion matrix.
    "cw_to_sw": """
        spheres = tr.SphericalBasis.default(30, 10)
        cylinders = tr.CylindricalBasis.default(np.linspace(-0.5, 0.5, 1000), 10)
        diff.expansion(spheres, cylinders, [1.0, 1.0])
    """,
    # The matrix-free cluster stores no dense matrix, but the translation plan of
    # lmax 30 couples its 1920 modes through 46 million terms: 1.1 GB.
    "iterative_sphere_cluster": """
        tr.iterative.SphereCluster(30, 1.0, [1.0], [2.0], [[0.0, 0.0, 0.0]])
    """,
    # 200 spheres at lmax 10 have 48000 modes: a 37 GB complex coupling matrix.
    "sphere_cluster": """
        grid = np.stack(np.meshgrid(*3 * [np.arange(6.0)], indexing="ij"), axis=-1)
        positions = grid.reshape(-1, 3)[:200]
        diff.sphere_cluster(10, 1.0, np.full(200, 0.1), np.full(200, 2.0), positions)
    """,
    # 100000 points and the 1920 modes of lmax 30: a 9.2 GB complex field matrix.
    "field_operator": """
        points = np.full((100_000, 3), 2.0)
        diff.field_operator(points, tr.SphericalBasis.default(30), [1.0, 1.0])
    """,
    # A circle of radius 1e5 pitches holds 3.1e10 orders: 500 GB of int64 pairs.
    "diffraction_orders": """
        tr.lattice.diffr_orders_circle(np.eye(2), 1e5)
    """,
}

#: Imports first, then the limit: only the request has to fit under it.
SCRIPT = """
import resource
import os

import numpy as np

import treams_rs as tr
from treams_rs import diff

_, hard = resource.getrlimit(resource.RLIMIT_AS)
with open("/proc/self/statm") as status:
    current = int(status.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
limit = current + {headroom}
limit = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
try:
{request}
except MemoryError as error:
    print("MemoryError:", error)
"""


@pytest.mark.skipif(
    sys.platform != "linux", reason="RLIMIT_AS caps the address space on Linux only"
)
@pytest.mark.parametrize("name", REQUESTS)
def test_request_beyond_the_address_space_raises_memory_error(name):
    request = textwrap.indent(textwrap.dedent(REQUESTS[name]).strip(), "    ")
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(headroom=HEADROOM, request=request)],
        env={**os.environ, "TREAMS_RS_NUM_THREADS": "2"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    # Rust's allocation-failure handler ends the process with SIGABRT (-6).
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert result.stdout.startswith("MemoryError: cannot allocate "), result.stdout

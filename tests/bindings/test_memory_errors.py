"""Native requests beyond the memory the system grants raise MemoryError.

Rust aborts the process when an allocation fails, which takes the interpreter and a
Jupyter kernel with it. treams-core reserves its large outputs and workspaces
fallibly, and the bindings raise the refusal (``Error::OutOfMemory``) as
``MemoryError``. Each request runs in a subprocess whose address space
``RLIMIT_AS`` caps, so the system refuses the request at once instead of granting
it and leaving the outcome to the out-of-memory killer. tests/lattice checks the
same error for ``lattice.cube`` without a limit.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.interface

#: Address space of the subprocess: room for the interpreter, NumPy and two pool
#: threads, but not for the dense output of any request below.
LIMIT = 8 << 30

#: Calls whose dense output exceeds ``LIMIT``, by native function.
REQUESTS = {
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
}

#: Imports first, then the limit: only the request has to fit under it.
SCRIPT = """
import resource

import numpy as np

import treams_rs as tr
from treams_rs import diff

_, hard = resource.getrlimit(resource.RLIMIT_AS)
limit = {limit} if hard == resource.RLIM_INFINITY else min({limit}, hard)
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
        [sys.executable, "-c", SCRIPT.format(limit=LIMIT, request=request)],
        env={**os.environ, "TREAMS_RS_NUM_THREADS": "2"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    # Rust's allocation-failure handler ends the process with SIGABRT (-6).
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert result.stdout.startswith("MemoryError: cannot allocate "), result.stdout

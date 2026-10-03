"""Import repository scripts as ordinary modules for the harness tests.

Scripts import their siblings by module name (``import benchmark_cluster``), as
they do when run as ``python scripts/<name>.py``. Tests patch module attributes
with ``monkeypatch.setattr``; functions resolve module globals at call time.
"""

import importlib
import sys
from types import ModuleType

from _support import ROOT

SCRIPTS = ROOT / "scripts"
# As for `python scripts/<name>.py`, including scripts run with runpy as __main__.
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load(name: str) -> ModuleType:
    """Return the imported ``scripts/<name>.py`` module."""
    return importlib.import_module(name)

"""CPU threads of the native numerical work.

treams-rs runs its parallel kernels and dense linear algebra on one pool of
worker threads that it owns. The pool starts on the first parallel call. A
forked child process (``multiprocessing``, ``ProcessPoolExecutor``, PyTorch
``DataLoader`` workers) builds its own pool instead of waiting on the parent's
threads, which do not exist after ``fork``.

The default budget is the first positive integer among
``TREAMS_RS_NUM_THREADS``, ``RAYON_NUM_THREADS`` and ``OMP_NUM_THREADS`` (the
outermost entry of an OpenMP list), otherwise the CPUs this process may use,
which respects CPU affinity and container quotas. The environment is read once,
when ``treams_rs`` is imported; later changes to it have no effect, so call
``set_num_threads`` instead, also in process-pool initializers. Empty and zero
values mean "not set". Unparsable values raise ``ThreadingWarning`` at import;
they, budgets above the available CPUs and an ``OMP_NUM_THREADS`` limit are
listed in ``thread_info()["diagnostics"]``. joblib's loky backend sets
``OMP_NUM_THREADS`` in its workers, so each worker uses its share of the CPUs.

When threadpoolctl is installed, ``threadpoolctl.threadpool_limits`` limits
treams-rs too: ``threadpool_info()`` lists it with ``user_api`` ``"treams"``.

Results do not depend on the budget: for a given build and CPU, every value and
gradient repeats bit for bit at any number of threads.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager as _contextmanager
from importlib import metadata
from importlib.util import find_spec as _find_spec
from numbers import Integral as _Integral
from typing import TYPE_CHECKING, Any, override

from . import _native

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "ThreadingWarning",
    "get_num_threads",
    "set_num_threads",
    "thread_info",
    "threads",
]


class ThreadingWarning(RuntimeWarning):
    """An ignored or oversubscribing thread setting."""


def thread_info() -> dict[str, Any]:
    """Return the thread budget, where it came from, and the pool state.

    Keys:

    - ``threads``: the budget of the next parallel call.
    - ``source``: ``"set_num_threads"``, the environment variable that set the
      budget, or ``"available_parallelism"``.
    - ``available``: the CPUs this process may use.
    - ``pool_threads``: the workers of the current pool, ``None`` before the
      first parallel call of this process.
    - ``forked``: whether this process was forked after treams-rs was loaded;
      it then builds a pool of its own.
    - ``diagnostics``: ignored, oversubscribing or limiting settings.
    - ``environment``: the variables consulted, in precedence order.

    Never starts the pool.

    Examples
    --------
    >>> import treams_rs as tr
    >>> tr.thread_info()["threads"] >= 1
    True
    """
    return dict(_native.thread_info())


def get_num_threads() -> int:
    """Return the worker-thread budget of the next native parallel call."""
    return int(_native.thread_info()["threads"])


def set_num_threads(threads: int | None) -> None:
    """Set the process-wide worker-thread budget; ``None`` restores the default.

    ``threads`` must be a positive integer. A parallel region that has started
    finishes on its pool; later regions use the new budget. A budget above the
    available CPUs is honored with a ``ThreadingWarning``, issued before the
    budget changes, because oversubscription slows dense linear algebra. Forked
    children inherit the budget and build their own pool of that size.

    Examples
    --------
    >>> import treams_rs as tr
    >>> tr.set_num_threads(1)
    >>> tr.get_num_threads()
    1
    >>> tr.set_num_threads(None)
    """
    if threads is not None:
        if (
            isinstance(threads, bool)
            or not isinstance(threads, _Integral)
            or threads < 1
        ):
            raise ValueError("threads must be a positive integer or None")
        threads = int(threads)
        available = int(_native.thread_info()["available"])
        if threads > available:
            warnings.warn(
                f"set_num_threads({threads}) exceeds the {available} CPUs this "
                "process may use; oversubscription slows dense linear algebra",
                ThreadingWarning,
                stacklevel=2,
            )
    _native.set_num_threads(threads)


@_contextmanager
def threads(n: int | None) -> Iterator[None]:
    """Set the process-wide worker-thread budget for the ``with`` block.

    Restores the previous budget on exit, like
    ``threadpoolctl.threadpool_limits``. The budget is process-wide, not
    thread-local: other Python threads that call treams-rs inside the block use
    it too. Alternating between two budgets reuses their pools.

    Blocks in different Python threads must not overlap: each restores the
    budget it saw on entry, in exit order. For a thread pool, call
    ``set_num_threads(k)`` once before starting the workers.

    Examples
    --------
    >>> import treams_rs as tr
    >>> with tr.threads(1):
    ...     tr.get_num_threads()
    1
    """
    info = _native.thread_info()
    previous = info["threads"] if info["source"] == "set_num_threads" else None
    try:
        set_num_threads(n)
        yield
    finally:
        _native.set_num_threads(previous)


def _register_threadpoolctl() -> None:
    """Teach threadpoolctl to find and limit the treams-rs pool."""
    import threadpoolctl

    class TreamsController(threadpoolctl.LibController):
        """The treams-rs pool: the extension module that exports its budget."""

        user_api = "treams"
        internal_api = "treams_rs"
        filename_prefixes = ("_native",)
        check_symbols = ("treams_rs_num_threads",)

        # threadpoolctl leaves these abstract return types unannotated; Pyrefly
        # infers None from their empty bodies.
        @override
        def get_num_threads(self) -> int:  # pyrefly: ignore[bad-override]
            return get_num_threads()

        @override
        def set_num_threads(self, num_threads: int) -> None:
            _native.set_num_threads(max(1, int(num_threads)))

        @override
        def get_version(self) -> str | None:  # pyrefly: ignore[bad-override]
            try:
                return metadata.version("treams-rs")
            except metadata.PackageNotFoundError:
                return None

    threadpoolctl.register(TreamsController)


# A forked child also detects the fork by its process id; the hook covers a
# reused process id as well.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_native.after_fork)

if _find_spec("threadpoolctl") is not None:
    _register_threadpoolctl()

# Unparsable settings are mistakes worth a warning. Oversubscription and an
# OMP_NUM_THREADS limit are deliberate often enough to stay in diagnostics.
for _message in _native.thread_info()["diagnostics"]:
    if _message.startswith("ignored "):
        warnings.warn(_message, ThreadingWarning, stacklevel=2)

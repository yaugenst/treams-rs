"""Select an existing derivative adapter from values, before any array conversion.

There is no ambient backend: plain Python/NumPy inputs keep the NumPy path.
Only the framework that owns an input is loaded. Physics objects carry their
backend, while supported containers and materials expose their numeric leaves.
"""

from __future__ import annotations

import importlib
import sys
from functools import wraps
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from ._framework_backend import Backend


_PLAIN_TYPES = (int, float, complex, bool, str, bytes, type(None), np.ndarray)


def backend_for(*values: Any) -> Backend | None:
    """The one framework owning these values, or None for ordinary NumPy work.

    Mixing distinct autodiff frameworks is an error, including nested inputs.
    This inspects container structure and types, never tracer payloads.
    """
    # Read current imports once per call: importing a framework later works,
    # while an ordinary NumPy process needs no input traversal at all.
    advect = sys.modules.get("advect")
    jax = sys.modules.get("jax")
    torch = sys.modules.get("torch")
    autograd = sys.modules.get("autograd.tracer")
    if advect is None and jax is None and torch is None and autograd is None:
        return None
    selected = None
    pending = list(values)
    for value in pending:
        cls = type(value)
        if cls in _PLAIN_TYPES:
            continue
        if isinstance(value, np.generic):
            continue
        if isinstance(value, (tuple, list)):
            pending.extend(value)
            continue
        if isinstance(value, dict):
            pending.extend(value.values())
            continue
        if cls.__name__ == "Material" and cls.__module__ in (
            "treams_rs._material",
            "treams_rs._framework_backend",
        ):
            pending.extend((value.epsilon, value.mu, value.kappa))
            continue
        # Root physics objects hold NumPy arrays. Avoid their expensive
        # missing-attribute diagnostics; framework objects live in _framework*.
        if cls.__module__.startswith("treams_rs.") and not cls.__module__.startswith(
            "treams_rs._framework"
        ):
            continue
        backend = getattr(value, "_backend", None)
        if backend is None:
            if advect is not None and advect.is_traced(value):
                backend = importlib.import_module(".advect", __package__)._backend
            elif jax is not None and isinstance(value, (jax.Array, jax.core.Tracer)):
                backend = importlib.import_module(".jax", __package__)._backend
            elif torch is not None and isinstance(value, torch.Tensor):
                backend = importlib.import_module(".torch", __package__)._backend
            elif autograd is not None and autograd.isbox(value):
                backend = importlib.import_module(".autograd", __package__)._backend
            else:
                continue
        if selected is not None and backend is not selected:
            raise TypeError("cannot mix autodiff backends in one treams-rs operation")
        selected = backend
    return selected


def namespace(backend: Backend) -> ModuleType:
    """The existing adapter namespace for a selected backend."""
    return importlib.import_module(f".{backend.xp.__name__.split('.')[0]}", __package__)


def autodiff_method[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    """Promote a constant NumPy receiver when a method receives framework inputs."""

    @wraps(method)
    def call(*args: P.args, **kwargs: P.kwargs) -> Any:
        backend = backend_for(args[1:], kwargs)
        if backend is None:
            return method(*args, **kwargs)
        from ._promotion import promote

        receiver = promote(args[0], backend)
        target = getattr(receiver, method.__name__, None)
        if receiver is args[0] or target is None:
            raise NotImplementedError(
                f"{type(args[0]).__name__}.{method.__name__} has no autodiff implementation"
            )
        return target(*args[1:], **kwargs)

    return call

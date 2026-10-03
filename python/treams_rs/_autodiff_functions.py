"""Array-function dispatch without importing optional frameworks on NumPy calls.

The wrapped native callable remains the NumPy implementation, including its
ufunc methods and attributes. Framework calls use an existing native record;
mutating ``out`` and partially initialized ``where`` outputs are NumPy-only.
"""

from __future__ import annotations

from functools import update_wrapper, wraps
from importlib import import_module
from inspect import isfunction
from typing import TYPE_CHECKING, Any

from ._dispatch import backend_for

if TYPE_CHECKING:
    from collections.abc import Callable


class Function:
    """A native callable with an input-selected differentiation implementation."""

    def __init__(
        self,
        original: Callable[..., Any],
        implementation: Callable[..., Any],
        *,
        module: str | None = None,
    ):
        self.original = original
        self.implementation = implementation
        update_wrapper(self, original)
        if module is not None:
            self.__module__ = module

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        backend = backend_for(*args, *kwargs.values())
        if backend is None:
            return self.original(*args, **kwargs)
        return self.implementation(backend, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def __reduce__(self) -> Any:
        return _load_function, (self.__module__, self.__name__)


def _load_function(module: str, name: str) -> Any:
    return getattr(import_module(module), name)


def transparent_function(
    original: Callable[..., Any],
    implementation: Callable[..., Any],
    *,
    module: str | None = None,
) -> Any:
    """Retain Python function introspection as well as native ufunc attributes."""
    dispatcher = Function(original, implementation, module=module)
    if not isfunction(original):
        return dispatcher

    @wraps(original)
    def call(*args: Any, **kwargs: Any) -> Any:
        return dispatcher(*args, **kwargs)

    return call


def require_no_out(out: Any = None, where: Any = True) -> None:
    """Reject mutation and unspecified masked values inside differentiation."""
    if out is not None or where is not True:
        raise TypeError(
            "differentiable calls do not support out or where; "
            "apply the framework's where function to the result instead"
        )

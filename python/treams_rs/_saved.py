"""Explicit array state for records whose derivatives can reuse their forward.

The schema depends only on input shapes, input domains and fixed record options.
Runtime branches must share that schema. Saved arrays contain values, never
addresses or handles into a Python or native object registry.

No framework is imported here. Ordinary adapters call ``SavedRecord`` like any
other record; adapters that retain array state use its three additional hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from inspect import signature
from math import prod
from typing import TYPE_CHECKING, Any, cast
from weakref import ref

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from inspect import Signature
    from typing import Protocol

    from numpy.typing import NDArray

    from ._records import Record

    class _ContextFactory(Protocol):
        def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ArraySpec:
    """Shape and exact NumPy dtype of one input, output or saved-state array."""

    shape: tuple[int, ...]
    dtype: np.dtype[Any]

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True)
class SavedRecord:
    """A normal record with a statically shaped, independently owned residual.

    ``state_spec`` receives the native input descriptions: shapes are unchanged,
    and supported input dtypes are promoted to float64 or complex128.
    ``save(context)`` returns arrays matching that schema exactly.
    ``restore(state, *native_primals)`` reconstructs a derivative context without
    rerunning the numerical forward. It may borrow the immutable state arrays;
    it must not modify them. Each restore can be used independently, including
    concurrent and repeated pullbacks or pushforwards.

    ``needs_primals=False`` promises that ``restore(state)`` needs no primal
    values, so an adapter can retain only their shape and dtype metadata.
    """

    evaluate: Record
    state_spec: Callable[[tuple[ArraySpec, ...]], tuple[ArraySpec, ...]]
    save: Callable[[Any], tuple[Any, ...]]
    restore: Callable[..., Any]
    needs_primals: bool = True

    def __call__(self, *values: Any) -> tuple[Any, Any]:
        return self.evaluate(*values)


def saved_record(record: Record) -> SavedRecord | None:
    """Find explicit state metadata without changing a public record signature."""
    if isinstance(record, SavedRecord):
        return record
    # functools.wraps copies function attributes, but a wrapper may change the
    # values or derivative context. Its original's state contract is not valid.
    owner = record.func if isinstance(record, partial) else record
    metadata = getattr(owner, "__treams_saved_record__", None)
    return (
        metadata[1](record) if metadata is not None and metadata[0]() is owner else None
    )


def preserve_state[Function: Callable[..., Any]](
    source: Record, evaluate: Function
) -> Function:
    """Carry state through an output check, resolving it only when requested."""

    def factory(checked: Record) -> SavedRecord | None:
        if isinstance(checked, partial):
            return None
        prepared = saved_record(source)
        return None if prepared is None else replace(prepared, evaluate=checked)

    cast("Any", evaluate).__treams_saved_record__ = (ref(evaluate), factory)
    return evaluate


def native_state[Function: Callable[..., Any]](
    context_type: Any,
    dimensions: Callable[[tuple[ArraySpec, ...]], tuple[Any, ...]],
    *,
    map_context: Callable[..., Any] | None = None,
    needs_primals: bool = True,
) -> Callable[[Function], Function]:
    """Attach native state; value-independent context mappings can drop primals."""

    def decorate(record: Function) -> Function:
        def factory(evaluate: Record) -> SavedRecord | None:
            return (
                None
                if isinstance(evaluate, partial)
                else _native_saved(
                    evaluate,
                    context_type,
                    dimensions,
                    map_context,
                    needs_primals=needs_primals,
                )
            )

        # Local physics records should expire with their last caller, without
        # waiting for cyclic GC. Only a requested SavedRecord owns evaluation.
        cast("Any", record).__treams_saved_record__ = (ref(record), factory)
        return record

    return decorate


def _native_saved(
    evaluate: Record,
    context_type: Any,
    dimensions: Callable[[tuple[ArraySpec, ...]], tuple[Any, ...]],
    map_context: Callable[..., Any] | None = None,
    *,
    needs_primals: bool = True,
) -> SavedRecord:
    def state_spec(inputs: tuple[ArraySpec, ...]) -> tuple[ArraySpec, ...]:
        size = context_type._state_spec(*dimensions(inputs))
        return (ArraySpec((size,), np.dtype(np.uint8)),)

    def save(context: Any) -> tuple[Any, ...]:
        if map_context is not None:
            context = context.native_context
        return (context._state(),)

    def restore(state: tuple[Any, ...], *primals: Any) -> Any:
        context = context_type._from_state(state[0])
        return context if map_context is None else map_context(context, *primals)

    return SavedRecord(
        evaluate,
        state_spec,
        save,
        restore,
        needs_primals=map_context is not None and needs_primals,
    )


def native_record[Function: Callable[..., Any]](
    context_type: Any,
    dimensions: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> Callable[[Function], Function]:
    """Attach state to a public diff record with named static configuration.

    ``dimensions`` sees the record's signature with defaults and partial
    arguments bound. Dynamic arguments are ArraySpec objects, static labels and
    options retain their ordinary values. Only explicit functools.partial
    binding is recognized; arbitrary Python closures remain opaque records.
    """

    def decorate(record: Function) -> Function:
        parameters = signature(record)

        def factory(evaluate: Record) -> SavedRecord:
            def shape(inputs: tuple[ArraySpec, ...]) -> tuple[Any, ...]:
                return dimensions(_arguments(parameters, evaluate, inputs))

            return _native_saved(evaluate, context_type, shape)

        cast("Any", record).__treams_saved_record__ = (ref(record), factory)
        return record

    return decorate


def primal_record[Function: Callable[..., Any]](
    make_context: _ContextFactory,
) -> Callable[[Function], Function]:
    """Restore an input-only context without evaluating its numerical outputs.

    Internal records have required positional inputs (or ``*arguments``) and
    keyword-only defaults. The context constructor receives the same arguments;
    defaults come from the record, and partial binding happens once per wrapper.
    """

    def decorate(record: Function) -> Function:
        defaults = (cast("Any", record).__kwdefaults__ or {}).copy()

        def factory(evaluate: Record) -> SavedRecord:
            args = evaluate.args if isinstance(evaluate, partial) else ()
            keywords = evaluate.keywords if isinstance(evaluate, partial) else {}
            construct = partial(make_context, *args, **(defaults | keywords))

            def restore(_state: tuple[Any, ...], *primals: Any) -> Any:
                return construct(*primals)

            return SavedRecord(
                evaluate, lambda _inputs: (), lambda _context: (), restore
            )

        cast("Any", record).__treams_saved_record__ = (ref(record), factory)
        return record

    return decorate


def _arguments(
    parameters: Signature, evaluate: Record, values: tuple[Any, ...]
) -> dict[str, Any]:
    args = evaluate.args if isinstance(evaluate, partial) else ()
    keywords = evaluate.keywords if isinstance(evaluate, partial) else {}
    bound = parameters.bind(*args, *values, **keywords)
    bound.apply_defaults()
    return bound.arguments


def pack_state(
    values: tuple[Any, ...], specs: tuple[ArraySpec, ...]
) -> tuple[NDArray[np.uint8], ...]:
    """Transport exact array bits even when the framework disables float64."""
    if not isinstance(values, tuple) or len(values) != len(specs):
        raise ValueError("saved state must return one array per state_spec entry")
    packed = []
    for value, spec in zip(values, specs, strict=True):
        array = np.asarray(value)
        if array.shape != spec.shape or array.dtype != spec.dtype:
            raise ValueError(
                "saved state must match state_spec; "
                f"expected {spec.shape} {spec.dtype}, received {array.shape} {array.dtype}"
            )
        if array.dtype.kind not in "buifc":
            raise TypeError("saved state arrays require numeric or boolean dtypes")
        packed.append(np.ascontiguousarray(array).reshape(-1).view(np.uint8))
    return tuple(packed)


def unpack_state(
    values: tuple[Any, ...], specs: tuple[ArraySpec, ...]
) -> tuple[NDArray[Any], ...]:
    """Recover typed views of the byte buffers owned by the framework."""
    return tuple(
        np.asarray(value).view(spec.dtype).reshape(spec.shape)
        for value, spec in zip(values, specs, strict=True)
    )

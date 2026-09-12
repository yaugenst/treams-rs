"""Shared array contract for optional framework bridges; no framework imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

type Array = NDArray[np.float64 | np.complex128]
# Native contexts have operation-specific pullback signatures. The bridge validates
# their flat array contract at execution; numerical validation remains in Rust.
type Record = Callable[..., tuple[Any, Any]]
type Pullback = Callable[..., Any]


def input_array(value: object) -> Array:
    array = np.asarray(value)
    require_dtype(array.dtype)
    return array


def require_dtype(dtype: np.dtype[Any]) -> None:
    if dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        raise TypeError("native adapters require float64 or complex128 parameters")


def execute(
    record: Record, values: tuple[Array, ...]
) -> tuple[tuple[Array, ...], Pullback, bool]:
    output, context = record(*values)
    multiple = isinstance(output, tuple)
    arrays = tuple(
        np.asarray(value, dtype=np.complex128 if np.iscomplexobj(value) else np.float64)
        for value in (output if multiple else (output,))
    )
    if not arrays:
        raise ValueError("a native operation must return at least one array")
    return arrays, context if callable(context) else context.pullback, multiple


def gradients(
    pullback: Pullback,
    cotangents: tuple[Array, ...],
    primals: tuple[Array, ...],
    *,
    conjugate: bool,
) -> tuple[Array, ...]:
    result = pullback(
        *(
            np.asarray(np.conjugate(value) if conjugate else value)
            for value in cotangents
        )
    )
    # A list can itself represent one gradient (e.g. native Euler angles).
    values = result if isinstance(result, tuple) else (result,)
    if len(values) != len(primals):
        raise ValueError("pullback must return one gradient per dynamic parameter")
    output: list[Array] = []
    for value, primal in zip(values, primals, strict=True):
        array = np.asarray(value)
        if array.shape != primal.shape:
            raise ValueError("pullback gradient shape must match its dynamic parameter")
        if conjugate:
            array = np.conjugate(array)
        output.append(
            np.asarray(
                array if np.iscomplexobj(primal) else array.real, dtype=primal.dtype
            )
        )
    return tuple(output)

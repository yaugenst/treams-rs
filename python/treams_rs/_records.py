"""NumPy-side execution of native records for framework adapters and testing.

Advect, JAX, PyTorch and ``treams_rs.testing`` run native records here.
``treams_rs.diff`` defines record, context, pullback and cotangent. The shared
rules:

- A record returns ``(outputs, context_or_pullback)``: one output or a flat tuple
  of outputs, and a context with a ``pullback`` method or the pullback itself.
- Outputs are coerced to float64, or to complex128 when they are complex.
- A pullback result that is not a tuple is one gradient; a list counts as one.
- Gradients of real inputs are projected to their real part.
- Each gradient has its input's shape and dtype. ``reshape=True`` (Advect)
  also accepts a gradient of the same size and reshapes it.
- ``conjugate=True`` implements JAX's bilinear pairing on top of the native real
  pairing by conjugating the cotangents and the gradients.

No framework is imported here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

type Array = NDArray[np.float64 | np.complex128]
# Native contexts have operation-specific pullback signatures. run_record and
# apply_pullback check the flat array structure; Rust validates the numbers.
type Record = Callable[..., tuple[Any, Any]]
type Pullback = Callable[..., Any]

__all__ = [
    "Array",
    "Pullback",
    "Record",
    "apply_pullback",
    "input_array",
    "native_array",
    "require_float64",
    "require_inexact",
    "run_record",
]


def input_array(value: object) -> Array:
    """Convert a dynamic input to a NumPy array without changing its dtype.

    Inputs are never cast: anything but float64 or complex128 raises the
    TypeError of ``require_float64`` instead of silently changing precision.
    """
    array = np.asarray(value)
    require_float64(array.dtype)
    return array


def require_float64(dtype: np.dtype[Any] | str) -> None:
    """Raise TypeError unless ``dtype`` is float64 or complex128.

    Native operations compute in double precision only. ``dtype`` is a NumPy
    dtype or a dtype name, so that a framework dtype NumPy lacks (bfloat16) is
    reported too. The message names the received dtype and the remedy, an
    explicit cast, so that every adapter that shares this check reports the
    same diagnostic.
    """
    if dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        raise TypeError(
            "native adapters require float64 or complex128 parameters; "
            f"received {dtype}. Cast the parameter explicitly before calling."
        )


def require_inexact(dtype: np.dtype[Any] | str) -> None:
    """Validate supported real/complex precision before framework coercion."""
    if dtype not in tuple(
        map(np.dtype, ("float32", "float64", "complex64", "complex128"))
    ):
        raise TypeError(
            "native adapters require float32, float64, complex64 or complex128 "
            f"parameters; received {dtype}. Cast the parameter explicitly before calling."
        )


def native_array(value: object) -> Array:
    """Promote supported floating-point inputs to the native double precision."""
    array = np.asarray(value)
    require_inexact(array.dtype)
    return np.asarray(
        array, dtype=np.complex128 if np.iscomplexobj(array) else np.float64
    )


def run_record(
    record: Record, values: tuple[object, ...]
) -> tuple[tuple[Array, ...], Pullback, bool]:
    """Run ``record(*values)`` and normalize its outputs and pullback.

    Returns the outputs as a nonempty tuple of float64/complex128 arrays, the
    pullback (the context's ``pullback`` method, or the returned callable
    itself) and whether the record returned a tuple, so that an adapter can
    return a single output unwrapped. An empty output tuple raises ValueError.
    """
    output, context = record(*values)
    multiple = isinstance(output, tuple)
    arrays = tuple(
        np.asarray(value, dtype=np.complex128 if np.iscomplexobj(value) else np.float64)
        for value in (output if multiple else (output,))
    )
    if not arrays:
        raise ValueError(
            "a native operation must return at least one array; "
            "received an empty output tuple"
        )
    return arrays, context if callable(context) else context.pullback, multiple


def apply_pullback(
    pullback: Pullback,
    cotangents: tuple[Array, ...],
    primals: tuple[Array, ...],
    *,
    conjugate: bool,
    reshape: bool = False,
) -> tuple[Array, ...]:
    """Apply ``pullback`` to output cotangents and return one gradient per input.

    ``primals`` are the dynamic inputs of the forward call. A result that is
    not a tuple is one gradient, a list included. Each gradient must have its
    input's shape and is returned in its input's dtype, as the real part for a
    real input; a wrong count or shape raises ValueError. ``reshape=True`` also
    accepts a gradient with its input's size and reshapes it to the input's
    shape.
    ``conjugate=True`` conjugates the cotangents before and the gradients after
    the native pullback, which maps JAX's bilinear pairing onto the native
    ``Re(vdot(cotangent, dx))`` pairing.
    """
    result = pullback(
        *(
            np.asarray(np.conjugate(value) if conjugate else value)
            for value in cotangents
        )
    )
    # A list can itself represent one gradient (e.g. native Euler angles).
    values = result if isinstance(result, tuple) else (result,)
    if len(values) != len(primals):
        raise ValueError(
            "pullback must return one gradient per dynamic parameter; "
            f"expected {len(primals)} gradients, received {len(values)}. "
            "Return a tuple in dynamic parameter order."
        )
    output: list[Array] = []
    for index, (value, primal) in enumerate(zip(values, primals, strict=True)):
        array = np.asarray(value)
        if array.shape != primal.shape:
            if reshape and array.size == primal.size:
                array = array.reshape(primal.shape)
            else:
                raise ValueError(
                    "pullback gradient shape must match its dynamic parameter; "
                    f"parameter[{index}] expected shape {primal.shape}, "
                    f"received {array.shape}"
                )
        if conjugate:
            array = np.conjugate(array)
        output.append(
            np.asarray(
                array if np.iscomplexobj(primal) else array.real, dtype=primal.dtype
            )
        )
    return tuple(output)

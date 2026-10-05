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

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

type Array = NDArray[np.float64 | np.complex128]
# Native contexts have operation-specific pullback signatures. run_record and
# apply_pullback check the flat array structure; Rust validates the numbers.
type Record = Callable[..., tuple[Any, Any]]
type Pullback = Callable[..., Any]

_INEXACT_DTYPES = tuple(
    map(np.dtype, ("float32", "float64", "complex64", "complex128"))
)


class ArrayMetadata(Protocol):
    """Only the shape and dtype are needed to normalize an output tangent."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...


__all__ = [
    "Array",
    "DerivativeContext",
    "Pullback",
    "Record",
    "apply_pullback",
    "apply_pushforward",
    "input_array",
    "native_array",
    "record_outputs",
    "require_float64",
    "require_inexact",
    "run_jvp",
    "run_record",
]


@dataclass(frozen=True)
class DerivativeContext:
    """Keep both derivative directions when a Python record reshapes its inputs."""

    pullback: Pullback
    pushforward: Callable[..., Any]
    native_context: Any = None


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
    if dtype not in _INEXACT_DTYPES:
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


def record_outputs(
    record: Record, values: tuple[object, ...]
) -> tuple[tuple[Array, ...], Any, bool]:
    """Run a record, retaining its context for either derivative direction."""
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
    return arrays, context, multiple


def run_record(
    record: Record, values: tuple[object, ...]
) -> tuple[tuple[Array, ...], Pullback, bool]:
    """Run ``record(*values)`` and normalize its outputs and pullback.

    Returns the outputs as a nonempty tuple of float64/complex128 arrays, the
    pullback (the context's ``pullback`` method, or the returned callable
    itself) and whether the record returned a tuple, so that an adapter can
    return a single output unwrapped. An empty output tuple raises ValueError.
    """
    arrays, context, multiple = record_outputs(record, values)
    return arrays, context if callable(context) else context.pullback, multiple


def run_jvp(
    record: Record, primals: tuple[object, ...], tangents: tuple[object, ...]
) -> tuple[tuple[Array, ...], tuple[Array, ...], bool]:
    """Run a fresh native record and its analytic directional derivative.

    Tangents have the shapes and real/complex domains of their inputs. Unlike
    cotangents, they need no conjugation for any framework pairing convention.
    A pullback-only custom record must supply ``pushforward`` to use forward AD.
    """
    values = tuple(native_array(value) for value in primals)
    arrays, context, multiple = record_outputs(record, values)
    return arrays, apply_pushforward(context, tangents, values, arrays), multiple


def apply_pushforward(
    context: Any,
    tangents: tuple[object, ...],
    primals: tuple[ArrayMetadata, ...],
    outputs: tuple[ArrayMetadata, ...],
) -> tuple[Array, ...]:
    """Apply a saved context's pushforward; ``None`` denotes an inactive input."""
    if len(tangents) != len(primals):
        raise ValueError("pushforward requires one tangent per dynamic parameter")
    directions = []
    for index, (primal, tangent) in enumerate(zip(primals, tangents, strict=True)):
        direction = (
            np.zeros(primal.shape, dtype=primal.dtype)
            if tangent is None
            else native_array(tangent)
        )
        if direction.shape != primal.shape:
            raise ValueError(
                "pushforward tangent shape must match its dynamic parameter; "
                f"parameter[{index}] expected shape {primal.shape}, "
                f"received {direction.shape}"
            )
        if not np.iscomplexobj(primal) and np.iscomplexobj(direction):
            raise TypeError("a real dynamic parameter requires a real tangent")
        directions.append(
            np.asarray(
                direction,
                dtype=np.complex128 if np.iscomplexobj(primal) else np.float64,
            )
        )
    pushforward = getattr(context, "pushforward", None)
    if pushforward is None:
        raise NotImplementedError(
            "forward-mode autodiff requires a record context with pushforward"
        )
    tangent_output = pushforward(*directions)
    tangent_values = (
        tangent_output if isinstance(tangent_output, tuple) else (tangent_output,)
    )
    if len(tangent_values) != len(outputs):
        raise ValueError("pushforward must return one tangent per output")
    result = []
    for primal, tangent in zip(outputs, tangent_values, strict=True):
        direction = np.asarray(tangent)
        if direction.shape != primal.shape:
            raise ValueError(
                "pushforward output tangent shape must match its primal output; "
                f"expected {primal.shape}, received {direction.shape}"
            )
        if not np.iscomplexobj(primal) and np.iscomplexobj(direction):
            raise TypeError("a real output requires a real tangent")
        result.append(np.asarray(direction, dtype=primal.dtype))
    return tuple(result)


def apply_pullback(
    pullback: Pullback,
    cotangents: tuple[Array, ...],
    primals: tuple[ArrayMetadata, ...],
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
            if (reshape and array.size == prod(primal.shape)) or (
                primal.shape == () and array.shape == (1,)
            ):
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

"""Directional finite-difference checks for first-order gradients and native pullbacks.

These are diagnostic test oracles, never production derivative implementations.
Complex derivatives use the native real pairing ``Re(vdot(gradient, direction))``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._adapters import execute, input_array

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

    from ._adapters import Array, Record

__all__ = ["check_gradient", "check_pullback"]


def _shaped(
    value: ArrayLike, reference: Array, label: str, *, gradient: bool = False
) -> Array:
    array = np.asarray(value)
    if array.shape != reference.shape:
        raise ValueError(
            f"{label}: expected shape {reference.shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: values must be finite")
    if not gradient and not np.iscomplexobj(reference) and np.any(array.imag != 0):
        raise ValueError(f"{label}: a real value requires a real probe/output")
    # The imaginary part of a gradient for a real input does not contribute to
    # the real pairing. Directions/cotangents are checked separately below.
    return np.asarray(
        array if np.iscomplexobj(reference) else array.real, dtype=reference.dtype
    )


def _probe(
    value: ArrayLike | None, reference: Array, rng: np.random.Generator, label: str
) -> Array:
    if value is None:
        value = rng.standard_normal(reference.shape)
        if np.iscomplexobj(reference):
            value = value + 1j * rng.standard_normal(reference.shape)
    return _shaped(value, reference, label)


def check_pullback(
    record: Record,
    *parameters: ArrayLike,
    directions: tuple[ArrayLike, ...] | None = None,
    cotangents: ArrayLike | tuple[ArrayLike, ...] | None = None,
    step: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    seed: int = 0,
) -> None:
    """Assert a native record's pullback agrees with central finite differences.

    ``record(*parameters)`` returns ``(output, context)``; context is callable or
    has ``pullback(*cotangents)``. An output is an array/scalar or a tuple of them.
    A pullback returns one gradient per parameter, as a tuple for multiple inputs.
    Static labels/options must be captured in a closure. Dynamic inputs require
    float64/complex128, and output/gradient shapes must remain unchanged.

    One random direction per parameter and random output cotangents are generated
    reproducibly from ``seed`` unless supplied. ``directions`` is always a tuple;
    ``cotangents`` must have the same single-array/tuple structure as the output.
    Each parameter is perturbed separately by ``+/- step * direction``. Repeat
    with different seeds for additional coverage; tune step/tolerances for scale
    and conditioning, and keep both perturbations inside the physical domain.

    Calls a fresh record and consumes its pullback exactly once. Further forward
    records supply the finite differences without consuming their contexts.
    Returns None on success; raises AssertionError with the parameter index and
    directional error on disagreement, or ValueError for an invalid contract.
    """
    if not parameters:
        raise ValueError("check_pullback requires at least one dynamic parameter")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    if not all(np.isfinite(x) and x >= 0 for x in (rtol, atol)):
        raise ValueError("rtol and atol must be finite and nonnegative")
    primals = tuple(input_array(value) for value in parameters)
    for i, value in enumerate(primals):
        if not value.size or not np.all(np.isfinite(value)):
            raise ValueError(f"parameter {i}: require nonempty, finite values")
    if directions is not None and (
        not isinstance(directions, tuple) or len(directions) != len(primals)
    ):
        raise ValueError("directions must be a tuple with one item per parameter")
    rng = np.random.default_rng(seed)
    probes = tuple(
        _probe(
            None if directions is None else directions[i], value, rng, f"direction {i}"
        )
        for i, value in enumerate(primals)
    )
    if any(not np.any(direction) for direction in probes):
        raise ValueError("each direction must be nonzero")
    outputs, pullback, multiple = execute(record, primals)
    if cotangents is not None and isinstance(cotangents, tuple) != multiple:
        raise ValueError(
            "cotangents must match the output's single-array/tuple structure"
        )
    supplied = cotangents if isinstance(cotangents, tuple) else (cotangents,)
    if cotangents is not None and len(supplied) != len(outputs):
        raise ValueError("cotangents must contain one item per output")
    weights = tuple(
        _probe(
            None if cotangents is None else supplied[i], value, rng, f"cotangent {i}"
        )
        for i, value in enumerate(outputs)
    )
    if not any(np.any(weight) for weight in weights):
        raise ValueError("cotangents must not all be zero")
    for i, output in enumerate(outputs):
        _shaped(output, output, f"output {i}")
    result = pullback(*weights)
    raw_gradients = result if isinstance(result, tuple) else (result,)
    if len(raw_gradients) != len(primals):
        raise ValueError(
            f"pullback returned {len(raw_gradients)} gradients for {len(primals)} parameters"
        )
    gradients = tuple(
        _shaped(gradient, value, f"gradient for parameter {i}", gradient=True)
        for i, (gradient, value) in enumerate(zip(raw_gradients, primals, strict=True))
    )
    for i, (gradient, direction) in enumerate(zip(gradients, probes, strict=True)):
        shifted_outputs = []
        for sign in (1, -1):
            shifted = list(primals)
            shifted[i] = np.asarray(
                primals[i] + sign * step * direction, dtype=primals[i].dtype
            )
            values, _, is_multiple = execute(record, tuple(shifted))
            if is_multiple != multiple or len(values) != len(outputs):
                raise ValueError(f"parameter {i}: perturbed output structure changed")
            shifted_outputs.append(
                tuple(
                    _shaped(value, reference, f"parameter {i}, perturbed output {j}")
                    for j, (value, reference) in enumerate(
                        zip(values, outputs, strict=True)
                    )
                )
            )
        numeric = sum(
            float(np.vdot(weight, (plus - minus) / (2 * step)).real)
            for weight, plus, minus in zip(weights, *shifted_outputs, strict=True)
        )
        analytic = float(np.vdot(gradient, direction).real)
        error = abs(analytic - numeric)
        tolerance = atol + rtol * abs(numeric)
        if not np.isfinite(analytic) or not np.isfinite(numeric) or error > tolerance:
            raise AssertionError(
                f"parameter {i} shape {primals[i].shape}: pullback={analytic:.10g}, "
                f"finite difference={numeric:.10g}, absolute error={error:.3g} "
                f"> tolerance={tolerance:.3g} (step={step:g}, seed={seed}). "
                "Check gradient order/shapes and the Re(vdot) complex convention; "
                "check step size and conditioning before changing tolerances."
            )


def check_gradient(
    function: Callable[..., ArrayLike],
    gradient: Callable[..., ArrayLike | tuple[ArrayLike, ...]],
    *parameters: ArrayLike,
    directions: tuple[ArrayLike, ...] | None = None,
    step: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    seed: int = 0,
) -> None:
    """Check a real scalar objective and its first-order gradient callable.

    ``gradient(*parameters)`` returns an array/scalar for one input or a tuple
    with one gradient per input. Complex gradients use ``Re(vdot(g, dx))``.
    Other options and failure diagnostics are those of :func:`check_pullback`.
    The gradient is evaluated once; the objective is evaluated 1 + 2*n times.
    """

    def record(*values: Array) -> tuple[Array, Callable[..., object]]:
        output = np.asarray(function(*values))
        if output.shape != () or np.iscomplexobj(output):
            raise ValueError("check_gradient requires a real scalar objective")

        def pullback(_: Array) -> ArrayLike | tuple[ArrayLike, ...]:
            return gradient(*values)

        return output, pullback

    check_pullback(
        record,
        *parameters,
        directions=directions,
        cotangents=np.asarray(1.0),
        step=step,
        rtol=rtol,
        atol=atol,
        seed=seed,
    )

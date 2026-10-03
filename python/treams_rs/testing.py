"""Finite-difference checks of first-order gradients and pullbacks.

A pullback maps the gradient of a real loss with respect to an output to the
gradients with respect to the inputs; ``treams_rs.diff`` defines records,
contexts and pullbacks. Complex gradients follow dL = Re(vdot(g, dx)). These
checks are test tools: treams-rs never computes a gradient by finite
differences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._records import apply_pullback as _apply_pullback
from ._records import input_array as _input_array
from ._records import run_record as _run_record

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

    from ._records import Array, Record

__all__ = ["check_gradient", "check_pullback"]


def _shaped(value: ArrayLike, reference: Array, label: str) -> Array:
    array = np.asarray(value)
    if array.shape != reference.shape:
        raise ValueError(
            f"{label}: expected shape {reference.shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: values must be finite")
    if not np.iscomplexobj(reference) and np.any(array.imag != 0):
        raise ValueError(f"{label}: a real value requires a real probe/output")
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
    """Check the pullback of a record against central finite differences.

    The check records once and calls the pullback once with the cotangents.
    For each parameter it compares Re(vdot(gradient, direction)) with the
    central difference of Re(vdot(cotangents, value)) along the direction;
    the other parameters stay fixed. The shifted forward calls never use
    their contexts.

    Args:
        record: function of the dynamic inputs that returns
            ``(value, context)``. The context has ``pullback(*cotangents)`` or
            is the pullback itself; the pullback returns one gradient per
            parameter, a tuple for several. The value is an array, a scalar or
            a tuple of them. Bind labels and options with a closure.
        *parameters: the dynamic inputs: nonempty, finite, float64 or
            complex128.
        directions: tuple with one direction per parameter, each nonzero.
            Default: random directions drawn from ``seed``.
        cotangents: gradient of the loss with respect to the value, one array
            or a tuple like the value. Default: random arrays drawn from
            ``seed``.
        step: finite-difference step; each parameter moves by
            ``+/- step * direction``. Keep both points inside the physical
            domain and scale the step to the problem.
        rtol: relative tolerance on the finite difference.
        atol: absolute tolerance; a parameter passes when
            ``|pullback - finite difference| <= atol + rtol * |finite
            difference|``.
        seed: seed of the random directions and cotangents. Repeat with other
            seeds for more coverage.

    Returns:
        None when every parameter passes.

    Raises:
        AssertionError: a pullback disagrees; the message names the
            parameter, both values and the error.
        ValueError: the record or the arguments break these rules, for example
            a wrong number of gradients, a wrong shape or non-finite values.
    """
    if not parameters:
        raise ValueError("check_pullback requires at least one dynamic parameter")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    if not all(np.isfinite(x) and x >= 0 for x in (rtol, atol)):
        raise ValueError("rtol and atol must be finite and nonnegative")
    primals = tuple(_input_array(value) for value in parameters)
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
    outputs, pullback, multiple = _run_record(record, primals)
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
    # The adapters' rules: arity, shapes and the real projection for real
    # parameters (whose imaginary gradient part does not enter the pairing).
    # Finiteness is checked before that projection, so a non-finite imaginary
    # part for a real parameter still exposes a pullback bug.
    result = pullback(*weights)

    def returned(*_: Array) -> object:
        return result

    gradients = _apply_pullback(returned, weights, primals, conjugate=False)
    for i, raw in enumerate(result if isinstance(result, tuple) else (result,)):
        if not np.all(np.isfinite(np.asarray(raw))):
            raise ValueError(f"gradient for parameter {i}: values must be finite")
    for i, (gradient, direction) in enumerate(zip(gradients, probes, strict=True)):
        shifted_outputs = []
        for sign in (1, -1):
            shifted = list(primals)
            shifted[i] = np.asarray(
                primals[i] + sign * step * direction, dtype=primals[i].dtype
            )
            values, _, is_multiple = _run_record(record, tuple(shifted))
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
    """Check the gradient of a real scalar objective against finite differences.

    The check calls ``gradient`` once and ``function`` 1 + 2n times for n
    parameters. It runs ``check_pullback`` on the record
    ``(function(*p), lambda g: gradient(*p))`` with the cotangent 1.0.

    Args:
        function: objective of the parameters; returns a real scalar.
        gradient: function of the parameters that returns the gradient, an
            array for one parameter or a tuple with one gradient per
            parameter. Complex gradients follow dL = Re(vdot(g, dx)).
        *parameters: as in ``check_pullback``.
        directions: as in ``check_pullback``.
        step: as in ``check_pullback``.
        rtol: as in ``check_pullback``.
        atol: as in ``check_pullback``.
        seed: as in ``check_pullback``.

    Returns:
        None when every parameter passes.

    Raises:
        AssertionError: as in ``check_pullback``.
        ValueError: as in ``check_pullback``, and for a complex or non-scalar
            objective.
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

"""First-order JAX bridges to native CPU operations, including jit and vmap.

Install ``treams-rs[jax]`` and enable ``jax_enable_x64``. Use the CPU backend;
host callbacks execute the Rust CPU implementation.
Backward recomputes one native forward to obtain its single-use pullback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from . import diff
from ._adapters import execute, gradients, input_array, require_dtype

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import ArrayLike

    from ._adapters import Array, Record

type Output = jax.Array | tuple[jax.Array, ...]


def _inputs(values: tuple[ArrayLike, ...]) -> tuple[jax.Array, ...]:
    if not bool(jax.config.read("jax_enable_x64")):
        raise ValueError("treams-rs JAX adapters require jax_enable_x64=True")
    if jax.default_backend() != "cpu":
        raise ValueError("treams-rs JAX adapters require the CPU backend")
    arrays = tuple(jnp.asarray(value) for value in values)
    for array in arrays:
        require_dtype(np.dtype(array.dtype))
        if not isinstance(array, jax.core.Tracer) and any(
            device.platform != "cpu" for device in array.devices()
        ):
            raise ValueError("treams-rs JAX adapters require CPU arrays")
    return arrays


def _operation(
    record: Record, specs: tuple[jax.ShapeDtypeStruct, ...], multiple: bool
) -> Callable[..., Output]:
    def forward_callback(*values: Array) -> tuple[Array, ...]:
        outputs, _, is_multiple = execute(record, tuple(np.asarray(v) for v in values))
        if is_multiple != multiple or len(outputs) != len(specs):
            raise ValueError("native operation changed its output structure")
        return outputs

    @jax.custom_vjp
    def primitive(*values: jax.Array) -> tuple[jax.Array, ...]:
        return jax.pure_callback(
            forward_callback, specs, *values, vmap_method="sequential"
        )

    def forward_rule(*values: jax.Array) -> tuple[tuple[jax.Array, ...], Any]:
        return primitive(*values), values

    def backward_rule(
        primals: tuple[jax.Array, ...], cotangents: tuple[jax.Array, ...]
    ) -> tuple[jax.Array, ...]:
        input_specs = tuple(jax.ShapeDtypeStruct(v.shape, v.dtype) for v in primals)

        def callback(*packed: Array) -> tuple[Array, ...]:
            values = tuple(np.asarray(v) for v in packed[: len(primals)])
            _, pullback, _ = execute(record, values)
            # JAX uses the bilinear complex convention; Rust uses Re(vdot(g, dx)).
            return gradients(
                pullback,
                tuple(np.asarray(v) for v in packed[len(primals) :]),
                values,
                conjugate=True,
            )

        return jax.pure_callback(
            callback, input_specs, *primals, *cotangents, vmap_method="sequential"
        )

    primitive.defvjp(forward_rule, backward_rule)

    def call(*values: ArrayLike) -> Output:
        outputs = primitive(*_inputs(values))
        return outputs if multiple else outputs[0]

    return call


def wrap(record: Record, *example_values: ArrayLike) -> Callable[..., Output]:
    """Prepare a native ``(outputs, context_or_pullback)`` function for JAX.

    Examples determine fixed input/output shapes and output dtypes, by executing
    one native forward. Capture static labels/options in ``record``. Its pullback
    must return gradients in exactly the dynamic parameter order. Inputs/outputs
    are arrays or scalars; multiple outputs are a flat tuple. See docs/adapters.md.
    """
    _inputs(example_values)
    examples = tuple(input_array(value) for value in example_values)
    outputs, _, multiple = execute(record, examples)
    operation = _operation(
        record, tuple(jax.ShapeDtypeStruct(v.shape, v.dtype) for v in outputs), multiple
    )
    signatures = tuple((v.shape, v.dtype) for v in examples)

    def call(*values: ArrayLike) -> Output:
        arrays = _inputs(values)
        if tuple((v.shape, v.dtype) for v in arrays) != signatures:
            raise ValueError("parameters must match the shapes and dtypes used by wrap")
        return operation(*arrays)

    return call


def _call(record: Record, shape: tuple[int, ...], *values: ArrayLike) -> jax.Array:
    return cast(
        "jax.Array",
        _operation(record, (jax.ShapeDtypeStruct(shape, np.complex128),), False)(
            *values
        ),
    )


def solve(operator: ArrayLike, rhs: ArrayLike) -> jax.Array:
    """Solve A X = B using native LU and its analytic first-order VJP."""
    return _call(diff.solve, np.shape(rhs), operator, rhs)


def interaction(local: ArrayLike, coupling: ArrayLike) -> jax.Array:
    """Solve (I - T C) X = T with native matrix pullbacks."""
    return _call(diff.interaction, np.shape(local), local, coupling)


def illuminate(local: ArrayLike, coupling: ArrayLike, incident: ArrayLike) -> jax.Array:
    """Solve only requested incident columns, with native T/C/incident adjoints."""
    return _call(diff.illuminate, np.shape(incident), local, coupling, incident)


def sphere(
    lmax: int,
    k0: ArrayLike,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike | None = None,
    kappa: ArrayLike | None = None,
) -> jax.Array:
    """Multilayer chiral sphere with native material, radius and frequency VJPs."""

    def record(k: Array, r: Array, e: Array, m: Array, c: Array) -> tuple[Any, Any]:
        return diff.sphere(lmax, float(k), r, e, m, c)

    size = 2 * lmax * (lmax + 2)
    return _call(
        record,
        (size, size),
        k0,
        radii,
        epsilon,
        np.ones(np.shape(epsilon)) if mu is None else mu,
        np.zeros(np.shape(epsilon)) if kappa is None else kappa,
    )


def bessel(
    z: ArrayLike,
    *,
    order: np.typing.ArrayLike,
    kind: str = "j",
    spherical: bool = False,
    derivative: bool = False,
) -> jax.Array:
    """Broadcast Bessel values with the order fixed and a native argument VJP."""

    def record(value: Array) -> tuple[Any, Any]:
        return diff.bessel(
            order, value, kind=kind, spherical=spherical, derivative=derivative
        )

    return _call(
        record,
        np.broadcast_shapes(np.shape(z), np.shape(order)),
        z,
    )

"""JAX adapter: physics objects and records on the CPU with first-order forward
and reverse mode, jit and sequential vmap. Records with explicit array state
retain their native residual for reverse mode, ``jacfwd`` and repeated
``linearize`` directions. JAX owns the saved arrays; ``checkpoint`` can choose
to recompute them. Opaque custom records retain their inputs and rerun the
record for each pullback or tangent application, including a direct JVP.

Inputs may be float32, float64, complex64 or complex128. Native work uses
double precision; outputs are float32/complex64 with JAX's default
configuration, or float64/complex128 with ``jax_enable_x64`` enabled. Input
gradients retain their input dtype. This adapter does not change JAX settings.

Differentiate a scattering cross section::

    import jax
    import treams_rs as tr

    def objective(radius):
        sphere = tr.sphere_tmatrix(k0=2.0, lmax=2, radius=radius, material=3.0)
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=2.0)
        return sphere.cross_sections(wave).scattering

    value, gradient = jax.jit(jax.value_and_grad(objective))(0.2)

A pushforward maps input tangents to output tangents. A pullback maps output
cotangents to input cotangents; ``treams_rs.diff`` defines records and their
derivative contexts. Install ``treams-rs[jax]`` and keep the CPU backend: host callbacks
run the Rust code, and ``vmap`` calls them one after another. Run another
record with ``wrap``.

Framework adapters guide: https://yaugenst.github.io/treams-rs/latest/differentiation/frameworks/
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir

# Physical objects share one implementation; this namespace selects execution.
from . import _framework, _framework_backend
from ._bases import CylindricalBasis, PlaneWaveBasis, PlaneWavePorts, SphericalBasis
from ._framework_backend import Material
from ._framework_smatrix import SMatrix, stack
from ._framework_tmatrix import Cluster, PeriodicResponse, TMatrix, solve_periodic
from ._framework_waves import PlaneWave, PortWave, Wave
from ._lattice import Lattice
from ._records import (
    apply_pullback,
    apply_pushforward,
    native_array,
    record_outputs,
    require_inexact,
    run_record,
)
from ._results import BandModes, CrossSections, PowerBalance, ScatteredPorts
from ._saved import ArraySpec, SavedRecord, pack_state, saved_record, unpack_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import ArrayLike

    from ._records import Array, ArrayMetadata, Record

type Output = jax.Array | tuple[jax.Array, ...]


def _inputs(values: tuple[ArrayLike, ...]) -> tuple[jax.Array, ...]:
    if jax.default_backend() != "cpu":
        raise ValueError("treams-rs JAX adapters require the CPU backend")
    arrays = tuple(jnp.asarray(value) for value in values)
    for array in arrays:
        require_inexact(np.dtype(array.dtype))
        if not isinstance(array, jax.core.Tracer) and any(
            device.platform != "cpu" for device in array.devices()
        ):
            raise ValueError("treams-rs JAX adapters require CPU arrays")
    return arrays


class _NativeCall:
    """Keep the record's numeric residual in JAX-owned arrays between calls."""

    def __init__(
        self,
        record: SavedRecord,
        specs: tuple[jax.ShapeDtypeStruct, ...],
        multiple: bool,
        inputs: tuple[jax.Array, ...],
    ) -> None:
        self.record = record
        self.specs = specs
        self.multiple = multiple
        self.input_specs = tuple(
            ArraySpec(value.shape, np.dtype(value.dtype)) for value in inputs
        )
        self.retained_count = len(inputs) if record.needs_primals else 0
        self.state_specs = record.state_spec(
            tuple(
                ArraySpec(
                    value.shape,
                    np.dtype(np.complex128 if np.iscomplexobj(value) else np.float64),
                )
                for value in inputs
            )
        )
        self.buffer_specs = tuple(
            jax.ShapeDtypeStruct((spec.nbytes,), np.uint8) for spec in self.state_specs
        )
        self.native_outputs = tuple(
            ArraySpec(
                spec.shape,
                np.dtype(
                    np.complex128
                    if np.issubdtype(spec.dtype, np.complexfloating)
                    else np.float64
                ),
            )
            for spec in specs
        )

    def _outputs(self, values: tuple[Array, ...], multiple: bool) -> tuple[Array, ...]:
        if multiple != self.multiple or len(values) != len(self.specs):
            raise ValueError("native operation changed its output structure")
        return tuple(
            np.asarray(value, dtype=spec.dtype)
            for value, spec in zip(values, self.specs, strict=True)
        )

    def forward(self, *values: Array) -> tuple[Array, ...]:
        outputs, _, multiple = run_record(
            self.record, tuple(native_array(value) for value in values)
        )
        return self._outputs(outputs, multiple)

    def record_and_save(self, *values: Array) -> tuple[Any, ...]:
        outputs, context, multiple = record_outputs(
            self.record, tuple(native_array(value) for value in values)
        )
        state = pack_state(self.record.save(context), self.state_specs)
        return (*self._outputs(outputs, multiple), *state)

    def _restore(
        self, packed: tuple[Any, ...]
    ) -> tuple[Any, tuple[ArrayMetadata, ...], int]:
        primals = packed[: self.retained_count]
        boundary = self.retained_count + len(self.state_specs)
        state = unpack_state(packed[self.retained_count : boundary], self.state_specs)
        context = self.record.restore(
            state, *(native_array(value) for value in primals)
        )
        return context, self.input_specs, boundary

    def tangent(self, *packed: Array) -> tuple[Array, ...]:
        context, primals, boundary = self._restore(packed)
        tangents = apply_pushforward(
            context, packed[boundary:], primals, self.native_outputs
        )
        return self._outputs(tangents, self.multiple)

    def pullback(self, *packed: Array) -> tuple[Array, ...]:
        context, primals, boundary = self._restore(packed)
        return apply_pullback(
            context if callable(context) else context.pullback,
            tuple(native_array(value) for value in packed[boundary:]),
            primals,
            conjugate=True,
        )


def _callback(*values: jax.Array, operation: _NativeCall) -> tuple[jax.Array, ...]:
    return jax.pure_callback(
        operation.tangent, operation.specs, *values, vmap_method="sequential"
    )


def _batch(
    values: tuple[jax.Array, ...],
    dimensions: tuple[int | None, ...],
    *,
    operation: _NativeCall,
) -> tuple[Any, tuple[int | None, ...]]:
    batched = tuple(
        jnp.moveaxis(value, dimension, 0)
        for value, dimension in zip(values, dimensions, strict=True)
        if dimension is not None
    )

    def mapped(items: tuple[jax.Array, ...]) -> Any:
        iterator = iter(items)
        arguments = tuple(
            value if dimension is None else next(iterator)
            for value, dimension in zip(values, dimensions, strict=True)
        )
        return _tangent.bind(*arguments, operation=operation)

    outputs = jax.lax.map(mapped, batched)
    return outputs, (0,) * len(outputs)


def _abstract(*_: Any, operation: _NativeCall) -> tuple[Any, ...]:
    return tuple(
        jax.core.ShapedArray(spec.shape, spec.dtype) for spec in operation.specs
    )


# Only this linear boundary needs a primitive: its transpose applies the native
# pullback to the same array-owned residual used by the pushforward.
_tangent = core.Primitive("treams_tangent")
_tangent.multiple_results = True
_tangent.def_impl(_callback)
_tangent.def_abstract_eval(_abstract)
mlir.register_lowering(_tangent, mlir.lower_fun(_callback, multiple_results=True))
batching.primitive_batchers[_tangent] = _batch


def _higher_order(*_: Any, **__: Any) -> Any:
    raise ValueError("treams-rs supports first-order differentiation only")


def _tangent_jvp(
    primals: tuple[Any, ...], tangents: tuple[Any, ...], *, operation: _NativeCall
) -> tuple[Any, Any]:
    boundary = operation.retained_count + len(operation.state_specs)
    # The recorded point stays fixed when differentiating a linearized map
    # with respect to its direction. Differentiating the state needs a Hessian.
    if any(not isinstance(value, ad.Zero) for value in tangents[:boundary]):
        _higher_order()
    outputs = _tangent.bind(*primals, operation=operation)
    output_tangents = _tangent.bind(
        *primals[:boundary],
        *(ad.instantiate_zeros(value) for value in tangents[boundary:]),
        operation=operation,
    )
    return outputs, output_tangents


ad.primitive_jvps[_tangent] = _tangent_jvp


def _saved_transpose(
    cotangents: tuple[Any, ...], *values: Any, operation: _NativeCall
) -> tuple[Any, ...]:
    count = operation.retained_count
    boundary = count + len(operation.state_specs)
    gradients = jax.pure_callback(
        operation.pullback,
        tuple(
            jax.ShapeDtypeStruct(value.shape, value.dtype)
            for value in operation.input_specs
        ),
        *values[:boundary],
        *(ad.instantiate_zeros(value) for value in cotangents),
        vmap_method="sequential",
    )
    return (None,) * boundary + tuple(gradients)


ad.primitive_transposes[_tangent] = _saved_transpose


def _primitive(
    record: Record,
    specs: tuple[jax.ShapeDtypeStruct, ...],
    multiple: bool,
    inputs: tuple[jax.Array, ...],
) -> Callable[..., Output]:
    """One custom JVP backed by native pushforward and pullback callbacks."""
    prepared = saved_record(record)
    if prepared is None:
        # Arbitrary user records have no array-state contract. Replaying the
        # record preserves reverse compatibility and enables a basic JVP.
        def state_spec(_inputs: tuple[ArraySpec, ...]) -> tuple[ArraySpec, ...]:
            return ()

        def save(_context: Any) -> tuple[Any, ...]:
            return ()

        def restore(_state: tuple[Any, ...], *primals: Any) -> Any:
            return record(*primals)[1]

        prepared = SavedRecord(record, state_spec, save, restore)

    # JAX caches callbacks by identity. Keep one operation per wrapper so eager
    # calls reuse their compiled callbacks as well as jitted calls do.
    operation = _NativeCall(prepared, specs, multiple, inputs)

    @jax.custom_jvp
    def primitive(*values: jax.Array) -> tuple[jax.Array, ...]:
        return jax.pure_callback(
            operation.forward, specs, *values, vmap_method="sequential"
        )

    @jax.custom_jvp
    def record_and_save(*values: jax.Array) -> tuple[Any, ...]:
        return jax.pure_callback(
            operation.record_and_save,
            (*specs, *operation.buffer_specs),
            *values,
            vmap_method="sequential",
        )

    record_and_save.defjvp(_higher_order)

    @primitive.defjvp
    def jvp_rule(
        primals: tuple[jax.Array, ...], tangents: tuple[jax.Array, ...]
    ) -> Any:
        recorded = record_and_save(*primals)
        outputs, state = recorded[: len(specs)], recorded[len(specs) :]
        output_tangents = _tangent.bind(
            *primals[: operation.retained_count],
            *state,
            *tangents,
            operation=operation,
        )
        return outputs, tuple(output_tangents)

    def call(*values: jax.Array) -> Output:
        outputs = primitive(*values)
        return outputs if multiple else outputs[0]

    return call


def wrap(record: Record, *example_values: ArrayLike) -> Callable[..., Output]:
    """Turn a record into a JAX function with first-order forward and reverse AD.

    ``wrap`` runs ``record(*example_values)`` once to fix the shapes and dtypes
    of the inputs and outputs; later calls must use the same input shapes and
    dtypes. For example, ``wrap(diff.eig, matrix)`` differentiates an
    eigensystem. Custom records:
    https://yaugenst.github.io/treams-rs/latest/differentiation/custom-records/

    Args:
        record: function of the dynamic inputs that returns ``(value, context)``
            or ``(value, pullback)``. The value is an array, a scalar or a flat
            tuple of them. The pullback returns one gradient per dynamic input,
            in argument order. Bind labels and options with a closure or
            ``functools.partial``. The record must be deterministic: JAX may
            skip or repeat its calls.
        example_values: valid float32/64 or complex64/128 inputs.

    Returns:
        A function of the dynamic inputs that returns JAX arrays, a tuple for
        several outputs. It works under ``jax.jit``, ``jax.grad`` and
        ``jax.vmap``, ``jax.jvp`` and ``jax.jacfwd``. Forward mode requires a
        context with a ``pushforward`` method.
    """
    arrays = _inputs(example_values)
    examples = tuple(native_array(value) for value in arrays)
    outputs, _, multiple = run_record(record, examples)
    operation = _primitive(
        record,
        tuple(
            jax.ShapeDtypeStruct(v.shape, jax.dtypes.canonicalize_dtype(v.dtype))
            for v in outputs
        ),
        multiple,
        arrays,
    )
    signatures = tuple((v.shape, v.dtype) for v in arrays)

    def call(*values: ArrayLike) -> Output:
        arrays = _inputs(values)
        if tuple((v.shape, v.dtype) for v in arrays) != signatures:
            raise ValueError("parameters must match the shapes and dtypes used by wrap")
        return operation(*arrays)

    return call


def _operation(
    record: Record, *values: ArrayLike, shape: tuple[int, ...], real: bool = False
) -> jax.Array:
    spec = jax.ShapeDtypeStruct(
        shape, jax.dtypes.canonicalize_dtype(np.float64 if real else np.complex128)
    )
    arrays = _inputs(values)
    return cast("jax.Array", _primitive(record, (spec,), False, arrays)(*arrays))


def _physics_validate(value: Any) -> None:
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        _inputs((value,))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _physics_validate(item)


def _physics_array(value: Any, *, dtype: Any) -> jax.Array:
    # Canonicalize explicitly so default JAX precision produces no truncation warning.
    return jnp.asarray(
        value, dtype=None if dtype is None else jax.dtypes.canonicalize_dtype(dtype)
    )


_backend = _framework_backend.Backend(
    jnp, _operation, asarray=_physics_array, validate=_physics_validate
)


# One shared implementation of the physical constructors, bound to this backend.
# _api and _ops keep their names: their bound methods pickle as references to them.
_api = _framework.Constructors(_backend, __name__)
plane_wave = _api.plane_wave
smatrix = _api.smatrix
wave = _api.wave
tmatrix = _api.tmatrix
sphere_tmatrix = _api.sphere_tmatrix
multilayer_sphere_tmatrix = _api.multilayer_sphere_tmatrix
cylinder_tmatrix = _api.cylinder_tmatrix
multilayer_cylinder_tmatrix = _api.multilayer_cylinder_tmatrix
slab = _api.slab
interface = _api.interface
multilayer_slab = _api.multilayer_slab
propagation = _api.propagation

# The expert operations shared with the other adapters.
_ops = _framework.Operations(_backend, __name__)
solve = _ops.solve
interaction = _ops.interaction
illuminate = _ops.illuminate
sphere = _ops.sphere
bessel = _ops.bessel


__all__ = [
    "BandModes",
    "Cluster",
    "CrossSections",
    "CylindricalBasis",
    "Lattice",
    "Material",
    "PeriodicResponse",
    "PlaneWave",
    "PlaneWaveBasis",
    "PlaneWavePorts",
    "PortWave",
    "PowerBalance",
    "SMatrix",
    "ScatteredPorts",
    "SphericalBasis",
    "TMatrix",
    "Wave",
    "bessel",
    "cylinder_tmatrix",
    "illuminate",
    "interaction",
    "interface",
    "multilayer_cylinder_tmatrix",
    "multilayer_slab",
    "multilayer_sphere_tmatrix",
    "plane_wave",
    "propagation",
    "slab",
    "smatrix",
    "solve",
    "solve_periodic",
    "sphere",
    "sphere_tmatrix",
    "stack",
    "tmatrix",
    "wave",
    "wrap",
]

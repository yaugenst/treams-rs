"""JAX adapter: physics objects and records as JAX operations on the CPU, in
first-order reverse mode only, with jit and sequential vmap. JAX stores only
the inputs; each gradient pass reruns the Rust forward and uses its pullback
once. Inputs must be float64 or complex128 (TypeError otherwise), so enable
``jax_enable_x64``; outputs are float64 or complex128.

Differentiate a scattering cross section::

    import jax
    import treams_rs.jax as tr

    jax.config.update("jax_enable_x64", True)

    def objective(radius):
        sphere = tr.sphere_tmatrix(k0=2.0, lmax=2, radius=radius, material=3.0)
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=2.0)
        return sphere.cross_sections(wave).scattering

    value, gradient = jax.jit(jax.value_and_grad(objective))(0.2)

A pullback maps the gradient with respect to an output to the gradients with
respect to the inputs; ``treams_rs.diff`` defines records, contexts and
pullbacks. Install ``treams-rs[jax]`` and keep the CPU backend: host callbacks
run the Rust code, and ``vmap`` calls them one after another. Run another
record with ``wrap``.

Framework adapters guide: https://yaugenst.github.io/treams-rs/latest/differentiation/frameworks/
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np

# Physical objects share one implementation; this namespace selects execution.
from . import _framework, _framework_backend
from ._bases import CylindricalBasis, PlaneWaveBasis, PlaneWavePorts, SphericalBasis
from ._framework_backend import Material
from ._framework_smatrix import SMatrix, stack
from ._framework_tmatrix import Cluster, PeriodicResponse, TMatrix, solve_periodic
from ._framework_waves import PlaneWave, PortWave, Wave
from ._lattice import Lattice
from ._records import apply_pullback, input_array, require_float64, run_record
from ._results import BandModes, CrossSections, PowerBalance, ScatteredPorts

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import ArrayLike

    from ._records import Array, Record

type Output = jax.Array | tuple[jax.Array, ...]


def _require_x64() -> None:
    if not bool(jax.config.read("jax_enable_x64")):
        raise ValueError("treams-rs JAX adapters require jax_enable_x64=True")


def _inputs(values: tuple[ArrayLike, ...]) -> tuple[jax.Array, ...]:
    _require_x64()
    if jax.default_backend() != "cpu":
        raise ValueError("treams-rs JAX adapters require the CPU backend")
    arrays = tuple(jnp.asarray(value) for value in values)
    for array in arrays:
        require_float64(np.dtype(array.dtype))
        if not isinstance(array, jax.core.Tracer) and any(
            device.platform != "cpu" for device in array.devices()
        ):
            raise ValueError("treams-rs JAX adapters require CPU arrays")
    return arrays


def _primitive(
    record: Record, specs: tuple[jax.ShapeDtypeStruct, ...], multiple: bool
) -> Callable[..., Output]:
    """The ``jax.custom_vjp`` function behind wrap and _operation, for fixed specs."""

    def forward_callback(*values: Array) -> tuple[Array, ...]:
        outputs, _, is_multiple = run_record(
            record, tuple(np.asarray(v) for v in values)
        )
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
            _, pullback, _ = run_record(record, values)
            # JAX uses the bilinear complex convention; Rust uses Re(vdot(g, dx)).
            return apply_pullback(
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
    """Turn a record into a JAX function with a first-order gradient.

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
        example_values: valid float64 or complex128 inputs.

    Returns:
        A function of the dynamic inputs that returns JAX arrays, a tuple for
        several outputs. It works under ``jax.jit``, ``jax.grad`` and
        ``jax.vmap``.
    """
    _inputs(example_values)
    examples = tuple(input_array(value) for value in example_values)
    outputs, _, multiple = run_record(record, examples)
    operation = _primitive(
        record, tuple(jax.ShapeDtypeStruct(v.shape, v.dtype) for v in outputs), multiple
    )
    signatures = tuple((v.shape, v.dtype) for v in examples)

    def call(*values: ArrayLike) -> Output:
        arrays = _inputs(values)
        if tuple((v.shape, v.dtype) for v in arrays) != signatures:
            raise ValueError("parameters must match the shapes and dtypes used by wrap")
        return operation(*arrays)

    return call


def _operation(
    record: Record, *values: ArrayLike, shape: tuple[int, ...], real: bool = False
) -> jax.Array:
    spec = jax.ShapeDtypeStruct(shape, np.float64 if real else np.complex128)
    return cast("jax.Array", _primitive(record, (spec,), False)(*values))


def _physics_validate(value: Any) -> None:
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        _inputs((value,))
    else:
        # Without x64, jnp.asarray would truncate Python and NumPy values to 32 bits.
        _require_x64()


_backend = _framework_backend.Backend(jnp, _operation, validate=_physics_validate)


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

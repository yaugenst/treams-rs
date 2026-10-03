"""HIPS Autograd adapter: physics objects and native records on the CPU.

First-order reverse differentiation supports float64 and complex128 inputs
and outputs. The first reverse pass consumes the Rust context saved by the
forward; repeated calls to a pullback rerun the forward from input snapshots.
Forward mode and higher derivatives are not available.

Differentiate a scattering cross section::

    import autograd
    import treams_rs as tr

    def objective(radius):
        sphere = tr.sphere_tmatrix(k0=2.0, lmax=2, radius=radius, material=3.0)
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=2.0)
        return sphere.cross_sections(wave).scattering

    value, gradient = autograd.value_and_grad(objective)(0.2)

Install ``treams-rs[autograd]``. Use ``autograd.numpy`` for array operations
inside objectives. ``wrap`` adapts another native record; ``treams_rs.diff``
defines records, contexts and pullbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import autograd.numpy as anp
import numpy as np
from autograd.builtins import isinstance as _isinstance
from autograd.extend import defvjp_argnums, primitive
from autograd.tracer import isbox

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
    from collections.abc import Callable, Sequence

    from ._records import Array, Pullback, Record

_FIRST_ORDER = "treams-rs Autograd adapters support first-order gradients only"


@dataclass
class _Invocation:
    record: Record
    primals: tuple[Array, ...] = ()
    pullback: Pullback | None = None
    multiple: bool = False


@primitive
def _execute(invocation: _Invocation, *values: Any) -> tuple[Array, ...]:
    invocation.primals = tuple(input_array(value).copy() for value in values)
    outputs, invocation.pullback, invocation.multiple = run_record(
        invocation.record, invocation.primals
    )
    return outputs


def _make_vjp(
    argnums: Sequence[int], answer: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Callable[[Any], tuple[Array, ...]]:
    if isbox(answer) or any(isbox(value) for value in args[1:]):
        raise NotImplementedError(_FIRST_ORDER)
    invocation: _Invocation = args[0]

    def backward(cotangents: Any) -> tuple[Array, ...]:
        if isbox(cotangents) or any(isbox(value) for value in cotangents):
            raise NotImplementedError(_FIRST_ORDER)
        pullback = invocation.pullback
        invocation.pullback = None
        if pullback is None:
            _, pullback, _ = run_record(invocation.record, invocation.primals)
        # Autograd uses bilinear complex cotangents; Rust uses Re(vdot(g, dx)).
        gradients = apply_pullback(
            pullback,
            tuple(np.asarray(value) for value in cotangents),
            invocation.primals,
            conjugate=True,
        )
        return tuple(gradients[index - 1] for index in argnums)

    return backward


# One pullback produces all input gradients, consuming the native context once.
# Register only this shared primitive: Autograd retains registrations globally.
defvjp_argnums(_execute, _make_vjp)


def wrap(record: Record) -> Callable[..., Any]:
    """Turn a record into an Autograd function with a first-order gradient.

    For example, ``wrap(diff.solve)`` differentiates a linear solve. A record
    returns ``(value, context)`` or ``(value, pullback)``; the value is an array,
    scalar or nonempty flat tuple of them. The pullback returns one gradient
    per dynamic input in argument order. Bind static labels and options with
    a closure or ``functools.partial``.

    Inputs must have dtype float64 or complex128. Outputs are NumPy arrays,
    or a flat tuple of arrays. Repeated pullbacks recompute the record, which
    must therefore be deterministic. Use ``autograd.numpy`` around this call.
    """

    def operation(*values: Any) -> Any:
        invocation = _Invocation(record)
        # Unlike asarray, Autograd's array constructor preserves nested boxes.
        outputs = _execute(invocation, *(anp.array(value) for value in values))
        return outputs if invocation.multiple else outputs[0]

    return operation


def _operation(
    record: Record, *values: Any, shape: tuple[int, ...], real: bool = False
) -> Any:
    # Backend.apply checks output shape and dtype against the native result.
    return wrap(record)(*values)


def _physics_array(value: Any, *, dtype: Any) -> Any:
    if _isinstance(value, (list, tuple)) and len(value):
        return anp.stack([_physics_array(item, dtype=dtype) for item in value])
    array = anp.array(value)
    if isbox(array):
        require_float64(array.dtype)
        # Autograd's cast VJP warns for complex-to-real cotangents; addition
        # supplies the same promotion with its explicit real-input projection.
        return array + 0j if dtype == np.complex128 else array
    return np.asarray(array, dtype=dtype)


_backend = _framework_backend.Backend(anp, _operation, asarray=_physics_array)


# The shared physical implementation, bound to Autograd execution.
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

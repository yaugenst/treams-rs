"""PyTorch adapter: physics objects and records as autograd operations on the
CPU, in first-order forward and reverse mode. Forward mode and the first
backward share the Rust context saved by the forward. The first backward
releases that context; repeated backward (``retain_graph=True``) reruns the
Rust forward from copies of the inputs.
Tensors must be CPU float32, float64,
complex64 or complex128 (TypeError for other dtypes, ValueError for other
devices). Native computation and outputs use float64/complex128; gradients
retain each input's dtype.

Differentiate a scattering cross section::

    import torch
    import treams_rs as tr

    radius = torch.tensor(0.2, requires_grad=True)
    sphere = tr.sphere_tmatrix(k0=2.0, lmax=2, radius=radius, material=3.0)
    wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=2.0)
    sphere.cross_sections(wave).scattering.backward()
    gradient = radius.grad

A pullback maps the gradient with respect to an output to the gradients with
respect to the inputs; ``treams_rs.diff`` defines records, contexts and
pullbacks. Install ``treams-rs[torch]``. Forward mode supports
``torch.func.jvp`` and ``torch.autograd.forward_ad``. Higher derivatives,
vectorized transforms and ``torch.compile`` are not available. Run another
record with ``wrap``.

Framework adapters guide: https://yaugenst.github.io/treams-rs/latest/differentiation/frameworks/
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import torch

# Physical objects share one implementation; this namespace selects execution.
from . import _framework, _framework_backend
from ._bases import CylindricalBasis, PlaneWaveBasis, PlaneWavePorts, SphericalBasis
from ._framework_backend import Material
from ._framework_smatrix import SMatrix, stack
from ._framework_tmatrix import Cluster, PeriodicResponse, TMatrix, solve_periodic
from ._framework_waves import PlaneWave, PortWave, Wave
from ._lattice import Lattice
from ._records import (
    _INEXACT_DTYPES,
    apply_pullback,
    apply_pushforward,
    native_array,
    record_outputs,
    require_inexact,
)
from ._results import BandModes, CrossSections, PowerBalance, ScatteredPorts

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

    from ._records import Array, Record

type Output = torch.Tensor | tuple[torch.Tensor, ...]

_TENSOR_DTYPES = tuple(getattr(torch, dtype.name) for dtype in _INEXACT_DTYPES)


def _numpy(value: torch.Tensor) -> Array:
    return value.numpy(force=True)


@dataclass
class _NativeCall:
    """Share the invocation's context, releasing it after the first backward."""

    record: Record
    context: Any
    multiple: bool

    def take_context(self, primals: tuple[Array, ...]) -> Any:
        context, self.context = self.context, None
        if context is None:
            _, context, _ = record_outputs(
                self.record, tuple(native_array(value) for value in primals)
            )
        return context


@dataclass
class _ForwardState:
    """Transport snapshots to setup_context without making them tensor outputs."""

    snapshots: tuple[torch.Tensor, ...]
    native: _NativeCall


# Native intermediates are auxiliary outputs so setup_context can save them.
# This is PyTorch's supported boundary for torch.func transforms.
class _Execute(torch.autograd.Function):
    @staticmethod
    @override
    def forward(*inputs: Any) -> tuple[Any, ...]:  # pyrefly: ignore[bad-override]
        record, *values = inputs
        # Reverse replay must own every input, including NumPy-backed constants.
        # Without reverse inputs, jvp consumes these values synchronously before
        # the operation returns; no later derivative can need a snapshot.
        snapshots = (
            tuple(v.clone() for v in values)
            if any(v.requires_grad for v in values)
            else tuple(values)
        )
        primals = tuple(native_array(_numpy(v)) for v in snapshots)
        outputs, context, multiple = record_outputs(record, primals)
        return (
            *(torch.from_numpy(np.array(v, copy=True)) for v in outputs),
            _ForwardState(snapshots, _NativeCall(record, context, multiple)),
        )

    @staticmethod
    @override
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        _, *values = inputs
        ctx.input_count = len(values)
        state = output[-1]
        ctx.native = state.native
        # PyTorch checks these tensors' version counters before any backward,
        # including repeats; caller mutation cannot change recomputed inputs.
        # Only the saved-tensor hooks retain snapshots after setup_context.
        ctx.save_for_backward(*values, *state.snapshots)
        ctx.save_for_forward(*state.snapshots, *output[:-1])

    @staticmethod
    @override
    def backward(
        ctx: Any, *cotangents: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise NotImplementedError(
                "treams-rs PyTorch adapters support first-order gradients only"
            )
        # Saved tensors validate original version counters and release snapshots
        # with the graph; NumPy aliases cannot alter the owned recomputation data.
        values = tuple(_numpy(v) for v in ctx.saved_tensors[ctx.input_count :])
        context = ctx.native.take_context(values)
        pullback = context if callable(context) else context.pullback
        result = apply_pullback(
            pullback,
            tuple(_numpy(cast("torch.Tensor", g)) for g in cotangents[:-1]),
            values,
            conjugate=False,
        )
        return (None, *(torch.from_numpy(np.array(g, copy=True)) for g in result))

    @staticmethod
    @override
    def jvp(
        ctx: Any, *tangents: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        directions = tangents[1:]  # The record is a non-tensor input.
        outputs = _Pushforward.apply(
            ctx.native, ctx.input_count, *ctx.saved_tensors, *directions
        )
        return (*outputs, None)


class _Pushforward(torch.autograd.Function):
    """Enter Rust with ordinary tensors, including from torch.func.jvp.

    Transformed tensors have no NumPy storage. A separate Function unwraps
    them through PyTorch's public dispatch and makes the first-order boundary
    explicit without private functorch APIs.
    """

    @staticmethod
    @override
    def forward(  # pyrefly: ignore[bad-override]
        native: _NativeCall, count: int, *values: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        primals = tuple(
            native_array(_numpy(cast("torch.Tensor", v))) for v in values[:count]
        )
        outputs = tuple(_numpy(cast("torch.Tensor", v)) for v in values[count:-count])
        tangents = tuple(
            np.zeros_like(primal) if tangent is None else _numpy(tangent)
            for primal, tangent in zip(primals, values[-count:], strict=True)
        )
        directions = apply_pushforward(native.context, tangents, primals, outputs)
        return tuple(torch.from_numpy(np.array(v, copy=True)) for v in directions)

    @staticmethod
    @override
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        pass

    @staticmethod
    @override
    def backward(ctx: Any, *cotangents: torch.Tensor) -> Any:
        raise NotImplementedError(
            "treams-rs PyTorch adapters support first-order gradients only"
        )

    @staticmethod
    @override
    def jvp(ctx: Any, *tangents: torch.Tensor | None) -> Any:
        raise NotImplementedError(
            "treams-rs PyTorch adapters support first-order gradients only"
        )


# PyTorch binds setup_context forwards with inspect.signature on every call.
# Cache only its declaration-derived signature; reconstructing it dominated
# adapter overhead for small native operations.
for _function in (_Execute, _Pushforward):
    cast("Any", _function.forward).__signature__ = inspect.signature(_function.forward)


def _check_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.device.type != "cpu":
        raise ValueError("treams-rs PyTorch adapters require CPU tensors")
    # Valid enums come from the shared domain contract; format a Torch-only
    # dtype (including those NumPy lacks) only when reporting an error.
    if value.dtype not in _TENSOR_DTYPES:
        require_inexact(str(value.dtype).removeprefix("torch."))
    return value


def _tensor(value: ArrayLike) -> torch.Tensor:
    # Torch warns on read-only NumPy memory (broadcast views, cached constants)
    # and rejects negative strides (reversed views); only those inputs are
    # copied here; reverse replay separately owns every input it needs.
    array = native_array(value)
    if not array.flags.writeable or any(s < 0 for s in array.strides):
        array = array.copy()
    return torch.from_numpy(array)


def wrap(record: Record) -> Callable[..., Output]:
    """Turn a record into a PyTorch function with first-order derivatives.

    For example, ``wrap(diff.solve)`` differentiates a linear solve. Custom
    records: https://yaugenst.github.io/treams-rs/latest/differentiation/custom-records/

    Args:
        record: function of the dynamic inputs that returns ``(value, context)``
            or ``(value, pullback)``. The value is an array, a scalar or a flat
            tuple of them. The pullback returns one gradient per dynamic input,
            in argument order. Forward mode additionally requires a context
            with ``pushforward(*input_tangents)``. Bind labels and options with a closure or
            ``functools.partial``.

    Returns:
        A function of tensors or array-likes that returns a tensor, or a tuple
        of tensors for several outputs.
    """

    def operation(*values: torch.Tensor | ArrayLike) -> Output:
        arrays = tuple(
            _check_tensor(value) if isinstance(value, torch.Tensor) else _tensor(value)
            for value in values
        )
        result = _Execute.apply(record, *arrays)
        outputs = cast("tuple[torch.Tensor, ...]", result[:-1])
        return outputs if result[-1].native.multiple else outputs[0]

    return operation


def _operation(
    record: Record, *values: Any, shape: tuple[int, ...], real: bool = False
) -> Any:
    # Autograd takes the output from the native forward, so shape and real are
    # unused; Backend.apply checks both.
    return wrap(record)(*values)


def _physics_array(value: Any, *, dtype: torch.dtype | None) -> torch.Tensor:
    # torch.as_tensor cannot stack tensors inside lists without detaching them.
    if isinstance(value, torch.Tensor):
        _check_tensor(value)
        return value.to(dtype=dtype) if dtype == torch.complex128 else value
    if isinstance(value, (list, tuple)) and value:
        return torch.stack([_physics_array(item, dtype=dtype) for item in value])
    # torch.tensor copies, but rejects reversed (negative-stride) NumPy views.
    if isinstance(value, np.ndarray) and any(s < 0 for s in value.strides):
        value = value.copy()
    # Python numbers stay double, as in NumPy, rather than torch's float32 default.
    return torch.tensor(np.asarray(value) if dtype is None else value, dtype=dtype)


_backend = _framework_backend.Backend(torch, _operation, asarray=_physics_array)


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

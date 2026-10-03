"""PyTorch adapter: physics objects and records as autograd operations on the
CPU, in first-order reverse mode only. The first backward uses the Rust context
saved by the forward; a repeated backward (``retain_graph=True``) reruns the
Rust forward from copies of the inputs. Tensors must be CPU float32, float64,
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
pullbacks. Install ``treams-rs[torch]``. Higher derivatives
(``create_graph=True``), forward mode, ``torch.func`` and ``torch.compile`` are
not available. Run another record with ``wrap``.

Framework adapters guide: https://yaugenst.github.io/treams-rs/latest/differentiation/frameworks/
"""

from __future__ import annotations

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
from ._records import apply_pullback, native_array, require_inexact, run_record
from ._results import BandModes, CrossSections, PowerBalance, ScatteredPorts

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

    from ._records import Array, Pullback, Record

type Output = torch.Tensor | tuple[torch.Tensor, ...]


def _numpy(value: torch.Tensor) -> Array:
    return value.detach().resolve_conj().resolve_neg().numpy()


# The autograd Function behind wrap and _operation: one native forward per call.
class _Execute(torch.autograd.Function):
    @staticmethod
    @override
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any, record: Record, *values: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        snapshots = tuple(v.detach().clone() for v in values)
        primals = tuple(native_array(_numpy(v)) for v in snapshots)
        outputs, pullback, _ = run_record(record, primals)
        ctx.record = record
        ctx.pullback = pullback
        ctx.input_count = len(values)
        # PyTorch checks these tensors' version counters before any backward,
        # including repeats; caller mutation cannot change recomputed inputs.
        ctx.save_for_backward(*values, *snapshots)
        return tuple(torch.from_numpy(np.array(v, copy=True)) for v in outputs)

    @staticmethod
    @override
    def backward(
        ctx: Any, *cotangents: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise NotImplementedError(
                "treams-rs PyTorch adapters support first-order gradients only"
            )
        # Saved tensors validate original version counters and release snapshots
        # with the graph; NumPy aliases cannot alter the owned recomputation data.
        values = tuple(_numpy(v) for v in ctx.saved_tensors[ctx.input_count :])
        pullback = ctx.pullback
        ctx.pullback = cast("Pullback | None", None)
        if pullback is None:
            _, pullback, _ = run_record(
                ctx.record, tuple(native_array(v) for v in values)
            )
        result = apply_pullback(
            pullback,
            tuple(native_array(_numpy(g)) for g in cotangents),
            values,
            conjugate=False,
        )
        return (None, *(torch.from_numpy(np.array(g, copy=True)) for g in result))


def _check_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.device.type != "cpu":
        raise ValueError("treams-rs PyTorch adapters require CPU tensors")
    # Check Torch's name before NumPy conversion, including dtypes NumPy lacks.
    require_inexact(str(value.dtype).removeprefix("torch."))
    return value


def _tensor(value: ArrayLike) -> torch.Tensor:
    # Torch warns on read-only NumPy memory (broadcast views, cached constants)
    # and rejects negative strides (reversed views); only those inputs are
    # copied, and forward snapshots its inputs anyway.
    array = native_array(value)
    if not array.flags.writeable or any(s < 0 for s in array.strides):
        array = array.copy()
    return torch.from_numpy(array)


def wrap(record: Record) -> Callable[..., Output]:
    """Turn a record into a PyTorch function with a first-order gradient.

    For example, ``wrap(diff.solve)`` differentiates a linear solve. Custom
    records: https://yaugenst.github.io/treams-rs/latest/differentiation/custom-records/

    Args:
        record: function of the dynamic inputs that returns ``(value, context)``
            or ``(value, pullback)``. The value is an array, a scalar or a flat
            tuple of them. The pullback returns one gradient per dynamic input,
            in argument order. Bind labels and options with a closure or
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
        # Structure is discovered during the same native forward, without a
        # second simulation or output signature registry.
        multiple = [False]

        def forward(*primals: Array) -> tuple[Any, Any]:
            output, context = record(*primals)
            multiple[0] = isinstance(output, tuple)
            return output, context

        outputs = cast("tuple[torch.Tensor, ...]", _Execute.apply(forward, *arrays))
        return outputs if multiple[0] else outputs[0]

    return operation


def _operation(
    record: Record, *values: Any, shape: tuple[int, ...], real: bool = False
) -> Any:
    # Autograd takes the output from the native forward, so shape and real are
    # unused; Backend.apply checks both.
    return wrap(record)(*values)


def _physics_array(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    # torch.as_tensor cannot stack tensors inside lists without detaching them.
    if isinstance(value, torch.Tensor):
        _check_tensor(value)
        return value.to(dtype=dtype) if dtype == torch.complex128 else value
    if isinstance(value, (list, tuple)) and value:
        return torch.stack([_physics_array(item, dtype=dtype) for item in value])
    # torch.tensor copies, but rejects reversed (negative-stride) NumPy views.
    if isinstance(value, np.ndarray) and any(s < 0 for s in value.strides):
        value = value.copy()
    return torch.tensor(value, dtype=dtype)


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

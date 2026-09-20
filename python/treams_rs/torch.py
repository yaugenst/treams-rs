"""First-order PyTorch bridges to native CPU operations.

Install ``treams-rs[torch]``. Dynamic tensors must be CPU float64/complex128.
The first backward uses the retained native context; repeated backward recomputes
the native forward. Higher-order AD and torch.func transforms are unsupported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import torch

# Physical objects share one implementation; this namespace selects execution.
from . import _framework as _physics
from . import diff
from ._adapters import execute, gradients, input_array
from ._core import (
    CylindricalWaveBasis,
    PlaneWaveBasisByComp,
    SphericalWaveBasis,
)
from ._core import CylindricalWaveBasis as CylindricalBasis
from ._core import PlaneWaveBasisByComp as PlaneWavePorts
from ._core import PlaneWaveBasisByUnitVector as PlaneWaveBasis
from ._core import SphericalWaveBasis as SphericalBasis
from ._framework import (
    BandModes as BandModes,
)
from ._framework import (
    Cluster as Cluster,
)
from ._framework import (
    CrossSections as CrossSections,
)
from ._framework import (
    Material as Material,
)
from ._framework import (
    PeriodicResponse as PeriodicResponse,
)
from ._framework import PortWave as PortWave
from ._framework import (
    PowerBalance as PowerBalance,
)
from ._framework import ScatteredPorts as ScatteredPorts
from ._framework import SMatrix as SMatrix
from ._framework import TMatrix as TMatrix
from ._framework import Wave as Wave
from ._lattice import Lattice as Lattice

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

    from ._adapters import Array, Pullback, Record

type Output = torch.Tensor | tuple[torch.Tensor, ...]


def _numpy(value: torch.Tensor) -> Array:
    return value.detach().resolve_conj().resolve_neg().numpy()


class _Execute(torch.autograd.Function):
    @staticmethod
    @override
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any, record: Record, *values: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        snapshots = tuple(v.detach().clone() for v in values)
        primals = tuple(_numpy(v) for v in snapshots)
        outputs, pullback, _ = execute(record, primals)
        ctx.record = record
        ctx.pullback = pullback
        ctx.input_count = len(values)
        # PyTorch checks these tensors' version counters before any backward,
        # including repeats; caller mutation cannot change recomputed primals.
        ctx.save_for_backward(*values, *snapshots)
        return tuple(torch.from_numpy(np.array(v, copy=True)) for v in outputs)

    @staticmethod
    @override
    def backward(
        ctx: Any, *cotangents: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise NotImplementedError(
                "treams-rs PyTorch adapters support first-order VJPs only"
            )
        # Saved tensors validate original version counters and release snapshots
        # with the graph; NumPy aliases cannot alter the owned recomputation data.
        values = tuple(_numpy(v) for v in ctx.saved_tensors[ctx.input_count :])
        pullback = ctx.pullback
        ctx.pullback = cast("Pullback | None", None)
        if pullback is None:
            _, pullback, _ = execute(ctx.record, values)
        result = gradients(
            pullback, tuple(_numpy(g) for g in cotangents), values, conjugate=False
        )
        return (None, *(torch.from_numpy(np.array(g, copy=True)) for g in result))


def wrap(record: Record) -> Callable[..., Output]:
    """Adapt a native forward/context function, holding static options in a closure.

    Native pullbacks must return one gradient per dynamic parameter, in order.
    A single output returns a Tensor; multiple outputs return a flat tuple.
    See docs/adapters.md for native operations with reordered/extra adjoints.
    """

    def operation(*values: torch.Tensor | ArrayLike) -> Output:
        arrays = tuple(
            value
            if isinstance(value, torch.Tensor)
            else torch.from_numpy(input_array(value))
            for value in values
        )
        for array in arrays:
            if array.device.type != "cpu":
                raise ValueError("treams-rs PyTorch adapters require CPU tensors")
            if array.dtype not in (torch.float64, torch.complex128):
                raise TypeError(
                    "native adapters require float64 or complex128 parameters"
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


def solve(
    operator: torch.Tensor | ArrayLike, rhs: torch.Tensor | ArrayLike
) -> torch.Tensor:
    """Solve A X = B using native LU and its analytic first-order VJP."""
    return cast("torch.Tensor", wrap(diff.solve)(operator, rhs))


def interaction(
    local: torch.Tensor | ArrayLike, coupling: torch.Tensor | ArrayLike
) -> torch.Tensor:
    """Solve (I - T C) X = T with native matrix pullbacks."""
    return cast("torch.Tensor", wrap(diff.interaction)(local, coupling))


def illuminate(
    local: torch.Tensor | ArrayLike,
    coupling: torch.Tensor | ArrayLike,
    incident: torch.Tensor | ArrayLike,
) -> torch.Tensor:
    """Solve only requested incident columns, with native T/C/incident adjoints."""
    return cast("torch.Tensor", wrap(diff.illuminate)(local, coupling, incident))


def sphere(
    lmax: int,
    k0: torch.Tensor | ArrayLike,
    radii: torch.Tensor | ArrayLike,
    epsilon: torch.Tensor | ArrayLike,
    mu: torch.Tensor | ArrayLike | None = None,
    kappa: torch.Tensor | ArrayLike | None = None,
) -> torch.Tensor:
    """Multilayer chiral sphere with native material, radius and frequency VJPs."""

    def record(k: Array, r: Array, e: Array, m: Array, c: Array) -> tuple[Any, Any]:
        return diff.sphere(lmax, float(k), r, e, m, c)

    return cast(
        "torch.Tensor",
        wrap(record)(
            k0,
            radii,
            epsilon,
            np.ones(np.shape(epsilon)) if mu is None else mu,
            np.zeros(np.shape(epsilon)) if kappa is None else kappa,
        ),
    )


def bessel(
    z: torch.Tensor | ArrayLike,
    *,
    order: ArrayLike,
    kind: str = "j",
    spherical: bool = False,
    derivative: bool = False,
) -> torch.Tensor:
    """Broadcast Bessel values with the order fixed and a native argument VJP."""

    def record(value: Array) -> tuple[Any, Any]:
        return diff.bessel(
            order, value, kind=kind, spherical=spherical, derivative=derivative
        )

    return cast("torch.Tensor", wrap(record)(z))


class PlaneWave(_physics.PlaneWave):
    """Plane illumination using this namespace's explicitly selected framework."""

    def __init__(
        self, direction: Any, polarization: Any, *, k0: Any, medium: Any = 1.0
    ):
        super().__init__(
            direction, polarization, k0=k0, medium=medium, backend=_backend
        )


def smatrix(
    array: Any,
    *,
    basis: PlaneWaveBasisByComp,
    k0: Any,
    negative_medium: Any = 1.0,
    positive_medium: Any = 1.0,
    polarization: str = "helicity",
) -> SMatrix:
    """Wrap scattering blocks (2,2,modes,modes) with explicit exterior media."""
    matrix = _backend.array(array, complex_=True)
    if matrix.shape != (2, 2, len(basis), len(basis)):
        raise ValueError("scattering blocks must have shape (2,2,modes,modes)")
    result = SMatrix(
        matrix,
        basis=basis,
        k0=_backend.array(k0),
        media=(
            _physics._material(positive_medium),
            _physics._material(negative_medium),
        ),
        backend=_backend,
    )
    result.polarization = polarization
    return result


def wave(
    coefficients: Any,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    k0: Any,
    medium: Any = 1.0,
    kind: str = "regular",
    polarization: str = "helicity",
    positions: Any = None,
) -> Wave:
    """Construct a physical multipole wave using this namespace; coefficients have shape (modes,)."""
    if kind not in ("regular", "outgoing"):
        raise ValueError("wave kind must be regular or outgoing")
    array = _backend.array(coefficients, complex_=True)
    if array.shape != (len(basis),):
        raise ValueError("one coefficient is required per basis mode")
    return Wave(
        array,
        basis=basis,
        k0=_backend.array(k0),
        medium=_physics._material(medium),
        backend=_backend,
        positions=positions,
        outgoing=kind == "outgoing",
        polarization=polarization,
    )


def tmatrix(
    array: Any,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    k0: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
    positions: Any = None,
) -> TMatrix:
    """Wrap a user response (modes,modes) with fixed labels and dynamic physical metadata."""
    matrix = _backend.array(array, complex_=True)
    if matrix.shape != (len(basis), len(basis)):
        raise ValueError("matrix dimensions must match the basis")
    return TMatrix(
        matrix,
        basis=basis,
        k0=_backend.array(k0),
        medium=_physics._material(medium),
        backend=_backend,
        positions=positions,
        polarization=polarization,
    )


def sphere_tmatrix(
    *,
    k0: Any,
    lmax: int,
    radius: Any,
    material: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
) -> TMatrix:
    """Homogeneous sphere with differentiable geometry, material and frequency."""
    return multilayer_sphere_tmatrix(
        k0=k0,
        lmax=lmax,
        radii=_backend.stack((_backend.array(radius),)),
        materials=(material,),
        medium=medium,
        polarization=polarization,
    )


def multilayer_sphere_tmatrix(
    *,
    k0: Any,
    lmax: int,
    radii: Any,
    materials: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
) -> TMatrix:
    """Concentric layers, one material per radius, with a separate exterior."""
    result = _backend.multilayer(
        k0=k0,
        radii=radii,
        materials=materials,
        medium=medium,
        basis=SphericalWaveBasis.default(lmax),
        polarization="helicity",
    )
    return (
        result if polarization == "helicity" else result.with_polarization(polarization)
    )


def cylinder_tmatrix(
    *,
    k0: Any,
    kz: Any,
    mmax: int,
    radius: Any,
    material: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
) -> TMatrix:
    """Homogeneous cylinder; axial mode labels are fixed configuration."""
    return multilayer_cylinder_tmatrix(
        k0=k0,
        kz=kz,
        mmax=mmax,
        radii=_backend.stack((_backend.array(radius),)),
        materials=(material,),
        medium=medium,
        polarization=polarization,
    )


def multilayer_cylinder_tmatrix(
    *,
    k0: Any,
    kz: Any,
    mmax: int,
    radii: Any,
    materials: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
) -> TMatrix:
    """Concentric cylinders with fixed axial labels and dynamic layers."""
    result = _backend.multilayer(
        k0=k0,
        radii=radii,
        materials=materials,
        medium=medium,
        basis=CylindricalWaveBasis.default(kz, mmax),
        polarization="helicity",
        kz=kz,
        mmax=mmax,
    )
    return (
        result if polarization == "helicity" else result.with_polarization(polarization)
    )


def plane_wave(
    direction: Any, polarization: Any, *, k0: Any, medium: Any = 1.0
) -> PlaneWave:
    """Fixed-direction plane wave with dynamic frequency, medium and amplitudes."""
    return PlaneWave(direction, polarization, k0=k0, medium=medium)


def solve_periodic(
    unit_cell: TMatrix | Cluster, *, lattice: Any, kpar: Any, eta: complex = 0
) -> PeriodicResponse:
    """Solve periodic coupling once, then use response.to_smatrix(ports)."""
    return _physics.solve_periodic(unit_cell, lattice=lattice, kpar=kpar, eta=eta)


def slab(
    *,
    basis: PlaneWaveBasisByComp,
    k0: Any,
    thickness: Any,
    material: Any,
    negative_medium: Any = 1.0,
    positive_medium: Any = 1.0,
    polarization: str = "helicity",
) -> SMatrix:
    """One layer with differentiable geometry and explicit exterior media."""
    return _physics.layer_stack(
        _backend,
        k0=k0,
        basis=basis,
        materials=(negative_medium, material, positive_medium),
        thickness=thickness,
    ).with_polarization(polarization)


def interface(
    *,
    basis: PlaneWaveBasisByComp,
    k0: Any,
    negative_medium: Any,
    positive_medium: Any,
    polarization: str = "helicity",
) -> SMatrix:
    """Interface from negative to positive side of the port normal."""
    return _physics.layer_stack(
        _backend,
        k0=k0,
        basis=basis,
        materials=(negative_medium, positive_medium),
        thickness=[],
    ).with_polarization(polarization)


def multilayer_slab(
    *,
    basis: PlaneWaveBasisByComp,
    k0: Any,
    thicknesses: Any,
    materials: Any,
    negative_medium: Any = 1.0,
    positive_medium: Any = 1.0,
    polarization: str = "helicity",
) -> SMatrix:
    """Interior layers in increasing normal order, one material per thickness."""
    return _physics.layer_stack(
        _backend,
        k0=k0,
        basis=basis,
        materials=(negative_medium, *materials, positive_medium),
        thickness=thicknesses,
    ).with_polarization(polarization)


def propagation(
    *,
    distance: Any,
    basis: PlaneWaveBasisByComp,
    k0: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
) -> SMatrix:
    """Homogeneous propagation with native distance and wavevector derivatives."""
    return _physics.propagation(
        _backend,
        distance=distance,
        basis=basis,
        k0=k0,
        medium=medium,
        polarization=polarization,
    )


def stack(layers: Any) -> SMatrix:
    """Cascade layers from the negative to positive side of the port normal."""
    if not layers:
        raise ValueError("stack requires at least one layer")
    result = layers[0]
    for layer in layers[1:]:
        result = result.cascade(layer)
    return result


def _physics_call(
    record: Any, shape: tuple[int, ...], *values: Any, real: bool = False
) -> Any:
    return wrap(record)(*values)


_backend = _physics.Backend(torch, _physics_call, torch=True)


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

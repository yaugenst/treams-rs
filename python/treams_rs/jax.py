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

# Physical objects share one implementation; this namespace selects execution.
from . import _framework as _physics
from . import diff
from ._adapters import execute, gradients, input_array, require_dtype
from ._core import (
    CylindricalWaveBasis,
    PlaneWaveBasisByComp,
    SphericalWaveBasis,
)
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
    dtype = np.float64 if real else np.complex128
    return _operation(record, (jax.ShapeDtypeStruct(shape, dtype),), False)(*values)


def _physics_validate(value: Any) -> None:
    if not bool(jax.config.read("jax_enable_x64")):
        raise ValueError("treams-rs JAX adapters require jax_enable_x64=True")
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        _inputs((value,))


_backend = _physics.Backend(jnp, _physics_call, validate=_physics_validate)


__all__ = [
    "BandModes",
    "Cluster",
    "CrossSections",
    "Material",
    "PeriodicResponse",
    "PlaneWave",
    "PortWave",
    "PowerBalance",
    "SMatrix",
    "ScatteredPorts",
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

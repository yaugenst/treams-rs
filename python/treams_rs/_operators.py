"""Explicit basis operators backed by native numerical kernels."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

import numpy as np

from . import diff
from . import lattice as _lattice
from ._core import (
    CylindricalWaveBasis,
    Material,
    PlaneWaveBasisByComp,
    PlaneWaveBasisByUnitVector,
    SphericalWaveBasis,
)
from ._lattice import Lattice, WaveVector, _geometry_inputs
from .config import _resolve_poltype

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

    from ._core import MaterialLike

    type Basis = SphericalWaveBasis | CylindricalWaveBasis
    type FieldBasis = Basis | PlaneWaveBasisByComp | PlaneWaveBasisByUnitVector


def changepoltype(
    poltype: str | tuple[str, str] | None = None,
    *,
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    where: ArrayLike = True,
) -> NDArray[np.float64]:
    """Explicit helicity/parity conversion, including rectangular basis subsets.

    poltype names the destination type, or a (destination, source) pair.
    The real transformation is its own inverse for complete polarization pairs.
    Mode labels, origins and masks are discrete metadata.
    """
    poltype = _resolve_poltype(None) if poltype is None else poltype
    if poltype not in (
        "helicity",
        "parity",
        ("helicity", "parity"),
        ("parity", "helicity"),
    ):
        raise ValueError("polarization conversion must switch helicity and parity")
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    if type(destination) is not type(source):
        raise ValueError("polarization conversion requires the same wave family")
    if (
        isinstance(destination, PlaneWaveBasisByComp)
        and isinstance(source, PlaneWaveBasisByComp)
        and destination.alignment != source.alignment
    ):
        raise ValueError("polarization conversion requires matching alignments")
    out, incoming = np.asarray(destination.modes), np.asarray(source.modes)
    same = np.all(out[:, None, :-1] == incoming[None, :, :-1], axis=-1)
    signs = np.where((destination.pol[:, None] == 0) & (source.pol == 0), -1, 1)
    return (same & np.asarray(where, dtype=bool)) * signs * np.sqrt(0.5)


def rotate(
    phi: float,
    theta: float = 0,
    psi: float = 0,
    *,
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Rotation matrix in the z-y-z convention.

    Multipole origins remain fixed. Plane-wave rotations preserve coefficients
    and rotate the direction labels; their output basis is basis.rotate(phi+psi).
    As in treams, plane-wave theta must be zero and component bases must be xy.
    """
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    if isinstance(source, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        if type(destination) is not type(source) or destination.modes != source.modes:
            raise ValueError("plane rotations require matching input and output bases")
        if theta != 0 or not np.isfinite([phi, psi]).all():
            raise ValueError("plane rotations require finite phi/psi and zero theta")
        if isinstance(source, PlaneWaveBasisByComp) and (
            source.alignment != "xy"
            or not isinstance(destination, PlaneWaveBasisByComp)
            or destination.alignment != "xy"
        ):
            raise ValueError("plane rotations require xy alignment")
        return _masked(np.eye(len(source), dtype=np.complex128), where)
    if not isinstance(destination, (SphericalWaveBasis, CylindricalWaveBasis)):
        raise ValueError("rotations require matching wave families")
    return _masked(diff.rotation([phi, theta, psi], destination, source)[0], where)


def permute(
    n: int = 1,
    *,
    basis: PlaneWaveBasisByComp | PlaneWaveBasisByUnitVector,
    k0: float | None = None,
    material: MaterialLike = 1,
    modetype: str = "up",
    poltype: str | None = None,
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Cyclic coordinate permutation; the output basis is basis.permute(n).

    Returns an explicit polarization matrix. Basis directions transform separately,
    preserving the same Cartesian field under the corresponding axis permutation.
    """
    poltype = _resolve_poltype(poltype)
    if isinstance(basis, PlaneWaveBasisByUnitVector):
        vectors = basis.directions
    elif isinstance(basis, PlaneWaveBasisByComp):
        if k0 is None:
            raise ValueError("component plane permutations require k0")
        vectors = np.column_stack(basis.kvecs(k0, material, modetype))
    else:
        raise TypeError("permutations require a plane-wave basis")
    coefficients, _ = diff.plane_permutation(vectors, basis.pol, n, poltype=poltype)
    same = _plane_wave_match(vectors, vectors)
    return _masked(
        np.where(
            same, coefficients[basis.pol[:, None], np.arange(len(basis))[None, :]], 0
        ),
        where,
    )


def expand(
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    modetype: str | tuple[str, str] | None = None,
    *,
    k0: float,
    material: MaterialLike = 1,
    poltype: str | None = None,
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Multipole expansion, including regular cylindrical-to-spherical waves.

    A basis pair is (destination, source). All explicit origin pairs are included.
    """
    poltype = _resolve_poltype(poltype)
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    medium = Material(material)
    if not np.isfinite(k0) or k0 <= 0 or (poltype == "parity" and medium.ischiral):
        raise ValueError(
            "invalid frequency or embedding medium for the polarization type"
        )
    mask = np.asarray(where, dtype=bool)
    if isinstance(source, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        if isinstance(destination, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
            sides = modetype if isinstance(modetype, tuple) else (modetype or "up",) * 2
            return (
                _plane_wave_match(
                    np.column_stack(destination.kvecs(k0, medium, sides[0])),
                    np.column_stack(source.kvecs(k0, medium, sides[1])),
                )
                & (destination.pol[:, None] == source.pol)
                & mask
            ).astype(np.complex128)
        types = (
            ("regular", "up")
            if modetype is None
            else (modetype if isinstance(modetype, tuple) else ("regular", modetype))
        )
        if types[0] != "regular" or (
            isinstance(source, PlaneWaveBasisByComp) and types[1] not in ("up", "down")
        ):
            raise ValueError(
                "plane waves expand into regular multipoles from up/down modes"
            )
        value = diff.plane_expansion(
            destination,
            np.column_stack(source.kvecs(k0, medium, types[1])),
            source.pol,
            poltype=poltype,
            fixed_vectors=True,
        )[0]
        return _masked(value, where)
    if isinstance(destination, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        raise ValueError(
            "multipole-to-plane expansion requires a periodic radiation operator"
        )

    types = (
        ("regular", "regular")
        if modetype is None
        else (modetype if isinstance(modetype, tuple) else (modetype, modetype))
    )
    if types not in (
        ("regular", "regular"),
        ("singular", "singular"),
        ("regular", "singular"),
    ):
        raise ValueError("unsupported multipole expansion mode types")
    if type(destination) is not type(source) and types != ("regular", "regular"):
        raise ValueError("cylindrical-to-spherical conversion requires regular waves")
    # Equal radial types use the regular addition theorem.
    value = diff.expansion(
        destination,
        source,
        medium.ks(k0),
        poltype=poltype,
        singular=types == ("regular", "singular"),
    )[0]
    return _masked(value, where)


def _masked(value: NDArray[np.complex128], where: ArrayLike) -> NDArray[np.complex128]:
    if where is not True:
        value *= np.asarray(where, dtype=bool)
    return value


def _plane_wave_match(
    destination: NDArray[np.complex128], source: NDArray[np.complex128]
) -> NDArray[np.bool_]:
    if not np.isfinite(destination).all() or not np.isfinite(source).all():
        raise ValueError("plane wavevectors must be finite")
    tolerance = (
        32
        * np.finfo(float).eps
        * np.maximum(
            np.max(abs(destination), axis=1)[:, None], np.max(abs(source), axis=1)
        )
    )
    same = np.ones((len(destination), len(source)), dtype=bool)
    for axis in range(3):
        same &= abs(destination[:, None, axis] - source[:, axis]) <= tolerance
    return same


def translate(
    r: ArrayLike,
    *,
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    k0: float,
    material: MaterialLike = 1,
    poltype: str | None = None,
    modetype: str = "up",
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Regular translation at fixed local origins, with displacement shape (..., 3).

    Multipole translations pair equal particle indices and ignore the stored
    origins. Use expand for all physical origin pairs. Plane translations apply
    exp(i k.r) to matching wavevectors and polarizations.
    """
    poltype = _resolve_poltype(poltype)
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    offsets = np.asarray(r, dtype=np.float64)
    medium = Material(material)
    if offsets.ndim == 0 or offsets.shape[-1] != 3 or not np.isfinite(offsets).all():
        raise ValueError("translations require finite Cartesian displacements (..., 3)")
    if not np.isfinite(k0) or k0 <= 0 or (poltype == "parity" and medium.ischiral):
        raise ValueError(
            "invalid frequency or embedding medium for the polarization type"
        )
    shape = (*offsets.shape[:-1], len(destination), len(source))
    if isinstance(
        source, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)
    ) and isinstance(destination, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        vectors = np.column_stack(source.kvecs(k0, medium, modetype))
        matching = _plane_wave_match(
            np.column_stack(destination.kvecs(k0, medium, modetype)), vectors
        ) & (destination.pol[:, None] == source.pol)
        phases = diff.plane_phases(offsets.reshape(-1, 3), vectors)[0]
        return _masked((phases[:, None, :] * matching).reshape(shape), where)
    if (
        not isinstance(source, (SphericalWaveBasis, CylindricalWaveBasis))
        or not isinstance(destination, (SphericalWaveBasis, CylindricalWaveBasis))
        or type(destination) is not type(source)
    ):
        raise ValueError("translation requires matching wave families")
    incoming = type(source)(source.modes, np.zeros_like(source.positions))
    matching = destination.pidx[:, None] == source.pidx
    points = offsets.reshape(-1, 3)
    ks = medium.ks(k0)
    result = np.empty((len(points), len(destination), len(source)), dtype=np.complex128)
    for i, offset in enumerate(points):
        outgoing = type(destination)(
            destination.modes, np.broadcast_to(offset, destination.positions.shape)
        )
        result[i] = (
            diff.expansion(outgoing, incoming, ks, poltype=poltype)[0] * matching
        )
    return _masked(result.reshape(shape), where)


def _field(
    kind: str,
    r: ArrayLike,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike,
    modetype: str | None,
    poltype: str,
    coefficients: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    medium = Material(material)
    points = np.asarray(r, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 3 or not np.isfinite(k0) or k0 <= 0:
        raise ValueError(
            "require Cartesian field points (..., 3) and positive finite k0"
        )
    plane = isinstance(basis, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector))
    modetype = ("up" if plane else "regular") if modetype is None else modetype
    if not isinstance(basis, PlaneWaveBasisByUnitVector) and modetype not in (
        ("up", "down") if plane else ("regular", "singular")
    ):
        raise ValueError("invalid field mode type for this basis")
    if poltype not in ("helicity", "parity") or (
        poltype == "parity" and medium.ischiral
    ):
        raise ValueError("invalid polarization type for embedding medium")
    weights = np.ones(len(basis), dtype=np.complex128)
    if kind in ("H", "B"):
        weights *= -1j / medium.impedance
        if poltype == "helicity":
            weights *= 2 * basis.pol - 1
        else:
            if isinstance(basis, PlaneWaveBasisByUnitVector):
                basis = type(basis)([(*mode[:3], 1 - mode[3]) for mode in basis.modes])
            else:
                modes = [(*mode[:-1], 1 - mode[-1]) for mode in basis.modes]
                basis = (
                    type(basis)(modes, basis.alignment)
                    if isinstance(basis, PlaneWaveBasisByComp)
                    else type(basis)(modes, basis.positions)
                )
    if kind == "D":
        weights *= (
            medium.nmp[basis.pol] / medium.impedance
            if poltype == "helicity"
            else medium.epsilon
        )
    if kind == "B":
        weights *= (
            medium.nmp[basis.pol] * medium.impedance
            if poltype == "helicity"
            else medium.mu
        )
    weighted = None
    if coefficients is not None:
        amplitudes = np.asarray(coefficients, dtype=np.complex128)
        if amplitudes.shape != (len(basis),):
            raise ValueError("field requires one amplitude per basis mode")
        weighted = amplitudes * weights
    if isinstance(basis, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        value, _ = diff.plane_field(
            weighted,
            points.reshape(-1, 3),
            np.column_stack(basis.kvecs(k0, medium, modetype)),
            basis.pol,
            poltype=poltype,
            fixed_vectors=True,
        )
    elif weighted is None:
        value, _ = diff.field_operator(
            points.reshape(-1, 3),
            basis,
            medium.ks(k0),
            poltype=poltype,
            singular=modetype == "singular",
        )
    else:
        value, _ = diff.field(
            weighted,
            points.reshape(-1, 3),
            basis,
            medium.ks(k0),
            poltype=poltype,
            singular=modetype == "singular",
        )
    if weighted is not None:
        return value.reshape(points.shape)
    return value.reshape((*points.shape[:-1], 3, len(basis))) * weights


def efield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Cartesian electric-field operator (..., 3, modes)."""
    poltype = _resolve_poltype(poltype)
    return _field("E", r, basis, k0, material, modetype, poltype)


def hfield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Magnetic-field operator in units of electric field / vacuum impedance."""
    poltype = _resolve_poltype(poltype)
    return _field("H", r, basis, k0, material, modetype, poltype)


def dfield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Electric-displacement operator in units of vacuum permittivity times E."""
    poltype = _resolve_poltype(poltype)
    return _field("D", r, basis, k0, material, modetype, poltype)


def bfield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Magnetic-flux operator in units of electric field / vacuum light speed."""
    poltype = _resolve_poltype(poltype)
    return _field("B", r, basis, k0, material, modetype, poltype)


def _rs_weights(
    pol: int, basis: FieldBasis, poltype: str
) -> tuple[NDArray[np.float64], float]:
    if pol not in (-1, 0, 1):
        raise ValueError("Riemann-Silberstein polarization must be -1, 0 or 1")
    pol = max(pol, 0)
    # Preserve upstream's different spherical and cylindrical/plane scalings.
    normalization = np.sqrt(2) if isinstance(basis, SphericalWaveBasis) else 1.0
    if poltype == "helicity":
        return normalization * (basis.pol == pol), 0.0
    if poltype == "parity":
        return np.full(len(basis), normalization), normalization * (2 * pol - 1)
    raise ValueError("polarization type must be helicity or parity")


def gfield(
    pol: int,
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Riemann-Silberstein G operator with treams' family/polarization scaling.

    Polarization -1 aliases 0. For a normalization independent of the basis
    convention, form (E +/- i Z H)/sqrt(2) from efield and hfield directly.
    """
    poltype = _resolve_poltype(poltype)
    electric, magnetic = _rs_weights(pol, basis, poltype)
    value = _field("E", r, basis, k0, material, modetype, poltype) * electric
    if magnetic:
        value += (
            1j
            * Material(material).impedance
            * magnetic
            * _field("H", r, basis, k0, material, modetype, poltype)
        )
    return value


def ffield(
    pol: int,
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Riemann-Silberstein F operator, including the chiral index weights."""
    poltype = _resolve_poltype(poltype)
    value = gfield(
        pol,
        r,
        basis=basis,
        k0=k0,
        material=material,
        modetype=modetype,
        poltype=poltype,
    )
    if poltype == "helicity":
        medium = Material(material)
        value *= medium.nmp[basis.pol] / medium.n
    return value


def _periodic_channels(
    source: Basis,
    destination: PlaneWaveBasisByComp,
    ks: ArrayLike,
    lattice: ArrayLike,
    kpar: ArrayLike,
    poltype: str,
) -> NDArray[np.complex128]:
    """Validate physical diffraction ports and return native incidence/emission blocks."""
    vectors, bloch = _geometry_inputs(
        lattice, kpar, "xy" if isinstance(source, SphericalWaveBasis) else "x"
    )
    q = destination.components
    if isinstance(source, SphericalWaveBasis):
        if (
            vectors.shape != (2, 2)
            or bloch.shape != (2,)
            or destination.alignment != "xy"
        ):
            raise ValueError(
                "spherical arrays require a 2D xy lattice, Bloch vector and plane basis"
            )
        orders = (q - bloch) @ vectors.T / (2 * np.pi)
        measure = float(abs(np.linalg.det(vectors)))
    else:
        if (
            vectors.shape != (1, 1)
            or bloch.shape != (1,)
            or destination.alignment != "zx"
        ):
            raise ValueError(
                "cylindrical arrays require a 1D x period, Bloch vector and zx plane basis"
            )
        orders = (q[:, 1] - bloch[0]) * vectors[0, 0] / (2 * np.pi)
        measure = float(abs(vectors[0, 0]))
    if not np.allclose(orders, np.round(orders), atol=1e-10, rtol=0):
        raise ValueError(
            "plane-wave channels must match the lattice diffraction orders"
        )
    # Both native channel signatures share the same scalar cell-measure argument.
    if isinstance(source, SphericalWaveBasis):
        return diff.spherical_channels(
            source, ks, q, destination.pol, measure, poltype=poltype, fixed_q=True
        )[0]
    return diff.cylindrical_channels(
        source, ks, q, destination.pol, measure, poltype=poltype, fixed_q=True
    )[0]


def expandlattice(
    lattice: ArrayLike | Lattice | None = None,
    kpar: ArrayLike | WaveVector | None = None,
    *,
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    k0: float,
    material: MaterialLike = 1,
    poltype: str | None = None,
    modetype: str | tuple[str, str] | None = None,
    eta: complex = 0,
) -> NDArray[np.complex128]:
    """Periodic multipole coupling or radiation, with explicit cell and Bloch vector.

    Spherical cells follow z/xy/xyz in 1D/2D/3D; cylindrical cells follow x/xy.
    A basis pair is (destination, source). Cross-family radiation includes all
    explicit origin pairs rather than an implicit matching-particle-index mask.
    """
    poltype = _resolve_poltype(poltype)
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    lattice = (destination.lattice or source.lattice) if lattice is None else lattice
    kpar = (
        (source.kpar if source.kpar is not None else destination.kpar)
        if kpar is None
        else kpar
    )
    if lattice is None or kpar is None:
        raise ValueError("periodic expansion requires a lattice and Bloch vector")
    medium = Material(material)
    if not np.isfinite(k0) or k0 <= 0 or (poltype == "parity" and medium.ischiral):
        raise ValueError(
            "invalid frequency or embedding medium for the polarization type"
        )
    if not isinstance(source, (SphericalWaveBasis, CylindricalWaveBasis)):
        raise ValueError("periodic expansion requires a multipole source")
    if isinstance(destination, PlaneWaveBasisByComp):
        side = (
            "up"
            if modetype is None
            else modetype[0]
            if isinstance(modetype, tuple)
            else modetype
        )
        if side not in ("up", "down") or (
            isinstance(modetype, tuple) and modetype[1] != "singular"
        ):
            raise ValueError("plane radiation requires up/down outgoing plane modes")
        channels = _periodic_channels(
            source, destination, medium.ks(k0), lattice, kpar, poltype
        )
        return channels[1, 0 if side == "up" else 1].T
    if isinstance(destination, CylindricalWaveBasis) and isinstance(
        source, SphericalWaveBasis
    ):
        vectors, bloch = _geometry_inputs(lattice, kpar, "z")
        if vectors.shape != (1, 1) or bloch.shape != (1,):
            raise ValueError(
                "spherical-to-cylindrical radiation requires a 1D z period and Bloch component"
            )
        orders = (destination.kz - bloch[0]) * vectors[0, 0] / (2 * np.pi)
        if not np.allclose(orders, np.round(orders), atol=1e-10, rtol=0):
            raise ValueError(
                "cylindrical axial wavenumbers must match diffraction orders"
            )
        if modetype not in (None, "singular", ("singular", "singular")):
            raise ValueError(
                "periodic spherical-to-cylindrical radiation requires outgoing waves"
            )
        return diff.periodic_conversion(
            destination,
            source,
            medium.ks(k0),
            float(abs(vectors[0, 0])),
            poltype=poltype,
        )[0]
    if type(destination) is type(source):
        if modetype not in (None, "regular", ("regular", "singular")):
            raise ValueError("periodic coupling maps outgoing to regular waves")
        return _lattice.expansion(
            destination, source, medium.ks(k0), lattice, kpar, poltype=poltype, eta=eta
        )
    raise ValueError("unsupported periodic wave-family conversion")


class Operator:
    """Reusable explicit operator; calls return arrays and @ infers input metadata."""

    __array_priority__ = 1.0
    _FUNC: ClassVar[Callable[..., NDArray[np.generic]]]
    _metadata: ClassVar[tuple[str, ...]] = ()
    _parameters: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(
        cls, *, func: Callable[..., NDArray[np.generic]] | None = None
    ) -> None:
        super().__init_subclass__()
        if func is not None:
            cls._FUNC = func
        cls._parameters = tuple(
            name
            for name in inspect.signature(cls.__init__).parameters
            if name != "self"
        )
        if hasattr(cls, "_FUNC"):
            cls._metadata = tuple(
                name
                for name, parameter in inspect.signature(cls._FUNC).parameters.items()
                if parameter.kind == inspect.Parameter.KEYWORD_ONLY
            )

    def __init__(self, *args: Any, isinv: bool = False) -> None:
        self._args = args
        self._isinv = bool(isinv)

    @property
    def isinv(self) -> bool:
        return self._isinv

    @property
    def inv(self) -> Self:
        return type(self)(*self._args, isinv=not self.isinv)

    @property
    def FUNC(self) -> Callable[..., NDArray[np.generic]]:  # noqa: N802 - upstream operator interface
        return type(self)._FUNC

    def __call__(self, **kwargs: Any) -> NDArray[np.generic]:
        if self.isinv:
            return self._call_inv(**kwargs)
        return self.FUNC(*self._args, **kwargs)

    def _call_inv(self, **kwargs: Any) -> NDArray[np.generic]:
        raise NotImplementedError("this operator has no inverse")

    def get_kwargs(self, obj: Any, dim: int = -1) -> dict[str, Any]:
        values = {}
        for name in self._metadata:
            value = getattr(obj, name, None)
            if isinstance(value, tuple):
                value = value[dim]
            if value is not None:
                values[name] = value
        return values

    def __matmul__(self, other: Any) -> NDArray[np.generic]:
        if isinstance(other, Operator):
            raise NotImplementedError(
                "evaluate operators in explicit bases before composing"
            )
        return self(
            **self.get_kwargs(other, -2 if np.ndim(other) > 1 else -1)
        ) @ np.asarray(other)

    def __rmatmul__(self, other: Any) -> NDArray[np.generic]:
        return np.asarray(other) @ self(**self.get_kwargs(other))

    def __repr__(self) -> str:
        arguments = ", ".join(repr(value) for value in self._args)
        return (
            f"{type(self).__name__}({arguments}{', isinv=True' if self.isinv else ''})"
        )


def _reverse_basis(kwargs: dict[str, Any]) -> None:
    if isinstance(kwargs.get("basis"), tuple):
        kwargs["basis"] = kwargs["basis"][::-1]


class Rotate(Operator, func=rotate):
    """Euler rotation with an exact inverse obtained by reversing the angles."""

    def __init__(
        self, phi: float, theta: float = 0, psi: float = 0, *, isinv: bool = False
    ) -> None:
        super().__init__(phi, theta, psi, isinv=isinv)

    @override
    def _call_inv(self, **kwargs: Any) -> NDArray[np.generic]:
        _reverse_basis(kwargs)
        return rotate(*(-value for value in self._args[::-1]), **kwargs)


class Translate(Operator, func=translate):
    """Translation with the inverse displacement in the reversed basis pair."""

    def __init__(self, r: ArrayLike, *, isinv: bool = False) -> None:
        super().__init__(np.array(r, dtype=np.float64, copy=True), isinv=isinv)

    @override
    def _call_inv(self, **kwargs: Any) -> NDArray[np.generic]:
        _reverse_basis(kwargs)
        return translate(-self._args[0], **kwargs)


class ChangePoltype(Operator, func=changepoltype):
    """Helicity/parity conversion; omission selects the opposite input convention."""

    def __init__(
        self, poltype: str | tuple[str, str] | None = None, *, isinv: bool = False
    ) -> None:
        super().__init__(*(() if poltype is None else (poltype,)), isinv=isinv)

    @override
    def get_kwargs(self, obj: Any, dim: int = -1) -> dict[str, Any]:
        kwargs = super().get_kwargs(obj, dim)
        if not self._args:
            poltype = getattr(obj, "poltype", None)
            if isinstance(poltype, tuple):
                poltype = poltype[dim]
            if poltype is not None:
                kwargs["poltype"] = "parity" if poltype == "helicity" else "helicity"
        return kwargs

    @override
    def _call_inv(self, **kwargs: Any) -> NDArray[np.generic]:
        _reverse_basis(kwargs)
        poltype = (
            self._args[0]
            if self._args
            else _resolve_poltype(kwargs.pop("poltype", None))
        )
        destination = (
            poltype[::-1]
            if isinstance(poltype, tuple)
            else "parity"
            if poltype == "helicity"
            else "helicity"
        )
        return changepoltype(destination, **kwargs)


class Expand(Operator, func=expand):
    """Expand into a destination basis, obtaining the source from keyword metadata."""

    def __init__(
        self,
        basis: FieldBasis | tuple[FieldBasis, FieldBasis],
        modetype: str | tuple[str, str] | None = None,
        *,
        isinv: bool = False,
    ) -> None:
        super().__init__(basis, modetype, isinv=isinv)

    def __call__(self, **kwargs: Any) -> NDArray[np.generic]:
        basis, modetype = self._args
        source = kwargs.pop("basis", None)
        if source is not None and not isinstance(basis, tuple):
            basis = (basis, source)
        source_type = kwargs.pop("modetype", None)
        if modetype is None:
            modetype = source_type
        elif source_type is not None and not isinstance(modetype, tuple):
            modetype = (modetype, source_type)
        if self.isinv:
            basis = basis[::-1] if isinstance(basis, tuple) else basis
            modetype = modetype[::-1] if isinstance(modetype, tuple) else modetype
        return expand(basis, modetype, **kwargs)

    @override
    def get_kwargs(self, obj: Any, dim: int = -1) -> dict[str, Any]:
        kwargs = super().get_kwargs(obj, dim)
        for name in ("basis", "modetype"):
            value = getattr(obj, name, None)
            if isinstance(value, tuple):
                value = value[dim]
            if value is not None:
                kwargs[name] = value
        return kwargs


class ExpandLattice(Operator, func=expandlattice):
    """Periodic expansion into an optional destination basis."""

    def __init__(
        self,
        lattice: ArrayLike | Lattice | None = None,
        kpar: ArrayLike | WaveVector | None = None,
        basis: FieldBasis | None = None,
        modetype: str | None = None,
        eta: complex = 0,
    ) -> None:
        super().__init__(lattice, kpar, basis, modetype, eta)

    @property
    @override
    def inv(self) -> Self:
        raise NotImplementedError("a periodic image sum has no inverse")

    def __call__(self, **kwargs: Any) -> NDArray[np.generic]:
        lattice, kpar, basis, modetype, eta = self._args
        if "basis" in kwargs:
            source = kwargs.pop("basis")
            basis = source if basis is None else (basis, source)
        inherited_lattice, inherited_kpar = (
            kwargs.pop("lattice", None),
            kwargs.pop("kpar", None),
        )
        lattice = inherited_lattice if lattice is None else lattice
        kpar = inherited_kpar if kpar is None else kpar
        source_type = kwargs.pop("modetype", None)
        if source_type is not None and not isinstance(modetype, tuple):
            destination, source = basis if isinstance(basis, tuple) else (basis, basis)
            if modetype is None:
                modetype = (
                    "up"
                    if isinstance(destination, PlaneWaveBasisByComp)
                    else "singular"
                    if isinstance(destination, CylindricalWaveBasis)
                    and isinstance(source, SphericalWaveBasis)
                    else "regular"
                )
            modetype = (modetype, source_type)
        eta = kwargs.pop("eta", eta)
        return expandlattice(
            lattice, kpar, basis=basis, modetype=modetype, eta=eta, **kwargs
        )

    @override
    def get_kwargs(self, obj: Any, dim: int = -1) -> dict[str, Any]:
        kwargs = super().get_kwargs(obj, dim)
        for name in ("lattice", "kpar"):
            value = getattr(obj, name, None)
            if value is not None:
                kwargs[name] = value
        return kwargs


class Permute(Operator, func=permute):
    """Cyclic plane-wave coordinate permutation."""

    def __init__(self, n: int = 1, *, isinv: bool = False) -> None:
        super().__init__(n, isinv=isinv)

    @override
    def _call_inv(self, **kwargs: Any) -> NDArray[np.generic]:
        kwargs["basis"] = kwargs["basis"].permute(self._args[0])
        return permute(-self._args[0], **kwargs)


class FieldOperator(Operator):
    """Cartesian field evaluation; no inverse from sampled fields is assumed."""

    def __init__(self, r: ArrayLike) -> None:
        super().__init__(np.array(r, dtype=np.float64, copy=True))

    @property
    @override
    def inv(self) -> Self:
        raise NotImplementedError("field evaluation has no general inverse")

    @override
    def __matmul__(self, other: Any) -> NDArray[np.generic]:
        # Existing wave objects evaluate weighted fields directly, without a
        # sample-by-mode matrix. Ordinary coefficient matrices use the operator.
        from ._array import PhysicsArray
        from ._plane import PlaneWave
        from ._source import MultipoleWave

        if isinstance(other, (PlaneWave, MultipoleWave)):
            return getattr(other, self.FUNC.__name__)(*self._args)
        if isinstance(other, PhysicsArray) and other.ndim == 1:
            return _field_samples(
                self.FUNC.__name__[0].upper(),
                self._args[-1],
                other.array,
                pol=self._args[0] if len(self._args) == 2 else 0,
                **self.get_kwargs(other),
            )
        return super().__matmul__(other)


class EField(FieldOperator, func=efield):
    """Electric field evaluation."""


class HField(FieldOperator, func=hfield):
    """Magnetic field evaluation."""


class DField(FieldOperator, func=dfield):
    """Displacement field evaluation."""


class BField(FieldOperator, func=bfield):
    """Magnetic flux density evaluation."""


class GField(FieldOperator, func=gfield):
    """Riemann-Silberstein field G, with explicit helicity and sample points."""

    def __init__(self, pol: int, r: ArrayLike) -> None:
        Operator.__init__(self, pol, np.array(r, dtype=np.float64, copy=True))


class FField(GField, func=ffield):
    """Riemann-Silberstein field F, with explicit helicity and sample points."""


class _WaveFields:
    """Shared weighted field evaluation for plane and multipole amplitudes."""

    if TYPE_CHECKING:

        @property
        def array(self) -> NDArray[np.complex128]: ...
        @property
        def basis(self) -> FieldBasis: ...

        k0: float
        material: Material
        modetype: str
        poltype: str

    def _field(
        self, kind: str, r: ArrayLike, coefficients: ArrayLike | None = None
    ) -> NDArray[np.complex128]:
        return _field(
            kind,
            r,
            self.basis,
            self.k0,
            self.material,
            self.modetype,
            self.poltype,
            self.array if coefficients is None else coefficients,
        )

    def efield(self, r: ArrayLike) -> NDArray[np.complex128]:
        """Cartesian electric samples (..., 3)."""
        return self._field("E", r)

    def hfield(self, r: ArrayLike) -> NDArray[np.complex128]:
        """Cartesian magnetic samples (..., 3)."""
        return self._field("H", r)

    def dfield(self, r: ArrayLike) -> NDArray[np.complex128]:
        """Cartesian displacement samples (..., 3)."""
        return self._field("D", r)

    def bfield(self, r: ArrayLike) -> NDArray[np.complex128]:
        """Cartesian flux-density samples (..., 3)."""
        return self._field("B", r)

    def gfield(self, pol: int, r: ArrayLike) -> NDArray[np.complex128]:
        """Weighted Riemann-Silberstein samples with the upstream normalization."""
        return _field_samples(
            "G",
            r,
            self.array,
            basis=self.basis,
            k0=self.k0,
            material=self.material,
            modetype=self.modetype,
            poltype=self.poltype,
            pol=pol,
        )

    def ffield(self, pol: int, r: ArrayLike) -> NDArray[np.complex128]:
        """Weighted Riemann-Silberstein F samples, including chiral index weights."""
        return _field_samples(
            "F",
            r,
            self.array,
            basis=self.basis,
            k0=self.k0,
            material=self.material,
            modetype=self.modetype,
            poltype=self.poltype,
            pol=pol,
        )


def _field_samples(
    kind: str,
    r: ArrayLike,
    coefficients: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
    pol: int = 0,
) -> NDArray[np.complex128]:
    poltype = _resolve_poltype(poltype)
    if kind not in ("G", "F"):
        return _field(kind, r, basis, k0, material, modetype, poltype, coefficients)
    medium = Material(material)
    electric, magnetic = _rs_weights(pol, basis, poltype)
    value = _field(
        "E",
        r,
        basis,
        k0,
        medium,
        modetype,
        poltype,
        np.asarray(coefficients) * electric,
    )
    if magnetic:
        value += (
            1j
            * medium.impedance
            * magnetic
            * _field("H", r, basis, k0, medium, modetype, poltype, coefficients)
        )
    if kind == "F" and poltype == "helicity":
        value *= medium.nmp[max(pol, 0)] / medium.n
    return value


class OperatorAttribute:
    """Bind an operator to one object; evaluated results are explicit arrays.

    Unlike a descriptor with shared mutable state, a saved bound attribute stays
    attached to its original object when another object's attribute is accessed.
    """

    def __init__(self, op: type[Operator], obj: Any = None) -> None:
        self._op, self._obj = op, obj

    @property
    def OP(self) -> type[Operator]:  # noqa: N802 - upstream operator interface
        return self._op

    def __get__(self, obj: Any, objtype: type | None = None) -> Self:
        return self if obj is None else type(self)(self.OP, obj)

    def _prepare(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], inverse: bool
    ) -> tuple[Operator, dict[str, Any]]:
        constructor = {
            key: kwargs.pop(key) for key in self.OP._parameters if key in kwargs
        }
        op = self.OP(*args, **constructor)
        if inverse:
            op = op.inv
        dim = -1 if inverse or np.ndim(self._obj) == 1 else -2
        inherited = op.get_kwargs(self._obj, dim)
        if "basis" in kwargs and "basis" in inherited:
            kwargs["basis"] = (kwargs["basis"], inherited["basis"])
        return op, inherited | kwargs

    def eval(self, *args: Any, **kwargs: Any) -> NDArray[np.generic]:
        op, metadata = self._prepare(args, kwargs, False)
        return op(**metadata)

    def eval_inv(self, *args: Any, **kwargs: Any) -> NDArray[np.generic]:
        op, metadata = self._prepare(args, kwargs, True)
        return op(**metadata)

    def apply_left(self, *args: Any, **kwargs: Any) -> NDArray[np.generic]:
        op, metadata = self._prepare(args, kwargs, False)
        if not kwargs and isinstance(op, FieldOperator):
            return op @ self._obj
        return op(**metadata) @ np.asarray(self._obj)

    def apply_right(self, *args: Any, **kwargs: Any) -> NDArray[np.generic]:
        return np.asarray(self._obj) @ self.eval_inv(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> NDArray[np.generic]:
        if np.ndim(self._obj) <= 1:
            return self.apply_left(*args, **kwargs)
        try:
            inverse = self.eval_inv(*args, **kwargs)
        except NotImplementedError:
            return self.apply_left(*args, **kwargs)
        return self.eval(*args, **kwargs) @ np.asarray(self._obj) @ inverse

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.OP.__name__})"

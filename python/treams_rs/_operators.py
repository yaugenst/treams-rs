"""Explicit basis operators backed by native numerical kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import diff
from ._core import (
    Material,
    PlaneWaveBasisByComp,
    PlaneWaveBasisByUnitVector,
    SphericalWaveBasis,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ._core import CylindricalWaveBasis, MaterialLike

    type Basis = SphericalWaveBasis | CylindricalWaveBasis
    type FieldBasis = Basis | PlaneWaveBasisByComp | PlaneWaveBasisByUnitVector


def rotate(
    phi: float, theta: float = 0, psi: float = 0, *, basis: Basis | tuple[Basis, Basis]
) -> NDArray[np.complex128]:
    """Rotation matrix in the z-y-z convention; basis origins remain fixed."""
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    return diff.rotation([phi, theta, psi], destination, source)[0]


def expand(
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    modetype: str | tuple[str, str] | None = None,
    *,
    k0: float,
    material: MaterialLike = 1,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Multipole expansion, including regular cylindrical-to-spherical waves.

    A basis pair is (destination, source). All explicit origin pairs are included.
    """
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    if isinstance(source, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        if not isinstance(destination, SphericalWaveBasis):
            raise ValueError(
                "plane expansion currently requires a spherical destination"
            )
        medium = Material(material)
        if not np.isfinite(k0) or k0 <= 0 or (poltype == "parity" and medium.ischiral):
            raise ValueError(
                "invalid frequency or embedding medium for the polarization type"
            )
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
        return diff.plane_expansion(
            destination,
            np.column_stack(source.kvecs(k0, medium, types[1])),
            source.pol,
            poltype=poltype,
            fixed_vectors=True,
        )[0]
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
    if not np.isfinite(k0) or k0 <= 0:
        raise ValueError("k0 must be positive and finite")
    if type(destination) is not type(source) and types != ("regular", "regular"):
        raise ValueError("cylindrical-to-spherical conversion requires regular waves")
    # Equal radial types use the regular addition theorem.
    return diff.expansion(
        destination,
        source,
        Material(material).ks(k0),
        poltype=poltype,
        singular=types == ("regular", "singular"),
    )[0]


def _field(
    kind: str,
    r: ArrayLike,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike,
    modetype: str | None,
    poltype: str,
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
    if isinstance(basis, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        value, _ = diff.plane_field(
            None,
            points.reshape(-1, 3),
            np.column_stack(basis.kvecs(k0, medium, modetype)),
            basis.pol,
            poltype=poltype,
            fixed_vectors=True,
        )
    else:
        value, _ = diff.field_operator(
            points.reshape(-1, 3),
            basis,
            medium.ks(k0),
            poltype=poltype,
            singular=modetype == "singular",
        )
    return value.reshape((*points.shape[:-1], 3, len(basis))) * weights


def efield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Cartesian electric-field operator (..., 3, modes)."""
    return _field("E", r, basis, k0, material, modetype, poltype)


def hfield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Magnetic-field operator in units of electric field / vacuum impedance."""
    return _field("H", r, basis, k0, material, modetype, poltype)


def dfield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Electric-displacement operator in units of vacuum permittivity times E."""
    return _field("D", r, basis, k0, material, modetype, poltype)


def bfield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Magnetic-flux operator in units of electric field / vacuum light speed."""
    return _field("B", r, basis, k0, material, modetype, poltype)

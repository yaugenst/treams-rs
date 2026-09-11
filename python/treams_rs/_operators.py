"""Explicit basis operators backed by native numerical kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import diff
from ._core import Material

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ._core import CylindricalWaveBasis, MaterialLike, SphericalWaveBasis

    type Basis = SphericalWaveBasis | CylindricalWaveBasis


def rotate(
    phi: float, theta: float = 0, psi: float = 0, *, basis: Basis | tuple[Basis, Basis]
) -> NDArray[np.complex128]:
    """Rotation matrix in the z-y-z convention; basis origins remain fixed."""
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    return diff.rotation([phi, theta, psi], destination, source)[0]


def _field(
    kind: str,
    r: ArrayLike,
    basis: Basis,
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
    modetype = "regular" if modetype is None else modetype
    if modetype not in ("regular", "singular"):
        raise ValueError("multipole fields require regular or singular waves")
    weights = np.ones(len(basis), dtype=np.complex128)
    if kind in ("H", "B"):
        weights *= -1j / medium.impedance
        if poltype == "helicity":
            weights *= 2 * basis.pol - 1
        else:
            basis = type(basis)(
                [(*mode[:3], 1 - mode[3]) for mode in basis.modes], basis.positions
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
    basis: Basis,
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
    basis: Basis,
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
    basis: Basis,
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
    basis: Basis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Magnetic-flux operator in units of electric field / vacuum light speed."""
    return _field("B", r, basis, k0, material, modetype, poltype)

"""Explicit multipole amplitudes, metadata and weighted native field evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._core import CylindricalWaveBasis, Material, MaterialLike, SphericalWaveBasis
from ._operators import _field, _rs_weights, changepoltype, expand

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NDArray

type Basis = SphericalWaveBasis | CylindricalWaveBasis


class MultipoleWave:
    """A multipole amplitude vector with explicit basis and physical metadata.

    Fields use weighted Rust kernels, avoiding a full sample-by-mode operator.
    Use .array for arithmetic or diff.field/advect.field for tracked parameters.
    """

    def __init__(
        self,
        coefficients: ArrayLike,
        *,
        basis: Basis,
        k0: float = 1.0,
        material: MaterialLike = 1,
        modetype: str = "regular",
        poltype: str = "helicity",
    ):
        self.array: NDArray[np.complex128] = np.array(
            coefficients, dtype=np.complex128, copy=True
        )
        if self.array.shape != (len(basis),) or not np.isfinite(self.array).all():
            raise ValueError("wave requires one finite amplitude per basis mode")
        if not np.isfinite(k0) or k0 <= 0:
            raise ValueError("wave requires positive finite k0")
        if modetype not in ("regular", "singular"):
            raise ValueError("multipole mode type must be regular or singular")
        self.material = Material(material)
        if poltype not in ("helicity", "parity") or (
            poltype == "parity" and self.material.ischiral
        ):
            raise ValueError("invalid polarization type for embedding medium")
        self.basis, self.k0, self.modetype, self.poltype = (
            basis,
            float(k0),
            modetype,
            poltype,
        )
        self.array.flags.writeable = False

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        return np.asarray(self.array, dtype=dtype, copy=copy)

    def changepoltype(self, poltype: str | None = None) -> MultipoleWave:
        """Express the same field in the other polarization convention."""
        poltype = (
            ("parity" if self.poltype == "helicity" else "helicity")
            if poltype is None
            else poltype
        )
        if poltype == self.poltype:
            return self
        change = changepoltype(poltype, basis=self.basis)
        if not np.all(np.count_nonzero(change, axis=0) == 2):
            raise ValueError(
                "polarization conversion requires both polarizations of each mode"
            )
        return type(self)(
            change @ self.array,
            basis=self.basis,
            k0=self.k0,
            material=self.material,
            modetype=self.modetype,
            poltype=poltype,
        )

    def expand(
        self, basis: Basis, *, modetype: str = "regular"
    ) -> NDArray[np.complex128]:
        """Coefficients in the destination basis; singular-to-regular uses outgoing translation."""
        if basis is self.basis and modetype == self.modetype:
            return self.array
        return (
            expand(
                (basis, self.basis),
                (modetype, self.modetype),
                k0=self.k0,
                material=self.material,
                poltype=self.poltype,
            )
            @ self.array
        )

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
        electric, magnetic = _rs_weights(pol, self.basis, self.poltype)
        value = self._field("E", r, self.array * electric)
        if magnetic:
            value += 1j * self.material.impedance * magnetic * self.hfield(r)
        return value

    def ffield(self, pol: int, r: ArrayLike) -> NDArray[np.complex128]:
        """Weighted Riemann-Silberstein F samples, including chiral index weights."""
        value = self.gfield(pol, r)
        if self.poltype == "helicity":
            value *= self.material.nmp[max(pol, 0)] / self.material.n
        return value


def spherical_wave(
    l: int,  # noqa: E741 - preserve the upstream multipole-degree argument
    m: int,
    pol: int,
    *,
    k0: float = 1.0,
    basis: SphericalWaveBasis | None = None,
    material: MaterialLike = 1,
    modetype: str = "regular",
    poltype: str = "helicity",
) -> MultipoleWave:
    """One spherical mode in a global basis (positive helicity is pol=1)."""
    basis = SphericalWaveBasis.default(l) if basis is None else basis
    if not basis.isglobal:
        raise ValueError("source basis must be global")
    index = basis.modes.index((int(basis.pidx[0]), l, m, pol))
    amplitudes = np.zeros(len(basis), complex)
    amplitudes[index] = 1
    return MultipoleWave(
        amplitudes,
        basis=basis,
        k0=k0,
        material=material,
        modetype=modetype,
        poltype=poltype,
    )


def cylindrical_wave(
    kz: float,
    m: int,
    pol: int,
    *,
    k0: float = 1.0,
    basis: CylindricalWaveBasis | None = None,
    material: MaterialLike = 1,
    modetype: str = "regular",
    poltype: str = "helicity",
) -> MultipoleWave:
    """One cylindrical mode in a global basis, with fixed real axial label kz."""
    basis = CylindricalWaveBasis.default([kz], abs(m)) if basis is None else basis
    if not basis.isglobal:
        raise ValueError("source basis must be global")
    index = basis.modes.index((int(basis.pidx[0]), kz, m, pol))
    amplitudes = np.zeros(len(basis), complex)
    amplitudes[index] = 1
    return MultipoleWave(
        amplitudes,
        basis=basis,
        k0=k0,
        material=material,
        modetype=modetype,
        poltype=poltype,
    )

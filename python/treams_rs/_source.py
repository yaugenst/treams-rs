"""Explicit multipole amplitudes, metadata and weighted native field evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np

from ._core import (
    CylindricalWaveBasis,
    Material,
    MaterialLike,
    PlaneWaveBasisByComp,
    PlaneWaveBasisByUnitVector,
    SphericalWaveBasis,
)
from ._operators import _field_samples, _WaveFields, changepoltype, expand
from .config import _resolve_poltype

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NDArray

type Basis = (
    SphericalWaveBasis
    | CylindricalWaveBasis
    | PlaneWaveBasisByComp
    | PlaneWaveBasisByUnitVector
)


class MultipoleWave(_WaveFields):
    """Physical wave amplitudes with explicit basis, medium and propagation kind.

    Fields use weighted Rust kernels, avoiding a full sample-by-mode operator.
    Use .coefficients for numerical work and framework namespaces for gradients.
    Coefficient batches retain one illumination per column.
    """

    basis: Basis
    array: NDArray[np.complex128]

    def __init__(
        self,
        coefficients: ArrayLike,
        *,
        basis: Basis,
        k0: float = 1.0,
        material: MaterialLike = 1,
        modetype: str = "regular",
        poltype: str | None = None,
        medium: MaterialLike | None = None,
        kind: str | None = None,
        polarization: str | None = None,
    ):
        """Construct wave amplitudes with explicit basis, medium and radial/directional kind.

        Kind is regular/outgoing for multipoles and up/down for plane ports.
        Coefficients have shape (modes,) or (modes, illuminations). Polarization
        is helicity/parity. Material, modetype and poltype retain the numerical
        kernel vocabulary for direct coefficient-level use.
        """
        if medium is not None:
            if Material(material) != Material():
                raise ValueError("specify medium only once")
            material = medium
        if kind is not None:
            if modetype != "regular":
                raise ValueError("specify wave kind only once")
            modetype = "singular" if kind == "outgoing" else kind
        if polarization is not None:
            if poltype is not None:
                raise ValueError("specify polarization convention only once")
            poltype = polarization
        poltype = _resolve_poltype(poltype)
        self.array = np.array(coefficients, dtype=np.complex128, copy=True)
        if (
            self.array.ndim not in (1, 2)
            or self.array.shape[0] != len(basis)
            or (self.array.ndim == 2 and self.array.shape[1] == 0)
            or not np.isfinite(self.array).all()
        ):
            raise ValueError(
                "wave coefficients require shape (modes,) or (modes, illuminations)"
            )
        if not np.isfinite(k0) or k0 <= 0:
            raise ValueError("wave requires positive finite k0")
        kinds = (
            ("up", "down")
            if isinstance(basis, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector))
            else ("regular", "singular")
        )
        if modetype not in kinds:
            raise ValueError(f"wave mode type must be one of {kinds}")
        self.material = Material(material)
        if poltype == "parity" and self.material.ischiral:
            raise ValueError("invalid polarization type for embedding medium")
        self.basis, self.k0, self.modetype, self.poltype = (
            basis,
            float(k0),
            modetype,
            poltype,
        )
        self.array.flags.writeable = False

    @property
    def coefficients(self) -> NDArray[np.complex128]:
        """Read-only amplitudes, shape (modes,) or (modes, illuminations)."""
        return self.array

    @property
    def medium(self) -> Material:
        """Homogeneous medium in which this wave is represented."""
        return self.material

    @property
    def polarization(self) -> str:
        """Polarization convention: helicity or parity."""
        return self.poltype

    @property
    def kind(self) -> str:
        """Regular/outgoing multipoles or up/down plane-port propagation."""
        return "outgoing" if self.modetype == "singular" else self.modetype

    def _samples(self, kind: str, r: ArrayLike, pol: int = 0) -> NDArray[np.complex128]:
        def sample(coefficients: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return _field_samples(
                kind,
                r,
                coefficients,
                basis=self.basis,
                k0=self.k0,
                material=self.material,
                modetype=self.modetype,
                poltype=self.poltype,
                pol=pol,
            )

        if self.array.ndim == 1:
            return sample(self.array)
        return np.stack([sample(column) for column in self.array.T], axis=-1)

    @override
    def _field(
        self, kind: str, r: ArrayLike, coefficients: ArrayLike | None = None
    ) -> NDArray[np.complex128]:
        if coefficients is not None:
            return _field_samples(
                kind,
                r,
                coefficients,
                basis=self.basis,
                k0=self.k0,
                material=self.material,
                modetype=self.modetype,
                poltype=self.poltype,
            )
        return self._samples(kind, r)

    @override
    def gfield(self, pol: int, r: ArrayLike) -> NDArray[np.complex128]:
        """G field samples (..., 3), or (..., 3, illuminations) for a batch."""
        return self._samples("G", r, pol)

    @override
    def ffield(self, pol: int, r: ArrayLike) -> NDArray[np.complex128]:
        """F field samples (..., 3), or (..., 3, illuminations) for a batch."""
        return self._samples("F", r, pol)

    def in_basis(self, basis: Basis, *, kind: str | None = None) -> MultipoleWave:
        """Represent this field in another basis, retaining physical metadata.

        Kind is regular/outgoing for multipoles, up/down for plane-wave ports.
        Omission preserves the current kind within a family; plane-to-multipole
        conversion produces regular waves. Truncated bases are approximations.
        """
        if kind is None:
            kind = (
                "regular"
                if isinstance(
                    self.basis, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)
                )
                and isinstance(basis, (SphericalWaveBasis, CylindricalWaveBasis))
                else self.modetype
            )
        elif kind == "outgoing":
            kind = "singular"
        return type(self)(
            self.expand(basis, modetype=kind),
            basis=basis,
            k0=self.k0,
            material=self.material,
            modetype=kind,
            poltype=self.poltype,
        )

    def with_polarization(self, polarization: str) -> MultipoleWave:
        """Represent the same physical wave in helicity or parity channels."""
        return self.changepoltype(polarization)

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


def spherical_wave(
    l: int,  # noqa: E741 - preserve the upstream multipole-degree argument
    m: int,
    pol: int,
    *,
    k0: float = 1.0,
    basis: SphericalWaveBasis | None = None,
    material: MaterialLike = 1,
    modetype: str = "regular",
    poltype: str | None = None,
    medium: MaterialLike | None = None,
    kind: str | None = None,
    polarization: str | None = None,
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
        medium=medium,
        kind=kind,
        polarization=polarization,
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
    poltype: str | None = None,
    medium: MaterialLike | None = None,
    kind: str | None = None,
    polarization: str | None = None,
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
        medium=medium,
        kind=kind,
        polarization=polarization,
    )

"""Spherical T-matrices with explicit arrays and familiar treams constructors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, override

import numpy as np

from . import diff
from ._core import CylindricalWaveBasis, Material, MaterialLike, SphericalWaveBasis
from ._plane import PlaneWave

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, DTypeLike, NDArray


class _TMatrix[B: (SphericalWaveBasis, CylindricalWaveBasis)]:
    """A matrix with physical metadata; use ``.array`` for arbitrary NumPy operations.

    Unlike an ndarray subclass, slicing or arithmetic never silently propagates
    physical metadata to a result with a different meaning.
    """

    _basis_type: type[B]

    def __init__(
        self,
        arr: ArrayLike,
        *,
        k0: float,
        basis: B | None = None,
        material: MaterialLike = 1,
        poltype: str = "helicity",
    ):
        self.array: NDArray[np.complex128] = np.array(
            arr, dtype=np.complex128, copy=True
        )
        if (
            self.array.ndim != 2
            or self.array.shape[0] != self.array.shape[1]
            or not np.isfinite(self.array).all()
        ):
            raise ValueError("T-matrix must be a finite square matrix")
        if not np.isfinite(k0) or k0 <= 0:
            raise ValueError("k0 must be finite and positive")
        self.basis = self._default_basis(len(self.array)) if basis is None else basis
        if not isinstance(self.basis, self._basis_type):
            raise ValueError("basis wave family does not match T-matrix type")
        if len(self.basis) != len(self.array):
            raise ValueError("basis dimension does not match matrix")
        self.k0 = float(k0)
        self.material = Material(material)
        if poltype not in ("helicity", "parity") or (
            poltype == "parity" and self.material.ischiral
        ):
            raise ValueError("invalid polarization type for embedding medium")
        self.poltype = poltype
        self.array.flags.writeable = False

    def _default_basis(self, dimension: int) -> B:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ks(self) -> NDArray[np.complex128]:
        return self.material.ks(self.k0)

    @property
    def isglobal(self) -> bool:
        return self.basis.isglobal

    def __len__(self) -> int:
        return len(self.array)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        return np.asarray(self.array, dtype=dtype, copy=copy)

    def __matmul__(self, other: ArrayLike | PlaneWave) -> NDArray[np.complex128]:
        return self.array @ self._incident(other)

    def _incident(self, value: ArrayLike | PlaneWave) -> NDArray[np.complex128]:
        if isinstance(value, PlaneWave):
            if (
                value.k0 != self.k0
                or value.material != self.material
                or value.poltype != self.poltype
            ):
                raise ValueError(
                    "illumination must have matching k0, material and polarization type"
                )
            return value.expand(self.basis)
        return np.asarray(value, dtype=np.complex128)

    @classmethod
    def cluster(cls, tmats: Sequence[Self], positions: ArrayLike) -> Self:
        if not tmats:
            raise ValueError("cluster must contain at least one T-matrix")
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (len(tmats), 3):
            raise ValueError("one Cartesian position required per T-matrix")
        first = tmats[0]
        dimension = sum(len(tm) for tm in tmats)
        value = np.zeros((dimension, dimension), dtype=np.complex128)
        modes: list[tuple[int, float, int, int]] = []
        offset = 0
        for particle, tm in enumerate(tmats):
            if (
                not isinstance(tm.basis, cls._basis_type)
                or not tm.isglobal
                or tm.k0 != first.k0
                or tm.material != first.material
                or tm.poltype != first.poltype
            ):
                raise ValueError(
                    "cluster requires global matrices with the same k0, material and polarization type"
                )
            end = offset + len(tm)
            value[offset:end, offset:end] = tm.array
            modes.extend(
                (particle, degree, order, pol) for _, degree, order, pol in tm.basis
            )
            offset = end
        return cls(
            value,
            k0=first.k0,
            material=first.material,
            poltype=first.poltype,
            basis=type(first.basis)(modes, positions),
        )

    @property
    def interaction(self) -> _Interaction[Self]:
        return _Interaction(self)

    def changepoltype(self, poltype: str | None = None) -> Self:
        poltype = (
            ("parity" if self.poltype == "helicity" else "helicity")
            if poltype is None
            else poltype
        )
        if poltype == self.poltype:
            return self
        if poltype not in ("helicity", "parity"):
            raise ValueError("invalid polarization type")
        modes = np.array(self.basis.modes)
        same = np.all(modes[:, None, :3] == modes[None, :, :3], axis=-1)
        signs = np.where((modes[:, None, 3] == 0) & (modes[None, :, 3] == 0), -1, 1)
        change = same * signs * np.sqrt(0.5)
        if not np.all(same.sum(axis=0) == 2):
            raise ValueError(
                "polarization change requires both polarizations of each mode"
            )
        return type(self)(
            change @ self.array @ change.T,
            k0=self.k0,
            basis=self.basis,
            material=self.material,
            poltype=poltype,
        )

    def expand(self, basis: B) -> Self:
        """Express outgoing and regular channels in another spherical basis."""
        outgoing, _ = diff.expansion(basis, self.basis, self.ks, poltype=self.poltype)
        incident, _ = diff.expansion(self.basis, basis, self.ks, poltype=self.poltype)
        return type(self)(
            outgoing @ self.array @ incident,
            k0=self.k0,
            basis=basis,
            material=self.material,
            poltype=self.poltype,
        )

    def _propagating_ks(self) -> NDArray[np.float64]:
        ks = self.ks
        if not self.material.isreal or np.any(ks.imag != 0) or np.any(ks.real == 0):
            raise NotImplementedError(
                "cross sections require a nonabsorbing propagating embedding medium"
            )
        return ks.real[self.basis.pol]

    def _cross_sections(
        self, illu: ArrayLike | PlaneWave, flux: float, power: int, factor: float
    ) -> tuple[float, float]:
        incident = self._incident(illu)
        if (
            incident.shape != (len(self),)
            or not np.isfinite(incident).all()
            or not np.isfinite(flux)
            or flux <= 0
        ):
            raise ValueError("require finite incident coefficients and positive flux")
        scattered = self @ incident
        weighted = scattered / self._propagating_ks() ** power
        overlap, _ = diff.expansion(
            self.basis, self.basis, self.ks, poltype=self.poltype
        )
        return float(
            np.vdot(scattered, overlap @ weighted).real * factor / flux
        ), float(-np.vdot(incident, weighted).real * factor / flux)


class _Interaction[M: _TMatrix[Any]]:
    def __init__(self, matrix: M):
        self.matrix = matrix

    def _coupling(self) -> NDArray[np.complex128]:
        tm = self.matrix
        return diff.expansion(
            tm.basis, tm.basis, tm.ks, poltype=tm.poltype, singular=True
        )[0]

    def __call__(self) -> NDArray[np.complex128]:
        return (
            np.eye(len(self.matrix), dtype=np.complex128)
            - self.matrix.array @ self._coupling()
        )

    def solve(self) -> M:
        tm = self.matrix
        result, _ = diff.interaction(tm.array, self._coupling())
        return type(tm)(
            result, k0=tm.k0, basis=tm.basis, material=tm.material, poltype=tm.poltype
        )


class TMatrix(_TMatrix[SphericalWaveBasis]):
    """Spherical T-matrix with a native scattering and adjoint core."""

    _basis_type = SphericalWaveBasis

    @override
    def _default_basis(self, dimension: int) -> SphericalWaveBasis:
        return SphericalWaveBasis.default(SphericalWaveBasis.defaultlmax(dimension))

    @classmethod
    def sphere(
        cls,
        lmax: int,
        k0: float,
        radii: ArrayLike,
        materials: Sequence[MaterialLike],
        poltype: str = "helicity",
    ) -> TMatrix:
        layers = [Material(m) for m in materials]
        if not layers:
            raise ValueError("sphere requires layer materials and an embedding medium")
        value, _ = diff.sphere(
            lmax,
            k0,
            np.atleast_1d(radii),
            [m.epsilon for m in layers],
            [m.mu for m in layers],
            [m.kappa for m in layers],
        )
        result = cls(value, k0=k0, material=layers[-1])
        return result if poltype == "helicity" else result.changepoltype(poltype)

    @property
    def xs_ext_avg(self) -> float:
        if not self.isglobal:
            raise NotImplementedError("expand to a global basis before averaging")
        ks = self._propagating_ks()
        return float(-2 * np.pi * np.sum(np.diag(self.array).real / ks**2))

    @property
    def xs_sca_avg(self) -> float:
        if not self.isglobal:
            raise NotImplementedError("expand to a global basis before averaging")
        ks = self._propagating_ks()
        return float(2 * np.pi * np.sum(abs(self.array / ks[:, None]) ** 2))

    def xs(self, illu: ArrayLike | PlaneWave, flux: float = 0.5) -> tuple[float, float]:
        """Scattering and extinction for incident spherical coefficients."""
        return self._cross_sections(illu, flux, 2, 0.5)


class TMatrixC(_TMatrix[CylindricalWaveBasis]):
    """Cylindrical T-matrix with fixed axial mode labels."""

    _basis_type = CylindricalWaveBasis

    @override
    def _default_basis(self, dimension: int) -> CylindricalWaveBasis:
        return CylindricalWaveBasis.default(
            [0], CylindricalWaveBasis.defaultmmax(dimension)
        )

    @classmethod
    def cylinder(
        cls,
        kzs: ArrayLike,
        mmax: int,
        k0: float,
        radii: ArrayLike,
        materials: Sequence[MaterialLike],
        poltype: str = "helicity",
    ) -> TMatrixC:
        layers = [Material(m) for m in materials]
        if not layers:
            raise ValueError(
                "cylinder requires layer materials and an embedding medium"
            )
        axial = np.atleast_1d(kzs)
        value, _ = diff.cylinder(
            axial,
            mmax,
            k0,
            np.atleast_1d(radii),
            [m.epsilon for m in layers],
            [m.mu for m in layers],
            [m.kappa for m in layers],
        )
        result = cls(
            value,
            k0=k0,
            basis=CylindricalWaveBasis.default(axial, mmax),
            material=layers[-1],
        )
        return result if poltype == "helicity" else result.changepoltype(poltype)

    @property
    def krhos(self) -> NDArray[np.complex128]:
        values = np.sqrt(self.ks[self.basis.pol] ** 2 - self.basis.kz**2)
        return np.where(values.imag < 0, -values, values)

    @property
    def xw_ext_avg(self) -> float:
        if not self.isglobal:
            raise NotImplementedError("expand to a global basis before averaging")
        return float(
            -2
            * np.sum(np.diag(self.array).real / self._propagating_ks())
            / len(np.unique(self.basis.kz))
        )

    @property
    def xw_sca_avg(self) -> float:
        if not self.isglobal:
            raise NotImplementedError("expand to a global basis before averaging")
        return float(
            2
            * np.sum(abs(self.array) ** 2 / self._propagating_ks()[:, None])
            / len(np.unique(self.basis.kz))
        )

    def xw(self, illu: ArrayLike | PlaneWave, flux: float = 0.5) -> tuple[float, float]:
        """Scattering and extinction widths for incident cylindrical coefficients."""
        return self._cross_sections(illu, flux, 1, 2.0)

"""Spherical T-matrices with explicit arrays and familiar treams constructors."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Any, Self, cast, override

import numpy as np

from . import _operators as op
from . import diff, lattice
from ._core import CylindricalWaveBasis, Material, MaterialLike, SphericalWaveBasis
from ._operators import Operator, changepoltype, expandlattice
from ._plane import PlaneWave
from ._source import MultipoleWave
from .config import _resolve_poltype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from . import _native


class _TMatrix[B: (SphericalWaveBasis, CylindricalWaveBasis)]:
    """A matrix with physical metadata; use ``.array`` for arbitrary NumPy operations.

    Unlike an ndarray subclass, slicing or arithmetic never silently propagates
    physical metadata to a result with a different meaning.
    """

    _basis_type: type[B]
    modetype = ("singular", "regular")
    translate = op.OperatorAttribute(op.Translate)
    expandlattice = op.OperatorAttribute(op.ExpandLattice)
    permute = op.OperatorAttribute(op.Permute)
    efield = op.OperatorAttribute(op.EField)
    hfield = op.OperatorAttribute(op.HField)
    dfield = op.OperatorAttribute(op.DField)
    bfield = op.OperatorAttribute(op.BField)
    gfield = op.OperatorAttribute(op.GField)
    ffield = op.OperatorAttribute(op.FField)

    def __init__(
        self,
        arr: ArrayLike,
        *,
        k0: float,
        basis: B | None = None,
        material: MaterialLike = 1,
        poltype: str | None = None,
    ):
        poltype = _resolve_poltype(poltype)
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
        self._cluster_sizes: tuple[int, ...] | None = None

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

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, self._basis_type):
            indices = [self.basis.modes.index(mode) for mode in key]
            return type(self)(
                self.array[np.ix_(indices, indices)],
                basis=cast("B", self.basis[indices]),
                k0=self.k0,
                material=self.material,
                poltype=self.poltype,
            )
        return self.array[key]

    def valid_points(self, grid: ArrayLike, radii: ArrayLike) -> NDArray[np.bool_]:
        """Points strictly outside the enclosing spheres or infinite cylinders."""
        points, radii = np.asarray(grid, dtype=float), np.asarray(radii, dtype=float)
        dim = 2 if isinstance(self.basis, CylindricalWaveBasis) else 3
        if (
            points.ndim == 0
            or points.shape[-1] not in (dim, 3)
            or not np.isfinite(points).all()
        ):
            raise ValueError("grid requires finite Cartesian points")
        if (
            radii.shape != (len(self.basis.positions),)
            or not np.isfinite(radii).all()
            or np.any(radii < 0)
        ):
            raise ValueError(
                "one finite nonnegative radius required per expansion origin"
            )
        valid = np.ones(points.shape[:-1], dtype=bool)
        for radius, position in zip(radii, self.basis.positions, strict=True):
            valid &= (
                np.sum((points[..., :dim] - position[:dim]) ** 2, axis=-1) > radius**2
            )
        return valid

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        return np.asarray(self.array, dtype=dtype, copy=copy)

    def __matmul__(
        self, other: ArrayLike | PlaneWave | MultipoleWave | Operator
    ) -> NDArray[np.complex128]:
        if isinstance(other, Operator):
            return NotImplemented
        return self.array @ self._incident(other)

    def _incident(
        self, value: ArrayLike | PlaneWave | MultipoleWave
    ) -> NDArray[np.complex128]:
        if isinstance(value, (PlaneWave, MultipoleWave)):
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
        result = cls(
            value,
            k0=first.k0,
            material=first.material,
            poltype=first.poltype,
            basis=type(first.basis)(modes, positions),
        )
        result._cluster_sizes = tuple(len(tm) for tm in tmats)
        return result

    @property
    def interaction(self) -> _Interaction[Self]:
        return _Interaction(self)

    @property
    def latticeinteraction(self) -> _PeriodicInteraction[Self]:
        return _PeriodicInteraction(self)

    def changepoltype(self, poltype: str | None = None) -> Self:
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

    def rotate(self, phi: float, theta: float = 0, psi: float = 0) -> Self:
        """Rotate the local multipole channels at their fixed expansion origins."""
        rotation, _ = diff.rotation([phi, theta, psi], self.basis)
        inverse, _ = diff.rotation([-psi, -theta, -phi], self.basis)
        return type(self)(
            rotation @ self.array @ inverse,
            k0=self.k0,
            basis=self.basis,
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

    def _cross_section_weights(self, power: int) -> NDArray[np.float64]:
        ks = self._propagating_ks()
        weights = 1 / ks**power
        if isinstance(self.basis, CylindricalWaveBasis):
            if np.any(abs(self.basis.kz) == abs(ks)):
                raise ValueError("cross widths are undefined at a diffraction cutoff")
            weights[abs(self.basis.kz) > abs(ks)] = 0
        return weights

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
        weights = self._cross_section_weights(power)
        if np.any(incident[weights == 0] != 0):
            raise ValueError("cross widths require propagating incident modes")
        scattered = self @ incident
        weighted = scattered * weights
        if self.isglobal:
            radiated = weighted
        else:
            overlap, _ = diff.expansion(
                self.basis, self.basis, self.ks, poltype=self.poltype
            )
            radiated = overlap @ weighted
        return float(np.vdot(scattered, radiated).real * factor / flux), float(
            -np.vdot(incident, weighted).real * factor / flux
        )


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
        if tm._cluster_sizes is not None:
            local = []
            bases = []
            offset = 0
            for size in tm._cluster_sizes:
                end = offset + size
                local.append(tm.array[offset:end, offset:end])
                bases.append(
                    type(tm.basis)(mode[1:] for mode in tm.basis.modes[offset:end])
                )
                offset = end
            result, _ = diff.particle_cluster(
                local,
                tm.basis.positions,
                tm.ks,
                bases=bases,
                poltype=tm.poltype,
            )
        else:
            result, _ = diff.interaction(tm.array, self._coupling())
        return type(tm)(
            result, k0=tm.k0, basis=tm.basis, material=tm.material, poltype=tm.poltype
        )

    def factor(self) -> _native.InteractionFactor:
        """Factor once for repeated incident-field solves without a full response matrix.

        The returned factor's solve/record methods accept complex128 arrays with
        one incident illumination per column.
        """
        tm = self.matrix
        coupling = self._coupling()
        if tm._cluster_sizes is None:
            return diff.factor_interaction(tm.array, coupling)
        offsets = np.cumsum((0, *tm._cluster_sizes))
        blocks = [tm.array[a:b, a:b] for a, b in pairwise(offsets)]
        return diff.factor_interaction_blocks(blocks, coupling)

    def illuminate(
        self, incident: ArrayLike | PlaneWave | MultipoleWave
    ) -> NDArray[np.complex128]:
        """Compute only requested scattered coefficients, preserving vector/batch shape."""
        values = self.matrix._incident(incident)
        vector = values.ndim == 1
        result = self.factor().solve(values[:, None] if vector else values)
        return result[:, 0] if vector else result


class _PeriodicInteraction[M: _TMatrix[Any]]:
    def __init__(self, matrix: M):
        self.matrix = matrix

    def coupling(
        self, a: ArrayLike, kpar: ArrayLike, *, eta: complex = 0
    ) -> NDArray[np.complex128]:
        tm = self.matrix
        return lattice.expansion(
            tm.basis, tm.basis, tm.ks, a, kpar, poltype=tm.poltype, eta=eta
        )

    def __call__(
        self, a: ArrayLike, kpar: ArrayLike, *, eta: complex = 0
    ) -> NDArray[np.complex128]:
        return np.eye(
            len(self.matrix), dtype=np.complex128
        ) - self.matrix.array @ self.coupling(a, kpar, eta=eta)

    def solve(
        self, a: ArrayLike, kpar: ArrayLike, *, eta: complex = 0
    ) -> NDArray[np.complex128]:
        """Effective periodic response in local multipole channels.

        This is not an isolated-particle T-matrix; finite-particle cross-section
        formulae do not apply. Use its array with incident channel coefficients.
        """
        return diff.interaction(self.matrix.array, self.coupling(a, kpar, eta=eta))[0]

    def factor(
        self, a: ArrayLike, kpar: ArrayLike, *, eta: complex = 0
    ) -> _native.InteractionFactor:
        """Reusable periodic factor for requested incident channel coefficients."""
        return diff.factor_interaction(
            self.matrix.array, self.coupling(a, kpar, eta=eta)
        )

    def illuminate(
        self,
        incident: ArrayLike | PlaneWave | MultipoleWave,
        a: ArrayLike,
        kpar: ArrayLike,
        *,
        eta: complex = 0,
    ) -> NDArray[np.complex128]:
        """Compute requested periodic response columns without the full effective T matrix."""
        values = self.matrix._incident(incident)
        vector = values.ndim == 1
        result = self.factor(a, kpar, eta=eta).solve(
            values[:, None] if vector else values
        )
        return result[:, 0] if vector else result


class TMatrix(_TMatrix[SphericalWaveBasis]):
    """Spherical T-matrix with a native scattering and adjoint core."""

    _basis_type = SphericalWaveBasis

    def _metric(self, kind: str) -> float:
        if not self.isglobal or self.poltype != "helicity":
            raise NotImplementedError("metric requires a global helicity T-matrix")
        ks = np.ones(2)
        if kind == "cd":
            self._propagating_ks()
            ks = self.ks.real
        return diff.tmatrix_metric(
            self.array, ks, polarizations=self.basis.pol, kind=kind
        )[0]

    @property
    def cd(self) -> float:
        """Rotationally averaged absorption circular dichroism."""
        return self._metric("cd")

    @property
    def db(self) -> float:
        """Fraction of scattering that changes helicity (duality breaking)."""
        return self._metric("db")

    @property
    def chi(self) -> float:
        """Normalized electromagnetic chirality from helicity-block singular values."""
        return self._metric("chi")

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
        poltype: str | None = None,
    ) -> TMatrix:
        poltype = _resolve_poltype(poltype)
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

    @classmethod
    def from_array(
        cls,
        tm: TMatrix,
        basis: CylindricalWaveBasis,
        *,
        lattice: ArrayLike,
        kpar: ArrayLike,
        eta: complex = 0,
    ) -> TMatrixC:
        """Solve a spherical 1D z-periodic unit cell in cylindrical channels.

        ``tm`` contains the uncoupled particles of one cell. The cylindrical axial
        wavenumbers must equal ``kpar + 2*pi*n/period`` for integer orders n.
        """
        if not isinstance(tm, TMatrix):
            raise ValueError("from_array requires a spherical unit cell")
        outgoing = expandlattice(
            lattice,
            kpar,
            basis=(basis, tm.basis),
            k0=tm.k0,
            material=tm.material,
            poltype=tm.poltype,
        )
        incident = diff.expansion(tm.basis, basis, tm.ks, poltype=tm.poltype)[0]
        response = tm.latticeinteraction.solve(lattice, kpar, eta=eta)
        return cls(
            outgoing @ response @ incident,
            k0=tm.k0,
            basis=basis,
            material=tm.material,
            poltype=tm.poltype,
        )

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
        poltype: str | None = None,
    ) -> TMatrixC:
        poltype = _resolve_poltype(poltype)
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

    def _averaging_weights(self) -> NDArray[np.float64]:
        if not self.isglobal:
            raise NotImplementedError("expand to a global basis before averaging")
        weights = self._cross_section_weights(1)
        channels = np.column_stack((self.basis.kz, self.basis.pol))[weights != 0]
        count = len(np.unique(channels, axis=0))
        if count == 0:
            raise ValueError("cross-width average requires propagating incident modes")
        return weights * (4 / count)

    @property
    def xw_ext_avg(self) -> float:
        """Mean over azimuth and the represented propagating (kz, pol) channels."""
        return float(-np.sum(np.diag(self.array).real * self._averaging_weights()))

    @property
    def xw_sca_avg(self) -> float:
        """Mean radiated width over the same incoming ensemble as xw_ext_avg."""
        weights = self._averaging_weights()
        return float(np.sum(abs(self.array[:, weights != 0]) ** 2 * weights[:, None]))

    def xw(self, illu: ArrayLike | PlaneWave, flux: float = 0.5) -> tuple[float, float]:
        """Radiated and extinguished widths for propagating incident coefficients.

        Evanescent outgoing orders remain in the solution but carry no far-field
        power. Evanescent illumination has no incident far-field flux and is not
        supported by this cross-width normalization.
        """
        return self._cross_sections(illu, flux, 1, 2.0)

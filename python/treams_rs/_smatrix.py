"""Planar interfaces and stable layer composition with native numerical kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native, coeffs, diff
from ._core import (
    Material,
    MaterialLike,
    PlaneWaveBasisByComp,
)
from ._operators import _periodic_channels, changepoltype, efield, hfield
from ._plane import PlaneWave

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import ArrayLike, NDArray

    from ._tmatrix import TMatrix, TMatrixC


class SMatrices:
    """Four scattering blocks indexed by outgoing and incoming up/down direction.

    ``material`` is ordered (positive side, negative side) along the basis normal.
    Interface/slab inputs run from the negative side to the positive side.
    Arrays have shape (2, 2, n, n). Use ``.array`` for arbitrary array operations.
    """

    def __init__(
        self,
        smats: ArrayLike,
        *,
        k0: float,
        basis: PlaneWaveBasisByComp,
        material: MaterialLike | tuple[MaterialLike, MaterialLike] = 1,
        poltype: str = "helicity",
    ):
        self.array: NDArray[np.complex128] = np.array(
            smats, dtype=np.complex128, copy=True, order="C"
        )
        if (
            self.array.shape != (2, 2, len(basis), len(basis))
            or not np.isfinite(self.array).all()
        ):
            raise ValueError(
                "S matrices require finite shape (2, 2, len(basis), len(basis))"
            )
        if not np.isfinite(k0) or k0 <= 0:
            raise ValueError("k0 must be finite and positive")
        if isinstance(material, tuple):
            if len(material) != 2:
                raise ValueError(
                    "material requires the media above and below the S matrix"
                )
            self.material = (Material(material[0]), Material(material[1]))
        else:
            self.material = (Material(material), Material(material))
        if poltype not in ("helicity", "parity") or (
            poltype == "parity" and any(m.ischiral for m in self.material)
        ):
            raise ValueError("invalid polarization type for embedding media")
        self.poltype, self.k0, self.basis = poltype, float(k0), basis
        self.array.flags.writeable = False

    def __getitem__(
        self, key: int | str | tuple[int | str, int | str]
    ) -> NDArray[np.complex128]:
        keys = {0: 0, 1: 1, "up": 0, "down": 1}
        if isinstance(key, tuple):
            return self.array[keys[key[0]], keys[key[1]]]
        return self.array[keys[key]]

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[NDArray[np.complex128]]:
        return iter(self.array)

    @classmethod
    def interface(
        cls,
        basis: PlaneWaveBasisByComp,
        k0: float,
        materials: Sequence[MaterialLike],
        poltype: str = "helicity",
    ) -> SMatrices:
        if len(materials) != 2:
            raise ValueError("an interface requires two materials, below then above")
        below, above = (Material(m) for m in materials)
        if poltype not in ("helicity", "parity") or (
            poltype == "parity" and (below.ischiral or above.ischiral)
        ):
            raise ValueError("invalid polarization type for embedding media")
        ks = np.array([below.ks(k0), above.ks(k0)])
        zs = np.array([below.impedance, above.impedance])
        result = np.zeros((2, 2, len(basis), len(basis)), dtype=np.complex128)
        for kx, ky in dict.fromkeys((kx, ky) for kx, ky, _ in basis):
            kz = np.array([m.kzs(k0, kx, ky) for m in (below, above)])
            value = (
                coeffs.fresnel(ks, kz, zs)
                if basis.alignment == "xy"
                else diff.interface(
                    ks, zs, [kx, ky], alignment=basis.alignment, fixed_q=True
                )[0]
            )
            if poltype == "parity":
                change = np.array([[-1, 1], [1, 1]]) * np.sqrt(0.5)
                value = change @ value @ change.T
            indices = [
                (i, pol) for i, (x, y, pol) in enumerate(basis) if (x, y) == (kx, ky)
            ]
            for i, pol in indices:
                for j, pol2 in indices:
                    result[:, :, i, j] = value[:, :, pol, pol2]
        return cls(result, basis=basis, k0=k0, material=(above, below), poltype=poltype)

    @classmethod
    def from_array(
        cls,
        tm: TMatrix | TMatrixC,
        basis: PlaneWaveBasisByComp,
        *,
        lattice: ArrayLike,
        kpar: ArrayLike,
        eta: complex = 0,
    ) -> SMatrices:
        """Solve an uncoupled unit cell and radiate into matching plane-wave ports.

        Spherical arrays use a 2D xy cell. Cylindrical arrays use a 1D period along
        x and zx-aligned ports, radiating toward positive/negative y.
        """
        channels = _periodic_channels(tm.basis, basis, tm.ks, lattice, kpar, tm.poltype)
        response = tm.latticeinteraction.solve(lattice, kpar, eta=eta)
        value, _ = diff.smatrix_from_array(response, channels)
        return cls(
            value, basis=basis, k0=tm.k0, material=tm.material, poltype=tm.poltype
        )

    @classmethod
    def propagation(
        cls,
        r: ArrayLike,
        basis: PlaneWaveBasisByComp,
        k0: float,
        material: MaterialLike = 1,
        poltype: str = "helicity",
    ) -> SMatrices:
        axis = basis.normal_axis
        distance = np.asarray(r, dtype=np.float64)
        if distance.ndim == 0:
            distance = np.eye(3)[axis] * float(distance)
        if distance.shape != (3,):
            raise ValueError("propagation requires a scalar or Cartesian displacement")
        axes = [(axis + 1) % 3, (axis + 2) % 3, axis]
        vectors = np.column_stack(basis.kvecs(k0, material))
        value, _ = diff.propagation(vectors[:, axes], distance[axes])
        return cls(
            value, basis=basis, k0=k0, material=Material(material), poltype=poltype
        )

    def add(self, upper: SMatrices) -> SMatrices:
        self._check_adjacent(upper)
        value, _ = diff.smatrix_add(self.array, upper.array)
        return type(self)(
            value,
            basis=self.basis,
            k0=self.k0,
            material=(upper.material[0], self.material[1]),
            poltype=self.poltype,
        )

    def _check_adjacent(self, upper: SMatrices) -> None:
        if (
            self.k0 != upper.k0
            or self.poltype != upper.poltype
            or self.basis.modes != upper.basis.modes
            or self.basis.alignment != upper.basis.alignment
            or self.material[0] != upper.material[1]
        ):
            raise ValueError(
                "coupled S matrices must match basis, k0, polarization and internal medium"
            )

    @classmethod
    def stack(cls, items: Sequence[SMatrices]) -> SMatrices:
        if not items:
            raise ValueError("stack requires at least one S matrix")
        result = items[0]
        for item in items[1:]:
            result = result.add(item)
        return result

    @classmethod
    def slab(
        cls,
        thickness: ArrayLike,
        basis: PlaneWaveBasisByComp,
        k0: float,
        materials: Sequence[MaterialLike],
        poltype: str = "helicity",
    ) -> SMatrices:
        values = np.atleast_1d(np.asarray(thickness, dtype=np.float64))
        if (
            len(materials) != len(values) + 2
            or not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError(
                "slabs require nonnegative thicknesses and two exterior materials"
            )
        groups: dict[tuple[float, float], list[int]] = {}
        for i, (x, y, _) in enumerate(basis):
            groups.setdefault((x, y), []).append(i)
        if all(len(indices) == 2 for indices in groups.values()):
            media = [Material(m) for m in materials]
            if poltype not in ("helicity", "parity") or (
                poltype == "parity" and any(m.ischiral for m in media)
            ):
                raise ValueError("invalid polarization type for embedding media")
            compact, _ = diff.layer_stack(
                [m.ks(k0) for m in media],
                [m.impedance for m in media],
                list(groups),
                values,
                alignment=basis.alignment,
                fixed_q=True,
            )
            if poltype == "parity":
                change = np.array([[-1, 1], [1, 1]]) * np.sqrt(0.5)
                compact = change @ compact @ change.T
            array = np.zeros((2, 2, len(basis), len(basis)), complex)
            for q, indices in enumerate(groups.values()):
                index = np.array(indices)
                pol = basis.pol[index]
                array[:, :, index[:, None], index] = compact[q][:, :, pol[:, None], pol]
            return cls(
                array,
                k0=k0,
                basis=basis,
                material=(media[-1], media[0]),
                poltype=poltype,
            )
        # A partial polarization basis retains its existing projected-step semantics.
        result = cls.interface(basis, k0, materials[:2], poltype)
        for d, lower, upper in zip(values, materials[1:-1], materials[2:], strict=True):
            result = result.add(cls.propagation(d, basis, k0, lower, poltype)).add(
                cls.interface(basis, k0, (lower, upper), poltype)
            )
        return result

    def double(self, n: int = 1) -> SMatrices:
        if n < 0:
            raise ValueError("doubling count must be nonnegative")
        result = self
        for _ in range(n):
            result = result.add(result)
        return result

    def changepoltype(self, poltype: str | None = None) -> SMatrices:
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
                "polarization conversion requires both polarizations per direction"
            )
        return type(self)(
            change @ self.array @ change.T,
            basis=self.basis,
            k0=self.k0,
            material=self.material,
            poltype=poltype,
        )

    def _incident(
        self, illu: ArrayLike | PlaneWave, modetype: str
    ) -> NDArray[np.complex128]:
        if isinstance(illu, PlaneWave):
            medium = self.material[1 if modetype == "up" else 0]
            if (
                illu.material != medium
                or illu.k0 != self.k0
                or illu.poltype != self.poltype
                or _direction(illu, None, self.basis.normal_axis) != modetype
            ):
                raise ValueError(
                    "illumination must match incident medium, k0, polarization and direction"
                )
            return illu.expand(self.basis)
        return np.asarray(illu, dtype=np.complex128)

    def illuminate(
        self,
        illu: ArrayLike | PlaneWave,
        illu2: ArrayLike | PlaneWave | None = None,
        *,
        modetype: str | None = None,
        smat: SMatrices | None = None,
    ) -> tuple[NDArray[np.complex128], ...]:
        """Outgoing fields, and optionally internal fields below an adjacent stack.

        With ``smat``, return outgoing up/down and internal up/down coefficients.
        Incident PlaneWave metadata refers to the outer media of the combined pair.
        """
        upper = self if smat is None else smat
        if smat is not None:
            self._check_adjacent(smat)
        modetype = _direction(illu, modetype, self.basis.normal_axis)
        first = (self if modetype == "up" else upper)._incident(illu, modetype)
        second = (
            np.zeros_like(first)
            if illu2 is None
            else (upper if modetype == "up" else self)._incident(
                illu2, "down" if modetype == "up" else "up"
            )
        )
        if (
            first.shape != (len(self.basis),)
            or second.shape != first.shape
            or not np.isfinite(first).all()
            or not np.isfinite(second).all()
        ):
            raise ValueError(
                "illumination requires one finite amplitude per basis mode"
            )
        up, down = (first, second) if modetype == "up" else (second, first)
        if smat is not None:
            fields = _native.smatrix_illuminate_forward(
                self.array, smat.array, up[:, None], down[:, None]
            )
            return tuple(fields[:, :, 0])
        return self[0, 0] @ up + self[0, 1] @ down, self[1, 0] @ up + self[1, 1] @ down

    def periodic(self) -> NDArray[np.complex128]:
        """Transfer matrix for repeating a cell along the basis normal.

        The outer media must match. Strongly evanescent channels can make a
        transfer matrix ill-conditioned; finite-stack composition remains in S form.
        """
        if self.material[0] != self.material[1]:
            raise ValueError("periodic repetition requires matching outer media")
        return diff.smatrix_periodic(self.array)[0]

    def bands_kz(
        self, az: float
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Bloch wavenumbers along the basis normal and their right eigenvectors.

        The name preserves treams' z-normal convention; yz/zx bases use x/y.
        ``az`` is the positive cell length along that normal.
        """
        if self.material[0] != self.material[1]:
            raise ValueError("periodic repetition requires matching outer media")
        return diff.bands(self.array, az)[0]

    def tr(
        self, illu: ArrayLike | PlaneWave, *, modetype: str | None = None
    ) -> tuple[float, float]:
        modetype = _direction(illu, modetype, self.basis.normal_axis)
        outgoing = self.illuminate(illu, modetype=modetype)
        transmission = 0 if modetype == "up" else 1
        reflection = 1 - transmission
        sign = 1 if modetype == "up" else -1
        trans, refl = outgoing[transmission], outgoing[reflection]
        source = _power_forms(
            self.basis, self.k0, self.material[reflection], self.poltype
        )
        target = (
            source
            if self.material[0] == self.material[1]
            else _power_forms(
                self.basis, self.k0, self.material[transmission], self.poltype
            )
        )
        incident = self._incident(illu, modetype)
        flux = sign * (
            np.vdot(incident, source[transmission, transmission] @ incident).real
            + 2 * np.vdot(incident, source[transmission, reflection] @ refl).real
        )
        if flux <= 0:
            raise ValueError("transmittance requires positive incident power flux")
        return float(
            sign
            * np.vdot(trans, target[transmission, transmission] @ trans).real
            / flux
        ), float(
            -sign * np.vdot(refl, source[reflection, reflection] @ refl).real / flux
        )

    def cd(
        self, illu: ArrayLike | PlaneWave, *, modetype: str | None = None
    ) -> tuple[float, float]:
        """Transmission and total-outgoing-power contrast against opposite polarization.

        These are upstream's two CD formulas: (T_opposite-T)/(T_opposite+T)
        and ((T+R)_opposite-(T+R))/((T+R)_opposite+(T+R)). The second quantity
        is normalized by outgoing power, although upstream calls it absorption CD.
        Helicity bases must contain both polarizations of each direction.
        """
        direction = _direction(illu, modetype, self.basis.normal_axis)
        transmission, reflection = self.tr(illu, modetype=direction)
        incident = self._incident(illu, direction)
        if self.poltype == "helicity":
            indices = {mode: i for i, mode in enumerate(self.basis.modes)}
            try:
                opposite = incident[
                    [indices[(x, y, 1 - p)] for x, y, p in self.basis.modes]
                ]
            except KeyError as error:
                raise ValueError(
                    "CD requires both helicities of each direction"
                ) from error
        else:
            opposite = incident * (2 * self.basis.pol - 1)
        opposite_t, opposite_r = self.tr(opposite, modetype=direction)
        total = transmission + reflection
        opposite_total = opposite_t + opposite_r
        if transmission + opposite_t == 0 or total + opposite_total == 0:
            raise ValueError(
                "CD is undefined for zero summed transmission or outgoing power"
            )
        return (opposite_t - transmission) / (opposite_t + transmission), (
            opposite_total - total
        ) / (opposite_total + total)


def _direction(illu: ArrayLike | PlaneWave, modetype: str | None, axis: int = 2) -> str:
    if modetype is None:
        kz = illu.kvecs[0, axis] if isinstance(illu, PlaneWave) else 1 + 0j
        return "down" if kz.imag < 0 or (kz.imag == 0 and kz.real < 0) else "up"
    if modetype not in ("up", "down"):
        raise ValueError("modetype must be up or down")
    return modetype


def _power_forms(
    basis: PlaneWaveBasisByComp, k0: float, material: MaterialLike, poltype: str
) -> NDArray[np.complex128]:
    """Hermitian up/down power blocks from Cartesian E cross H*, averaged in the cell."""
    electric = np.stack(
        [
            efield(
                [0, 0, 0],
                basis=basis,
                k0=k0,
                material=material,
                modetype=side,
                poltype=poltype,
            )
            for side in ("up", "down")
        ]
    )
    magnetic = np.stack(
        [
            hfield(
                [0, 0, 0],
                basis=basis,
                k0=k0,
                material=material,
                modetype=side,
                poltype=poltype,
            )
            for side in ("up", "down")
        ]
    )
    a, b = (basis.normal_axis + 1) % 3, (basis.normal_axis + 2) % 3
    q = basis.components
    same = np.all(q[:, None, :] == q[None, :, :], axis=-1)
    result = np.empty((2, 2, len(basis), len(basis)), complex)
    for i in range(2):
        for j in range(2):
            result[i, j] = (
                0.25
                * same
                * (
                    magnetic[i, b].conj()[:, None] * electric[j, a]
                    - magnetic[i, a].conj()[:, None] * electric[j, b]
                    + electric[i, a].conj()[:, None] * magnetic[j, b]
                    - electric[i, b].conj()[:, None] * magnetic[j, a]
                )
            )
    return result


def chirality_density(
    basis: PlaneWaveBasisByComp,
    k0: float,
    material: MaterialLike = 1,
    poltype: str = "helicity",
    z: ArrayLike = (0.0, 0.0),
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """Up/down/coherent-cross forms of 2 Re(E* . i Z H), along the basis normal.

    Equal endpoints evaluate at that plane. z gives coordinates along the normal.
    For amplitudes u,d the density is
    Re(u* U u + d* D d + d* X u). X can be complex for a shifted interval.
    This corrects upstream's attenuation average and discarded cross phase.
    """
    medium = Material(material)
    if poltype not in ("helicity", "parity") or (
        poltype == "parity" and medium.ischiral
    ):
        raise ValueError("invalid polarization type for the medium")
    normal = basis.kvecs(k0, medium)[basis.normal_axis]
    if basis.alignment != "xy":
        q = basis.components
        if poltype == "helicity":
            values, _ = diff.oriented_chirality(
                q, normal, z, polarizations=basis.pol, axis=basis.normal_axis
            )
            up, down, cross = (np.diag(row) for row in values)
        else:
            values, _ = diff.oriented_chirality(
                np.repeat(q, 2, axis=0),
                np.repeat(normal, 2),
                z,
                polarizations=np.tile([0, 1], len(basis)),
                axis=basis.normal_axis,
            )
            values = values.reshape(3, len(basis), 2)
            same = np.all(q[:, None, :] == q[None, :, :], axis=-1)
            sign = 2 * basis.pol - 1
            pair = sign[:, None] * sign
            up, down, cross = (
                0.5 * same * (pair * row[:, 0] + row[:, 1]) for row in values
            )
        return up, down, cross
    values, _ = diff.chirality_density(
        medium.ks(k0)[basis.pol],
        normal,
        z,
    )
    if poltype == "helicity":
        values = values * (2 * basis.pol - 1)
        up, down, cross = (np.diag(row) for row in values)
        return up, down, cross
    if poltype == "parity" and not medium.ischiral:
        same = np.all(
            basis.components[:, None, :] == basis.components[None, :, :], axis=-1
        )
        paired = same & (basis.pol[:, None] != basis.pol)
        up, down, cross = (paired * row for row in values)
        return up, down, cross
    raise ValueError("invalid polarization type for the medium")


def poynting_avg_z(
    basis: PlaneWaveBasisByComp,
    k0: float,
    material: MaterialLike = 1,
    poltype: str = "helicity",
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Same- and opposite-direction time-averaged axial power-flux forms."""
    if basis.alignment != "xy":
        raise ValueError("axial power forms currently require xy-aligned plane bases")
    medium = Material(material)
    kx, ky, kz = basis.kvecs(k0, medium)
    gamma = kz / (medium.ks(k0)[basis.pol] * medium.impedance)
    selection = (kx[:, None] == kx) & (ky[:, None] == ky)
    pol = basis.pol
    if poltype == "parity" and not medium.ischiral:
        selection &= pol[:, None] == pol
        return selection * (
            (1 - pol) * gamma.conj() + pol * gamma
        ) * 0.25, selection * ((1 - pol) * gamma.conj() - pol * gamma) * 0.25
    if poltype == "helicity":
        signs = 2 * pol - 1
        return selection * (
            signs[:, None] * signs * gamma[:, None].conj() + gamma
        ) * 0.25, selection * (
            signs[:, None] * signs * gamma[:, None].conj() - gamma
        ) * 0.25
    raise ValueError("invalid polarization type for the medium")

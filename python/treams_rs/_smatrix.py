"""Planar interfaces and stable layer composition with native numerical kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import coeffs, diff
from ._core import Material, MaterialLike, PlaneWaveBasisByComp, SphericalWaveBasis
from ._plane import PlaneWave

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import ArrayLike, NDArray

    from ._tmatrix import TMatrix


class SMatrices:
    """Four scattering blocks indexed by outgoing and incoming up/down direction.

    ``material`` is ordered (above, below); interface/slab inputs run bottom to top.
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
        if basis.alignment != "xy":
            raise ValueError("S matrices currently require xy-aligned plane bases")
        self.array: NDArray[np.complex128] = np.array(
            smats, dtype=np.complex128, copy=True
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
        ks = np.array([below.ks(k0), above.ks(k0)])
        zs = np.array([below.impedance, above.impedance])
        result = np.zeros((2, 2, len(basis), len(basis)), dtype=np.complex128)
        for kx, ky in dict.fromkeys((kx, ky) for kx, ky, _ in basis):
            kz = np.array([m.kzs(k0, kx, ky) for m in (below, above)])
            value = coeffs.fresnel(ks, kz, zs)
            indices = [
                (i, pol) for i, (x, y, pol) in enumerate(basis) if (x, y) == (kx, ky)
            ]
            for i, pol in indices:
                for j, pol2 in indices:
                    result[:, :, i, j] = value[:, :, pol, pol2]
        return cls(result, basis=basis, k0=k0, material=(above, below)).changepoltype(
            poltype
        )

    @classmethod
    def from_array(
        cls,
        tm: TMatrix,
        basis: PlaneWaveBasisByComp,
        *,
        lattice: ArrayLike,
        kpar: ArrayLike,
        eta: complex = 0,
    ) -> SMatrices:
        """Solve one uncoupled spherical unit cell and radiate into plane-wave ports.

        Lattice and Bloch vector are explicit. The input T matrix describes one
        uncoupled unit cell; this constructor solves its periodic interaction.
        """
        if not isinstance(tm.basis, SphericalWaveBasis):
            raise ValueError("array radiation currently requires spherical multipoles")
        vectors = np.asarray(lattice, dtype=np.float64)
        bloch = np.asarray(kpar, dtype=np.float64)
        if vectors.shape != (2, 2) or bloch.shape != (2,):
            raise ValueError("array radiation requires a 2D lattice and Bloch vector")
        q = basis.components
        orders = (q - bloch) @ vectors.T / (2 * np.pi)
        if not np.allclose(orders, np.round(orders), atol=1e-10, rtol=0):
            raise ValueError(
                "plane-wave channels must match the lattice diffraction orders"
            )
        response = tm.latticeinteraction.solve(vectors, bloch, eta=eta)
        channels, _ = diff.spherical_channels(
            tm.basis,
            tm.ks,
            q,
            basis.pol,
            float(abs(np.linalg.det(vectors))),
            poltype=tm.poltype,
            fixed_q=True,
        )
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
        distance = np.asarray(r, dtype=np.float64)
        if distance.ndim == 0:
            distance = np.array([0, 0, float(distance)])
        vectors = np.stack(basis.kvecs(k0, material), axis=-1)
        value, _ = diff.propagation(vectors, distance)
        return cls(
            value, basis=basis, k0=k0, material=Material(material), poltype=poltype
        )

    def add(self, upper: SMatrices) -> SMatrices:
        if (
            self.k0 != upper.k0
            or self.poltype != upper.poltype
            or self.basis.modes != upper.basis.modes
            or self.material[0] != upper.material[1]
        ):
            raise ValueError(
                "coupled S matrices must match basis, k0, polarization and internal medium"
            )
        value, _ = diff.smatrix_add(self.array, upper.array)
        return type(self)(
            value,
            basis=self.basis,
            k0=self.k0,
            material=(upper.material[0], self.material[1]),
            poltype=self.poltype,
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
        q = self.basis.components
        same = np.all(q[:, None, :] == q[None, :, :], axis=-1)
        if not np.all(same.sum(axis=0) == 2):
            raise ValueError(
                "polarization conversion requires both polarizations per direction"
            )
        signs = np.where((self.basis.pol[:, None] == 0) & (self.basis.pol == 0), -1, 1)
        change = same * signs * np.sqrt(0.5)
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
                or _direction(illu, None) != modetype
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
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        modetype = _direction(illu, modetype)
        first = self._incident(illu, modetype)
        second = (
            np.zeros_like(first)
            if illu2 is None
            else self._incident(illu2, "down" if modetype == "up" else "up")
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
        return self[0, 0] @ up + self[0, 1] @ down, self[1, 0] @ up + self[1, 1] @ down

    def tr(
        self, illu: ArrayLike | PlaneWave, *, modetype: str | None = None
    ) -> tuple[float, float]:
        modetype = _direction(illu, modetype)
        trans, refl = self.illuminate(illu, modetype=modetype)
        materials = self.material
        if modetype == "down":
            trans, refl, materials = refl, trans, materials[::-1]
        a, _ = poynting_avg_z(self.basis, self.k0, materials[0], self.poltype)
        b, cross = poynting_avg_z(self.basis, self.k0, materials[1], self.poltype)
        incident = self._incident(illu, modetype)
        flux = (
            np.vdot(incident, b @ incident).real
            + (np.vdot(refl, cross @ incident) - np.vdot(incident, cross @ refl)).real
        )
        if flux <= 0:
            raise ValueError("transmittance requires positive incident power flux")
        return float(np.vdot(trans, a @ trans).real / flux), float(
            np.vdot(refl, b @ refl).real / flux
        )


def _direction(illu: ArrayLike | PlaneWave, modetype: str | None) -> str:
    if modetype is None:
        kz = illu.kvecs[0, 2] if isinstance(illu, PlaneWave) else 1 + 0j
        return "down" if kz.imag < 0 or (kz.imag == 0 and kz.real < 0) else "up"
    if modetype not in ("up", "down"):
        raise ValueError("modetype must be up or down")
    return modetype


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

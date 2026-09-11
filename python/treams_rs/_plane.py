"""Plane-wave illumination with explicit amplitudes and native basis conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native, diff
from ._core import (
    CylindricalWaveBasis,
    Material,
    MaterialLike,
    PlaneWaveBasisByComp,
    SphericalWaveBasis,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class PlaneWave:
    """A plane wave with amplitudes ordered by polarization index (0, 1)."""

    def __init__(
        self,
        kvec: ArrayLike,
        pol: ArrayLike,
        *,
        k0: float = 1.0,
        material: MaterialLike = 1,
        poltype: str = "helicity",
    ):
        vector = np.asarray(kvec, dtype=np.complex128)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("kvec must contain three finite components")
        norm = np.sqrt(np.sum(vector**2))
        if norm == 0 or not np.isfinite(k0) or k0 <= 0:
            raise ValueError("require nonzero wavevector norm and positive k0")
        self.direction = vector / norm
        self.direction.flags.writeable = False
        self.k0 = float(k0)
        self.material = Material(material)
        if poltype not in ("helicity", "parity") or (
            poltype == "parity" and self.material.ischiral
        ):
            raise ValueError("invalid polarization type for embedding medium")
        self.poltype = poltype
        polarization = np.asarray(pol, dtype=np.complex128)
        if polarization.ndim == 0:
            index = complex(polarization.item())
            if index not in (-1, 0, 1):
                raise ValueError("polarization index must be 0 or 1")
            self.amplitudes = np.array(
                [1, 0] if index in (-1, 0) else [0, 1], dtype=np.complex128
            )
        elif polarization.shape == (2,):
            self.amplitudes = polarization.copy()
        elif polarization.shape == (3,):
            if poltype == "helicity":
                self.amplitudes = np.array(
                    [
                        np.dot(
                            _native.plane_polarization(
                                tuple(complex(v) for v in self.kvecs[pol]),
                                1 - pol,
                                True,
                            ),
                            polarization,
                        )
                        for pol in (0, 1)
                    ]
                )
            else:
                self.amplitudes = np.array(
                    [
                        (-1 if pol == 0 else 1)
                        * np.dot(
                            _native.plane_polarization(
                                tuple(complex(v) for v in self.kvecs[pol]), pol, False
                            ),
                            polarization,
                        )
                        for pol in (0, 1)
                    ]
                )
        else:
            raise ValueError(
                "pol must be an index, two amplitudes, or three electric components"
            )
        if not np.isfinite(self.amplitudes).all():
            raise ValueError("polarization amplitudes must be finite")
        self.amplitudes.flags.writeable = False

    @property
    def kvecs(self) -> NDArray[np.complex128]:
        return self.material.ks(self.k0)[:, None] * self.direction

    def efield(self, r: ArrayLike) -> NDArray[np.complex128]:
        """Cartesian samples using the native weighted plane-field kernel."""
        points = np.asarray(r, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError("require Cartesian field points (..., 3)")
        value, _ = diff.plane_field(
            self.amplitudes,
            points.reshape(-1, 3),
            self.kvecs,
            [0, 1],
            poltype=self.poltype,
            fixed_vectors=True,
        )
        return value.reshape(points.shape)

    def expand(
        self, basis: SphericalWaveBasis | CylindricalWaveBasis | PlaneWaveBasisByComp
    ) -> NDArray[np.complex128]:
        """Regular multipole amplitudes at the supplied basis origins."""
        values = np.zeros(len(basis), dtype=np.complex128)
        if isinstance(basis, PlaneWaveBasisByComp):
            for pol in (0, 1):
                if self.amplitudes[pol] == 0:
                    continue
                matching = (
                    (basis.pol == pol)
                    & np.isclose(basis.kx, self.kvecs[pol, 0], rtol=1e-13, atol=1e-14)
                    & np.isclose(basis.ky, self.kvecs[pol, 1], rtol=1e-13, atol=1e-14)
                )
                if np.count_nonzero(matching) != 1:
                    raise ValueError(
                        "plane-wave illumination requires exactly one matching basis mode"
                    )
                values[matching] = self.amplitudes[pol]
            return values
        for pol in (0, 1):
            if self.amplitudes[pol] != 0:
                vector = tuple(complex(v) for v in self.kvecs[pol])
                if isinstance(basis, SphericalWaveBasis):
                    converted = _native.plane_to_spherical(
                        list(basis.modes),
                        basis.positions.tolist(),
                        vector,
                        pol,
                        self.poltype == "helicity",
                    )
                else:
                    converted = _native.plane_to_cylindrical(
                        list(basis.modes), basis.positions.tolist(), vector, pol
                    )
                values += self.amplitudes[pol] * converted
        return values


def plane_wave(
    kvec: ArrayLike,
    pol: ArrayLike,
    *,
    k0: float = 1.0,
    material: MaterialLike = 1,
    poltype: str = "helicity",
) -> PlaneWave:
    """Define an incident plane wave from a real or complex propagation direction."""
    return PlaneWave(kvec, pol, k0=k0, material=material, poltype=poltype)


def plane_wave_angle(
    theta: float,
    phi: float,
    pol: ArrayLike,
    *,
    k0: float = 1.0,
    material: MaterialLike = 1,
    poltype: str = "helicity",
) -> PlaneWave:
    """Define an incident plane wave by polar and azimuthal angles in radians."""
    return plane_wave(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        pol,
        k0=k0,
        material=material,
        poltype=poltype,
    )

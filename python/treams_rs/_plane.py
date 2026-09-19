"""Plane-wave illumination with explicit amplitudes and native basis conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np

from . import _native, diff
from ._core import (
    CylindricalWaveBasis,
    Material,
    MaterialLike,
    PlaneWaveBasisByComp,
    PlaneWaveBasisByUnitVector,
    SphericalWaveBasis,
    _unit_vectors,
)
from ._operators import _WaveFields
from .config import _resolve_poltype

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._source import MultipoleWave


class PlaneWave(_WaveFields):
    """A plane wave with amplitudes ordered by polarization index (0, 1)."""

    def __init__(
        self,
        kvec: ArrayLike,
        pol: ArrayLike,
        *,
        k0: float = 1.0,
        material: MaterialLike = 1,
        poltype: str | None = None,
    ):
        poltype = _resolve_poltype(poltype)
        vector = np.asarray(kvec, dtype=np.complex128)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("kvec must contain three finite components")
        if not np.isfinite(k0) or k0 <= 0:
            raise ValueError("require positive finite k0")
        self.direction = _unit_vectors(vector[None, :])[0]
        self.direction.flags.writeable = False
        self.k0 = float(k0)
        self.material = Material(material)
        if poltype == "parity" and self.material.ischiral:
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

    modetype = "up"

    @property
    def coefficients(self) -> NDArray[np.complex128]:
        """Read-only polarization amplitudes in negative/positive label order."""
        return self.array

    @property
    def medium(self) -> Material:
        """Homogeneous propagation medium."""
        return self.material

    @property
    def polarization(self) -> str:
        """Polarization basis convention: helicity or parity."""
        return self.poltype

    def with_polarization(self, polarization: str) -> PlaneWave:
        """Represent the same plane wave with helicity or parity amplitudes."""
        wave = self.in_basis(self.basis).with_polarization(polarization)
        return type(self)(
            self.direction,
            wave.coefficients,
            k0=self.k0,
            material=self.medium,
            poltype=polarization,
        )

    def in_basis(
        self,
        basis: SphericalWaveBasis
        | CylindricalWaveBasis
        | PlaneWaveBasisByComp
        | PlaneWaveBasisByUnitVector,
    ) -> MultipoleWave:
        """Expand into a typed regular multipole wave or directional plane wave."""
        from ._source import MultipoleWave

        kind = "regular"
        if isinstance(basis, PlaneWaveBasisByComp):
            normal = self.kvecs[0, basis.normal_axis]
            kind = (
                "down"
                if normal.imag < 0 or (normal.imag == 0 and normal.real < 0)
                else "up"
            )
        elif isinstance(basis, PlaneWaveBasisByUnitVector):
            kind = "up"
        return MultipoleWave(
            self.expand(basis),
            basis=basis,
            k0=self.k0,
            material=self.material,
            modetype=kind,
            poltype=self.poltype,
        )

    @property
    @override
    def basis(self) -> PlaneWaveBasisByUnitVector:
        return PlaneWaveBasisByUnitVector([(*self.direction, pol) for pol in (0, 1)])

    @property
    @override
    def array(self) -> NDArray[np.complex128]:
        return self.amplitudes

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        return np.asarray(self.array, dtype=dtype, copy=copy)

    @property
    def kvecs(self) -> NDArray[np.complex128]:
        return self.material.ks(self.k0)[:, None] * self.direction

    @override
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
        self,
        basis: SphericalWaveBasis
        | CylindricalWaveBasis
        | PlaneWaveBasisByComp
        | PlaneWaveBasisByUnitVector,
    ) -> NDArray[np.complex128]:
        """Regular multipole amplitudes at the supplied basis origins."""
        values = np.zeros(len(basis), dtype=np.complex128)
        if isinstance(basis, PlaneWaveBasisByUnitVector):
            for pol in (0, 1):
                if self.amplitudes[pol] == 0:
                    continue
                matching = (basis.pol == pol) & np.all(
                    np.isclose(
                        basis.directions, self.direction, rtol=1e-13, atol=1e-14
                    ),
                    axis=1,
                )
                if np.count_nonzero(matching) != 1:
                    raise ValueError(
                        "plane-wave illumination requires exactly one matching basis mode"
                    )
                values[matching] = self.amplitudes[pol]
            return values
        if isinstance(basis, PlaneWaveBasisByComp):
            for pol in (0, 1):
                if self.amplitudes[pol] == 0:
                    continue
                matching = (basis.pol == pol) & np.all(
                    np.isclose(
                        basis.components,
                        self.kvecs[
                            pol, ["xyz".index(axis) for axis in basis.alignment]
                        ],
                        rtol=1e-13,
                        atol=1e-14,
                    ),
                    axis=1,
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
    kvec: ArrayLike | None = None,
    pol: ArrayLike | None = None,
    *,
    k0: float = 1.0,
    material: MaterialLike = 1,
    poltype: str | None = None,
    direction: ArrayLike | None = None,
    polarization: ArrayLike | str | None = None,
    medium: MaterialLike | None = None,
) -> PlaneWave:
    """Define a plane wave from direction, polarization, k0 and medium.

    Direction is normalized, independently of the vacuum angular wavenumber k0.
    Polarization is ``positive_helicity``/``negative_helicity``, a pair of
    channel amplitudes, or three Cartesian electric components. Explicit
    helicity names select the helicity convention. The positional kvec/pol and
    material/poltype parameters expose the coefficient-level convention.
    """
    if direction is not None:
        if kvec is not None:
            raise ValueError("specify direction only once")
        kvec = direction
    if polarization is not None:
        if pol is not None:
            raise ValueError("specify polarization only once")
        pol = polarization
    if isinstance(pol, str):
        if pol not in ("positive_helicity", "negative_helicity"):
            raise ValueError(
                "named polarization must be positive_helicity or negative_helicity"
            )
        if poltype not in (None, "helicity"):
            raise ValueError("helicity polarization requires helicity channels")
        poltype = "helicity"
        pol = int(pol == "positive_helicity")
    if kvec is None or pol is None:
        raise TypeError("plane_wave requires direction and polarization")
    if medium is not None:
        if Material(material) != Material():
            raise ValueError("specify medium only once")
        material = medium
    poltype = _resolve_poltype(poltype)
    return PlaneWave(kvec, pol, k0=k0, material=material, poltype=poltype)


def plane_wave_angle(
    theta: float,
    phi: float,
    pol: ArrayLike,
    *,
    k0: float = 1.0,
    material: MaterialLike = 1,
    poltype: str | None = None,
) -> PlaneWave:
    """Define an incident plane wave by polar and azimuthal angles in radians."""
    poltype = _resolve_poltype(poltype)
    return plane_wave(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        pol,
        k0=k0,
        material=material,
        poltype=poltype,
    )

"""Typed physical metadata; numerical execution lives in the native extension."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, init=False)
class Material:
    """Relative permittivity, permeability and chirality, in treams conventions."""

    epsilon: complex
    mu: complex
    kappa: complex

    def __init__(self, epsilon: MaterialLike = 1, mu: complex = 1, kappa: complex = 0):
        if isinstance(epsilon, Material):
            epsilon, mu, kappa = epsilon()
        elif isinstance(epsilon, (tuple, list)):
            if len(epsilon) > 3:
                raise ValueError("invalid material definition")
            epsilon, mu, kappa = tuple(epsilon) + (1, 1, 0)[len(epsilon) :]
        object.__setattr__(self, "epsilon", complex(epsilon))
        object.__setattr__(self, "mu", complex(mu))
        object.__setattr__(self, "kappa", complex(kappa))

    def __call__(self) -> tuple[complex, complex, complex]:
        return self.epsilon, self.mu, self.kappa

    def __iter__(self) -> Iterator[complex]:
        return iter(self())

    @classmethod
    def from_n(
        cls, n: complex = 1, impedance: complex | None = None, kappa: complex = 0
    ) -> Material:
        impedance = 1 / n if impedance is None else impedance
        return cls(n / impedance, n * impedance, kappa)

    @classmethod
    def from_nmp(
        cls, ns: tuple[complex, complex] = (1, 1), impedance: complex | None = None
    ) -> Material:
        return cls.from_n(sum(ns) / 2, impedance, (ns[1] - ns[0]) / 2)

    @property
    def n(self) -> complex:
        value = cmath.sqrt(self.epsilon * self.mu)
        return -value if value.imag < 0 else value

    @property
    def nmp(self) -> NDArray[np.complex128]:
        index = cmath.sqrt(self.epsilon * self.mu)
        values = np.array([index - self.kappa, index + self.kappa], dtype=np.complex128)
        return np.where(values.imag < 0, -values, values)

    @property
    def impedance(self) -> complex:
        return cmath.sqrt(self.mu / self.epsilon)

    @property
    def isreal(self) -> bool:
        return all(value.imag == 0 for value in self)

    @property
    def ischiral(self) -> bool:
        return self.kappa != 0

    def ks(self, k0: float) -> NDArray[np.complex128]:
        return k0 * self.nmp


type MaterialLike = Material | complex | tuple[complex, ...] | list[complex]
type Mode = tuple[int, int, int, int]


class _WaveBasis[M: tuple[int, float, int, int]]:
    def __init__(self, modes: Sequence[M], positions: ArrayLike):
        if not modes or len(set(modes)) != len(modes):
            raise ValueError("basis must contain distinct modes")
        self.modes: tuple[M, ...] = tuple(modes)
        self.positions: NDArray[np.float64] = np.array(
            positions, dtype=np.float64, copy=True
        )
        if (
            self.positions.ndim != 2
            or self.positions.shape[1] != 3
            or not np.isfinite(self.positions).all()
            or max(p for p, *_ in modes) >= len(self.positions)
        ):
            raise ValueError("invalid basis positions")
        self.positions.flags.writeable = False

    def __len__(self) -> int:
        return len(self.modes)

    def __iter__(self) -> Iterator[M]:
        return iter(self.modes)

    @property
    def pidx(self) -> NDArray[np.int64]:
        return np.array([m[0] for m in self.modes], dtype=np.int64)

    @property
    def m(self) -> NDArray[np.int64]:
        return np.array([m[2] for m in self.modes], dtype=np.int64)

    @property
    def pol(self) -> NDArray[np.int64]:
        return np.array([m[3] for m in self.modes], dtype=np.int64)

    @property
    def isglobal(self) -> bool:
        return len({mode[0] for mode in self.modes}) == 1


class SphericalWaveBasis(_WaveBasis[Mode]):
    """Modes (particle, l, m, pol), with positive helicity first by default."""

    def __init__(
        self, modes: Iterable[Sequence[float]], positions: ArrayLike = ((0, 0, 0),)
    ):
        values: list[Mode] = []
        for row in modes:
            mode = tuple(row)
            if len(mode) == 3:
                mode = (0, *mode)
            if len(mode) != 4:
                raise ValueError("modes require (l, m, pol) or (particle, l, m, pol)")
            p, degree, order, pol = mode
            if (
                any(int(v) != v for v in mode)
                or p < 0
                or not 1 <= degree <= 128
                or abs(order) > degree
                or pol not in (0, 1)
            ):
                raise ValueError("invalid spherical mode")
            values.append((int(p), int(degree), int(order), int(pol)))
        super().__init__(values, positions)

    @staticmethod
    def defaultdim(lmax: int) -> int:
        return 2 * lmax * (lmax + 2)

    @staticmethod
    def defaultlmax(dim: int) -> int:
        order = math.isqrt(dim // 2 + 1) - 1
        if order < 1 or SphericalWaveBasis.defaultdim(order) != dim:
            raise ValueError("dimension does not define a complete spherical basis")
        return order

    @classmethod
    def default(
        cls, lmax: int, nmax: int = 1, positions: ArrayLike | None = None
    ) -> SphericalWaveBasis:
        if not 1 <= lmax <= 128 or nmax < 1:
            raise ValueError("require 1 <= lmax <= 128 and nmax >= 1")
        return cls(
            (
                (p, degree, order, pol)
                for p in range(nmax)
                for degree in range(1, lmax + 1)
                for order in range(-degree, degree + 1)
                for pol in (1, 0)
            ),
            np.zeros((nmax, 3)) if positions is None else positions,
        )

    @property
    def l(self) -> NDArray[np.int64]:  # noqa: E743
        return np.array([m[1] for m in self.modes], dtype=np.int64)

    @property
    def lms(self) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        return self.l, self.m, self.pol


type CylindricalMode = tuple[int, float, int, int]


class CylindricalWaveBasis(_WaveBasis[CylindricalMode]):
    """Modes (particle, kz, m, pol), with fixed real axial mode labels."""

    def __init__(
        self, modes: Iterable[Sequence[float]], positions: ArrayLike = ((0, 0, 0),)
    ):
        values: list[CylindricalMode] = []
        for row in modes:
            mode = tuple(row)
            if len(mode) == 3:
                mode = (0, *mode)
            if len(mode) != 4:
                raise ValueError("modes require (kz, m, pol) or (particle, kz, m, pol)")
            p, kz, order, pol = mode
            if (
                not math.isfinite(kz)
                or any(int(v) != v for v in (p, order, pol))
                or p < 0
                or abs(order) > 128
                or pol not in (0, 1)
            ):
                raise ValueError("invalid cylindrical mode")
            values.append((int(p), float(kz), int(order), int(pol)))
        super().__init__(values, positions)

    @staticmethod
    def defaultdim(nkz: int, mmax: int, nmax: int = 1) -> int:
        return 2 * nkz * (2 * mmax + 1) * nmax

    @staticmethod
    def defaultmmax(dim: int, nkz: int = 1, nmax: int = 1) -> int:
        if nkz < 1 or nmax < 1:
            raise ValueError("nkz and nmax must be positive")
        order = (dim // (2 * nkz * nmax) - 1) // 2
        if order < 0 or CylindricalWaveBasis.defaultdim(nkz, order, nmax) != dim:
            raise ValueError("dimension does not define a complete cylindrical basis")
        return order

    @classmethod
    def default(
        cls,
        kzs: ArrayLike,
        mmax: int,
        nmax: int = 1,
        positions: ArrayLike | None = None,
    ) -> CylindricalWaveBasis:
        axial = np.atleast_1d(np.asarray(kzs, dtype=np.float64))
        if axial.ndim != 1 or not 0 <= mmax <= 128 or nmax < 1:
            raise ValueError("require a kz vector, 0 <= mmax <= 128 and nmax >= 1")
        return cls(
            (
                (p, float(kz), order, pol)
                for p in range(nmax)
                for kz in axial
                for order in range(-mmax, mmax + 1)
                for pol in (1, 0)
            ),
            np.zeros((nmax, 3)) if positions is None else positions,
        )

    @property
    def kz(self) -> NDArray[np.float64]:
        return np.array([m[1] for m in self.modes], dtype=np.float64)

    @property
    def zms(self) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        return self.kz, self.m, self.pol

"""Typed physical metadata; numerical execution lives in the native extension."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from functools import cached_property
from types import EllipsisType
from typing import TYPE_CHECKING, Self, overload, override

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

    def kzs(
        self, k0: float, kx: ArrayLike, ky: ArrayLike, pol: ArrayLike = (0, 1)
    ) -> NDArray[np.complex128]:
        """Axial wavevectors on the outgoing branch (nonnegative imaginary part)."""
        value = np.sqrt(
            self.ks(k0)[np.asarray(pol, dtype=np.int64)] ** 2
            - np.asarray(kx) ** 2
            - np.asarray(ky) ** 2
        )
        return np.where(value.imag < 0, -value, value)


type MaterialLike = Material | complex | tuple[complex, ...] | list[complex]
type Mode = tuple[int, int, int, int]


def _unit_vectors(vectors: ArrayLike) -> NDArray[np.complex128]:
    values = np.asarray(vectors, dtype=np.complex128)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("wavevectors require finite shape (n, 3)")
    scale = np.max(np.abs(values), axis=1)
    if np.any(scale == 0):
        raise ValueError("wavevectors must have nonzero algebraic norm")
    scaled = values / scale[:, None]
    norm = np.sqrt(np.sum(scaled**2, axis=1))
    if np.any(norm == 0):
        raise ValueError("wavevectors must have nonzero algebraic norm")
    return scaled / norm[:, None]


type _Selection = (
    slice
    | Sequence[int]
    | Sequence[bool]
    | NDArray[np.integer]
    | NDArray[np.bool_]
    | EllipsisType
)


class _Basis[M: tuple[object, ...]]:
    """Ordered mode labels; selections preserve their physical metadata."""

    modes: tuple[M, ...]
    _labels: tuple[str, ...]
    _keys: str

    def __len__(self) -> int:
        return len(self.modes)

    def __iter__(self) -> Iterator[M]:
        return iter(self.modes)

    def __getattr__(self, key: str) -> tuple[NDArray[np.generic], ...]:
        labels = dict(zip(self._keys, self._labels, strict=True))
        try:
            return tuple(getattr(self, labels[part]) for part in key)
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {key!r}"
            ) from None

    def __contains__(self, mode: object) -> bool:
        return mode in self.modes

    def index(self, mode: M, start: int = 0, stop: int | None = None) -> int:
        return self.modes.index(mode, start, len(self) if stop is None else stop)

    def count(self, mode: M) -> int:
        return self.modes.count(mode)

    @overload
    def __getitem__(self, key: int | np.integer) -> M: ...
    @overload
    def __getitem__(self, key: tuple[()]) -> tuple[NDArray[np.generic], ...]: ...
    @overload
    def __getitem__(self, key: _Selection) -> Self: ...

    def __getitem__(
        self, key: int | np.integer | tuple[()] | _Selection
    ) -> M | tuple[NDArray[np.generic], ...] | Self:
        if isinstance(key, (int, np.integer)):
            return self.modes[int(key)]
        if isinstance(key, tuple) and not key:
            return tuple(getattr(self, label) for label in self._labels)
        selected = np.arange(len(self))[key]
        if selected.ndim != 1:
            raise IndexError("basis selections must be one-dimensional")
        indices = np.fromiter(dict.fromkeys(selected.tolist()), dtype=np.intp)
        return self._from_modes(tuple(self.modes[i] for i in indices))

    def _from_modes(self, modes: tuple[M, ...]) -> Self:
        raise NotImplementedError

    def _sets(self, other: Self) -> tuple[set[M], set[M]]:
        if type(other) is not type(self):
            raise TypeError("basis operations require the same family")
        if (
            isinstance(self, _WaveBasis)
            and isinstance(other, _WaveBasis)
            and not np.array_equal(self.positions, other.positions)
        ):
            raise ValueError("basis operations require identical origin tables")
        if (
            isinstance(self, PlaneWaveBasisByComp)
            and isinstance(other, PlaneWaveBasisByComp)
            and self.alignment != other.alignment
        ):
            raise ValueError("basis operations require identical plane alignment")
        return set(self.modes), set(other.modes)

    def __or__(self, other: Self) -> Self:
        left, _ = self._sets(other)
        return self._from_modes(
            self.modes + tuple(mode for mode in other if mode not in left)
        )

    def __and__(self, other: Self) -> Self:
        left, _ = self._sets(other)
        return self._from_modes(tuple(mode for mode in other if mode in left))

    def __sub__(self, other: Self) -> Self:
        _, right = self._sets(other)
        return self._from_modes(tuple(mode for mode in self if mode not in right))

    def __xor__(self, other: Self) -> Self:
        left, right = self._sets(other)
        return self._from_modes(
            tuple(mode for mode in self if mode not in right)
            + tuple(mode for mode in other if mode not in left)
        )

    def __le__(self, other: Self) -> bool:
        left, right = self._sets(other)
        return left <= right

    def __lt__(self, other: Self) -> bool:
        left, right = self._sets(other)
        return left < right

    def __ge__(self, other: Self) -> bool:
        left, right = self._sets(other)
        return left >= right

    def __gt__(self, other: Self) -> bool:
        left, right = self._sets(other)
        return left > right

    def isdisjoint(self, other: Self) -> bool:
        left, right = self._sets(other)
        return left.isdisjoint(right)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _Basis)
            and type(other) is type(self)
            and self.modes == other.modes
        )


class PlaneWaveBasisByUnitVector(_Basis[tuple[complex, complex, complex, int]]):
    """Full complex directions satisfying q.q=1, followed by polarization 0 or 1."""

    isglobal = True
    _labels = ("qx", "qy", "qz", "pol")
    _keys = "xyzs"

    def __init__(self, modes: Iterable[Sequence[complex]]):
        rows = list(dict.fromkeys(tuple(row) for row in modes))
        values = (
            np.asarray(rows, dtype=np.complex128)
            if rows
            else np.empty((0, 4), dtype=np.complex128)
        )
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("plane modes require (qx, qy, qz, pol) rows")
        if not np.all((values[:, 3] == 0) | (values[:, 3] == 1)):
            raise ValueError("polarizations must be 0 or 1")
        self.directions = _unit_vectors(values[:, :3])
        self.directions.flags.writeable = False
        self.modes: tuple[tuple[complex, complex, complex, int], ...] = tuple(
            (complex(q[0]), complex(q[1]), complex(q[2]), int(pol.real))
            for q, pol in zip(self.directions, values[:, 3], strict=True)
        )
        if len(set(self.modes)) != len(self.modes):
            raise ValueError(
                "basis must contain distinct normalized directions and polarizations"
            )

    @override
    def _from_modes(
        self, modes: tuple[tuple[complex, complex, complex, int], ...]
    ) -> Self:
        result = type(self).__new__(type(self))
        result.modes = modes
        result.directions = np.array(
            [mode[:3] for mode in modes], dtype=np.complex128
        ).reshape(-1, 3)
        result.directions.flags.writeable = False
        return result

    @property
    def qx(self) -> NDArray[np.complex128]:
        return self.directions[:, 0]

    @property
    def qy(self) -> NDArray[np.complex128]:
        return self.directions[:, 1]

    @property
    def qz(self) -> NDArray[np.complex128]:
        return self.directions[:, 2]

    @cached_property
    def pol(self) -> NDArray[np.int64]:
        result = np.array([m[3] for m in self.modes], dtype=np.int64)
        result.flags.writeable = False
        return result

    @classmethod
    def default(cls, kvecs: ArrayLike) -> PlaneWaveBasisByUnitVector:
        vectors = np.atleast_2d(np.asarray(kvecs, dtype=np.complex128))
        if vectors.shape[1] != 3:
            raise ValueError("wavevectors require shape (n, 3)")
        return cls((*vector, pol) for vector in vectors for pol in (1, 0))

    def kvecs(
        self, k0: float, material: MaterialLike = 1, modetype: str | None = None
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
        values = self.directions * Material(material).ks(k0)[self.pol, None]
        return values[:, 0], values[:, 1], values[:, 2]

    def rotate(self, phi: float) -> PlaneWaveBasisByUnitVector:
        """Rotate direction labels around z, preserving their polarizations."""
        if not math.isfinite(phi):
            raise ValueError("rotation angle must be finite")
        c, s = math.cos(phi), math.sin(phi)
        return type(self)(
            (c * x - s * y, s * x + c * y, z, p) for x, y, z, p in self.modes
        )

    def permute(self, n: int = 1) -> PlaneWaveBasisByUnitVector:
        if n != int(n):
            raise ValueError("number of permutations must be integer")
        n = int(n)
        vectors = np.roll(self.directions, n % 3, axis=1)
        return type(self)((*v, int(p)) for v, p in zip(vectors, self.pol, strict=True))

    def bycomp(
        self, k0: float, alignment: str = "xy", material: MaterialLike = 1
    ) -> PlaneWaveBasisByComp:
        if alignment not in ("xy", "yz", "zx"):
            raise ValueError("alignment must be xy, yz or zx")
        axes = ["xyz".index(axis) for axis in alignment]
        q = np.real_if_close(np.column_stack(self.kvecs(k0, material))[:, axes])
        if np.iscomplexobj(q):
            raise ValueError("component bases require real transverse wavevectors")
        return PlaneWaveBasisByComp(
            (
                (float(v[0]), float(v[1]), int(p))
                for v, p in zip(q, self.pol, strict=True)
            ),
            alignment,
        )


class PlaneWaveBasisByComp(_Basis[tuple[float, float, int]]):
    """Plane modes (k1, k2, pol), aligned with xy, yz or zx."""

    isglobal = True
    _labels = ("_k1", "_k2", "pol")
    _keys = "xys"

    def __init__(self, modes: Iterable[Sequence[float]], alignment: str = "xy"):
        if alignment not in ("xy", "yz", "zx"):
            raise ValueError("alignment must be xy, yz or zx")
        self.alignment = alignment
        values = []
        for row in modes:
            if len(row) != 3:
                raise ValueError("plane modes require (k1, k2, pol)")
            kx, ky, pol = row
            if not math.isfinite(kx) or not math.isfinite(ky) or pol not in (0, 1):
                raise ValueError(
                    "plane components must be finite and polarization 0 or 1"
                )
            values.append((float(kx), float(ky), int(pol)))
        self.modes: tuple[tuple[float, float, int], ...] = tuple(dict.fromkeys(values))

    @override
    def _from_modes(self, modes: tuple[tuple[float, float, int], ...]) -> Self:
        return type(self)(modes, self.alignment)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PlaneWaveBasisByComp)
            and self.alignment == other.alignment
            and super().__eq__(other)
        )

    @property
    def _k1(self) -> NDArray[np.float64]:
        return self.components[:, 0]

    @property
    def _k2(self) -> NDArray[np.float64]:
        return self.components[:, 1]

    @classmethod
    def default(cls, kpars: ArrayLike, alignment: str = "xy") -> PlaneWaveBasisByComp:
        values = np.atleast_2d(np.asarray(kpars, dtype=np.float64))
        if values.shape[1] != 2:
            raise ValueError("transverse wavevectors require shape (n, 2)")
        return cls(
            ((float(kx), float(ky), pol) for kx, ky in values for pol in (1, 0)),
            alignment,
        )

    @classmethod
    def diffr_orders(
        cls, kpar: ArrayLike, lattice: ArrayLike, bmax: float, *, alignment: str = "xy"
    ) -> PlaneWaveBasisByComp:
        """Both polarizations of all reciprocal vectors with length <= bmax.

        The cutoff applies before adding the Bloch vector. Opposite diffraction
        orders are adjacent, following treams ordering for rectangular lattices.
        """
        a = np.asarray(lattice, dtype=np.float64)
        q = np.asarray(kpar, dtype=np.float64)
        if (
            a.shape != (2, 2)
            or q.shape != (2,)
            or not np.isfinite(a).all()
            or not np.isfinite(q).all()
            or not math.isfinite(bmax)
            or bmax < 0
        ):
            raise ValueError(
                "require a finite 2D lattice, Bloch vector and nonnegative cutoff"
            )
        reciprocal = 2 * np.pi * np.linalg.inv(a).T
        # |G.a_i| <= bmax |a_i| bounds every integer coordinate, also for skew cells.
        bounds = np.ceil(bmax * np.linalg.norm(a, axis=1) / (2 * np.pi)).astype(int)
        orders = [(0, 0)]
        for m in range(int(bounds[0]) + 1):
            ns = [*range(int(bounds[1]) + 1), *range(-1, -int(bounds[1]) - 1, -1)]
            for n in ns:
                if m == 0 and n <= 0:
                    continue
                vector = np.array([m, n]) @ reciprocal
                if np.linalg.norm(vector) <= bmax:
                    orders.extend([(m, n), (-m, -n)])
        return cls.default(q + np.asarray(orders) @ reciprocal, alignment)

    @property
    def normal_axis(self) -> int:
        """Cartesian axis normal to the stored component plane."""
        return {"xy": 2, "yz": 0, "zx": 1}[self.alignment]

    @cached_property
    def components(self) -> NDArray[np.float64]:
        """The two real stored components, in alignment order."""
        result = np.array([(m[0], m[1]) for m in self.modes], dtype=np.float64).reshape(
            -1, 2
        )
        result.flags.writeable = False
        return result

    def _component(self, axis: str) -> NDArray[np.float64] | None:
        return (
            self.components[:, self.alignment.index(axis)]
            if axis in self.alignment
            else None
        )

    @property
    def kx(self) -> NDArray[np.float64] | None:
        return self._component("x")

    @property
    def ky(self) -> NDArray[np.float64] | None:
        return self._component("y")

    @property
    def kz(self) -> NDArray[np.float64] | None:
        return self._component("z")

    def rotate(self, phi: float) -> PlaneWaveBasisByComp:
        """Rotate xy direction labels around z, preserving their polarizations."""
        if self.alignment != "xy":
            raise ValueError("z rotation of component bases requires xy alignment")
        if not math.isfinite(phi):
            raise ValueError("rotation angle must be finite")
        c, s = math.cos(phi), math.sin(phi)
        return type(self)((c * x - s * y, s * x + c * y, p) for x, y, p in self.modes)

    def permute(self, n: int = 1) -> PlaneWaveBasisByComp:
        if n != int(n):
            raise ValueError("number of permutations must be integer")
        n = int(n)
        alignments = ("xy", "yz", "zx")
        return type(self)(
            self.modes, alignments[(alignments.index(self.alignment) + n) % 3]
        )

    @cached_property
    def pol(self) -> NDArray[np.int64]:
        result = np.array([m[2] for m in self.modes], dtype=np.int64)
        result.flags.writeable = False
        return result

    def kvecs(
        self, k0: float, material: MaterialLike = 1, modetype: str = "up"
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
        if modetype not in ("up", "down"):
            raise ValueError("modetype must be up or down")
        k1, k2 = self.components.astype(np.complex128).T
        normal = Material(material).kzs(k0, k1, k2, self.pol)
        if modetype == "down":
            normal = -normal
        if self.alignment == "yz":
            return normal, k1, k2
        if self.alignment == "zx":
            return k2, normal, k1
        return k1, k2, normal

    def byunitvector(
        self, k0: float, material: MaterialLike = 1, modetype: str = "up"
    ) -> PlaneWaveBasisByUnitVector:
        vectors = np.column_stack(self.kvecs(k0, material, modetype))
        return PlaneWaveBasisByUnitVector(
            (*v, int(p)) for v, p in zip(vectors, self.pol, strict=True)
        )


class _WaveBasis[M: tuple[int, float, int, int]](_Basis[M]):
    def __init__(self, modes: Sequence[M], positions: ArrayLike):
        self.modes = tuple(dict.fromkeys(modes))
        self.positions: NDArray[np.float64] = np.array(
            ((0, 0, 0),) if positions is None else positions,
            dtype=np.float64,
            copy=True,
        )
        if self.positions.ndim == 1:
            self.positions = self.positions[None, :]
        if (
            self.positions.ndim != 2
            or self.positions.shape[1] != 3
            or not np.isfinite(self.positions).all()
            or max((p for p, *_ in modes), default=-1) >= len(self.positions)
        ):
            raise ValueError("invalid basis positions")
        self.positions.flags.writeable = False

    @override
    def _from_modes(self, modes: tuple[M, ...]) -> Self:
        return type(self)(modes, self.positions)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _WaveBasis)
            and np.array_equal(self.positions, other.positions)
            and super().__eq__(other)
        )

    @property
    def pidx(self) -> NDArray[np.int64]:
        return np.array([m[0] for m in self.modes], dtype=np.int64)

    @property
    def m(self) -> NDArray[np.int64]:
        return np.array([m[2] for m in self.modes], dtype=np.int64)

    @cached_property
    def pol(self) -> NDArray[np.int64]:
        result = np.array([m[3] for m in self.modes], dtype=np.int64)
        result.flags.writeable = False
        return result

    @property
    def isglobal(self) -> bool:
        return len({mode[0] for mode in self.modes}) <= 1


class SphericalWaveBasis(_WaveBasis[Mode]):
    """Modes (particle, l, m, pol), with positive helicity first by default."""

    _labels = ("pidx", "l", "m", "pol")
    _keys = "plms"

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
    def defaultdim(lmax: int, nmax: int = 1) -> int:
        if lmax < 0 or nmax < 0:
            raise ValueError("degree and particle count must be nonnegative")
        return 2 * lmax * (lmax + 2) * nmax

    @staticmethod
    def defaultlmax(dim: int, nmax: int = 1) -> int:
        if dim < 0 or nmax < 1:
            raise ValueError(
                "require nonnegative dimension and positive particle count"
            )
        order = math.isqrt(dim // (2 * nmax) + 1) - 1
        if SphericalWaveBasis.defaultdim(order, nmax) != dim:
            raise ValueError("dimension does not define a complete spherical basis")
        return order

    @classmethod
    def default(
        cls, lmax: int, nmax: int = 1, positions: ArrayLike | None = None
    ) -> SphericalWaveBasis:
        if not 0 <= lmax <= 128 or nmax < 1:
            raise ValueError("require 0 <= lmax <= 128 and nmax >= 1")
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

    @classmethod
    def ebcm(
        cls,
        lmax: int,
        nmax: int = 1,
        mmax: int = -1,
        positions: ArrayLike | None = None,
    ) -> Self:
        """Order modes by particle, azimuthal order, degree and polarization."""
        if mmax == -1:
            mmax = lmax
        if not 0 <= mmax <= lmax <= 128 or nmax < 1:
            raise ValueError(
                "require 0 <= mmax <= lmax <= 128 and positive particle count"
            )
        return cls(
            (
                (p, degree, order, pol)
                for p in range(nmax)
                for order in range(-mmax, mmax + 1)
                for degree in range(max(1, abs(order)), lmax + 1)
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

    _labels = ("pidx", "kz", "m", "pol")
    _keys = "pzms"

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

    @classmethod
    def diffr_orders(
        cls,
        kz: float,
        mmax: int,
        lattice: float,
        bmax: float,
        nmax: int = 1,
        positions: ArrayLike | None = None,
    ) -> CylindricalWaveBasis:
        """Axial diffraction orders with reciprocal shifts bounded by bmax."""
        if (
            not math.isfinite(lattice)
            or lattice == 0
            or not math.isfinite(bmax)
            or bmax < 0
        ):
            raise ValueError("require a finite nonzero period and nonnegative cutoff")
        reciprocal = 2 * np.pi / lattice
        count = math.floor(bmax / abs(reciprocal))
        return cls.default(
            kz + np.arange(-count, count + 1) * reciprocal, mmax, nmax, positions
        )

    @property
    def kz(self) -> NDArray[np.float64]:
        return np.array([m[1] for m in self.modes], dtype=np.float64)

    @property
    def zms(self) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        return self.kz, self.m, self.pol

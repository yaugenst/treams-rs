"""Mode bases of the spherical, cylindrical and plane-wave families (treams._core)."""

from __future__ import annotations

import math
from functools import cached_property
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Self, cast, overload, override

import numpy as np

from . import _native
from ._lattice import Lattice, WaveVector, geometry_inputs, turns
from ._material import Material
from ._validation import MAX_DEGREE, cos_sin, unit_vectors

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from numpy.typing import ArrayLike, NDArray

    from ._material import MaterialLike

__all__ = [
    "ALIGNMENT_AXIS",
    "CylindricalBasis",
    "FieldBasis",
    "PlaneWaveBasis",
    "PlaneWavePorts",
    "SphericalBasis",
    "mode_lookup",
]

# Cartesian axis normal to each plane-port alignment.
ALIGNMENT_AXIS = {"xy": 2, "yz": 0, "zx": 1}


type SphericalMode = tuple[int, int, int, int]


type _Selection = (
    slice
    | Sequence[int]
    | Sequence[bool]
    | NDArray[np.integer]
    | NDArray[np.bool_]
    | EllipsisType
)


class _Basis[M: tuple[object, ...]]:
    """Ordered, distinct mode labels; selections keep the basis geometry.

    Each label column has a one-letter key. An attribute whose name consists
    of keys returns those columns as a tuple, in the order of the letters, as
    in treams: ``basis.lms`` is ``(basis.l, basis.m, basis.pol)``. The keys
    are ``p`` (pidx), ``l``, ``m`` and ``s`` (pol) for SphericalBasis; ``p``,
    ``z`` (kz), ``m`` and ``s`` for CylindricalBasis; ``x``, ``y``, ``z``
    (qx, qy, qz) and ``s`` for PlaneWaveBasis; and ``x``, ``y`` (the two stored
    components, in alignment order) and ``s`` for PlaneWavePorts.
    ``basis[()]`` returns every column. Set operators (``|``, ``&``, ``-``,
    ``^``, ``<=``, ...) combine bases of one family and geometry.
    """

    lattice: Lattice | None = None
    """Lattice of a periodic arrangement, or None."""
    kpar: WaveVector | None = None
    """Bloch vector of a periodic arrangement, or None."""
    modes: tuple[M, ...]
    """The mode labels, in order."""
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

    @cached_property
    def _lookup(self) -> dict[M, int]:
        """Index of each mode label; labels are distinct."""
        return {mode: index for index, mode in enumerate(self.modes)}

    def index(self, mode: M, start: int = 0, stop: int | None = None) -> int:
        """Index of ``mode``, like ``tuple.index``; ValueError when it is absent."""
        if start == 0 and stop is None:
            try:
                return self._lookup[mode]
            except (KeyError, TypeError):
                pass  # tuple.index raises the usual ValueError
        return self.modes.index(mode, start, len(self) if stop is None else stop)

    def count(self, mode: M) -> int:
        """Number of times ``mode`` occurs: 0 or 1, because modes are distinct."""
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

    def _geometry_mismatch(self, other: Self) -> str | None:
        """Why a basis of the same family has different geometry, or None."""
        return None

    def _inherit[T: _Basis[Any]](self, result: T, method: str = "", *args: float) -> T:
        """Give ``result`` this basis's lattice and kpar, transformed by ``method``."""
        lattice, kpar = self.lattice, self.kpar
        if method:
            lattice = None if lattice is None else getattr(lattice, method)(*args)
            kpar = None if kpar is None else getattr(kpar, method)(*args)
        result.lattice, result.kpar = lattice, kpar
        return result

    def _sets(self, other: Self) -> tuple[set[M], set[M]]:
        if type(other) is not type(self):
            raise TypeError("basis operations require the same family")
        mismatch = self._geometry_mismatch(other)
        if mismatch is not None:
            raise ValueError(mismatch)
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
        """Whether the bases share no mode; both need one family and geometry."""
        left, right = self._sets(other)
        return left.isdisjoint(right)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _Basis)
            and type(other) is type(self)
            and self.modes == other.modes
            and self._geometry_mismatch(other) is None
        )


class PlaneWaveBasis(_Basis[tuple[complex, complex, complex, int]]):
    """Ordered plane modes ``(qx, qy, qz, pol)`` with complex unit directions.

    Mirrors ``treams.PlaneWaveBasisByUnitVector``. Directions are normalized by
    the algebraic norm sqrt(q.q), without complex conjugation; zero or null
    directions are invalid. Labels pol are 0 or 1; the wave or operator that
    uses the basis sets the polarization convention. The basis stores
    directions, not wavevectors; ``kvecs`` scales them with k0 and the medium.
    Exact duplicate rows are removed in order.
    """

    isglobal = True
    """Always True: plane waves have no expansion positions."""
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
            raise ValueError("pol labels must be 0 or 1")
        self.directions = unit_vectors(values[:, :3])
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
        return self._inherit(result)

    @property
    def qx(self) -> NDArray[np.complex128]:
        """x component of each unit direction, shape (modes,)."""
        return self.directions[:, 0]

    @property
    def qy(self) -> NDArray[np.complex128]:
        """y component of each unit direction, shape (modes,)."""
        return self.directions[:, 1]

    @property
    def qz(self) -> NDArray[np.complex128]:
        """z component of each unit direction, shape (modes,)."""
        return self.directions[:, 2]

    @cached_property
    def pol(self) -> NDArray[np.int64]:
        """Label ``pol`` (0 or 1) of each mode, shape (modes,); read-only and cached."""
        result = np.array([m[3] for m in self.modes], dtype=np.int64)
        result.flags.writeable = False
        return result

    @classmethod
    def default(cls, kvecs: ArrayLike) -> PlaneWaveBasis:
        """Normalize shape-(n, 3) directions and include pol 0 and 1.

        A single shape-(3,) vector is accepted. Each direction contributes
        labels (1, 0), in that order. Complex directions allow evanescent waves.
        """
        vectors = np.atleast_2d(np.asarray(kvecs, dtype=np.complex128))
        if vectors.shape[1] != 3:
            raise ValueError("wavevectors require shape (n, 3)")
        return cls((*vector, pol) for vector in vectors for pol in (1, 0))

    def kvecs(
        self, k0: float, material: MaterialLike = 1, modetype: str | None = None
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
        """Return (kx, ky, kz), each shape (modes,), in inverse length units.

        Each direction is scaled by its medium wavenumber at the vacuum angular
        wavenumber k0. The direction fixes the propagation, so ``modetype`` has no
        effect; it is accepted for a signature shared with PlaneWavePorts.kvecs.
        """
        values = self.directions * Material(material).ks(k0)[self.pol, None]
        return values[:, 0], values[:, 1], values[:, 2]

    def rotate(self, phi: float) -> PlaneWaveBasis:
        """Rotate direction labels around z, preserving their pol labels.

        Quarter turns are exact, matching the rotated lattice metadata.
        """
        c, s = cos_sin(phi)
        result = type(self)(
            (c * x - s * y, s * x + c * y, z, p) for x, y, z, p in self.modes
        )
        return self._inherit(result, "rotate", phi)

    def permute(self, n: int = 1) -> PlaneWaveBasis:
        """Permute the direction components cyclically, x -> y -> z, n times.

        Labels pol stay; lattice and Bloch vector permute along.
        """
        vectors = np.roll(self.directions, turns(n), axis=1)
        result = type(self)(
            (*v, int(p)) for v, p in zip(vectors, self.pol, strict=True)
        )
        return self._inherit(result, "permute", n)

    def bycomp(
        self, k0: float, alignment: str = "xy", material: MaterialLike = 1
    ) -> PlaneWavePorts:
        """The same modes as PlaneWavePorts, which store two wavevector components.

        Mirrors ``treams.PlaneWaveBasisByUnitVector.bycomp``. The components named
        by ``alignment`` (``"xy"``, ``"yz"`` or ``"zx"``) of every wavevector at
        k0 in ``material`` must be real.
        """
        if alignment not in ("xy", "yz", "zx"):
            raise ValueError("alignment must be xy, yz or zx")
        axes = ["xyz".index(axis) for axis in alignment]
        q = np.real_if_close(np.column_stack(self.kvecs(k0, material))[:, axes])
        if np.iscomplexobj(q):
            raise ValueError("component bases require real transverse wavevectors")
        result = PlaneWavePorts(
            (
                (float(v[0]), float(v[1]), int(p))
                for v, p in zip(q, self.pol, strict=True)
            ),
            alignment,
        )
        return self._inherit(result)


class PlaneWavePorts(_Basis[tuple[float, float, int]]):
    """Ordered plane modes ``(k1, k2, pol)`` storing two real wavevector components.

    Mirrors ``treams.PlaneWaveBasisByComp``. The alignment ``"xy"``, ``"yz"``
    or ``"zx"`` names the stored Cartesian components (k1, k2), in inverse
    length units. Labels pol are 0 or 1; the wave or operator that uses the
    basis sets the polarization convention. ``kvecs`` adds the third component
    from k0, the medium and the direction up or down, including evanescent
    waves. Exact duplicate modes are removed in order.
    """

    isglobal = True
    """Always True: plane waves have no expansion positions."""
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
                raise ValueError("plane components must be finite and pol 0 or 1")
            values.append((float(kx), float(ky), int(pol)))
        self.modes: tuple[tuple[float, float, int], ...] = tuple(dict.fromkeys(values))

    @override
    def _from_modes(self, modes: tuple[tuple[float, float, int], ...]) -> Self:
        return self._inherit(type(self)(modes, self.alignment))

    @override
    def _geometry_mismatch(self, other: Self) -> str | None:
        if self.alignment != other.alignment:
            return "basis operations require identical plane alignment"
        return None

    @property
    def _k1(self) -> NDArray[np.float64]:
        return self.components[:, 0]

    @property
    def _k2(self) -> NDArray[np.float64]:
        return self.components[:, 1]

    @classmethod
    def default(cls, kpars: ArrayLike, alignment: str = "xy") -> PlaneWavePorts:
        """Include labels (1, 0) for each shape-(n, 2) transverse wavevector.

        A single shape-(2,) pair is accepted. Components are real and dimensional,
        in the Cartesian order specified by alignment (xy, yz or zx).
        """
        values = np.atleast_2d(np.asarray(kpars, dtype=np.float64))
        if values.shape[1] != 2:
            raise ValueError("transverse wavevectors require shape (n, 2)")
        return cls(
            ((float(kx), float(ky), pol) for kx, ky in values for pol in (1, 0)),
            alignment,
        )

    @classmethod
    def diffr_orders(
        cls,
        kpar: ArrayLike,
        lattice: ArrayLike,
        bmax: float,
        *,
        alignment: str | None = None,
    ) -> PlaneWavePorts:
        """Pol 0 and 1 of all reciprocal vectors with length <= bmax.

        The cutoff applies before adding the Bloch vector. Opposite diffraction
        orders are adjacent, following treams ordering for rectangular lattices.
        """
        cell = (
            lattice
            if isinstance(lattice, Lattice) and alignment in (None, lattice.alignment)
            else Lattice(lattice, alignment)
        )
        a, q = geometry_inputs(cell, kpar, cell.alignment)
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
        reciprocal = cell.reciprocal
        orders = _native.diffraction_orders(reciprocal, bmax)
        result = cls.default(q + orders @ reciprocal, cell.alignment)
        result.lattice, result.kpar = cell, WaveVector(q, cell.alignment)
        return result

    @property
    def normal_axis(self) -> int:
        """Cartesian axis normal to the stored component plane."""
        return ALIGNMENT_AXIS[self.alignment]

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
        """x component of each wavevector, shape (modes,); None if not stored."""
        return self._component("x")

    @property
    def ky(self) -> NDArray[np.float64] | None:
        """y component of each wavevector, shape (modes,); None if not stored."""
        return self._component("y")

    @property
    def kz(self) -> NDArray[np.float64] | None:
        """z component of each wavevector, shape (modes,); None if not stored."""
        return self._component("z")

    def rotate(self, phi: float) -> PlaneWavePorts:
        """Rotate xy direction labels around z, preserving their pol labels.

        Quarter turns are exact, matching the rotated lattice metadata.
        """
        if self.alignment != "xy":
            raise ValueError("z rotation of component bases requires xy alignment")
        c, s = cos_sin(phi)
        result = type(self)((c * x - s * y, s * x + c * y, p) for x, y, p in self.modes)
        return self._inherit(result, "rotate", phi)

    def permute(self, n: int = 1) -> PlaneWavePorts:
        """Permute the alignment cyclically, xy -> yz -> zx, n times.

        The stored components and labels pol stay; lattice and Bloch
        vector permute along.
        """
        alignments = ("xy", "yz", "zx")
        result = type(self)(
            self.modes, alignments[(alignments.index(self.alignment) + turns(n)) % 3]
        )
        return self._inherit(result, "permute", n)

    @cached_property
    def pol(self) -> NDArray[np.int64]:
        """Label ``pol`` (0 or 1) of each mode, shape (modes,); read-only and cached."""
        result = np.array([m[2] for m in self.modes], dtype=np.int64)
        result.flags.writeable = False
        return result

    def kvecs(
        self, k0: float, material: MaterialLike = 1, modetype: str = "up"
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
        """Return (kx, ky, kz), each shape (modes,), at vacuum wavenumber k0.

        The missing normal component uses the medium's outgoing square-root
        branch. ``modetype='down'`` reverses that component; ``'up'`` keeps it.
        Stored transverse components remain unchanged, in inverse length units.
        """
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
    ) -> PlaneWaveBasis:
        """The same modes as a PlaneWaveBasis of unit directions.

        Mirrors ``treams.PlaneWaveBasisByComp.byunitvector``. The third component
        comes from k0, ``material`` and ``modetype`` (``"up"`` or ``"down"``), as
        in ``kvecs``.
        """
        vectors = np.column_stack(self.kvecs(k0, material, modetype))
        result = PlaneWaveBasis(
            (*v, int(p)) for v, p in zip(vectors, self.pol, strict=True)
        )
        return self._inherit(result)


def _positions(nmax: int, positions: ArrayLike | None) -> ArrayLike:
    return np.zeros((nmax, 3)) if positions is None else positions


class _WaveBasis[M: tuple[int, float, int, int]](_Basis[M]):
    """Multipole modes ``(pidx, label, m, pol)`` at Cartesian positions."""

    @staticmethod
    def _rows(
        modes: Iterable[Sequence[float]], message: str
    ) -> Iterator[tuple[float, ...]]:
        """Rows as (particle, ...) 4-tuples; 3-tuples imply particle 0."""
        for row in modes:
            mode = tuple(row)
            if len(mode) == 3:
                mode = (0, *mode)
            if len(mode) != 4:
                raise ValueError(message)
            yield mode

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
        # Distinct modes of this basis, or of one with the same positions, are
        # already valid for these read-only, shared positions.
        result = type(self).__new__(type(self))
        result.modes, result.positions = modes, self.positions
        return self._inherit(result)

    @override
    def _geometry_mismatch(self, other: Self) -> str | None:
        if not np.array_equal(self.positions, other.positions):
            return "basis operations require identical origin tables"
        return None

    def _column(self, index: int, dtype: type[np.generic]) -> NDArray[Any]:
        result = np.array([mode[index] for mode in self.modes], dtype=dtype)
        result.flags.writeable = False
        return result

    @cached_property
    def pidx(self) -> NDArray[np.int64]:
        """Particle index of each mode, shape (modes,).

        A read-only column cached per basis, like ``pol``; call ``.copy()`` to
        edit it.
        """
        return self._column(0, np.int64)

    @cached_property
    def m(self) -> NDArray[np.int64]:
        """Azimuthal order of each mode, shape (modes,).

        A read-only column cached per basis, like ``pol``; call ``.copy()`` to
        edit it.
        """
        return self._column(2, np.int64)

    @cached_property
    def pol(self) -> NDArray[np.int64]:
        """Label ``pol`` (0 or 1) of each mode, shape (modes,); read-only and cached."""
        return self._column(3, np.int64)

    @cached_property
    def isglobal(self) -> bool:
        """Whether every mode has the same particle index."""
        return len({mode[0] for mode in self.modes}) <= 1


class SphericalBasis(_WaveBasis[SphericalMode]):
    """Ordered modes ``(pidx, l, m, pol)`` and Cartesian expansion positions.

    Mirrors ``treams.SphericalWaveBasis``. Rows ``(l, m, pol)`` imply pidx=0.
    Positions have shape (particles, 3) in the simulation's length unit; a
    shape-(3,) position is also accepted. Require 1 <= l <= 128, -l <= m <= l
    and pol in {0, 1}. Exact duplicate modes are removed in order. The wave or
    operator that uses the basis sets the polarization convention: helicity
    labels 0/1 mean negative/positive, parity labels magnetic/electric.
    Selecting modes keeps the positions.
    """

    _labels = ("pidx", "l", "m", "pol")
    _keys = "plms"

    def __init__(
        self, modes: Iterable[Sequence[float]], positions: ArrayLike = ((0, 0, 0),)
    ):
        values: list[SphericalMode] = []
        for mode in self._rows(
            modes, "modes require (l, m, pol) or (particle, l, m, pol)"
        ):
            p, degree, order, pol = mode
            if (
                any(int(v) != v for v in mode)
                or p < 0
                or not 1 <= degree <= MAX_DEGREE
                or abs(order) > degree
                or pol not in (0, 1)
            ):
                raise ValueError("invalid spherical mode")
            values.append((int(p), int(degree), int(order), int(pol)))
        super().__init__(values, positions)

    @staticmethod
    def defaultdim(lmax: int, nmax: int = 1) -> int:
        """Return ``2*lmax*(lmax+2)*nmax`` channels for a complete default basis."""
        if lmax < 0 or nmax < 0:
            raise ValueError("degree and particle count must be nonnegative")
        return 2 * lmax * (lmax + 2) * nmax

    @staticmethod
    def defaultlmax(dim: int, nmax: int = 1) -> int:
        """Invert ``defaultdim``; reject dimensions that omit part of a mode shell."""
        if dim < 0 or nmax < 1:
            raise ValueError(
                "require nonnegative dimension and positive particle count"
            )
        lmax = math.isqrt(dim // (2 * nmax) + 1) - 1
        if SphericalBasis.defaultdim(lmax, nmax) != dim:
            raise ValueError("dimension does not define a complete spherical basis")
        return lmax

    @classmethod
    def default(
        cls, lmax: int, nmax: int = 1, positions: ArrayLike | None = None
    ) -> Self:
        """Create complete spherical mode shells at nmax expansion positions.

        Order is particle, l=1..lmax, m=-l..l, then pol=(1, 0). The dimension is
        ``2*lmax*(lmax+2)*nmax``. Require 0 <= lmax <= 128 and nmax >= 1; lmax=0
        makes an empty basis. Positions have shape (nmax, 3), defaulting to zeros,
        and use the simulation's length unit.
        """
        if not 0 <= lmax <= MAX_DEGREE or nmax < 1:
            raise ValueError(f"require 0 <= lmax <= {MAX_DEGREE} and nmax >= 1")
        return cls(
            (
                (p, degree, order, pol)
                for p in range(nmax)
                for degree in range(1, lmax + 1)
                for order in range(-degree, degree + 1)
                for pol in (1, 0)
            ),
            _positions(nmax, positions),
        )

    @classmethod
    def ebcm(
        cls,
        lmax: int,
        nmax: int = 1,
        mmax: int = -1,
        positions: ArrayLike | None = None,
    ) -> Self:
        """Complete mode shells in EBCM order: particle, then m, degree and pol=(1, 0).

        Mirrors ``treams.SphericalWaveBasis.ebcm``. Orders run over -mmax..mmax
        and degrees over max(1, |m|)..lmax; ``mmax=-1`` uses lmax. Require
        0 <= mmax <= lmax <= 128 and nmax >= 1. Positions have shape (nmax, 3),
        zeros by default.
        """
        if mmax == -1:
            mmax = lmax
        if not 0 <= mmax <= lmax <= MAX_DEGREE or nmax < 1:
            raise ValueError(
                f"require 0 <= mmax <= lmax <= {MAX_DEGREE} and positive particle count"
            )
        return cls(
            (
                (p, degree, order, pol)
                for p in range(nmax)
                for order in range(-mmax, mmax + 1)
                for degree in range(max(1, abs(order)), lmax + 1)
                for pol in (1, 0)
            ),
            _positions(nmax, positions),
        )

    @cached_property
    def l(self) -> NDArray[np.int64]:  # noqa: E743
        """Degree of each mode, shape (modes,).

        A read-only column cached per basis, like ``pol``; call ``.copy()`` to
        edit it.
        """
        return self._column(1, np.int64)

    @property
    def lms(self) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """The columns ``(l, m, pol)``, as treams' ``lms``."""
        return self.l, self.m, self.pol


type CylindricalMode = tuple[int, float, int, int]


class CylindricalBasis(_WaveBasis[CylindricalMode]):
    """Ordered modes ``(pidx, kz, m, pol)`` around parallel z-directed axes.

    Mirrors ``treams.CylindricalWaveBasis``. Rows ``(kz, m, pol)`` imply
    pidx=0. kz is a finite real axial wavenumber in inverse length units,
    |m| <= 128 and pol is 0 or 1. The axis positions have shape (particles, 3)
    in the matching length unit. Exact duplicate modes are removed in order.
    The wave or operator that uses the basis sets the polarization convention:
    helicity labels 0/1 mean negative/positive, parity labels
    magnetic/electric.
    """

    _labels = ("pidx", "kz", "m", "pol")
    _keys = "pzms"

    def __init__(
        self, modes: Iterable[Sequence[float]], positions: ArrayLike = ((0, 0, 0),)
    ):
        values: list[CylindricalMode] = []
        for mode in self._rows(
            modes, "modes require (kz, m, pol) or (particle, kz, m, pol)"
        ):
            p, kz, order, pol = mode
            if (
                not math.isfinite(kz)
                or any(int(v) != v for v in (p, order, pol))
                or p < 0
                or abs(order) > MAX_DEGREE
                or pol not in (0, 1)
            ):
                raise ValueError("invalid cylindrical mode")
            values.append((int(p), float(kz), int(order), int(pol)))
        super().__init__(values, positions)

    @staticmethod
    def defaultdim(nkz: int, mmax: int, nmax: int = 1) -> int:
        """Return ``2*nkz*(2*mmax+1)*nmax`` channels for distinct axial labels."""
        return 2 * nkz * (2 * mmax + 1) * nmax

    @staticmethod
    def defaultmmax(dim: int, nkz: int = 1, nmax: int = 1) -> int:
        """Invert ``defaultdim`` for positive nkz/nmax; reject incomplete mode sets."""
        if nkz < 1 or nmax < 1:
            raise ValueError("nkz and nmax must be positive")
        mmax = (dim // (2 * nkz * nmax) - 1) // 2
        if mmax < 0 or CylindricalBasis.defaultdim(nkz, mmax, nmax) != dim:
            raise ValueError("dimension does not define a complete cylindrical basis")
        return mmax

    @classmethod
    def default(
        cls,
        kzs: ArrayLike,
        mmax: int,
        nmax: int = 1,
        positions: ArrayLike | None = None,
    ) -> Self:
        """Create pol 0 and 1 of orders -mmax..mmax for each axial label.

        Kzs is a scalar or one-dimensional real wavenumber array. Order is
        particle, kz in input order, m, then pol=(1, 0); duplicate labels are
        removed. Require 0 <= mmax <= 128 and nmax >= 1. Positions have shape
        (nmax, 3), defaulting to zeros, in the length unit inverse to kzs.
        """
        axial = np.atleast_1d(np.asarray(kzs, dtype=np.float64))
        if axial.ndim != 1 or not 0 <= mmax <= MAX_DEGREE or nmax < 1:
            raise ValueError(
                f"require a kz vector, 0 <= mmax <= {MAX_DEGREE} and nmax >= 1"
            )
        return cls(
            (
                (p, float(kz), order, pol)
                for p in range(nmax)
                for kz in axial
                for order in range(-mmax, mmax + 1)
                for pol in (1, 0)
            ),
            _positions(nmax, positions),
        )

    @classmethod
    def diffr_orders(
        cls,
        kz: float,
        mmax: int,
        lattice: float | Lattice,
        bmax: float,
        nmax: int = 1,
        positions: ArrayLike | None = None,
    ) -> Self:
        """Modes at the axial diffraction orders of a chain with period along z.

        Mirrors ``treams.CylindricalWaveBasis.diffr_orders``. The axial
        wavenumbers are ``kz + 2 pi j / period`` for every integer j with
        ``|2 pi j / period| <= bmax``, each with orders -mmax..mmax and both pol,
        ordered as in ``default``. ``lattice`` is the period or a 1D Lattice. The
        basis keeps the lattice and the Bloch vector kz.
        """
        cell = Lattice(lattice, "z")
        period = cell.volume
        if (
            not math.isfinite(period)
            or period == 0
            or not math.isfinite(bmax)
            or bmax < 0
        ):
            raise ValueError("require a finite nonzero period and nonnegative cutoff")
        reciprocal = 2 * np.pi / period
        count = math.floor(bmax / abs(reciprocal))
        result = cls.default(
            kz + np.arange(-count, count + 1) * reciprocal, mmax, nmax, positions
        )
        result.lattice, result.kpar = cell, WaveVector(kz)
        return result

    @cached_property
    def kz(self) -> NDArray[np.float64]:
        """Axial wavenumber label of each mode, shape (modes,).

        A read-only column cached per basis, like ``pol``; call ``.copy()`` to
        edit it.
        """
        return self._column(1, np.float64)

    @property
    def zms(self) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        """The columns ``(kz, m, pol)``, as treams' ``zms``."""
        return self.kz, self.m, self.pol


type FieldBasis = SphericalBasis | CylindricalBasis | PlaneWavePorts | PlaneWaveBasis


def mode_lookup(basis: FieldBasis) -> dict[tuple[object, ...], int]:
    """Index of each mode label of ``basis``, cached per basis."""
    return cast("dict[tuple[object, ...], int]", basis._lookup)

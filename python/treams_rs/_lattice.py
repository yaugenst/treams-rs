"""Lattice and WaveVector (treams._lattice) and their native cell rows."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, overload, override

import numpy as np

from . import _native

if TYPE_CHECKING:
    from types import EllipsisType

    from numpy.typing import ArrayLike, DTypeLike, NDArray


__all__ = [
    "Lattice",
    "WaveVector",
    "cell_geometry",
    "framework_cell",
    "geometry_inputs",
    "infer_dimension",
    "on_diffraction_orders",
    "periodic_alignment",
    "periodic_geometry",
    "sum_cell",
    "turns",
    "z_rotation",
]

_ALIGNMENTS = {1: ("z", "x", "y"), 2: ("xy", "yz", "zx"), 3: ("xyz",)}


def _alignment(axes: set[str]) -> str:
    value = "".join(sorted(axes))
    return "zx" if value == "xz" else value


def periodic_alignment(dim: int, spherical: bool) -> str:
    """Cartesian axes of a periodic cell: z (spheres) or x (cylinders) in 1D."""
    return ("z" if spherical else "x") if dim == 1 else "xyz"[:dim]


def on_diffraction_orders(orders: ArrayLike) -> bool:
    """Whether (possibly complex) reciprocal coordinates are integer orders."""
    values = np.asarray(orders)
    return np.allclose(values, np.round(values.real), atol=1e-10, rtol=0)


def turns(n: int) -> int:
    """Permutation count modulo 3; non-integer counts raise ValueError."""
    value = int(n)
    if value != n:
        raise ValueError("permutation count must be an integer")
    return value % 3


def z_rotation(phi: float) -> NDArray[np.float64]:
    """Rotation matrix about z, exact at quarter turns."""
    if not math.isfinite(phi):
        raise ValueError("rotation angle must be finite")
    c, s = math.cos(phi), math.sin(phi)
    if math.remainder(phi, math.pi / 2) == 0:
        c, s = round(c), round(s)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


@dataclass(frozen=True, init=False, eq=False)
class Lattice:
    """Lattice vectors as rows, along named Cartesian axes.

    Mirrors ``treams.Lattice``. The array is a scalar period (1D), the
    diagonal of 2 or 3 lengths, or a square 2x2 or 3x3 matrix of row vectors.
    ``alignment`` names the Cartesian axes of the rows: ``"z"`` (default),
    ``"x"`` or ``"y"`` in 1D; ``"xy"`` (default), ``"yz"`` or ``"zx"`` in 2D;
    ``"xyz"`` in 3D. ``Lattice(lattice, "xy")`` selects the sublattice of a 3D
    lattice along x and y. The Rust core computes the volume and the reciprocal
    vectors.
    """

    _array: NDArray[np.float64]
    _reciprocal: NDArray[np.float64]
    alignment: str
    """Cartesian axes of the rows, for example ``"xy"``."""

    def __init__(self, arr: ArrayLike | Lattice, alignment: str | None = None):
        if isinstance(arr, Lattice):
            alignment = arr.alignment if alignment is None else alignment
            values = arr._sublattice(alignment)
        else:
            values = np.array(arr, dtype=np.float64)
            if values.ndim < 3 and values.size == 1:
                values = values.reshape(())
            elif values.ndim == 1 and values.size in (2, 3):
                values = np.diag(values)
            elif (
                values.ndim != 2
                or values.shape[0] != values.shape[1]
                or values.shape[0] not in (2, 3)
            ):
                raise ValueError(
                    "lattice requires a scalar, diagonal or square matrix in dimension 1, 2 or 3"
                )
        dim = 1 if values.ndim == 0 else values.shape[0]
        alignment = _ALIGNMENTS[dim][0] if alignment is None else alignment
        if alignment not in _ALIGNMENTS[dim]:
            raise ValueError(f"invalid lattice alignment: {alignment}")
        reciprocal = _native.cell_reciprocal(np.atleast_2d(values))
        if dim == 1:
            reciprocal = reciprocal.reshape(())
        values.flags.writeable = False
        reciprocal.flags.writeable = False
        object.__setattr__(self, "_array", values)
        object.__setattr__(self, "_reciprocal", reciprocal)
        object.__setattr__(self, "alignment", alignment)

    @property
    def dim(self) -> int:
        """Number of lattice dimensions, 1 to 3."""
        return len(self.alignment)

    @property
    def volume(self) -> float:
        """Signed pitch, area or volume."""
        return float(_native.cell_volume(np.atleast_2d(self._array)))

    @property
    def reciprocal(self) -> NDArray[np.float64]:
        """Reciprocal row vectors satisfying a @ b.T = 2 pi I."""
        return self._reciprocal

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        return np.array(self._array, dtype=dtype, copy=copy)

    def __getitem__(
        self,
        key: int
        | slice
        | EllipsisType
        | NDArray[np.integer | np.bool_]
        | tuple[int | slice | EllipsisType, ...],
    ) -> NDArray[np.float64] | np.float64:
        return self._array[key]

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Lattice)
            and self.alignment == other.alignment
            and np.array_equal(self._array, other._array)
        )

    @override
    def __str__(self) -> str:
        return str(self._array)

    @override
    def __repr__(self) -> str:
        values = str(self._array).replace("\n", "\n        ")
        return f"Lattice({values}, alignment='{self.alignment}')"

    @classmethod
    def square(cls, pitch: float, alignment: str | None = None) -> Lattice:
        """Square lattice with side ``pitch``.

        Args:
            pitch: Lattice constant.
            alignment: Plane of the lattice: ``"xy"`` (default), ``"yz"`` or
                ``"zx"``. The rows lie along its first and second axis.
        """
        return cls([pitch, pitch], alignment)

    @classmethod
    def cubic(cls, pitch: float, alignment: str | None = None) -> Lattice:
        """Cubic lattice with side ``pitch``; the alignment can only be ``"xyz"``."""
        return cls([pitch, pitch, pitch], alignment)

    @classmethod
    def rectangular(cls, x: float, y: float, alignment: str | None = None) -> Lattice:
        """Rectangular lattice with sides ``x`` and ``y``.

        Args:
            x: Length along the first axis of the alignment.
            y: Length along the second axis of the alignment.
            alignment: Plane of the lattice: ``"xy"`` (default), ``"yz"`` or
                ``"zx"``. The rows lie along its first and second axis.
        """
        return cls([x, y], alignment)

    @classmethod
    def orthorhombic(
        cls, x: float, y: float, z: float, alignment: str | None = None
    ) -> Lattice:
        """Orthorhombic lattice with sides ``x``, ``y`` and ``z``.

        The alignment can only be ``"xyz"``.
        """
        return cls([x, y, z], alignment)

    @classmethod
    def hexagonal(
        cls, pitch: float, height: float | None = None, alignment: str | None = None
    ) -> Lattice:
        """Hexagonal lattice with side ``pitch``, 2D or, with ``height``, 3D.

        The rows are (pitch, 0) and (pitch/2, sqrt(3) pitch/2) in the plane of
        the alignment. ``height`` adds the row (0, 0, height) and needs the
        alignment ``"xyz"``.

        Args:
            pitch: Lattice constant in the plane.
            height: Period along the third axis; None for a 2D lattice.
            alignment: Plane of a 2D lattice: ``"xy"`` (default), ``"yz"`` or
                ``"zx"``.
        """
        values = np.array([[pitch, 0], [0.5 * pitch, np.sqrt(0.75) * pitch]])
        if height is not None:
            values = np.pad(values, ((0, 1), (0, 1)))
            values[2, 2] = height
        return cls(values, alignment)

    def _sublattice(self, alignment: str) -> NDArray[np.float64]:
        if alignment not in _ALIGNMENTS.get(len(alignment), ()) or not set(
            alignment
        ) <= set(self.alignment):
            raise ValueError(f"unavailable sublattice: {alignment}")
        if alignment == self.alignment:
            return self._array.copy()
        columns = [self.alignment.index(axis) for axis in alignment]
        remaining = [i for i in range(self.dim) if i not in columns]
        selected = np.any(self._array[:, columns] != 0, axis=1)
        if np.count_nonzero(selected) != len(columns) or np.any(
            self._array[selected][:, remaining] != 0
        ):
            raise ValueError("cannot determine sublattice")
        result = self._array[selected][:, columns]
        return result.reshape(()) if len(columns) == 1 else result

    def permute(self, n: int = 1) -> Lattice:
        """Permute the axes cyclically, x -> y -> z, n times."""
        shift = turns(n)
        if self.dim == 3:
            return Lattice(np.roll(self._array, shift, axis=1), self.alignment)
        alignment = "".join("xyz"[("xyz".index(c) + shift) % 3] for c in self.alignment)
        return Lattice(self._array, alignment)

    def rotate(self, phi: float) -> Lattice:
        """Rotate around z; the resulting span must remain Cartesian aligned."""
        axes = ["xyz".index(axis) for axis in self.alignment]
        rotation = z_rotation(phi)[:, axes].T
        columns = np.flatnonzero(np.any(rotation != 0, axis=0))
        if len(columns) != self.dim:
            raise ValueError(
                "rotation produces a lattice span outside Cartesian alignment"
            )
        alignment = _alignment({"xyz"[i] for i in columns})
        columns = ["xyz".index(axis) for axis in alignment]
        values = np.atleast_2d(self._array) @ rotation[:, columns]
        return Lattice(values, alignment)

    def __le__(self, other: Lattice) -> bool:
        try:
            return self == Lattice(other, self.alignment)
        except ValueError:
            return False

    def __or__(self, other: Lattice | None) -> Lattice:
        if other is None or self == other:
            return Lattice(self)
        axes = set(self.alignment) | set(other.alignment)
        alignment = _alignment(axes)
        for parent, child in ((self, other), (other, self)):
            if set(parent.alignment) == axes:
                if child <= parent:
                    return Lattice(parent)
                raise ValueError("cannot combine lattices")
        overlap = set(self.alignment) & set(other.alignment)
        if overlap:
            # Two partially overlapping planes can merge only when their shared
            # axes are separable sublattices, as treams requires.
            values = []
            for axis in alignment:
                candidates = [
                    Lattice(parent, axis)
                    for parent in (self, other)
                    if axis in parent.alignment
                ]
                if len(candidates) == 2 and candidates[0] != candidates[1]:
                    raise ValueError("cannot combine lattices")
                values.append(float(candidates[0]._array))
            return Lattice(values, alignment)
        values = np.zeros((len(alignment), len(alignment)))
        for parent in (self, other):
            index = [alignment.index(c) for c in parent.alignment]
            values[np.ix_(index, index)] = np.atleast_2d(parent._array)
        return Lattice(values, alignment)

    def __and__(self, other: Lattice | None) -> Lattice | None:
        if other is None:
            return None
        alignment = _alignment(set(self.alignment) & set(other.alignment))
        if not alignment:
            raise ValueError("cannot intersect disjoint lattices")
        left, right = Lattice(self, alignment), Lattice(other, alignment)
        if left != right:
            raise ValueError("cannot combine lattices")
        return left

    def isdisjoint(self, other: Lattice) -> bool:
        """Whether the two lattices share no Cartesian axis."""
        return set(self.alignment).isdisjoint(other.alignment)


@dataclass(frozen=True, init=False, eq=False)
class WaveVector(Sequence[complex | float]):
    """Partial Cartesian wavevector; NaN marks an unspecified component.

    Mirrors ``treams.WaveVector``. Three values give (kx, ky, kz); one or two
    values give the components along ``alignment`` (``"z"`` or ``"xy"`` by
    default), and the others are NaN. ``&`` combines compatible constraints;
    ``|`` keeps only the common ones.
    """

    _values: tuple[complex | float, ...]

    def __init__(self, seq: ArrayLike | WaveVector = (), alignment: str | None = None):
        if isinstance(seq, WaveVector):
            values = seq._values
        else:
            array = np.atleast_1d(seq)
            if array.ndim != 1 or array.size > 3 or array.dtype.kind not in "biufc":
                raise ValueError("wavevector requires zero to three numeric components")
            values = tuple(array.tolist())
        if len(values) in (0, 3):
            result = values if values else (np.nan,) * 3
        else:
            alignment = _ALIGNMENTS[len(values)][0] if alignment is None else alignment
            if alignment not in _ALIGNMENTS[len(values)]:
                raise ValueError(f"invalid wavevector alignment: {alignment}")
            result = tuple(
                values[alignment.index(axis)] if axis in alignment else np.nan
                for axis in "xyz"
            )
        object.__setattr__(self, "_values", result)

    @override
    def __len__(self) -> int:
        return 3

    @overload
    def __getitem__(self, key: int) -> complex | float: ...
    @overload
    def __getitem__(self, key: slice) -> tuple[complex | float, ...]: ...
    @override
    def __getitem__(
        self, key: int | slice
    ) -> complex | float | tuple[complex | float, ...]:
        return self._values[key]

    @override
    def __iter__(self) -> Iterator[complex | float]:
        return iter(self._values)

    @override
    def __str__(self) -> str:
        return str(self._values)

    @override
    def __repr__(self) -> str:
        return f"WaveVector{self._values}"

    @override
    def __eq__(self, other: object) -> bool:
        try:
            other = WaveVector(cast("ArrayLike | WaveVector", other))
        except (ValueError, TypeError):
            return False
        return all(
            a == b or (np.isnan(a) and np.isnan(b))
            for a, b in zip(self, other, strict=True)
        )

    def _combine(self, other: ArrayLike | WaveVector, common: bool) -> WaveVector:
        other = WaveVector(other)
        result = []
        for a, b in zip(self, other, strict=True):
            if a != b and not (np.isnan(a) or np.isnan(b)):
                raise ValueError("non-matching wavevector constraints")
            result.append(
                np.nan
                if common and (np.isnan(a) or np.isnan(b))
                else b
                if np.isnan(a)
                else a
            )
        return WaveVector(result)

    def __or__(self, other: ArrayLike | WaveVector) -> WaveVector:
        return self._combine(other, True)

    def __and__(self, other: ArrayLike | WaveVector) -> WaveVector:
        return self._combine(other, False)

    def __le__(self, other: ArrayLike | WaveVector) -> bool:
        return all(
            a == b or np.isnan(b) for a, b in zip(self, WaveVector(other), strict=True)
        )

    def isdisjoint(self, other: ArrayLike | WaveVector) -> bool:
        """Whether no component is specified in both wavevectors."""
        return all(
            np.isnan(a) or np.isnan(b)
            for a, b in zip(self, WaveVector(other), strict=True)
        )

    def permute(self, n: int = 1) -> WaveVector:
        """Permute the components cyclically, x -> y -> z, n times."""
        shift = turns(n)
        return WaveVector(
            self._values[-shift:] + self._values[:-shift] if shift else self._values
        )

    def rotate(self, phi: float) -> WaveVector:
        """Rotate around z without discarding partial wavevector constraints."""
        values = np.asarray(self._values)
        rotated = [np.sum(row[row != 0] * values[row != 0]) for row in z_rotation(phi)]
        if np.count_nonzero(np.isnan(rotated)) != np.count_nonzero(np.isnan(values)):
            raise ValueError(
                "rotation produces constraints outside Cartesian alignment"
            )
        return WaveVector(rotated)


def geometry_inputs(
    a: ArrayLike | Lattice, kpar: ArrayLike | WaveVector, alignment: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Lattice rows and Bloch vector of ``alignment`` as arrays, for a native call."""
    if isinstance(a, Lattice):
        values = a._array if a.alignment == alignment else a._sublattice(alignment)
    else:
        values = np.asarray(a, dtype=np.float64)
        if values.ndim == 1 and values.size in (2, 3):
            values = np.diag(values)
        if values.ndim == 2 and values.shape[0] > len(alignment):
            values = Lattice(values)._sublattice(alignment)
    if isinstance(kpar, WaveVector):
        kpar = tuple(kpar["xyz".index(axis)] for axis in alignment)
    return np.atleast_2d(values), np.atleast_1d(np.asarray(kpar, dtype=np.float64))


def infer_dimension(a: ArrayLike, kpar: ArrayLike) -> int:
    """Lattice dimension: the specified Bloch components, capped by the cell size.

    Only the shapes of arrays are read, so framework arrays stay unconverted.
    """
    dim = (
        int(np.count_nonzero(~np.isnan(np.atleast_1d(kpar))))
        if isinstance(kpar, WaveVector)
        else math.prod(np.shape(kpar))
    )
    return min(dim, (np.shape(a) or (1,))[0])


def periodic_geometry(
    a: ArrayLike, kpar: ArrayLike, spherical: bool
) -> tuple[list[list[float]], list[float]]:
    """Native cell rows and Bloch vector, with the inferred lattice dimension."""
    return cell_geometry(infer_dimension(a, kpar), a, kpar, spherical)


def cell_geometry(
    dim: int, a: ArrayLike, kpar: ArrayLike, spherical: bool
) -> tuple[list[list[float]], list[float]]:
    """Native cell rows and Bloch vector of a ``dim``-dimensional cell."""
    matrix, bloch = geometry_inputs(a, kpar, periodic_alignment(dim, spherical))
    if matrix.shape != (dim, dim) or bloch.shape != (dim,):
        raise ValueError("lattice and Bloch vector must match the requested dimension")
    return matrix.tolist(), bloch.tolist()


def sum_cell(
    dim: int, a: ArrayLike, kpar: ArrayLike, spherical: bool
) -> tuple[ArrayLike, ArrayLike]:
    """Native (lattice, Bloch) inputs of a per-dimension sum.

    Lattice/WaveVector metadata is resolved (to scalars in 1D) and 2D/3D
    diagonals are expanded; other raw 1D inputs pass through unchanged.
    """
    if isinstance(a, Lattice) or isinstance(kpar, WaveVector):
        matrix, bloch = cell_geometry(dim, a, kpar, spherical)
        return (matrix[0][0], bloch[0]) if dim == 1 else (np.asarray(matrix), bloch)
    if dim > 1:
        a = np.asarray(a)
        if a.shape == (dim,):
            a = np.diag(a)
    return a, kpar


def framework_cell(
    backend: Any, a: Any, kpar: Any, spherical: bool, dim: int | None = None
) -> tuple[Any, Any]:
    """Framework (lattice, Bloch) inputs, resolved as in ``geometry_inputs``.

    Lattice/WaveVector metadata gives the components of the cell's alignment,
    and a 2D/3D diagonal expands into rows that keep its derivatives. ``dim``
    defaults to the inferred dimension.
    """
    if not isinstance(a, Lattice):
        a = backend.array(a)
    if not isinstance(kpar, WaveVector):
        kpar = backend.array(kpar)
    dim = infer_dimension(a, kpar) if dim is None else dim
    alignment = periodic_alignment(dim, spherical)
    if isinstance(a, Lattice):
        a = backend.array(np.asarray(Lattice(a, alignment)))
    if isinstance(kpar, WaveVector):
        kpar = backend.array([kpar["xyz".index(axis)] for axis in alignment])
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif dim > 1 and a.shape == (dim,):
        a = a[:, None] * backend.array(np.eye(dim))
    return a, kpar.reshape(1) if kpar.ndim == 0 else kpar

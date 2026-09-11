"""Immutable lattice and partial-wavevector metadata in treams conventions."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, overload, override

import numpy as np

from . import _native

if TYPE_CHECKING:
    from types import EllipsisType

    from numpy.typing import ArrayLike, DTypeLike, NDArray


_ALIGNMENTS = {1: ("z", "x", "y"), 2: ("xy", "yz", "zx"), 3: ("xyz",)}


def _alignment(axes: set[str]) -> str:
    value = "".join(sorted(axes))
    return "zx" if value == "xz" else value


def _turns(n: int) -> int:
    value = int(n)
    if value != n:
        raise ValueError("permutation count must be an integer")
    return value % 3


@dataclass(frozen=True, init=False, eq=False)
class Lattice:
    """Row lattice vectors embedded along Cartesian axes; numerical geometry is native."""

    _array: NDArray[np.float64]
    _reciprocal: NDArray[np.float64]
    alignment: str

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

    def __bool__(self) -> bool:
        return True

    @classmethod
    def square(cls, pitch: float, alignment: str | None = None) -> Lattice:
        return cls([pitch, pitch], alignment)

    @classmethod
    def cubic(cls, pitch: float, alignment: str | None = None) -> Lattice:
        return cls([pitch, pitch, pitch], alignment)

    @classmethod
    def rectangular(cls, x: float, y: float, alignment: str | None = None) -> Lattice:
        return cls([x, y], alignment)

    @classmethod
    def orthorhombic(
        cls, x: float, y: float, z: float, alignment: str | None = None
    ) -> Lattice:
        return cls([x, y, z], alignment)

    @classmethod
    def hexagonal(
        cls, pitch: float, height: float | None = None, alignment: str | None = None
    ) -> Lattice:
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
        turns = _turns(n)
        if self.dim == 3:
            return Lattice(np.roll(self._array, turns, axis=1), self.alignment)
        alignment = "".join("xyz"[("xyz".index(c) + turns) % 3] for c in self.alignment)
        return Lattice(self._array, alignment)

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
            # axes are separable sublattices, matching upstream's contract.
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
        return set(self.alignment).isdisjoint(other.alignment)


@dataclass(frozen=True, init=False, eq=False)
class WaveVector(Sequence[complex | float]):
    """Partial Cartesian wavevector; NaN denotes an unspecified component.

    ``&`` combines compatible constraints; ``|`` keeps only common constraints.
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
        return all(
            np.isnan(a) or np.isnan(b)
            for a, b in zip(self, WaveVector(other), strict=True)
        )

    def permute(self, n: int = 1) -> WaveVector:
        turns = _turns(n)
        return WaveVector(
            self._values[-turns:] + self._values[:-turns] if turns else self._values
        )


def _geometry_inputs(
    a: ArrayLike | Lattice, kpar: ArrayLike | WaveVector, alignment: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resolve explicit metadata once at the numerical geometry boundary."""
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

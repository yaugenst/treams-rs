"""Argument validation shared by physical objects."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ._lattice import z_rotation

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NDArray

__all__ = [
    "MAX_DEGREE",
    "MAX_LABEL",
    "MAX_ORDER",
    "check_k0",
    "check_kind",
    "check_particle_positions",
    "cos_sin",
    "frozen",
    "one_of",
    "unit_vectors",
]

# Label limits of the native core, named as treams_core::MAX_DEGREE,
# special::MAX_ORDER and special::MAX_LABEL. treams has no such limits.
MAX_DEGREE = 128
"""The largest multipole degree l and cylindrical |m|: the native recurrence limit."""
MAX_ORDER = 2 * MAX_DEGREE
"""The largest radial-function order and cylindrical order difference."""
MAX_LABEL = 2 * MAX_DEGREE + 4
"""The largest |n| of a Kambe order or a Wigner label."""


def check_k0(k0: float) -> float:
    """The vacuum angular wavenumber as a float; it must be finite and positive."""
    # Floats skip the NumPy ufunc, which costs a microsecond on every evaluation;
    # other values keep its checks and exception types.
    finite = math.isfinite(k0) if isinstance(k0, float) else np.isfinite(k0)
    if not finite or k0 <= 0:
        raise ValueError("k0 must be finite and positive")
    return float(k0)


def one_of[T](
    name: str, value: T | None, alias: str, alias_value: T | None, default: T
) -> T:
    """The value given under ``name`` or its treams alias, else the default.

    Giving both raises, so neither keyword silently overrides the other.
    """
    if value is not None and alias_value is not None:
        raise ValueError(f"specify {name} or {alias}, not both")
    if value is not None:
        return value
    return default if alias_value is None else alias_value


def check_kind(kind: str) -> str:
    """Reject the ambiguous kind 'outgoing'; other kinds pass through unchanged."""
    if kind == "outgoing":
        raise ValueError(
            "kind must be 'regular' or 'singular'; outgoing multipole waves are "
            "'singular' (treams modetype)"
        )
    return kind


def check_particle_positions(
    positions: ArrayLike, *, cylindrical: bool = False
) -> None:
    """Particles need distinct centres; infinite cylinders need distinct axes."""
    points = np.asarray(positions)
    if cylindrical:
        points = points[:, :2]
    if len(np.unique(points, axis=0)) != len(points):
        message = (
            "cylinders require distinct transverse positions"
            if cylindrical
            else "particle modes must be grouped at distinct positions"
        )
        raise ValueError(message)


def frozen(value: ArrayLike, dtype: DTypeLike) -> NDArray[Any]:
    """An owned, read-only copy."""
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def cos_sin(phi: float) -> tuple[float, float]:
    """Cosine and sine of a z rotation, exact at quarter turns like the lattice."""
    rotation = z_rotation(phi)
    return float(rotation[0, 0]), float(rotation[1, 0])


def unit_vectors(vectors: ArrayLike) -> NDArray[np.complex128]:
    """Rows of shape (n, 3) scaled to the algebraic norm sqrt(q.q) = 1.

    The norm has no complex conjugation, so complex rows describe evanescent
    directions; rows with a zero norm raise ValueError.
    """
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

"""Mode selection, material branches and reciprocal-cell reduction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def refractive_index(
    epsilon: ArrayLike = 1, mu: ArrayLike = 1, kappa: ArrayLike = 0
) -> NDArray[np.float64 | np.complex128]:
    """Negative/positive helicity indices, with nonnegative imaginary parts."""
    return _native.refractive_indices(epsilon, mu, kappa)


def wave_vec_z(kx: ArrayLike, ky: ArrayLike, k: ArrayLike) -> NDArray[np.complex128]:
    """Normal wavenumber on the outgoing square-root branch, including zero."""
    return _native.wave_vector_z(kx, ky, k)


def _mode_rows(columns: ArrayLike) -> NDArray[np.generic]:
    values = np.asarray(columns)
    if values.ndim != 2 or values.shape[0] not in (3, 4):
        raise ValueError("modes require three or four equally sized column arrays")
    return values.T


def pickmodes(out: ArrayLike, in_: ArrayLike) -> NDArray[np.bool_]:
    """Exact mode-selection matrix, with output modes as rows."""
    destination, source = _mode_rows(out), _mode_rows(in_)
    return np.all(destination[:, None, :] == source, axis=-1)


def basischange(out: ArrayLike, in_: ArrayLike | None = None) -> NDArray[np.float64]:
    """Helicity/parity conversion between matching nonpolarization labels."""
    destination = _mode_rows(out)
    source = destination if in_ is None else _mode_rows(in_)
    match = np.all(destination[:, None, :-1] == source[:, :-1], axis=-1)
    negative = (destination[:, None, -1] == 0) & (source[:, -1] == 0)
    return np.where(negative, -np.sqrt(0.5), np.sqrt(0.5)) * match


firstbrillouin1d = _native.first_brillouin_1d


def firstbrillouin2d(kpar: ArrayLike, b: ArrayLike, n: int = 2) -> NDArray[np.float64]:
    """Reduce a wavevector using a minimal two-dimensional reciprocal basis."""
    return _native.first_brillouin(
        np.asarray(kpar, dtype=np.float64), np.asarray(b, dtype=np.float64), 2, n
    )


def firstbrillouin3d(kpar: ArrayLike, b: ArrayLike, n: int = 2) -> NDArray[np.float64]:
    """Iterative nearest-neighbour reduction in three dimensions."""
    return _native.first_brillouin(
        np.asarray(kpar, dtype=np.float64), np.asarray(b, dtype=np.float64), 3, n
    )

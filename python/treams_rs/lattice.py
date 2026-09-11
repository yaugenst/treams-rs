# ruff: noqa: E741 - preserve upstream multipole degree argument l
"""Native Ewald sums; multipole arrays broadcast over one lattice geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native
from ._core import CylindricalWaveBasis, SphericalWaveBasis
from ._lattice import WaveVector, _geometry_inputs

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

type SumResult = complex | np.ndarray[tuple[int, ...], np.dtype[np.complex128]]


def _geometry(
    dim: int, a: ArrayLike, kpar: ArrayLike, spherical: bool
) -> tuple[list[list[float]], list[float]]:
    alignment = ("z" if spherical else "x") if dim == 1 else "xyz"[:dim]
    matrix, bloch = _geometry_inputs(a, kpar, alignment)
    if matrix.shape != (dim, dim) or bloch.shape != (dim,):
        raise ValueError("lattice and Bloch vector must match the requested dimension")
    return matrix.tolist(), bloch.tolist()


def _sum(
    spherical: bool,
    dim: int,
    degree: ArrayLike,
    order: ArrayLike,
    k: complex,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: complex,
) -> SumResult:
    matrix, bloch = _geometry(dim, a, kpar, spherical)
    shift = np.asarray(r, dtype=np.float64)
    if shift.shape != (3 if spherical else 2,):
        raise ValueError(
            "shift requires three spherical or two cylindrical coordinates"
        )
    if not spherical:
        shift = np.append(shift, 0.0)
    degrees, orders = np.broadcast_arrays(degree, order)
    if np.any(degrees != np.floor(degrees)) or np.any(orders != np.floor(orders)):
        raise ValueError("multipole indices must be integers")
    modes = [
        (int(degree), int(order))
        for degree, order in zip(degrees.flat, orders.flat, strict=True)
    ]
    values = np.asarray(
        _native.lattice_sum(
            spherical,
            modes,
            k,
            bloch,
            matrix,
            (float(shift[0]), float(shift[1]), float(shift[2])),
            eta,
        ),
        dtype=np.complex128,
    ).reshape(degrees.shape)
    return complex(values) if values.ndim == 0 else values


def lsumsw(
    dim: int,
    l: ArrayLike,
    m: ArrayLike,
    k: complex,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: complex = 0,
) -> SumResult:
    """Spherical sum h_l(k|-r-R|) Y_lm(-r-R) exp(i kpar.R), excluding zero distance.

    A one-dimensional lattice runs along z; two-dimensional lattices lie in xy.
    Exact diffraction thresholds are singular. Supply loss or a limiting k.
    """
    return _sum(True, dim, l, m, k, kpar, a, r, eta)


def lsumcw(
    dim: int,
    m: ArrayLike,
    k: complex,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: complex = 0,
) -> SumResult:
    """Cylindrical sum; a one-dimensional lattice runs along x."""
    return _sum(False, dim, 0, m, k, kpar, a, r, eta)


def lsumsw1d(
    l: ArrayLike, k: complex, kpar: float, a: float, r: float, eta: complex = 0
) -> SumResult:
    return lsumsw(1, l, 0, k, kpar, a, [0, 0, r], eta)


def lsumsw1d_shift(
    l: ArrayLike,
    m: ArrayLike,
    k: complex,
    kpar: float,
    a: float,
    r: ArrayLike,
    eta: complex = 0,
) -> SumResult:
    return lsumsw(1, l, m, k, kpar, a, r, eta)


def lsumsw2d(
    l: ArrayLike,
    m: ArrayLike,
    k: complex,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: complex = 0,
) -> SumResult:
    return lsumsw(2, l, m, k, kpar, a, np.append(r, 0), eta)


def lsumsw2d_shift(
    l: ArrayLike,
    m: ArrayLike,
    k: complex,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: complex = 0,
) -> SumResult:
    return lsumsw(2, l, m, k, kpar, a, r, eta)


def lsumsw3d(
    l: ArrayLike,
    m: ArrayLike,
    k: complex,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: complex = 0,
) -> SumResult:
    return lsumsw(3, l, m, k, kpar, a, r, eta)


def lsumcw1d(
    m: ArrayLike, k: complex, kpar: float, a: float, r: float, eta: complex = 0
) -> SumResult:
    return lsumcw(1, m, k, kpar, a, [r, 0], eta)


def lsumcw1d_shift(
    m: ArrayLike, k: complex, kpar: float, a: float, r: ArrayLike, eta: complex = 0
) -> SumResult:
    return lsumcw(1, m, k, kpar, a, r, eta)


def lsumcw2d(
    m: ArrayLike,
    k: complex,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: complex = 0,
) -> SumResult:
    return lsumcw(2, m, k, kpar, a, r, eta)


def expansion_with_context(
    destination: SphericalWaveBasis | CylindricalWaveBasis,
    source: SphericalWaveBasis | CylindricalWaveBasis,
    ks: ArrayLike,
    a: ArrayLike,
    kpar: ArrayLike,
    *,
    poltype: str = "helicity",
    eta: complex = 0,
) -> tuple[NDArray[np.complex128], _native.PeriodicContext]:
    """Periodic outgoing-to-regular coupling, including nonzero self images.

    Cylindrical ``context.pullback_axial`` appends gradients of the sorted
    distinct axial wavenumbers from both bases, with matching groups fixed.
    """
    wavenumbers = np.asarray(ks, dtype=np.complex128)
    if wavenumbers.shape != (2,):
        raise ValueError(
            "context requires two medium wavenumbers, one per polarization"
        )
    if poltype not in ("helicity", "parity") or (
        poltype == "parity" and wavenumbers[0] != wavenumbers[1]
    ):
        raise ValueError("invalid polarization type for embedding medium")
    components = np.atleast_1d(kpar)
    dim = (
        int(np.count_nonzero(~np.isnan(components)))
        if isinstance(kpar, WaveVector)
        else components.size
    )
    dim = min(dim, np.atleast_2d(a).shape[0] if np.ndim(a) != 1 else np.size(a))
    matrix, bloch = _geometry(dim, a, kpar, isinstance(source, SphericalWaveBasis))
    pair = (complex(wavenumbers[0]), complex(wavenumbers[1]))
    if isinstance(destination, SphericalWaveBasis) and isinstance(
        source, SphericalWaveBasis
    ):
        return _native.periodic_expansion(
            list(destination.modes),
            list(source.modes),
            destination.positions.tolist(),
            source.positions.tolist(),
            pair,
            poltype == "helicity",
            bloch,
            matrix,
            eta,
        )
    if isinstance(destination, CylindricalWaveBasis) and isinstance(
        source, CylindricalWaveBasis
    ):
        return _native.periodic_cyl_expansion(
            list(destination.modes),
            list(source.modes),
            destination.positions.tolist(),
            source.positions.tolist(),
            pair,
            bloch,
            matrix,
            eta,
        )
    raise ValueError("periodic expansion requires matching wave families")


def expansion(
    destination: SphericalWaveBasis | CylindricalWaveBasis,
    source: SphericalWaveBasis | CylindricalWaveBasis,
    ks: ArrayLike,
    a: ArrayLike,
    kpar: ArrayLike,
    *,
    poltype: str = "helicity",
    eta: complex = 0,
) -> NDArray[np.complex128]:
    """Periodic outgoing-to-regular coupling, including nonzero self images.

    Cylindrical ``context.pullback_axial`` appends gradients of the sorted
    distinct axial wavenumbers from both bases, with matching groups fixed.
    """
    return expansion_with_context(
        destination,
        source,
        np.broadcast_to(np.asarray(ks, dtype=np.complex128), (2,)),
        a,
        kpar,
        poltype=poltype,
        eta=eta,
    )[0]


volume = _native.cell_volume
area = volume
reciprocal = _native.cell_reciprocal


def cube(d: int, n: int) -> NDArray[np.int64]:
    """All integer points in [-n,n]^d in lexicographic order, for d=1,2,3."""
    return _native.lattice_cube(d, n, False)


def cubeedge(d: int, n: int) -> NDArray[np.int64]:
    """Boundary points of [-n,n]^d, including the single origin when n=0."""
    return _native.lattice_cube(d, n, True)


def diffr_orders_circle(b: ArrayLike, rmax: float) -> NDArray[np.int64]:
    """All reciprocal orders within radius rmax, with adjacent opposite pairs.

    The enumeration covers skew lattices and preserves upstream ordering on
    orthogonal cells. Negative radii return an empty (0,2) integer array.
    """
    return _native.diffraction_orders(np.asarray(b, dtype=np.float64), rmax)

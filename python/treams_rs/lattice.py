# ruff: noqa: E741 - preserve upstream multipole degree argument l
"""Native Ewald components and direct shells with NumPy broadcasting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native
from ._core import CylindricalWaveBasis, SphericalWaveBasis
from ._lattice import Lattice, WaveVector, _geometry_inputs
from .config import _resolve_poltype

if TYPE_CHECKING:
    from typing import Any

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
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    parameter: ArrayLike,
    part: int,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if not 1 <= dim <= (3 if spherical else 2):
        raise ValueError("invalid lattice dimension")
    if isinstance(a, Lattice) or isinstance(kpar, WaveVector):
        a, kpar = _geometry(dim, a, kpar, spherical)
    a = np.asarray(a)
    kpar = np.asarray(kpar)
    if dim == 1:
        if a.shape == (1, 1):
            a = a[0, 0]
        if kpar.shape == (1,):
            kpar = kpar[0]
    elif a.shape == (dim,):
        a = np.diag(a)
    functions = _SPHERICAL_SUMS if spherical else _CYLINDRICAL_SUMS
    function = functions[part][dim - 1]
    args = (degree, order) if spherical else (order,)
    return function(*args, k, kpar, a, r, parameter, out=out, **kwargs)


def lsumsw(
    dim: int,
    l: ArrayLike,
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    """Ewald sum; row lattice vectors, excluding zero distance."""
    return _sum(True, dim, l, m, k, kpar, a, r, eta, 0, out, **kwargs)


def lsumcw(
    dim: int,
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    """Ewald sum; row lattice vectors, excluding zero distance."""
    return _sum(False, dim, 0, m, k, kpar, a, r, eta, 0, out, **kwargs)


def lsumsw1d(
    l: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if isinstance(a, Lattice) or isinstance(kpar, WaveVector):
        a, kpar = _geometry(1, a, kpar, True)
        a, kpar = a[0][0], kpar[0]
    return _native.lsumsw1d(l, k, kpar, a, r, eta, out=out, **kwargs)


def lsumsw1d_shift(
    l: ArrayLike,
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if isinstance(a, Lattice) or isinstance(kpar, WaveVector):
        a, kpar = _geometry(1, a, kpar, True)
        a, kpar = a[0][0], kpar[0]
    return _native.lsumsw1d_shift(l, m, k, kpar, a, r, eta, out=out, **kwargs)


def lsumsw2d(
    l: ArrayLike,
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if isinstance(a, Lattice) or isinstance(kpar, WaveVector):
        a, kpar = _geometry(2, a, kpar, True)
    a = np.asarray(a)
    if a.shape == (2,):
        a = np.diag(a)
    return _native.lsumsw2d(l, m, k, kpar, a, r, eta, out=out, **kwargs)


def lsumsw2d_shift(
    l: ArrayLike,
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if isinstance(a, Lattice) or isinstance(kpar, WaveVector):
        a, kpar = _geometry(2, a, kpar, True)
    a = np.asarray(a)
    if a.shape == (2,):
        a = np.diag(a)
    return _native.lsumsw2d_shift(l, m, k, kpar, a, r, eta, out=out, **kwargs)


def lsumsw3d(
    l: ArrayLike,
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if isinstance(a, Lattice) or isinstance(kpar, WaveVector):
        a, kpar = _geometry(3, a, kpar, True)
    a = np.asarray(a)
    if a.shape == (3,):
        a = np.diag(a)
    return _native.lsumsw3d(l, m, k, kpar, a, r, eta, out=out, **kwargs)


def lsumcw1d(
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if isinstance(a, Lattice) or isinstance(kpar, WaveVector):
        a, kpar = _geometry(1, a, kpar, False)
        a, kpar = a[0][0], kpar[0]
    return _native.lsumcw1d(m, k, kpar, a, r, eta, out=out, **kwargs)


def lsumcw1d_shift(
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if isinstance(a, Lattice) or isinstance(kpar, WaveVector):
        a, kpar = _geometry(1, a, kpar, False)
        a, kpar = a[0][0], kpar[0]
    return _native.lsumcw1d_shift(m, k, kpar, a, r, eta, out=out, **kwargs)


def lsumcw2d(
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if isinstance(a, Lattice) or isinstance(kpar, WaveVector):
        a, kpar = _geometry(2, a, kpar, False)
    a = np.asarray(a)
    if a.shape == (2,):
        a = np.diag(a)
    return _native.lsumcw2d(m, k, kpar, a, r, eta, out=out, **kwargs)


def realsumsw(
    dim: int,
    l: ArrayLike,
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    """Ewald real-space component; row lattice vectors, excluding zero distance."""
    return _sum(True, dim, l, m, k, kpar, a, r, eta, 1, out, **kwargs)


def realsumcw(
    dim: int,
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    """Ewald real-space component; row lattice vectors, excluding zero distance."""
    return _sum(False, dim, 0, m, k, kpar, a, r, eta, 1, out, **kwargs)


realsumsw1d = _native.realsumsw1d
realsumsw1d_shift = _native.realsumsw1d_shift
realsumsw2d = _native.realsumsw2d
realsumsw2d_shift = _native.realsumsw2d_shift
realsumsw3d = _native.realsumsw3d
realsumcw1d = _native.realsumcw1d
realsumcw1d_shift = _native.realsumcw1d_shift
realsumcw2d = _native.realsumcw2d


def recsumsw(
    dim: int,
    l: ArrayLike,
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    """Ewald reciprocal-space component; row lattice vectors, excluding zero distance."""
    return _sum(True, dim, l, m, k, kpar, a, r, eta, 2, out, **kwargs)


def recsumcw(
    dim: int,
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    """Ewald reciprocal-space component; row lattice vectors, excluding zero distance."""
    return _sum(False, dim, 0, m, k, kpar, a, r, eta, 2, out, **kwargs)


recsumsw1d = _native.recsumsw1d
recsumsw1d_shift = _native.recsumsw1d_shift
recsumsw2d = _native.recsumsw2d
recsumsw2d_shift = _native.recsumsw2d_shift
recsumsw3d = _native.recsumsw3d
recsumcw1d = _native.recsumcw1d
recsumcw1d_shift = _native.recsumcw1d_shift
recsumcw2d = _native.recsumcw2d


def dsumsw(
    dim: int,
    l: ArrayLike,
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    i: ArrayLike,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    """One integer cube shell; sum successive shells for a direct truncation."""
    return _sum(True, dim, l, m, k, kpar, a, r, i, 3, out, **kwargs)


def dsumcw(
    dim: int,
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    i: ArrayLike,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    """One integer cube shell; sum successive shells for a direct truncation."""
    return _sum(False, dim, 0, m, k, kpar, a, r, i, 3, out, **kwargs)


dsumsw1d = _native.dsumsw1d
dsumsw1d_shift = _native.dsumsw1d_shift
dsumsw2d = _native.dsumsw2d
dsumsw2d_shift = _native.dsumsw2d_shift
dsumsw3d = _native.dsumsw3d


def dsumcw1d(
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    i: ArrayLike,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if (
        out is None
        and not kwargs
        and isinstance(m, int)
        and isinstance(i, int)
        and isinstance(k, (float, complex))
        and isinstance(kpar, (int, float))
        and isinstance(a, (int, float))
        and isinstance(r, (int, float))
    ):
        return _native.direct_cylindrical_1d(m, k, kpar, a, r, i)
    return _native.dsumcw1d(m, k, kpar, a, r, i, out=out, **kwargs)


def dsumcw1d_shift(
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    i: ArrayLike,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if (
        out is None
        and not kwargs
        and isinstance(m, int)
        and isinstance(i, int)
        and isinstance(k, (float, complex))
        and isinstance(kpar, (int, float))
        and isinstance(a, (int, float))
        and isinstance(r, np.ndarray)
        and r.shape == (2,)
        and r.dtype == np.float64
    ):
        return _native.direct_cylindrical_1d_shift(m, k, kpar, a, r, i)
    return _native.dsumcw1d_shift(m, k, kpar, a, r, i, out=out, **kwargs)


def dsumcw2d(
    m: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    i: ArrayLike,
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if (
        out is None
        and not kwargs
        and isinstance(m, int)
        and isinstance(i, int)
        and isinstance(k, (float, complex))
        and isinstance(kpar, np.ndarray)
        and kpar.shape == (2,)
        and kpar.dtype == np.float64
        and isinstance(a, np.ndarray)
        and a.shape == (2, 2)
        and a.dtype == np.float64
        and isinstance(r, np.ndarray)
        and r.shape == (2,)
        and r.dtype == np.float64
    ):
        return _native.direct_cylindrical_2d(m, k, kpar, a, r, i)
    return _native.dsumcw2d(m, k, kpar, a, r, i, out=out, **kwargs)


_SPHERICAL_SUMS = (
    (_native.lsumsw1d_shift, _native.lsumsw2d_shift, _native.lsumsw3d),
    (_native.realsumsw1d_shift, _native.realsumsw2d_shift, _native.realsumsw3d),
    (_native.recsumsw1d_shift, _native.recsumsw2d_shift, _native.recsumsw3d),
    (_native.dsumsw1d_shift, _native.dsumsw2d_shift, _native.dsumsw3d),
)


_CYLINDRICAL_SUMS = (
    (_native.lsumcw1d_shift, _native.lsumcw2d),
    (_native.realsumcw1d_shift, _native.realsumcw2d),
    (_native.recsumcw1d_shift, _native.recsumcw2d),
    (_native.dsumcw1d_shift, _native.dsumcw2d),
)


def expansion_with_context(
    destination: SphericalWaveBasis | CylindricalWaveBasis,
    source: SphericalWaveBasis | CylindricalWaveBasis,
    ks: ArrayLike,
    a: ArrayLike,
    kpar: ArrayLike,
    *,
    poltype: str | None = None,
    eta: complex = 0,
) -> tuple[NDArray[np.complex128], _native.PeriodicContext]:
    """Periodic outgoing-to-regular coupling, including nonzero self images.

    Cylindrical ``context.pullback_axial`` appends gradients of the sorted
    distinct axial wavenumbers from both bases, with matching groups fixed.
    """
    poltype = _resolve_poltype(poltype)
    wavenumbers = np.asarray(ks, dtype=np.complex128)
    if wavenumbers.shape != (2,):
        raise ValueError(
            "context requires two medium wavenumbers, one per polarization"
        )
    if poltype == "parity" and wavenumbers[0] != wavenumbers[1]:
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
    poltype: str | None = None,
    eta: complex = 0,
) -> NDArray[np.complex128]:
    """Periodic outgoing-to-regular coupling, including nonzero self images.

    Cylindrical ``context.pullback_axial`` appends gradients of the sorted
    distinct axial wavenumbers from both bases, with matching groups fixed.
    """
    poltype = _resolve_poltype(poltype)
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

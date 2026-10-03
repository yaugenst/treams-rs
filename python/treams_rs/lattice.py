"""Lattice sums of spherical and cylindrical waves, and lattice geometry.

Mirrors ``treams.lattice``. A lattice of dimension d holds the points
``R = n_1 a_1 + ... + n_d a_d`` with integers ``n_i``. The rows of ``a`` are
the lattice vectors ``a_i``. For a wavenumber ``k``, a Bloch vector ``kpar``
(the real wavevector components along the lattice) and a shift ``r``, the
spherical and cylindrical sums are

    D_lm(k, kpar, r) = sum'_R h_l(k |r + R|) Y_lm(-r - R) exp(i kpar . R)
    D_m(k, kpar, r)  = sum'_R H_m(k |r + R|) exp(i m phi) exp(i kpar . R)

with the outgoing (first-kind) Hankel functions ``h_l`` and ``H_m``, the
spherical harmonic ``Y_lm`` in the direction of ``-r - R`` and the azimuth
``phi`` of ``-r - R``. The prime drops the term with ``r + R = 0``.

Spherical sums (``*sw*``) take a degree ``l`` and an order ``m`` and run over
lattices in 1, 2 or 3 dimensions; cylindrical sums (``*cw*``) take an order
``m`` and run over lattices in 1 or 2 dimensions. A 1D spherical lattice lies
on the z axis, a 1D cylindrical lattice on the x axis, and a 2D lattice in the
plane z = 0. In 1D, ``a`` is the period and ``kpar`` a number. In 2D and 3D,
``a`` is a (d, d) array of rows, or the d side lengths of a rectangular cell,
and ``kpar`` has d components. The functions that take the dimension ``dim``
as their first argument, and the ``lsum*`` functions, also take a
``Lattice`` for ``a`` and a ``WaveVector`` for ``kpar``, and use their
components along the lattice axes.

The Ewald method splits each sum into a real-space part (``realsum*``), a
sum over lattice points, and a reciprocal-space part (``recsum*``), a sum over
the diffraction orders ``kpar + G`` with reciprocal lattice vectors ``G``.
Both parts converge exponentially. The Ewald split parameter ``eta`` sets how
the sum divides between them; the exact sum does not depend on it. ``eta=0``
(the default) selects ``sqrt(2 pi) / (k L) max(|k L| / 8, 1)``, where ``L`` is
the length, the square root of the area or the cube root of the volume of the
cell. The direct sums ``dsum*`` add one shell of lattice points at a time and
serve as a check. ``volume``, ``area``, ``reciprocal``, ``cube``, ``cubeedge``
and ``diffr_orders_circle`` describe the lattice itself.

Accuracy
--------

The two Ewald parts grow and cancel as ``eta`` gets small. An explicit split
far below the automatic one loses about ``eps * exp(Re(1 / (2 * eta**2)))`` of
relative accuracy, with the machine precision ``eps = 2.2e-16``.

- A split is small when ``Re(1 / (2 * eta**2)) > 16 / pi``, which holds below
  every automatic split; a real split is small below 0.31.
- A sum at a small split predicts its loss from the terms it adds. It raises
  ValueError "Ewald split too small" where the loss exceeds 1e-3 of
  ``max(|S|, 1)`` for the sum ``S``. Off the axis, a 1D spherical sum takes its
  spectral series instead, where that series applies.
- At a small split the real-space part needs more terms: its cost grows like
  ``1 / eta**2`` up to the shell limit. A sum fails as soon as its predicted
  loss exceeds 2e-3 of ``max(|S|, 1)`` for the sum ``S`` at the automatic split.
- At a small split, a complete sum whose real-space shells reach their limit is
  returned only if its last shells add a negligible amount and it matches the
  sum at the automatic split within the predicted loss. Otherwise it raises
  ValueError "did not converge".
- ``realsum*`` and ``recsum*`` apply the 1e-3 rule with the scale
  ``max(|P|, 1)`` of their part ``P``, so a part can succeed at a split where
  the complete sum fails.
- Gradients from ``diff.lattice_sum`` fail the same way where the loss
  predicted for a derivative exceeds 1e-3 of ``|dS| + max(|S|, 1) s``, and as
  soon as it exceeds 2e-3 of that at the automatic split. The factor ``s`` is
  ``|k|`` for the shift and the lattice vectors and ``1 / |k|`` for ``k`` and
  ``kpar``. A value can succeed where its gradient fails. Gradients of a single
  part need an explicit nonzero ``eta``.
- Spherical Ewald sums need ``Re k > 0``; for ``Re k < 0`` they fail or give
  the values of another function.

Differences from treams:
    - ``a`` may be a ``Lattice`` and ``kpar`` a ``WaveVector``, as described
      above.
    - ``eta`` defaults to 0, the automatic split; treams requires it.
    - The automatic split in 3D uses the cell length ``L = V**(1/3)`` of the
      volume ``V``; treams divides by ``k V**(2/3)``. Both give the same sum
      where the parts converge.
    - Each Ewald part stops when two consecutive shells add less than
      ``2e-13 max(|P|, 1)`` and raises ValueError at its shell limit or at a
      small split, as described under Accuracy. treams stops at a change below
      1e-10 or after 200, 20 or 10 shells, and returns the sum without warning.
    - Both Ewald parts sum over a reduced basis, with the shift reduced into a
      cell of the lattice and ``kpar`` into a cell of the reciprocal lattice.
      treams sums over the given basis and can miss short lattice vectors of a
      skewed basis.
    - At a complex split and ``Im k >= 0``, the reciprocal part continues the
      sum from the automatic split, so the value does not depend on the split.
      treams takes the principal branches.
    - For odd ``l`` (spherical) or odd ``m`` (cylindrical), ``kpar = 0`` and a
      shift ``r`` in the lattice line or plane with ``2 r`` a lattice vector,
      the sum is exactly zero at every split. treams returns exact zeros only
      at ``r = 0`` (and at ``r = ±a / 2`` in 1D).
    - Off the axis, the Ewald parts of a 1D spherical sum can cancel strongly.
      There the sum comes from a spectral series, a sum over diffraction orders
      of cylindrical waves, wherever that is more accurate.

Example::

    import numpy as np
    from treams_rs import Lattice, WaveVector, lattice

    a, kpar = Lattice([1.7, 1.7]), WaveVector([0.1, 0.0])
    r = np.array([0.2, 0.1])
    total = lattice.lsumsw2d(2, 0, 1.3, kpar, a, r)  # l, m, k, kpar, a, r
    # The sum does not depend on the Ewald split eta.
    split = lattice.lsumsw2d(2, 0, 1.3, kpar, a, r, eta=0.6)
    np.testing.assert_allclose(split, total, rtol=1e-12)
    # A shift by the lattice vector R = (1.7, 0) multiplies it by exp(-i kpar . R).
    shifted = lattice.lsumsw2d(2, 0, 1.3, kpar, a, r + [1.7, 0.0])
    np.testing.assert_allclose(shifted, np.exp(-0.17j) * total, rtol=1e-12)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native
from ._lattice import sum_cell as _sum_cell

if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "SumResult",
    "area",
    "cube",
    "cubeedge",
    "diffr_orders_circle",
    "dsumcw",
    "dsumcw1d",
    "dsumcw1d_shift",
    "dsumcw2d",
    "dsumsw",
    "dsumsw1d",
    "dsumsw1d_shift",
    "dsumsw2d",
    "dsumsw2d_shift",
    "dsumsw3d",
    "lsumcw",
    "lsumcw1d",
    "lsumcw1d_shift",
    "lsumcw2d",
    "lsumsw",
    "lsumsw1d",
    "lsumsw1d_shift",
    "lsumsw2d",
    "lsumsw2d_shift",
    "lsumsw3d",
    "realsumcw",
    "realsumcw1d",
    "realsumcw1d_shift",
    "realsumcw2d",
    "realsumsw",
    "realsumsw1d",
    "realsumsw1d_shift",
    "realsumsw2d",
    "realsumsw2d_shift",
    "realsumsw3d",
    "reciprocal",
    "recsumcw",
    "recsumcw1d",
    "recsumcw1d_shift",
    "recsumcw2d",
    "recsumsw",
    "recsumsw1d",
    "recsumsw1d_shift",
    "recsumsw2d",
    "recsumsw2d_shift",
    "recsumsw3d",
    "volume",
]

type SumResult = complex | np.ndarray[tuple[int, ...], np.dtype[np.complex128]]
"""Lattice-sum value: a complex scalar, or a complex128 array of the broadcast shape."""


# The native sums of each part, by lattice dimension (1D, 2D, 3D). The part names
# are those of diff.lattice_sum.
_SPHERICAL_SUMS = {
    "full": (_native.lsumsw1d_shift, _native.lsumsw2d_shift, _native.lsumsw3d),
    "real": (_native.realsumsw1d_shift, _native.realsumsw2d_shift, _native.realsumsw3d),
    "reciprocal": (
        _native.recsumsw1d_shift,
        _native.recsumsw2d_shift,
        _native.recsumsw3d,
    ),
    "direct": (_native.dsumsw1d_shift, _native.dsumsw2d_shift, _native.dsumsw3d),
}
_CYLINDRICAL_SUMS = {
    "full": (_native.lsumcw1d_shift, _native.lsumcw2d),
    "real": (_native.realsumcw1d_shift, _native.realsumcw2d),
    "reciprocal": (_native.recsumcw1d_shift, _native.recsumcw2d),
    "direct": (_native.dsumcw1d_shift, _native.dsumcw2d),
}


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
    part: Literal["full", "real", "reciprocal", "direct"],
    out: Any = None,
    **kwargs: Any,
) -> SumResult:
    if not 1 <= dim <= (3 if spherical else 2):
        raise ValueError("invalid lattice dimension")
    a, kpar = _sum_cell(dim, a, kpar, spherical)
    a, kpar = np.asarray(a), np.asarray(kpar)
    if dim == 1:
        if a.shape == (1, 1):
            a = a[0, 0]
        if kpar.shape == (1,):
            kpar = kpar[0]
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
    """Ewald sum D_lm of spherical waves over a 1D, 2D or 3D lattice.

    Mirrors ``treams.lattice.lsumsw``. Calls ``lsumsw1d_shift``, ``lsumsw2d_shift`` or
    ``lsumsw3d``.

    Args:
        dim: Lattice dimension, 1, 2 or 3.
        l: Degree, 0 to 128.
        m: Order, ``|m| <= l``.
        k: Wavenumber with ``Re k > 0``; for ``Re k < 0`` the sum fails or gives
            the values of another function.
        kpar: Bloch vector: a number in 1D, ``dim`` components otherwise,
            or a ``WaveVector``.
        a: Lattice vectors as rows: the period in 1D, a ``(dim, dim)`` array
            or the side lengths of a rectangular cell, or a ``Lattice``.
        r: Shift ``(x, y, z)``.
        eta: Ewald split parameter; 0 selects it automatically.
        out: Array for the result, as for a NumPy ufunc.
        **kwargs: Further ufunc keywords.

    Returns:
        The sum, complex128 with the broadcast shape of the arguments.

    See the Accuracy section of ``treams_rs.lattice``.
    """
    return _sum(True, dim, l, m, k, kpar, a, r, eta, "full", out, **kwargs)


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
    """Ewald sum D_m of cylindrical waves over a 1D or 2D lattice.

    Mirrors ``treams.lattice.lsumcw``. Calls ``lsumcw1d_shift`` or ``lsumcw2d``.

    Args:
        dim: Lattice dimension, 1 or 2.
        m: Order, ``|m|`` up to 128.
        k: Wavenumber, real or complex, nonzero.
        kpar: Bloch vector: a number in 1D, ``dim`` components otherwise,
            or a ``WaveVector``.
        a: Lattice vectors as rows: the period in 1D, a ``(dim, dim)`` array
            or the side lengths of a rectangular cell, or a ``Lattice``.
        r: Shift ``(x, y)``.
        eta: Ewald split parameter; 0 selects it automatically.
        out: Array for the result, as for a NumPy ufunc.
        **kwargs: Further ufunc keywords.

    Returns:
        The sum, complex128 with the broadcast shape of the arguments.

    See the Accuracy section of ``treams_rs.lattice``.
    """
    return _sum(False, dim, 0, m, k, kpar, a, r, eta, "full", out, **kwargs)


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
    """Ewald sum D_lm with ``m = 0`` over a 1D lattice on the z axis.

    Mirrors ``treams.lattice.lsumsw1d``. ``a`` is the period or a ``Lattice``, and
    ``kpar`` the Bloch wavenumber along z or a ``WaveVector``. ``r`` is the shift along
    z. ``eta`` is the Ewald split parameter, and ``eta=0`` selects it automatically.
    Arguments broadcast as for a NumPy ufunc; ``out`` and ``**kwargs`` go to the ufunc.
    See the Accuracy section of ``treams_rs.lattice``.
    """
    a, kpar = _sum_cell(1, a, kpar, True)
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
    """Ewald sum D_lm over a 1D lattice on the z axis.

    Mirrors ``treams.lattice.lsumsw1d_shift``. ``a`` is the period or a ``Lattice``, and
    ``kpar`` the Bloch wavenumber along z or a ``WaveVector``. ``r`` is the shift ``(x,
    y, z)``. ``eta`` is the Ewald split parameter, and ``eta=0`` selects it
    automatically. Arguments broadcast as for a NumPy ufunc; ``out`` and ``**kwargs`` go
    to the ufunc. See the Accuracy section of ``treams_rs.lattice``.
    """
    a, kpar = _sum_cell(1, a, kpar, True)
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
    """Ewald sum D_lm over a 2D lattice in the plane z = 0.

    Mirrors ``treams.lattice.lsumsw2d``. The rows of ``a``, shape ``(2, 2)``, are the
    lattice vectors; ``a`` may also be the two side lengths of a rectangular cell or a
    ``Lattice``. ``kpar`` has shape ``(2,)`` or is a ``WaveVector``. ``r`` is the shift
    ``(x, y)`` in the plane. ``eta`` is the Ewald split parameter, and ``eta=0`` selects
    it automatically. Arguments broadcast as for a NumPy ufunc; ``out`` and ``**kwargs``
    go to the ufunc. See the Accuracy section of ``treams_rs.lattice``.
    """
    a, kpar = _sum_cell(2, a, kpar, True)
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
    """Ewald sum D_lm over a 2D lattice in the plane z = 0.

    Mirrors ``treams.lattice.lsumsw2d_shift``. The rows of ``a``, shape ``(2, 2)``, are
    the lattice vectors; ``a`` may also be the two side lengths of a rectangular cell or
    a ``Lattice``. ``kpar`` has shape ``(2,)`` or is a ``WaveVector``. ``r`` is the
    shift ``(x, y, z)``. ``eta`` is the Ewald split parameter, and ``eta=0`` selects it
    automatically. Arguments broadcast as for a NumPy ufunc; ``out`` and ``**kwargs`` go
    to the ufunc. See the Accuracy section of ``treams_rs.lattice``.
    """
    a, kpar = _sum_cell(2, a, kpar, True)
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
    """Ewald sum D_lm over a 3D lattice.

    Mirrors ``treams.lattice.lsumsw3d``. The rows of ``a``, shape ``(3, 3)``, are the
    lattice vectors; ``a`` may also be the three side lengths of a rectangular cell or a
    ``Lattice``. ``kpar`` has shape ``(3,)`` or is a ``WaveVector``. ``r`` is the shift
    ``(x, y, z)``. ``eta`` is the Ewald split parameter, and ``eta=0`` selects it
    automatically. Arguments broadcast as for a NumPy ufunc; ``out`` and ``**kwargs`` go
    to the ufunc. See the Accuracy section of ``treams_rs.lattice``.
    """
    a, kpar = _sum_cell(3, a, kpar, True)
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
    """Ewald sum D_m over a 1D lattice on the x axis.

    Mirrors ``treams.lattice.lsumcw1d``. ``a`` is the period or a ``Lattice``, and
    ``kpar`` the Bloch wavenumber along x or a ``WaveVector``. ``r`` is the shift along
    x. ``eta`` is the Ewald split parameter, and ``eta=0`` selects it automatically.
    Arguments broadcast as for a NumPy ufunc; ``out`` and ``**kwargs`` go to the ufunc.
    See the Accuracy section of ``treams_rs.lattice``.
    """
    a, kpar = _sum_cell(1, a, kpar, False)
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
    """Ewald sum D_m over a 1D lattice on the x axis.

    Mirrors ``treams.lattice.lsumcw1d_shift``. ``a`` is the period or a ``Lattice``, and
    ``kpar`` the Bloch wavenumber along x or a ``WaveVector``. ``r`` is the shift ``(x,
    y)``. ``eta`` is the Ewald split parameter, and ``eta=0`` selects it automatically.
    Arguments broadcast as for a NumPy ufunc; ``out`` and ``**kwargs`` go to the ufunc.
    See the Accuracy section of ``treams_rs.lattice``.
    """
    a, kpar = _sum_cell(1, a, kpar, False)
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
    """Ewald sum D_m over a 2D lattice in the plane z = 0.

    Mirrors ``treams.lattice.lsumcw2d``. The rows of ``a``, shape ``(2, 2)``, are the
    lattice vectors; ``a`` may also be the two side lengths of a rectangular cell or a
    ``Lattice``. ``kpar`` has shape ``(2,)`` or is a ``WaveVector``. ``r`` is the shift
    ``(x, y)``. ``eta`` is the Ewald split parameter, and ``eta=0`` selects it
    automatically. Arguments broadcast as for a NumPy ufunc; ``out`` and ``**kwargs`` go
    to the ufunc. See the Accuracy section of ``treams_rs.lattice``.
    """
    a, kpar = _sum_cell(2, a, kpar, False)
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
    """Real-space part of the Ewald sum D_lm over a 1D, 2D or 3D lattice.

    Mirrors ``treams.lattice.realsumsw``. Calls ``realsumsw1d_shift``,
    ``realsumsw2d_shift`` or ``realsumsw3d``. Adding ``recsumsw`` at the same ``eta``
    gives ``lsumsw``.

    Args:
        dim: Lattice dimension, 1, 2 or 3.
        l: Degree, 0 to 128.
        m: Order, ``|m| <= l``.
        k: Wavenumber with ``Re k > 0``; for ``Re k < 0`` the sum fails or gives
            the values of another function.
        kpar: Bloch vector: a number in 1D, ``dim`` components otherwise,
            or a ``WaveVector``.
        a: Lattice vectors as rows: the period in 1D, a ``(dim, dim)`` array
            or the side lengths of a rectangular cell, or a ``Lattice``.
        r: Shift ``(x, y, z)``.
        eta: Ewald split parameter; 0 selects it automatically.
        out: Array for the result, as for a NumPy ufunc.
        **kwargs: Further ufunc keywords.

    Returns:
        The part, complex128 with the broadcast shape of the arguments.

    See the Accuracy section of ``treams_rs.lattice``.
    """
    return _sum(True, dim, l, m, k, kpar, a, r, eta, "real", out, **kwargs)


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
    """Real-space part of the Ewald sum D_m over a 1D or 2D lattice.

    Mirrors ``treams.lattice.realsumcw``. Calls ``realsumcw1d_shift`` or
    ``realsumcw2d``. Adding ``recsumcw`` at the same ``eta`` gives ``lsumcw``.

    Args:
        dim: Lattice dimension, 1 or 2.
        m: Order, ``|m|`` up to 128.
        k: Wavenumber, real or complex, nonzero.
        kpar: Bloch vector: a number in 1D, ``dim`` components otherwise,
            or a ``WaveVector``.
        a: Lattice vectors as rows: the period in 1D, a ``(dim, dim)`` array
            or the side lengths of a rectangular cell, or a ``Lattice``.
        r: Shift ``(x, y)``.
        eta: Ewald split parameter; 0 selects it automatically.
        out: Array for the result, as for a NumPy ufunc.
        **kwargs: Further ufunc keywords.

    Returns:
        The part, complex128 with the broadcast shape of the arguments.

    See the Accuracy section of ``treams_rs.lattice``.
    """
    return _sum(False, dim, 0, m, k, kpar, a, r, eta, "real", out, **kwargs)


# The per-dimension parts and the spherical direct shells are the native ufuncs,
# which carry their docstrings from the treams-py ufunc registry.
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
    """Reciprocal-space part of the Ewald sum D_lm over a 1D, 2D or 3D lattice.

    Mirrors ``treams.lattice.recsumsw``. Calls ``recsumsw1d_shift``,
    ``recsumsw2d_shift`` or ``recsumsw3d``. At ``r = 0`` the part includes the
    correction for the dropped term at ``R = 0``, as in treams.

    Args:
        dim: Lattice dimension, 1, 2 or 3.
        l: Degree, 0 to 128.
        m: Order, ``|m| <= l``.
        k: Wavenumber with ``Re k > 0``; for ``Re k < 0`` the sum fails or gives
            the values of another function.
        kpar: Bloch vector: a number in 1D, ``dim`` components otherwise,
            or a ``WaveVector``.
        a: Lattice vectors as rows: the period in 1D, a ``(dim, dim)`` array
            or the side lengths of a rectangular cell, or a ``Lattice``.
        r: Shift ``(x, y, z)``.
        eta: Ewald split parameter; 0 selects it automatically.
        out: Array for the result, as for a NumPy ufunc.
        **kwargs: Further ufunc keywords.

    Returns:
        The part, complex128 with the broadcast shape of the arguments.

    See the Accuracy section of ``treams_rs.lattice``.
    """
    return _sum(True, dim, l, m, k, kpar, a, r, eta, "reciprocal", out, **kwargs)


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
    """Reciprocal-space part of the Ewald sum D_m over a 1D or 2D lattice.

    Mirrors ``treams.lattice.recsumcw``. Calls ``recsumcw1d_shift`` or ``recsumcw2d``.
    At ``r = 0`` the part includes the correction for the dropped term at ``R = 0``, as
    in treams.

    Args:
        dim: Lattice dimension, 1 or 2.
        m: Order, ``|m|`` up to 128.
        k: Wavenumber, real or complex, nonzero.
        kpar: Bloch vector: a number in 1D, ``dim`` components otherwise,
            or a ``WaveVector``.
        a: Lattice vectors as rows: the period in 1D, a ``(dim, dim)`` array
            or the side lengths of a rectangular cell, or a ``Lattice``.
        r: Shift ``(x, y)``.
        eta: Ewald split parameter; 0 selects it automatically.
        out: Array for the result, as for a NumPy ufunc.
        **kwargs: Further ufunc keywords.

    Returns:
        The part, complex128 with the broadcast shape of the arguments.

    See the Accuracy section of ``treams_rs.lattice``.
    """
    return _sum(False, dim, 0, m, k, kpar, a, r, eta, "reciprocal", out, **kwargs)


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
    """Shell ``i`` of the direct sum D_lm over a 1D, 2D or 3D lattice.

    Mirrors ``treams.lattice.dsumsw``. Calls ``dsumsw1d_shift``, ``dsumsw2d_shift`` or
    ``dsumsw3d``. Shell ``i`` holds the lattice points whose integer coordinates have
    the largest magnitude ``i``. In 1D at a shift ``r = s a / 2`` on the axis, with
    ``s = 1`` or ``-1``, shell ``i`` holds ``n = s i`` and ``n = -s (i + 1)`` instead.
    The shells ``0, 1, 2, ...`` add up to the sum, but slowly.

    Args:
        dim: Lattice dimension, 1, 2 or 3.
        l: Degree, 0 to 128.
        m: Order, ``|m| <= l``.
        k: Wavenumber, real or complex, nonzero.
        kpar: Bloch vector: a number in 1D, ``dim`` components otherwise,
            or a ``WaveVector``.
        a: Lattice vectors as rows: the period in 1D, a ``(dim, dim)`` array
            or the side lengths of a rectangular cell, or a ``Lattice``.
        r: Shift ``(x, y, z)``.
        i: Shell index, 0 or more.
        out: Array for the result, as for a NumPy ufunc.
        **kwargs: Further ufunc keywords.

    Returns:
        The shell sum, complex128 with the broadcast shape of the arguments.
    """
    return _sum(True, dim, l, m, k, kpar, a, r, i, "direct", out, **kwargs)


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
    """Shell ``i`` of the direct sum D_m over a 1D or 2D lattice.

    Mirrors ``treams.lattice.dsumcw``. Calls ``dsumcw1d_shift`` or ``dsumcw2d``. Shell
    ``i`` holds the lattice points whose integer coordinates have the largest magnitude
    ``i``. In 1D at a shift ``r = s a / 2`` on the axis, with ``s = 1`` or ``-1``,
    shell ``i`` holds ``n = s i`` and ``n = -s (i + 1)`` instead. The shells
    ``0, 1, 2, ...`` add up to the sum, but slowly.

    Args:
        dim: Lattice dimension, 1 or 2.
        m: Order, ``|m|`` up to 128.
        k: Wavenumber, real or complex, nonzero.
        kpar: Bloch vector: a number in 1D, ``dim`` components otherwise,
            or a ``WaveVector``.
        a: Lattice vectors as rows: the period in 1D, a ``(dim, dim)`` array
            or the side lengths of a rectangular cell, or a ``Lattice``.
        r: Shift ``(x, y)``.
        i: Shell index, 0 or more.
        out: Array for the result, as for a NumPy ufunc.
        **kwargs: Further ufunc keywords.

    Returns:
        The shell sum, complex128 with the broadcast shape of the arguments.
    """
    return _sum(False, dim, 0, m, k, kpar, a, r, i, "direct", out, **kwargs)


dsumsw1d = _native.dsumsw1d
dsumsw1d_shift = _native.dsumsw1d_shift
dsumsw2d = _native.dsumsw2d
dsumsw2d_shift = _native.dsumsw2d_shift
dsumsw3d = _native.dsumsw3d


# Python-scalar fast paths: called with Python ints for m and i, a Python float
# or complex for k, Python numbers for the 1D kpar, a and r, float64 arrays of
# the core shapes for the vector arguments and no ``out`` or ufunc keywords, the
# dsumcw* wrappers call a native scalar function instead of the ufunc, because
# NumPy's ufunc dispatch costs more than one evaluation. Each fast path returns
# the value of the ufunc as a Python complex, or raises the same ValueError.


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
    """Shell ``i`` of the direct sum D_m over a 1D lattice on the x axis.

    Mirrors ``treams.lattice.dsumcw1d``. ``a`` is the period, ``kpar`` the Bloch
    wavenumber along x and ``r`` the shift along x. Shell ``i`` holds the lattice points
    ``n a`` with ``|n| = i``. At ``r = s a / 2``, with ``s = 1`` or ``-1``, it holds
    ``n = s i`` and ``n = -s (i + 1)`` instead. Arguments broadcast as for a NumPy
    ufunc; ``out`` and ``**kwargs`` go to the ufunc.
    """
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
        return _native.dsumcw1d_scalar(m, k, kpar, a, r, i)
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
    """Shell ``i`` of the direct sum D_m over a 1D lattice on the x axis.

    Mirrors ``treams.lattice.dsumcw1d_shift``. ``a`` is the period, ``kpar`` the Bloch
    wavenumber along x and ``r`` the shift ``(x, y)``. Shell ``i`` holds the lattice
    points ``n a`` with ``|n| = i``. At ``r = (s a / 2, 0)``, with ``s = 1`` or ``-1``,
    it holds ``n = s i`` and ``n = -s (i + 1)`` instead. Arguments broadcast as for a
    NumPy ufunc; ``out`` and ``**kwargs`` go to the ufunc.
    """
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
        return _native.dsumcw1d_shift_scalar(m, k, kpar, a, r, i)
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
    """Shell ``i`` of the direct sum D_m over a 2D lattice in the plane z = 0.

    Mirrors ``treams.lattice.dsumcw2d``. The rows of ``a``, shape ``(2, 2)``, are the
    lattice vectors; ``kpar`` and ``r`` have shape ``(2,)``. Shell ``i`` holds the
    lattice points whose integer coordinates have the largest magnitude ``i``. Arguments
    broadcast as for a NumPy ufunc; ``out`` and ``**kwargs`` go to the ufunc.
    """
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
        return _native.dsumcw2d_scalar(m, k, kpar, a, r, i)
    return _native.dsumcw2d(m, k, kpar, a, r, i, out=out, **kwargs)


volume = _native.cell_volume
area = volume
reciprocal = _native.cell_reciprocal


def cube(d: int, n: int) -> NDArray[np.int64]:
    """All integer points in the cube [-n, n]^d, in lexicographic order.

    Mirrors ``treams.lattice.cube``.

    Args:
        d: Space dimension 1, 2 or 3.
        n: Nonnegative half side length of the cube.

    Returns:
        int64 array of shape ((2 n + 1)**d, d).
    """
    return _native.lattice_cube(d, n, False)


def cubeedge(d: int, n: int) -> NDArray[np.int64]:
    """Integer points on the surface of the cube [-n, n]^d, in lexicographic order.

    Mirrors ``treams.lattice.cubeedge``. For ``n = 0`` the surface is the
    origin alone.

    Args:
        d: Space dimension 1, 2 or 3.
        n: Nonnegative half side length of the cube.

    Returns:
        int64 array of shape (points, d).
    """
    return _native.lattice_cube(d, n, True)


def diffr_orders_circle(b: ArrayLike, rmax: float) -> NDArray[np.int64]:
    """All diffraction orders within a circle of radius rmax.

    Mirrors ``treams.lattice.diffr_orders_circle``. Each order is followed by its
    opposite; the enumeration covers skew lattices and keeps the treams order on
    orthogonal cells.

    Args:
        b: Reciprocal lattice vectors as the rows of a (2, 2) array.
        rmax: Radius of the circle in reciprocal space.

    Returns:
        int64 array of shape (orders, 2).

    Differences from treams:
        A negative ``rmax`` gives an empty (0, 2) array; treams gives an empty
        one-dimensional array.
    """
    return _native.diffraction_orders(np.asarray(b, dtype=np.float64), rmax)

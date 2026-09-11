# ruff: noqa: E741 - preserve upstream degree argument l
"""Direct spherical-wave coefficients, backed by the native operator kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native, lattice
from ._core import SphericalWaveBasis, _poltype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray


def translate(
    lambda_: ArrayLike,
    mu: ArrayLike,
    pol: ArrayLike,
    l: ArrayLike,
    m: ArrayLike,
    qol: ArrayLike,
    kr: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    poltype: str | None = None,
    singular: bool = True,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Polarized translation; outgoing self coefficients follow upstream's zero convention."""
    if _poltype(poltype):
        function = _native.sw_translate_sh if singular else _native.sw_translate_rh
    else:
        function = _native.sw_translate_sp if singular else _native.sw_translate_rp
    return function(lambda_, mu, pol, l, m, qol, kr, theta, phi, *args, **kwargs)


def rotate(
    lambda_: ArrayLike,
    mu: ArrayLike,
    pol: ArrayLike,
    l: ArrayLike,
    m: ArrayLike,
    qol: ArrayLike,
    phi: ArrayLike,
    theta: ArrayLike = 0,
    psi: ArrayLike = 0,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Euler rotation coefficient with degree and polarization selection."""
    return _native.sw_rotate(
        lambda_, mu, pol, l, m, qol, phi, theta, psi, *args, **kwargs
    )


def translate_periodic(
    ks: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    rs: ArrayLike,
    out: Sequence[ArrayLike],
    in_: Sequence[ArrayLike] | None = None,
    rsin: ArrayLike | None = None,
    poltype: str | None = None,
    eta: complex = 0,
    func: object = lattice.lsumsw,
) -> NDArray[np.complex128]:
    """Periodic coupling with optional independent source modes and origins."""
    if func is not lattice.lsumsw:
        raise NotImplementedError("custom lattice-sum callbacks are not implemented")
    destination = SphericalWaveBasis(_mode_rows(out), np.atleast_2d(rs))
    source = SphericalWaveBasis(
        _mode_rows(out if in_ is None else in_),
        np.atleast_2d(rs if rsin is None else rsin),
    )
    return lattice.expansion(
        destination,
        source,
        ks,
        a,
        kpar,
        poltype="helicity" if _poltype(poltype) else "parity",
        eta=eta,
    )


def _mode_rows(labels: Sequence[ArrayLike]) -> list[tuple[float, ...]]:
    if len(labels) not in (3, 4):
        raise ValueError("modes require three or four label arrays")
    arrays = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in labels))
    return [
        tuple(float(v) for v in row)
        for row in zip(*(v.flat for v in arrays), strict=True)
    ]


def periodic_to_pw(
    kx: ArrayLike,
    ky: ArrayLike,
    kz: ArrayLike,
    pol: ArrayLike,
    l: ArrayLike,
    m: ArrayLike,
    qol: ArrayLike,
    area: ArrayLike,
    poltype: str | None = None,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Radiation coefficient of a planar periodic spherical array."""
    function = _native.sw_to_pw_h if _poltype(poltype) else _native.sw_to_pw_p
    return function(kx, ky, kz, pol, l, m, qol, area, *args, **kwargs)


def periodic_to_cw(
    kz: ArrayLike,
    mu: ArrayLike,
    pol: ArrayLike,
    l: ArrayLike,
    m: ArrayLike,
    qol: ArrayLike,
    k: ArrayLike,
    a: ArrayLike,
    poltype: str | None = None,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Radiation coefficient of a z-periodic spherical array."""
    function = _native.sw_to_cw_h if _poltype(poltype) else _native.sw_to_cw_p
    return function(kz, mu, pol, l, m, qol, k, a, *args, **kwargs)

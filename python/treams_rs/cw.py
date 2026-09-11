# ruff: noqa: E741 - preserve upstream degree argument l
"""Direct cylindrical-wave coefficients and periodic coupling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native, lattice
from ._core import CylindricalWaveBasis, _poltype
from .sw import _mode_rows

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

rotate = _native.cw_rotate


def translate(
    kz: ArrayLike,
    mu: ArrayLike,
    pol: ArrayLike,
    qz: ArrayLike,
    m: ArrayLike,
    qol: ArrayLike,
    krr: ArrayLike,
    phi: ArrayLike,
    z: ArrayLike,
    singular: bool = True,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Translation with exact axial/polarization selection and outgoing self exclusion."""
    if (
        not args
        and not kwargs
        and isinstance(krr, (int, float, complex))
        and isinstance(mu, int)
        and isinstance(m, int)
        and isinstance(pol, int)
        and isinstance(qol, int)
        and isinstance(kz, (int, float))
        and isinstance(qz, (int, float))
        and isinstance(phi, (int, float))
        and isinstance(z, (int, float))
    ):
        if pol not in (0, 1) or qol not in (0, 1):
            raise ValueError("polarization must be 0 or 1")
        if pol != qol or (singular and abs(krr) < 1e-16 and abs(z) < 1e-16):
            return 0j
        return _native.cylindrical_translation_scalar(
            kz, mu, qz, m, krr, phi, z, singular
        )
    function = _native.cw_translate_s if singular else _native.cw_translate_r
    return function(kz, mu, pol, qz, m, qol, krr, phi, z, *args, **kwargs)


def to_sw(
    l: ArrayLike,
    m: ArrayLike,
    polsw: ArrayLike,
    kz: ArrayLike,
    mu: ArrayLike,
    polcw: ArrayLike,
    k: ArrayLike,
    poltype: str | None = None,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Coincident-origin conversion to regular spherical waves."""
    function = _native.cw_to_sw_h if _poltype(poltype) else _native.cw_to_sw_p
    return function(l, m, polsw, kz, mu, polcw, k, *args, **kwargs)


def translate_periodic(
    ks: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    rs: ArrayLike,
    out: Sequence[ArrayLike],
    in_: Sequence[ArrayLike] | None = None,
    rsin: ArrayLike | None = None,
    eta: complex = 0,
) -> NDArray[np.complex128]:
    """Periodic coupling with optional independent source modes and origins."""
    destination = CylindricalWaveBasis(_mode_rows(out), np.atleast_2d(rs))
    source = CylindricalWaveBasis(
        _mode_rows(out if in_ is None else in_),
        np.atleast_2d(rs if rsin is None else rsin),
    )
    return lattice.expansion(destination, source, ks, a, kpar, eta=eta)


periodic_to_pw = _native.cw_to_pw

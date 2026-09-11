# ruff: noqa: E741 - preserve upstream degree argument l
"""Direct spherical-wave coefficients, backed by the native operator kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native, diff, lattice
from ._core import SphericalWaveBasis, _poltype
from ._lattice import WaveVector

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

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
    func: Callable[..., Any] = lattice.lsumsw,
) -> NDArray[np.complex128]:
    """Periodic coupling with independent source modes/origins and optional callback.

    A custom func receives one broadcast call using the same scalar-harmonic
    signature as lattice.lsumsw. Rust contracts its output with the angular map.
    For callback adjoints, compose advect.periodic_from_table with a differentiable
    table producer; arbitrary Python callbacks do not imply a derivative.
    """
    destination = SphericalWaveBasis(_mode_rows(out), np.atleast_2d(rs))
    source = SphericalWaveBasis(
        _mode_rows(out if in_ is None else in_),
        np.atleast_2d(rs if rsin is None else rsin),
    )
    if func is not lattice.lsumsw:
        helicity = _poltype(poltype)
        wavenumbers = np.atleast_1d(np.asarray(ks, dtype=np.complex128))
        if wavenumbers.shape not in ((1,), (2,)) or not np.isfinite(wavenumbers).all():
            raise ValueError("require one or two finite medium wavenumbers")
        if wavenumbers.size == 2 and wavenumbers[0] == wavenumbers[1]:
            wavenumbers = wavenumbers[:1]
        if not helicity and wavenumbers.size != 1:
            raise ValueError("parity requires an achiral embedding medium")
        maximum = int(np.max(destination.l) + np.max(source.l))
        modes = np.array(
            [
                (degree, order)
                for degree in range(maximum + 1)
                for order in range(-degree, degree + 1)
            ]
        )
        shifts = (
            source.positions[None, :, None, None, :]
            - destination.positions[:, None, None, None, :]
        )
        components = np.atleast_1d(kpar)
        dim = (
            int(np.count_nonzero(~np.isnan(components)))
            if isinstance(kpar, WaveVector)
            else components.size
        )
        dim = min(dim, np.atleast_2d(a).shape[0] if np.ndim(a) != 1 else np.size(a))
        cell, bloch = lattice._geometry(dim, a, kpar, True)
        values = func(
            dim,
            modes[:, 0],
            modes[:, 1],
            wavenumbers[:, None],
            bloch[0] if dim == 1 else bloch,
            cell[0][0] if dim == 1 else cell,
            shifts,
            eta,
        )
        shape = (
            len(destination.positions),
            len(source.positions),
            wavenumbers.size,
            len(modes),
        )
        return diff.periodic_from_table(
            np.broadcast_to(values, shape),
            destination,
            source,
            poltype="helicity" if helicity else "parity",
        )[0]
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

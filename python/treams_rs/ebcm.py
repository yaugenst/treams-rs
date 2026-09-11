"""Axisymmetric EBCM surface integrals evaluated by the Rust core."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from . import diff
from ._core import SphericalWaveBasis

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

type Modes = SphericalWaveBasis | tuple[ArrayLike, ArrayLike, ArrayLike]


def _modes(value: Modes) -> list[tuple[int, int, int]]:
    if isinstance(value, SphericalWaveBasis):
        if not value.isglobal or np.any(value.positions != 0):
            raise ValueError("EBCM basis must have one origin at zero")
        return [(int(degree), int(m), int(p)) for _, degree, m, p in value.modes]
    modes = np.column_stack(value)
    if (
        modes.ndim != 2
        or modes.shape[1] != 3
        or not np.isfinite(modes).all()
        or np.any(modes != np.floor(modes))
    ):
        raise ValueError("EBCM modes must be integer degree/order/polarization triples")
    return [(int(degree), int(m), int(p)) for degree, m, p in modes]


@lru_cache(maxsize=8)
def _quadrature(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = (nodes + 1) * (np.pi / 2)
    weights *= np.pi / 2
    nodes.flags.writeable = weights.flags.writeable = False
    return nodes, weights


def qmat(
    r: Callable[[float], float],
    dr: Callable[[float], float],
    ks: ArrayLike,
    zs: ArrayLike,
    out: Modes,
    in_: Modes | None = None,
    singular: bool = True,
    *,
    order: int = 96,
    legacy: bool = False,
) -> NDArray[np.complex128]:
    """Surface Q integral using fixed Gauss-Legendre quadrature on [0, pi].

    r(theta) is the positive radius and dr(theta) its polar-angle derivative.
    ks and zs are ordered inner/outer, with negative/positive helicity within ks.
    Increase order to check convergence for the chosen shape and multipole degree.
    Use diff.ebcm_qmat or advect.ebcm_qmat for differentiable sampled surfaces.
    The default includes the radial surface-area factor omitted by treams.
    Set legacy=True only to reproduce that upstream integral.
    """
    theta, weights = _quadrature(order)
    radii = np.array([r(float(t)) for t in theta])
    slopes = np.array([dr(float(t)) for t in theta])
    return diff.ebcm_qmat(
        radii,
        slopes,
        ks,
        zs,
        theta=theta,
        weights=weights,
        out=out,
        in_=in_,
        singular=singular,
        legacy=legacy,
    )[0]

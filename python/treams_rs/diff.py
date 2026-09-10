"""Framework-neutral native operations; all pullbacks use the real Hermitian pairing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ._core import SphericalWaveBasis


def sphere(
    lmax: int,
    k0: float,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike | None = None,
    kappa: ArrayLike | None = None,
) -> tuple[NDArray[np.complex128], _native.SphereContext]:
    """Helicity T-matrix and pullback returning (k0, radii, epsilon, mu, kappa)."""
    eps = np.ascontiguousarray(epsilon, dtype=np.complex128)
    return _native.sphere(
        lmax,
        k0,
        np.ascontiguousarray(radii, dtype=np.float64),
        eps,
        np.ones_like(eps)
        if mu is None
        else np.ascontiguousarray(mu, dtype=np.complex128),
        np.zeros_like(eps)
        if kappa is None
        else np.ascontiguousarray(kappa, dtype=np.complex128),
    )


def cluster(
    lmax: int,
    k0: float,
    radii: ArrayLike,
    epsilon: ArrayLike,
    positions: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.ClusterContext]:
    """Homogeneous nonmagnetic spheres in vacuum; pullback returns (radii, positions, epsilon, k0)."""
    return _native.cluster(
        lmax,
        k0,
        np.ascontiguousarray(radii, dtype=np.float64),
        np.ascontiguousarray(epsilon, dtype=np.complex128),
        np.ascontiguousarray(positions, dtype=np.float64),
    )


def interaction(
    local: ArrayLike, coupling: ArrayLike
) -> tuple[NDArray[np.complex128], _native.InteractionContext]:
    """Solve (I - T C) X = T; pullback returns (T_bar, C_bar)."""
    return _native.interact(
        np.ascontiguousarray(local, dtype=np.complex128),
        np.ascontiguousarray(coupling, dtype=np.complex128),
    )


def expansion(
    destination: SphericalWaveBasis,
    source: SphericalWaveBasis,
    ks: ArrayLike,
    *,
    poltype: str = "helicity",
    singular: bool = False,
) -> tuple[NDArray[np.complex128], _native.ExpansionContext]:
    """Spherical expansion; pullback returns (destination positions, source positions, ks)."""
    values = np.asarray(ks, dtype=np.complex128)
    if values.shape != (2,):
        raise ValueError("ks must contain negative and positive helicity wavenumbers")
    if poltype not in ("helicity", "parity"):
        raise ValueError("invalid polarization type")
    return _native.expansion(
        list(destination.modes),
        list(source.modes),
        destination.positions.tolist(),
        source.positions.tolist(),
        (complex(values[0]), complex(values[1])),
        poltype == "helicity",
        singular,
    )

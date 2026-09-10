"""Scattering coefficients in the upstream treams helicity convention."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def mie_with_context(
    l: int,  # noqa: E741 - upstream multipole-degree argument
    x: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike,
    kappa: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.MieContext]:
    """Mie matrix and its one-use native pullback for x, epsilon, mu and kappa.

    The pullback uses dL = Re(sum(conj(cotangent) * doutput)).
    """
    return _native.mie(
        l,
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(epsilon, dtype=np.complex128),
        np.ascontiguousarray(mu, dtype=np.complex128),
        np.ascontiguousarray(kappa, dtype=np.complex128),
    )


def mie(
    l: int,  # noqa: E741 - upstream multipole-degree argument
    x: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike,
    kappa: ArrayLike,
) -> NDArray[np.complex128]:
    """Multilayer sphere coefficients, ordered as treams.coeffs.mie."""
    return mie_with_context(l, x, epsilon, mu, kappa)[0]


def mie_cyl_with_context(
    kz: float,
    m: int,
    k0: float,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike,
    kappa: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.CylinderContext]:
    """Cylinder coefficients; pullback returns (kz, k0, radii, epsilon, mu, kappa)."""
    return _native.mie_cyl(
        kz,
        m,
        k0,
        np.ascontiguousarray(radii, dtype=np.float64),
        np.ascontiguousarray(epsilon, dtype=np.complex128),
        np.ascontiguousarray(mu, dtype=np.complex128),
        np.ascontiguousarray(kappa, dtype=np.complex128),
    )


def mie_cyl(
    kz: float,
    m: int,
    k0: float,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike,
    kappa: ArrayLike,
) -> NDArray[np.complex128]:
    """Multilayer chiral-cylinder coefficients, following treams argument order."""
    return mie_cyl_with_context(kz, m, k0, radii, epsilon, mu, kappa)[0]

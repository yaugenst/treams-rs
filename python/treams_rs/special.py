"""Broadcast mathematical functions evaluated in Rust.

Results use complex128, including real inputs. Singular/nonfinite evaluations
raise ValueError. Use diff.bessel or advect.bessel for native argument pullbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _native, diff

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray


def _bessel(
    order: ArrayLike,
    z: ArrayLike,
    *,
    kind: str = "j",
    spherical: bool = False,
    derivative: bool = False,
) -> NDArray[np.complex128]:
    orders, arguments, shape, _ = diff._bessel_inputs(order, z)
    return _native.bessel_forward(
        orders, arguments, kind, spherical, int(derivative), shape
    )


def jv(v: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Cylindrical Bessel J for real order and complex argument."""
    return _bessel(v, z)


def yv(v: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Cylindrical Bessel Y for real order and complex argument."""
    return _bessel(v, z, kind="y")


def hankel1(v: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Cylindrical outgoing Hankel function."""
    return _bessel(v, z, kind="h1")


def hankel2(v: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Cylindrical incoming Hankel function."""
    return _bessel(v, z, kind="h2")


def jv_d(v: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Complex-argument derivative of cylindrical Bessel J."""
    return _bessel(v, z, derivative=True)


def yv_d(v: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Complex-argument derivative of cylindrical Bessel Y."""
    return _bessel(v, z, kind="y", derivative=True)


def hankel1_d(v: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Complex-argument derivative of outgoing cylindrical Hankel."""
    return _bessel(v, z, kind="h1", derivative=True)


def hankel2_d(v: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Complex-argument derivative of incoming cylindrical Hankel."""
    return _bessel(v, z, kind="h2", derivative=True)


def spherical_jn(
    n: ArrayLike, z: ArrayLike, derivative: bool = False
) -> NDArray[np.complex128]:
    """Spherical regular Bessel function, with analytic values at the origin."""
    return _bessel(n, z, spherical=True, derivative=derivative)


def spherical_yn(
    n: ArrayLike, z: ArrayLike, derivative: bool = False
) -> NDArray[np.complex128]:
    """Spherical second-kind Bessel function."""
    return _bessel(n, z, kind="y", spherical=True, derivative=derivative)


def spherical_hankel1(n: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Spherical outgoing Hankel function."""
    return _bessel(n, z, kind="h1", spherical=True)


def spherical_hankel2(n: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Spherical incoming Hankel function."""
    return _bessel(n, z, kind="h2", spherical=True)


def spherical_jn_d(n: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Complex-argument derivative of spherical regular Bessel."""
    return spherical_jn(n, z, True)


def spherical_yn_d(n: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Complex-argument derivative of spherical second-kind Bessel."""
    return spherical_yn(n, z, True)


def spherical_hankel1_d(n: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Complex-argument derivative of outgoing spherical Hankel."""
    return _bessel(n, z, kind="h1", spherical=True, derivative=True)


def spherical_hankel2_d(n: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Complex-argument derivative of incoming spherical Hankel."""
    return _bessel(n, z, kind="h2", spherical=True, derivative=True)

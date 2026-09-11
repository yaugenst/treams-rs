"""Rust special functions with NumPy broadcasting, ``out`` and ``where``.

Results use complex128, including real inputs. Singular/nonfinite evaluations
raise ValueError. diff/advect.bessel and .angular provide native argument pullbacks.
Angular degrees and orders must be integers, with degree between zero and 128.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _native

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

jv = _native.jv
yv = _native.yv
jv_d = _native.jv_d
yv_d = _native.yv_d
hankel1_d = _native.hankel1_d
hankel2_d = _native.hankel2_d
spherical_hankel1 = _native.spherical_hankel1
spherical_hankel2 = _native.spherical_hankel2
spherical_jn_d = _native.spherical_jn_d
spherical_yn_d = _native.spherical_yn_d
spherical_hankel1_d = _native.spherical_hankel1_d
spherical_hankel2_d = _native.spherical_hankel2_d


def spherical_jn(
    n: ArrayLike,
    z: ArrayLike,
    derivative: bool = False,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Spherical regular Bessel, optionally its first argument derivative."""
    function = _native.spherical_jn_d if derivative else _native.spherical_jn
    return function(n, z, out=out, where=where)


def spherical_yn(
    n: ArrayLike,
    z: ArrayLike,
    derivative: bool = False,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Spherical second-kind Bessel, optionally its first argument derivative."""
    function = _native.spherical_yn_d if derivative else _native.spherical_yn
    return function(n, z, out=out, where=where)


def hankel1(
    v: ArrayLike,
    z: ArrayLike,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> complex | NDArray[np.complex128]:
    """Outgoing cylindrical Hankel, with a direct native Python-scalar path."""
    if (
        out is None
        and where is True
        and isinstance(v, (int, float))
        and isinstance(z, (int, float, complex))
    ):
        return _native.hankel_scalar(v, z, True)
    return _native.hankel1(v, z, out=out, where=where)


def hankel2(
    v: ArrayLike,
    z: ArrayLike,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> complex | NDArray[np.complex128]:
    """Incoming cylindrical Hankel, with a direct native Python-scalar path."""
    if (
        out is None
        and where is True
        and isinstance(v, (int, float))
        and isinstance(z, (int, float, complex))
    ):
        return _native.hankel_scalar(v, z, False)
    return _native.hankel2(v, z, out=out, where=where)


def lpmv(
    m: ArrayLike,
    n: ArrayLike,
    z: ArrayLike,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> complex | NDArray[np.complex128]:
    """Integer-degree legendre function with native broadcasting and polar limits."""
    if (
        out is None
        and where is True
        and isinstance(n, (int, float))
        and isinstance(m, (int, float))
        and isinstance(z, (int, float, complex))
    ):
        return _native.angular_value(n, m, z, "legendre")
    return _native.lpmv(m, n, z, out=out, where=where)


def pi_fun(
    n: ArrayLike,
    m: ArrayLike,
    z: ArrayLike,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> complex | NDArray[np.complex128]:
    """Integer-degree pi function with native broadcasting and polar limits."""
    if (
        out is None
        and where is True
        and isinstance(n, (int, float))
        and isinstance(m, (int, float))
        and isinstance(z, (int, float, complex))
    ):
        return _native.angular_value(n, m, z, "pi")
    return _native.pi_fun(n, m, z, out=out, where=where)


def tau_fun(
    n: ArrayLike,
    m: ArrayLike,
    z: ArrayLike,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> complex | NDArray[np.complex128]:
    """Integer-degree tau function with native broadcasting and polar limits."""
    if (
        out is None
        and where is True
        and isinstance(n, (int, float))
        and isinstance(m, (int, float))
        and isinstance(z, (int, float, complex))
    ):
        return _native.angular_value(n, m, z, "tau")
    return _native.tau_fun(n, m, z, out=out, where=where)

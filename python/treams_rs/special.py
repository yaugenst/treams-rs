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


wignersmalld = _native.wignersmalld
wignerd = _native.wignerd


def incgamma(
    n: ArrayLike,
    z: ArrayLike,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> complex | NDArray[np.complex128]:
    """Upper incomplete gamma for integer and half-integer degree."""
    if (
        out is None
        and where is True
        and isinstance(n, (int, float))
        and isinstance(z, (int, float, complex))
    ):
        return _native.incgamma(n, z)
    return _native.incgamma_ufunc(n, z, out=out, where=where)


def intkambe(
    n: ArrayLike,
    z: ArrayLike,
    eta: ArrayLike,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> complex | NDArray[np.complex128]:
    """Kambe integral with integer order and broadcast complex arguments."""
    if (
        out is None
        and where is True
        and isinstance(n, int)
        and isinstance(z, (int, float, complex))
        and isinstance(eta, (int, float, complex))
    ):
        return _native.intkambe(n, z, eta)
    return _native.intkambe_ufunc(n, z, eta, out=out, where=where)


def wigner3j(
    j1: ArrayLike,
    j2: ArrayLike,
    j3: ArrayLike,
    m1: ArrayLike,
    m2: ArrayLike,
    m3: ArrayLike,
    *,
    out: NDArray[np.float64] | None = None,
    where: ArrayLike = True,
) -> float | NDArray[np.float64]:
    """Integer Wigner 3j symbol with selection rules and native broadcasting."""
    if (
        out is None
        and where is True
        and isinstance(j1, int)
        and isinstance(j2, int)
        and isinstance(j3, int)
        and isinstance(m1, int)
        and isinstance(m2, int)
        and isinstance(m3, int)
    ):
        return _native.wigner3j_scalar(j1, j2, j3, m1, m2, m3)
    return _native.wigner3j(j1, j2, j3, m1, m2, m3, out=out, where=where)


car2cyl = _native.car2cyl
car2sph = _native.car2sph
cyl2car = _native.cyl2car
cyl2sph = _native.cyl2sph
sph2car = _native.sph2car
sph2cyl = _native.sph2cyl
car2pol = _native.car2pol
pol2car = _native.pol2car
vcar2cyl = _native.vcar2cyl
vcar2sph = _native.vcar2sph
vcyl2car = _native.vcyl2car
vcyl2sph = _native.vcyl2sph
vsph2car = _native.vsph2car
vsph2cyl = _native.vsph2cyl
vcar2pol = _native.vcar2pol
vpol2car = _native.vpol2car

sph_harm = _native.sph_harm
vsh_X = _native.vsh_X  # noqa: N816 - upstream public function name
vsh_Y = _native.vsh_Y  # noqa: N816 - upstream public function name
vsh_Z = _native.vsh_Z  # noqa: N816 - upstream public function name
vsw_M = _native.vsw_M  # noqa: N816 - upstream public function name
vsw_N = _native.vsw_N  # noqa: N816 - upstream public function name
vsw_A = _native.vsw_A  # noqa: N816 - upstream public function name
vsw_rM = _native.vsw_rM  # noqa: N816 - upstream public function name
vsw_rN = _native.vsw_rN  # noqa: N816 - upstream public function name
vsw_rA = _native.vsw_rA  # noqa: N816 - upstream public function name
vcw_M = _native.vcw_M  # noqa: N816 - upstream public function name
vcw_N = _native.vcw_N  # noqa: N816 - upstream public function name
vcw_A = _native.vcw_A  # noqa: N816 - upstream public function name
vcw_rM = _native.vcw_rM  # noqa: N816 - upstream public function name
vcw_rN = _native.vcw_rN  # noqa: N816 - upstream public function name
vcw_rA = _native.vcw_rA  # noqa: N816 - upstream public function name
vpw_M = _native.vpw_M  # noqa: N816 - upstream public function name
vpw_N = _native.vpw_N  # noqa: N816 - upstream public function name
vpw_A = _native.vpw_A  # noqa: N816 - upstream public function name

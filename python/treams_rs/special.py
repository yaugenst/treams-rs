"""Special functions of multipole waves, with NumPy broadcasting.

Mirrors ``treams.special``. Every function broadcasts its arguments as a NumPy
ufunc does. With NumPy inputs the functions accept ``out`` and ``where``;
the coordinate transforms and ``vpw_*`` accept ``out`` only.

Advect, JAX, PyTorch and Autograd inputs select their differentiation adapter
automatically for continuous arguments. Degrees, orders and polarization labels
stay fixed. Differentiable calls do not accept ``out`` or ``where``; use the
framework's array operations to combine or mask results.

- Bessel and Hankel functions: ``jv``, ``yv``, ``hankel1``, ``hankel2``, the
  spherical ``spherical_jn``, ``spherical_yn``, ``spherical_hankel1``,
  ``spherical_hankel2``, and their derivatives ``*_d``.
- Angular functions: ``lpmv``, ``pi_fun``, ``tau_fun``, ``sph_harm`` and the
  Wigner symbols ``wignersmalld``, ``wignerd`` and ``wigner3j``.
- Lattice-sum integrals: ``incgamma`` and ``intkambe``.
- Vector waves: spherical ``vsw_*``, cylindrical ``vcw_*``, plane ``vpw_*``
  and the vector spherical harmonics ``vsh_*``. An ``r`` after the underscore
  (``vsw_rN``) selects the regular wave; without it the wave is singular
  (outgoing, built on the Hankel function H1).
- Translation coefficients: ``tl_vsw_A``, ``tl_vsw_B``, ``tl_vcw`` and their
  regular versions ``tl_vsw_rA``, ``tl_vsw_rB`` and ``tl_vcw_r``.
- Coordinate transforms of points (``car2sph``, ``sph2car``, ...) and of
  vector components (``vcar2sph``, ``vsph2car``, ...).

Results are complex128, also for real arguments. ``lpmv`` with real
arguments, ``wigner3j`` and the transforms of real points return float64.
Bessel functions, angular functions and waves raise ValueError where they have
no finite value, for example ``hankel1(0, 0.0)``. Angular functions, Wigner d
functions and spherical waves take degrees l from 0 to 128.
``diff.bessel``, ``diff.angular`` and their ``advect`` versions also return
gradients with respect to the argument.

Differences from treams:
    - Native ufuncs are exposed through dispatching callables, so they are not
      instances of ``numpy.ufunc``. NumPy calls, ufunc attributes and methods
      such as ``outer`` still delegate to the original ufunc, available as
      ``__wrapped__``. Ufunc methods themselves are NumPy-only.
    - ValueError instead of NaN or infinity: ``hankel1(0, 0.0)`` raises,
      treams returns ``nan+nanj``. ``incgamma`` and ``intkambe`` return
      infinity at their poles, as treams does.
    - complex128 instead of float64 for real arguments: ``jv(0, 1.0)`` is
      ``0.765...+0j``.
    - Degrees l above 128 raise ValueError; treams has no limit.
    - ``hankel1``, ``hankel2``, ``lpmv``, ``pi_fun``, ``tau_fun``,
      ``incgamma``, ``intkambe``, ``wigner3j``, ``tl_vcw`` and ``tl_vcw_r``
      are Python functions, not ufuncs: they have no ufunc methods such as
      ``outer``.
    - The coordinate transforms (``car2cyl``, ..., ``vpol2car``) and
      ``vpw_M``, ``vpw_N``, ``vpw_A`` are native functions, not ufuncs: they
      take no ``where`` and have no ufunc methods.

Example::

    import numpy as np
    from treams_rs import special

    x = 2.0
    assert np.isclose(special.jv(0, 1.0), 0.7651976865579666)
    assert np.isclose(special.spherical_jn(1, x), np.sin(x) / x**2 - np.cos(x) / x)
    assert np.isclose(special.lpmv(2, 2, 0.5), 3 * (1 - 0.5**2))
"""

from __future__ import annotations

from functools import partial as _partial
from typing import TYPE_CHECKING

from . import _native
from . import _special_functions as _ad
from ._autodiff_functions import transparent_function as _function

_transparent_function = _partial(_function, module=__name__)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "car2cyl",
    "car2pol",
    "car2sph",
    "cyl2car",
    "cyl2sph",
    "hankel1",
    "hankel1_d",
    "hankel2",
    "hankel2_d",
    "incgamma",
    "intkambe",
    "jv",
    "jv_d",
    "lpmv",
    "pi_fun",
    "pol2car",
    "sph2car",
    "sph2cyl",
    "sph_harm",
    "spherical_hankel1",
    "spherical_hankel1_d",
    "spherical_hankel2",
    "spherical_hankel2_d",
    "spherical_jn",
    "spherical_jn_d",
    "spherical_yn",
    "spherical_yn_d",
    "tau_fun",
    "tl_vcw",
    "tl_vcw_r",
    "tl_vsw_A",
    "tl_vsw_B",
    "tl_vsw_rA",
    "tl_vsw_rB",
    "vcar2cyl",
    "vcar2pol",
    "vcar2sph",
    "vcw_A",
    "vcw_M",
    "vcw_N",
    "vcw_rA",
    "vcw_rM",
    "vcw_rN",
    "vcyl2car",
    "vcyl2sph",
    "vpol2car",
    "vpw_A",
    "vpw_M",
    "vpw_N",
    "vsh_X",
    "vsh_Y",
    "vsh_Z",
    "vsph2car",
    "vsph2cyl",
    "vsw_A",
    "vsw_M",
    "vsw_N",
    "vsw_rA",
    "vsw_rM",
    "vsw_rN",
    "wigner3j",
    "wignerd",
    "wignersmalld",
    "yv",
    "yv_d",
]

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


# The ufunc aliases (jv, ..., sph_harm, vsh_*, vsw_*, vcw_*, tl_vsw_*,
# wignersmalld, wignerd) keep NumPy's positional ``out``, keyword ``where`` and
# ufunc methods. The coordinate transforms and vpw_* are native functions with
# their own scalar path; they take ``out`` but not ``where``. The Python functions
# in this module mirror their treams signatures instead: spherical_jn and
# spherical_yn take the ``derivative`` flag, tl_vcw and tl_vcw_r forward extra
# arguments to the ufunc, and the others take ``out`` and ``where`` as keywords.
#
# Python-scalar fast paths: called with Python numbers and no ``out`` or
# ``where``, a wrapper calls a native scalar function instead of the ufunc,
# because NumPy's ufunc dispatch costs more than one evaluation. Each fast path
# returns the value and dtype of the ufunc, or raises the same ValueError;
# test_python_scalar_fast_paths_match_the_ufunc in
# tests/bindings/test_ufunc_contract.py checks this for every wrapper.


def spherical_jn(
    n: ArrayLike,
    z: ArrayLike,
    derivative: bool = False,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Spherical Bessel function j_n(z) = sqrt(pi / (2 z)) J_{n + 1/2}(z).

    Mirrors ``treams.special.spherical_jn``.

    Args:
        n: Order n, usually an integer n >= 0; a noninteger n evaluates the
            formula above.
        z: Argument, real or complex.
        derivative: If True, return the derivative dj_n/dz instead.
        out: Array for the result, as for a NumPy ufunc.
        where: Mask of the elements to compute, as for a NumPy ufunc.

    Returns:
        complex128 array with the broadcast shape of ``n`` and ``z``.

    Differences from treams:
        treams returns float64 for real ``z`` and truncates a noninteger ``n``;
        here a noninteger ``n`` evaluates the formula above. Non-finite input
        and overflow raise ValueError.
    """
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
    """Spherical Bessel function y_n(z) = sqrt(pi / (2 z)) Y_{n + 1/2}(z).

    Mirrors ``treams.special.spherical_yn``.

    Args:
        n: Order n, usually an integer n >= 0; a noninteger n evaluates the
            formula above.
        z: Argument, real or complex, nonzero.
        derivative: If True, return the derivative dy_n/dz instead.
        out: Array for the result, as for a NumPy ufunc.
        where: Mask of the elements to compute, as for a NumPy ufunc.

    Returns:
        complex128 array with the broadcast shape of ``n`` and ``z``.

    Differences from treams:
        treams returns float64 for real ``z``, ``-inf`` at ``z = 0`` and
        truncates a noninteger ``n``; here ``z = 0`` raises ValueError and a
        noninteger ``n`` evaluates the formula above. Non-finite input,
        overflow and ``n = 0`` at ``|z|`` below about ``1e-162`` raise
        ValueError.
    """
    function = _native.spherical_yn_d if derivative else _native.spherical_yn
    return function(n, z, out=out, where=where)


def hankel1(
    v: ArrayLike,
    z: ArrayLike,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> complex | NDArray[np.complex128]:
    """Hankel function of the first kind H1_v(z) = J_v(z) + i Y_v(z).

    Mirrors ``treams.special.hankel1``. H1 describes outgoing (singular)
    cylindrical waves.

    Args:
        v: Real order.
        z: Argument, real or complex, nonzero.
        out: Array for the result, as for a NumPy ufunc.
        where: Mask of the elements to compute, as for a NumPy ufunc.

    Returns:
        complex128 array with the broadcast shape of ``v`` and ``z``. Python
        numbers give one complex number.

    Differences from treams:
        ``z = 0``, non-finite input and overflow raise ValueError; treams
        returns NaN or infinity.
    """
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
    """Hankel function of the second kind H2_v(z) = J_v(z) - i Y_v(z).

    Mirrors ``treams.special.hankel2``. H2 describes incoming cylindrical waves.

    Args:
        v: Real order.
        z: Argument, real or complex, nonzero.
        out: Array for the result, as for a NumPy ufunc.
        where: Mask of the elements to compute, as for a NumPy ufunc.

    Returns:
        complex128 array with the broadcast shape of ``v`` and ``z``. Python
        numbers give one complex number.

    Differences from treams:
        ``z = 0``, non-finite input and overflow raise ValueError; treams
        returns NaN or infinity.
    """
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
    out: NDArray[np.complex128] | NDArray[np.float64] | None = None,
    where: ArrayLike = True,
) -> float | complex | NDArray[np.complex128] | NDArray[np.float64]:
    """Associated Legendre function P_v^m(z), with the Condon-Shortley phase (-1)^m.

    Mirrors ``treams.special.lpmv``. The order ``m`` comes first, as in SciPy.
    For real arguments the degree may be noninteger. Orders with
    ``|m| > v`` give 0.

    Args:
        m: Integer order m.
        n: Degree v, from 0 to 128; integer for complex ``z``.
        z: Argument. Real ``z`` outside [-1, 1] with odd ``m`` and ``|m| <= v``
            raises ValueError; pass complex ``z`` for the continuation.
        out: Array for the result, as for a NumPy ufunc.
        where: Mask of the elements to compute, as for a NumPy ufunc.

    Returns:
        float64 array for real arguments, complex128 array for complex ``z``,
        with the broadcast shape of ``m``, ``n`` and ``z``. Python numbers give
        one float or complex number.

    Differences from treams:
        At real ``|z| > 1`` and an integer degree ``v >= |m| > 0``, treams
        returns NaN; here even ``m`` gives the value and odd ``m`` raises
        ValueError. A noninteger degree ``v > |m|`` needs ``z`` in ``(-1, 1]``
        with zero imaginary part and raises ValueError elsewhere.
    """
    if (
        out is None
        and where is True
        and isinstance(n, (int, float))
        and isinstance(m, (int, float))
    ):
        if isinstance(z, complex):
            return _native.angular_scalar(n, m, z, "legendre")
        if isinstance(z, (int, float)):
            return _native.lpmv_real_scalar(n, m, z)
    return _native.lpmv(m, n, z, out=out, where=where)


def pi_fun(
    n: ArrayLike,
    m: ArrayLike,
    z: ArrayLike,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> complex | NDArray[np.complex128]:
    """Angular function pi_l^m(x) = m P_l^m(x) / sqrt(1 - x^2).

    Mirrors ``treams.special.pi_fun``. At ``x = +-1`` it returns the limit.

    Args:
        n: Integer degree l, from 0 to 128.
        m: Integer order m.
        z: Argument x, real or complex; ``x = cos(theta)`` on the real line.
        out: Array for the result, as for a NumPy ufunc.
        where: Mask of the elements to compute, as for a NumPy ufunc.

    Returns:
        complex128 array with the broadcast shape of ``n``, ``m`` and ``z``.
        Python numbers give one complex number.

    Differences from treams:
        treams returns float64 for real ``x`` and NaN at real ``|x| > 1``,
        where this function returns the value.
    """
    if (
        out is None
        and where is True
        and isinstance(n, (int, float))
        and isinstance(m, (int, float))
        and isinstance(z, (int, float, complex))
    ):
        return _native.angular_scalar(n, m, z, "pi")
    return _native.pi_fun(n, m, z, out=out, where=where)


def tau_fun(
    n: ArrayLike,
    m: ArrayLike,
    z: ArrayLike,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> complex | NDArray[np.complex128]:
    """Angular function tau_l^m(x) = d P_l^m(cos theta) / d theta at x = cos theta.

    Mirrors ``treams.special.tau_fun``. At ``x = +-1`` it returns the limit.

    Args:
        n: Integer degree l, from 0 to 128.
        m: Integer order m.
        z: Argument x, real or complex.
        out: Array for the result, as for a NumPy ufunc.
        where: Mask of the elements to compute, as for a NumPy ufunc.

    Returns:
        complex128 array with the broadcast shape of ``n``, ``m`` and ``z``.
        Python numbers give one complex number.

    Differences from treams:
        treams returns float64 for real ``x`` and NaN at real ``|x| > 1``
        unless ``l = |m| <= 1``; this function returns the value. For
        ``|m| > l`` it returns 0, where treams returns, for example,
        ``tau_fun(0, -1, x) = 0.5``.
    """
    if (
        out is None
        and where is True
        and isinstance(n, (int, float))
        and isinstance(m, (int, float))
        and isinstance(z, (int, float, complex))
    ):
        return _native.angular_scalar(n, m, z, "tau")
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
    """Upper incomplete gamma function Gamma(n, z).

    ``Gamma(n, z) = integral_z^inf t^(n - 1) exp(-t) dt``.

    Mirrors ``treams.special.incgamma``. The negative real axis is the branch
    cut. For ``n <= 0``, ``Gamma(n, 0)`` is infinite.

    Args:
        n: Integer or half-integer degree from -128 to 128.
        z: Argument, real or complex.
        out: Array for the result, as for a NumPy ufunc.
        where: Mask of the elements to compute, as for a NumPy ufunc.

    Returns:
        complex128 array with the broadcast shape of ``n`` and ``z``. Python
        numbers give one complex number.

    Differences from treams:
        treams returns float64 for real ``z`` and NaN at ``z < 0`` for
        half-integer ``l`` and for ``l <= 0``; this function returns the
        principal-branch value (``arg z = pi``). Degrees outside the integers
        and half-integers from -128 to 128 raise ValueError.
    """
    if (
        out is None
        and where is True
        and isinstance(n, (int, float))
        and isinstance(z, (int, float, complex))
    ):
        return _native.incgamma_scalar(n, z)
    return _native.incgamma(n, z, out=out, where=where)


def intkambe(
    n: ArrayLike,
    z: ArrayLike,
    eta: ArrayLike,
    *,
    out: NDArray[np.complex128] | None = None,
    where: ArrayLike = True,
) -> complex | NDArray[np.complex128]:
    """Kambe integral I_n(z, eta).

    ``I_n(z, eta) = integral_eta^inf t^n exp(-z^2 t^2 / 2 + 1 / (2 t^2)) dt``.

    Mirrors ``treams.special.intkambe``. The lattice sums of
    ``treams_rs.lattice`` use it. Odd orders at imaginary ``eta = -i / w``
    (the arguments of the 1D spherical lattice sums) come from a series that
    cancels by up to ``exp(|w|^2)``: they lose up to about 1e-4 relative at
    ``|w| = 5``, 0.1 at ``|w| = 6`` and every digit from about ``|w| = 8``.

    Args:
        n: Integer order.
        z: Argument, real or complex.
        eta: Lower limit of the integral, real or complex.
        out: Array for the result, as for a NumPy ufunc.
        where: Mask of the elements to compute, as for a NumPy ufunc.

    Returns:
        complex128 array with the broadcast shape of ``n``, ``z`` and ``eta``.
        Python numbers give one complex number.
    """
    if (
        out is None
        and where is True
        and isinstance(n, int)
        and isinstance(z, (int, float, complex))
        and isinstance(eta, (int, float, complex))
    ):
        return _native.intkambe_scalar(n, z, eta)
    return _native.intkambe(n, z, eta, out=out, where=where)


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
    """Wigner 3j symbol (j1 j2 j3; m1 m2 m3).

    Mirrors ``treams.special.wigner3j``. Labels that break the selection rules
    (``m1 + m2 + m3 = 0``, the triangle rule, ``|mi| <= ji``) give 0.

    Args:
        j1: Integer degree.
        j2: Integer degree.
        j3: Integer degree.
        m1: Integer order.
        m2: Integer order.
        m3: Integer order.
        out: Array for the result, as for a NumPy ufunc.
        where: Mask of the elements to compute, as for a NumPy ufunc.

    Returns:
        float64 array with the broadcast shape of the six labels. Python
        integers give one float.

    Differences from treams:
        Labels must be integers from -260 to 260; others raise ValueError.
    """
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
vsh_X = _native.vsh_X
vsh_Y = _native.vsh_Y
vsh_Z = _native.vsh_Z
vsw_M = _native.vsw_M
vsw_N = _native.vsw_N
vsw_A = _native.vsw_A
vsw_rM = _native.vsw_rM
vsw_rN = _native.vsw_rN
vsw_rA = _native.vsw_rA
vcw_M = _native.vcw_M
vcw_N = _native.vcw_N
vcw_A = _native.vcw_A
vcw_rM = _native.vcw_rM
vcw_rN = _native.vcw_rN
vcw_rA = _native.vcw_rA
vpw_M = _native.vpw_M
vpw_N = _native.vpw_N
vpw_A = _native.vpw_A

tl_vsw_A = _native.tl_vsw_A
tl_vsw_B = _native.tl_vsw_B
tl_vsw_rA = _native.tl_vsw_rA
tl_vsw_rB = _native.tl_vsw_rB


def tl_vcw(
    kz: ArrayLike,
    mu: ArrayLike,
    qz: ArrayLike,
    m: ArrayLike,
    krr: ArrayLike,
    phi: ArrayLike,
    z: ArrayLike,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Singular cylindrical translation coefficient of one mode pair.

    ``H1_{m - mu}(krr) exp(i ((m - mu) phi + kz z))`` if ``kz == qz``, else 0.

    Mirrors ``treams.special.tl_vcw``. It expands a singular wave about the
    source origin in regular waves about the destination origin;
    ``cw.translate`` adds the polarization.

    Args:
        kz: Axial wavenumber of the destination mode.
        mu: Integer order of the destination mode.
        qz: Axial wavenumber of the source mode.
        m: Integer order of the source mode.
        krr: Radial distance of the displacement times the radial wavenumber,
            real or complex, nonzero.
        phi: Azimuthal angle of the displacement.
        z: Axial distance of the displacement, not scaled by a wavenumber.
        *args: Further ufunc arguments, such as ``out``.
        **kwargs: Further ufunc keywords, such as ``where``.

    Returns:
        complex128 array with the broadcast shape of the arguments. Python
        numbers give one complex number.

    Differences from treams:
        ``krr = 0`` raises ValueError; treams returns ``nan+nanj``.
    """
    if (
        not args
        and not kwargs
        and isinstance(mu, int)
        and isinstance(m, int)
        and isinstance(krr, (int, float, complex))
        and isinstance(kz, (int, float))
        and isinstance(qz, (int, float))
        and isinstance(phi, (int, float))
        and isinstance(z, (int, float))
    ):
        return _native.tl_vcw_scalar(kz, mu, qz, m, krr, phi, z, True)
    return _native.tl_vcw(kz, mu, qz, m, krr, phi, z, *args, **kwargs)


def tl_vcw_r(
    kz: ArrayLike,
    mu: ArrayLike,
    qz: ArrayLike,
    m: ArrayLike,
    krr: ArrayLike,
    phi: ArrayLike,
    z: ArrayLike,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Regular cylindrical translation coefficient of one mode pair.

    ``J_{m - mu}(krr) exp(i ((m - mu) phi + kz z))`` if ``kz == qz``, else 0.

    Mirrors ``treams.special.tl_vcw_r``. It expands a wave about the source
    origin in waves of the same kind about the destination origin. At
    ``krr = 0`` it returns the limit.

    Args:
        kz: Axial wavenumber of the destination mode.
        mu: Integer order of the destination mode.
        qz: Axial wavenumber of the source mode.
        m: Integer order of the source mode.
        krr: Radial distance of the displacement times the radial wavenumber,
            real or complex.
        phi: Azimuthal angle of the displacement.
        z: Axial distance of the displacement, not scaled by a wavenumber.
        *args: Further ufunc arguments, such as ``out``.
        **kwargs: Further ufunc keywords, such as ``where``.

    Returns:
        complex128 array with the broadcast shape of the arguments. Python
        numbers give one complex number.
    """
    if (
        not args
        and not kwargs
        and isinstance(mu, int)
        and isinstance(m, int)
        and isinstance(krr, (int, float, complex))
        and isinstance(kz, (int, float))
        and isinstance(qz, (int, float))
        and isinstance(phi, (int, float))
        and isinstance(z, (int, float))
    ):
        return _native.tl_vcw_scalar(kz, mu, qz, m, krr, phi, z, False)
    return _native.tl_vcw_r(kz, mu, qz, m, krr, phi, z, *args, **kwargs)


# NumPy still runs the original callable, including ufunc methods and out/where.
# Framework inputs select the same analytic records used by the explicit adapters.
jv = _transparent_function(jv, _partial(_ad.bessel, function="j"))
yv = _transparent_function(yv, _partial(_ad.bessel, function="y"))
hankel1 = _transparent_function(hankel1, _partial(_ad.bessel, function="h1"))
hankel2 = _transparent_function(hankel2, _partial(_ad.bessel, function="h2"))
jv_d = _transparent_function(jv_d, _partial(_ad.bessel, function="j", derivative=True))
yv_d = _transparent_function(yv_d, _partial(_ad.bessel, function="y", derivative=True))
hankel1_d = _transparent_function(
    hankel1_d, _partial(_ad.bessel, function="h1", derivative=True)
)
hankel2_d = _transparent_function(
    hankel2_d, _partial(_ad.bessel, function="h2", derivative=True)
)
spherical_jn = _transparent_function(
    spherical_jn, _partial(_ad.spherical_bessel, function="j")
)
spherical_yn = _transparent_function(
    spherical_yn, _partial(_ad.spherical_bessel, function="y")
)
spherical_hankel1 = _transparent_function(
    spherical_hankel1, _partial(_ad.bessel, function="h1", spherical=True)
)
spherical_hankel2 = _transparent_function(
    spherical_hankel2, _partial(_ad.bessel, function="h2", spherical=True)
)
spherical_jn_d = _transparent_function(
    spherical_jn_d, _partial(_ad.bessel, function="j", spherical=True, derivative=True)
)
spherical_yn_d = _transparent_function(
    spherical_yn_d, _partial(_ad.bessel, function="y", spherical=True, derivative=True)
)
spherical_hankel1_d = _transparent_function(
    spherical_hankel1_d,
    _partial(_ad.bessel, function="h1", spherical=True, derivative=True),
)
spherical_hankel2_d = _transparent_function(
    spherical_hankel2_d,
    _partial(_ad.bessel, function="h2", spherical=True, derivative=True),
)
lpmv = _transparent_function(lpmv, _ad.lpmv)
pi_fun = _transparent_function(pi_fun, _partial(_ad.angular, function="pi"))
tau_fun = _transparent_function(tau_fun, _partial(_ad.angular, function="tau"))
incgamma = _transparent_function(incgamma, _ad.incgamma)
intkambe = _transparent_function(intkambe, _ad.intkambe)
wignerd = _transparent_function(wignerd, _ad.wignerd)
wignersmalld = _transparent_function(wignersmalld, _ad.wignersmalld)
car2cyl = _transparent_function(car2cyl, _partial(_ad.coordinates, function="car2cyl"))
car2sph = _transparent_function(car2sph, _partial(_ad.coordinates, function="car2sph"))
cyl2car = _transparent_function(cyl2car, _partial(_ad.coordinates, function="cyl2car"))
cyl2sph = _transparent_function(cyl2sph, _partial(_ad.coordinates, function="cyl2sph"))
sph2car = _transparent_function(sph2car, _partial(_ad.coordinates, function="sph2car"))
sph2cyl = _transparent_function(sph2cyl, _partial(_ad.coordinates, function="sph2cyl"))
car2pol = _transparent_function(car2pol, _partial(_ad.coordinates, function="car2pol"))
pol2car = _transparent_function(pol2car, _partial(_ad.coordinates, function="pol2car"))
vcar2cyl = _transparent_function(
    vcar2cyl, _partial(_ad.vector_coordinates, function="car2cyl")
)
vcar2sph = _transparent_function(
    vcar2sph, _partial(_ad.vector_coordinates, function="car2sph")
)
vcyl2car = _transparent_function(
    vcyl2car, _partial(_ad.vector_coordinates, function="cyl2car")
)
vcyl2sph = _transparent_function(
    vcyl2sph, _partial(_ad.vector_coordinates, function="cyl2sph")
)
vsph2car = _transparent_function(
    vsph2car, _partial(_ad.vector_coordinates, function="sph2car")
)
vsph2cyl = _transparent_function(
    vsph2cyl, _partial(_ad.vector_coordinates, function="sph2cyl")
)
vcar2pol = _transparent_function(
    vcar2pol, _partial(_ad.vector_coordinates, function="car2pol")
)
vpol2car = _transparent_function(
    vpol2car, _partial(_ad.vector_coordinates, function="pol2car")
)
sph_harm = _transparent_function(sph_harm, _ad.sph_harm)
vsh_X = _transparent_function(vsh_X, _partial(_ad.spherical_wave, function="vsh_X"))
vsh_Y = _transparent_function(vsh_Y, _partial(_ad.spherical_wave, function="vsh_Y"))
vsh_Z = _transparent_function(vsh_Z, _partial(_ad.spherical_wave, function="vsh_Z"))
vsw_M = _transparent_function(vsw_M, _partial(_ad.spherical_wave, function="vsw_M"))
vsw_N = _transparent_function(vsw_N, _partial(_ad.spherical_wave, function="vsw_N"))
vsw_A = _transparent_function(vsw_A, _partial(_ad.spherical_wave, function="vsw_A"))
vsw_rM = _transparent_function(vsw_rM, _partial(_ad.spherical_wave, function="vsw_rM"))
vsw_rN = _transparent_function(vsw_rN, _partial(_ad.spherical_wave, function="vsw_rN"))
vsw_rA = _transparent_function(vsw_rA, _partial(_ad.spherical_wave, function="vsw_rA"))
vcw_M = _transparent_function(vcw_M, _partial(_ad.cylindrical_wave, function="vcw_M"))
vcw_N = _transparent_function(vcw_N, _partial(_ad.cylindrical_wave, function="vcw_N"))
vcw_A = _transparent_function(vcw_A, _partial(_ad.cylindrical_wave, function="vcw_A"))
vcw_rM = _transparent_function(
    vcw_rM, _partial(_ad.cylindrical_wave, function="vcw_rM")
)
vcw_rN = _transparent_function(
    vcw_rN, _partial(_ad.cylindrical_wave, function="vcw_rN")
)
vcw_rA = _transparent_function(
    vcw_rA, _partial(_ad.cylindrical_wave, function="vcw_rA")
)
vpw_M = _transparent_function(vpw_M, _partial(_ad.plane_wave, function="vpw_M"))
vpw_N = _transparent_function(vpw_N, _partial(_ad.plane_wave, function="vpw_N"))
vpw_A = _transparent_function(vpw_A, _partial(_ad.plane_wave, function="vpw_A"))
tl_vsw_A = _transparent_function(
    tl_vsw_A, _partial(_ad.spherical_translation, singular=True, cross=False)
)
tl_vsw_B = _transparent_function(
    tl_vsw_B, _partial(_ad.spherical_translation, singular=True, cross=True)
)
tl_vsw_rA = _transparent_function(
    tl_vsw_rA, _partial(_ad.spherical_translation, singular=False, cross=False)
)
tl_vsw_rB = _transparent_function(
    tl_vsw_rB, _partial(_ad.spherical_translation, singular=False, cross=True)
)
tl_vcw = _transparent_function(
    tl_vcw, _partial(_ad.cylindrical_translation, singular=True)
)
tl_vcw_r = _transparent_function(
    tl_vcw_r, _partial(_ad.cylindrical_translation, singular=False)
)

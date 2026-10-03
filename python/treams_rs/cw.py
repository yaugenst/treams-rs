"""Coefficients of cylindrical waves: translation, rotation and periodic arrays.

Mirrors ``treams.cw``. A cylindrical mode has an axial wavenumber kz, an
order m and a polarization index pol (0 or 1). Coefficients relate a source
mode to a destination mode: ``kz``, ``mu`` and ``pol`` label the
destination, ``qz``, ``m`` and ``qol`` the source. ``to_sw`` takes the
spherical destination (``l``, ``m``, ``polsw``) first, so there ``m`` is the
destination order and ``mu`` the source order, as in treams.
``periodic_to_pw`` takes the plane-wave destination first. Label arguments
broadcast like NumPy ufunc arguments, and further positional and keyword
arguments (``out``, ``where``) go to the ufunc.

Translation and rotation coefficients are the same for both polarization
conventions. ``to_sw`` takes ``poltype``: ``"helicity"`` (the default when
omitted), where pol 0 is negative and 1 positive helicity, or ``"parity"``,
where pol 0 is the transverse-electric and 1 the transverse-magnetic wave.
``singular=True`` selects singular (outgoing) waves at the source;
``singular=False`` regular ones.

Differences from treams:
    - Invalid modes and nonfinite arguments raise ValueError; treams returns
      0 or NaN for them, or raises ZeroDivisionError.
    - There is no ``config.POLTYPE``: ``poltype=None`` always means
      ``"helicity"``.
    - ``translate_periodic`` drops exact duplicate rows of ``out`` and of
      ``in_`` and returns a matrix with one row or column per distinct mode,
      so it can be smaller than treams' result.

Example::

    import numpy as np
    from treams_rs import cw

    # Rotation about z by phi multiplies the mode of order m by exp(-i m phi).
    assert np.isclose(cw.rotate(0.1, 2, 1, 0.1, 2, 1, 0.3), np.exp(-0.6j))
    # Different axial wavenumbers do not couple.
    assert cw.translate(0.1, 1, 1, 0.2, 1, 1, 1.5, 0.3, 0.2) == 0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _modes, _native, diff
from . import _wave_functions as _autodiff
from ._autodiff_functions import transparent_function as _transparent_function
from ._bases import CylindricalBasis as _CylindricalBasis
from ._polarization import is_helicity as _is_helicity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "periodic_to_pw",
    "rotate",
    "to_sw",
    "translate",
    "translate_periodic",
]


# The wrappers mirror the treams signatures, which forward extra positional
# and keyword arguments to the ufunc. periodic_to_pw is the native ufunc.
#
# Python-scalar fast paths: called with Python numbers and no extra
# arguments, rotate and translate call a native scalar function instead of
# the ufunc, because NumPy's ufunc dispatch costs more than one evaluation.
# Each fast path returns the value and dtype of the ufunc, or raises the same
# ValueError; test_python_scalar_fast_paths_match_the_ufunc in
# tests/bindings/test_ufunc_contract.py checks this.


def rotate(
    kz: ArrayLike,
    mu: ArrayLike,
    pol: ArrayLike,
    qz: ArrayLike,
    m: ArrayLike,
    qol: ArrayLike,
    phi: ArrayLike,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Rotation coefficient exp(-i m phi) if kz = qz, mu = m and pol = qol, else 0.

    Mirrors ``treams.cw.rotate``. ``phi`` rotates about the z axis.

    Args:
        kz: Axial wavenumber of the destination mode.
        mu: Order of the destination mode.
        pol: Polarization index of the destination mode.
        qz: Axial wavenumber of the source mode.
        m: Order of the source mode.
        qol: Polarization index of the source mode.
        phi: Rotation angle about z.
        *args: Further ufunc arguments, such as ``out``.
        **kwargs: Further ufunc keywords, such as ``where``.

    Returns:
        complex128 array with the broadcast shape of the label and coordinate
        arguments. Python numbers give one complex number.

    Autodiff:
        Mode matching is held fixed. Axial labels may move together, but changing
        whether ``kz == qz`` is a discrete transition without a derivative.
    """
    if (
        not args
        and not kwargs
        and isinstance(phi, (int, float))
        and isinstance(kz, (int, float))
        and isinstance(mu, int)
        and isinstance(pol, int)
        and isinstance(qz, (int, float))
        and isinstance(m, int)
        and isinstance(qol, int)
    ):
        return _native.cw_rotate_scalar(kz, mu, pol, qz, m, qol, phi)
    return _native.cw_rotate(kz, mu, pol, qz, m, qol, phi, *args, **kwargs)


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
    """Translation coefficient of one cylindrical mode pair.

    Mirrors ``treams.cw.translate``. The coefficient is ``special.tl_vcw``
    (``special.tl_vcw_r`` for regular waves) if pol = qol, else 0, in either
    polarization convention. ``krr`` is the radial distance times the radial
    wavenumber; ``phi`` and ``z`` are the azimuth and axial distance of the
    displacement ``r_destination - r_source``. The singular coefficient for
    zero displacement is 0, as in treams.

    Args:
        kz: Axial wavenumber of the destination mode.
        mu: Order of the destination mode.
        pol: Polarization index of the destination mode.
        qz: Axial wavenumber of the source mode.
        m: Order of the source mode.
        qol: Polarization index of the source mode.
        krr: Radial distance times the radial wavenumber, real or complex.
        phi: Azimuthal angle of the displacement.
        z: Axial distance of the displacement, not scaled by a wavenumber.
        singular: True expands a singular wave at the source in regular waves at
            the destination; False expands a regular wave in regular waves.
        *args: Further ufunc arguments, such as ``out``.
        **kwargs: Further ufunc keywords, such as ``where``.

    Returns:
        complex128 array with the broadcast shape of the label and coordinate
        arguments. Python numbers give one complex number.

    Autodiff:
        Mode matching is held fixed. To differentiate the shared axial
        wavenumber, use the same varying parameter for ``kz`` and ``qz``. The
        phase derivative is credited to the source ``qz``. Independent changes
        that break ``kz == qz`` are discontinuous and have no derivative.
    """
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
        return _native.cw_translate_scalar(
            kz, mu, pol, qz, m, qol, krr, phi, z, bool(singular)
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
    """Coefficient of a regular cylindrical wave in regular spherical waves.

    Mirrors ``treams.cw.to_sw``. Both waves share the origin. The spherical
    destination mode comes first, so ``m`` is the destination order and ``mu``
    the source order.

    Args:
        l: Degree of the spherical destination mode.
        m: Order of the spherical destination mode.
        polsw: Polarization index of the spherical destination mode.
        kz: Axial wavenumber of the cylindrical source mode.
        mu: Order of the cylindrical source mode.
        polcw: Polarization index of the cylindrical source mode.
        k: Wavenumber of the medium.
        poltype: ``"helicity"`` (default) or ``"parity"``.
        *args: Further ufunc arguments, such as ``out``.
        **kwargs: Further ufunc keywords, such as ``where``.

    Returns:
        complex128 array with the broadcast shape of the label arguments.
    """
    function = _native.cw_to_sw_h if _is_helicity(poltype) else _native.cw_to_sw_p
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
    """Coupling matrix of cylindrical modes in a 1D or 2D lattice.

    Mirrors ``treams.cw.translate_periodic``. Entry (i, j) sums the singular
    translation coefficients from source mode j, repeated at every lattice point
    with the Bloch phase of ``kpar``, to destination mode i. The lattice sums
    converge fast because they split into a real-space and a reciprocal-space
    series (the Ewald split); ``eta`` sets where they split.

    Args:
        ks: Wavenumber of the medium, or two wavenumbers (negative, positive
            helicity) in a chiral medium.
        kpar: Bloch wavevector along the lattice, shape (D,) with D = 1 or 2; a
            number for D = 1.
        a: Lattice vectors as rows, shape (D, D); a number for D = 1.
        rs: Positions of the cylinders in the unit cell, shape (P, 3).
        out: Destination modes: three label arrays (kz, m, pol) for cylinder 0,
            or four arrays (cylinder index, kz, m, pol).
        in_: Source modes in the same form; ``out`` when omitted.
        rsin: Positions of the source cylinders; ``rs`` when omitted.
        eta: Ewald split parameter; 0 selects it automatically.

    Returns:
        complex128 array of shape (destination modes, source modes), with exact
        duplicate modes counted once.

    Differences from treams:
        treams returns one row per entry of ``out`` and one column per entry of
        ``in_``, duplicates included.
    """
    destination, source = _modes.periodic_bases(_CylindricalBasis, rs, out, in_, rsin)
    return diff.lattice_expansion(
        destination,
        source,
        np.broadcast_to(np.asarray(ks, dtype=np.complex128), (2,)),
        kpar,
        a,
        eta=eta,
    )[0]


periodic_to_pw = _native.cw_periodic_to_pw


rotate = _transparent_function(rotate, _autodiff.cw_rotate, module=__name__)
translate = _transparent_function(translate, _autodiff.cw_translate, module=__name__)
to_sw = _transparent_function(to_sw, _autodiff.cw_to_sw, module=__name__)
translate_periodic = _transparent_function(
    translate_periodic, _autodiff.cw_translate_periodic, module=__name__
)
periodic_to_pw = _transparent_function(
    periodic_to_pw, _autodiff.periodic_to_pw, module=__name__
)

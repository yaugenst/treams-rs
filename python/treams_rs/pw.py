"""Coefficients of plane waves: translation, expansion and axis permutation.

Mirrors ``treams.pw``. A plane wave has a wavevector (kx, ky, kz) and a
polarization index pol (0 or 1). Each coefficient relates a source mode to a
destination mode, and the destination comes first: ``to_sw`` and ``to_cw``
take the spherical or cylindrical destination, then the plane-wave source.
Label arguments broadcast like NumPy ufunc arguments, and further positional
and keyword arguments (``out``, ``where``) go to the ufunc.

- ``translate``: the phase ``exp(i k . r)`` of a translation by r.
- ``to_sw`` and ``to_cw``: the expansion of a plane wave in regular
  spherical or cylindrical waves at the origin.
- ``permute_xyz``: the polarization coefficients of a plane wave after a
  cyclic permutation of the Cartesian axes.

``poltype`` selects the polarization convention: ``"helicity"`` (the default
when omitted), where pol 0 is negative and 1 positive helicity, or
``"parity"``, where pol 0 is the transverse-electric and 1 the
transverse-magnetic wave. ``translate`` and ``to_cw`` are the same in both.

Autodiff holds the axial-label matching in ``to_cw`` fixed. If the axial
wavenumber varies, use that same parameter for both ``kzcw`` and ``kzpw``.
Changing whether the labels match is a discrete transition without a
derivative. With matching labels the coefficient is independent of their
shared axial wavenumber.

Differences from treams:
    - Invalid modes and nonfinite arguments raise ValueError; treams returns
      NaN for them, for example in ``to_sw`` with ``|m| > l``.
    - There is no ``config.POLTYPE``: ``poltype=None`` always means
      ``"helicity"``.

Example::

    import numpy as np
    from treams_rs import pw

    k, r = np.array([0.3, 0.2, 0.9]), np.array([1.0, 2.0, 3.0])
    assert np.isclose(pw.translate(*k, *r), np.exp(1j * k @ r))
    # Helicity survives the permutation of the axes.
    assert pw.permute_xyz(*k, 1, 0) == 0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _native
from . import _wave_functions as _autodiff
from ._autodiff_functions import transparent_function as _transparent_function
from ._polarization import is_helicity as _is_helicity

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

__all__ = ["permute_xyz", "to_cw", "to_sw", "translate"]

translate = _native.pw_translate
to_cw = _native.pw_to_cw


# translate and to_cw are native: translate takes a scalar fast path in
# Rust, to_cw is the ufunc. The Python wrappers mirror the treams signatures,
# which forward extra positional and keyword arguments to the ufunc.
#
# Python-scalar fast path: called with Python numbers and no extra
# arguments, permute_xyz calls a native scalar function instead of the ufunc,
# because NumPy's ufunc dispatch costs more than one evaluation. It returns the
# value and dtype of the ufunc, or raises the same ValueError;
# test_python_scalar_fast_paths_match_the_ufunc in
# tests/bindings/test_ufunc_contract.py checks this.


def to_sw(
    l: ArrayLike,
    m: ArrayLike,
    polsw: ArrayLike,
    kx: ArrayLike,
    ky: ArrayLike,
    kz: ArrayLike,
    polpw: ArrayLike,
    poltype: str | None = None,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Coefficient of a plane wave in regular spherical waves.

    Mirrors ``treams.pw.to_sw``. The spherical wave is centred at the origin.
    The spherical destination mode comes first.

    Args:
        l: Degree of the spherical destination mode.
        m: Order of the spherical destination mode.
        polsw: Polarization index of the spherical destination mode.
        kx: x component of the source wavevector.
        ky: y component of the source wavevector.
        kz: z component of the source wavevector.
        polpw: Polarization index of the plane-wave source mode.
        poltype: ``"helicity"`` (default) or ``"parity"``.
        *args: Further ufunc arguments, such as ``out``.
        **kwargs: Further ufunc keywords, such as ``where``.

    Returns:
        complex128 array with the broadcast shape of the label arguments.

    Differences from treams:
        ``|m| > l`` raises ValueError; treams returns NaN.
    """
    function = _native.pw_to_sw_h if _is_helicity(poltype) else _native.pw_to_sw_p
    return function(l, m, polsw, kx, ky, kz, polpw, *args, **kwargs)


def permute_xyz(
    kx: ArrayLike,
    ky: ArrayLike,
    kz: ArrayLike,
    p: ArrayLike,
    q: ArrayLike,
    poltype: str | None = None,
    inverse: bool = False,
    *args: object,
    **kwargs: object,
) -> complex | NDArray[np.complex128]:
    """Polarization coefficient of a plane wave in cyclically permuted axes.

    Mirrors ``treams.pw.permute_xyz``. The permuted axes are
    (x', y', z') = (z, x, y), in which the wavevector reads (kz, kx, ky). The
    coefficient is the amplitude of polarization ``p`` in the permuted axes
    for a wave of polarization ``q`` in the original axes. ``inverse=True``
    uses (x', y', z') = (y, z, x) instead. The values equal treams'; the
    treams docstring writes (y, z, x) for ``inverse=False``.

    Args:
        kx: x component of the wavevector, real or complex.
        ky: y component of the wavevector, real or complex.
        kz: z component of the wavevector, real or complex.
        p: Polarization index in the permuted axes (destination).
        q: Polarization index in the original axes (source).
        poltype: ``"helicity"`` (default) or ``"parity"``.
        inverse: If True, use the inverse permutation.
        *args: Further ufunc arguments, such as ``out``.
        **kwargs: Further ufunc keywords, such as ``where``.

    Returns:
        complex128 array with the broadcast shape of the arguments. Python
        numbers give one complex number.
    """
    helicity = _is_helicity(poltype)
    if (
        not args
        and not kwargs
        and isinstance(p, int)
        and isinstance(q, int)
        and isinstance(kx, (int, float, complex))
        and isinstance(ky, (int, float, complex))
        and isinstance(kz, (int, float, complex))
    ):
        return _native.pw_permute_xyz_scalar(kx, ky, kz, p, q, helicity, bool(inverse))
    if helicity:
        function = (
            _native.pw_permute_xyz_inverse_h if inverse else _native.pw_permute_xyz_h
        )
    else:
        function = (
            _native.pw_permute_xyz_inverse_p if inverse else _native.pw_permute_xyz_p
        )
    return function(kx, ky, kz, p, q, *args, **kwargs)


translate = _transparent_function(translate, _autodiff.pw_translate, module=__name__)
to_sw = _transparent_function(to_sw, _autodiff.pw_to_sw, module=__name__)
to_cw = _transparent_function(to_cw, _autodiff.pw_to_cw, module=__name__)
permute_xyz = _transparent_function(
    permute_xyz, _autodiff.pw_permute_xyz, module=__name__
)

# ruff: noqa: E741 - preserve upstream degree argument l
"""Direct plane-wave phases, multipole coefficients and coordinate permutations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _native
from ._core import _poltype

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

translate = _native.pw_translate
to_cw = _native.pw_to_cw


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
    """Regular spherical expansion coefficient in helicity or parity convention."""
    function = _native.pw_to_sw_h if _poltype(poltype) else _native.pw_to_sw_p
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
    """Cyclic xyz-to-zxy coefficient; inverse selects the opposite cycle."""
    helicity = _poltype(poltype)
    if (
        not args
        and not kwargs
        and isinstance(p, int)
        and isinstance(q, int)
        and isinstance(kx, (int, float, complex))
        and isinstance(ky, (int, float, complex))
        and isinstance(kz, (int, float, complex))
    ):
        return _native.plane_permutation_scalar(kx, ky, kz, p, q, helicity, inverse)
    if helicity:
        function = _native.pw_inverse_h if inverse else _native.pw_permute_h
    else:
        function = _native.pw_inverse_p if inverse else _native.pw_permute_p
    return function(kx, ky, kz, p, q, *args, **kwargs)

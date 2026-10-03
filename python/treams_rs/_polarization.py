"""Polarization convention: helicity or parity channels (treams ``poltype``).

Every wave basis labels its two polarizations ``pol`` 0 and 1:

- ``"helicity"``: 0 is negative and 1 is positive helicity. A helicity wave is
  ``(N + (2 pol - 1) M) / sqrt(2)`` in terms of the parity waves.
- ``"parity"``: 0 is the transverse-electric (magnetic, ``M``) wave and 1 the
  transverse-magnetic (electric, ``N``) wave.

Parity channels need achiral media; helicity channels work in any medium.

Omitted conventions mean helicity. treams reads its default from the mutable
``treams.config.POLTYPE``; treams-rs has no such setting, so a result depends
only on its arguments. Pass ``poltype="parity"`` (or ``polarization="parity"``
on physics objects) to use parity channels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._bases import mode_lookup

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._bases import FieldBasis
    from ._material import Material

__all__ = [
    "DEFAULT_POLTYPE",
    "OPPOSITE_POLTYPE",
    "PARITY_CHANGE",
    "change_polarization",
    "check_poltype_medium",
    "is_helicity",
    "pol_partners",
    "resolve_poltype",
    "target_poltype",
]

DEFAULT_POLTYPE = "helicity"


def resolve_poltype(value: str | None) -> str:
    """Checked convention name; None gives ``DEFAULT_POLTYPE``."""
    value = DEFAULT_POLTYPE if value is None else value
    if value not in ("helicity", "parity"):
        raise ValueError("polarization type must be helicity or parity")
    return value


def check_poltype_medium[P: str | None](poltype: P, *media: Material | None) -> P:
    """Parity channels require every given medium to be achiral."""
    if poltype == "parity" and any(m is not None and m.ischiral for m in media):
        raise ValueError("parity polarization requires an achiral embedding medium")
    return poltype


def is_helicity(value: str | None) -> bool:
    """True for helicity channels (also when omitted), False for parity."""
    return resolve_poltype(value) == "helicity"


OPPOSITE_POLTYPE = {"helicity": "parity", "parity": "helicity"}


def target_poltype(current: str, target: str | None) -> str | None:
    """Destination convention, the opposite one by default; None when unchanged."""
    target = OPPOSITE_POLTYPE[current] if target is None else target
    if target not in OPPOSITE_POLTYPE:
        raise ValueError("polarization conversion must switch helicity and parity")
    return None if target == current else target


def pol_partners(basis: FieldBasis) -> NDArray[np.intp]:
    """Index of the mode with the same labels and the other ``pol`` of each mode.

    Every mode must have its partner in the basis.
    """
    lookup = mode_lookup(basis)
    try:
        return np.array(
            [lookup[(*mode[:-1], 1 - mode[-1])] for mode in basis.modes],
            dtype=np.intp,
        )
    except KeyError:
        raise ValueError(
            "polarization conversion requires both polarizations of each mode"
        ) from None


def change_polarization(
    values: NDArray[np.complex128], basis: FieldBasis, axes: tuple[int, ...]
) -> NDArray[np.complex128]:
    """Apply ``changepoltype(basis=basis)`` along each axis without a dense matmul.

    The square change matrix C is symmetric with two entries per row,
    ``C[i, i] = +/-sqrt(1/2)`` (negative for label 0) and ``sqrt(1/2)`` at the
    partner of mode i, so ``C @ x`` and ``x @ C.T`` both cost O(size).
    """
    partners = pol_partners(basis)
    half = np.sqrt(0.5)
    diagonal = np.where(basis.pol == 0, -half, half)
    for axis in axes:
        shape = [1] * values.ndim
        shape[axis] = -1
        values = diagonal.reshape(shape) * values + half * np.take(
            values, partners, axis=axis
        )
    return values


# Helicity-to-parity change of one (pol 0, pol 1) pair; it is its own inverse.
PARITY_CHANGE = np.array([[-1, 1], [1, 1]]) * np.sqrt(0.5)

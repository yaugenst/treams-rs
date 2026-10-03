"""Mode labels: label arrays turned into mode rows and bases for EBCM and periodic couplings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._bases import SphericalBasis

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from ._bases import CylindricalBasis

__all__ = ["Modes", "ebcm_modes", "label_rows", "periodic_bases"]

type Modes = SphericalBasis | tuple[ArrayLike, ArrayLike, ArrayLike]
"""EBCM modes: a ``SphericalBasis`` at the origin, or (degree, order, pol) arrays."""


def ebcm_modes(value: Modes) -> list[tuple[int, int, int]]:
    """Integer (degree, order, pol) rows; a basis needs one position, the origin."""
    if isinstance(value, SphericalBasis):
        if not value.isglobal or np.any(value.positions != 0):
            raise ValueError("EBCM basis must have one origin at zero")
        return [(int(degree), int(m), int(p)) for _, degree, m, p in value.modes]
    modes = np.column_stack(value)
    if (
        modes.ndim != 2
        or modes.shape[1] != 3
        or not np.isfinite(modes).all()
        or np.any(modes != np.floor(modes))
    ):
        raise ValueError("EBCM modes must be integer degree/order/polarization triples")
    return [(int(degree), int(m), int(p)) for degree, m, p in modes]


def label_rows(labels: Sequence[ArrayLike]) -> list[tuple[float, ...]]:
    """Mode rows from three or four broadcast label arrays."""
    if len(labels) not in (3, 4):
        raise ValueError("modes require three or four label arrays")
    arrays = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in labels))
    return [
        tuple(float(v) for v in row)
        for row in zip(*(v.flat for v in arrays), strict=True)
    ]


def periodic_bases[B: (SphericalBasis, CylindricalBasis)](
    family: type[B],
    rs: ArrayLike,
    out: Sequence[ArrayLike],
    in_: Sequence[ArrayLike] | None,
    rsin: ArrayLike | None,
) -> tuple[B, B]:
    """Destination and source bases of the label-array periodic couplings."""
    destination = family(label_rows(out), np.atleast_2d(rs))
    source = family(
        label_rows(out if in_ is None else in_),
        np.atleast_2d(rs if rsin is None else rsin),
    )
    return destination, source

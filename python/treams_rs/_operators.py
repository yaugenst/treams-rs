"""Explicit basis operators backed by native numerical kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import diff

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ._core import CylindricalWaveBasis, SphericalWaveBasis

    type Basis = SphericalWaveBasis | CylindricalWaveBasis


def rotate(
    phi: float, theta: float = 0, psi: float = 0, *, basis: Basis | tuple[Basis, Basis]
) -> NDArray[np.complex128]:
    """Rotation matrix in the z-y-z convention; basis origins remain fixed."""
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    return diff.rotation([phi, theta, psi], destination, source)[0]

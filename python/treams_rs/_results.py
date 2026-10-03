"""Named results of cross-section, power, band and port queries.

The NumPy objects and the framework adapters (advect, jax, torch) return the
same classes; the type parameter is the field type.
"""

from __future__ import annotations

from typing import Any, NamedTuple

__all__ = [
    "BandModes",
    "CircularDichroism",
    "CrossSections",
    "PowerBalance",
    "ScatteredPorts",
]


class CrossSections[T](NamedTuple):
    """Scattering and extinction cross sections; absorption is their difference.

    Units: areas (length squared) from TMatrix and widths (length) from
    CylindricalTMatrix, in the length unit of 1/k0. Fields are floats for
    TMatrix and CylindricalTMatrix, and framework arrays in advect, jax and
    torch.
    """

    scattering: T
    extinction: T

    @property
    def absorption(self) -> T:
        """Extinguished power minus scattered power, with the same normalization."""
        # T has no arithmetic bound; floats and every array type subtract.
        extinction: Any = self.extinction
        return extinction - self.scattering


class PowerBalance[T](NamedTuple):
    """Transmitted and reflected power as fractions of the incident power.

    Units: none; a lossless network gives transmission + reflection = 1.
    Fields are floats for SMatrix.power, and framework arrays in advect, jax
    and torch.
    """

    transmission: T
    reflection: T

    @property
    def absorption(self) -> T:
        """Fraction not carried by transmitted or reflected power."""
        # T has no arithmetic bound; floats and every array type subtract.
        transmission: Any = self.transmission
        return 1 - transmission - self.reflection


class BandModes[T](NamedTuple):
    """Bloch wavenumbers along the basis normal and matching right eigenvectors.

    Units: wavenumbers in inverse length, the unit of k0; eigenvectors
    dimensionless, with ``eigenvectors[:, i]`` the mode of ``wavenumbers[i]``.
    Fields are complex NumPy arrays for SMatrix.bands, and framework arrays in
    advect, jax and torch.
    """

    wavenumbers: T
    eigenvectors: T


class ScatteredPorts[W](NamedTuple):
    """Outgoing waves on the positive and negative side of the stack normal.

    Units: plane-wave amplitudes in the normalization of the incident wave.
    Fields are Wave objects for SMatrix.scatter, and PortWave objects in
    advect, jax and torch.
    """

    positive: W
    negative: W


class CircularDichroism(NamedTuple):
    """Contrasts against opposite polarization, normalized by the respective summed powers.

    Units: none; each contrast lies between -1 and 1.
    """

    transmission: float
    outgoing_power: float

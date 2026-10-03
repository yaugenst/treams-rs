"""Isotropic chiral materials; mirrors treams._material."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import _native
from ._upstream import UpstreamMembers

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import ArrayLike, NDArray

__all__ = ["Material", "MaterialLike", "as_material"]


@dataclass(frozen=True, init=False)
class Material(UpstreamMembers):
    """Isotropic reciprocal material with relative epsilon, mu and chirality kappa.

    Mirrors ``treams.Material``. All three parameters are dimensionless complex
    scalars. A single scalar specifies epsilon with mu=1 and kappa=0; a
    tuple/list specifies ``(epsilon, mu, kappa)``, with omitted trailing values
    taking those defaults. Passing another Material copies its three values.
    Instances are immutable. Call or iterate the object to get the parameters
    in that order.
    """

    epsilon: complex
    """Relative permittivity."""
    mu: complex
    """Relative permeability."""
    kappa: complex
    """Chirality parameter, dimensionless."""

    def __init__(self, epsilon: MaterialLike = 1, mu: complex = 1, kappa: complex = 0):
        if isinstance(epsilon, Material):
            epsilon, mu, kappa = epsilon()
        elif isinstance(epsilon, (tuple, list)) or (
            isinstance(epsilon, np.ndarray) and epsilon.ndim > 0
        ):
            if len(epsilon) > 3:
                raise ValueError("invalid material definition")
            epsilon, mu, kappa = tuple(epsilon) + (1, 1, 0)[len(epsilon) :]
        object.__setattr__(self, "epsilon", complex(epsilon))
        object.__setattr__(self, "mu", complex(mu))
        object.__setattr__(self, "kappa", complex(kappa))

    def __call__(self) -> tuple[complex, complex, complex]:
        """The parameters as a tuple ``(epsilon, mu, kappa)``."""
        return self.epsilon, self.mu, self.kappa

    def __iter__(self) -> Iterator[complex]:
        return iter(self())

    @classmethod
    def from_refractive_index(
        cls, n: complex = 1, impedance: complex | None = None, kappa: complex = 0
    ) -> Material:
        """Construct epsilon=n/impedance and mu=n*impedance.

        ``n`` is the mean refractive index and impedance is relative to vacuum.
        Omitting impedance uses 1/n, giving mu=1. Kappa stays dimensionless.
        """
        impedance = 1 / n if impedance is None else impedance
        return cls(n / impedance, n * impedance, kappa)

    @classmethod
    def from_helicity_indices(
        cls, ns: tuple[complex, complex] = (1, 1), impedance: complex | None = None
    ) -> Material:
        """Construct from negative/positive-helicity indices ``(n_minus, n_plus)``.

        Their mean gives n and half their difference gives kappa. Omitting
        impedance uses the same nonmagnetic convention as ``from_refractive_index``.
        """
        return cls.from_refractive_index(sum(ns) / 2, impedance, (ns[1] - ns[0]) / 2)

    @property
    def n(self) -> complex:
        """Square root of epsilon*mu, with sign chosen for nonnegative imaginary part."""
        value = cmath.sqrt(self.epsilon * self.mu)
        return -value if value.imag < 0 else value

    @property
    def nmp(self) -> NDArray[np.complex128]:
        """Shape-(2,) complex128 refractive indices for the labels pol 0 and 1.

        Start from the principal square root of epsilon*mu minus/plus kappa,
        then choose each sign for a nonnegative imaginary part. Array order is
        negative/positive helicity; default mode bases instead list label 1 first.
        """
        return _native.refractive_indices(self.epsilon, self.mu, self.kappa)

    @property
    def impedance(self) -> complex:
        """Principal square root of mu/epsilon, relative to vacuum impedance."""
        return cmath.sqrt(self.mu / self.epsilon)

    @property
    def isreal(self) -> bool:
        """Whether epsilon, mu and kappa are all real, as for a lossless medium."""
        return all(value.imag == 0 for value in self)

    @property
    def ischiral(self) -> bool:
        """Whether the chirality kappa is nonzero; parity channels need it zero."""
        return self.kappa != 0

    def ks(self, k0: float) -> NDArray[np.complex128]:
        """Return ``k0 * nmp`` in negative/positive-helicity order, shape (2,).

        The vacuum angular wavenumber k0 is 2*pi/wavelength, in inverse units
        of the lengths used elsewhere in the simulation.
        """
        return k0 * self.nmp

    def kzs(
        self, k0: float, kx: ArrayLike, ky: ArrayLike, pol: ArrayLike = (0, 1)
    ) -> NDArray[np.complex128]:
        """Axial wavevectors on the outgoing branch (nonnegative imaginary part)."""
        return _native.wave_vector_z(
            kx, ky, self.ks(k0)[np.asarray(pol, dtype=np.int64)]
        )

    def krhos(
        self, k0: float, kz: ArrayLike, pol: ArrayLike = (0, 1)
    ) -> NDArray[np.complex128]:
        """Radial wavevectors on the same outgoing branch as kzs."""
        return self.kzs(k0, kz, 0, pol)


type MaterialLike = (
    Material | complex | tuple[complex, ...] | list[complex] | NDArray[np.generic]
)


def as_material(material: MaterialLike) -> Material:
    """``material`` as a Material; an existing one is reused, being immutable."""
    return material if type(material) is Material else Material(material)

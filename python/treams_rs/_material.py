"""Isotropic chiral materials; mirrors treams._material."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _native
from ._dispatch import backend_for
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

    epsilon: Any
    """Relative permittivity."""
    mu: Any
    """Relative permeability."""
    kappa: Any
    """Chirality parameter, dimensionless."""

    def __init__(self, epsilon: Any = 1, mu: Any = 1, kappa: Any = 0):
        if isinstance(epsilon, Material):
            epsilon, mu, kappa = epsilon()
        else:
            if (backend := backend_for(epsilon)) is not None:
                # Framework array conversion retains traced sequence components.
                epsilon = backend.asarray(epsilon, dtype=None)
            if isinstance(epsilon, (tuple, list)) or getattr(epsilon, "ndim", 0) > 0:
                if len(epsilon) > 3:
                    raise ValueError("invalid material definition")
                epsilon, mu, kappa = tuple(epsilon) + (1, 1, 0)[len(epsilon) :]
        backend = backend_for(epsilon, mu, kappa)
        for name, value in (("epsilon", epsilon), ("mu", mu), ("kappa", kappa)):
            if backend is None:
                value = complex(value)
            elif getattr(value, "ndim", 0) != 0:
                raise ValueError("material parameters must be scalars")
            object.__setattr__(self, name, value)

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
        if (backend := backend_for(self)) is not None:
            return backend.upper_half(
                backend.xp.sqrt(backend.array(self.epsilon * self.mu, complex_=True))
            )
        value = cmath.sqrt(self.epsilon * self.mu)
        return -value if value.imag < 0 else value

    @property
    def nmp(self) -> NDArray[np.complex128]:
        """Shape-(2,) complex128 refractive indices for the labels pol 0 and 1.

        Start from the principal square root of epsilon*mu minus/plus kappa,
        then choose each sign for a nonnegative imaginary part. Array order is
        negative/positive helicity; default mode bases instead list label 1 first.
        """
        if (backend := backend_for(self)) is not None:
            return backend.ks(self, 1.0)
        return _native.refractive_indices(self.epsilon, self.mu, self.kappa)

    @property
    def impedance(self) -> complex:
        """Principal square root of mu/epsilon, relative to vacuum impedance."""
        if (backend := backend_for(self)) is not None:
            return backend.impedance(self)
        return cmath.sqrt(self.mu / self.epsilon)

    @property
    def isreal(self) -> bool:
        """Whether epsilon, mu and kappa are all real, as for a lossless medium."""
        if (backend := backend_for(self)) is not None:
            return backend.xp.all(
                backend.xp.imag(backend.array(tuple(self), complex_=True)) == 0
            )
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
        if (backend := backend_for(self, k0)) is not None:
            return backend.ks(self, k0)
        return k0 * self.nmp

    def kzs(
        self, k0: float, kx: ArrayLike, ky: ArrayLike, pol: ArrayLike = (0, 1)
    ) -> NDArray[np.complex128]:
        """Axial wavevectors on the outgoing branch (nonnegative imaginary part)."""
        if (backend := backend_for(self, k0, kx, ky)) is not None:
            ks = backend.ks(self, k0)[np.asarray(pol, dtype=np.int64)]
            return backend.upper_half(
                backend.xp.sqrt(
                    ks**2
                    - backend.array(kx, complex_=True) ** 2
                    - backend.array(ky, complex_=True) ** 2
                )
            )
        return _native.wave_vector_z(
            kx, ky, self.ks(k0)[np.asarray(pol, dtype=np.int64)]
        )

    def _plane_ks(self, k0: float) -> NDArray[np.complex128]:
        """Wavenumbers after checking the plane-wave polarization convention."""
        if (backend := backend_for(self, k0)) is not None:
            return backend.plane_ks(self, k0)
        _check_plane_material(self.epsilon, self.mu, self.kappa)
        return self.ks(k0)

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


def _check_plane_material(epsilon: complex, mu: complex, kappa: complex) -> None:
    """Reject constitutive branches the principal-norm plane polarizations cannot represent.

    Their curl eigenvalues use the principal sqrt(k dot k), whereas Maxwell's
    equations require mu/Z minus/plus kappa. These agree only when mu/Z equals
    the principal sqrt(epsilon*mu) and both chiral indices are principal roots.
    Gain with positive real indices remains supported on the outgoing branch.
    """
    epsilon, mu, kappa = complex(epsilon), complex(mu), complex(kappa)
    if epsilon == 0 or mu == 0:
        raise ValueError("plane waves require nonzero permittivity and permeability")
    n = cmath.sqrt(epsilon * mu)
    physical_n = mu / cmath.sqrt(mu / epsilon)
    indices = (n - kappa, n + kappa)
    if abs(physical_n - n) > abs(physical_n + n) or any(
        index.real < 0 or (index.real == 0 and index.imag < 0) for index in indices
    ):
        raise ValueError(
            "plane-wave polarizations do not support this material branch: "
            "require mu/Z = sqrt(epsilon*mu) and principal chiral indices"
        )

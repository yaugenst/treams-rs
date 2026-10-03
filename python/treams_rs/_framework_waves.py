"""Framework waves: multipole waves, plane waves and port waves."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from ._framework_smatrix import SMatrix

import numpy as np

from . import diff
from ._bases import ALIGNMENT_AXIS, CylindricalBasis, PlaneWavePorts, SphericalBasis
from ._framework_backend import Backend, Basis, Material, as_material, port_modes

__all__ = ["HasPorts", "PlaneWave", "PortSet", "PortWave", "Wave"]


def _select_gradient(context: Any, index: int) -> Callable[[Any], Any]:
    """Pullback of ``context`` that keeps only the gradient of argument ``index``."""
    return lambda g: context.pullback(g)[index]


class _Fields:
    # H, D, B, G and F from one electric-field evaluation. These weights match
    # advect.hfield and _fields.rs_weights (which advect.gfield and
    # advect.ffield reuse). They have their own expressions for framework
    # arrays, because the NumPy expressions in _fields must stay bitwise
    # identical.

    medium: Material
    """Medium in which the wave travels."""
    polarization: str
    """Polarization convention, "helicity" or "parity"."""
    _backend: Backend

    def efield(self, points: Any) -> Any:
        raise NotImplementedError

    def _combined(self, electric: Any, magnetic: Any, points: Any) -> Any:
        """``electric * E + magnetic * H`` from one native electric-field evaluation."""
        raise NotImplementedError

    def _weights(self, electric: Any, magnetic: Any, signs: Any) -> Any:
        # In the helicity basis H is E of the coefficients times -i(2 pol - 1)/Z,
        # so a linear combination of E and H is E of reweighted coefficients.
        b = self._backend
        return electric + magnetic * (-1j * b.array(signs) / b.impedance(self.medium))

    def hfield(self, points: Any) -> Any:
        """Magnetic field in relative vacuum impedance units."""
        return self._combined(0.0, 1.0, points)

    def dfield(self, points: Any) -> Any:
        """Electric displacement divided by vacuum permittivity."""
        return self._combined(self.medium.epsilon, 1j * self.medium.kappa, points)

    def bfield(self, points: Any) -> Any:
        """Magnetic flux density times vacuum light speed."""
        return self._combined(-1j * self.medium.kappa, self.medium.mu, points)

    def gfield(self, pol: int, points: Any) -> Any:
        """Helicity-resolved G field, normalized as in treams for this wave family."""
        if pol not in (0, 1):
            raise ValueError("helicity label must be 0 or 1")
        # PlaneWave and PortWave have no multipole basis.
        spherical = isinstance(getattr(self, "basis", None), SphericalBasis)
        normalization = np.sqrt(2.0) if spherical else 1.0
        if self.polarization == "helicity":
            normalization *= 0.5
        impedance = self._backend.impedance(self.medium)
        return normalization * self._combined(
            1.0, (2 * pol - 1) * 1j * impedance, points
        )

    def ffield(self, pol: int, points: Any) -> Any:
        """G field with the chiral index weighting in the helicity convention."""
        value = self.gfield(pol, points)
        if self.polarization == "parity":
            return value
        ks = self._backend.ks(self.medium, 1.0)
        return value * 2 * ks[pol] / self._backend.xp.sum(ks)


class Wave(_Fields):
    """Multipole wave of framework arrays: coefficients with basis, medium and kind.

    The constructor ``wave`` of advect, jax and torch builds it, and
    ``TMatrix.scatter``, ``Cluster.scatter`` and ``in_basis`` return one. The
    basis labels and the kind are fixed; the coefficients, k0, the medium and
    the positions may carry gradients. Every method uses ``positions`` and
    ignores ``basis.positions``.
    """

    coefficients: Any
    """One coefficient per basis mode, a framework array."""
    basis: Basis
    """Fixed mode labels."""
    k0: Any
    """Vacuum wavenumber."""
    positions: Any
    """Expansion centres, shape (positions, 3), a framework array."""

    @property
    def kind(self) -> str:
        """Radial kind: "regular" (incident) or "singular" (outgoing)."""
        return "singular" if self.singular else "regular"

    def __init__(
        self,
        coefficients: Any,
        *,
        basis: Basis,
        k0: Any,
        medium: Material,
        backend: Backend,
        positions: Any = None,
        singular: bool = False,
        polarization: str = "helicity",
    ):
        if polarization == "parity":
            coefficients = backend.require_achiral(coefficients, medium)
        self.coefficients, self.basis, self.k0 = coefficients, basis, k0
        self.medium, self._backend, self.singular = medium, backend, singular
        self.polarization = polarization
        self.positions = backend.positions(basis, positions)

    @override
    def efield(self, points: Any) -> Any:
        """Electric field at points (..., 3), differentiable in points and wave values."""
        b = self._backend
        points = b.array(points)
        shape = tuple(points.shape)
        basis = self.basis

        def record(c: Any, p: Any, o: Any, ks: Any) -> Any:
            return diff.field(
                c,
                p,
                type(basis)(basis.modes, o),
                ks,
                poltype=self.polarization,
                singular=self.singular,
            )

        return b.apply(
            record,
            (int(np.prod(shape[:-1])), 3),
            self.coefficients,
            points.reshape(-1, 3),
            self.positions,
            b.ks(self.medium, self.k0),
        ).reshape(shape)

    @override
    def _combined(self, electric: Any, magnetic: Any, points: Any) -> Any:
        if self.polarization != "helicity":
            return self.with_polarization("helicity")._combined(
                electric, magnetic, points
            )
        weights = self._weights(electric, magnetic, 2 * self.basis.pol - 1)
        return Wave(
            self.coefficients * weights,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=self._backend,
            positions=self.positions,
            singular=self.singular,
        ).efield(points)

    def in_basis(
        self, basis: Basis, *, positions: Any = None, singular: bool | None = None
    ) -> Wave:
        """Re-expand into another multipole basis; the field stays the same."""
        target_singular = self.singular if singular is None else singular
        if target_singular and not self.singular:
            raise ValueError("a regular wave cannot be converted to singular waves")
        b = self._backend
        destination_positions = b.positions(basis, positions)
        operator = b.expansion(
            basis,
            self.basis,
            destination_positions,
            self.positions,
            b.ks(self.medium, self.k0),
            poltype=self.polarization,
            singular=self.singular and not target_singular,
        )
        return Wave(
            operator @ self.coefficients,
            basis=basis,
            k0=self.k0,
            medium=self.medium,
            backend=b,
            positions=destination_positions,
            singular=target_singular,
            polarization=self.polarization,
        )

    def with_polarization(self, polarization: str) -> Wave:
        """Change the static multipole polarization convention."""
        return Wave(
            self._backend.change_polarization(
                self.coefficients, self.basis, self.polarization, polarization, (0,)
            ),
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=self._backend,
            positions=self.positions,
            singular=self.singular,
            polarization=polarization,
        )


class PlaneWave(_Fields):
    """Plane wave with a fixed direction and two helicity amplitudes.

    The constructor ``plane_wave`` of advect, jax and torch builds it. The
    direction is fixed; the amplitudes, k0 and the medium may carry
    gradients. The direction fixes the angular factors and the phases carry
    the k0 and position derivatives, so incidence along an axis has gradients
    too.
    """

    def __init__(
        self,
        direction: Any,
        pol: Any,
        *,
        k0: Any,
        medium: Any,
        backend: Backend,
    ):
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction) if direction.shape == (3,) else 0.0
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("direction must be a finite nonzero Cartesian vector")
        self.direction = direction / norm
        choices = {"positive_helicity": (0.0, 1.0), "negative_helicity": (1.0, 0.0)}
        if isinstance(pol, str):
            if pol not in choices:
                raise ValueError(
                    "named pol must be positive_helicity or negative_helicity"
                )
            pol = choices[pol]
        elif isinstance(pol, (int, np.integer)):
            if pol not in (0, 1):
                raise ValueError("helicity label must be 0 or 1")
            pol = (1.0, 0.0) if pol == 0 else (0.0, 1.0)
        self.coefficients = backend.array(pol, complex_=True)
        if self.coefficients.shape != (2,):
            raise ValueError("pol requires two helicity amplitudes or a helicity name")
        self.k0, self.medium = backend.array(k0), as_material(medium)
        self._backend = backend
        self.polarization = "helicity"

    def in_basis(
        self, basis: Basis, *, positions: Any = None, singular: bool | None = None
    ) -> Wave:
        """Expand into regular multipole waves of ``basis`` at ``positions``.

        ``positions`` defaults to ``basis.positions``; ``singular=True``
        raises ValueError.
        """
        if singular:
            raise ValueError("plane illumination expands to regular multipoles")
        b = self._backend
        positions = b.positions(basis, positions)
        vectors = self._vectors()
        # Fixed-direction angular coefficients and dynamic translation phases
        # separate the axis gauge from physical frequency/position derivatives.
        if isinstance(basis, CylindricalBasis) and (
            np.any(basis.kz != 0) or self.direction[2] != 0
        ):
            raise NotImplementedError(
                "cylindrical plane illumination requires kz=0 and transverse incidence; treams_rs.advect.plane_expansion handles fixed axial sectors"
            )
        zero_basis = type(basis)(basis.modes, np.zeros_like(basis.positions))

        def record_angular(vectors: Any) -> Any:
            value, context = diff.plane_expansion(
                zero_basis, vectors, [0, 1], fixed_vectors=True
            )
            return value, _select_gradient(context, 1)

        angular = b.apply(record_angular, (len(basis), 2), vectors)
        phases = b.apply(diff.plane_phases, (positions.shape[0], 2), positions, vectors)
        operator = angular * phases[basis.pidx.copy()]
        return Wave(
            operator @ self.coefficients,
            basis=basis,
            k0=self.k0,
            medium=self.medium,
            backend=b,
            positions=positions,
        )

    def _vectors(self) -> Any:
        """Wavevectors of the two helicities, one row each."""
        b = self._backend
        return b.ks(self.medium, self.k0)[:, None] * b.array(self.direction)[None, :]

    @override
    def efield(self, points: Any) -> Any:
        """Electric field at points (..., 3), differentiable in points and wave values."""
        return _plane_efield(self, points, [0, 1], lambda _: True)

    @override
    def _combined(self, electric: Any, magnetic: Any, points: Any) -> Any:
        weights = self._weights(electric, magnetic, [-1.0, 1.0])
        return PlaneWave(
            self.direction,
            self.coefficients * weights,
            k0=self.k0,
            medium=self.medium,
            backend=self._backend,
        ).efield(points)


@dataclass(frozen=True, slots=True)
class PortSet:
    """Plane-wave ports shared by an S-matrix and the port waves it returns.

    ``modes`` labels each port ``(group, pol)``; ``groups`` and ``pols`` hold
    the two labels as index arrays. The ports of one group share one row of
    ``transverse_wavevectors``: the two wavevector components in the port
    plane, in ``alignment`` order. Ports built from a basis have fixed
    wavevectors (``fixed_q=True``). Diffraction-order ports have ``basis=None``
    and ``fixed_q=False``: their wavevectors follow the lattice and the Bloch
    vector and carry derivatives.
    """

    basis: PlaneWavePorts | None
    modes: tuple[tuple[int, int], ...]
    alignment: str
    transverse_wavevectors: Any
    fixed_q: bool
    groups: NDArray[np.int_] = field(init=False, repr=False, compare=False)
    pols: NDArray[np.int_] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Writable index arrays: Torch warns on read-only NumPy index arrays.
        groups, pols = np.array(self.modes, dtype=int).reshape(-1, 2).T
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "pols", pols)

    @classmethod
    def from_basis(cls, basis: PlaneWavePorts, backend: Backend) -> PortSet:
        """Fixed ports of a static basis, one group per distinct wavevector."""
        groups, modes = port_modes(basis)
        return cls(
            basis=basis,
            modes=modes,
            alignment=basis.alignment,
            transverse_wavevectors=backend.array(np.array(groups)),
            fixed_q=True,
        )


class HasPorts:
    """Read-only views of the ``ports`` of an S-matrix or port wave."""

    ports: PortSet

    @property
    def modes(self) -> tuple[tuple[int, int], ...]:
        """(group, pol) label of each port."""
        return self.ports.modes

    @property
    def transverse_wavevectors(self) -> Any:
        """In-plane wavevector components, one row per port group."""
        return self.ports.transverse_wavevectors


class PortWave(HasPorts, _Fields):
    """Plane-wave amplitudes on the ports of one side of an S-matrix.

    ``SMatrix.scatter`` returns two in a ScatteredPorts. ``positive`` is True
    for the wave on the positive side, which travels along the positive
    normal. The amplitudes, k0, the exterior medium and the transverse
    wavevectors of diffraction-order ports may carry gradients.
    """

    coefficients: Any
    """One amplitude per port, a framework array."""
    k0: Any
    """Vacuum wavenumber."""
    ports: PortSet
    """Plane-wave ports, shared with the S-matrix."""
    positive: bool
    """True on the positive side of the S-matrix, False on the negative side."""

    def __init__(self, coefficients: Any, *, system: SMatrix, positive: bool):
        self.coefficients = coefficients
        self.k0 = system.k0
        self.medium = system.media[0 if positive else 1]  # (positive, negative)
        self._backend = system._backend
        self.polarization = system.polarization
        self.ports = system.ports
        self.positive = positive

    def _vectors(self) -> Any:
        """Wavevector of each port, one row each."""
        b = self._backend
        local = b.plane_vectors(
            self.ports.transverse_wavevectors[self.ports.groups],
            b.ks(self.medium, self.k0)[self.ports.pols],
            positive=self.positive,
        )
        axis = ALIGNMENT_AXIS[self.ports.alignment]
        order = np.argsort([(axis + 1) % 3, (axis + 2) % 3, axis])
        return local[:, order]

    @override
    def efield(self, points: Any) -> Any:
        """Electric field at points (..., 3), differentiable in points and wave values."""
        axis = ALIGNMENT_AXIS[self.ports.alignment]

        def fixed(v: Any) -> bool:
            return bool(
                self.ports.fixed_q
                and np.all(v[:, [i for i in range(3) if i != axis]] == 0)
            )

        return _plane_efield(self, points, self.ports.pols, fixed)

    @override
    def _combined(self, electric: Any, magnetic: Any, points: Any) -> Any:
        if self.polarization != "helicity":
            return self.with_polarization("helicity")._combined(
                electric, magnetic, points
            )
        signs = 2 * self.ports.pols - 1
        # Derived port waves copy themselves: the constructor reads an SMatrix.
        wave = copy(self)
        wave.coefficients = self.coefficients * self._weights(electric, magnetic, signs)
        return wave.efield(points)

    def with_polarization(self, polarization: str) -> PortWave:
        """Change the polarization convention with a fixed basis matrix."""
        wave = copy(self)
        wave.coefficients = self._backend.change_port_polarization(
            self.coefficients, self.ports.modes, self.polarization, polarization, (0,)
        )
        wave.polarization = polarization
        return wave


def _plane_efield(
    wave: PlaneWave | PortWave,
    points: Any,
    pols: Sequence[int] | NDArray[np.int_],
    fixed_vectors: Callable[[Any], bool],
) -> Any:
    """Electric field of plane-wave amplitudes, one native field per wavevector.

    ``fixed_vectors(vectors)`` decides inside the record whether the
    wavevectors are constants of the native field.
    """
    b = wave._backend
    points = b.array(points)
    shape = tuple(points.shape)
    vectors = wave._vectors()

    def record(v: Any) -> Any:
        value, context = diff.plane_field(
            None,
            np.zeros((1, 3)),
            v,
            pols,
            poltype=wave.polarization,
            fixed_vectors=fixed_vectors(v),
        )
        return value, _select_gradient(context, 2)

    electric = b.apply(record, (1, 3, len(pols)), vectors)[0]
    phases = b.apply(
        diff.plane_phases,
        (int(np.prod(shape[:-1])), len(pols)),
        points.reshape(-1, 3),
        vectors,
    )
    return ((phases * wave.coefficients) @ electric.T).reshape(shape)

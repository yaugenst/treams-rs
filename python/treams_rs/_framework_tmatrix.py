"""Framework T-matrices, clusters and periodic responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np

from . import diff
from ._bases import CylindricalBasis, PlaneWavePorts, SphericalBasis
from ._framework_backend import Backend, Basis, Material, Recorded, with_zero_metadata
from ._framework_smatrix import SMatrix
from ._framework_waves import PlaneWave, PortSet, Wave
from ._results import CrossSections
from ._validation import one_of

__all__ = ["Cluster", "PeriodicResponse", "TMatrix", "solve_periodic"]


def _regular_incident(target: TMatrix | Cluster, incident: PlaneWave | Wave) -> Wave:
    """Regular illumination in the target's basis, positions and polarization."""
    wave = incident.in_basis(target.basis, positions=target.positions, singular=False)
    wave.coefficients = target._backend.require_same_medium(
        wave.coefficients, target, incident
    )
    if wave.polarization != target.polarization:
        wave = wave.with_polarization(target.polarization)
    return wave


class TMatrix:
    """T-matrix of framework arrays: regular incident to singular scattered waves.

    Singular waves are the outgoing ones. The constructors ``tmatrix``,
    ``sphere_tmatrix``, ``multilayer_sphere_tmatrix``, ``cylinder_tmatrix`` and
    ``multilayer_cylinder_tmatrix`` of advect, jax and torch build it, and
    ``Cluster.solve`` returns one; ``PeriodicResponse.response`` holds the
    solved unit cell of ``solve_periodic``. The basis labels are
    fixed; the matrix, k0, the medium and the positions may carry gradients.
    Every method uses ``positions`` and ignores ``basis.positions``.
    """

    array: Any
    """Matrix of shape (modes, modes), a framework array."""
    basis: Basis
    """Fixed mode labels of the rows and columns."""
    k0: Any
    """Vacuum wavenumber."""
    medium: Material
    """Embedding medium."""
    polarization: str
    """Polarization convention, "helicity" or "parity"."""
    positions: Any
    """Expansion centres, shape (positions, 3), a framework array."""

    def __init__(
        self,
        array: Any,
        *,
        basis: Basis,
        k0: Any,
        medium: Material,
        backend: Backend,
        positions: Any = None,
        polarization: str = "helicity",
    ):
        self.array, self.basis, self.k0, self.medium = array, basis, k0, medium
        self._backend, self.polarization = backend, polarization
        self.positions = backend.positions(basis, positions)

    def scatter(self, incident: PlaneWave | Wave) -> Wave:
        """Scatter an incident wave; returns the singular (scattered) wave."""
        wave = _regular_incident(self, incident)
        return Wave(
            self.array @ wave.coefficients,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=self._backend,
            positions=self.positions,
            singular=True,
            polarization=self.polarization,
        )

    def cross_sections(
        self, incident: PlaneWave | Wave, *, flux: Any = 0.5
    ) -> CrossSections[Any]:
        """Named cross sections (area), or cylindrical widths (length).

        The exterior must be nonabsorbing and propagating, and achiral for a
        parity response. For a cylindrical response, ``cross_widths`` makes the
        length units explicit.
        """
        b = self._backend
        wave = _regular_incident(self, incident)
        scattered = self.array @ wave.coefficients
        power = 2 if isinstance(self.basis, SphericalBasis) else 1

        def require_propagating(ks: Any, flux: Any) -> None:
            if (
                np.any(ks.imag != 0)
                or np.any(ks.real == 0)
                or not np.isfinite(flux)
                or flux <= 0
            ):
                raise ValueError(
                    "cross sections require a nonabsorbing propagating exterior and positive flux"
                )
            # The weights are helicity wavenumbers; parity labels have none in
            # a chiral exterior (every position, with or without an overlap).
            if self.polarization == "parity" and ks[0] != ks[1]:
                raise ValueError("parity requires an achiral medium")
            if isinstance(self.basis, CylindricalBasis) and np.any(
                np.abs(self.basis.kz) >= np.abs(ks[self.basis.pol])
            ):
                raise ValueError("cross widths require propagating axial sectors")

        ks = b.guard(require_propagating, b.ks(self.medium, self.k0), b.array(flux))
        weights = ks[self.basis.pol.copy()].real ** (-power)
        weighted = scattered * weights
        if self.positions.shape[0] > 1:
            # Outgoing power of several positions includes their mutual overlap;
            # for one position the regular self-translation is the identity.
            overlap = b.expansion(
                self.basis,
                self.basis,
                self.positions,
                self.positions,
                ks,
                poltype=self.polarization,
            )
            weighted_scattered = overlap @ weighted
        else:
            weighted_scattered = weighted
        factor = 0.5 if power == 2 else 2.0
        return CrossSections(
            b.xp.sum(b.xp.conj(scattered) * weighted_scattered).real * factor / flux,
            -b.xp.sum(b.xp.conj(wave.coefficients) * weighted).real * factor / flux,
        )

    def cross_widths(
        self, incident: PlaneWave | Wave, *, flux: Any = 0.5
    ) -> CrossSections[Any]:
        """Named cylindrical scattering/extinction/absorption widths in length units."""
        if not isinstance(self.basis, CylindricalBasis):
            raise ValueError("cross widths require a cylindrical basis")
        return self.cross_sections(incident, flux=flux)

    def with_polarization(self, polarization: str) -> TMatrix:
        """Convert the response convention with a fixed analytic basis transform."""
        change = self._backend.polarization_change(
            self.basis, self.polarization, polarization
        )
        return TMatrix(
            change @ self.array @ change.T,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=self._backend,
            positions=self.positions,
            polarization=polarization,
        )


class Cluster:
    """Particles at framework-valued positions, before their interaction is solved.

    Build it as ``Cluster(particles, *, positions)``. ``particles`` is a
    sequence of TMatrix objects in global bases of one wave family, with one
    polarization convention, k0 and medium. ``positions`` holds one Cartesian
    position per particle, shape (particles, 3), and may carry gradients.
    The cluster basis labels each mode with its particle index; its
    ``basis.positions`` are zero placeholders, and ``positions`` holds the
    centres.
    """

    def __init__(self, particles: Sequence[TMatrix], *, positions: Any):
        if not particles:
            raise ValueError("cluster needs at least one particle")
        first = particles[0]
        self._backend = first._backend
        self.positions = self._backend.array(positions)
        if self.positions.shape != (len(particles), 3):
            raise ValueError("one Cartesian position is required per particle")
        for particle in particles:
            if (
                type(particle.basis) is not type(first.basis)
                or not particle.basis.isglobal
                or particle.polarization != first.polarization
            ):
                raise ValueError(
                    "cluster particles require one global wave family and polarization convention"
                )
        self.particles = tuple(particles)
        modes = [
            (p, *mode[1:])
            for p, particle in enumerate(particles)
            for mode in particle.basis.modes
        ]
        # The labels carry the particle index; the basis positions are zero
        # placeholders, and the framework array self.positions holds the centres.
        self.basis = type(first.basis)(modes, np.zeros((len(particles), 3)))
        self.k0, self.medium = first.k0, first.medium
        self.polarization = first.polarization

    def _blocks(self) -> list[Any]:
        return [
            self._backend.require_same_medium(p.array, self, p) for p in self.particles
        ]

    def _local(self) -> Any:
        """Block-diagonal local response, one concatenation per particle row."""
        b = self._backend
        total, offset = len(self.basis), 0
        rows: list[Any] = []
        for block in self._blocks():
            size = block.shape[0]
            right = total - offset - size
            parts = [block]
            if offset:
                parts.insert(0, b.array(np.zeros((size, offset)), complex_=True))
            if right:
                parts.append(b.array(np.zeros((size, right)), complex_=True))
            rows.append(b.concat(parts, axis=1))
            offset += size
        return b.concat(rows, axis=0)

    def _coupling(self) -> Any:
        b = self._backend
        return b.expansion(
            self.basis,
            self.basis,
            self.positions,
            self.positions,
            b.ks(self.medium, self.k0),
            poltype=self.polarization,
            singular=True,
        )

    def solve(self) -> TMatrix:
        """The T-matrix of the coupled particles, for any illumination."""
        b = self._backend
        local = self._local()
        result = b.apply(diff.interaction, tuple(local.shape), local, self._coupling())
        return TMatrix(
            result,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=b,
            positions=self.positions,
            polarization=self.polarization,
        )

    def scatter(self, incident: PlaneWave | Wave) -> Wave:
        """The scattered wave of one illumination, without the full coupled T-matrix."""
        b = self._backend
        wave = _regular_incident(self, incident)

        # The native factor keeps the particle blocks separate; no dense local
        # matrix or full interacting response is formed.
        def record(coupling: Any, incident: Any, *blocks: Any) -> Any:
            factor = diff.factor_interaction_blocks(list(blocks), coupling)
            value, context = factor.record(np.asarray(incident, dtype=np.complex128))

            def pullback(g: Any) -> Any:
                local, coupling, incident = context.pullback_blocks(g)
                return (coupling, incident, *local)

            return value, pullback

        coefficients = b.apply(
            record,
            (len(self.basis), 1),
            self._coupling(),
            wave.coefficients[:, None],
            *self._blocks(),
        )[:, 0]
        return Wave(
            coefficients,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=b,
            positions=self.positions,
            singular=True,
            polarization=self.polarization,
        )


class PeriodicResponse:
    """Solved response of a periodic array, ready for conversion to plane-wave ports.

    ``solve_periodic`` returns it. ``response`` is the TMatrix of one unit
    cell with the lattice interaction solved; ``lattice`` and ``kpar`` are
    framework arrays and carry gradients.
    """

    def __init__(self, response: TMatrix, lattice: Any, kpar: Any):
        self.response, self.lattice, self.kpar = response, lattice, kpar

    def _determinant(self) -> Any:
        """Signed area of the two-dimensional unit cell."""
        a = self.lattice
        return a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]

    def _order_ports(self, orders: Any, spherical: bool, alignment: str) -> PortSet:
        """Ports of diffraction orders, whose wavevectors move with the cell."""
        tm, b = self.response, self.response._backend
        labels = np.asarray(orders)
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.round(labels)):
            raise ValueError("diffraction orders must be finite integers")
        if spherical:
            if labels.ndim != 2 or labels.shape[1] != 2:
                raise ValueError("spherical orders require shape (groups,2)")
            a = self.lattice
            # The reciprocal cell is built from framework operations, so the
            # order wavevectors stay differentiable in the lattice vectors.
            determinant = self._determinant()
            reciprocal = (
                b.stack((b.stack((a[1, 1], -a[1, 0])), b.stack((-a[0, 1], a[0, 0]))))
                / determinant
            )
            q = self.kpar + 2 * np.pi * (b.array(labels) @ reciprocal)
        else:
            if labels.ndim != 1:
                raise ValueError("cylindrical orders require a vector")
            axial = np.unique(cast("CylindricalBasis", tm.basis).kz)
            if len(axial) != 1:
                raise NotImplementedError(
                    "order ports require one cylindrical axial sector"
                )
            transverse = self.kpar[0] + 2 * np.pi * b.array(labels) / self.lattice[0, 0]
            q = b.stack((transverse * 0 + axial[0], transverse), axis=1)
        if len(np.unique(labels, axis=0)) != len(labels) or len(labels) == 0:
            raise ValueError("diffraction orders must be nonempty and distinct")
        modes = tuple((i, pol) for i in range(len(labels)) for pol in (1, 0))
        return PortSet(
            basis=None,
            modes=modes,
            alignment=alignment,
            transverse_wavevectors=q,
            fixed_q=False,
        )

    def to_smatrix(
        self,
        basis: PlaneWavePorts | None = None,
        *,
        diffraction_orders: Any = None,
        orders: Any = None,
    ) -> SMatrix:
        """Convert to fixed ports or integer diffraction orders following cell/Bloch changes.

        Spherical orders have shape (groups,2); cylindrical orders are a vector
        for one fixed axial sector. Order ports include both helicities (1,0).
        The returned transverse_wavevectors retain framework derivatives.
        ``orders`` is an alias of ``diffraction_orders``.
        """
        orders = one_of(
            "diffraction_orders", diffraction_orders, "orders", orders, None
        )
        tm, b = self.response, self.response._backend
        spherical = isinstance(tm.basis, SphericalBasis)
        alignment = "xy" if spherical else "zx"
        if (basis is None) == (orders is None):
            raise ValueError("provide a plane basis or integer diffraction orders")
        if orders is None:
            assert basis is not None
            if basis.alignment != alignment:
                raise ValueError("ports must be xy for spheres or zx for cylinders")
            ports = PortSet.from_basis(basis, b)
        else:
            ports = self._order_ports(orders, spherical, alignment)
        modes, fixed_q, pols = ports.modes, ports.fixed_q, ports.pols
        channel_q = ports.transverse_wavevectors[ports.groups]
        measure = b.xp.abs(self._determinant() if spherical else self.lattice[0, 0])

        def record(
            positions: Any, ks: Any, q: Any, measure: Any, a: Any, bloch: Any
        ) -> Recorded:
            diffraction = (
                (q - bloch) @ a.T / (2 * np.pi)
                if spherical
                else (q[:, 1] - bloch[0]) * a[0, 0] / (2 * np.pi)
            )
            if not np.allclose(diffraction, np.round(diffraction), atol=1e-9, rtol=0):
                raise ValueError("plane ports must match lattice diffraction orders")
            dynamic = type(tm.basis)(tm.basis.modes, positions)
            channels = cast(
                "Any",
                diff.spherical_channels if spherical else diff.cylindrical_channels,
            )
            value, context = channels(
                dynamic,
                ks,
                q,
                pols,
                float(measure),
                poltype=tm.polarization,
                fixed_q=fixed_q,
            )

            return value, with_zero_metadata(context.pullback, a, bloch)

        channels = b.apply(
            record,
            (2, 2, len(tm.basis), len(modes)),
            tm.positions,
            b.ks(tm.medium, tm.k0),
            channel_q,
            measure,
            self.lattice,
            self.kpar,
        )
        array = b.apply(
            diff.smatrix_from_array, (2, 2, len(modes), len(modes)), tm.array, channels
        )
        return SMatrix(
            array,
            ports=ports,
            k0=tm.k0,
            media=(tm.medium, tm.medium),
            backend=b,
            polarization=tm.polarization,
        )


def solve_periodic(
    unit_cell: TMatrix | Cluster, *, lattice: Any, kpar: Any, eta: complex = 0
) -> PeriodicResponse:
    """Solve periodic coupling once, then use response.to_smatrix(ports).

    Lattice vectors and the Bloch wavevector remain dynamic framework values.
    """
    if not isinstance(unit_cell, Cluster) and not unit_cell.basis.isglobal:
        raise ValueError(
            "periodic cells require an unsolved Cluster or a global particle response"
        )
    b, basis = unit_cell._backend, unit_cell.basis
    a, q = b.array(lattice), b.array(kpar)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    if q.ndim == 0:
        q = q.reshape(1)
    coupling = b.lattice_expansion(
        basis,
        unit_cell.positions,
        b.ks(unit_cell.medium, unit_cell.k0),
        q,
        a,
        poltype=unit_cell.polarization,
        eta=eta,
    )
    local = unit_cell._local() if isinstance(unit_cell, Cluster) else unit_cell.array
    response = b.apply(diff.interaction, (len(basis), len(basis)), local, coupling)
    return PeriodicResponse(
        TMatrix(
            response,
            basis=basis,
            k0=unit_cell.k0,
            medium=unit_cell.medium,
            backend=b,
            positions=unit_cell.positions,
            polarization=unit_cell.polarization,
        ),
        a,
        q,
    )

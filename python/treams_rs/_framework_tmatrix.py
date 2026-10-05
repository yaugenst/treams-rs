"""Framework T-matrices, clusters and periodic responses."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np

from . import _native, diff
from ._bases import CylindricalBasis, PlaneWavePorts, SphericalBasis
from ._framework_backend import Backend, Basis, Material, Recorded, with_zero_metadata
from ._framework_smatrix import SMatrix
from ._framework_waves import PlaneWave, PortSet, PortWave, Wave
from ._lattice import framework_cell, on_diffraction_orders, periodic_alignment
from ._promotion import promote
from ._records import DerivativeContext
from ._results import CrossSections
from ._saved import native_state
from ._validation import check_particle_positions, one_of

__all__ = ["Cluster", "PeriodicResponse", "TMatrix", "solve_periodic"]


def _regular_incident(target: TMatrix | Cluster, incident: Any) -> Wave:
    """Regular illumination in the target's basis, positions and polarization."""
    incident = promote(incident, target._backend)
    if not isinstance(incident, (PlaneWave, PortWave, Wave)):
        # Raw coefficients already describe the target's local channels. A
        # regular translation would incorrectly add other centres' fields.
        return Wave(
            target._backend.array(incident, complex_=True),
            basis=target.basis,
            k0=target.k0,
            medium=target.medium,
            backend=target._backend,
            positions=target.positions,
            polarization=target.polarization,
        )
    wave = incident.in_basis(target.basis, positions=target.positions, singular=False)
    wave.coefficients = target._backend.require_same_medium(
        wave.coefficients, target, incident
    )
    if wave.polarization != target.polarization:
        wave = wave.with_polarization(target.polarization)
    return wave


class _InteractionFactor:
    """Prepared coupling for differentiable incident-column solves.

    The coupling is reused. Each solve records its own native factorization so
    its derivative remains connected to the local response and the coupling.
    """

    def __init__(self, matrix: TMatrix, coupling: Any):
        self.matrix, self.coupling = matrix, coupling

    def solve(self, incident: Any) -> Any:
        """Scattered coefficients for a vector or a batch of incident columns."""
        b = self.matrix._backend
        incident = b.array(incident, complex_=True)
        if incident.ndim not in (1, 2) or incident.shape[0] != len(self.matrix):
            raise ValueError(
                "incident coefficients must have shape (modes,) or (modes, illuminations)"
            )
        vector = incident.ndim == 1
        columns = incident[:, None] if vector else incident
        result = b.apply(
            diff.illuminate,
            tuple(columns.shape),
            self.matrix.array,
            self.coupling,
            columns,
        )
        return result[:, 0] if vector else result


class _Interaction:
    """Finite multiple scattering of a response in its local multipole basis."""

    def __init__(self, matrix: TMatrix):
        self.matrix = matrix

    def _coupling(self) -> Any:
        tm = self.matrix
        return tm._backend.expansion(
            tm.basis,
            tm.basis,
            tm.positions,
            tm.positions,
            tm.ks,
            poltype=tm.polarization,
            singular=True,
        )

    def __call__(self) -> Any:
        tm = self.matrix
        identity = tm._backend.array(np.eye(len(tm)), complex_=True)
        return identity - tm.array @ self._coupling()

    def solve(self) -> TMatrix:
        """Return the full coupled response ``(I - T*C)^-1 T``."""
        tm = self.matrix
        return tm._with_array(
            tm._backend.apply(diff.interaction, tm.shape, tm.array, self._coupling())
        )

    def factor(self) -> _InteractionFactor:
        """Prepare coupling for repeated differentiable ``solve(incident)`` calls.

        Each call records a native factorization. Unlike the NumPy factor,
        this helper does not retain native LU factors or expose manual records.
        """
        return _InteractionFactor(self.matrix, self._coupling())

    def illuminate(self, incident: Any) -> Any:
        """Scattered coefficients for physical waves or incident-column arrays."""
        incident = promote(incident, self.matrix._backend)
        if isinstance(incident, (PlaneWave, PortWave, Wave)):
            incident = _regular_incident(self.matrix, incident).coefficients
        return self.factor().solve(incident)


class _PeriodicInteraction:
    """Legacy periodic interaction methods backed by the shared native records."""

    def __init__(self, matrix: TMatrix):
        self.matrix = matrix

    def coupling(self, lattice: Any, kpar: Any, *, eta: complex = 0) -> Any:
        """Lattice coupling from every other cell into this cell's regular modes."""
        tm, b = self.matrix, self.matrix._backend
        a, q = framework_cell(b, lattice, kpar, isinstance(tm.basis, SphericalBasis))
        return b.lattice_expansion(
            tm.basis,
            tm.positions,
            tm.ks,
            q,
            a,
            poltype=tm.polarization,
            eta=eta,
        )

    def __call__(self, lattice: Any, kpar: Any, *, eta: complex = 0) -> Any:
        tm = self.matrix
        return tm._backend.array(
            np.eye(len(tm)), complex_=True
        ) - tm.array @ self.coupling(lattice, kpar, eta=eta)

    def solve(self, lattice: Any, kpar: Any, *, eta: complex = 0) -> Any:
        """Effective periodic response array in local multipole channels."""
        tm = self.matrix
        return tm._backend.apply(
            diff.interaction, tm.shape, tm.array, self.coupling(lattice, kpar, eta=eta)
        )

    def factor(
        self, lattice: Any, kpar: Any, *, eta: complex = 0
    ) -> _InteractionFactor:
        """Prepare coupling; each differentiated solve records its factorization."""
        return _InteractionFactor(self.matrix, self.coupling(lattice, kpar, eta=eta))

    def illuminate(
        self, incident: Any, lattice: Any, kpar: Any, *, eta: complex = 0
    ) -> Any:
        """Periodic scattered coefficients for a wave or incident columns."""
        coefficients = _regular_incident(self.matrix, incident).coefficients
        return self.factor(lattice, kpar, eta=eta).solve(coefficients)


class TMatrix:
    """T-matrix of framework arrays: regular incident to singular scattered waves.

    Singular waves are the outgoing ones. The constructors ``tmatrix``,
    ``sphere_tmatrix``, ``multilayer_sphere_tmatrix``, ``cylinder_tmatrix`` and
    ``multilayer_cylinder_tmatrix`` of the adapters build it, and
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
        if polarization == "parity":
            array = backend.require_achiral(array, medium)
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
        weights = b.xp.real(ks[self.basis.pol.copy()]) ** (-power)
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
            b.xp.real(b.xp.sum(b.xp.conj(scattered) * weighted_scattered))
            * factor
            / flux,
            -b.xp.real(b.xp.sum(b.xp.conj(wave.coefficients) * weighted))
            * factor
            / flux,
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
        return TMatrix(
            self._backend.change_polarization(
                self.array, self.basis, self.polarization, polarization, (0, 1)
            ),
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=self._backend,
            positions=self.positions,
            polarization=polarization,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the response matrix."""
        return tuple(self.array.shape)

    @property
    def isglobal(self) -> bool:
        """Whether the response has one expansion centre."""
        return self.basis.isglobal

    @property
    def ks(self) -> Any:
        """Embedding wavenumbers of the two helicities."""
        return self._backend.ks(self.medium, self.k0)

    @property
    def interaction(self) -> _Interaction:
        """Finite multiple scattering within the local multipole basis."""
        return _Interaction(self)

    @property
    def latticeinteraction(self) -> _PeriodicInteraction:
        """Periodic coupling, solve, factor and illuminate methods."""
        return _PeriodicInteraction(self)

    def __len__(self) -> int:
        return len(self.basis)

    def __matmul__(self, incident: Any) -> Any:
        """Scattered coefficients of an incident wave or coefficient array."""
        return self.scatter(incident).coefficients

    def _with_array(self, array: Any) -> TMatrix:
        return TMatrix(
            array,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=self._backend,
            positions=self.positions,
            polarization=self.polarization,
        )

    def rotate(self, phi: Any, theta: Any = 0, psi: Any = 0) -> TMatrix:
        """Rotate local multipole channels by differentiable z-y-z Euler angles."""
        b = self._backend

        rotation = b.apply(
            partial(diff.rotation, destination=self.basis),
            self.shape,
            b.array([phi, theta, psi]),
        )
        return self._with_array(rotation @ self.array @ b.xp.conj(rotation).T)

    def translate(self, r: Any) -> TMatrix:
        """Represent the response about expansion centres shifted by r."""
        b = self._backend
        offset = b.array(r)
        if tuple(offset.shape) != (3,):
            raise ValueError("T-matrix translation requires one Cartesian displacement")
        origin = b.array(np.zeros_like(self.basis.positions))
        mask = b.array(self.basis.pidx[:, None] == self.basis.pidx)

        def shift(displacement: Any) -> Any:
            return (
                b.expansion(
                    self.basis,
                    self.basis,
                    origin + displacement,
                    origin,
                    self.ks,
                    poltype=self.polarization,
                )
                * mask
            )

        return self._with_array(shift(offset) @ self.array @ shift(-offset))

    def in_basis(self, basis: Basis) -> TMatrix:
        """Represent this response in another fixed multipole basis."""
        b = self._backend
        destination = b.positions(basis)
        outgoing = b.expansion(
            basis,
            self.basis,
            destination,
            self.positions,
            self.ks,
            poltype=self.polarization,
        )
        incident = b.expansion(
            self.basis,
            basis,
            self.positions,
            destination,
            self.ks,
            poltype=self.polarization,
        )
        return TMatrix(
            outgoing @ self.array @ incident,
            basis=basis,
            k0=self.k0,
            medium=self.medium,
            backend=b,
            positions=destination,
            polarization=self.polarization,
        )

    def _propagating_ks(self) -> Any:
        def require_propagating(ks: Any) -> None:
            if np.any(ks.imag != 0) or np.any(ks.real == 0):
                raise NotImplementedError(
                    "cross sections require a nonabsorbing propagating embedding medium"
                )

        b = self._backend
        return b.xp.real(b.guard(require_propagating, self.ks))

    @property
    def average_cross_sections(self) -> CrossSections[Any]:
        """Rotationally and polarization-averaged spherical cross sections."""
        if not isinstance(self.basis, SphericalBasis):
            raise ValueError(
                "spherical cross-section averages require a spherical basis"
            )
        if not self.isglobal:
            raise NotImplementedError("expand to a global basis before averaging")
        b = self._backend
        ks = self._propagating_ks()[self.basis.pol.copy()]
        diagonal = self.array[np.arange(len(self)), np.arange(len(self))]
        return CrossSections(
            2 * np.pi * b.xp.sum(b.xp.abs(self.array / ks[:, None]) ** 2),
            -2 * np.pi * b.xp.sum(b.xp.real(diagonal) / ks**2),
        )

    @property
    def average_cross_widths(self) -> CrossSections[Any]:
        """Cylindrical mean over azimuth and propagating axial/polarization channels."""
        if not isinstance(self.basis, CylindricalBasis):
            raise ValueError("cross-width averages require a cylindrical basis")
        if not self.isglobal:
            raise NotImplementedError("expand to a global basis before averaging")
        b = self._backend
        ks = self._propagating_ks()[self.basis.pol.copy()]
        kz = b.array(self.basis.kz)

        def require_channels(ks: Any) -> None:
            if np.any(np.abs(self.basis.kz) == np.abs(ks)):
                raise ValueError("cross widths are undefined at a diffraction cutoff")
            if not np.any(np.abs(self.basis.kz) < np.abs(ks)):
                raise ValueError(
                    "cross-width average requires propagating incident modes"
                )

        # Keep the check in a native callback so JIT checks concrete values too.
        ks = b.xp.real(b.guard(require_channels, b.array(ks, complex_=True)))
        active = b.xp.abs(kz) < b.xp.abs(ks)
        groups = np.unique(np.column_stack((self.basis.kz, self.basis.pol)), axis=0)
        count = b.xp.sum(
            b.xp.abs(b.array(groups[:, 0]))
            < b.xp.abs(self._propagating_ks()[groups[:, 1].astype(int)])
        )
        weights = b.xp.where(active, 4.0 / (count * ks), 0.0)
        diagonal = self.array[np.arange(len(self)), np.arange(len(self))]
        return CrossSections(
            b.xp.sum(b.xp.abs(self.array) ** 2 * weights[:, None] * active[None, :]),
            -b.xp.sum(b.xp.real(diagonal) * weights),
        )

    def _metric(self, kind: str) -> Any:
        if not self.isglobal or self.polarization != "helicity":
            raise NotImplementedError("metric requires a global helicity T-matrix")
        b = self._backend
        ks = self._propagating_ks() if kind == "cd" else b.array([1.0, 1.0])

        record = partial(diff.tmatrix_metric, polarizations=self.basis.pol, metric=kind)
        return b.apply(record, (), self.array, ks, real=True)

    @property
    def circular_dichroism(self) -> Any:
        """Absorption contrast between the two helicities."""
        return self._metric("cd")

    @property
    def duality_breaking(self) -> Any:
        """Fraction of the scattering norm that changes helicity."""
        return self._metric("db")

    @property
    def electromagnetic_chirality(self) -> Any:
        """Normalized electromagnetic chirality from helicity-block singular values."""
        return self._metric("chi")

    # Upstream spellings delegate to the same differentiable operations.
    @property
    def material(self) -> Material:
        """Upstream name of the embedding medium."""
        return self.medium

    @property
    def poltype(self) -> str:
        """Upstream name of the polarization convention."""
        return self.polarization

    cd = circular_dichroism
    db = duality_breaking
    chi = electromagnetic_chirality

    @property
    def xs_ext_avg(self) -> Any:
        """Upstream name of the mean extinction cross section."""
        return self.average_cross_sections.extinction

    @property
    def xs_sca_avg(self) -> Any:
        """Upstream name of the mean scattering cross section."""
        return self.average_cross_sections.scattering

    @property
    def xw_ext_avg(self) -> Any:
        """Upstream name of the mean extinction cross width."""
        return self.average_cross_widths.extinction

    @property
    def xw_sca_avg(self) -> Any:
        """Upstream name of the mean scattering cross width."""
        return self.average_cross_widths.scattering

    def xs(self, illu: Any, flux: Any = 0.5) -> tuple[Any, Any]:
        """Upstream cross sections as (scattering, extinction)."""
        result = self.cross_sections(illu, flux=flux)
        return result.scattering, result.extinction

    def xw(self, illu: Any, flux: Any = 0.5) -> tuple[Any, Any]:
        """Upstream cross widths as (scattering, extinction)."""
        result = self.cross_widths(illu, flux=flux)
        return result.scattering, result.extinction

    def changepoltype(self, poltype: str | None = None) -> TMatrix:
        """Change polarization convention, defaulting to the other convention."""
        target = poltype or (
            "parity" if self.polarization == "helicity" else "helicity"
        )
        return self.with_polarization(target)


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
        from ._dispatch import backend_for

        backend = backend_for(particles, positions)
        if backend is None:
            raise TypeError(
                "framework clusters require a framework particle or positions"
            )
        particles = tuple(promote(particle, backend) for particle in particles)
        first = particles[0]
        self._backend = backend
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
        bases = [particle.basis for particle in self.particles]
        polarization = self.polarization

        def map_context(context: Any, *_primals: Any) -> DerivativeContext:
            def pullback(g: Any) -> Any:
                local, positions, ks = context.pullback(g)
                return (positions, ks, *local)

            def pushforward(positions: Any, ks: Any, *blocks: Any) -> Any:
                return context.pushforward(list(blocks), positions, ks)

            return DerivativeContext(pullback, pushforward, context)

        # The native cluster keeps the particle blocks separate and builds the
        # coupling itself; neither the dense local matrix nor the coupling of
        # _coupling() is formed. It also rejects particles at one position.
        @native_state(
            _native.ParticleClusterContext,
            lambda inputs: (
                [block.shape[0] for block in inputs[2:]],
                isinstance(self.basis, CylindricalBasis),
            ),
            map_context=map_context,
            needs_primals=False,
        )
        def record(positions: Any, ks: Any, *blocks: Any) -> Recorded:
            value, context = diff.particle_cluster(
                list(blocks), positions, ks, bases=bases, poltype=polarization
            )
            return value, map_context(context)

        size = len(self.basis)
        result = b.apply(
            record,
            (size, size),
            self.positions,
            b.ks(self.medium, self.k0),
            *self._blocks(),
        )
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
        return self.factor().scatter(incident)

    def factor(self) -> _ClusterFactor:
        """Prepare coupling and particle blocks for repeated illumination solves.

        Framework inputs retain their derivatives. Each scatter call records
        its native factorization; only constant NumPy clusters retain LU factors.
        """
        return _ClusterFactor(self)

    def _scatter(self, incident: Any, coupling: Any, blocks: Sequence[Any]) -> Wave:
        b = self._backend
        wave = _regular_incident(self, incident)

        def map_context(
            context: Any, _coupling: Any, _incident: Any, positions: Any, *_blocks: Any
        ) -> DerivativeContext:
            def pullback(g: Any) -> Any:
                local, coupling, incident = context.pullback_blocks(g)
                return (coupling, incident, np.zeros_like(positions), *local)

            def pushforward(
                coupling: Any, incident: Any, _positions: Any, *blocks: Any
            ) -> Any:
                return context.pushforward_blocks(list(blocks), coupling, incident)

            return DerivativeContext(pullback, pushforward, context)

        # The native factor keeps the particle blocks separate; no dense local
        # matrix or full interacting response is formed. The positions only
        # pass through for the check that solve() gets from its native cluster.
        @native_state(
            _native.IlluminateContext,
            lambda inputs: (
                [block.shape[0] for block in inputs[3:]],
                inputs[1].shape[1],
            ),
            map_context=map_context,
        )
        def record(coupling: Any, incident: Any, positions: Any, *blocks: Any) -> Any:
            check_particle_positions(
                positions, cylindrical=isinstance(self.basis, CylindricalBasis)
            )
            factor = diff.factor_interaction_blocks(list(blocks), coupling)
            value, context = factor.record(np.asarray(incident, dtype=np.complex128))
            return value, map_context(context, coupling, incident, positions, *blocks)

        vector = wave.coefficients.ndim == 1
        columns = wave.coefficients[:, None] if vector else wave.coefficients
        coefficients = b.apply(
            record,
            tuple(columns.shape),
            coupling,
            columns,
            self.positions,
            *blocks,
        )
        return Wave(
            coefficients[:, 0] if vector else coefficients,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=b,
            positions=self.positions,
            singular=True,
            polarization=self.polarization,
        )


class _ClusterFactor:
    """Reuse a differentiable cluster's prepared coupling and particle blocks."""

    def __init__(self, cluster: Cluster):
        self._cluster = cluster
        self._coupling, self._blocks = cluster._coupling(), cluster._blocks()

    def scatter(self, incident: Any) -> Wave:
        """Solve incident columns with a fresh native differentiation record."""
        return self._cluster._scatter(incident, self._coupling, self._blocks)


class PeriodicWave:
    """Outgoing cell amplitudes with their Bloch-periodic radiation geometry."""

    def __init__(self, local: Wave, lattice: Any, kpar: Any):
        self._local, self.lattice, self.kpar = local, lattice, kpar
        self._backend = local._backend

    @property
    def coefficients(self) -> Any:
        """Outgoing coefficients of the reference cell."""
        return self._local.coefficients

    @property
    def basis(self) -> Basis:
        """Multipole labels of the cell."""
        return self._local.basis

    @property
    def k0(self) -> Any:
        """Vacuum wavenumber."""
        return self._local.k0

    @property
    def medium(self) -> Material:
        """Exterior material."""
        return self._local.medium

    @property
    def polarization(self) -> str:
        """Polarization convention."""
        return self._local.polarization

    def in_basis(self, basis: Basis | PlaneWavePorts, *, kind: str) -> Wave | PortWave:
        """Represent periodic outgoing waves in diffraction ports or multipoles."""
        b, wave = self._backend, self._local
        if isinstance(basis, PlaneWavePorts):
            if kind not in ("up", "down"):
                raise ValueError(
                    "plane radiation requires up/down outgoing plane modes"
                )
            channels = _radiation_channels(
                wave, self.lattice, self.kpar, PortSet.from_basis(basis, b)
            )
            coefficients = channels[1, 0 if kind == "up" else 1].T @ self.coefficients
            return PortWave._from_basis(
                coefficients,
                basis=basis,
                k0=self.k0,
                medium=self.medium,
                backend=b,
                positive=kind == "up",
                polarization=self.polarization,
            )
        if isinstance(basis, CylindricalBasis) and isinstance(
            wave.basis, SphericalBasis
        ):
            if kind != "singular":
                raise ValueError(
                    "periodic spherical-to-cylindrical radiation requires outgoing waves"
                )
            operator = _cylindrical_radiation(wave, basis, self.lattice, self.kpar)
        else:
            if kind != "regular" or type(basis) is not type(wave.basis):
                raise ValueError(
                    "periodic coupling maps outgoing to regular waves of the same family"
                )

            @native_state(
                _native.LatticeExpansionContext,
                lambda inputs: (
                    len(basis),
                    len(wave.basis),
                    inputs[0].shape[0],
                    inputs[1].shape[0],
                    isinstance(basis, CylindricalBasis),
                ),
            )
            def record(
                destination: Any, source: Any, ks: Any, q: Any, a: Any
            ) -> Recorded:
                return diff.lattice_expansion(
                    type(basis)(basis.modes, destination),
                    type(wave.basis)(wave.basis.modes, source),
                    ks,
                    q,
                    a,
                    poltype=self.polarization,
                )

            operator = b.apply(
                record,
                (len(basis), len(wave.basis)),
                b.positions(basis),
                wave.positions,
                b.ks(self.medium, self.k0),
                self.kpar,
                self.lattice,
            )
        return Wave(
            operator @ self.coefficients,
            basis=basis,
            k0=self.k0,
            medium=self.medium,
            backend=b,
            singular=kind == "singular",
            polarization=self.polarization,
        )


def _cylindrical_radiation(
    source: TMatrix | Wave, destination: CylindricalBasis, lattice: Any, kpar: Any
) -> Any:
    """Radiation of a sphere chain with fixed cylindrical axial labels."""
    from ._tmatrix import axis_mask

    b = source._backend
    if not isinstance(source.basis, SphericalBasis):
        raise ValueError("cylindrical conversion requires a spherical unit cell")
    if tuple(lattice.shape) != (1, 1) or tuple(kpar.shape) != (1,):
        raise ValueError("spherical-to-cylindrical radiation requires a 1D z period")

    def map_context(
        context: Any, _destination: Any, _source: Any, _ks: Any, a: Any, q: Any
    ) -> DerivativeContext:
        def pullback(g: Any) -> tuple[Any, ...]:
            gd, gs, gks, _gkz, period = context.pullback(g)
            return (
                gd,
                gs,
                gks,
                np.asarray([[period * np.sign(a[0, 0])]]),
                np.zeros_like(q),
            )

        def pushforward(gd: Any, gs: Any, gks: Any, da: Any, _q: Any) -> Any:
            return context.pushforward(
                gd,
                gs,
                gks,
                np.zeros_like(destination.kz),
                float(da[0, 0] * np.sign(a[0, 0])),
            )

        return DerivativeContext(pullback, pushforward, context)

    @native_state(
        _native.PeriodicToCwContext,
        lambda inputs: (
            len(destination),
            inputs[0].shape[0],
            len(source.basis),
            inputs[1].shape[0],
        ),
        map_context=map_context,
    )
    def record(
        destination_at: Any, source_at: Any, ks: Any, a: Any, q: Any
    ) -> Recorded:
        orders = (destination.kz - q[0]) * a[0, 0] / (2 * np.pi)
        if not on_diffraction_orders(orders):
            raise ValueError(
                "cylindrical axial wavenumbers must match diffraction orders"
            )
        value, context = diff.periodic_to_cw(
            CylindricalBasis(destination.modes, destination_at),
            SphericalBasis(source.basis.modes, source_at),
            ks,
            float(abs(a[0, 0])),
            poltype=source.polarization,
        )
        return value, map_context(context, destination_at, source_at, ks, a, q)

    result = b.apply(
        record,
        (len(destination), len(source.basis)),
        b.positions(destination),
        source.positions,
        b.ks(source.medium, source.k0),
        lattice,
        kpar,
    )
    mask = axis_mask(destination, source.basis)
    return result if mask is None else result * b.array(mask)


def _radiation_channels(
    tm: TMatrix | Wave, lattice: Any, kpar: Any, ports: PortSet
) -> Any:
    """Shared incidence/emission channels of a periodic response or outgoing wave."""
    b = tm._backend
    spherical = isinstance(tm.basis, SphericalBasis)
    dimension = 2 if spherical else 1
    if tuple(lattice.shape) != (dimension, dimension) or tuple(kpar.shape) != (
        dimension,
    ):
        raise ValueError(
            "plane channels require a 2D xy sphere lattice or 1D x cylinder lattice"
        )
    if ports.alignment != ("xy" if spherical else "zx"):
        raise ValueError("ports must be xy for spheres or zx for cylinders")
    measure = b.xp.abs(
        lattice[0, 0] * lattice[1, 1] - lattice[0, 1] * lattice[1, 0]
        if spherical
        else lattice[0, 0]
    )

    def map_context(context: Any, *_primals: Any) -> DerivativeContext:
        return with_zero_metadata(context, *_primals[-2:])

    @native_state(
        _native.SphericalChannelsContext
        if spherical
        else _native.CylindricalChannelsContext,
        lambda inputs: (len(tm.basis), inputs[0].shape[0], inputs[2].shape[0]),
        map_context=map_context,
    )
    def record(
        positions: Any, ks: Any, q: Any, measure: Any, a: Any, bloch: Any
    ) -> Recorded:
        diffraction = (
            (q - bloch) @ a.T / (2 * np.pi)
            if spherical
            else (q[:, 1] - bloch[0]) * a[0, 0] / (2 * np.pi)
        )
        if not on_diffraction_orders(diffraction):
            raise ValueError("plane ports must match lattice diffraction orders")
        channels = cast(
            "Any", diff.spherical_channels if spherical else diff.cylindrical_channels
        )
        value, context = channels(
            type(tm.basis)(tm.basis.modes, positions),
            ks,
            q,
            ports.pols,
            float(measure),
            poltype=tm.polarization,
            fixed_q=ports.fixed_q,
        )
        return value, map_context(context, positions, ks, q, measure, a, bloch)

    return b.apply(
        record,
        (2, 2, len(tm.basis), len(ports.modes)),
        tm.positions,
        b.ks(tm.medium, tm.k0),
        ports.transverse_wavevectors[ports.groups],
        measure,
        lattice,
        kpar,
    )


class PeriodicResponse:
    """Solved response of a periodic array, ready for conversion to plane-wave ports.

    ``solve_periodic`` returns it. ``response`` is the TMatrix of one unit
    cell with the lattice interaction solved; ``lattice`` and ``kpar`` are
    framework arrays and carry gradients.
    """

    def __init__(self, response: TMatrix, lattice: Any, kpar: Any):
        self.response, self.lattice, self.kpar = response, lattice, kpar
        self._backend = response._backend

    @property
    def array(self) -> Any:
        """Solved response in local multipole channels."""
        return self.response.array

    @property
    def basis(self) -> Basis:
        """Local multipole labels of the periodic response."""
        return self.response.basis

    @property
    def k0(self) -> Any:
        """Vacuum wavenumber."""
        return self.response.k0

    @property
    def medium(self) -> Material:
        """Embedding material."""
        return self.response.medium

    @property
    def polarization(self) -> str:
        """Polarization convention."""
        return self.response.polarization

    def scatter(self, incident: Any) -> PeriodicWave:
        """Scatter an illumination whose plane-wave orders match this Bloch vector."""
        b, response = self._backend, self.response
        incident = promote(incident, b)
        wave = _regular_incident(response, incident)
        if isinstance(incident, (PlaneWave, PortWave)):
            vectors = incident._vectors()
            alignment = periodic_alignment(
                len(self.kpar), isinstance(response.basis, SphericalBasis)
            )
            axes = ["xyz".index(axis) for axis in alignment]

            def require_orders(
                values: Any, vectors: Any, amplitudes: Any, a: Any, q: Any
            ) -> None:
                active = amplitudes != 0
                if active.ndim == 2:
                    active = np.any(active, axis=1)
                orders = (vectors[active][:, axes] - q) @ a.T / (2 * np.pi)
                if not on_diffraction_orders(orders):
                    raise ValueError(
                        "plane-wave direction does not match the Bloch wavevector"
                    )

            wave.coefficients = b.guard(
                require_orders,
                wave.coefficients,
                vectors,
                incident.coefficients,
                self.lattice,
                self.kpar,
            )
        scattered = Wave(
            response.array @ wave.coefficients,
            basis=response.basis,
            k0=response.k0,
            medium=response.medium,
            backend=b,
            positions=response.positions,
            singular=True,
            polarization=response.polarization,
        )
        return PeriodicWave(scattered, self.lattice, self.kpar)

    def to_cylindrical(self, basis: CylindricalBasis) -> TMatrix:
        """Represent a spherical z-periodic response in outgoing cylindrical modes."""
        from ._tmatrix import axis_mask

        tm, b = self.response, self._backend
        outgoing = _cylindrical_radiation(tm, basis, self.lattice, self.kpar)
        incoming = b.expansion(
            tm.basis,
            basis,
            tm.positions,
            b.positions(basis),
            tm.ks,
            poltype=tm.polarization,
        )
        mask = axis_mask(basis, cast("SphericalBasis", tm.basis))
        if mask is not None:
            incoming = incoming * b.array(mask.T)
        return TMatrix(
            outgoing @ tm.array @ incoming,
            basis=basis,
            k0=tm.k0,
            medium=tm.medium,
            backend=b,
            polarization=tm.polarization,
        )

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
        dimension = 2 if spherical else 1
        if tuple(self.lattice.shape) != (dimension, dimension) or tuple(
            self.kpar.shape
        ) != (dimension,):
            raise ValueError(
                "plane ports require a (2, 2) xy lattice and a (2,) Bloch vector "
                "for spheres, or a (1, 1) period and a (1,) Bloch vector for "
                "cylinders"
            )
        if (basis is None) == (orders is None):
            raise ValueError("provide a plane basis or integer diffraction orders")
        if orders is None:
            assert basis is not None
            if basis.alignment != alignment:
                raise ValueError("ports must be xy for spheres or zx for cylinders")
            ports = PortSet.from_basis(basis, b)
        else:
            ports = self._order_ports(orders, spherical, alignment)
        modes = ports.modes
        channels = _radiation_channels(tm, self.lattice, self.kpar, ports)
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
    from ._dispatch import backend_for

    backend = backend_for(unit_cell, lattice, kpar)
    if backend is None:
        raise TypeError("framework periodic solves require a framework input")
    unit_cell = promote(unit_cell, backend)
    if not isinstance(unit_cell, Cluster) and not unit_cell.basis.isglobal:
        raise ValueError(
            "periodic cells require an unsolved Cluster or a global particle response"
        )
    b, basis = unit_cell._backend, unit_cell.basis
    a, q = framework_cell(b, lattice, kpar, isinstance(basis, SphericalBasis))
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

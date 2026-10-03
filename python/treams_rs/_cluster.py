"""Finite multiple-scattering clusters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from . import diff
from ._tmatrix import CylindricalTMatrix, TMatrix, interaction_coupling, solve_columns
from ._waves import check_compatible

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from . import _native
    from ._bases import CylindricalBasis, SphericalBasis
    from ._material import Material
    from ._waves import PlaneWave, Wave

__all__ = ["Cluster", "DelegatesToLocal", "ScatteringFactor", "assemble"]


def assemble(
    particles: Sequence[TMatrix] | Sequence[CylindricalTMatrix], positions: ArrayLike
) -> tuple[
    TMatrix | CylindricalTMatrix,
    tuple[NDArray[np.complex128], ...],
    tuple[SphericalBasis | CylindricalBasis, ...],
]:
    """Block-diagonal local matrix of uncoupled particles, with their blocks and bases.

    Internal helper of Cluster. Particles are global matrices of one wave
    family that share k0, medium and polarization; mode counts may differ.
    Positions, shape (n, 3), set each particle's position. The local
    matrix has dimension equal to the sum of the particle dimensions; Cluster
    solves its multiple scattering from the blocks.
    """
    if not particles:
        raise ValueError("cluster requires at least one particle")
    cls = TMatrix if isinstance(particles[0], TMatrix) else CylindricalTMatrix
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (len(particles), 3):
        raise ValueError("one Cartesian position required per T-matrix")
    first = particles[0]
    dimension = sum(len(tm) for tm in particles)
    value = np.zeros((dimension, dimension), dtype=np.complex128)
    modes: list[tuple[int, float, int, int]] = []
    offset = 0
    for particle, tm in enumerate(particles):
        if not isinstance(tm.basis, cls._basis_type) or not tm.isglobal:
            raise ValueError("cluster requires global matrices of one wave family")
        check_compatible(
            (tm.k0, tm.medium, tm.polarization),
            (first.k0, first.medium, first.polarization),
            "cluster particles",
        )
        end = offset + len(tm)
        value[offset:end, offset:end] = tm.array
        modes.extend(
            (particle, degree, order, pol) for _, degree, order, pol in tm.basis
        )
        offset = end
    # Adopting the block-diagonal array avoids a second dense copy, which
    # would also fault in every page of its mostly untouched zeros.
    local = cls._adopt(
        value,
        k0=first.k0,
        material=first.medium,
        poltype=first.polarization,
        basis=cast("Any", type(first.basis)(modes, positions)),
    )
    return (
        local,
        tuple(tm.array for tm in particles),
        tuple(tm.basis for tm in particles),
    )


class ScatteringFactor:
    """Reusable dense factor for requested finite-cluster illuminations."""

    def __init__(
        self, local: TMatrix | CylindricalTMatrix, factor: _native.InteractionFactor
    ):
        self._local, self._factor = local, factor

    def scatter(self, incident: ArrayLike | PlaneWave | Wave) -> Wave:
        """Solve one source or a (modes, illuminations) batch with the stored LU factors."""
        return self._local._outgoing(
            solve_columns(self._factor, self._local._incident(incident))
        )


class DelegatesToLocal[L: TMatrix | CylindricalTMatrix | Wave]:
    """Exposes basis, k0, medium and polarization of the wrapped local response or wave."""

    # The wrapped object; solve_periodic reads it from a Cluster.
    _local: L

    @property
    def basis(self) -> SphericalBasis | CylindricalBasis:
        """Local multipole modes and their positions."""
        return cast("SphericalBasis | CylindricalBasis", self._local.basis)

    @property
    def k0(self) -> float:
        """Vacuum angular wavenumber."""
        return self._local.k0

    @property
    def medium(self) -> Material:
        """Homogeneous exterior (embedding) medium."""
        return self._local.medium

    @property
    def polarization(self) -> str:
        """Polarization channel convention: helicity or parity."""
        return self._local.polarization


class Cluster(DelegatesToLocal[TMatrix | CylindricalTMatrix]):
    """Unsolved finite collection of particle responses at Cartesian positions.

    Construction only assembles particles. Solve creates a full coupled response;
    scatter computes requested illuminations. No numerical matrix multiplication
    is provided on an unsolved cluster.

    Complete finite-scattering workflow::

        import treams_rs as tr

        particles = [tr.sphere_tmatrix(k0=2., lmax=2, radius=r, material=3.)
                     for r in (0.12, 0.16)]
        system = tr.Cluster(particles, positions=[[0, 0, 0], [0.6, 0, 0]])
        incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=2.)
        response = system.solve()
        cross = response.cross_sections(incident)
        points = [[0.2, 0.1, 0.8]]
        total_e = incident.efield(points) + response.scatter(incident).efield(points)
        assert total_e.shape == (1, 3)
        assert abs(cross.absorption) < 1e-12  # lossless materials

    For fields alone, use ``system.scatter(incident).efield(points)`` and add
    the incident field. For geometry/material gradients choose an explicit
    framework namespace and construct changing quantities inside the objective.
    """

    def __init__(
        self,
        particles: Sequence[TMatrix] | Sequence[CylindricalTMatrix],
        *,
        positions: ArrayLike,
    ):
        self._local, self._blocks, self._bases = assemble(particles, positions)

    def solve(self) -> TMatrix | CylindricalTMatrix:
        """Compute the complete coupled finite response, including multiple scattering."""
        local = self._local
        result, _ = diff.particle_cluster(
            self._blocks,
            local.basis.positions,
            local.ks,
            bases=self._bases,
            poltype=local.polarization,
        )
        return local._adopt(
            result,
            k0=local.k0,
            basis=cast("Any", local.basis),
            material=local.medium,
            poltype=local.polarization,
        )

    def factor(self) -> ScatteringFactor:
        """Factor once for repeated requested-source solves."""
        coupling = interaction_coupling(self._local)
        return ScatteringFactor(
            self._local, diff.factor_interaction_blocks(self._blocks, coupling)
        )

    def scatter(self, incident: ArrayLike | PlaneWave | Wave) -> Wave:
        """Compute only the requested outgoing waves, including particle interactions."""
        return self.factor().scatter(incident)

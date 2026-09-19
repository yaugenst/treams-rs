"""Physical construction and solved responses, independent of array conventions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from . import diff
from ._core import PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector
from ._lattice import WaveVector
from ._operators import _periodic_channels, expandlattice
from ._plane import PlaneWave
from ._smatrix import SMatrices
from ._source import MultipoleWave
from ._tmatrix import TMatrix, TMatrixC
from .lattice import _geometry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from . import _native
    from ._core import (
        CylindricalWaveBasis,
        Material,
        MaterialLike,
        SphericalWaveBasis,
    )


def sphere_tmatrix(
    *,
    k0: float,
    lmax: int,
    radius: float,
    material: MaterialLike,
    medium: MaterialLike = 1,
    polarization: str = "helicity",
) -> TMatrix:
    """Homogeneous sphere response. Radius and 1/k0 use the same length unit."""
    return TMatrix.sphere(lmax, k0, radius, [material, medium], polarization)


def multilayer_sphere_tmatrix(
    *,
    k0: float,
    lmax: int,
    radii: ArrayLike,
    materials: Sequence[MaterialLike],
    medium: MaterialLike = 1,
    polarization: str = "helicity",
) -> TMatrix:
    """Concentric layers ordered inside-out; one material per positive increasing radius."""
    return TMatrix.sphere(lmax, k0, radii, [*materials, medium], polarization)


def cylinder_tmatrix(
    *,
    k0: float,
    kz: ArrayLike,
    mmax: int,
    radius: float,
    material: MaterialLike,
    medium: MaterialLike = 1,
    polarization: str = "helicity",
) -> TMatrixC:
    """Homogeneous infinite z cylinder; kz is one or more real axial wavenumbers."""
    return TMatrixC.cylinder(kz, mmax, k0, radius, [material, medium], polarization)


def multilayer_cylinder_tmatrix(
    *,
    k0: float,
    kz: ArrayLike,
    mmax: int,
    radii: ArrayLike,
    materials: Sequence[MaterialLike],
    medium: MaterialLike = 1,
    polarization: str = "helicity",
) -> TMatrixC:
    """Concentric infinite cylinders; materials run outward, excluding the medium."""
    return TMatrixC.cylinder(kz, mmax, k0, radii, [*materials, medium], polarization)


class ScatteringFactor:
    """Reusable dense factor for requested finite-cluster illuminations."""

    def __init__(self, local: TMatrix | TMatrixC, factor: _native.InteractionFactor):
        self._local, self._factor = local, factor

    def scatter(self, incident: ArrayLike | PlaneWave | MultipoleWave) -> MultipoleWave:
        """Solve one source or a (modes, illuminations) batch using the retained LU."""
        values = self._local._incident(incident)
        vector = values.ndim == 1
        result = self._factor.solve(values[:, None] if vector else values)
        return MultipoleWave(
            result[:, 0] if vector else result,
            basis=self._local.basis,
            k0=self._local.k0,
            material=self._local.medium,
            modetype="singular",
            poltype=self._local.polarization,
        )


class Cluster:
    """Unsolved finite collection of particle responses at Cartesian positions.

    Construction only assembles particles. Solve creates a full coupled response;
    scatter computes requested illuminations. No numerical matrix multiplication
    is provided on an unsolved cluster.
    """

    def __init__(
        self, particles: Sequence[TMatrix] | Sequence[TMatrixC], *, positions: ArrayLike
    ):
        if not particles:
            raise ValueError("cluster requires at least one particle")
        self._local = (
            TMatrix._assemble(cast("Sequence[TMatrix]", particles), positions)
            if isinstance(particles[0], TMatrix)
            else TMatrixC._assemble(cast("Sequence[TMatrixC]", particles), positions)
        )

    @property
    def basis(self) -> SphericalWaveBasis | CylindricalWaveBasis:
        """Local multipole modes and particle positions."""
        return self._local.basis

    @property
    def k0(self) -> float:
        """Vacuum angular wavenumber."""
        return self._local.k0

    @property
    def medium(self) -> Material:
        """Common exterior medium."""
        return self._local.medium

    @property
    def polarization(self) -> str:
        """Common polarization channel convention."""
        return self._local.polarization

    def solve(self) -> TMatrix | TMatrixC:
        """Compute the complete coupled finite response, including multiple scattering."""
        return self._local.interaction.solve()

    def factor(self) -> ScatteringFactor:
        """Factor once for repeated requested-source solves."""
        return ScatteringFactor(self._local, self._local.interaction.factor())

    def scatter(self, incident: ArrayLike | PlaneWave | MultipoleWave) -> MultipoleWave:
        """Compute only the requested outgoing waves, including particle interactions."""
        return self.factor().scatter(incident)


class PeriodicWave:
    """Solved local-cell outgoing coefficients with Bloch-periodic geometry.

    Local coefficients alone are not a finite radiating wave. Choose a plane-port
    or cylindrical representation with in_basis before evaluating fields.
    """

    def __init__(self, local: MultipoleWave, lattice: ArrayLike, kpar: ArrayLike):
        self._local = local
        self.lattice = np.array(lattice, dtype=float, copy=True)
        self.kpar = np.array(kpar, dtype=float, copy=True)
        self.lattice.flags.writeable = self.kpar.flags.writeable = False

    @property
    def coefficients(self) -> NDArray[np.complex128]:
        """Read-only outgoing coefficients of the reference cell."""
        return self._local.coefficients

    @property
    def basis(self) -> SphericalWaveBasis | CylindricalWaveBasis:
        """Local multipole basis of the reference cell."""
        return cast("SphericalWaveBasis | CylindricalWaveBasis", self._local.basis)

    @property
    def k0(self) -> float:
        """Vacuum angular wavenumber."""
        return self._local.k0

    @property
    def medium(self) -> Material:
        """Propagation medium."""
        return self._local.medium

    @property
    def polarization(self) -> str:
        """Polarization channel convention."""
        return self._local.polarization

    def in_basis(
        self,
        basis: PlaneWaveBasisByComp | CylindricalWaveBasis | SphericalWaveBasis,
        *,
        kind: str,
    ) -> MultipoleWave:
        """Represent all cells in plane ports, cylindrical waves or local regular modes.

        Field evaluation is valid in the exterior region of that expansion.
        Diffraction orders must match the lattice and Bloch vector.
        """
        kind = "singular" if kind == "outgoing" else kind
        conversion = expandlattice(
            self.lattice,
            self.kpar,
            basis=(basis, self.basis),
            k0=self.k0,
            material=self.medium,
            poltype=self.polarization,
            modetype=kind,
        )
        return MultipoleWave(
            conversion @ self.coefficients,
            basis=basis,
            k0=self.k0,
            material=self.medium,
            poltype=self.polarization,
            modetype=kind,
        )


class PeriodicResponse:
    """Solved Bloch-periodic response, distinct from an isolated-particle T-matrix.

    Conversion methods use the solved array without another interaction solve.
    Isolated-particle cross-section formulas do not apply to this response.
    """

    def __init__(
        self,
        local: TMatrix | TMatrixC,
        array: ArrayLike,
        *,
        lattice: ArrayLike,
        kpar: ArrayLike,
    ):
        self._local = local
        self.array = np.array(array, dtype=complex, copy=True)
        if self.array.shape != local.shape or not np.isfinite(self.array).all():
            raise ValueError(
                "periodic response must match the local finite matrix shape"
            )
        self.lattice = np.array(lattice, dtype=float, copy=True)
        self.kpar = np.array(kpar, dtype=float, copy=True)
        self.array.flags.writeable = False
        self.lattice.flags.writeable = self.kpar.flags.writeable = False

    @property
    def basis(self) -> SphericalWaveBasis | CylindricalWaveBasis:
        """Local multipole basis."""
        return self._local.basis

    @property
    def k0(self) -> float:
        """Vacuum angular wavenumber."""
        return self._local.k0

    @property
    def medium(self) -> Material:
        """Homogeneous exterior medium."""
        return self._local.medium

    @property
    def polarization(self) -> str:
        """Polarization channel convention."""
        return self._local.polarization

    def scatter(self, incident: ArrayLike | PlaneWave | MultipoleWave) -> PeriodicWave:
        """Apply this solved response to reference-cell incident coefficients.

        Supplied coefficients define the Bloch-periodic excitation via this
        response's kpar. A plane wave must match that Bloch vector modulo the
        reciprocal lattice; use a diffraction basis to construct such sources.
        """
        vectors = None
        if isinstance(incident, PlaneWave):
            vectors = incident.kvecs[incident.amplitudes != 0]
        elif isinstance(incident, MultipoleWave) and isinstance(
            incident.basis, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)
        ):
            active = incident.coefficients != 0
            if active.ndim == 2:
                active = np.any(active, axis=1)
            vectors = np.column_stack(
                incident.basis.kvecs(incident.k0, incident.medium, incident.modetype)
            )[active]
        if vectors is not None:
            dimension = len(self.kpar)
            axes = (
                ([2] if isinstance(self._local, TMatrix) else [0])
                if dimension == 1
                else list(range(dimension))
            )
            vectors = vectors[:, axes]
            orders = (vectors - self.kpar) @ self.lattice.T / (2 * np.pi)
            if not np.allclose(orders, np.round(orders.real), atol=1e-10, rtol=0):
                raise ValueError(
                    "plane-wave direction does not match the Bloch wavevector"
                )
        values = self._local._incident(incident)
        local = MultipoleWave(
            self.array @ values,
            basis=self.basis,
            k0=self.k0,
            material=self.medium,
            poltype=self.polarization,
            modetype="singular",
        )
        return PeriodicWave(local, self.lattice, self.kpar)

    def to_smatrix(self, basis: PlaneWaveBasisByComp) -> SMatrices:
        """Convert the solved response to matching up/down diffraction ports."""
        channels = _periodic_channels(
            self.basis,
            basis,
            self.medium.ks(self.k0),
            self.lattice,
            self.kpar,
            self.polarization,
        )
        array, _ = diff.smatrix_from_array(self.array, channels)
        return SMatrices(
            array,
            basis=basis,
            k0=self.k0,
            material=self.medium,
            poltype=self.polarization,
        )

    def to_cylindrical(self, basis: CylindricalWaveBasis) -> TMatrixC:
        """Convert a solved spherical z-periodic response to outgoing cylindrical modes."""
        if not isinstance(self._local, TMatrix):
            raise ValueError("cylindrical conversion requires a spherical unit cell")
        outgoing = expandlattice(
            self.lattice,
            self.kpar,
            basis=(basis, self.basis),
            k0=self.k0,
            material=self.medium,
            poltype=self.polarization,
        )
        incoming = diff.expansion(
            self.basis, basis, self.medium.ks(self.k0), poltype=self.polarization
        )[0]
        return TMatrixC(
            outgoing @ self.array @ incoming,
            basis=basis,
            k0=self.k0,
            material=self.medium,
            poltype=self.polarization,
        )


def solve_periodic(
    unit_cell: TMatrix | TMatrixC | Cluster,
    *,
    lattice: ArrayLike,
    kpar: ArrayLike,
    eta: complex = 0,
) -> PeriodicResponse:
    """Solve cell interactions once at a fixed lattice and Bloch wavevector.

    The unit cell contains isolated particles (a single T-matrix or an unsolved
    Cluster). Do not pass an already coupled cluster response. Lattice lengths,
    basis positions and inverse k0 use the same units. Eta controls the Ewald
    splitting; zero asks the native kernel to choose it.
    """
    if not isinstance(unit_cell, Cluster) and not unit_cell.isglobal:
        raise ValueError(
            "periodic cells require an unsolved Cluster or a single-origin response; "
            "a solved local cluster would count particle interactions twice"
        )
    local = unit_cell._local if isinstance(unit_cell, Cluster) else unit_cell
    components = np.atleast_1d(kpar)
    dimension = (
        int(np.count_nonzero(~np.isnan(components)))
        if isinstance(kpar, WaveVector)
        else components.size
    )
    dimension = min(
        dimension,
        np.atleast_2d(lattice).shape[0] if np.ndim(lattice) != 1 else np.size(lattice),
    )
    vectors, bloch = _geometry(dimension, lattice, kpar, isinstance(local, TMatrix))
    response = local.latticeinteraction.solve(vectors, bloch, eta=eta)
    return PeriodicResponse(local, response, lattice=vectors, kpar=bloch)


def interface(
    *,
    basis: PlaneWaveBasisByComp,
    k0: float,
    negative_medium: MaterialLike,
    positive_medium: MaterialLike,
    polarization: str = "helicity",
) -> SMatrices:
    """Planar interface from negative to positive side of the basis normal."""
    return SMatrices.interface(
        basis, k0, [negative_medium, positive_medium], polarization
    )


def slab(
    *,
    basis: PlaneWaveBasisByComp,
    k0: float,
    thickness: float,
    material: MaterialLike,
    negative_medium: MaterialLike = 1,
    positive_medium: MaterialLike = 1,
    polarization: str = "helicity",
) -> SMatrices:
    """One homogeneous layer between explicitly named exterior media."""
    return multilayer_slab(
        basis=basis,
        k0=k0,
        thicknesses=[thickness],
        materials=[material],
        negative_medium=negative_medium,
        positive_medium=positive_medium,
        polarization=polarization,
    )


def multilayer_slab(
    *,
    basis: PlaneWaveBasisByComp,
    k0: float,
    thicknesses: ArrayLike,
    materials: Sequence[MaterialLike],
    negative_medium: MaterialLike = 1,
    positive_medium: MaterialLike = 1,
    polarization: str = "helicity",
) -> SMatrices:
    """Layers ordered along the positive normal, with one material per thickness."""
    return SMatrices.slab(
        thicknesses,
        basis,
        k0,
        [negative_medium, *materials, positive_medium],
        polarization,
    )


def propagation(
    *,
    distance: ArrayLike,
    basis: PlaneWaveBasisByComp,
    k0: float,
    medium: MaterialLike = 1,
    polarization: str = "helicity",
) -> SMatrices:
    """Homogeneous propagation by a normal distance or Cartesian displacement."""
    return SMatrices.propagation(distance, basis, k0, medium, polarization)


def stack(layers: Sequence[SMatrices]) -> SMatrices:
    """Cascade layers in order from negative to positive side of their normal."""
    return SMatrices.stack(layers)

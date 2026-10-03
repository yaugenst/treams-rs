"""Bloch-periodic arrays: solve_periodic and its responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ._bases import CylindricalBasis, PlaneWaveBasis, PlaneWavePorts, SphericalBasis
from ._cluster import Cluster, DelegatesToLocal
from ._dispatch import autodiff_method, backend_for, namespace
from ._lattice import on_diffraction_orders, periodic_alignment, periodic_geometry
from ._operators import expandlattice
from ._smatrix import SMatrix
from ._tmatrix import CylindricalTMatrix, TMatrix, axis_mask
from ._validation import check_kind, frozen, one_of
from ._waves import PlaneWave, Wave

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["PeriodicResponse", "PeriodicWave", "solve_periodic"]


class PeriodicWave(DelegatesToLocal[Wave]):
    """Solved local-cell outgoing coefficients with Bloch-periodic geometry.

    Local coefficients alone are not a finite radiating wave. Choose a plane-port
    or cylindrical representation with in_basis before evaluating fields.
    """

    def __new__(cls, local: Any = None, lattice: Any = None, kpar: Any = None) -> Any:
        backend = backend_for(local, lattice, kpar)
        if backend is None:
            return super().__new__(cls)
        from ._framework_tmatrix import PeriodicWave as FrameworkPeriodicWave
        from ._promotion import promote

        return FrameworkPeriodicWave(
            promote(local, backend), backend.array(lattice), backend.array(kpar)
        )

    def __init__(self, local: Wave, lattice: ArrayLike, kpar: ArrayLike):
        self._local = local
        self.lattice = frozen(lattice, np.float64)
        self.kpar = frozen(kpar, np.float64)

    @property
    def coefficients(self) -> NDArray[np.complex128]:
        """Read-only outgoing coefficients of the reference cell."""
        return self._local.coefficients

    def in_basis(
        self,
        basis: PlaneWavePorts | CylindricalBasis | SphericalBasis,
        *,
        kind: str,
    ) -> Wave:
        """Represent all cells in plane ports, cylindrical waves or local regular modes.

        ``kind`` has no default because a periodic wave has no default radial
        kind: choose "up" or "down" for plane ports and "regular" or
        "singular" for multipole bases. Field evaluation is valid in the
        exterior region of that expansion.
        Diffraction orders must match the lattice and Bloch vector. A
        cylindrical basis with several axes needs one axis per particle
        position, as in ``to_cylindrical``.
        """
        kind = check_kind(kind)
        mask = None
        if isinstance(basis, CylindricalBasis) and isinstance(
            self.basis, SphericalBasis
        ):
            mask = axis_mask(basis, self.basis)
        conversion = expandlattice(
            self.lattice,
            self.kpar,
            basis=(basis, self.basis),
            k0=self.k0,
            material=self.medium,
            poltype=self.polarization,
            modetype=kind,
        )
        if mask is not None:
            # Each axis already holds the whole field, so it pairs only with its own particle.
            conversion = conversion * mask
        return Wave(
            conversion @ self.coefficients,
            basis=basis,
            k0=self.k0,
            medium=self.medium,
            polarization=self.polarization,
            kind=kind,
        )


class PeriodicResponse(DelegatesToLocal[TMatrix | CylindricalTMatrix]):
    """Solved Bloch-periodic response, distinct from an isolated-particle T-matrix.

    Conversion methods use the solved array without another interaction solve.
    Isolated-particle cross-section formulas do not apply to this response.
    """

    def __new__(cls, local: Any = None, array: Any = None, **kwargs: Any) -> Any:
        backend = backend_for(local, array, kwargs)
        if backend is None:
            return super().__new__(cls)
        from ._framework_tmatrix import PeriodicResponse as FrameworkPeriodicResponse
        from ._promotion import promote

        response = promote(local, backend)
        values = backend.array(array, complex_=True)
        if values.shape != response.shape:
            raise ValueError(
                "periodic response must match the local finite matrix shape"
            )
        return FrameworkPeriodicResponse(
            response._with_array(values),
            backend.array(kwargs["lattice"]),
            backend.array(kwargs["kpar"]),
        )

    def __init__(
        self,
        local: TMatrix | CylindricalTMatrix,
        array: ArrayLike,
        *,
        lattice: ArrayLike,
        kpar: ArrayLike,
    ):
        self._local = local
        self.array = frozen(array, np.complex128)
        if self.array.shape != local.shape or not np.isfinite(self.array).all():
            raise ValueError(
                "periodic response must match the local finite matrix shape"
            )
        self.lattice = frozen(lattice, np.float64)
        self.kpar = frozen(kpar, np.float64)

    @autodiff_method
    def scatter(self, incident: ArrayLike | PlaneWave | Wave) -> PeriodicWave:
        """Apply this solved response to reference-cell incident coefficients.

        Supplied coefficients define the Bloch-periodic excitation via this
        response's kpar. A plane wave must match that Bloch vector modulo the
        reciprocal lattice; use a diffraction basis to construct such sources.
        """
        vectors = None
        if isinstance(incident, PlaneWave):
            vectors = incident.kvecs[incident.coefficients != 0]
        elif isinstance(incident, Wave) and isinstance(
            incident.basis, (PlaneWavePorts, PlaneWaveBasis)
        ):
            active = incident.coefficients != 0
            if active.ndim == 2:
                active = np.any(active, axis=1)
            vectors = np.column_stack(
                incident.basis.kvecs(incident.k0, incident.medium, incident.kind)
            )[active]
        if vectors is not None:
            alignment = periodic_alignment(
                len(self.kpar), isinstance(self._local, TMatrix)
            )
            vectors = vectors[:, ["xyz".index(axis) for axis in alignment]]
            orders = (vectors - self.kpar) @ self.lattice.T / (2 * np.pi)
            if not on_diffraction_orders(orders):
                raise ValueError(
                    "plane-wave direction does not match the Bloch wavevector"
                )
        local = self._local._outgoing(self.array @ self._local._incident(incident))
        return PeriodicWave(local, self.lattice, self.kpar)

    def to_smatrix(
        self,
        basis: PlaneWavePorts | None = None,
        *,
        diffraction_orders: ArrayLike | None = None,
        orders: ArrayLike | None = None,
    ) -> SMatrix:
        """Convert to a fixed plane basis or explicit integer diffraction orders.

        Spherical arrays take orders with shape (groups, 2), cylindrical arrays
        a vector of orders within one axial sector. Each order includes both
        helicities. ``orders`` is an alias of ``diffraction_orders``.
        """
        orders = one_of(
            "diffraction_orders", diffraction_orders, "orders", orders, None
        )
        if (basis is None) == (orders is None):
            raise ValueError("provide a plane basis or integer diffraction orders")
        if orders is not None:
            labels = np.asarray(orders)
            if not np.all(np.isfinite(labels)) or not np.all(
                labels == np.round(labels)
            ):
                raise ValueError("diffraction orders must be finite integers")
            spherical = isinstance(self.basis, SphericalBasis)
            if spherical:
                if labels.ndim != 2 or labels.shape[1] != 2:
                    raise ValueError("spherical orders require shape (groups,2)")
                if self.lattice.shape != (2, 2) or self.kpar.shape != (2,):
                    raise ValueError("spherical plane ports require a 2D xy lattice")
                q = self.kpar + 2 * np.pi * labels @ np.linalg.inv(self.lattice).T
            else:
                if labels.ndim != 1:
                    raise ValueError("cylindrical orders require a vector")
                axial = np.unique(cast("CylindricalBasis", self.basis).kz)
                if len(axial) != 1:
                    raise NotImplementedError(
                        "order ports require one cylindrical axial sector"
                    )
                if self.lattice.shape != (1, 1) or self.kpar.shape != (1,):
                    raise ValueError("cylindrical plane ports require a 1D x lattice")
                transverse = self.kpar[0] + 2 * np.pi * labels / self.lattice[0, 0]
                q = np.column_stack((np.full_like(transverse, axial[0]), transverse))
            if len(labels) == 0 or len(np.unique(labels, axis=0)) != len(labels):
                raise ValueError("diffraction orders must be nonempty and distinct")
            basis = PlaneWavePorts(
                [(*vector, pol) for vector in q for pol in (1, 0)],
                alignment="xy" if spherical else "zx",
            )
        assert basis is not None
        return SMatrix._from_response(
            self._local, basis, self.lattice, self.kpar, lambda: self.array
        )

    def to_cylindrical(self, basis: CylindricalBasis) -> CylindricalTMatrix:
        """Convert a solved spherical z-periodic response to outgoing cylindrical modes.

        A basis with several axes needs one axis per particle position; axis i
        then collects the waves of particle i only, as in treams.
        """
        return CylindricalTMatrix._from_response(
            self._local, basis, self.lattice, self.kpar, lambda: self.array
        )


def solve_periodic(
    unit_cell: TMatrix | CylindricalTMatrix | Cluster,
    *,
    lattice: ArrayLike,
    kpar: ArrayLike,
    eta: complex = 0,
) -> PeriodicResponse:
    """Solve cell interactions once at a fixed lattice and Bloch wavevector.

    The unit cell contains isolated particles (a single T-matrix or an unsolved
    Cluster). Do not pass an already coupled cluster response. Lattice lengths,
    basis positions and inverse k0 use the same units. The lattice sums use
    the Ewald method, which splits each sum into a real-space and a
    reciprocal-space series; eta sets that split, and 0 picks it
    automatically.

    A subwavelength square array, optionally combined with planar layers::

        import treams_rs as tr

        k0 = 2.0
        cell = tr.Lattice.square(0.9)
        particle = tr.sphere_tmatrix(k0=k0, lmax=2, radius=0.12, material=3.)
        response = tr.solve_periodic(particle, lattice=cell, kpar=[0, 0])
        ports = tr.PlaneWavePorts.default([0, 0])  # zeroth order, both helicities
        array = response.to_smatrix(ports)
        layer = tr.slab(k0=k0, basis=ports, thickness=0.2, material=2.)
        gap = tr.propagation(k0=k0, basis=ports, distance=0.3)
        network = tr.stack([layer, gap, array])  # negative to positive z
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
        power = network.power(wave, side="negative")
        assert abs(power.transmission + power.reflection - 1) < 1e-10

    Include every open diffraction order for shorter wavelengths or larger
    periods; ``PlaneWavePorts.diffr_orders`` constructs those mode channels.
    The propagation distance ends at the array's particle-center plane.
    """
    backend = backend_for(unit_cell, lattice, kpar, eta)
    if backend is not None:
        return namespace(backend).solve_periodic(
            unit_cell, lattice=lattice, kpar=kpar, eta=eta
        )
    if not isinstance(unit_cell, Cluster) and not unit_cell.isglobal:
        raise ValueError(
            "periodic cells require an unsolved Cluster or a response at one position; "
            "a solved local cluster would count particle interactions twice"
        )
    local = unit_cell._local if isinstance(unit_cell, Cluster) else unit_cell
    vectors, bloch = periodic_geometry(lattice, kpar, isinstance(local, TMatrix))
    response = local.latticeinteraction.solve(vectors, bloch, eta=eta)
    return PeriodicResponse(local, response, lattice=vectors, kpar=bloch)

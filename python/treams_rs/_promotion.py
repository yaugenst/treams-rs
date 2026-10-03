"""Lift constant NumPy physics objects into a selected differentiation backend.

Only the package's known physics types are translated. Framework objects keep
their identity; a second framework is an error, never an array conversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._framework_backend import Backend


def promote(value: Any, backend: Backend) -> Any:
    """Retain an object in ``backend``, or copy its constant physical metadata."""
    current = getattr(value, "_backend", None)
    if current is not None:
        backend.require_same(
            current, "physical objects must use the same autodiff backend"
        )
        return value

    # Local imports keep ordinary NumPy imports independent of framework classes
    # and avoid a cycle when a framework method promotes one of its arguments.
    from . import _cluster, _periodic, _smatrix, _tmatrix, _waves
    from ._bases import CylindricalBasis, PlaneWavePorts, SphericalBasis
    from ._framework_backend import as_material
    from ._framework_smatrix import SMatrix
    from ._framework_tmatrix import Cluster, PeriodicResponse, PeriodicWave, TMatrix
    from ._framework_waves import PlaneWave, PortSet, PortWave, Wave

    if isinstance(value, _tmatrix._PeriodicInteraction):
        return promote(value.matrix, backend).latticeinteraction

    if isinstance(value, (_tmatrix.TMatrix, _tmatrix.CylindricalTMatrix)):
        return TMatrix(
            backend.array(value.array, complex_=True),
            basis=value.basis,
            k0=backend.array(value.k0),
            medium=as_material(value.medium),
            backend=backend,
            polarization=value.polarization,
        )
    if isinstance(value, _waves.PlaneWave):
        # Framework plane waves store helicity amplitudes. Convert the fixed
        # source before promotion so a parity source describes the same field.
        if value.polarization != "helicity":
            value = value.with_polarization("helicity")
        if np.any(value.direction.imag != 0):
            raise NotImplementedError("autodiff plane waves require a real direction")
        return PlaneWave(
            value.direction.real,
            value.coefficients,
            k0=value.k0,
            medium=value.medium,
            backend=backend,
        )
    if isinstance(value, _waves.Wave):
        if isinstance(value.basis, (SphericalBasis, CylindricalBasis)):
            return Wave(
                backend.array(value.coefficients, complex_=True),
                basis=value.basis,
                k0=backend.array(value.k0),
                medium=as_material(value.medium),
                backend=backend,
                singular=value.kind == "singular",
                polarization=value.polarization,
            )
        if isinstance(value.basis, PlaneWavePorts):
            return PortWave._from_basis(
                value.coefficients,
                basis=value.basis,
                k0=value.k0,
                medium=value.medium,
                backend=backend,
                positive=value.kind == "up",
                polarization=value.polarization,
            )
        raise NotImplementedError(
            "autodiff waves require a multipole basis or PlaneWavePorts"
        )
    if isinstance(value, _smatrix.SMatrix):
        return SMatrix(
            backend.array(value.array, complex_=True),
            ports=PortSet.from_basis(value.basis, backend),
            k0=backend.array(value.k0),
            media=(
                as_material(value.positive_medium),
                as_material(value.negative_medium),
            ),
            backend=backend,
            polarization=value.polarization,
        )
    if isinstance(value, _cluster.Cluster):
        local = value._local
        particles = [
            TMatrix(
                backend.array(block, complex_=True),
                basis=basis,
                k0=backend.array(local.k0),
                medium=as_material(local.medium),
                backend=backend,
                polarization=local.polarization,
            )
            for block, basis in zip(value._blocks, value._bases, strict=True)
        ]
        return Cluster(particles, positions=local.basis.positions)
    if isinstance(value, _periodic.PeriodicResponse):
        response = promote(value._local, backend)
        response.array = backend.array(value.array, complex_=True)
        return PeriodicResponse(
            response, backend.array(value.lattice), backend.array(value.kpar)
        )
    if isinstance(value, _periodic.PeriodicWave):
        return PeriodicWave(
            promote(value._local, backend),
            backend.array(value.lattice),
            backend.array(value.kpar),
        )
    return value

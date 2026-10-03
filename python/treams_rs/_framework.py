"""Functions shared by the framework adapters (advect, jax, torch).

Each adapter binds one Constructors and one Operations instance to its
Backend. The physical objects live in _framework_waves, _framework_tmatrix and
_framework_smatrix.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike

    from ._records import Array, Record

import numpy as np

from . import diff
from ._bases import CylindricalBasis, PlaneWavePorts, SphericalBasis
from ._framework_backend import (
    Backend,
    Basis,
    Recorded,
    as_material,
    material_defaults,
)
from ._framework_smatrix import SMatrix
from ._framework_tmatrix import TMatrix
from ._framework_waves import PlaneWave, PortSet, Wave
from ._validation import check_kind

__all__ = ["Constructors", "Operations"]


def _shape(value: Any) -> tuple[int, ...]:
    """``np.shape(value)``, read directly from framework arrays."""
    # np.shape sends an Advect traced array through __array_function__, which
    # costs about as much as a small native record.
    shape = getattr(value, "shape", None)
    return np.shape(value) if shape is None else tuple(shape)


def _sphere_record(lmax: int) -> Record:
    """Record of a multilayer sphere T-matrix in (k0, radii, epsilon, mu, kappa)."""

    def record(k: Array, r: Array, e: Array, m: Array, c: Array) -> Recorded:
        return diff.sphere(lmax, float(k), r, e, m, c)

    return record


def _adapter_instance(module: str, name: str) -> Any:
    """The shared instance ``name`` of an adapter namespace, as unpickled."""
    return getattr(importlib.import_module(module), name)


class _BoundToBackend:
    """Methods bound to one framework backend and exposed by one adapter namespace.

    ``module`` names that namespace, which holds this instance as ``_name``.
    """

    _name: ClassVar[str]

    def __init__(self, backend: Backend, module: str):
        self.backend = backend
        self.module = module

    def __reduce__(self) -> tuple[Callable[[str, str], Any], tuple[str, str]]:
        # Pickle by reference, as for module functions: the backend holds the
        # framework module, which does not pickle.
        return _adapter_instance, (self.module, self._name)


class Constructors(_BoundToBackend):
    """Physical constructors bound to one framework backend.

    They build the framework PlaneWave, Wave, TMatrix and SMatrix objects. Each
    adapter namespace exposes these bound methods as module functions, for
    example ``treams_rs.jax.slab``; the source catalog documents them there.
    ``module`` names that namespace, whose ``_api`` instance this is.
    """

    _name = "_api"

    def plane_wave(
        self, direction: Any, pol: Any, *, k0: Any, medium: Any = 1.0
    ) -> PlaneWave:
        """Fixed-direction plane wave with dynamic frequency, medium and amplitudes."""
        return PlaneWave(direction, pol, k0=k0, medium=medium, backend=self.backend)

    def smatrix(
        self,
        array: Any,
        *,
        basis: PlaneWavePorts,
        k0: Any,
        negative_medium: Any = 1.0,
        positive_medium: Any = 1.0,
        polarization: str = "helicity",
    ) -> SMatrix:
        """Wrap scattering blocks (2,2,modes,modes) with explicit exterior media."""
        b = self.backend
        matrix = b.array(array, complex_=True)
        if matrix.shape != (2, 2, len(basis), len(basis)):
            raise ValueError("scattering blocks must have shape (2,2,modes,modes)")
        return SMatrix(
            matrix,
            ports=PortSet.from_basis(basis, b),
            k0=b.array(k0),
            media=(as_material(positive_medium), as_material(negative_medium)),
            backend=b,
            polarization=polarization,
        )

    def wave(
        self,
        coefficients: Any,
        *,
        basis: SphericalBasis | CylindricalBasis,
        k0: Any,
        medium: Any = 1.0,
        kind: str = "regular",
        polarization: str = "helicity",
        positions: Any = None,
    ) -> Wave:
        """Multipole wave with one coefficient per basis mode, shape (modes,)."""
        kind = check_kind(kind)
        if kind not in ("regular", "singular"):
            raise ValueError("kind must be 'regular' or 'singular'")
        b = self.backend
        array = b.array(coefficients, complex_=True)
        if array.shape != (len(basis),):
            raise ValueError("one coefficient is required per basis mode")
        return Wave(
            array,
            basis=basis,
            k0=b.array(k0),
            medium=as_material(medium),
            backend=b,
            positions=positions,
            singular=kind == "singular",
            polarization=polarization,
        )

    def tmatrix(
        self,
        array: Any,
        *,
        basis: SphericalBasis | CylindricalBasis,
        k0: Any,
        medium: Any = 1.0,
        polarization: str = "helicity",
        positions: Any = None,
    ) -> TMatrix:
        """Wrap a user response (modes,modes) with fixed labels and dynamic physical metadata."""
        b = self.backend
        matrix = b.array(array, complex_=True)
        if matrix.shape != (len(basis), len(basis)):
            raise ValueError("matrix dimensions must match the basis")
        return TMatrix(
            matrix,
            basis=basis,
            k0=b.array(k0),
            medium=as_material(medium),
            backend=b,
            positions=positions,
            polarization=polarization,
        )

    def sphere_tmatrix(
        self,
        *,
        k0: Any,
        lmax: int,
        radius: Any,
        material: Any,
        medium: Any = 1.0,
        polarization: str = "helicity",
    ) -> TMatrix:
        """Homogeneous sphere with differentiable geometry, material and frequency."""
        return self.multilayer_sphere_tmatrix(
            k0=k0,
            lmax=lmax,
            radii=radius,
            materials=(material,),
            medium=medium,
            polarization=polarization,
        )

    def multilayer_sphere_tmatrix(
        self,
        *,
        k0: Any,
        lmax: int,
        radii: Any,
        materials: Any,
        medium: Any = 1.0,
        polarization: str = "helicity",
    ) -> TMatrix:
        """Concentric layers, one material per radius, with a separate exterior."""
        return self._multilayer(
            k0=k0,
            radii=radii,
            materials=materials,
            medium=medium,
            basis=SphericalBasis.default(lmax),
            polarization=polarization,
        )

    def cylinder_tmatrix(
        self,
        *,
        k0: Any,
        kz: Any,
        mmax: int,
        radius: Any,
        material: Any,
        medium: Any = 1.0,
        polarization: str = "helicity",
    ) -> TMatrix:
        """Homogeneous cylinder; axial mode labels are fixed configuration."""
        return self.multilayer_cylinder_tmatrix(
            k0=k0,
            kz=kz,
            mmax=mmax,
            radii=radius,
            materials=(material,),
            medium=medium,
            polarization=polarization,
        )

    def multilayer_cylinder_tmatrix(
        self,
        *,
        k0: Any,
        kz: Any,
        mmax: int,
        radii: Any,
        materials: Any,
        medium: Any = 1.0,
        polarization: str = "helicity",
    ) -> TMatrix:
        """Concentric cylinders with fixed axial labels and dynamic layers."""
        return self._multilayer(
            k0=k0,
            radii=radii,
            materials=materials,
            medium=medium,
            basis=CylindricalBasis.default(kz, mmax),
            polarization=polarization,
            kz=kz,
            mmax=mmax,
        )

    def _multilayer(
        self,
        *,
        k0: Any,
        radii: Any,
        materials: Sequence[Any],
        medium: Any,
        basis: Basis,
        polarization: str,
        kz: Any = None,
        mmax: int = 0,
    ) -> TMatrix:
        b = self.backend
        layers = [as_material(value) for value in (*materials, medium)]
        e, m, c = (
            b.stack([b.array(getattr(layer, name), complex_=True) for layer in layers])
            for name in ("epsilon", "mu", "kappa")
        )
        r = b.vector(radii)
        k = b.array(k0)
        if isinstance(basis, SphericalBasis):
            record = _sphere_record(max(basis.l, default=0))
            values = (k, r, e, m, c)
        else:

            def record(kz: Any, k: Any, r: Any, e: Any, m: Any, c: Any) -> Any:
                return diff.cylinder(kz, mmax, float(k), r, e, m, c)

            values = (b.vector(kz), k, r, e, m, c)
        array = b.apply(record, (len(basis), len(basis)), *values)
        # The native T-matrices use helicity; conversion is a fixed basis change.
        result = TMatrix(array, basis=basis, k0=k, medium=layers[-1], backend=b)
        if polarization == "helicity":
            return result
        return result.with_polarization(polarization)

    def slab(
        self,
        *,
        basis: PlaneWavePorts,
        k0: Any,
        thickness: Any,
        material: Any,
        negative_medium: Any = 1.0,
        positive_medium: Any = 1.0,
        polarization: str = "helicity",
    ) -> SMatrix:
        """One layer with differentiable geometry and explicit exterior media."""
        return self._layers(
            k0=k0,
            basis=basis,
            materials=(negative_medium, material, positive_medium),
            thickness=thickness,
        ).with_polarization(polarization)

    def interface(
        self,
        *,
        basis: PlaneWavePorts,
        k0: Any,
        negative_medium: Any,
        positive_medium: Any,
        polarization: str = "helicity",
    ) -> SMatrix:
        """Interface from negative to positive side of the port normal."""
        return self._layers(
            k0=k0,
            basis=basis,
            materials=(negative_medium, positive_medium),
            thickness=[],
        ).with_polarization(polarization)

    def multilayer_slab(
        self,
        *,
        basis: PlaneWavePorts,
        k0: Any,
        thicknesses: Any,
        materials: Any,
        negative_medium: Any = 1.0,
        positive_medium: Any = 1.0,
        polarization: str = "helicity",
    ) -> SMatrix:
        """Interior layers in increasing normal order, one material per thickness."""
        return self._layers(
            k0=k0,
            basis=basis,
            materials=(negative_medium, *materials, positive_medium),
            thickness=thicknesses,
        ).with_polarization(polarization)

    def propagation(
        self,
        *,
        distance: Any,
        basis: PlaneWavePorts,
        k0: Any,
        medium: Any = 1.0,
        polarization: str = "helicity",
    ) -> SMatrix:
        """Propagation through a homogeneous medium, differentiable in the distance and the wavevectors."""
        b = self.backend
        material = as_material(medium)
        d = b.array(distance)
        axis = basis.normal_axis
        if d.ndim == 0:
            d = d * b.array(np.eye(3)[axis])
        if d.shape != (3,):
            raise ValueError("distance must be a scalar or Cartesian displacement")
        vectors = b.plane_vectors(
            b.array(basis.components), b.ks(material, k0)[basis.pol.copy()]
        )
        ordering = np.array([(axis + 1) % 3, (axis + 2) % 3, axis])
        result = b.apply(
            diff.propagation_matrix,
            (2, 2, len(basis), len(basis)),
            vectors,
            d[ordering],
        )
        return SMatrix(
            result,
            ports=PortSet.from_basis(basis, b),
            k0=k0,
            media=(material, material),
            backend=b,
        ).with_polarization(polarization)

    def _layers(
        self,
        *,
        k0: Any,
        basis: PlaneWavePorts,
        materials: Sequence[Any],
        thickness: Any,
    ) -> SMatrix:
        """Layer sequence below-to-above, with one thickness per interior medium."""
        b = self.backend
        media = tuple(as_material(m) for m in materials)
        ports = PortSet.from_basis(basis, b)
        group, pol = ports.groups, ports.pols

        def record(ks: Any, zs: Any, q: Any, d: Any) -> Any:
            return diff.layer_stack(
                ks, zs, q, d, alignment=basis.alignment, fixed_q=True
            )

        compact = b.apply(
            record,
            (ports.transverse_wavevectors.shape[0], 2, 2, 2, 2),
            b.stack([b.ks(m, k0) for m in media]),
            b.stack([b.impedance(m) for m in media]),
            ports.transverse_wavevectors,
            b.vector(thickness),
        )
        # One gather builds the (2, 2, ports, ports) blocks from the compact
        # per-group channels; ports in different transverse groups do not couple.
        outgoing = np.arange(2)[:, None, None, None]
        incoming = np.arange(2)[None, :, None, None]
        rows, columns = group[:, None], group[None, :]
        array = compact[rows, outgoing, incoming, pol[:, None], pol[None, :]] * b.array(
            (rows == columns).astype(float)
        )
        return SMatrix(
            array, ports=ports, k0=k0, media=(media[-1], media[0]), backend=b
        )


class Operations(_BoundToBackend):
    """Linear solves, sphere T-matrices and Bessel functions for one backend, run in Rust.

    Each adapter namespace exposes these bound methods as module functions, for
    example ``treams_rs.jax.solve``; ``module`` names that namespace, whose
    ``_ops`` instance this is. Each method declares its output shape, which JAX
    needs to build its callback.
    """

    _name = "_ops"

    def solve(self, operator: Any, rhs: Any) -> Any:
        """Solve A X = B by LU decomposition, differentiable in A and B."""
        return self.backend.operation(diff.solve, operator, rhs, shape=_shape(rhs))

    def interaction(self, local: Any, coupling: Any) -> Any:
        """Solve (I - T C) X = T, differentiable in T (local) and C (coupling)."""
        return self.backend.operation(
            diff.interaction, local, coupling, shape=_shape(local)
        )

    def illuminate(self, local: Any, coupling: Any, incident: Any) -> Any:
        """Solve only the requested incident columns, differentiable in all three inputs."""
        return self.backend.operation(
            diff.illuminate, local, coupling, incident, shape=_shape(incident)
        )

    def sphere(
        self,
        lmax: int,
        k0: Any,
        radii: Any,
        epsilon: Any,
        mu: Any = None,
        kappa: Any = None,
    ) -> Any:
        """Multilayer chiral sphere T-matrix in the helicity basis.

        Differentiable in k0, the radii and the layer materials.
        """

        size = 2 * lmax * (lmax + 2)
        return self.backend.operation(
            _sphere_record(lmax),
            k0,
            radii,
            epsilon,
            *material_defaults(epsilon, mu, kappa),
            shape=(size, size),
        )

    def bessel(
        self,
        z: Any,
        *,
        order: ArrayLike,
        function: str = "j",
        spherical: bool = False,
        derivative: bool = False,
    ) -> Any:
        """Broadcast Bessel values, differentiable in z; the order stays fixed."""

        def record(value: Array) -> tuple[Any, Any]:
            return diff.bessel(
                order,
                value,
                function=function,
                spherical=spherical,
                derivative=derivative,
            )

        return self.backend.operation(
            record, z, shape=np.broadcast_shapes(_shape(z), np.shape(order))
        )

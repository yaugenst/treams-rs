"""Shared physical objects for optional first-order CPU framework adapters.

Arrays and continuous metadata stay in their framework. Discrete bases stay in
Python; native callbacks reconstruct their moving origins at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast, override

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import numpy as np

from . import diff
from . import lattice as lattice_ops
from ._core import CylindricalWaveBasis, PlaneWaveBasisByComp, SphericalWaveBasis

type Basis = SphericalWaveBasis | CylindricalWaveBasis


@dataclass(frozen=True)
class Material:
    """Relative permittivity, permeability and chirality, retaining framework values."""

    epsilon: Any = 1.0
    mu: Any = 1.0
    kappa: Any = 0.0


class CrossSections(NamedTuple):
    """Scattering and extinction area (spheres) or width (cylinders)."""

    scattering: Any
    extinction: Any

    @property
    def absorption(self) -> Any:
        return self.extinction - self.scattering


class PowerBalance(NamedTuple):
    """Incident-normalized transmitted and reflected power."""

    transmission: Any
    reflection: Any

    @property
    def absorption(self) -> Any:
        return 1 - self.transmission - self.reflection


class BandModes(NamedTuple):
    """Normal Bloch wavenumbers and right eigenvectors."""

    wavenumbers: Any
    eigenvectors: Any


def _material(value: Any) -> Material:
    return value if isinstance(value, Material) else Material(value)


class Backend:
    """Framework array operations and one native analytic recording boundary."""

    def __init__(
        self,
        xp: Any,
        call: Callable[..., Any],
        *,
        torch: bool = False,
        validate: Callable[[Any], None] | None = None,
    ):
        self.xp, self.call, self.torch = xp, call, torch
        self.validate = validate

    def array(self, value: Any, *, complex_: bool = False) -> Any:
        if self.validate is not None:
            self.validate(value)
        dtype = self.xp.complex128 if complex_ else self.xp.float64
        if self.torch:
            if isinstance(value, self.xp.Tensor):
                if value.device.type != "cpu" or value.dtype not in (
                    self.xp.float64,
                    self.xp.complex128,
                ):
                    raise TypeError(
                        "framework physics requires CPU float64/complex128 tensors"
                    )
                return value.to(dtype=dtype) if complex_ else value
            if isinstance(value, (list, tuple)) and value:
                return self.stack(
                    [self.array(item, complex_=complex_) for item in value]
                )
            return self.xp.tensor(value, dtype=dtype)
        return self.xp.asarray(value, dtype=dtype)

    def vector(self, value: Any) -> Any:
        array = self.array(value)
        return self.stack((array,)) if array.ndim == 0 else array.reshape(-1)

    def stack(self, values: Sequence[Any], axis: int = 0) -> Any:
        return (
            self.xp.stack(tuple(values), dim=axis)
            if self.torch
            else self.xp.stack(tuple(values), axis=axis)
        )

    def concat(self, values: Sequence[Any], axis: int = 0) -> Any:
        return (
            self.xp.cat(tuple(values), dim=axis)
            if self.torch
            else self.xp.concatenate(tuple(values), axis=axis)
        )

    def ks(self, medium: Material, k0: Any) -> Any:
        n = self.xp.sqrt(
            self.array(medium.epsilon, complex_=True)
            * self.array(medium.mu, complex_=True)
        )
        raw = self.stack((n - medium.kappa, n + medium.kappa))
        return k0 * self.xp.where(raw.imag < 0, -raw, raw)

    def impedance(self, medium: Material) -> Any:
        return self.xp.sqrt(
            self.array(medium.mu, complex_=True)
            / self.array(medium.epsilon, complex_=True)
        )

    def invoke(
        self,
        record: Callable[..., Any],
        shape: tuple[int, ...],
        *values: Any,
        real: bool = False,
    ) -> Any:
        return self.call(record, shape, *values, real=real)

    def match(self, value: Any, first: Any, second: Any) -> Any:
        """Check dynamic physical compatibility at the native recording boundary."""
        if first._backend is not second._backend:
            raise TypeError("physical objects must use the same framework namespace")

        def metadata(item: Any) -> Any:
            return self.stack(
                [
                    self.array(x, complex_=True)
                    for x in (
                        item.k0,
                        item.medium.epsilon,
                        item.medium.mu,
                        item.medium.kappa,
                    )
                ]
            )

        def record(array: Any, left: Any, right: Any) -> Any:
            if not np.array_equal(left, right):
                raise ValueError(
                    "physical operations require matching k0 and embedding medium"
                )

            def pullback(g: Any) -> Any:
                return g, np.zeros_like(left), np.zeros_like(right)

            return array, pullback

        return self.invoke(
            record, tuple(value.shape), value, metadata(first), metadata(second)
        )

    def multilayer(
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
        if polarization != "helicity":
            raise NotImplementedError(
                "framework constructors use helicity; convert with with_polarization"
            )
        layers = [_material(value) for value in (*materials, medium)]
        e, m, c = (
            self.stack(
                [self.array(getattr(layer, name), complex_=True) for layer in layers]
            )
            for name in ("epsilon", "mu", "kappa")
        )
        r = self.vector(radii)
        k = self.array(k0)
        if isinstance(basis, SphericalWaveBasis):
            lmax = max(basis.l, default=0)

            def record(k: Any, r: Any, e: Any, m: Any, c: Any) -> Any:
                return diff.sphere(lmax, float(k), r, e, m, c)

            values = (k, r, e, m, c)
        else:

            def record(kz: Any, k: Any, r: Any, e: Any, m: Any, c: Any) -> Any:
                return diff.cylinder(kz, mmax, float(k), r, e, m, c)

            values = (self.vector(kz), k, r, e, m, c)
        array = self.invoke(record, (len(basis), len(basis)), *values)
        return TMatrix(array, basis=basis, k0=k, medium=layers[-1], backend=self, kz=kz)


class _Fields:
    medium: Material
    _backend: Backend

    def efield(self, points: Any) -> Any:
        raise NotImplementedError

    def hfield(self, points: Any) -> Any:
        raise NotImplementedError

    def dfield(self, points: Any) -> Any:
        """Electric displacement divided by vacuum permittivity."""
        return self.medium.epsilon * self.efield(
            points
        ) + 1j * self.medium.kappa * self.hfield(points)

    def bfield(self, points: Any) -> Any:
        """Magnetic flux density times vacuum light speed."""
        return self.medium.mu * self.hfield(
            points
        ) - 1j * self.medium.kappa * self.efield(points)

    def gfield(self, pol: int, points: Any) -> Any:
        """Helicity-resolved G field with the package's wave-family normalization."""
        if pol not in (0, 1):
            raise ValueError("helicity label must be 0 or 1")
        spherical = isinstance(getattr(self, "basis", None), SphericalWaveBasis)
        normalization = np.sqrt(2.0) if spherical else 1.0
        if getattr(self, "polarization", "helicity") == "helicity":
            normalization *= 0.5
        return normalization * (
            self.efield(points)
            + (2 * pol - 1)
            * 1j
            * self._backend.impedance(self.medium)
            * self.hfield(points)
        )

    def ffield(self, pol: int, points: Any) -> Any:
        """G field with the chiral index weighting in the helicity convention."""
        value = self.gfield(pol, points)
        if getattr(self, "polarization", "helicity") == "parity":
            return value
        ks = self._backend.ks(self.medium, 1.0)
        return value * 2 * ks[pol] / self._backend.xp.sum(ks)


class Wave(_Fields):
    """Multipole coefficients carrying their basis, medium and outgoing convention."""

    coefficients: Any
    basis: Basis
    k0: Any
    medium: Material
    polarization: str
    positions: Any

    @property
    def kind(self) -> str:
        return "outgoing" if self.outgoing else "regular"

    def __init__(
        self,
        coefficients: Any,
        *,
        basis: Basis,
        k0: Any,
        medium: Material,
        backend: Backend,
        positions: Any = None,
        outgoing: bool = False,
        polarization: str = "helicity",
        kz: Any = None,
    ):
        self.coefficients, self.basis, self.k0, self.medium = (
            coefficients,
            basis,
            k0,
            medium,
        )
        self._backend, self.outgoing, self.polarization, self.kz = (
            backend,
            outgoing,
            polarization,
            kz,
        )
        self.positions = backend.array(
            basis.positions if positions is None else positions
        )

    @override
    def efield(self, points: Any) -> Any:
        """Electric field (...,3), differentiable in coefficients, points and medium."""
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
                singular=self.outgoing,
            )

        return b.invoke(
            record,
            (int(np.prod(shape[:-1])), 3),
            self.coefficients,
            points.reshape(-1, 3),
            self.positions,
            b.ks(self.medium, self.k0),
        ).reshape(shape)

    @override
    def hfield(self, points: Any) -> Any:
        """Magnetic field in relative vacuum impedance units."""
        b = self._backend
        if self.polarization != "helicity":
            return self.with_polarization("helicity").hfield(points)
        coefficients = (
            -1j
            * self.coefficients
            * b.array(2 * self.basis.pol - 1)
            / b.impedance(self.medium)
        )
        return Wave(
            coefficients,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=b,
            positions=self.positions,
            outgoing=self.outgoing,
        ).efield(points)

    def in_basis(
        self, basis: Basis, *, positions: Any = None, outgoing: bool | None = None
    ) -> Wave:
        """Re-expand into another multipole basis, retaining field semantics."""
        target_outgoing = self.outgoing if outgoing is None else outgoing
        if target_outgoing and not self.outgoing:
            raise ValueError("a regular wave cannot be converted to outgoing waves")
        b = self._backend
        origins = b.array(basis.positions if positions is None else positions)
        source = self.basis

        def record(to: Any, source_origins: Any, ks: Any) -> Any:
            return diff.expansion(
                type(basis)(basis.modes, to),
                type(source)(source.modes, source_origins),
                ks,
                poltype=self.polarization,
                singular=self.outgoing and not target_outgoing,
            )

        operator = b.invoke(
            record,
            (len(basis), len(source)),
            origins,
            self.positions,
            b.ks(self.medium, self.k0),
        )
        return Wave(
            operator @ self.coefficients,
            basis=basis,
            k0=self.k0,
            medium=self.medium,
            backend=b,
            positions=origins,
            outgoing=target_outgoing,
            polarization=self.polarization,
        )

    def with_polarization(self, polarization: str) -> Wave:
        """Change the static multipole polarization convention."""
        from ._operators import changepoltype

        change = self._backend.array(
            changepoltype((polarization, self.polarization), basis=self.basis),
            complex_=True,
        )
        return Wave(
            change @ self.coefficients,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=self._backend,
            positions=self.positions,
            outgoing=self.outgoing,
            polarization=polarization,
        )


class PlaneWave(_Fields):
    """Plane illumination with a fixed direction and differentiable two-helicity amplitudes.

    The fixed polarization gauge is separated from native dynamic phase derivatives,
    including axis-aligned incidence. Direction itself is static configuration.
    """

    def __init__(
        self,
        direction: Any,
        polarization: Any,
        *,
        k0: Any,
        medium: Any,
        backend: Backend,
    ):
        direction = np.asarray(direction, dtype=float)
        if (
            direction.shape != (3,)
            or not np.all(np.isfinite(direction))
            or np.linalg.norm(direction) == 0
        ):
            raise ValueError("direction must be a finite nonzero Cartesian vector")
        self.direction = direction / np.linalg.norm(direction)
        choices = {"positive_helicity": (0.0, 1.0), "negative_helicity": (1.0, 0.0)}
        if isinstance(polarization, str):
            polarization = choices[polarization]
        elif isinstance(polarization, (int, np.integer)):
            if polarization not in (0, 1):
                raise ValueError("helicity label must be 0 or 1")
            polarization = (1.0, 0.0) if polarization == 0 else (0.0, 1.0)
        self.coefficients = backend.array(polarization, complex_=True)
        if self.coefficients.shape != (2,):
            raise ValueError(
                "polarization requires two helicity amplitudes or a helicity name"
            )
        self.k0, self.medium, self._backend = (
            backend.array(k0),
            _material(medium),
            backend,
        )
        self.polarization = "helicity"

    def in_basis(
        self, basis: Basis, *, positions: Any = None, outgoing: bool | None = None
    ) -> Wave:
        if outgoing:
            raise ValueError("plane illumination expands to regular multipoles")
        b = self._backend
        origins = b.array(basis.positions if positions is None else positions)
        vectors = b.ks(self.medium, self.k0)[:, None] * b.array(self.direction)[None, :]
        # Fixed-direction angular coefficients and dynamic translation phases
        # separate the axis gauge from physical frequency/origin derivatives.
        if isinstance(basis, CylindricalWaveBasis) and (
            np.any(basis.kz != 0) or self.direction[2] != 0
        ):
            raise NotImplementedError(
                "cylindrical plane illumination currently requires kz=0 and transverse incidence; use the expert plane_expansion primitive for fixed axial sectors"
            )
        zero_basis = type(basis)(basis.modes, np.zeros_like(basis.positions))
        angular, _ = diff.plane_expansion(
            zero_basis, np.tile(self.direction, (2, 1)), [0, 1], fixed_vectors=True
        )
        phases = b.invoke(diff.plane_phases, (origins.shape[0], 2), origins, vectors)
        operator = b.array(angular, complex_=True) * phases[basis.pidx.copy()]
        return Wave(
            operator @ self.coefficients,
            basis=basis,
            k0=self.k0,
            medium=self.medium,
            backend=b,
            positions=origins,
        )

    @override
    def efield(self, points: Any) -> Any:
        b = self._backend
        points = b.array(points)
        shape = tuple(points.shape)
        vectors = b.ks(self.medium, self.k0)[:, None] * b.array(self.direction)[None, :]
        electric, _ = diff.plane_field(
            None,
            np.zeros((1, 3)),
            np.tile(self.direction, (2, 1)),
            [0, 1],
            fixed_vectors=True,
        )
        phases = b.invoke(
            diff.plane_phases,
            (int(np.prod(shape[:-1])), 2),
            points.reshape(-1, 3),
            vectors,
        )
        return (
            (phases * self.coefficients)
            @ b.array(electric.reshape(3, 2).T, complex_=True)
        ).reshape(shape)

    @override
    def hfield(self, points: Any) -> Any:
        """Magnetic field with the same native phase and material derivatives."""
        b = self._backend
        wave = PlaneWave(
            self.direction,
            -1j * self.coefficients * b.array([-1.0, 1.0]) / b.impedance(self.medium),
            k0=self.k0,
            medium=self.medium,
            backend=b,
        )
        return wave.efield(points)


class PortWave(_Fields):
    """Plane-port amplitudes with continuous transverse wavevectors and exterior medium."""

    coefficients: Any
    k0: Any
    medium: Material
    transverse_wavevectors: Any
    polarization: str

    def __init__(self, coefficients: Any, *, system: SMatrix, positive: bool):
        self.coefficients = coefficients
        self.k0 = system.k0
        self.medium = system.media[0 if positive else 1]
        self._backend = system._backend
        self.polarization = system.polarization
        self.modes = system.modes
        self.alignment = system.alignment
        self.transverse_wavevectors = system.transverse_wavevectors
        self.positive = positive
        self.fixed_q = system.fixed_q

    def _vectors(self) -> Any:
        b = self._backend
        groups = np.array([group for group, _ in self.modes])
        pols = np.array([pol for _, pol in self.modes])
        q = self.transverse_wavevectors[groups]
        ks = b.ks(self.medium, self.k0)[pols]
        transverse = b.xp.sum(q * q, dim=1) if b.torch else b.xp.sum(q * q, axis=1)
        kz = b.xp.sqrt(ks**2 - transverse)
        kz = b.xp.where(kz.imag < 0, -kz, kz)
        if not self.positive:
            kz = -kz
        local = b.concat((b.array(q, complex_=True), kz[:, None]), axis=1)
        axis = {"xy": 2, "yz": 0, "zx": 1}[self.alignment]
        order = np.argsort([(axis + 1) % 3, (axis + 2) % 3, axis])
        return local[:, order]

    @override
    def efield(self, points: Any) -> Any:
        b = self._backend
        points = b.array(points)
        shape = tuple(points.shape)
        vectors = self._vectors()
        pols = [pol for _, pol in self.modes]
        axis = {"xy": 2, "yz": 0, "zx": 1}[self.alignment]

        def record(v: Any) -> Any:
            fixed = self.fixed_q and np.all(
                v[:, [i for i in range(3) if i != axis]] == 0
            )
            value, context = diff.plane_field(
                None,
                np.zeros((1, 3)),
                v,
                pols,
                poltype=self.polarization,
                fixed_vectors=bool(fixed),
            )

            def pullback(g: Any) -> Any:
                return context.pullback(g)[2]

            return value, pullback

        electric = b.invoke(record, (1, 3, len(self.modes)), vectors)[0]
        phases = b.invoke(
            diff.plane_phases,
            (int(np.prod(shape[:-1])), len(self.modes)),
            points.reshape(-1, 3),
            vectors,
        )
        return ((phases * self.coefficients) @ electric.T).reshape(shape)

    @override
    def hfield(self, points: Any) -> Any:
        b = self._backend
        if self.polarization != "helicity":
            return self.with_polarization("helicity").hfield(points)
        weights = b.array([2 * pol - 1 for _, pol in self.modes])
        from copy import copy

        wave = copy(self)
        wave.coefficients = -1j * self.coefficients * weights / b.impedance(self.medium)
        return wave.efield(points)

    def with_polarization(self, polarization: str) -> PortWave:
        from copy import copy

        from ._operators import changepoltype

        labels = PlaneWaveBasisByComp(
            [(float(group), 0.0, pol) for group, pol in self.modes]
        )
        change = self._backend.array(
            changepoltype((polarization, self.polarization), basis=labels),
            complex_=True,
        )
        wave = copy(self)
        wave.coefficients = change @ self.coefficients
        wave.polarization = polarization
        return wave


class ScatteredPorts(NamedTuple):
    """Outgoing physical waves on the positive and negative sides."""

    positive: PortWave
    negative: PortWave


class TMatrix:
    """Framework-valued response mapping regular illumination to outgoing waves."""

    array: Any
    basis: Basis
    k0: Any
    medium: Material
    polarization: str
    positions: Any

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
        kz: Any = None,
    ):
        self.array, self.basis, self.k0, self.medium = array, basis, k0, medium
        self._backend, self.polarization, self.kz = backend, polarization, kz
        self.positions = backend.array(
            basis.positions if positions is None else positions
        )

    def scatter(self, incident: PlaneWave | Wave) -> Wave:
        """Scatter physical illumination and return an outgoing physical wave."""
        wave = incident.in_basis(self.basis, positions=self.positions, outgoing=False)
        wave.coefficients = self._backend.match(wave.coefficients, self, incident)
        if wave.polarization != self.polarization:
            wave = wave.with_polarization(self.polarization)
        return Wave(
            self.array @ wave.coefficients,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=self._backend,
            positions=self.positions,
            outgoing=True,
            polarization=self.polarization,
        )

    def cross_sections(
        self, incident: PlaneWave | Wave, *, flux: Any = 0.5
    ) -> CrossSections:
        """Named scattering/extinction for a nonabsorbing propagating exterior."""
        b = self._backend
        wave = incident.in_basis(self.basis, positions=self.positions, outgoing=False)
        wave.coefficients = b.match(wave.coefficients, self, incident)
        if wave.polarization != self.polarization:
            wave = wave.with_polarization(self.polarization)
        scattered = self.array @ wave.coefficients
        power = 2 if isinstance(self.basis, SphericalWaveBasis) else 1

        def propagating(ks: Any, flux: Any) -> Any:
            if (
                np.any(ks.imag != 0)
                or np.any(ks.real == 0)
                or not np.isfinite(flux)
                or flux <= 0
            ):
                raise ValueError(
                    "cross sections require a nonabsorbing propagating exterior and positive flux"
                )
            if isinstance(self.basis, CylindricalWaveBasis) and np.any(
                np.abs(self.basis.kz) >= np.abs(ks[self.basis.pol])
            ):
                raise ValueError("cross widths require propagating axial sectors")

            def pullback(g: Any) -> Any:
                return g, np.zeros_like(flux)

            return ks, pullback

        ks = b.invoke(propagating, (2,), b.ks(self.medium, self.k0), b.array(flux))
        weights = ks[self.basis.pol.copy()].real ** (-power)
        weighted = scattered * weights

        def record(o1: Any, o2: Any, ks: Any) -> Any:
            return diff.expansion(
                type(self.basis)(self.basis.modes, o1),
                type(self.basis)(self.basis.modes, o2),
                ks,
                poltype=self.polarization,
            )

        overlap = b.invoke(
            record,
            tuple(self.array.shape),
            self.positions,
            self.positions,
            b.ks(self.medium, self.k0),
        )
        factor = 0.5 if power == 2 else 2.0
        return CrossSections(
            b.xp.sum(b.xp.conj(scattered) * (overlap @ weighted)).real * factor / flux,
            -b.xp.sum(b.xp.conj(wave.coefficients) * weighted).real * factor / flux,
        )

    def with_polarization(self, polarization: str) -> TMatrix:
        """Convert the response convention with a fixed analytic basis transform."""
        from ._operators import changepoltype

        change = self._backend.array(
            changepoltype((polarization, self.polarization), basis=self.basis),
            complex_=True,
        )
        return TMatrix(
            change @ self.array @ change.T,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=self._backend,
            positions=self.positions,
            polarization=polarization,
            kz=self.kz,
        )


class Cluster:
    """Uncoupled particles; solve interactions explicitly or only scatter an illumination."""

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
        self.basis = type(first.basis)(modes, np.zeros((len(particles), 3)))
        self.k0, self.medium, self.polarization = (
            first.k0,
            first.medium,
            first.polarization,
        )

    def _local(self) -> Any:
        b = self._backend
        return b.concat(
            [
                b.concat(
                    [
                        b.match(particle.array, self, particle)
                        if i == j
                        else b.array(
                            np.zeros((len(particle.basis), len(other.basis))),
                            complex_=True,
                        )
                        for j, other in enumerate(self.particles)
                    ],
                    axis=1,
                )
                for i, particle in enumerate(self.particles)
            ],
            axis=0,
        )

    def _coupling(self) -> Any:
        b = self._backend

        def record(to: Any, source: Any, ks: Any) -> Any:
            return diff.expansion(
                type(self.basis)(self.basis.modes, to),
                type(self.basis)(self.basis.modes, source),
                ks,
                singular=True,
                poltype=self.polarization,
            )

        return b.invoke(
            record,
            (len(self.basis), len(self.basis)),
            self.positions,
            self.positions,
            b.ks(self.medium, self.k0),
        )

    def solve(self) -> TMatrix:
        b = self._backend
        local = self._local()
        result = b.invoke(diff.interaction, tuple(local.shape), local, self._coupling())
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
        b = self._backend
        wave = incident.in_basis(self.basis, positions=self.positions, outgoing=False)
        wave.coefficients = b.match(wave.coefficients, self, incident)
        if wave.polarization != self.polarization:
            wave = wave.with_polarization(self.polarization)
        coefficients = b.invoke(
            diff.illuminate,
            (len(self.basis), 1),
            self._local(),
            self._coupling(),
            wave.coefficients[:, None],
        )[:, 0]
        return Wave(
            coefficients,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            backend=b,
            positions=self.positions,
            outgoing=True,
            polarization=self.polarization,
        )


class PeriodicResponse:
    """Solved periodic multipole response, ready for conversion to physical ports."""

    def __init__(self, response: TMatrix, lattice: Any, kpar: Any):
        self.response, self.lattice, self.kpar = response, lattice, kpar

    def to_smatrix(
        self, basis: PlaneWaveBasisByComp | None = None, *, orders: Any = None
    ) -> SMatrix:
        """Convert to fixed ports or integer diffraction orders following cell/Bloch changes.

        Spherical orders have shape (groups,2); cylindrical orders are a vector
        for one fixed axial sector. Order ports include both helicities (1,0).
        The returned transverse_wavevectors retain framework derivatives.
        """
        tm, b = self.response, self.response._backend
        spherical = isinstance(tm.basis, SphericalWaveBasis)
        alignment = "xy" if spherical else "zx"
        if (basis is None) == (orders is None):
            raise ValueError("provide a plane basis or integer diffraction orders")
        if orders is None:
            assert basis is not None
            if basis.alignment != alignment:
                raise ValueError("ports must be xy for spheres or zx for cylinders")
            groups = list(dict.fromkeys(tuple(q) for q in basis.components))
            modes = tuple(
                (groups.index(tuple(q)), int(pol))
                for q, pol in zip(basis.components, basis.pol, strict=True)
            )
            q = b.array(groups)
            fixed_q = True
        else:
            labels = np.asarray(orders)
            if not np.all(np.isfinite(labels)) or not np.all(
                labels == np.round(labels)
            ):
                raise ValueError("diffraction orders must be finite integers")
            if spherical:
                if labels.ndim != 2 or labels.shape[1] != 2:
                    raise ValueError("spherical orders require shape (groups,2)")
                a = self.lattice
                determinant = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
                reciprocal = (
                    b.stack(
                        (b.stack((a[1, 1], -a[1, 0])), b.stack((-a[0, 1], a[0, 0])))
                    )
                    / determinant
                )
                q = self.kpar + 2 * np.pi * (b.array(labels) @ reciprocal)
            else:
                if labels.ndim != 1:
                    raise ValueError("cylindrical orders require a vector")
                axial = np.unique(cast("CylindricalWaveBasis", tm.basis).kz)
                if len(axial) != 1:
                    raise NotImplementedError(
                        "order ports require one cylindrical axial sector"
                    )
                transverse = (
                    self.kpar[0] + 2 * np.pi * b.array(labels) / self.lattice[0, 0]
                )
                q = b.stack((transverse * 0 + axial[0], transverse), axis=1)
            if len(np.unique(labels, axis=0)) != len(labels) or len(labels) == 0:
                raise ValueError("diffraction orders must be nonempty and distinct")
            modes = tuple((i, pol) for i in range(len(labels)) for pol in (1, 0))
            fixed_q = False
        indices = np.array([group for group, _ in modes])
        pols = np.array([pol for _, pol in modes])
        channel_q = q[indices]
        measure = (
            b.xp.abs(
                self.lattice[0, 0] * self.lattice[1, 1]
                - self.lattice[0, 1] * self.lattice[1, 0]
            )
            if spherical
            else b.xp.abs(self.lattice[0, 0])
        )

        def record(
            origins: Any, ks: Any, q: Any, measure: Any, a: Any, bloch: Any
        ) -> Any:
            diffraction = (
                (q - bloch) @ a.T / (2 * np.pi)
                if spherical
                else (q[:, 1] - bloch[0]) * a[0, 0] / (2 * np.pi)
            )
            if not np.allclose(diffraction, np.round(diffraction), atol=1e-9, rtol=0):
                raise ValueError("plane ports must match lattice diffraction orders")
            dynamic = type(tm.basis)(tm.basis.modes, origins)
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

            def pullback(g: Any) -> Any:
                return (*context.pullback(g), np.zeros_like(a), np.zeros_like(bloch))

            return value, pullback

        channels = b.invoke(
            record,
            (2, 2, len(tm.basis), len(modes)),
            tm.positions,
            b.ks(tm.medium, tm.k0),
            channel_q,
            measure,
            self.lattice,
            self.kpar,
        )
        array = b.invoke(
            diff.smatrix_from_array, (2, 2, len(modes), len(modes)), tm.array, channels
        )
        return SMatrix(
            array,
            basis=basis,
            k0=tm.k0,
            media=(tm.medium, tm.medium),
            backend=b,
            transverse=q,
            modes=modes,
            alignment=alignment,
            fixed_q=fixed_q,
        )


class SMatrix:
    """Four scattering blocks with explicit positive/negative-side media and plane-wave ports."""

    array: Any
    basis: PlaneWaveBasisByComp | None
    k0: Any
    transverse_wavevectors: Any
    modes: tuple[tuple[int, int], ...]
    polarization: str

    @property
    def negative_medium(self) -> Material:
        """Exterior on the negative side of the plane normal."""
        return self.media[1]

    @property
    def positive_medium(self) -> Material:
        """Exterior on the positive side of the plane normal."""
        return self.media[0]

    def __init__(
        self,
        array: Any,
        *,
        basis: PlaneWaveBasisByComp | None,
        k0: Any,
        media: tuple[Material, Material],
        backend: Backend,
        transverse: Any = None,
        modes: tuple[tuple[int, int], ...] | None = None,
        alignment: str = "xy",
        fixed_q: bool = True,
    ):
        self.array, self.basis, self.k0, self.media, self._backend = (
            array,
            basis,
            k0,
            media,
            backend,
        )

        self.polarization = "helicity"
        self.fixed_q = fixed_q
        if basis is not None and transverse is None:
            groups = list(dict.fromkeys(tuple(q) for q in basis.components))
            transverse = backend.array(groups)
            modes = tuple(
                (groups.index(tuple(q)), int(pol))
                for q, pol in zip(basis.components, basis.pol, strict=True)
            )
            alignment = basis.alignment
        assert modes is not None
        self.modes, self.alignment = modes, alignment
        self.transverse_wavevectors = transverse

    def with_polarization(self, polarization: str) -> SMatrix:
        """Convert the port polarization convention using a static basis matrix."""
        if polarization == getattr(self, "polarization", "helicity"):
            return self
        from ._operators import changepoltype

        labels = PlaneWaveBasisByComp(
            [(float(group), 0.0, pol) for group, pol in self.modes]
        )
        change = self._backend.array(
            changepoltype(
                (polarization, getattr(self, "polarization", "helicity")), basis=labels
            ),
            complex_=True,
        )
        result = SMatrix(
            change @ self.array @ change.T,
            basis=self.basis,
            k0=self.k0,
            media=self.media,
            backend=self._backend,
            transverse=self.transverse_wavevectors,
            modes=self.modes,
            alignment=self.alignment,
            fixed_q=self.fixed_q,
        )
        result.polarization = polarization
        return result

    def _incident(self, incident: Any, side: str) -> Any:
        b = self._backend
        if not isinstance(incident, (PlaneWave, PortWave)):
            if isinstance(incident, Wave):
                raise ValueError(
                    "planar scattering requires a plane wave or plane-port wave"
                )
            return b.array(incident, complex_=True)
        if incident._backend is not b:
            raise TypeError("incident wave must use the same framework namespace")
        medium = self.media[1 if side == "negative" else 0]
        axis = {"xy": 2, "yz": 0, "zx": 1}[self.alignment]
        if isinstance(incident, PlaneWave):
            if (incident.direction[axis] > 0) != (
                side == "negative"
            ) or incident.direction[axis] == 0:
                raise ValueError(
                    "plane wave propagates away from the selected incident side"
                )
            vectors = (
                b.ks(incident.medium, incident.k0)[:, None]
                * b.array(incident.direction)[None, :]
            )
            transverse = vectors[:, [(axis + 1) % 3, (axis + 2) % 3]]
            pols = (0, 1)
        else:
            if (
                incident.positive != (side == "negative")
                or incident.alignment != self.alignment
            ):
                raise ValueError(
                    "port wave propagates away from the selected incident side"
                )
            transverse = incident.transverse_wavevectors[
                np.array([group for group, _ in incident.modes])
            ]
            pols = tuple(pol for _, pol in incident.modes)

        def metadata(k0: Any, material: Material) -> Any:
            return b.stack(
                [
                    b.array(v, complex_=True)
                    for v in (k0, material.epsilon, material.mu, material.kappa)
                ]
            )

        def record(c: Any, vectors: Any, ports: Any, left: Any, right: Any) -> Any:
            if not np.array_equal(left, right):
                raise ValueError(
                    "incident wave requires matching k0 and exterior medium"
                )
            selection = np.zeros((len(self.modes), len(pols)), dtype=np.complex128)
            for column, pol in enumerate(pols):
                indices = [
                    i
                    for i, (group, p) in enumerate(self.modes)
                    if p == pol
                    and np.allclose(
                        ports[group], vectors[column], atol=1e-12, rtol=1e-12
                    )
                ]
                if len(indices) != 1:
                    if c[column] != 0:
                        raise ValueError(
                            "incident plane wave must match one represented diffraction port"
                        )
                    continue
                selection[indices[0], column] = 1

            def pullback(g: Any) -> Any:
                return (
                    selection.T @ g,
                    np.zeros_like(vectors),
                    np.zeros_like(ports),
                    np.zeros_like(left),
                    np.zeros_like(right),
                )

            return selection @ c, pullback

        value = b.invoke(
            record,
            (len(self.modes),),
            incident.coefficients,
            transverse,
            self.transverse_wavevectors,
            metadata(self.k0, medium),
            metadata(incident.k0, incident.medium),
        )
        if incident.polarization != self.polarization:
            from ._operators import changepoltype

            labels = PlaneWaveBasisByComp(
                [(float(group), 0.0, pol) for group, pol in self.modes]
            )
            value = (
                b.array(
                    changepoltype(
                        (self.polarization, incident.polarization), basis=labels
                    ),
                    complex_=True,
                )
                @ value
            )
        return value

    def scatter(self, *, negative: Any = None, positive: Any = None) -> ScatteredPorts:
        """Scatter one coherent illumination supplied on either or both sides."""
        if negative is None and positive is None:
            raise ValueError("supply an incident wave on at least one side")
        b = self._backend
        zero = b.array(np.zeros(len(self.modes)), complex_=True)
        up = zero if negative is None else self._incident(negative, "negative")
        down = zero if positive is None else self._incident(positive, "positive")
        return ScatteredPorts(
            PortWave(
                self.array[0, 0] @ up + self.array[0, 1] @ down,
                system=self,
                positive=True,
            ),
            PortWave(
                self.array[1, 0] @ up + self.array[1, 1] @ down,
                system=self,
                positive=False,
            ),
        )

    def power(self, incident: Any, *, side: str | None = None) -> PowerBalance:
        """Transmission/reflection for port amplitudes; fixed port wavevectors."""
        if side is None:
            axis = {"xy": 2, "yz": 0, "zx": 1}[self.alignment]
            side = (
                ("negative" if incident.direction[axis] > 0 else "positive")
                if isinstance(incident, PlaneWave)
                else ("negative" if incident.positive else "positive")
                if isinstance(incident, PortWave)
                else "negative"
            )
        if side not in ("negative", "positive"):
            raise ValueError("incident side must be negative or positive")
        direction = "up" if side == "negative" else "down"
        b = self._backend
        incoming = self._incident(incident, side).reshape(len(self.modes), -1)

        def record(a: Any, i: Any, ks: Any, zs: Any, q: Any) -> Any:
            value, context = diff.smatrix_tr(
                a,
                i,
                ks,
                zs,
                q,
                modes=self.modes,
                poltype=self.polarization,
                axis={"xy": 2, "yz": 0, "zx": 1}[self.alignment],
                modetype=direction,
                fixed_q=self.fixed_q,
            )

            def pullback(g: Any) -> Any:
                return context.pullback(np.asarray(g, dtype=np.complex128))

            return value, pullback

        power = b.invoke(
            record,
            (2, incoming.shape[1]),
            self.array,
            incoming,
            b.stack([b.ks(m, self.k0) for m in self.media]),
            b.stack([b.impedance(m) for m in self.media]),
            self.transverse_wavevectors,
            real=True,
        )
        return PowerBalance(b.xp.squeeze(power[0]), b.xp.squeeze(power[1]))

    def cascade(self, upper: SMatrix) -> SMatrix:
        """Compose this lower system with the adjacent upper system."""
        if (
            self.modes != upper.modes
            or self.alignment != upper.alignment
            or self.polarization != upper.polarization
        ):
            raise ValueError("stacked systems require the same port basis")
        b = self._backend
        if b is not upper._backend:
            raise TypeError("stacked systems require one framework namespace")

        def record(
            lower: Any,
            higher: Any,
            first_q: Any,
            second_q: Any,
            first_k: Any,
            second_k: Any,
            first_medium: Any,
            second_medium: Any,
        ) -> Any:
            if not (
                np.array_equal(first_q, second_q)
                and np.array_equal(first_k, second_k)
                and np.array_equal(first_medium, second_medium)
            ):
                raise ValueError(
                    "stacked systems require matching wavevectors, k0 and adjacent medium"
                )
            value, context = diff.smatrix_add(lower, higher)

            def pullback(g: Any) -> Any:
                return (
                    *context.pullback(g),
                    np.zeros_like(first_q),
                    np.zeros_like(second_q),
                    np.zeros_like(first_k),
                    np.zeros_like(second_k),
                    np.zeros_like(first_medium),
                    np.zeros_like(second_medium),
                )

            return value, pullback

        def parameters(medium: Material) -> Any:
            return b.stack(
                [
                    b.array(v, complex_=True)
                    for v in (medium.epsilon, medium.mu, medium.kappa)
                ]
            )

        array = b.invoke(
            record,
            tuple(self.array.shape),
            self.array,
            upper.array,
            self.transverse_wavevectors,
            upper.transverse_wavevectors,
            b.array(self.k0),
            b.array(upper.k0),
            parameters(self.media[0]),
            parameters(upper.media[1]),
        )
        result = SMatrix(
            array,
            basis=self.basis,
            k0=self.k0,
            media=(upper.media[0], self.media[1]),
            backend=b,
            transverse=self.transverse_wavevectors,
            modes=self.modes,
            alignment=self.alignment,
            fixed_q=self.fixed_q,
        )
        result.polarization = self.polarization
        return result

    def bands(self, period: Any) -> BandModes:
        """Principal-branch normal Bloch wavenumbers and right vectors."""
        b = self._backend
        n = 2 * len(self.modes)

        def record(a: Any, p: Any, outer: Any) -> Any:
            if not np.array_equal(outer[0], outer[1]):
                raise ValueError("bands require equal outer media")
            (values, vectors), context = diff.bands(a, float(p))

            def pullback(g: Any) -> Any:
                return (*context.pullback(g[0], g[1:]), np.zeros_like(outer))

            return np.vstack((values, vectors)), pullback

        packed = b.invoke(
            record,
            (n + 1, n),
            self.array,
            b.array(period),
            b.stack(
                [
                    b.stack(
                        [b.array(v, complex_=True) for v in (m.epsilon, m.mu, m.kappa)]
                    )
                    for m in self.media
                ]
            ),
        )
        return BandModes(packed[0], packed[1:])


def solve_periodic(
    unit_cell: TMatrix | Cluster, *, lattice: Any, kpar: Any, eta: complex = 0
) -> PeriodicResponse:
    """Solve unit-cell coupling once, retaining dynamic lattice/Bloch coordinates."""
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

    def record(to: Any, source: Any, ks: Any, q: Any, a: Any) -> Any:
        return lattice_ops.expansion_with_context(
            type(basis)(basis.modes, to),
            type(basis)(basis.modes, source),
            ks,
            a,
            q,
            poltype=unit_cell.polarization,
            eta=eta,
        )

    shape = (len(basis), len(basis))
    coupling = b.invoke(
        record,
        shape,
        unit_cell.positions,
        unit_cell.positions,
        b.ks(unit_cell.medium, unit_cell.k0),
        q,
        a,
    )
    local = unit_cell._local() if isinstance(unit_cell, Cluster) else unit_cell.array
    response = b.invoke(diff.interaction, shape, local, coupling)
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


def layer_stack(
    backend: Backend,
    *,
    k0: Any,
    basis: PlaneWaveBasisByComp,
    materials: Sequence[Any],
    thickness: Any,
) -> SMatrix:
    """Layer sequence below-to-above, with one thickness per interior medium."""
    b = backend
    media = tuple(_material(m) for m in materials)
    q = np.asarray(list(dict.fromkeys(tuple(q) for q in basis.components)))

    def record(ks: Any, zs: Any, q: Any, d: Any) -> Any:
        return diff.layer_stack(ks, zs, q, d, alignment=basis.alignment, fixed_q=True)

    compact = b.invoke(
        record,
        (len(q), 2, 2, 2, 2),
        b.stack([b.ks(m, k0) for m in media]),
        b.stack([b.impedance(m) for m in media]),
        b.array(q),
        b.vector(thickness),
    )
    blocks = []
    for outgoing in range(2):
        row = []
        for incoming in range(2):
            rows = []
            for qi, pi in zip(basis.components, basis.pol, strict=True):
                group = next(
                    i for i, value in enumerate(q) if np.array_equal(qi, value)
                )
                rows.append(
                    b.stack(
                        [
                            compact[group, outgoing, incoming, int(pi), int(pj)]
                            if np.array_equal(qi, qj)
                            else b.array(0.0, complex_=True)
                            for qj, pj in zip(basis.components, basis.pol, strict=True)
                        ]
                    )
                )
            row.append(b.stack(rows))
        blocks.append(b.stack(row))
    return SMatrix(
        b.stack(blocks), basis=basis, k0=k0, media=(media[-1], media[0]), backend=b
    )


def propagation(
    backend: Backend,
    *,
    distance: Any,
    basis: PlaneWaveBasisByComp,
    k0: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
) -> SMatrix:
    """Native homogeneous propagation; distance may be scalar or Cartesian."""
    b = backend
    material = _material(medium)
    d = b.array(distance)
    axis = basis.normal_axis
    if d.ndim == 0:
        d = d * b.array(np.eye(3)[axis])
    if d.shape != (3,):
        raise ValueError("distance must be a scalar or Cartesian displacement")
    q = b.array(basis.components)
    ks = b.ks(material, k0)[basis.pol.copy()]
    kz = b.xp.sqrt(
        ks**2 - b.xp.sum(q * q, axis=1)
        if not b.torch
        else ks**2 - b.xp.sum(q * q, dim=1)
    )
    kz = b.xp.where(kz.imag < 0, -kz, kz)
    vectors = b.concat((b.array(q, complex_=True), kz[:, None]), axis=1)
    ordering = np.array([(axis + 1) % 3, (axis + 2) % 3, axis])
    result = b.invoke(
        diff.propagation, (2, 2, len(basis), len(basis)), vectors, d[ordering]
    )
    return SMatrix(
        result, basis=basis, k0=k0, media=(material, material), backend=b
    ).with_polarization(polarization)

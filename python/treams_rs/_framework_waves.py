"""Framework waves: multipole waves, plane waves and port waves."""

from __future__ import annotations

import sys
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from ._framework_smatrix import SMatrix

import numpy as np

from . import _native, diff
from ._bases import ALIGNMENT_AXIS, CylindricalBasis, PlaneWavePorts, SphericalBasis
from ._framework_backend import Backend, Basis, Material, as_material, port_modes
from ._records import DerivativeContext
from ._saved import native_state
from ._validation import check_kind, one_of

__all__ = ["HasPorts", "PlaneWave", "PortSet", "PortWave", "Wave"]


def _select_input(context: Any, index: int, *primals: Any) -> DerivativeContext:
    """Differentiate one input while the other native inputs remain constant."""

    def pullback(g: Any) -> Any:
        return context.pullback(g)[index]

    def pushforward(tangent: Any) -> Any:
        return context.pushforward(
            *(
                tangent if i == index else np.zeros_like(value)
                for i, value in enumerate(primals)
            )
        )

    return DerivativeContext(pullback, pushforward, native_context=context)


def _direction_is_dynamic(value: Any) -> bool:
    """Concrete framework direction constants need no angular derivative."""
    from ._dispatch import backend_for

    if isinstance(value, (list, tuple)):
        return any(_direction_is_dynamic(item) for item in value)
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return (
            value.requires_grad
            or torch.autograd.forward_ad.unpack_dual(value).tangent is not None
        )
    jax = sys.modules.get("jax")
    if jax is not None:
        if isinstance(value, jax.core.Tracer):
            return True
        if isinstance(value, jax.Array):
            return False
    return backend_for(value) is not None


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
    coefficients: Any

    @property
    def array(self) -> Any:
        """The framework-valued wave coefficients."""
        return self.coefficients

    @property
    def material(self) -> Material:
        """treams name of ``medium``."""
        return self.medium

    @property
    def poltype(self) -> str:
        """treams name of ``polarization``."""
        return self.polarization

    def efield(self, points: Any = None, *, r: Any = None) -> Any:
        raise NotImplementedError

    def _combined(self, electric: Any, magnetic: Any, points: Any) -> Any:
        """``electric * E + magnetic * H`` from one native electric-field evaluation."""
        raise NotImplementedError

    def _weights(self, electric: Any, magnetic: Any, signs: Any) -> Any:
        # In the helicity basis H is E of the coefficients times -i(2 pol - 1)/Z,
        # so a linear combination of E and H is E of reweighted coefficients.
        b = self._backend
        return electric + magnetic * (-1j * b.array(signs) / b.impedance(self.medium))

    def hfield(self, points: Any = None, *, r: Any = None) -> Any:
        """Magnetic field in relative vacuum impedance units."""
        points = one_of("points", points, "r", r, None)
        return self._combined(0.0, 1.0, points)

    def dfield(self, points: Any = None, *, r: Any = None) -> Any:
        """Electric displacement divided by vacuum permittivity."""
        points = one_of("points", points, "r", r, None)
        return self._combined(self.medium.epsilon, 1j * self.medium.kappa, points)

    def bfield(self, points: Any = None, *, r: Any = None) -> Any:
        """Magnetic flux density times vacuum light speed."""
        points = one_of("points", points, "r", r, None)
        return self._combined(-1j * self.medium.kappa, self.medium.mu, points)

    def gfield(self, pol: int, points: Any = None, *, r: Any = None) -> Any:
        """Helicity-resolved G field, normalized as in treams for this wave family."""
        points = one_of("points", points, "r", r, None)
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

    def ffield(self, pol: int, points: Any = None, *, r: Any = None) -> Any:
        """G field with the chiral index weighting in the helicity convention."""
        points = one_of("points", points, "r", r, None)
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

    @property
    def modetype(self) -> str:
        """treams name of ``kind``."""
        return self.kind

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
    def efield(self, points: Any = None, *, r: Any = None) -> Any:
        """Electric field at points (..., 3), differentiable in points and wave values."""
        points = one_of("points", points, "r", r, None)
        b = self._backend
        points = b.array(points)
        shape = tuple(points.shape)
        basis = self.basis

        if self.coefficients.ndim == 2:

            @native_state(
                _native.FieldOperatorContext,
                lambda inputs: (
                    len(basis),
                    inputs[1].shape[0],
                    inputs[0].shape[0],
                    isinstance(basis, CylindricalBasis),
                ),
            )
            def record_operator(p: Any, o: Any, ks: Any) -> Any:
                return diff.field_operator(
                    p,
                    type(basis)(basis.modes, o),
                    ks,
                    poltype=self.polarization,
                    singular=self.singular,
                )

            operator = b.apply(
                record_operator,
                (int(np.prod(shape[:-1])), 3, len(basis)),
                points.reshape(-1, 3),
                self.positions,
                b.ks(self.medium, self.k0),
            )
            return (operator @ self.coefficients).reshape(
                (*shape, self.coefficients.shape[1])
            )

        @native_state(
            _native.FieldContext,
            lambda inputs: (
                len(basis),
                inputs[2].shape[0],
                inputs[1].shape[0],
                isinstance(basis, CylindricalBasis),
            ),
        )
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
        if self.coefficients.ndim == 2:
            weights = weights[:, None]
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
        self,
        basis: Basis,
        *,
        positions: Any = None,
        singular: bool | None = None,
        kind: str | None = None,
    ) -> Wave:
        """Re-expand into another multipole basis; the field stays the same."""
        target_kind = one_of(
            "kind",
            kind,
            "singular",
            None if singular is None else ("singular" if singular else "regular"),
            self.kind,
        )
        if check_kind(target_kind) not in ("regular", "singular"):
            raise ValueError("multipole waves require regular or singular kind")
        target_singular = target_kind == "singular"
        if (
            basis is self.basis
            and target_singular == self.singular
            and (positions is None or positions is self.positions)
        ):
            return self
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
    """Plane wave with real direction and helicity or parity amplitudes.

    Direction, amplitudes, k0 and medium may carry gradients. Fixed directions
    support incidence along a polarization axis; direction derivatives require
    off-axis incidence, where the native polarization gauge is differentiable.
    """

    kind = "up"
    """Directional plane-wave kind in the treams convention."""
    modetype = kind
    """treams name of ``kind``."""

    def __init__(
        self,
        direction: Any,
        pol: Any,
        *,
        k0: Any,
        medium: Any,
        backend: Backend,
        polarization: str = "helicity",
    ):
        from ._dispatch import backend_for

        direction_backend = backend_for(direction)
        if direction_backend is not None:
            backend.require_same(
                direction_backend, "direction must use the same autodiff backend"
            )
        self._fixed_direction = not _direction_is_dynamic(direction)
        if self._fixed_direction:
            direction = np.asarray(direction)
            if np.iscomplexobj(direction):
                if np.any(direction.imag != 0):
                    raise ValueError("autodiff plane waves require a real direction")
                direction = direction.real
            norm = np.linalg.norm(direction) if direction.shape == (3,) else 0.0
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("direction must be a finite nonzero Cartesian vector")
            self.direction = direction / norm
        else:
            direction = backend.array(direction, complex_=True)
            if direction.shape != (3,):
                raise ValueError("direction must be a finite nonzero Cartesian vector")

            def check_direction(vector: Any) -> None:
                if np.any(vector.imag != 0):
                    raise ValueError("autodiff plane waves require a real direction")
                if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
                    raise ValueError(
                        "direction must be a finite nonzero Cartesian vector"
                    )

            direction = backend.xp.real(backend.guard(check_direction, direction))
            self.direction = direction / backend.xp.sqrt(backend.xp.sum(direction**2))
        if polarization not in ("helicity", "parity"):
            raise ValueError("polarization must be helicity or parity")
        self.k0, self.medium = backend.array(k0), as_material(medium)
        self._backend, self.polarization = backend, polarization
        choices = {"positive_helicity": (0.0, 1.0), "negative_helicity": (1.0, 0.0)}
        if isinstance(pol, str):
            if pol not in choices:
                raise ValueError(
                    "named pol must be positive_helicity or negative_helicity"
                )
            pol = backend.change_port_polarization(
                backend.array(choices[pol], complex_=True),
                ((0, 0), (0, 1)),
                "helicity",
                polarization,
                (0,),
            )
        elif backend_for(pol) is None and np.ndim(pol) == 0:
            label = complex(np.asarray(pol).item())
            if label not in (-1, 0, 1):
                raise ValueError("helicity label must be 0 or 1")
            pol = (1.0, 0.0) if label in (-1, 0) else (0.0, 1.0)
        self.coefficients = backend.array(pol, complex_=True)
        if self.coefficients.shape == (3,):
            helicity = polarization == "helicity"
            projection = _plane_operator(
                self,
                self._vectors(),
                [1, 0] if helicity else [0, 1],
                lambda _: self._fixed_direction,
            )
            self.coefficients = projection.T @ self.coefficients
            if not helicity:
                self.coefficients = self.coefficients * backend.array([-1.0, 1.0])
        if self.coefficients.shape != (2,):
            raise ValueError(
                "pol requires two amplitudes, three electric components or a helicity name"
            )
        if polarization == "parity":
            self.coefficients = backend.require_achiral(self.coefficients, self.medium)

    def with_polarization(self, polarization: str) -> PlaneWave:
        """Represent the same plane wave with helicity or parity amplitudes."""
        wave = copy(self)
        wave.coefficients = self._backend.change_port_polarization(
            self.coefficients, ((0, 0), (0, 1)), self.polarization, polarization, (0,)
        )
        if polarization == "parity":
            wave.coefficients = self._backend.require_achiral(
                wave.coefficients, self.medium
            )
        wave.polarization = polarization
        return wave

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
        if isinstance(basis, CylindricalBasis):

            def check_axial(_: Any, direction: Any) -> None:
                if np.any(basis.kz != 0) or direction[2] != 0:
                    raise NotImplementedError(
                        "cylindrical plane illumination requires kz=0 and transverse incidence"
                    )

            if self._fixed_direction:
                check_axial(None, self.direction)
            else:
                vectors = b.guard(check_axial, vectors, self.direction)
        return _plane_expansion(
            self, basis, positions, vectors, [0, 1], lambda _: self._fixed_direction
        )

    def _vectors(self) -> Any:
        """Wavevectors of the two helicities, one row each."""
        b = self._backend
        return (
            b.plane_ks(self.medium, self.k0)[:, None] * b.array(self.direction)[None, :]
        )

    @override
    def efield(self, points: Any = None, *, r: Any = None) -> Any:
        """Electric field at points (..., 3), differentiable in points and wave values."""
        points = one_of("points", points, "r", r, None)
        return _plane_efield(self, points, [0, 1], lambda _: self._fixed_direction)

    @override
    def _combined(self, electric: Any, magnetic: Any, points: Any) -> Any:
        if self.polarization != "helicity":
            return self.with_polarization("helicity")._combined(
                electric, magnetic, points
            )
        weights = self._weights(electric, magnetic, [-1.0, 1.0])
        wave = copy(self)
        wave.coefficients = self.coefficients * weights
        return wave.efield(points)


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
    def basis(self) -> PlaneWavePorts | None:
        """Fixed port basis, or None for ports following diffraction orders."""
        return self.ports.basis

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

    @classmethod
    def _from_basis(
        cls,
        coefficients: Any,
        *,
        basis: PlaneWavePorts,
        k0: Any,
        medium: Any,
        backend: Backend,
        positive: bool,
        polarization: str = "helicity",
    ) -> PortWave:
        """Construct a wave on fixed plane ports without a scattering system."""
        wave = cls.__new__(cls)
        wave.coefficients = backend.array(coefficients, complex_=True)
        if wave.coefficients.ndim not in (1, 2) or wave.coefficients.shape[0] != len(
            basis
        ):
            raise ValueError(
                "plane-port coefficients require shape (modes,) or (modes, illuminations)"
            )
        wave.k0, wave.medium = backend.array(k0), as_material(medium)
        wave._backend, wave.polarization = backend, polarization
        wave.ports, wave.positive = PortSet.from_basis(basis, backend), positive
        if polarization == "parity":
            wave.coefficients = backend.require_achiral(wave.coefficients, wave.medium)
        return wave

    @property
    def kind(self) -> str:
        """Propagation direction along the port normal."""
        return "up" if self.positive else "down"

    @property
    def modetype(self) -> str:
        """treams name of ``kind``."""
        return self.kind

    def _vectors(self) -> Any:
        """Wavevector of each port, one row each."""
        b = self._backend
        local = b.plane_vectors(
            self.ports.transverse_wavevectors[self.ports.groups],
            b.plane_ks(self.medium, self.k0)[self.ports.pols],
            positive=self.positive,
        )
        axis = ALIGNMENT_AXIS[self.ports.alignment]
        order = np.argsort([(axis + 1) % 3, (axis + 2) % 3, axis])
        return local[:, order]

    def _fixed_vectors(self, vectors: Any) -> bool:
        axis = ALIGNMENT_AXIS[self.ports.alignment]
        return bool(
            self.ports.fixed_q
            and np.all(vectors[:, [i for i in range(3) if i != axis]] == 0)
        )

    def in_basis(
        self,
        basis: Basis,
        *,
        positions: Any = None,
        singular: bool | None = None,
        kind: str | None = None,
    ) -> Wave:
        """Expand port amplitudes into regular multipoles, retaining their phases."""
        if singular or kind not in (None, "regular"):
            raise ValueError("plane illumination expands to regular multipoles")
        b = self._backend
        positions = b.positions(basis, positions)
        return _plane_expansion(
            self,
            basis,
            positions,
            self._vectors(),
            self.ports.pols,
            self._fixed_vectors,
        )

    @override
    def efield(self, points: Any = None, *, r: Any = None) -> Any:
        """Electric field at points (..., 3), differentiable in points and wave values."""
        points = one_of("points", points, "r", r, None)
        return _plane_efield(self, points, self.ports.pols, self._fixed_vectors)

    @override
    def _combined(self, electric: Any, magnetic: Any, points: Any) -> Any:
        if self.polarization != "helicity":
            return (
                self._complete_polarizations()
                .with_polarization("helicity")
                ._combined(electric, magnetic, points)
            )
        signs = 2 * self.ports.pols - 1
        # Derived port waves copy themselves: the constructor reads an SMatrix.
        wave = copy(self)
        weights = self._weights(electric, magnetic, signs)
        wave.coefficients = self.coefficients * (
            weights[:, None] if self.coefficients.ndim == 2 else weights
        )
        return wave.efield(points)

    def _complete_polarizations(self) -> PortWave:
        """Internal representation with zero amplitudes for missing partner ports."""

        wave = self
        modes = tuple(
            dict.fromkeys((group, p) for group, _ in self.ports.modes for p in (0, 1))
        )
        if len(modes) != len(self.ports.modes):
            # A missing parity amplitude is zero, but its partner is needed
            # to represent the magnetic field in the helicity convention.
            b = self._backend
            indices = {mode: i for i, mode in enumerate(self.ports.modes)}
            wave = copy(self)
            wave.ports = PortSet(
                None,
                modes,
                self.ports.alignment,
                self.ports.transverse_wavevectors,
                self.ports.fixed_q,
            )
            zero = np.zeros((1, *self.coefficients.shape[1:]))
            padded = b.concat((self.coefficients, b.array(zero, complex_=True)))
            wave.coefficients = padded[
                np.array([indices.get(mode, len(indices)) for mode in modes])
            ]
        return wave

    def with_polarization(self, polarization: str) -> PortWave:
        """Change the polarization convention with a fixed basis matrix."""
        wave = copy(self)
        wave.coefficients = self._backend.change_port_polarization(
            self.coefficients, self.ports.modes, self.polarization, polarization, (0,)
        )
        if polarization == "parity":
            wave.coefficients = self._backend.require_achiral(
                wave.coefficients, self.medium
            )
        wave.polarization = polarization
        return wave


def _plane_expansion(
    wave: PlaneWave | PortWave,
    basis: Basis,
    positions: Any,
    vectors: Any,
    pols: Sequence[int] | NDArray[np.int_],
    fixed_vectors: Callable[[Any], bool],
) -> Wave:
    """Separate angular coefficients from position-dependent translation phases."""
    b = wave._backend
    zero_basis = type(basis)(basis.modes, np.zeros_like(basis.positions))

    def map_context(context: Any, vectors: Any) -> DerivativeContext:
        return _select_input(context, 1, zero_basis.positions, vectors)

    @native_state(
        _native.PlaneExpansionContext,
        lambda inputs: (
            len(basis),
            len(zero_basis.positions),
            inputs[0].shape[0],
            isinstance(basis, CylindricalBasis),
        ),
        map_context=map_context,
    )
    def record(v: Any) -> Any:
        value, context = diff.plane_expansion(
            zero_basis,
            v,
            pols,
            poltype=wave.polarization,
            fixed_vectors=fixed_vectors(v),
        )
        return value, map_context(context, v)

    angular = b.apply(record, (len(basis), len(pols)), vectors)
    phases = b.apply(
        diff.plane_phases, (positions.shape[0], len(pols)), positions, vectors
    )
    operator = angular * phases[basis.pidx.copy()]
    return Wave(
        operator @ wave.coefficients,
        basis=basis,
        k0=wave.k0,
        medium=wave.medium,
        backend=b,
        positions=positions,
        polarization=wave.polarization,
    )


def _plane_operator(
    wave: PlaneWave | PortWave,
    vectors: Any,
    pols: Sequence[int] | NDArray[np.int_],
    fixed_vectors: Callable[[Any], bool],
) -> Any:
    """Electric polarization vectors at the origin, with their native derivatives."""

    def map_context(context: Any, vectors: Any) -> DerivativeContext:
        return _select_input(
            context, 2, np.empty(0, dtype=np.complex128), np.zeros((1, 3)), vectors
        )

    @native_state(
        _native.PlaneFieldContext,
        lambda inputs: (1, inputs[0].shape[0], False),
        map_context=map_context,
    )
    def record(v: Any) -> Any:
        value, context = diff.plane_field(
            None,
            np.zeros((1, 3)),
            v,
            pols,
            poltype=wave.polarization,
            fixed_vectors=fixed_vectors(v),
        )
        return value, map_context(context, v)

    return wave._backend.apply(record, (1, 3, len(pols)), vectors)[0]


def _plane_efield(
    wave: PlaneWave | PortWave,
    points: Any,
    pols: Sequence[int] | NDArray[np.int_],
    fixed_vectors: Callable[[Any], bool],
) -> Any:
    """Electric field of plane-wave amplitudes, one native field per wavevector."""
    b = wave._backend
    points = b.array(points)
    shape = tuple(points.shape)
    vectors = wave._vectors()
    electric = _plane_operator(wave, vectors, pols, fixed_vectors)
    phases = b.apply(
        diff.plane_phases,
        (int(np.prod(shape[:-1])), len(pols)),
        points.reshape(-1, 3),
        vectors,
    )
    if wave.coefficients.ndim == 2:
        operator = phases[:, None, :] * electric[None, :, :]
        return (operator @ wave.coefficients).reshape(
            (*shape, wave.coefficients.shape[1])
        )
    return ((phases * wave.coefficients) @ electric.T).reshape(shape)

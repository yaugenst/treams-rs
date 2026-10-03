"""Finite spherical/cylindrical responses with explicit physical operations."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Self, cast, override

import numpy as np

from . import _operator_objects as op
from . import diff
from ._bases import CylindricalBasis, SphericalBasis
from ._material import Material, MaterialLike
from ._operator_objects import Operator
from ._operators import expandlattice, translate
from ._polarization import (
    change_polarization,
    check_poltype_medium,
    resolve_poltype,
    target_poltype,
)
from ._results import CrossSections
from ._upstream import UpstreamMembers
from ._validation import check_k0, one_of
from ._waves import PlaneWave, Wave, check_compatible

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from . import _native

__all__ = [
    "CylindricalTMatrix",
    "TMatrix",
    "axis_mask",
    "cylinder_tmatrix",
    "interaction_coupling",
    "multilayer_cylinder_tmatrix",
    "multilayer_sphere_tmatrix",
    "solve_columns",
    "sphere_tmatrix",
]


class _TMatrix[B: (SphericalBasis, CylindricalBasis)](UpstreamMembers):
    """A matrix with physical metadata; use ``.array`` for arbitrary NumPy operations.

    Unlike an ndarray subclass, slicing or arithmetic never passes physical
    metadata on to a result with a different meaning.
    """

    _basis_type: type[B]
    medium: Material
    """Exterior propagation medium."""
    polarization: str
    """Polarization channel convention."""

    def __init__(
        self,
        arr: ArrayLike,
        *,
        k0: float,
        basis: B | None = None,
        medium: MaterialLike | None = None,
        polarization: str | None = None,
        material: MaterialLike | None = None,
        poltype: str | None = None,
    ):
        """Copy a finite square matrix mapping regular to outgoing coefficients.

        Args:
            arr: shape (modes, modes), outgoing rows and incident columns.
                ``.array`` holds an owned, read-only complex128 copy.
            k0: positive vacuum angular wavenumber, in inverse units of the
                basis positions.
            basis: modes of the rows and columns, of the matrix's wave family.
                Omitted, it is the complete basis at one position for this
                dimension (with kz=0 for cylinders).
            medium: the embedding medium, vacuum by default.
            polarization: "helicity" (default) or "parity". A chiral medium
                needs "helicity".
            material: treams name of ``medium``.
            poltype: treams name of ``polarization``.

        Give each quantity under one name.
        """
        # treams keywords
        medium = one_of("medium", medium, "material", material, 1)
        polarization = one_of("polarization", polarization, "poltype", poltype, None)
        self._setup(
            np.array(arr, dtype=np.complex128, copy=True),
            k0=k0,
            basis=basis,
            material=medium,
            poltype=polarization,
        )

    # Package-internal protocol of the response objects. _adopt wraps a freshly
    # computed array without copying it; _incident turns a source into incident
    # coefficients in this basis, through the _expanded method of Wave and
    # PlaneWave; _outgoing wraps scattered coefficients as a singular Wave.
    # Cluster, ScatteringFactor and PeriodicResponse call them on their local
    # matrix, and SMatrix._incident uses _expanded the same way.

    @classmethod
    def _adopt(
        cls,
        array: NDArray[np.complex128],
        *,
        k0: float,
        basis: B | None = None,
        material: MaterialLike = 1,
        poltype: str | None = None,
    ) -> Self:
        """Wrap a freshly computed internal array without copying it.

        The array must not be referenced elsewhere; it becomes read-only.
        Validation matches the public constructor.
        """
        result = cls.__new__(cls)
        result._setup(
            np.asarray(array, dtype=np.complex128),
            k0=k0,
            basis=basis,
            material=material,
            poltype=poltype,
        )
        return result

    def _setup(
        self,
        array: NDArray[np.complex128],
        *,
        k0: float,
        basis: B | None,
        material: MaterialLike,
        poltype: str | None,
    ) -> None:
        poltype = resolve_poltype(poltype)
        if (
            array.ndim != 2
            or array.shape[0] != array.shape[1]
            or not np.isfinite(array).all()
        ):
            raise ValueError("T-matrix must be a finite square matrix")
        self.k0 = check_k0(k0)
        self.basis: B = self._default_basis(len(array)) if basis is None else basis
        if not isinstance(self.basis, self._basis_type):
            raise ValueError("basis wave family does not match T-matrix type")
        if len(self.basis) != len(array):
            raise ValueError("basis dimension does not match matrix")
        self.medium = Material(material)
        self.polarization = check_poltype_medium(poltype, self.medium)
        array.flags.writeable = False
        self.array: NDArray[np.complex128] = array

    def _default_basis(self, dimension: int) -> B:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape (modes, modes) of the matrix."""
        return self.array.shape

    @property
    def ks(self) -> NDArray[np.complex128]:
        """Embedding wavenumbers for labels (0, 1), shape (2,), in inverse length."""
        return self.medium.ks(self.k0)

    @property
    def isglobal(self) -> bool:
        """Whether all multipoles of the basis share one position."""
        return self.basis.isglobal

    def __len__(self) -> int:
        return len(self.array)

    def scatter(self, incident: ArrayLike | PlaneWave | Wave) -> Wave:
        """Apply this solved response and retain outgoing-wave metadata.

        Incident is a physical source, a (modes,) coefficient vector or a
        (modes, illuminations) coefficient batch. Raw coefficients are regular
        incident channels in this matrix's basis.
        """
        return self._outgoing(self.array @ self._incident(incident))

    def _outgoing(self, coefficients: NDArray[np.complex128]) -> Wave:
        """Singular (outgoing) wave with this response's basis, k0 and medium."""
        return Wave(
            coefficients,
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            kind="singular",
            polarization=self.polarization,
        )

    def select(self, basis: B) -> Self:
        """Select the same subset of incoming and outgoing modes."""
        if not isinstance(basis, self._basis_type) or not np.array_equal(
            basis.positions, self.basis.positions
        ):
            raise ValueError("selection must use the same wave family and positions")
        return self[basis]

    def in_basis(self, basis: B) -> Self:
        """Represent the same response in another multipole basis."""
        outgoing, _ = diff.expansion(
            basis, self.basis, self.ks, poltype=self.polarization
        )
        incident, _ = diff.expansion(
            self.basis, basis, self.ks, poltype=self.polarization
        )
        return type(self)._adopt(
            outgoing @ self.array @ incident,
            k0=self.k0,
            basis=basis,
            material=self.medium,
            poltype=self.polarization,
        )

    def with_polarization(self, polarization: str) -> Self:
        """Represent the same response in helicity or parity channels.

        Every mode needs its partner of opposite polarization in the basis.
        """
        return self._with_polarization_target(polarization)

    def _with_polarization_target(self, target: str | None) -> Self:
        """The response in the named convention, or the other one for None."""
        target = target_poltype(self.polarization, target)
        if target is None:
            return self
        return type(self)._adopt(
            change_polarization(self.array, self.basis, (0, 1)),
            k0=self.k0,
            basis=self.basis,
            material=self.medium,
            poltype=target,
        )

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, self._basis_type):
            lookup = cast("dict[object, int]", self.basis._lookup)
            missing = [mode for mode in key if mode not in lookup]
            if missing:
                raise ValueError(f"{missing[0]!r} is not in the basis")
            indices = [lookup[mode] for mode in key]
            return type(self)._adopt(
                self.array[np.ix_(indices, indices)],
                basis=cast("B", self.basis[indices]),
                k0=self.k0,
                material=self.medium,
                poltype=self.polarization,
            )
        return self.array[key]

    def valid_points(self, grid: ArrayLike, radii: ArrayLike) -> NDArray[np.bool_]:
        """Points strictly outside the enclosing spheres or infinite cylinders."""
        points, radii = np.asarray(grid, dtype=float), np.asarray(radii, dtype=float)
        dim = 2 if isinstance(self.basis, CylindricalBasis) else 3
        if (
            points.ndim == 0
            or points.shape[-1] not in (dim, 3)
            or not np.isfinite(points).all()
        ):
            raise ValueError("grid requires finite Cartesian points")
        if (
            radii.shape != (len(self.basis.positions),)
            or not np.isfinite(radii).all()
            or np.any(radii < 0)
        ):
            raise ValueError(
                "one finite nonnegative radius required per basis position"
            )
        valid = np.ones(points.shape[:-1], dtype=bool)
        for radius, position in zip(radii, self.basis.positions, strict=True):
            valid &= (
                np.sum((points[..., :dim] - position[:dim]) ** 2, axis=-1) > radius**2
            )
        return valid

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        return np.asarray(self.array, dtype=dtype, copy=copy)

    def __matmul__(
        self, other: ArrayLike | PlaneWave | Wave | Operator
    ) -> NDArray[np.complex128]:
        if isinstance(other, Operator):
            return NotImplemented
        return self.array @ self._incident(other)

    def _incident(self, value: ArrayLike | PlaneWave | Wave) -> NDArray[np.complex128]:
        """Regular incident coefficients in this basis; arrays pass through."""
        if isinstance(value, (PlaneWave, Wave)):
            check_compatible(
                (value.k0, value.medium, value.polarization),
                (self.k0, self.medium, self.polarization),
                "illumination and T-matrix",
            )
            return value._expanded(self.basis)
        return np.asarray(value, dtype=np.complex128)

    def rotate(self, phi: float, theta: float = 0, psi: float = 0) -> Self:
        """Rotate the local multipole channels about their fixed positions.

        The angles are z-y-z Euler angles in radians; cylindrical bases allow
        only theta=0.
        """
        rotation, _ = diff.rotation([phi, theta, psi], self.basis)
        # D(-psi, -theta, -phi) = D(phi, theta, psi)^H entrywise, on any basis.
        return type(self)._adopt(
            rotation @ self.array @ rotation.conj().T,
            k0=self.k0,
            basis=self.basis,
            material=self.medium,
            poltype=self.polarization,
        )

    def translate(self, r: ArrayLike) -> Self:
        """Represent the same response about expansion positions shifted by r.

        The displacement r has shape (3,), in units inverse to k0. Shifting the
        positions by r equals moving the scatterer by -r. The result is
        ``operators.translate(r) @ T @ operators.translate(-r)`` in this basis.
        """
        offset = np.asarray(r, dtype=np.float64)
        if offset.shape != (3,):
            raise ValueError("T-matrix translation requires one Cartesian displacement")
        common = {
            "basis": self.basis,
            "k0": self.k0,
            "material": self.medium,
            "poltype": self.polarization,
        }
        return type(self)._adopt(
            translate(offset, **common) @ self.array @ translate(-offset, **common),
            k0=self.k0,
            basis=self.basis,
            material=self.medium,
            poltype=self.polarization,
        )

    def _propagating_ks(self) -> NDArray[np.float64]:
        ks = self.ks
        if not self.medium.isreal or np.any(ks.imag != 0) or np.any(ks.real == 0):
            raise NotImplementedError(
                "cross sections require a nonabsorbing propagating embedding medium"
            )
        return ks.real[self.basis.pol]

    def _cross_section_weights(self, k_exponent: int) -> NDArray[np.float64]:
        ks = self._propagating_ks()
        weights = 1 / ks**k_exponent
        if isinstance(self.basis, CylindricalBasis):
            if np.any(abs(self.basis.kz) == abs(ks)):
                raise ValueError("cross widths are undefined at a diffraction cutoff")
            weights[abs(self.basis.kz) > abs(ks)] = 0
        return weights

    @cached_property
    def _radiation_overlap(self) -> NDArray[np.complex128]:
        """Regular expansion between all position pairs, reused by cross sections."""
        overlap, _ = diff.expansion(
            self.basis, self.basis, self.ks, poltype=self.polarization
        )
        overlap.flags.writeable = False
        return overlap

    def _cross_sections(
        self,
        illu: ArrayLike | PlaneWave | Wave,
        flux: float,
        k_exponent: int,
        factor: float,
    ) -> tuple[float, float]:
        """(scattering, extinction) of one illumination, weighting modes by k**-k_exponent.

        k_exponent is 2 for cross sections (areas) and 1 for cross widths (lengths).
        """
        incident = self._incident(illu)
        if (
            incident.shape != (len(self),)
            or not np.isfinite(incident).all()
            or not np.isfinite(flux)
            or flux <= 0
        ):
            raise ValueError("require finite incident coefficients and positive flux")
        weights = self._cross_section_weights(k_exponent)
        if np.any(incident[weights == 0] != 0):
            raise ValueError("cross widths require propagating incident modes")
        scattered = self @ incident
        weighted = scattered * weights
        radiated = weighted if self.isglobal else self._radiation_overlap @ weighted
        return float(np.vdot(scattered, radiated).real * factor / flux), float(
            -np.vdot(incident, weighted).real * factor / flux
        )

    # treams-compatible names (see treams_rs._upstream.UPSTREAM_MEMBERS)

    @property
    def material(self) -> Material:
        """treams name of ``medium``."""
        return self.medium

    @property
    def poltype(self) -> str:
        """treams name of ``polarization``."""
        return self.polarization

    modetype = ("singular", "regular")
    """treams annotation: singular (outgoing) rows, regular incident columns."""

    def changepoltype(self, poltype: str | None = None) -> Self:
        """treams name of ``with_polarization``; omitting poltype switches convention."""
        return self._with_polarization_target(poltype)

    expand = in_basis

    @property
    def interaction(self) -> _Interaction[Self]:
        """Finite-cluster coupling with solve(), factor() and illuminate() methods.

        Calling the object returns I-T*C. Solve builds the full coupled T-matrix;
        illuminate solves only supplied incident columns; factor keeps the LU
        factors for repeated calls. All use the current local matrix and basis
        positions.
        """
        return _Interaction(self)

    @property
    def latticeinteraction(self) -> _PeriodicInteraction[Self]:
        """Periodic solve/factor/illuminate interface taking lattice and Bloch vector.

        Returned responses are local periodic coefficients; isolated-particle
        cross-section formulas do not apply to them.
        """
        return _PeriodicInteraction(self)

    expandlattice = op.OperatorAttribute(op.ExpandLattice)
    """treams operator: ``tm.expandlattice(lattice=..., kpar=...)`` returns the ndarray
    ``operators.expandlattice(lattice, kpar, basis=tm.basis, ...) @ tm.array``."""

    def permute(self, n: int = 1) -> Self:
        """Unsupported: cyclic coordinate permutations act on plane-wave ports.

        Always raises TypeError; permute an ``SMatrix`` or plane-wave basis instead.
        """
        raise TypeError(
            "coordinate permutations apply to plane-wave bases; use SMatrix.permute"
        )


def solve_columns(
    factor: _native.InteractionFactor, values: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Solve one incident vector or a (modes, illuminations) batch with a factor."""
    vector = values.ndim == 1
    result = factor.solve(values[:, None] if vector else values)
    return result[:, 0] if vector else result


def axis_mask(
    cylindrical: CylindricalBasis, spherical: SphericalBasis
) -> NDArray[np.bool_] | None:
    """Pair each cylinder axis with its own particle, shape (cylindrical, spherical).

    Returns None for a single axis, which collects the field of every particle.
    """
    if len(cylindrical.positions) == 1:
        return None
    if len(cylindrical.positions) != len(spherical.positions):
        raise ValueError(
            "a cylindrical basis with several axes needs one axis per particle position"
        )
    return cylindrical.pidx[:, None] == spherical.pidx


def interaction_coupling(tm: _TMatrix[Any]) -> NDArray[np.complex128]:
    """Coupling matrix C of a local basis.

    C expands the singular waves leaving each position in regular waves at every
    other position.
    """
    return diff.expansion(
        tm.basis, tm.basis, tm.ks, poltype=tm.polarization, singular=True
    )[0]


class _Interaction[M: _TMatrix[Any]]:
    def __init__(self, matrix: M):
        self.matrix = matrix

    def __call__(self) -> NDArray[np.complex128]:
        return np.eye(
            len(self.matrix), dtype=np.complex128
        ) - self.matrix.array @ interaction_coupling(self.matrix)

    def solve(self) -> M:
        """Return the full coupled matrix ``(I - T*C)^-1 T`` in the local basis."""
        tm = self.matrix
        result, _ = diff.interaction(tm.array, interaction_coupling(tm))
        return tm._adopt(
            result,
            k0=tm.k0,
            basis=tm.basis,
            material=tm.medium,
            poltype=tm.polarization,
        )

    def factor(self) -> _native.InteractionFactor:
        """Factor once for repeated incident-field solves without a full response matrix.

        The returned factor's solve/record methods accept complex128 arrays with
        one incident illumination per column.
        """
        tm = self.matrix
        return diff.factor_interaction(tm.array, interaction_coupling(tm))

    def illuminate(
        self, incident: ArrayLike | PlaneWave | Wave
    ) -> NDArray[np.complex128]:
        """Solve ``(I-T*C) scattered = T*incident`` for requested illuminations.

        Incident may be a matching PlaneWave/Wave, a shape-(modes,)
        coefficient vector or an array (modes, illuminations). Returns complex128
        scattered coefficients with the same vector/batch rank. A fresh factor
        is built per call; use ``factor().solve`` for repeated right-hand sides.
        """
        return solve_columns(self.factor(), self.matrix._incident(incident))


class _PeriodicInteraction[M: _TMatrix[Any]]:
    def __init__(self, matrix: M):
        self.matrix = matrix

    def coupling(
        self, lattice: ArrayLike, kpar: ArrayLike, *, eta: complex = 0
    ) -> NDArray[np.complex128]:
        """Return the coupling matrix C of the lattice.

        C expands the waves scattered by every other lattice site in regular
        waves at this site. ``lattice`` holds the lattice vectors and ``kpar`` the
        Bloch vector. ``eta`` sets where the Ewald method splits each lattice
        sum into a real-space and a reciprocal-space part; 0 picks it
        automatically.
        """
        tm = self.matrix
        return diff.lattice_expansion(
            tm.basis, tm.basis, tm.ks, kpar, lattice, poltype=tm.polarization, eta=eta
        )[0]

    def __call__(
        self, lattice: ArrayLike, kpar: ArrayLike, *, eta: complex = 0
    ) -> NDArray[np.complex128]:
        return np.eye(
            len(self.matrix), dtype=np.complex128
        ) - self.matrix.array @ self.coupling(lattice, kpar, eta=eta)

    def solve(
        self, lattice: ArrayLike, kpar: ArrayLike, *, eta: complex = 0
    ) -> NDArray[np.complex128]:
        """Effective periodic response in local multipole channels.

        This is not an isolated-particle T-matrix; finite-particle cross-section
        formulae do not apply. Use its array with incident channel coefficients.
        """
        return diff.interaction(
            self.matrix.array, self.coupling(lattice, kpar, eta=eta)
        )[0]

    def factor(
        self, lattice: ArrayLike, kpar: ArrayLike, *, eta: complex = 0
    ) -> _native.InteractionFactor:
        """Reusable periodic factor for requested incident channel coefficients."""
        return diff.factor_interaction(
            self.matrix.array, self.coupling(lattice, kpar, eta=eta)
        )

    def illuminate(
        self,
        incident: ArrayLike | PlaneWave | Wave,
        lattice: ArrayLike,
        kpar: ArrayLike,
        *,
        eta: complex = 0,
    ) -> NDArray[np.complex128]:
        """Compute requested periodic response columns without the full effective T matrix."""
        return solve_columns(
            self.factor(lattice, kpar, eta=eta), self.matrix._incident(incident)
        )


class TMatrix(_TMatrix[SphericalBasis]):
    """Spherical T-matrix: regular incident coefficients to singular (outgoing) ones.

    The treams counterpart is ``treams.TMatrix``. Build one with
    ``sphere_tmatrix``, ``multilayer_sphere_tmatrix`` or ``Cluster.solve``, or
    from a square array. Rows and columns follow ``basis``, a SphericalBasis.
    ``scatter(incident)`` returns the scattered Wave; ``array`` and ``@`` give
    the coefficients. For gradients use ``treams_rs.advect``, ``jax`` or
    ``torch``.

    The treams-rs names come first. The treams names (``xs``, ``cd``,
    ``poltype``, ``changepoltype``, ...) follow at the end of the class and
    call them.

    Treat it as immutable: the constructor checks the metadata attributes once
    and never again.
    """

    _basis_type = SphericalBasis

    def cross_sections(
        self, incident: ArrayLike | PlaneWave | Wave, *, flux: float = 0.5
    ) -> CrossSections[float]:
        """Named scattering/extinction/absorption cross sections in squared length units.

        Incident is a matching PlaneWave or Wave, or a (modes,) regular
        coefficient vector. Flux is its finite positive incident flux in the
        field normalization. The exterior must be nonabsorbing with propagating
        helicity wavenumbers. A solved finite cluster may keep several
        positions.
        """
        return CrossSections(*self._cross_sections(incident, flux, 2, 0.5))

    @property
    def average_cross_sections(self) -> CrossSections[float]:
        """Rotationally and polarization-averaged cross sections in squared length units.

        Requires one position and a nonabsorbing propagating exterior.
        Expand a solved cluster to a global basis before taking this average.
        """
        return CrossSections(self._average_scattering(), self._average_extinction())

    @property
    def circular_dichroism(self) -> float:
        """Rotationally averaged absorption circular dichroism."""
        return self._metric("cd")

    @property
    def duality_breaking(self) -> float:
        """Fraction of scattering that changes helicity."""
        return self._metric("db")

    @property
    def electromagnetic_chirality(self) -> float:
        """Normalized electromagnetic chirality from helicity-block singular values."""
        return self._metric("chi")

    def _metric(self, kind: str) -> float:
        if not self.isglobal or self.polarization != "helicity":
            raise NotImplementedError("metric requires a global helicity T-matrix")
        ks = np.ones(2)
        if kind == "cd":
            self._propagating_ks()
            ks = self.ks.real
        return diff.tmatrix_metric(
            self.array, ks, polarizations=self.basis.pol, metric=kind
        )[0]

    def _average_extinction(self) -> float:
        if not self.isglobal:
            raise NotImplementedError("expand to a global basis before averaging")
        ks = self._propagating_ks()
        return float(-2 * np.pi * np.sum(np.diag(self.array).real / ks**2))

    def _average_scattering(self) -> float:
        if not self.isglobal:
            raise NotImplementedError("expand to a global basis before averaging")
        ks = self._propagating_ks()
        return float(2 * np.pi * np.sum(abs(self.array / ks[:, None]) ** 2))

    @override
    def _default_basis(self, dimension: int) -> SphericalBasis:
        return SphericalBasis.default(SphericalBasis.defaultlmax(dimension))

    # treams-compatible names (see treams_rs._upstream.UPSTREAM_MEMBERS)

    def xs(self, illu: ArrayLike | PlaneWave, flux: float = 0.5) -> tuple[float, float]:
        """treams name of ``cross_sections``, returning (scattering, extinction)."""
        return self._cross_sections(illu, flux, 2, 0.5)

    @property
    def xs_ext_avg(self) -> float:
        """treams name of ``average_cross_sections.extinction``."""
        return self._average_extinction()

    @property
    def xs_sca_avg(self) -> float:
        """treams name of ``average_cross_sections.scattering``."""
        return self._average_scattering()

    cd = circular_dichroism
    db = duality_breaking
    chi = electromagnetic_chirality

    @classmethod
    def sphere(
        cls,
        lmax: int,
        k0: float,
        radii: ArrayLike,
        materials: Sequence[MaterialLike],
        poltype: str | None = None,
    ) -> TMatrix:
        """Construct a concentric multilayer, optionally chiral sphere at the origin.

        Lmax is the maximum spherical degree (1..128); k0 is the positive vacuum
        angular wavenumber. Radii is a positive scalar or strictly increasing
        one-dimensional sequence of layer boundaries, in units inverse to k0.
        Materials contains one entry per layer plus the exterior medium, ordered
        from the center outward. Each entry follows the Material constructor.

        Returns a matrix with dimension ``2*lmax*(lmax+2)`` and default spherical
        ordering (l, m, pol=(1, 0)). Poltype selects helicity/parity, defaulting to
        helicity; parity requires an achiral exterior. ``diff.sphere`` returns
        the same matrix in helicity with a context for its gradients.
        ``sphere_tmatrix`` and ``multilayer_sphere_tmatrix`` take keywords instead.
        """
        poltype = resolve_poltype(poltype)
        layers = [Material(m) for m in materials]
        if not layers:
            raise ValueError("sphere requires layer materials and an embedding medium")
        value, _ = diff.sphere(
            lmax,
            k0,
            np.atleast_1d(radii),
            [m.epsilon for m in layers],
            [m.mu for m in layers],
            [m.kappa for m in layers],
        )
        result = cls._adopt(value, k0=k0, material=layers[-1], poltype="helicity")
        return result if poltype == "helicity" else result.with_polarization(poltype)


class CylindricalTMatrix(_TMatrix[CylindricalBasis]):
    """Cylindrical T-matrix: regular incident coefficients to singular (outgoing) ones.

    The treams counterpart is ``treams.TMatrixC``. Rows and columns follow
    ``basis``, a CylindricalBasis with fixed real axial wavenumbers kz. Use
    ``cylinder_tmatrix`` for concentric infinite cylinders and ``Cluster`` for
    parallel cylinders. Cross widths have length units. ``scatter(incident)``
    returns the scattered Wave; ``array`` and ``@`` give the coefficients. For
    gradients use ``treams_rs.advect``, ``jax`` or ``torch``.

    The treams-rs names come first. The treams names (``xw``, ``poltype``,
    ``changepoltype``, ...) follow at the end of the class and call them.

    Treat it as immutable: the constructor checks the metadata attributes once
    and never again.
    """

    _basis_type = CylindricalBasis

    @classmethod
    def _from_response(
        cls,
        local: TMatrix | CylindricalTMatrix,
        basis: CylindricalBasis,
        lattice: ArrayLike,
        kpar: ArrayLike,
        response: Callable[[], NDArray[np.complex128]],
    ) -> CylindricalTMatrix:
        """Represent a periodic response of a spherical cell in cylindrical modes.

        Part of the package-internal protocol: PeriodicResponse.to_cylindrical
        calls it. ``response()`` supplies the solved array once the axial
        orders are valid.
        """
        if not isinstance(local, TMatrix):
            raise ValueError("cylindrical conversion requires a spherical unit cell")
        outgoing = expandlattice(
            lattice,
            kpar,
            basis=(basis, local.basis),
            k0=local.k0,
            material=local.medium,
            poltype=local.polarization,
        )
        incident = diff.expansion(
            local.basis, basis, local.ks, poltype=local.polarization
        )[0]
        mask = axis_mask(basis, local.basis)
        if mask is not None:
            # Each axis already holds the whole field, so it pairs only with its own particle.
            outgoing = outgoing * mask
            incident = incident * mask.T
        return cls._adopt(
            outgoing @ response() @ incident,
            k0=local.k0,
            basis=basis,
            material=local.medium,
            poltype=local.polarization,
        )

    @override
    def _default_basis(self, dimension: int) -> CylindricalBasis:
        return CylindricalBasis.default([0], CylindricalBasis.defaultmmax(dimension))

    @property
    def krhos(self) -> NDArray[np.complex128]:
        """Per-mode radial wavenumbers, shape (modes,), with nonnegative imaginary part."""
        return self.medium.krhos(self.k0, self.basis.kz, self.basis.pol)

    def cross_widths(
        self, incident: ArrayLike | PlaneWave | Wave, *, flux: float = 0.5
    ) -> CrossSections[float]:
        """Named scattering/extinction/absorption widths, in length units.

        Evanescent outgoing orders stay in the solution but carry no far-field
        power. Evanescent illumination has no incident far-field flux and raises
        ValueError.
        """
        return CrossSections(*self._cross_sections(incident, flux, 1, 2.0))

    def cross_sections(
        self, incident: ArrayLike | PlaneWave | Wave, *, flux: float = 0.5
    ) -> CrossSections[float]:
        """Cross widths, in length units, for cylinders; cross_widths is the explicit name."""
        return CrossSections(*self._cross_sections(incident, flux, 1, 2.0))

    @property
    def average_cross_widths(self) -> CrossSections[float]:
        """Mean cross widths over azimuth and propagating axial/polarization channels.

        Scattering is the mean radiated width over the same incoming ensemble.
        """
        return CrossSections(self._average_scattering(), self._average_extinction())

    def _averaging_weights(self) -> NDArray[np.float64]:
        if not self.isglobal:
            raise NotImplementedError("expand to a global basis before averaging")
        weights = self._cross_section_weights(1)
        channels = np.column_stack((self.basis.kz, self.basis.pol))[weights != 0]
        count = len(np.unique(channels, axis=0))
        if count == 0:
            raise ValueError("cross-width average requires propagating incident modes")
        return weights * (4 / count)

    def _average_extinction(self) -> float:
        return float(-np.sum(np.diag(self.array).real * self._averaging_weights()))

    def _average_scattering(self) -> float:
        weights = self._averaging_weights()
        return float(np.sum(abs(self.array[:, weights != 0]) ** 2 * weights[:, None]))

    # treams-compatible names (see treams_rs._upstream.UPSTREAM_MEMBERS)

    def xw(self, illu: ArrayLike | PlaneWave, flux: float = 0.5) -> tuple[float, float]:
        """treams name of ``cross_widths``, returning (scattering, extinction)."""
        return self._cross_sections(illu, flux, 1, 2.0)

    @property
    def xw_ext_avg(self) -> float:
        """treams name of ``average_cross_widths.extinction``."""
        return self._average_extinction()

    @property
    def xw_sca_avg(self) -> float:
        """treams name of ``average_cross_widths.scattering``."""
        return self._average_scattering()

    @classmethod
    def cylinder(
        cls,
        kzs: ArrayLike,
        mmax: int,
        k0: float,
        radii: ArrayLike,
        materials: Sequence[MaterialLike],
        poltype: str | None = None,
    ) -> CylindricalTMatrix:
        """Construct concentric infinite cylinders around the z axis at the origin.

        Kzs is a scalar or nonempty one-dimensional array of distinct finite real
        axial wavenumbers. Mmax (0..128) includes orders -mmax..mmax. K0 is the
        positive vacuum angular wavenumber, in the same inverse length units as
        kzs. Radii is a positive scalar or strictly increasing sequence of layer
        boundaries. Materials lists layers from the axis outward, followed by
        one exterior medium; entries follow the Material constructor.

        For nkz supplied axial values, the dimension is ``2*nkz*(2*mmax+1)``,
        ordered by kz input order, then m and pol=(1, 0). Poltype selects
        helicity/parity, defaulting to helicity; parity requires an
        achiral exterior. ``diff.cylinder`` returns the same matrix in helicity
        with a context for its gradients. ``cylinder_tmatrix`` and
        ``multilayer_cylinder_tmatrix`` take keywords instead.
        """
        poltype = resolve_poltype(poltype)
        layers = [Material(m) for m in materials]
        if not layers:
            raise ValueError(
                "cylinder requires layer materials and an embedding medium"
            )
        axial = np.atleast_1d(kzs)
        value, _ = diff.cylinder(
            axial,
            mmax,
            k0,
            np.atleast_1d(radii),
            [m.epsilon for m in layers],
            [m.mu for m in layers],
            [m.kappa for m in layers],
        )
        result = cls._adopt(
            value,
            k0=k0,
            basis=CylindricalBasis.default(axial, mmax),
            material=layers[-1],
            poltype="helicity",
        )
        return result if poltype == "helicity" else result.with_polarization(poltype)


def sphere_tmatrix(
    *,
    k0: float,
    lmax: int,
    radius: float,
    material: MaterialLike,
    medium: MaterialLike = 1,
    polarization: str = "helicity",
) -> TMatrix:
    """Homogeneous sphere response, ready for illumination or cluster assembly.

    ``k0=2*pi/vacuum_wavelength``; radius and 1/k0 share a length unit.
    ``material`` is a permittivity or Material(epsilon, mu, kappa); ``medium``
    is the exterior, vacuum by default. lmax is the fixed multipole cutoff.
    Use ``response.cross_sections(plane_wave(...))`` for named scattering,
    extinction and absorption areas, ``response.average_cross_sections`` for
    rotational/polarization averages, or ``response.scatter(wave).efield(xyz)``
    for scattered fields. Use treams_rs.advect/jax/torch for traced parameters.
    """
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
) -> CylindricalTMatrix:
    """Homogeneous infinite z cylinder; kz is one or more real axial wavenumbers."""
    return CylindricalTMatrix.cylinder(
        kz, mmax, k0, radius, [material, medium], polarization
    )


def multilayer_cylinder_tmatrix(
    *,
    k0: float,
    kz: ArrayLike,
    mmax: int,
    radii: ArrayLike,
    materials: Sequence[MaterialLike],
    medium: MaterialLike = 1,
    polarization: str = "helicity",
) -> CylindricalTMatrix:
    """Concentric infinite cylinders; materials run outward, excluding the medium."""
    return CylindricalTMatrix.cylinder(
        kz, mmax, k0, radii, [*materials, medium], polarization
    )

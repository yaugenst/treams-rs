"""Wave objects: multipole and plane-port amplitudes, and incident plane waves.

Each wave carries its basis, medium and polarization convention, and samples
fields with the weighted native kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import numpy as np

from . import _native, diff
from ._bases import (
    CylindricalBasis,
    PlaneWaveBasis,
    PlaneWavePorts,
    SphericalBasis,
)
from ._dispatch import backend_for, namespace
from ._fields import WaveFields
from ._material import Material, MaterialLike
from ._operators import expand
from ._polarization import (
    change_polarization,
    check_poltype_medium,
    resolve_poltype,
    target_poltype,
)
from ._upstream import UpstreamMembers
from ._validation import check_k0, check_kind, one_of, unit_vectors

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NDArray

__all__ = [
    "PlaneWave",
    "Wave",
    "check_compatible",
    "cylindrical_wave",
    "plane_wave",
    "plane_wave_angle",
    "propagation_side",
    "spherical_wave",
]

type Basis = SphericalBasis | CylindricalBasis | PlaneWavePorts | PlaneWaveBasis


def propagation_side(kvec: NDArray[np.complex128], axis: int) -> str:
    """Direction "up" or "down" of a wave vector along the Cartesian ``axis``.

    The wave travels down when that component has a negative imaginary part,
    or is real and negative; otherwise it travels up, toward the positive side.
    """
    normal = kvec[axis]
    return "down" if normal.imag < 0 or (normal.imag == 0 and normal.real < 0) else "up"


def check_compatible(
    a: tuple[float, Material, str], b: tuple[float, Material, str], what: str
) -> None:
    """Raise a ValueError naming ``what`` unless both (k0, medium, polarization) match."""
    if a != b:
        raise ValueError(
            f"{what} must have matching k0, medium and polarization convention"
        )


class Wave(WaveFields, UpstreamMembers):
    """Physical wave amplitudes with explicit basis, medium and kind.

    The treams counterpart is the PhysicsArray that ``treams.spherical_wave``
    and the other wave functions return. ``kind`` is the radial kind of
    multipoles (regular or singular) or the direction of plane-wave ports (up
    or down). Fields use weighted Rust kernels and never form the full
    sample-by-mode operator. Use ``coefficients`` for numerical work;
    framework-valued inputs retain their gradients. A batch keeps one
    illumination per column.

    The treams-rs names come first. The treams names (``material``,
    ``poltype``, ``modetype``, ``expand``, ...) follow at the end of the class
    and call them.

    Treat it as immutable: the constructor checks the metadata attributes once
    and never again.
    """

    basis: Basis
    array: NDArray[np.complex128]
    medium: Material
    """Homogeneous medium in which this wave is represented."""
    polarization: str
    """Polarization convention: helicity or parity."""
    kind: str
    """Radial kind (regular or singular) of multipoles, or port direction (up or down)."""

    def __new__(cls, coefficients: Any = None, **kwargs: Any) -> Any:
        backend = backend_for(coefficients, kwargs)
        if backend is None:
            return super().__new__(cls)
        medium = one_of(
            "medium",
            kwargs.pop("medium", None),
            "material",
            kwargs.pop("material", None),
            1,
        )
        kind = check_kind(
            one_of(
                "kind",
                kwargs.pop("kind", None),
                "modetype",
                kwargs.pop("modetype", None),
                "regular",
            )
        )
        polarization = resolve_poltype(
            one_of(
                "polarization",
                kwargs.pop("polarization", None),
                "poltype",
                kwargs.pop("poltype", None),
                None,
            )
        )
        basis = kwargs["basis"]
        if isinstance(basis, PlaneWavePorts):
            from ._framework_waves import PortWave

            if kind not in ("up", "down"):
                raise ValueError("plane-port wave kind must be 'up' or 'down'")
            return PortWave._from_basis(
                coefficients,
                backend=backend,
                medium=medium,
                positive=kind == "up",
                polarization=polarization,
                **kwargs,
            )
        if not isinstance(basis, (SphericalBasis, CylindricalBasis)):
            raise NotImplementedError(
                "autodiff Wave construction requires a multipole basis or PlaneWavePorts"
            )
        return namespace(backend).wave(
            coefficients,
            medium=medium,
            kind=kind,
            polarization=polarization,
            **kwargs,
        )

    def __init__(
        self,
        coefficients: ArrayLike,
        *,
        basis: Basis,
        k0: float,
        material: MaterialLike | None = None,
        modetype: str | None = None,
        poltype: str | None = None,
        medium: MaterialLike | None = None,
        kind: str | None = None,
        polarization: str | None = None,
    ):
        """Construct wave amplitudes with explicit basis, medium and radial/directional kind.

        Kind is regular/singular for multipoles and up/down for plane ports.
        Coefficients have shape (modes,) or (modes, illuminations). Polarization
        is helicity/parity. Material, modetype and poltype are the treams names
        of medium, kind and polarization; give each quantity under one name.
        """
        medium = one_of("medium", medium, "material", material, 1)
        kind = check_kind(one_of("kind", kind, "modetype", modetype, "regular"))
        polarization = one_of("polarization", polarization, "poltype", poltype, None)
        polarization = resolve_poltype(polarization)
        self.array = np.array(coefficients, dtype=np.complex128, copy=True)
        if (
            self.array.ndim not in (1, 2)
            or self.array.shape[0] != len(basis)
            or (self.array.ndim == 2 and self.array.shape[1] == 0)
            or not np.isfinite(self.array).all()
        ):
            raise ValueError(
                "wave coefficients require shape (modes,) or (modes, illuminations)"
            )
        k0 = check_k0(k0)
        kinds = (
            ("up", "down")
            if isinstance(basis, (PlaneWavePorts, PlaneWaveBasis))
            else ("regular", "singular")
        )
        if kind not in kinds:
            raise ValueError(f"wave mode type must be one of {kinds}")
        self.medium = Material(medium)
        self.polarization = check_poltype_medium(polarization, self.medium)
        self.basis, self.k0, self.kind = basis, k0, kind
        self.array.flags.writeable = False

    @property
    def coefficients(self) -> NDArray[np.complex128]:
        """Read-only amplitudes, shape (modes,) or (modes, illuminations)."""
        return self.array

    def in_basis(self, basis: Basis, *, kind: str | None = None) -> Wave:
        """Represent this field in another basis, retaining physical metadata.

        Kind is regular/singular for multipoles, up/down for plane-wave ports.
        Omission preserves the current kind within a family; plane-to-multipole
        conversion produces regular waves. Truncated bases are approximations.
        """
        if kind is None:
            kind = (
                "regular"
                if isinstance(self.basis, (PlaneWavePorts, PlaneWaveBasis))
                and isinstance(basis, (SphericalBasis, CylindricalBasis))
                else self.kind
            )
        else:
            kind = check_kind(kind)
        return type(self)(
            self._expanded(basis, kind),
            basis=basis,
            k0=self.k0,
            medium=self.medium,
            kind=kind,
            polarization=self.polarization,
        )

    def with_polarization(self, polarization: str) -> Wave:
        """Represent the same physical wave in helicity or parity channels.

        Every mode needs its partner of opposite polarization in the basis.
        """
        return self._with_polarization_target(polarization)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        return np.asarray(self.array, dtype=dtype, copy=copy)

    def _with_polarization_target(self, target: str | None) -> Wave:
        """The same field in the named convention, or the other one for None."""
        target = target_poltype(self.polarization, target)
        if target is None:
            return self
        return type(self)(
            change_polarization(self.array, self.basis, (0,)),
            basis=self.basis,
            k0=self.k0,
            medium=self.medium,
            kind=self.kind,
            polarization=target,
        )

    def _expanded(self, basis: Basis, kind: str = "regular") -> NDArray[np.complex128]:
        """Coefficients in ``basis`` with ``kind``; singular-to-regular translates outgoing waves."""
        if basis is self.basis and kind == self.kind:
            return self.array
        return (
            expand(
                (basis, self.basis),
                (kind, self.kind),
                k0=self.k0,
                material=self.medium,
                poltype=self.polarization,
            )
            @ self.array
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

    @property
    def modetype(self) -> str:
        """treams name of ``kind``."""
        return self.kind

    def changepoltype(self, poltype: str | None = None) -> Wave:
        """treams name of ``with_polarization``; omitting poltype switches convention."""
        return self._with_polarization_target(poltype)

    def expand(
        self, basis: Basis, *, modetype: str = "regular"
    ) -> NDArray[np.complex128]:
        """Coefficients only of ``in_basis(basis, kind=modetype)``."""
        return self._expanded(basis, modetype)


def spherical_wave(
    l: int,  # noqa: E741 - preserve the upstream multipole-degree argument
    m: int,
    pol: int,
    *,
    k0: float,
    basis: SphericalBasis | None = None,
    material: MaterialLike | None = None,
    modetype: str | None = None,
    poltype: str | None = None,
    medium: MaterialLike | None = None,
    kind: str | None = None,
    polarization: str | None = None,
) -> Wave:
    """One spherical mode in a global basis (positive helicity is pol=1)."""
    basis = SphericalBasis.default(l) if basis is None else basis
    if not basis.isglobal:
        raise ValueError("source basis must be global")
    index = basis.modes.index((int(basis.pidx[0]), l, m, pol))
    amplitudes = np.zeros(len(basis), complex)
    amplitudes[index] = 1
    return Wave(
        amplitudes,
        basis=basis,
        k0=k0,
        material=material,
        modetype=modetype,
        poltype=poltype,
        medium=medium,
        kind=kind,
        polarization=polarization,
    )


def cylindrical_wave(
    kz: float,
    m: int,
    pol: int,
    *,
    k0: float,
    basis: CylindricalBasis | None = None,
    material: MaterialLike | None = None,
    modetype: str | None = None,
    poltype: str | None = None,
    medium: MaterialLike | None = None,
    kind: str | None = None,
    polarization: str | None = None,
) -> Wave:
    """One cylindrical mode in a global basis, with fixed real axial label kz."""
    basis = CylindricalBasis.default([kz], abs(m)) if basis is None else basis
    if not basis.isglobal:
        raise ValueError("source basis must be global")
    index = basis.modes.index((int(basis.pidx[0]), kz, m, pol))
    amplitudes = np.zeros(len(basis), complex)
    amplitudes[index] = 1
    return Wave(
        amplitudes,
        basis=basis,
        k0=k0,
        material=material,
        modetype=modetype,
        poltype=poltype,
        medium=medium,
        kind=kind,
        polarization=polarization,
    )


class PlaneWave(WaveFields, UpstreamMembers):
    """An incident plane wave with two amplitudes, ordered by pol index 0, 1.

    The treams counterpart is the PhysicsArray that ``treams.plane_wave``
    returns. The treams-rs names come first; the treams names
    (``material``, ``poltype``, ``modetype``, ``expand``) follow at the end of
    the class and call them.

    Treat it as immutable: the constructor checks the metadata attributes once
    and never again.
    """

    medium: Material
    """Homogeneous propagation medium."""
    polarization: str
    """Polarization basis convention: helicity or parity."""
    kind = "up"
    """Always "up": the basis lists the wave's own direction."""

    def __new__(cls, kvec: Any = None, pol: Any = None, **kwargs: Any) -> Any:
        backend = backend_for(kvec, pol, kwargs)
        if backend is None:
            return super().__new__(cls)
        medium = one_of(
            "medium",
            kwargs.pop("medium", None),
            "material",
            kwargs.pop("material", None),
            1,
        )
        polarization = resolve_poltype(
            one_of(
                "polarization",
                kwargs.pop("polarization", None),
                "poltype",
                kwargs.pop("poltype", None),
                None,
            )
        )
        return namespace(backend).plane_wave(
            kvec,
            pol,
            medium=medium,
            polarization=polarization,
            **kwargs,
        )

    def __init__(
        self,
        kvec: ArrayLike,
        pol: ArrayLike,
        *,
        k0: float,
        medium: MaterialLike | None = None,
        polarization: str | None = None,
        material: MaterialLike | None = None,
        poltype: str | None = None,
    ):
        """Construct a plane wave from its direction and polarization state.

        Args:
            kvec: three finite components of the propagation direction; only the
                direction counts.
            pol: the pol index 0 or 1, two amplitudes for pol 0 and 1, or three
                Cartesian electric components.
            k0: positive vacuum angular wavenumber.
            medium: the propagation medium, vacuum by default.
            polarization: "helicity" (default) or "parity".
            material: treams name of ``medium``.
            poltype: treams name of ``polarization``.

        Give each quantity under one name. ``plane_wave`` also accepts the
        named states "positive_helicity" and "negative_helicity".
        """
        # treams keywords
        medium = one_of("medium", medium, "material", material, 1)
        poltype = resolve_poltype(
            one_of("polarization", polarization, "poltype", poltype, None)
        )
        vector = np.asarray(kvec, dtype=np.complex128)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("kvec must contain three finite components")
        self.k0 = check_k0(k0)
        self.direction = unit_vectors(vector[None, :])[0]
        self.direction.flags.writeable = False
        self.medium = Material(medium)
        self.polarization = check_poltype_medium(poltype, self.medium)
        state = np.asarray(pol, dtype=np.complex128)
        if state.ndim == 0:
            index = complex(state.item())
            if index not in (-1, 0, 1):
                raise ValueError("pol index must be 0 or 1")
            self._amplitudes = np.array(
                [1, 0] if index in (-1, 0) else [0, 1], dtype=np.complex128
            )
        elif state.shape == (2,):
            self._amplitudes = state.copy()
        elif state.shape == (3,):
            # Project onto each channel's polarization vector: helicity label
            # 1 - pol, or the signed parity vector (magnetic negated).
            helicity = poltype == "helicity"
            kvecs = self.kvecs
            self._amplitudes = np.array(
                [
                    (1 if helicity else 2 * pol - 1)
                    * np.dot(
                        _native.plane_polarization(
                            tuple(complex(v) for v in kvecs[pol]),
                            1 - pol if helicity else pol,
                            helicity,
                        ),
                        state,
                    )
                    for pol in (0, 1)
                ]
            )
        else:
            raise ValueError(
                "pol must be an index, two amplitudes, or three electric components"
            )
        if not np.isfinite(self._amplitudes).all():
            raise ValueError("polarization amplitudes must be finite")
        self._amplitudes.flags.writeable = False

    @property
    def coefficients(self) -> NDArray[np.complex128]:
        """Read-only polarization amplitudes in negative/positive label order."""
        return self.array

    def with_polarization(self, polarization: str) -> PlaneWave:
        """Represent the same plane wave with helicity or parity amplitudes."""
        wave = self.in_basis(self.basis).with_polarization(polarization)
        return type(self)(
            self.direction,
            wave.coefficients,
            k0=self.k0,
            medium=self.medium,
            polarization=polarization,
        )

    def in_basis(
        self,
        basis: SphericalBasis | CylindricalBasis | PlaneWavePorts | PlaneWaveBasis,
    ) -> Wave:
        """Expand into a typed regular multipole wave or directional plane wave."""
        kind = "regular"
        if isinstance(basis, PlaneWavePorts):
            kind = propagation_side(self.kvecs[0], basis.normal_axis)
        elif isinstance(basis, PlaneWaveBasis):
            kind = "up"
        return Wave(
            self._expanded(basis),
            basis=basis,
            k0=self.k0,
            medium=self.medium,
            kind=kind,
            polarization=self.polarization,
        )

    @property
    @override
    def basis(self) -> PlaneWaveBasis:
        """The two plane-wave modes of the direction, pol 0 then pol 1."""
        return PlaneWaveBasis([(*self.direction, pol) for pol in (0, 1)])

    @property
    @override
    def array(self) -> NDArray[np.complex128]:
        """The two amplitudes, shape (2,), ordered by pol index 0, 1."""
        return self._amplitudes

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        return np.asarray(self.array, dtype=dtype, copy=copy)

    @property
    def kvecs(self) -> NDArray[np.complex128]:
        """Wave vectors in the medium, shape (2, 3), one row per pol index."""
        return self.medium._plane_ks(self.k0)[:, None] * self.direction

    def _expanded(
        self,
        basis: SphericalBasis | CylindricalBasis | PlaneWavePorts | PlaneWaveBasis,
    ) -> NDArray[np.complex128]:
        """Regular multipole amplitudes at the supplied basis positions.

        Plane-wave bases must contain exactly one matching mode per nonzero
        polarization amplitude.
        """
        if isinstance(basis, (PlaneWaveBasis, PlaneWavePorts)):
            values = np.zeros(len(basis), dtype=np.complex128)
            kvecs = self.kvecs
            for pol in (0, 1):
                if self._amplitudes[pol] == 0:
                    continue
                if isinstance(basis, PlaneWaveBasis):
                    stored, target = basis.directions, self.direction
                else:
                    axes = ["xyz".index(axis) for axis in basis.alignment]
                    stored, target = basis.components, kvecs[pol, axes]
                matching = (basis.pol == pol) & np.all(
                    np.isclose(stored, target, rtol=1e-13, atol=1e-14), axis=1
                )
                if np.count_nonzero(matching) != 1:
                    raise ValueError(
                        "plane-wave illumination requires exactly one matching basis mode"
                    )
                values[matching] = self._amplitudes[pol]
            return values
        active = np.flatnonzero(self._amplitudes)
        if not active.size:
            return np.zeros(len(basis), dtype=np.complex128)
        expansion, _ = diff.plane_expansion(
            basis,
            self.kvecs[active],
            active,
            poltype=self.polarization,
            fixed_vectors=True,
        )
        return expansion @ self._amplitudes[active]

    # treams-compatible names (see treams_rs._upstream.UPSTREAM_MEMBERS)

    @property
    def material(self) -> Material:
        """treams name of ``medium``."""
        return self.medium

    @property
    def poltype(self) -> str:
        """treams name of ``polarization``."""
        return self.polarization

    modetype = kind

    def expand(
        self,
        basis: SphericalBasis | CylindricalBasis | PlaneWavePorts | PlaneWaveBasis,
    ) -> NDArray[np.complex128]:
        """Coefficients only of ``in_basis(basis)``."""
        return self._expanded(basis)


def plane_wave(
    kvec: ArrayLike | None = None,
    pol: ArrayLike | str | None = None,
    *,
    k0: float,
    material: MaterialLike | None = None,
    poltype: str | None = None,
    direction: ArrayLike | None = None,
    medium: MaterialLike | None = None,
    polarization: str | None = None,
) -> PlaneWave:
    """Define a plane wave from direction, polarization state pol, k0 and medium.

    Direction is normalized, independently of the vacuum angular wavenumber k0.
    The state ``pol`` is ``"positive_helicity"``/``"negative_helicity"``, a
    pol index 0 or 1, a pair of channel amplitudes, or three Cartesian
    electric components. A named helicity selects the helicity convention.
    ``polarization`` is the convention, ``"helicity"`` or ``"parity"``.
    kvec, material and poltype are the treams names of direction, medium and
    polarization; give each quantity under one name.
    """
    kvec = one_of("direction", direction, "kvec", kvec, None)
    material = one_of("medium", medium, "material", material, 1)
    poltype = one_of("polarization", polarization, "poltype", poltype, None)
    if poltype is not None and not (
        isinstance(poltype, str) and poltype in ("helicity", "parity")
    ):
        raise ValueError(
            "polarization is the convention 'helicity' or 'parity'; "
            "give the plane-wave state as pol"
        )
    if isinstance(pol, str):
        if pol not in ("positive_helicity", "negative_helicity"):
            raise ValueError("named pol must be positive_helicity or negative_helicity")
        if poltype not in (None, "helicity"):
            raise ValueError("a named helicity pol requires the helicity convention")
        poltype = "helicity"
        pol = int(pol == "positive_helicity")
    if kvec is None or pol is None:
        raise TypeError("plane_wave requires direction and pol")
    poltype = resolve_poltype(poltype)
    return PlaneWave(kvec, pol, k0=k0, medium=material, polarization=poltype)


def plane_wave_angle(
    theta: float,
    phi: float,
    pol: ArrayLike | str,
    *,
    k0: float,
    material: MaterialLike | None = None,
    poltype: str | None = None,
    medium: MaterialLike | None = None,
    polarization: str | None = None,
) -> PlaneWave:
    """Define an incident plane wave by polar and azimuthal angles in radians.

    Takes the same pol, medium and polarization keywords as plane_wave.
    """
    material = one_of("medium", medium, "material", material, 1)
    poltype = one_of("polarization", polarization, "poltype", poltype, None)
    backend = backend_for(theta, phi, pol, k0, material)
    xp = np if backend is None else backend.xp
    return plane_wave(
        [xp.sin(theta) * xp.cos(phi), xp.sin(theta) * xp.sin(phi), xp.cos(theta)],
        pol,
        k0=k0,
        material=material,
        poltype=poltype,
    )

"""Explicit basis operators backed by native numerical kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import diff
from . import lattice as _lattice
from ._core import (
    CylindricalWaveBasis,
    Material,
    PlaneWaveBasisByComp,
    PlaneWaveBasisByUnitVector,
    SphericalWaveBasis,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ._core import MaterialLike

    type Basis = SphericalWaveBasis | CylindricalWaveBasis
    type FieldBasis = Basis | PlaneWaveBasisByComp | PlaneWaveBasisByUnitVector


def rotate(
    phi: float, theta: float = 0, psi: float = 0, *, basis: Basis | tuple[Basis, Basis]
) -> NDArray[np.complex128]:
    """Rotation matrix in the z-y-z convention; basis origins remain fixed."""
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    return diff.rotation([phi, theta, psi], destination, source)[0]


def expand(
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    modetype: str | tuple[str, str] | None = None,
    *,
    k0: float,
    material: MaterialLike = 1,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Multipole expansion, including regular cylindrical-to-spherical waves.

    A basis pair is (destination, source). All explicit origin pairs are included.
    """
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    if isinstance(source, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        if not isinstance(destination, (SphericalWaveBasis, CylindricalWaveBasis)):
            raise ValueError("plane expansion requires a multipole destination")
        medium = Material(material)
        if not np.isfinite(k0) or k0 <= 0 or (poltype == "parity" and medium.ischiral):
            raise ValueError(
                "invalid frequency or embedding medium for the polarization type"
            )
        types = (
            ("regular", "up")
            if modetype is None
            else (modetype if isinstance(modetype, tuple) else ("regular", modetype))
        )
        if types[0] != "regular" or (
            isinstance(source, PlaneWaveBasisByComp) and types[1] not in ("up", "down")
        ):
            raise ValueError(
                "plane waves expand into regular multipoles from up/down modes"
            )
        return diff.plane_expansion(
            destination,
            np.column_stack(source.kvecs(k0, medium, types[1])),
            source.pol,
            poltype=poltype,
            fixed_vectors=True,
        )[0]
    if isinstance(destination, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        raise ValueError(
            "multipole-to-plane expansion requires a periodic radiation operator"
        )

    types = (
        ("regular", "regular")
        if modetype is None
        else (modetype if isinstance(modetype, tuple) else (modetype, modetype))
    )
    if types not in (
        ("regular", "regular"),
        ("singular", "singular"),
        ("regular", "singular"),
    ):
        raise ValueError("unsupported multipole expansion mode types")
    if not np.isfinite(k0) or k0 <= 0:
        raise ValueError("k0 must be positive and finite")
    if type(destination) is not type(source) and types != ("regular", "regular"):
        raise ValueError("cylindrical-to-spherical conversion requires regular waves")
    # Equal radial types use the regular addition theorem.
    return diff.expansion(
        destination,
        source,
        Material(material).ks(k0),
        poltype=poltype,
        singular=types == ("regular", "singular"),
    )[0]


def _field(
    kind: str,
    r: ArrayLike,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike,
    modetype: str | None,
    poltype: str,
) -> NDArray[np.complex128]:
    medium = Material(material)
    points = np.asarray(r, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 3 or not np.isfinite(k0) or k0 <= 0:
        raise ValueError(
            "require Cartesian field points (..., 3) and positive finite k0"
        )
    plane = isinstance(basis, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector))
    modetype = ("up" if plane else "regular") if modetype is None else modetype
    if not isinstance(basis, PlaneWaveBasisByUnitVector) and modetype not in (
        ("up", "down") if plane else ("regular", "singular")
    ):
        raise ValueError("invalid field mode type for this basis")
    if poltype not in ("helicity", "parity") or (
        poltype == "parity" and medium.ischiral
    ):
        raise ValueError("invalid polarization type for embedding medium")
    weights = np.ones(len(basis), dtype=np.complex128)
    if kind in ("H", "B"):
        weights *= -1j / medium.impedance
        if poltype == "helicity":
            weights *= 2 * basis.pol - 1
        else:
            if isinstance(basis, PlaneWaveBasisByUnitVector):
                basis = type(basis)([(*mode[:3], 1 - mode[3]) for mode in basis.modes])
            else:
                modes = [(*mode[:-1], 1 - mode[-1]) for mode in basis.modes]
                basis = (
                    type(basis)(modes, basis.alignment)
                    if isinstance(basis, PlaneWaveBasisByComp)
                    else type(basis)(modes, basis.positions)
                )
    if kind == "D":
        weights *= (
            medium.nmp[basis.pol] / medium.impedance
            if poltype == "helicity"
            else medium.epsilon
        )
    if kind == "B":
        weights *= (
            medium.nmp[basis.pol] * medium.impedance
            if poltype == "helicity"
            else medium.mu
        )
    if isinstance(basis, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        value, _ = diff.plane_field(
            None,
            points.reshape(-1, 3),
            np.column_stack(basis.kvecs(k0, medium, modetype)),
            basis.pol,
            poltype=poltype,
            fixed_vectors=True,
        )
    else:
        value, _ = diff.field_operator(
            points.reshape(-1, 3),
            basis,
            medium.ks(k0),
            poltype=poltype,
            singular=modetype == "singular",
        )
    return value.reshape((*points.shape[:-1], 3, len(basis))) * weights


def efield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Cartesian electric-field operator (..., 3, modes)."""
    return _field("E", r, basis, k0, material, modetype, poltype)


def hfield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Magnetic-field operator in units of electric field / vacuum impedance."""
    return _field("H", r, basis, k0, material, modetype, poltype)


def dfield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Electric-displacement operator in units of vacuum permittivity times E."""
    return _field("D", r, basis, k0, material, modetype, poltype)


def bfield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Magnetic-flux operator in units of electric field / vacuum light speed."""
    return _field("B", r, basis, k0, material, modetype, poltype)


def _periodic_channels(
    source: Basis,
    destination: PlaneWaveBasisByComp,
    ks: ArrayLike,
    lattice: ArrayLike,
    kpar: ArrayLike,
    poltype: str,
) -> NDArray[np.complex128]:
    """Validate physical diffraction ports and return native incidence/emission blocks."""
    vectors = np.atleast_2d(np.asarray(lattice, dtype=np.float64))
    bloch = np.atleast_1d(np.asarray(kpar, dtype=np.float64))
    q = destination.components
    if isinstance(source, SphericalWaveBasis):
        if (
            vectors.shape != (2, 2)
            or bloch.shape != (2,)
            or destination.alignment != "xy"
        ):
            raise ValueError(
                "spherical arrays require a 2D xy lattice, Bloch vector and plane basis"
            )
        orders = (q - bloch) @ vectors.T / (2 * np.pi)
        measure = float(abs(np.linalg.det(vectors)))
    else:
        if (
            vectors.shape != (1, 1)
            or bloch.shape != (1,)
            or destination.alignment != "zx"
        ):
            raise ValueError(
                "cylindrical arrays require a 1D x period, Bloch vector and zx plane basis"
            )
        orders = (q[:, 1] - bloch[0]) * vectors[0, 0] / (2 * np.pi)
        measure = float(abs(vectors[0, 0]))
    if not np.allclose(orders, np.round(orders), atol=1e-10, rtol=0):
        raise ValueError(
            "plane-wave channels must match the lattice diffraction orders"
        )
    # Both native channel signatures share the same scalar cell-measure argument.
    if isinstance(source, SphericalWaveBasis):
        return diff.spherical_channels(
            source, ks, q, destination.pol, measure, poltype=poltype, fixed_q=True
        )[0]
    return diff.cylindrical_channels(
        source, ks, q, destination.pol, measure, poltype=poltype, fixed_q=True
    )[0]


def expandlattice(
    lattice: ArrayLike,
    kpar: ArrayLike,
    *,
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    k0: float,
    material: MaterialLike = 1,
    poltype: str = "helicity",
    modetype: str | tuple[str, str] | None = None,
    eta: complex = 0,
) -> NDArray[np.complex128]:
    """Periodic multipole coupling or radiation, with explicit cell and Bloch vector.

    Spherical cells follow z/xy/xyz in 1D/2D/3D; cylindrical cells follow x/xy.
    A basis pair is (destination, source). Cross-family radiation includes all
    explicit origin pairs rather than an implicit matching-particle-index mask.
    """
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    medium = Material(material)
    if (
        not np.isfinite(k0)
        or k0 <= 0
        or poltype not in ("helicity", "parity")
        or (poltype == "parity" and medium.ischiral)
    ):
        raise ValueError(
            "invalid frequency or embedding medium for the polarization type"
        )
    if not isinstance(source, (SphericalWaveBasis, CylindricalWaveBasis)):
        raise ValueError("periodic expansion requires a multipole source")
    if isinstance(destination, PlaneWaveBasisByComp):
        side = (
            "up"
            if modetype is None
            else modetype[0]
            if isinstance(modetype, tuple)
            else modetype
        )
        if side not in ("up", "down") or (
            isinstance(modetype, tuple) and modetype[1] != "singular"
        ):
            raise ValueError("plane radiation requires up/down outgoing plane modes")
        channels = _periodic_channels(
            source, destination, medium.ks(k0), lattice, kpar, poltype
        )
        return channels[1, 0 if side == "up" else 1].T
    if isinstance(destination, CylindricalWaveBasis) and isinstance(
        source, SphericalWaveBasis
    ):
        vectors = np.atleast_2d(np.asarray(lattice, dtype=np.float64))
        bloch = np.atleast_1d(np.asarray(kpar, dtype=np.float64))
        if vectors.shape != (1, 1) or bloch.shape != (1,):
            raise ValueError(
                "spherical-to-cylindrical radiation requires a 1D z period and Bloch component"
            )
        orders = (destination.kz - bloch[0]) * vectors[0, 0] / (2 * np.pi)
        if not np.allclose(orders, np.round(orders), atol=1e-10, rtol=0):
            raise ValueError(
                "cylindrical axial wavenumbers must match diffraction orders"
            )
        if modetype not in (None, "singular", ("singular", "singular")):
            raise ValueError(
                "periodic spherical-to-cylindrical radiation requires outgoing waves"
            )
        return diff.periodic_conversion(
            destination,
            source,
            medium.ks(k0),
            float(abs(vectors[0, 0])),
            poltype=poltype,
        )[0]
    if type(destination) is type(source):
        if modetype not in (None, "regular", ("regular", "singular")):
            raise ValueError("periodic coupling maps outgoing to regular waves")
        return _lattice.expansion(
            destination, source, medium.ks(k0), lattice, kpar, poltype=poltype, eta=eta
        )
    raise ValueError("unsupported periodic wave-family conversion")

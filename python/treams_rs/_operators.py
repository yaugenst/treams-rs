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
from ._lattice import Lattice, WaveVector, _geometry_inputs

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ._core import MaterialLike

    type Basis = SphericalWaveBasis | CylindricalWaveBasis
    type FieldBasis = Basis | PlaneWaveBasisByComp | PlaneWaveBasisByUnitVector


def changepoltype(
    poltype: str | tuple[str, str] | None = None,
    *,
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    where: ArrayLike = True,
) -> NDArray[np.float64]:
    """Explicit helicity/parity conversion, including rectangular basis subsets.

    poltype names the destination type, or a (destination, source) pair.
    The real transformation is its own inverse for complete polarization pairs.
    Mode labels, origins and masks are discrete metadata.
    """
    poltype = "helicity" if poltype is None else poltype
    if poltype not in (
        "helicity",
        "parity",
        ("helicity", "parity"),
        ("parity", "helicity"),
    ):
        raise ValueError("polarization conversion must switch helicity and parity")
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    if type(destination) is not type(source):
        raise ValueError("polarization conversion requires the same wave family")
    if (
        isinstance(destination, PlaneWaveBasisByComp)
        and isinstance(source, PlaneWaveBasisByComp)
        and destination.alignment != source.alignment
    ):
        raise ValueError("polarization conversion requires matching alignments")
    out, incoming = np.asarray(destination.modes), np.asarray(source.modes)
    same = np.all(out[:, None, :-1] == incoming[None, :, :-1], axis=-1)
    signs = np.where((destination.pol[:, None] == 0) & (source.pol == 0), -1, 1)
    return (same & np.asarray(where, dtype=bool)) * signs * np.sqrt(0.5)


def rotate(
    phi: float,
    theta: float = 0,
    psi: float = 0,
    *,
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Rotation matrix in the z-y-z convention.

    Multipole origins remain fixed. Plane-wave rotations preserve coefficients
    and rotate the direction labels; their output basis is basis.rotate(phi+psi).
    As in treams, plane-wave theta must be zero and component bases must be xy.
    """
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    if isinstance(source, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        if type(destination) is not type(source) or destination.modes != source.modes:
            raise ValueError("plane rotations require matching input and output bases")
        if theta != 0 or not np.isfinite([phi, psi]).all():
            raise ValueError("plane rotations require finite phi/psi and zero theta")
        if isinstance(source, PlaneWaveBasisByComp) and (
            source.alignment != "xy"
            or not isinstance(destination, PlaneWaveBasisByComp)
            or destination.alignment != "xy"
        ):
            raise ValueError("plane rotations require xy alignment")
        return _masked(np.eye(len(source), dtype=np.complex128), where)
    if not isinstance(destination, (SphericalWaveBasis, CylindricalWaveBasis)):
        raise ValueError("rotations require matching wave families")
    return _masked(diff.rotation([phi, theta, psi], destination, source)[0], where)


def permute(
    n: int = 1,
    *,
    basis: PlaneWaveBasisByComp | PlaneWaveBasisByUnitVector,
    k0: float | None = None,
    material: MaterialLike = 1,
    modetype: str = "up",
    poltype: str = "helicity",
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Cyclic coordinate permutation; the output basis is basis.permute(n).

    Returns an explicit polarization matrix. Basis directions transform separately,
    preserving the same Cartesian field under the corresponding axis permutation.
    """
    if isinstance(basis, PlaneWaveBasisByUnitVector):
        vectors = basis.directions
    elif isinstance(basis, PlaneWaveBasisByComp):
        if k0 is None:
            raise ValueError("component plane permutations require k0")
        vectors = np.column_stack(basis.kvecs(k0, material, modetype))
    else:
        raise TypeError("permutations require a plane-wave basis")
    coefficients, _ = diff.plane_permutation(vectors, basis.pol, n, poltype=poltype)
    same = _plane_wave_match(vectors, vectors)
    return _masked(
        np.where(
            same, coefficients[basis.pol[:, None], np.arange(len(basis))[None, :]], 0
        ),
        where,
    )


def expand(
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    modetype: str | tuple[str, str] | None = None,
    *,
    k0: float,
    material: MaterialLike = 1,
    poltype: str = "helicity",
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Multipole expansion, including regular cylindrical-to-spherical waves.

    A basis pair is (destination, source). All explicit origin pairs are included.
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
    mask = np.asarray(where, dtype=bool)
    if isinstance(source, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        if isinstance(destination, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
            sides = modetype if isinstance(modetype, tuple) else (modetype or "up",) * 2
            return (
                _plane_wave_match(
                    np.column_stack(destination.kvecs(k0, medium, sides[0])),
                    np.column_stack(source.kvecs(k0, medium, sides[1])),
                )
                & (destination.pol[:, None] == source.pol)
                & mask
            ).astype(np.complex128)
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
        value = diff.plane_expansion(
            destination,
            np.column_stack(source.kvecs(k0, medium, types[1])),
            source.pol,
            poltype=poltype,
            fixed_vectors=True,
        )[0]
        return _masked(value, where)
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
    if type(destination) is not type(source) and types != ("regular", "regular"):
        raise ValueError("cylindrical-to-spherical conversion requires regular waves")
    # Equal radial types use the regular addition theorem.
    value = diff.expansion(
        destination,
        source,
        medium.ks(k0),
        poltype=poltype,
        singular=types == ("regular", "singular"),
    )[0]
    return _masked(value, where)


def _masked(value: NDArray[np.complex128], where: ArrayLike) -> NDArray[np.complex128]:
    if where is not True:
        value *= np.asarray(where, dtype=bool)
    return value


def _plane_wave_match(
    destination: NDArray[np.complex128], source: NDArray[np.complex128]
) -> NDArray[np.bool_]:
    if not np.isfinite(destination).all() or not np.isfinite(source).all():
        raise ValueError("plane wavevectors must be finite")
    tolerance = (
        32
        * np.finfo(float).eps
        * np.maximum(
            np.max(abs(destination), axis=1)[:, None], np.max(abs(source), axis=1)
        )
    )
    same = np.ones((len(destination), len(source)), dtype=bool)
    for axis in range(3):
        same &= abs(destination[:, None, axis] - source[:, axis]) <= tolerance
    return same


def translate(
    r: ArrayLike,
    *,
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    k0: float,
    material: MaterialLike = 1,
    poltype: str = "helicity",
    modetype: str = "up",
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Regular translation at fixed local origins, with displacement shape (..., 3).

    Multipole translations pair equal particle indices and ignore the stored
    origins. Use expand for all physical origin pairs. Plane translations apply
    exp(i k.r) to matching wavevectors and polarizations.
    """
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    offsets = np.asarray(r, dtype=np.float64)
    medium = Material(material)
    if offsets.ndim == 0 or offsets.shape[-1] != 3 or not np.isfinite(offsets).all():
        raise ValueError("translations require finite Cartesian displacements (..., 3)")
    if (
        not np.isfinite(k0)
        or k0 <= 0
        or poltype not in ("helicity", "parity")
        or (poltype == "parity" and medium.ischiral)
    ):
        raise ValueError(
            "invalid frequency or embedding medium for the polarization type"
        )
    shape = (*offsets.shape[:-1], len(destination), len(source))
    if isinstance(
        source, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)
    ) and isinstance(destination, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        vectors = np.column_stack(source.kvecs(k0, medium, modetype))
        matching = _plane_wave_match(
            np.column_stack(destination.kvecs(k0, medium, modetype)), vectors
        ) & (destination.pol[:, None] == source.pol)
        phases = diff.plane_phases(offsets.reshape(-1, 3), vectors)[0]
        return _masked((phases[:, None, :] * matching).reshape(shape), where)
    if (
        not isinstance(source, (SphericalWaveBasis, CylindricalWaveBasis))
        or not isinstance(destination, (SphericalWaveBasis, CylindricalWaveBasis))
        or type(destination) is not type(source)
    ):
        raise ValueError("translation requires matching wave families")
    incoming = type(source)(source.modes, np.zeros_like(source.positions))
    matching = destination.pidx[:, None] == source.pidx
    points = offsets.reshape(-1, 3)
    ks = medium.ks(k0)
    result = np.empty((len(points), len(destination), len(source)), dtype=np.complex128)
    for i, offset in enumerate(points):
        outgoing = type(destination)(
            destination.modes, np.broadcast_to(offset, destination.positions.shape)
        )
        result[i] = (
            diff.expansion(outgoing, incoming, ks, poltype=poltype)[0] * matching
        )
    return _masked(result.reshape(shape), where)


def _field(
    kind: str,
    r: ArrayLike,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike,
    modetype: str | None,
    poltype: str,
    coefficients: ArrayLike | None = None,
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
    weighted = None
    if coefficients is not None:
        amplitudes = np.asarray(coefficients, dtype=np.complex128)
        if amplitudes.shape != (len(basis),):
            raise ValueError("field requires one amplitude per basis mode")
        weighted = amplitudes * weights
    if isinstance(basis, (PlaneWaveBasisByComp, PlaneWaveBasisByUnitVector)):
        value, _ = diff.plane_field(
            weighted,
            points.reshape(-1, 3),
            np.column_stack(basis.kvecs(k0, medium, modetype)),
            basis.pol,
            poltype=poltype,
            fixed_vectors=True,
        )
    elif weighted is None:
        value, _ = diff.field_operator(
            points.reshape(-1, 3),
            basis,
            medium.ks(k0),
            poltype=poltype,
            singular=modetype == "singular",
        )
    else:
        value, _ = diff.field(
            weighted,
            points.reshape(-1, 3),
            basis,
            medium.ks(k0),
            poltype=poltype,
            singular=modetype == "singular",
        )
    if weighted is not None:
        return value.reshape(points.shape)
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


def _rs_weights(
    pol: int, basis: FieldBasis, poltype: str
) -> tuple[NDArray[np.float64], float]:
    if pol not in (-1, 0, 1):
        raise ValueError("Riemann-Silberstein polarization must be -1, 0 or 1")
    pol = max(pol, 0)
    # Preserve upstream's different spherical and cylindrical/plane scalings.
    normalization = np.sqrt(2) if isinstance(basis, SphericalWaveBasis) else 1.0
    if poltype == "helicity":
        return normalization * (basis.pol == pol), 0.0
    if poltype == "parity":
        return np.full(len(basis), normalization), normalization * (2 * pol - 1)
    raise ValueError("polarization type must be helicity or parity")


def gfield(
    pol: int,
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Riemann-Silberstein G operator with treams' family/polarization scaling.

    Polarization -1 aliases 0. For a normalization independent of the basis
    convention, form (E +/- i Z H)/sqrt(2) from efield and hfield directly.
    """
    electric, magnetic = _rs_weights(pol, basis, poltype)
    value = _field("E", r, basis, k0, material, modetype, poltype) * electric
    if magnetic:
        value += (
            1j
            * Material(material).impedance
            * magnetic
            * _field("H", r, basis, k0, material, modetype, poltype)
        )
    return value


def ffield(
    pol: int,
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Riemann-Silberstein F operator, including the chiral index weights."""
    value = gfield(
        pol,
        r,
        basis=basis,
        k0=k0,
        material=material,
        modetype=modetype,
        poltype=poltype,
    )
    if poltype == "helicity":
        medium = Material(material)
        value *= medium.nmp[basis.pol] / medium.n
    return value


def _periodic_channels(
    source: Basis,
    destination: PlaneWaveBasisByComp,
    ks: ArrayLike,
    lattice: ArrayLike,
    kpar: ArrayLike,
    poltype: str,
) -> NDArray[np.complex128]:
    """Validate physical diffraction ports and return native incidence/emission blocks."""
    vectors, bloch = _geometry_inputs(
        lattice, kpar, "xy" if isinstance(source, SphericalWaveBasis) else "x"
    )
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
    lattice: ArrayLike | Lattice | None = None,
    kpar: ArrayLike | WaveVector | None = None,
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
    lattice = (destination.lattice or source.lattice) if lattice is None else lattice
    kpar = (
        (source.kpar if source.kpar is not None else destination.kpar)
        if kpar is None
        else kpar
    )
    if lattice is None or kpar is None:
        raise ValueError("periodic expansion requires a lattice and Bloch vector")
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
        vectors, bloch = _geometry_inputs(lattice, kpar, "z")
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

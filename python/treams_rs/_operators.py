"""Functional operator builders that return plain arrays.

Mirrors the functional half of ``treams._operators``. The operator classes live in
``_operator_objects`` and field sampling in ``_fields``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import diff
from ._bases import (
    CylindricalBasis,
    PlaneWaveBasis,
    PlaneWavePorts,
    SphericalBasis,
)
from ._fields import field, riemann
from ._lattice import Lattice, WaveVector, geometry_inputs, on_diffraction_orders
from ._material import as_material
from ._polarization import (
    DEFAULT_POLTYPE,
    check_poltype_medium,
    resolve_poltype,
)
from ._validation import check_k0

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ._bases import FieldBasis
    from ._material import MaterialLike

    type Basis = SphericalBasis | CylindricalBasis

__all__ = [
    "bfield",
    "changepoltype",
    "dfield",
    "efield",
    "expand",
    "expandlattice",
    "ffield",
    "gfield",
    "hfield",
    "periodic_channels",
    "permute",
    "rotate",
    "translate",
]


def changepoltype(
    poltype: str | tuple[str, str] | None = None,
    *,
    basis: FieldBasis | tuple[FieldBasis, FieldBasis],
    where: ArrayLike = True,
) -> NDArray[np.float64]:
    """Matrix that changes the polarization convention between helicity and parity.

    Mirrors ``treams.changepoltype``. The real matrix is the same in both
    directions and is its own inverse when the basis holds both ``pol`` of every
    mode. A (destination, source) basis pair selects a rectangular part.

    Args:
        poltype: Destination convention, ``"helicity"`` or ``"parity"``, or a
            (destination, source) pair; it must name a change. ``"helicity"``
            when omitted.
        basis: Basis, or (destination, source) pair, of one wave family
            (metadata).
        where: Boolean mask of the entries to keep, broadcast to the output.
    """
    poltype = DEFAULT_POLTYPE if poltype is None else poltype
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
        isinstance(destination, PlaneWavePorts)
        and isinstance(source, PlaneWavePorts)
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
    """Rotation matrix for the Euler angles phi, theta, psi (z-y-z convention).

    Mirrors ``treams.rotate``. Multipole positions stay fixed. A plane-wave
    rotation keeps the coefficients and rotates the direction labels: its
    output basis is ``basis.rotate(phi + psi)``. As in treams, plane waves need
    theta = 0, and PlaneWavePorts need the alignment ``"xy"``.

    Args:
        phi: First angle, about z.
        theta: Second angle, about y.
        psi: Third angle, about z.
        basis: Basis, or (destination, source) pair (metadata).
        where: Boolean mask of the entries to keep, broadcast to the output.
    """
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    if isinstance(source, (PlaneWavePorts, PlaneWaveBasis)):
        if type(destination) is not type(source) or destination.modes != source.modes:
            raise ValueError("plane rotations require matching input and output bases")
        if theta != 0 or not np.isfinite([phi, psi]).all():
            raise ValueError("plane rotations require finite phi/psi and zero theta")
        if isinstance(source, PlaneWavePorts) and (
            source.alignment != "xy"
            or not isinstance(destination, PlaneWavePorts)
            or destination.alignment != "xy"
        ):
            raise ValueError("plane rotations require xy alignment")
        return _masked(np.eye(len(source), dtype=np.complex128), where)
    if not isinstance(destination, (SphericalBasis, CylindricalBasis)):
        raise ValueError("rotations require matching wave families")
    return _masked(diff.rotation([phi, theta, psi], destination, source)[0], where)


def permute(
    n: int = 1,
    *,
    basis: PlaneWavePorts | PlaneWaveBasis,
    k0: float | None = None,
    material: MaterialLike = 1,
    modetype: str = "up",
    poltype: str | None = None,
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Matrix of n cyclic permutations x -> y -> z of the plane-wave coordinates.

    Mirrors ``treams.permute``. The matrix maps the polarization amplitudes;
    the directions change with the basis, so the output basis is
    ``basis.permute(n)``. The permuted coefficients describe the same field
    with its coordinate axes permuted.

    Args:
        n: Number of cyclic permutations.
        basis: PlaneWaveBasis or PlaneWavePorts (metadata).
        k0: Vacuum angular wavenumber (metadata); PlaneWavePorts need it.
        material: Medium (metadata).
        modetype: ``"up"`` or ``"down"`` for PlaneWavePorts (metadata).
        poltype: Polarization convention (metadata), ``"helicity"`` by default.
        where: Boolean mask of the entries to keep, broadcast to the output.
    """
    poltype = resolve_poltype(poltype)
    if isinstance(basis, PlaneWaveBasis):
        vectors = basis.directions
    elif isinstance(basis, PlaneWavePorts):
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
    poltype: str | None = None,
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Expansion matrix from a source basis into a destination basis.

    Mirrors ``treams.expand``. It covers multipoles of one family, regular
    cylindrical waves into regular spherical waves, and plane waves into plane
    waves or regular multipoles. The matrix includes every pair of destination
    and source positions.

    Args:
        basis: Basis, or (destination, source) pair (metadata).
        modetype: Kinds of the destination and source waves (metadata): one
            kind for both, or a (destination, source) pair. Multipoles take
            ``("regular", "regular")`` (default), ``("singular", "singular")``
            or ``("regular", "singular")``; plane waves take ``"up"`` or
            ``"down"``.
        k0: Vacuum angular wavenumber (metadata).
        material: Medium (metadata).
        poltype: Polarization convention (metadata), ``"helicity"`` by default.
        where: Boolean mask of the entries to keep, broadcast to the output.

    Differences from treams:
        Between spherical and cylindrical bases with several positions, the
        matrix adds every pair of positions; treams pairs only equal particle
        indices.
    """
    poltype = resolve_poltype(poltype)
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    medium = as_material(material)
    check_k0(k0)
    check_poltype_medium(poltype, medium)
    mask = np.asarray(where, dtype=bool)
    if isinstance(source, (PlaneWavePorts, PlaneWaveBasis)):
        if isinstance(destination, (PlaneWavePorts, PlaneWaveBasis)):
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
            isinstance(source, PlaneWavePorts) and types[1] not in ("up", "down")
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
    if isinstance(destination, (PlaneWavePorts, PlaneWaveBasis)):
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
    poltype: str | None = None,
    modetype: str = "up",
    where: ArrayLike = True,
) -> NDArray[np.complex128]:
    """Translation matrix of a displacement r, shape (..., 3), at fixed positions.

    Mirrors ``treams.translate``. Multipole translations pair equal particle
    indices and do not use the stored positions; use ``expand`` to couple
    waves at different positions. Plane-wave translations multiply matching
    wavevectors and polarizations by exp(i k.r).

    Args:
        r: Displacements, shape (..., 3); the output has shape
            (..., destination modes, source modes).
        basis: Basis, or (destination, source) pair of one family (metadata).
        k0: Vacuum angular wavenumber (metadata).
        material: Medium (metadata).
        poltype: Polarization convention (metadata), ``"helicity"`` by default.
        modetype: ``"up"`` or ``"down"`` for PlaneWavePorts (metadata).
        where: Boolean mask of the entries to keep, broadcast to the output.
    """
    poltype = resolve_poltype(poltype)
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    offsets = np.asarray(r, dtype=np.float64)
    medium = as_material(material)
    if offsets.ndim == 0 or offsets.shape[-1] != 3 or not np.isfinite(offsets).all():
        raise ValueError("translations require finite Cartesian displacements (..., 3)")
    check_k0(k0)
    check_poltype_medium(poltype, medium)
    shape = (*offsets.shape[:-1], len(destination), len(source))
    if isinstance(source, (PlaneWavePorts, PlaneWaveBasis)) and isinstance(
        destination, (PlaneWavePorts, PlaneWaveBasis)
    ):
        vectors = np.column_stack(source.kvecs(k0, medium, modetype))
        matching = _plane_wave_match(
            np.column_stack(destination.kvecs(k0, medium, modetype)), vectors
        ) & (destination.pol[:, None] == source.pol)
        phases = diff.plane_phases(offsets.reshape(-1, 3), vectors)[0]
        return _masked((phases[:, None, :] * matching).reshape(shape), where)
    if (
        not isinstance(source, (SphericalBasis, CylindricalBasis))
        or not isinstance(destination, (SphericalBasis, CylindricalBasis))
        or type(destination) is not type(source)
    ):
        raise ValueError("translation requires matching wave families")
    # Every kept block pairs equal particle indices displaced by the same offset,
    # so it is one particle-independent translation restricted to its labels.
    family = type(source)
    labels = tuple(
        dict.fromkeys(mode[1:] for mode in (*destination.modes, *source.modes))
    )
    position = {label: i for i, label in enumerate(labels)}
    selection = np.ix_(
        [position[mode[1:]] for mode in destination.modes],
        [position[mode[1:]] for mode in source.modes],
    )
    origin = family(labels)
    matching = destination.pidx[:, None] == source.pidx
    points = offsets.reshape(-1, 3)
    ks = medium.ks(k0)
    result = np.empty((len(points), len(destination), len(source)), dtype=np.complex128)
    for i, offset in enumerate(points):
        shifted = family(labels, offset)
        translation = diff.expansion(shifted, origin, ks, poltype=poltype)[0]
        result[i] = translation[selection] * matching
    return _masked(result.reshape(shape), where)


def efield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Electric-field matrix of shape (..., 3, modes) at the sample points r.

    Mirrors ``treams.efield``.

    Args:
        r: Sample points, shape (..., 3).
        basis: Basis of the coefficients (metadata).
        k0: Vacuum angular wavenumber (metadata).
        material: Medium at the sample points (metadata).
        modetype: Kind of the waves (metadata): ``"regular"`` (default) or
            ``"singular"`` for multipoles, ``"up"`` (default) or ``"down"``
            for PlaneWavePorts.
        poltype: Polarization convention (metadata), ``"helicity"`` by default.
    """
    poltype = resolve_poltype(poltype)
    return field("E", r, basis, k0, material, modetype, poltype)


def hfield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Magnetic-field matrix, in units of E over the vacuum impedance.

    Mirrors ``treams.hfield``. The output has shape (..., 3, modes).

    Args:
        r: Sample points, shape (..., 3).
        basis: Basis of the coefficients (metadata).
        k0: Vacuum angular wavenumber (metadata).
        material: Medium at the sample points (metadata).
        modetype: Kind of the waves (metadata): ``"regular"`` (default) or
            ``"singular"`` for multipoles, ``"up"`` (default) or ``"down"``
            for PlaneWavePorts.
        poltype: Polarization convention (metadata), ``"helicity"`` by default.
    """
    poltype = resolve_poltype(poltype)
    return field("H", r, basis, k0, material, modetype, poltype)


def dfield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Electric-displacement matrix, in units of the vacuum permittivity times E.

    Mirrors ``treams.dfield``. The output has shape (..., 3, modes).

    Args:
        r: Sample points, shape (..., 3).
        basis: Basis of the coefficients (metadata).
        k0: Vacuum angular wavenumber (metadata).
        material: Medium at the sample points (metadata).
        modetype: Kind of the waves (metadata): ``"regular"`` (default) or
            ``"singular"`` for multipoles, ``"up"`` (default) or ``"down"``
            for PlaneWavePorts.
        poltype: Polarization convention (metadata), ``"helicity"`` by default.
    """
    poltype = resolve_poltype(poltype)
    return field("D", r, basis, k0, material, modetype, poltype)


def bfield(
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Magnetic-flux-density matrix, in units of E over the vacuum speed of light.

    Mirrors ``treams.bfield``. The output has shape (..., 3, modes).

    Args:
        r: Sample points, shape (..., 3).
        basis: Basis of the coefficients (metadata).
        k0: Vacuum angular wavenumber (metadata).
        material: Medium at the sample points (metadata).
        modetype: Kind of the waves (metadata): ``"regular"`` (default) or
            ``"singular"`` for multipoles, ``"up"`` (default) or ``"down"``
            for PlaneWavePorts.
        poltype: Polarization convention (metadata), ``"helicity"`` by default.
    """
    poltype = resolve_poltype(poltype)
    return field("B", r, basis, k0, material, modetype, poltype)


def gfield(
    pol: int,
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Riemann-Silberstein field G of helicity pol, scaled like treams.

    Mirrors ``treams.gfield``. The scaling depends on the wave family and the
    polarization convention, as in treams. For a scaling independent of the
    basis, form (E +/- i Z H) / sqrt(2) from ``efield`` and ``hfield``. The
    output has shape (..., 3, modes).

    Args:
        pol: Helicity of the field: 1 positive, 0 or -1 negative.
        r: Sample points, shape (..., 3).
        basis: Basis of the coefficients (metadata).
        k0: Vacuum angular wavenumber (metadata).
        material: Medium at the sample points (metadata).
        modetype: Kind of the waves (metadata), as for ``efield``.
        poltype: Polarization convention (metadata), ``"helicity"`` by default.
    """
    poltype = resolve_poltype(poltype)
    return riemann("G", r, basis, k0, as_material(material), modetype, poltype, pol)


def ffield(
    pol: int,
    r: ArrayLike,
    *,
    basis: FieldBasis,
    k0: float,
    material: MaterialLike = 1,
    modetype: str | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Riemann-Silberstein field F of helicity pol, with the chiral index weights.

    Mirrors ``treams.ffield``. The output has shape (..., 3, modes).

    Args:
        pol: Helicity of the field: 1 positive, 0 or -1 negative.
        r: Sample points, shape (..., 3).
        basis: Basis of the coefficients (metadata).
        k0: Vacuum angular wavenumber (metadata).
        material: Medium at the sample points (metadata).
        modetype: Kind of the waves (metadata), as for ``efield``.
        poltype: Polarization convention (metadata), ``"helicity"`` by default.
    """
    poltype = resolve_poltype(poltype)
    return riemann("F", r, basis, k0, as_material(material), modetype, poltype, pol)


def periodic_channels(
    source: Basis,
    destination: PlaneWavePorts,
    ks: ArrayLike,
    lattice: ArrayLike,
    kpar: ArrayLike,
    poltype: str,
) -> NDArray[np.complex128]:
    """Coupling blocks between a periodic multipole array and its diffraction channels.

    Checks that the plane-wave ports lie on the diffraction orders of the
    lattice, then returns the native incidence and emission blocks.
    """
    vectors, bloch = geometry_inputs(
        lattice, kpar, "xy" if isinstance(source, SphericalBasis) else "x"
    )
    q = destination.components
    if isinstance(source, SphericalBasis):
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
    if not on_diffraction_orders(orders):
        raise ValueError(
            "plane-wave channels must match the lattice diffraction orders"
        )
    # Both native channel signatures share the same scalar cell-measure argument.
    if isinstance(source, SphericalBasis):
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
    poltype: str | None = None,
    modetype: str | tuple[str, str] | None = None,
    eta: complex = 0,
) -> NDArray[np.complex128]:
    """Coupling or radiation matrix of a periodic array of multipoles.

    Mirrors ``treams.expandlattice``. A same-family destination gives the
    lattice coupling (singular to regular waves); PlaneWavePorts give the
    radiation into diffraction channels; a cylindrical destination of a
    spherical source gives the radiation of sphere chains into cylindrical
    waves. Spherical lattices are z, xy or xyz in 1D, 2D or 3D; cylindrical
    lattices are x or xy.

    Args:
        lattice: Lattice vectors or a ``Lattice`` (metadata); the basis
            lattice when omitted.
        kpar: Bloch vector, or a ``WaveVector`` (metadata); the basis Bloch
            vector when omitted.
        basis: Basis, or (destination, source) pair (metadata).
        k0: Vacuum angular wavenumber (metadata).
        material: Medium (metadata).
        poltype: Polarization convention (metadata), ``"helicity"`` by default.
        modetype: Kinds of the destination and source waves (metadata), one
            kind or a (destination, source) pair.
        eta: Ewald split parameter, which balances the real-space and
            reciprocal-space parts of the lattice sum; 0 chooses it
            automatically.

    Differences from treams:
        Between spherical and cylindrical bases with several positions, the
        matrix adds every pair of positions; treams pairs only equal particle
        indices.
    """
    poltype = resolve_poltype(poltype)
    destination, source = basis if isinstance(basis, tuple) else (basis, basis)
    lattice = (destination.lattice or source.lattice) if lattice is None else lattice
    kpar = (
        (source.kpar if source.kpar is not None else destination.kpar)
        if kpar is None
        else kpar
    )
    if lattice is None or kpar is None:
        raise ValueError("periodic expansion requires a lattice and Bloch vector")
    medium = as_material(material)
    check_k0(k0)
    check_poltype_medium(poltype, medium)
    if not isinstance(source, (SphericalBasis, CylindricalBasis)):
        raise ValueError("periodic expansion requires a multipole source")
    if isinstance(destination, PlaneWavePorts):
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
        channels = periodic_channels(
            source, destination, medium.ks(k0), lattice, kpar, poltype
        )
        return channels[1, 0 if side == "up" else 1].T
    if isinstance(destination, CylindricalBasis) and isinstance(source, SphericalBasis):
        vectors, bloch = geometry_inputs(lattice, kpar, "z")
        if vectors.shape != (1, 1) or bloch.shape != (1,):
            raise ValueError(
                "spherical-to-cylindrical radiation requires a 1D z period and Bloch component"
            )
        orders = (destination.kz - bloch[0]) * vectors[0, 0] / (2 * np.pi)
        if not on_diffraction_orders(orders):
            raise ValueError(
                "cylindrical axial wavenumbers must match diffraction orders"
            )
        if modetype not in (None, "singular", ("singular", "singular")):
            raise ValueError(
                "periodic spherical-to-cylindrical radiation requires outgoing waves"
            )
        return diff.periodic_to_cw(
            destination,
            source,
            medium.ks(k0),
            float(abs(vectors[0, 0])),
            poltype=poltype,
        )[0]
    if type(destination) is type(source):
        if modetype not in (None, "regular", ("regular", "singular")):
            raise ValueError("periodic coupling maps outgoing to regular waves")
        return diff.lattice_expansion(
            destination, source, medium.ks(k0), kpar, lattice, poltype=poltype, eta=eta
        )[0]
    raise ValueError("unsupported periodic wave-family conversion")

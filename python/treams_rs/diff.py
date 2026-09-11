"""Framework-neutral native operations; all pullbacks use the real Hermitian pairing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _native
from ._core import CylindricalWaveBasis, SphericalWaveBasis

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def solve(
    operator: ArrayLike, rhs: ArrayLike
) -> tuple[NDArray[np.complex128], _native.SolveContext]:
    """Solve A X = B for a matrix B; VJP returns (A_bar, B_bar) using retained LU."""
    return _native.linear_solve(
        np.ascontiguousarray(operator, dtype=np.complex128),
        np.ascontiguousarray(rhs, dtype=np.complex128),
    )


def eig(
    operator: ArrayLike,
) -> tuple[tuple[NDArray[np.complex128], NDArray[np.complex128]], _native.EigenContext]:
    """Complex eigenvalues and unit right eigenvectors, plus their native pullback.

    Each vector's largest component is real positive. Pullback takes separate
    value/vector cotangents. Individual modes at repeated eigenvalues have no VJP;
    equal value weights with zero vector cotangents support spectral sums there.
    """
    values, vectors, context = _native.eig(
        np.ascontiguousarray(operator, dtype=np.complex128)
    )
    return (values, vectors), context


def smatrix_from_array(
    response: ArrayLike, channels: ArrayLike
) -> tuple[NDArray[np.complex128], _native.ArrayContext]:
    """Radiate an effective periodic response; VJP returns (response, channels)."""
    return _native.smatrix_from_array(
        np.ascontiguousarray(response, dtype=np.complex128),
        np.ascontiguousarray(channels, dtype=np.complex128),
    )


def smatrix_illuminate(
    lower: ArrayLike, upper: ArrayLike, up: ArrayLike, down: ArrayLike
) -> tuple[NDArray[np.complex128], _native.IlluminationContext]:
    """Outgoing up/down and internal up/down coefficients of two adjacent stacks.

    Inputs up/down have shape (modes, illuminations); output shape is
    (4, modes, illuminations). The native solve only computes these right-hand
    sides. Pullback returns lower, upper, up and down cotangents.
    """
    return _native.smatrix_illuminate(
        np.ascontiguousarray(lower, dtype=np.complex128),
        np.ascontiguousarray(upper, dtype=np.complex128),
        np.ascontiguousarray(up, dtype=np.complex128),
        np.ascontiguousarray(down, dtype=np.complex128),
    )


def smatrix_periodic(
    smats: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.SMatrixPeriodicContext]:
    """Periodic transfer matrix and its four-block scattering-matrix pullback."""
    return _native.smatrix_periodic(np.ascontiguousarray(smats, dtype=np.complex128))


def bands(
    smats: ArrayLike, period: float
) -> tuple[tuple[NDArray[np.complex128], NDArray[np.complex128]], _native.BandContext]:
    """Normal Bloch wavenumbers and right vectors, with S-matrix/period pullback.

    Uses the principal logarithm. Derivatives hold its branch and eigenvector
    pivot phase fixed; individual modes at repeated eigenvalues are undefined.
    """
    wavenumbers, vectors, context = _native.bands(
        np.ascontiguousarray(smats, dtype=np.complex128), period
    )
    return (wavenumbers, vectors), context


def spherical_channels(
    basis: SphericalWaveBasis,
    ks: ArrayLike,
    q: ArrayLike,
    polarizations: ArrayLike,
    area: float,
    *,
    poltype: str = "helicity",
    fixed_q: bool = False,
) -> tuple[NDArray[np.complex128], _native.ChannelsContext]:
    """Incident/emitted, up/down arrays, shaped (2, 2, multipoles, plane modes).

    Emitted arrays are transposed: ``channels[1, side].T`` maps multipoles to
    outgoing plane waves. VJP returns (positions, ks, q, area). At exactly normal
    incidence the azimuth is undefined; set fixed_q for derivatives at fixed incidence.
    """
    if poltype not in ("helicity", "parity"):
        raise ValueError("invalid polarization type")
    pols = np.asarray(polarizations)
    if not np.all((pols == 0) | (pols == 1)):
        raise ValueError("polarizations must be 0 or 1")
    return _native.spherical_channels(
        list(basis.modes),
        basis.positions.tolist(),
        np.asarray(ks, dtype=np.complex128).tolist(),
        np.asarray(q, dtype=np.float64).tolist(),
        pols.astype(np.int64).tolist(),
        area,
        poltype == "helicity",
        fixed_q,
    )


def smatrix_add(
    lower: ArrayLike, upper: ArrayLike
) -> tuple[NDArray[np.complex128], _native.SMatrixContext]:
    """Couple four-block S matrices; pullback returns (lower_bar, upper_bar)."""
    return _native.smatrix_add(
        np.ascontiguousarray(lower, dtype=np.complex128),
        np.ascontiguousarray(upper, dtype=np.complex128),
    )


def propagation(
    vectors: ArrayLike, distance: ArrayLike
) -> tuple[NDArray[np.complex128], _native.PropagationContext]:
    """Propagate upgoing wavevectors by a Cartesian distance; VJP returns (vectors, distance)."""
    return _native.propagation(
        np.asarray(vectors, dtype=np.complex128).tolist(),
        np.asarray(distance, dtype=np.float64).tolist(),
    )


def sphere(
    lmax: int,
    k0: float,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike | None = None,
    kappa: ArrayLike | None = None,
) -> tuple[NDArray[np.complex128], _native.SphereContext]:
    """Helicity T-matrix and pullback returning (k0, radii, epsilon, mu, kappa)."""
    eps = np.ascontiguousarray(epsilon, dtype=np.complex128)
    return _native.sphere(
        lmax,
        k0,
        np.ascontiguousarray(radii, dtype=np.float64),
        eps,
        np.ones_like(eps)
        if mu is None
        else np.ascontiguousarray(mu, dtype=np.complex128),
        np.zeros_like(eps)
        if kappa is None
        else np.ascontiguousarray(kappa, dtype=np.complex128),
    )


def cluster(
    lmax: int,
    k0: float,
    radii: ArrayLike,
    epsilon: ArrayLike,
    positions: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.ClusterContext]:
    """Homogeneous nonmagnetic spheres in vacuum; pullback returns (radii, positions, epsilon, k0)."""
    return _native.cluster(
        lmax,
        k0,
        np.ascontiguousarray(radii, dtype=np.float64),
        np.ascontiguousarray(epsilon, dtype=np.complex128),
        np.ascontiguousarray(positions, dtype=np.float64),
    )


def interaction(
    local: ArrayLike, coupling: ArrayLike
) -> tuple[NDArray[np.complex128], _native.InteractionContext]:
    """Solve (I - T C) X = T; pullback returns (T_bar, C_bar)."""
    return _native.interact(
        np.ascontiguousarray(local, dtype=np.complex128),
        np.ascontiguousarray(coupling, dtype=np.complex128),
    )


def expansion(
    destination: SphericalWaveBasis | CylindricalWaveBasis,
    source: SphericalWaveBasis | CylindricalWaveBasis,
    ks: ArrayLike,
    *,
    poltype: str = "helicity",
    singular: bool = False,
) -> tuple[NDArray[np.complex128], _native.ExpansionContext]:
    """Expansion VJP returns (destination positions, source positions, ks).

    Cylindrical axial wavenumbers are fixed mode labels; different labels decouple.
    """
    values = np.asarray(ks, dtype=np.complex128)
    if values.shape != (2,):
        raise ValueError("ks must contain negative and positive helicity wavenumbers")
    if poltype not in ("helicity", "parity"):
        raise ValueError("invalid polarization type")
    if isinstance(destination, CylindricalWaveBasis) and isinstance(
        source, CylindricalWaveBasis
    ):
        if poltype == "parity" and values[0] != values[1]:
            raise ValueError("parity requires an achiral medium")
        return _native.cyl_expansion(
            list(destination.modes),
            list(source.modes),
            destination.positions.tolist(),
            source.positions.tolist(),
            (complex(values[0]), complex(values[1])),
            singular,
        )
    if isinstance(destination, SphericalWaveBasis) and isinstance(
        source, CylindricalWaveBasis
    ):
        if singular:
            raise ValueError(
                "cylindrical-to-spherical conversion requires regular waves"
            )
        return _native.cw_to_sw(
            list(destination.modes),
            list(source.modes),
            destination.positions.tolist(),
            source.positions.tolist(),
            (complex(values[0]), complex(values[1])),
            poltype == "helicity",
        )
    if not isinstance(destination, SphericalWaveBasis) or not isinstance(
        source, SphericalWaveBasis
    ):
        raise ValueError("unsupported wave-family conversion")
    return _native.expansion(
        list(destination.modes),
        list(source.modes),
        destination.positions.tolist(),
        source.positions.tolist(),
        (complex(values[0]), complex(values[1])),
        poltype == "helicity",
        singular,
    )


def rotation(
    angles: ArrayLike,
    destination: SphericalWaveBasis | CylindricalWaveBasis,
    source: SphericalWaveBasis | CylindricalWaveBasis | None = None,
) -> tuple[NDArray[np.complex128], _native.RotationContext]:
    """Native z-y-z rotation; pullback returns the three Euler-angle cotangents.

    Cylindrical bases permit only theta=0, which remains a fixed constraint.
    Origins are local expansion labels and are not moved by this operator.
    """
    source = destination if source is None else source
    values = np.asarray(angles, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("rotation requires three Euler angles")
    args = (
        destination.positions.tolist(),
        source.positions.tolist(),
        (float(values[0]), float(values[1]), float(values[2])),
    )
    if isinstance(destination, SphericalWaveBasis) and isinstance(
        source, SphericalWaveBasis
    ):
        return _native.rotation(list(destination.modes), list(source.modes), *args)
    if isinstance(destination, CylindricalWaveBasis) and isinstance(
        source, CylindricalWaveBasis
    ):
        return _native.cyl_rotation(list(destination.modes), list(source.modes), *args)
    raise ValueError("rotation bases must belong to the same wave family")


def field_operator(
    points: ArrayLike,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    ks: ArrayLike,
    *,
    poltype: str = "helicity",
    singular: bool = False,
) -> tuple[NDArray[np.complex128], _native.FieldOperatorContext]:
    """Field matrix (samples, 3, modes); VJP returns (points, origins, ks)."""
    values = np.asarray(ks, dtype=np.complex128)
    if values.shape != (2,) or poltype not in ("helicity", "parity"):
        raise ValueError("require two medium wavenumbers and a valid polarization type")
    args = (
        basis.positions.tolist(),
        np.ascontiguousarray(points, dtype=np.float64),
        (complex(values[0]), complex(values[1])),
        poltype == "helicity",
        singular,
    )
    if isinstance(basis, CylindricalWaveBasis):
        return _native.cylindrical_field_operator(list(basis.modes), *args)
    return _native.field_operator(list(basis.modes), *args)


def field(
    coefficients: ArrayLike,
    points: ArrayLike,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    ks: ArrayLike,
    *,
    poltype: str = "helicity",
    singular: bool = False,
) -> tuple[NDArray[np.complex128], _native.FieldContext]:
    """Electric samples (N, 3); VJP returns (coefficients, points, origins, ks).

    Inputs are multipole amplitudes, Cartesian points (N, 3), a multipole basis,
    and negative/positive helicity wavenumbers. The residual uses linear storage.
    Cylindrical axial wavenumbers remain fixed mode labels.
    """
    values = np.asarray(ks, dtype=np.complex128)
    if values.shape != (2,):
        raise ValueError("ks must contain negative and positive helicity wavenumbers")
    if poltype not in ("helicity", "parity"):
        raise ValueError("invalid polarization type")
    args = (
        basis.positions.tolist(),
        np.ascontiguousarray(coefficients, dtype=np.complex128),
        np.ascontiguousarray(points, dtype=np.float64),
        (complex(values[0]), complex(values[1])),
        poltype == "helicity",
        singular,
    )
    if isinstance(basis, CylindricalWaveBasis):
        return _native.cylindrical_field(list(basis.modes), *args)
    return _native.field(list(basis.modes), *args)


def cylinder(
    kzs: ArrayLike,
    mmax: int,
    k0: float,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike | None = None,
    kappa: ArrayLike | None = None,
) -> tuple[NDArray[np.complex128], _native.CylinderMatrixContext]:
    """Cylinder T-matrix; VJP returns (kzs, k0, radii, epsilon, mu, kappa)."""
    eps = np.ascontiguousarray(epsilon, dtype=np.complex128)
    return _native.cylinder(
        np.ascontiguousarray(kzs, dtype=np.float64),
        mmax,
        k0,
        np.ascontiguousarray(radii, dtype=np.float64),
        eps,
        np.ones_like(eps)
        if mu is None
        else np.ascontiguousarray(mu, dtype=np.complex128),
        np.zeros_like(eps)
        if kappa is None
        else np.ascontiguousarray(kappa, dtype=np.complex128),
    )


def plane_field(
    coefficients: ArrayLike | None,
    points: ArrayLike,
    vectors: ArrayLike,
    polarizations: ArrayLike,
    *,
    poltype: str = "helicity",
    fixed_vectors: bool = False,
) -> tuple[NDArray[np.complex128], _native.PlaneFieldContext]:
    """Weighted Cartesian plane fields or their full operator when coefficients=None.

    Pullback returns (amplitudes, points, full complex wavevectors). The amplitude
    gradient is empty for an operator. Use fixed_vectors at the polarization axis,
    where a full direction derivative is undefined in the upstream convention.
    """
    if poltype not in ("helicity", "parity"):
        raise ValueError("invalid polarization type")
    pols = np.asarray(polarizations)
    if not np.all((pols == 0) | (pols == 1)):
        raise ValueError("polarizations must be 0 or 1")
    return _native.plane_field(
        np.ascontiguousarray(vectors, dtype=np.complex128),
        pols.astype(np.int64).tolist(),
        np.ascontiguousarray(points, dtype=np.float64),
        None
        if coefficients is None
        else np.ascontiguousarray(coefficients, dtype=np.complex128),
        poltype == "helicity",
        fixed_vectors,
    )


def plane_expansion(
    destination: SphericalWaveBasis | CylindricalWaveBasis,
    vectors: ArrayLike,
    polarizations: ArrayLike,
    *,
    poltype: str = "helicity",
    fixed_vectors: bool = False,
) -> tuple[NDArray[np.complex128], _native.PlaneExpansionContext]:
    """Regular plane-to-multipole expansion; VJP returns (origins, wavevectors).

    Cylindrical axial components are fixed labels with zero cotangents.
    At axial propagation, fixed_vectors enables origin gradients at fixed incidence.
    """
    if poltype not in ("helicity", "parity"):
        raise ValueError("invalid polarization type")
    pols = np.asarray(polarizations)
    if not np.all((pols == 0) | (pols == 1)):
        raise ValueError("polarizations must be 0 or 1")
    if isinstance(destination, CylindricalWaveBasis):
        return _native.cylindrical_plane_expansion(
            list(destination.modes),
            destination.positions.tolist(),
            np.asarray(vectors, dtype=np.complex128).tolist(),
            pols.astype(np.int64).tolist(),
            poltype == "helicity",
            fixed_vectors,
        )
    return _native.plane_expansion(
        list(destination.modes),
        destination.positions.tolist(),
        np.asarray(vectors, dtype=np.complex128).tolist(),
        pols.astype(np.int64).tolist(),
        poltype == "helicity",
        fixed_vectors,
    )


def cylindrical_channels(
    basis: CylindricalWaveBasis,
    ks: ArrayLike,
    q: ArrayLike,
    polarizations: ArrayLike,
    period: float,
    *,
    poltype: str = "helicity",
    fixed_q: bool = False,
) -> tuple[NDArray[np.complex128], _native.ChannelsContext]:
    """Cylindrical incidence/emission channels for a periodic array along x.

    q=(kz,kx) identifies zx-aligned plane modes; up/down refers to +/-y.
    Pullback returns (origins, ks, q, period). Axial kz labels are fixed, so
    q[:,0] cotangents are zero. fixed_q also holds kx constant.
    """
    if poltype not in ("helicity", "parity"):
        raise ValueError("invalid polarization type")
    pols = np.asarray(polarizations)
    if not np.all((pols == 0) | (pols == 1)):
        raise ValueError("polarizations must be 0 or 1")
    return _native.cylindrical_channels(
        list(basis.modes),
        basis.positions.tolist(),
        np.asarray(ks, dtype=np.complex128).tolist(),
        np.asarray(q, dtype=np.float64).tolist(),
        pols.astype(np.int64).tolist(),
        period,
        poltype == "helicity",
        fixed_q,
    )


def interface(
    ks: ArrayLike,
    zs: ArrayLike,
    q: ArrayLike,
    *,
    alignment: str = "xy",
    fixed_q: bool = False,
) -> tuple[NDArray[np.complex128], _native.InterfaceContext]:
    """Cartesian interface; VJP returns (two-media wavenumbers, impedances, q).

    Wavenumbers have shape (2, 2), ordered below/above then polarization 0/1.
    Transverse components follow alignment xy/yz/zx; the remaining axis is normal.
    """
    if alignment not in ("xy", "yz", "zx"):
        raise ValueError("interface alignment must be xy, yz or zx")
    return _native.interface(
        np.asarray(ks, dtype=np.complex128).tolist(),
        np.asarray(zs, dtype=np.complex128).tolist(),
        np.asarray(q, dtype=np.float64).tolist(),
        {"xy": 2, "yz": 0, "zx": 1}[alignment],
        fixed_q,
    )


def layer_stack(
    ks: ArrayLike,
    zs: ArrayLike,
    q: ArrayLike,
    thickness: ArrayLike,
    *,
    alignment: str = "xy",
    fixed_q: bool = False,
) -> tuple[NDArray[np.complex128], _native.LayersContext]:
    """Independent planar channels, shape (channels, 2, 2, 2, 2).

    Final axes are outgoing/incoming direction then polarization 0/1.
    Media run below to above; thickness has one entry per interior medium.
    VJP returns (wavenumbers, impedances, transverse components, thickness).
    """
    if alignment not in ("xy", "yz", "zx"):
        raise ValueError("layer alignment must be xy, yz or zx")
    return _native.layer_stack(
        np.asarray(ks, dtype=np.complex128).tolist(),
        np.asarray(zs, dtype=np.complex128).tolist(),
        np.asarray(q, dtype=np.float64).tolist(),
        np.atleast_1d(np.asarray(thickness, dtype=np.float64)).tolist(),
        {"xy": 2, "yz": 0, "zx": 1}[alignment],
        fixed_q,
    )


def periodic_conversion(
    destination: CylindricalWaveBasis,
    source: SphericalWaveBasis,
    ks: ArrayLike,
    period: float,
    *,
    poltype: str = "helicity",
) -> tuple[NDArray[np.complex128], _native.PeriodicConversionContext]:
    """Spherical z-periodic radiation into outgoing cylindrical waves.

    VJP returns (destination origins, source origins, ks, destination kz, period).
    Each cylindrical mode's real kz is differentiable independently; physical
    diffraction orders satisfy kz=kpar+2*pi*n/period.
    """
    if poltype not in ("helicity", "parity"):
        raise ValueError("invalid polarization type")
    return _native.periodic_conversion(
        list(destination.modes),
        list(source.modes),
        destination.positions.tolist(),
        source.positions.tolist(),
        np.asarray(ks, dtype=np.complex128).tolist(),
        period,
        poltype == "helicity",
    )

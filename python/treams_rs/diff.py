"""Framework-neutral native operations; all pullbacks use the real Hermitian pairing."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, cast

import numpy as np

from . import _native
from ._core import CylindricalWaveBasis, SphericalWaveBasis
from .config import _resolve_poltype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from .ebcm import Modes


def bessel(
    order: ArrayLike,
    z: ArrayLike,
    *,
    kind: str = "j",
    spherical: bool = False,
    derivative: bool = False,
) -> tuple[NDArray[np.complex128], _native.BesselContext]:
    """Broadcast Bessel values/first derivatives and a complex-argument pullback.

    Order is held fixed. kind is j, y, h1 or h2. The pullback reduces broadcast
    axes to the original z shape; scalar arguments retain only one native value.
    """
    if isinstance(order, (int, float)) and isinstance(z, (int, float, complex)):
        return _native.bessel_scalar(order, z, kind, spherical, int(derivative))
    orders, arguments, shape, argument_shape = _bessel_inputs(order, z)
    return _native.bessel(
        orders, arguments, kind, spherical, int(derivative), shape, argument_shape
    )


def _bessel_inputs(
    order: ArrayLike, z: ArrayLike
) -> tuple[
    NDArray[np.float64], NDArray[np.complex128], tuple[int, ...], tuple[int, ...]
]:
    orders = np.asarray(order, dtype=np.float64)
    arguments = np.asarray(z, dtype=np.complex128)
    degrees, values = (
        (orders, arguments)
        if orders.shape == arguments.shape
        or (orders.size == 1 and orders.ndim <= arguments.ndim)
        else np.broadcast_arrays(orders, arguments)
    )
    return (
        (orders if orders.size == 1 else degrees).ravel(),
        (arguments if arguments.size == 1 else values).ravel(),
        values.shape,
        arguments.shape,
    )


def lattice_sum(
    dim: int,
    degree: ArrayLike,
    order: ArrayLike,
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    *,
    spherical: bool = True,
    part: str = "full",
    shell: ArrayLike = 0,
) -> tuple[NDArray[np.complex128], _native.LatticeSumContext]:
    """Broadcast lattice sums and k/kpar/a/r/eta pullbacks, in that order.

    Row lattice vectors form a square matrix (or a scalar in 1D). Spherical
    shifts have three Cartesian coordinates, cylindrical shifts have two.
    Degree/order and direct-shell index remain fixed. Component adjoints require
    an explicit nonzero eta; the full sum's arbitrary split has zero derivative.
    """
    if not 1 <= dim <= (3 if spherical else 2):
        raise ValueError("invalid lattice dimension")
    try:
        component = ("full", "real", "reciprocal", "direct").index(part)
    except ValueError:
        raise ValueError("part must be full, real, reciprocal or direct") from None
    arguments = (
        np.asarray(k, dtype=np.complex128),
        np.asarray(kpar, dtype=np.float64),
        np.asarray(a, dtype=np.float64),
        np.asarray(r, dtype=np.float64),
        np.asarray(eta, dtype=np.complex128),
    )
    shapes = [argument.shape for argument in arguments]
    wavenumbers, bloch, cells, shifts, etas = arguments
    if dim == 1:
        if bloch.shape[-1:] != (1,):
            bloch = bloch[..., None]
        if cells.shape[-2:] != (1, 1):
            cells = cells[..., None, None]
    coordinates = 3 if spherical else 2
    if (
        bloch.shape[-1:] != (dim,)
        or cells.shape[-2:] != (dim, dim)
        or shifts.shape[-1:] != (coordinates,)
    ):
        raise ValueError(
            "lattice, Bloch vector and Cartesian shift dimensions must agree"
        )
    degrees, orders, shells = (
        np.asarray(value, dtype=np.float64) for value in (degree, order, shell)
    )
    if any(
        not v.item().is_integer()
        if v.ndim == 0
        else (~np.isfinite(v) | (v != np.floor(v))).any()
        for v in (degrees, orders, shells)
    ):
        raise ValueError("degree, order and shell must be finite integers")
    if (
        (np.abs(degrees) > 128).any()
        or (np.abs(orders) > 128).any()
        or ((shells < 0) | (shells > np.iinfo(np.int32).max)).any()
    ):
        raise ValueError("lattice labels or shell exceed supported bounds")
    shape = np.broadcast_shapes(
        degrees.shape,
        orders.shape,
        shells.shape,
        wavenumbers.shape,
        bloch.shape[:-1],
        cells.shape[:-2],
        shifts.shape[:-1],
        etas.shape,
    )

    def pack[T: np.generic](
        value: NDArray[T], core: tuple[int, ...] = ()
    ) -> NDArray[T]:
        outer = value.shape[: value.ndim - len(core)]
        return (
            value
            if outer == shape or prod(outer) == 1
            else np.broadcast_to(value, shape + core)
        ).reshape((-1, *core))

    mode_degree, mode_order = (
        (degrees.ravel(), orders.ravel())
        if degrees.size == orders.size == 1
        else (
            np.broadcast_to(degrees, shape).ravel(),
            np.broadcast_to(orders, shape).ravel(),
        )
    )
    modes = [
        (int(degree), int(order))
        for degree, order in zip(mode_degree, mode_order, strict=True)
    ]
    return _native.lattice_record(
        spherical,
        dim,
        modes,
        pack(wavenumbers),
        pack(bloch, (dim,)),
        pack(cells, (dim, dim)),
        pack(shifts, (coordinates,)),
        pack(etas),
        component,
        pack(shells).astype(np.int64).tolist(),
        shape,
        shapes,
    )


def incgamma(
    n: ArrayLike, z: ArrayLike
) -> tuple[NDArray[np.complex128], _native.GammaContext]:
    """Upper incomplete gamma and a native z pullback, with fixed half-integer degree."""
    if isinstance(n, (int, float)) and isinstance(z, (int, float, complex)):
        return _native.gamma_record_scalar(n, z)
    degrees, arguments, shape, argument_shape = _bessel_inputs(n, z)
    return _native.gamma_record(degrees, arguments, shape, argument_shape)


def intkambe(
    n: ArrayLike, z: ArrayLike, eta: ArrayLike
) -> tuple[NDArray[np.complex128], _native.KambeContext]:
    """Kambe integral and native z/eta pullbacks; integer order stays fixed.

    Derivatives follow the selected complex branch. The n=-2 cusp at z=0
    is singular; n<=-3 has a zero first z derivative there.
    """
    if (
        isinstance(n, int)
        and isinstance(z, (int, float, complex))
        and isinstance(eta, (int, float, complex))
    ):
        return _native.kambe_record_scalar(n, z, eta)
    arguments = (
        np.asarray(z, dtype=np.complex128),
        np.asarray(eta, dtype=np.complex128),
    )
    if isinstance(n, (int, np.integer)):
        if not -260 <= n <= 260:
            raise ValueError("Kambe orders must be integers in [-260, 260]")
        orders = np.asarray(n, dtype=np.int32)
    else:
        orders = np.asarray(n, dtype=np.float64)
        if np.any(
            ~np.isfinite(orders) | (orders != np.floor(orders)) | (np.abs(orders) > 260)
        ):
            raise ValueError("Kambe orders must be integers in [-260, 260]")
        orders = orders.astype(np.int32)
    arrays = (orders, *arguments)
    shape = max((a.shape for a in arrays), key=len)
    if not all(
        a.shape == shape or (a.size == 1 and a.ndim <= len(shape)) for a in arrays
    ):
        shape = np.broadcast_shapes(*(a.shape for a in arrays))
    return _native.kambe_record(
        (
            orders
            if orders.size == 1 or orders.shape == shape
            else np.broadcast_to(orders, shape)
        ).ravel(),
        (
            arguments[0]
            if arguments[0].size == 1 or arguments[0].shape == shape
            else np.broadcast_to(arguments[0], shape)
        ).ravel(),
        (
            arguments[1]
            if arguments[1].size == 1 or arguments[1].shape == shape
            else np.broadcast_to(arguments[1], shape)
        ).ravel(),
        shape,
        (arguments[0].shape, arguments[1].shape),
    )


def angular(
    degree: ArrayLike,
    order: ArrayLike,
    z: ArrayLike,
    *,
    kind: str = "legendre",
) -> tuple[NDArray[np.complex128], _native.AngularContext]:
    """Integer-degree Legendre/pi/tau values and an argument VJP; labels stay fixed.

    kind is legendre, pi or tau. A derivative at a branch point raises ValueError
    unless its cotangent is zero. Broadcast axes reduce to the original z shape.
    """
    if (
        isinstance(degree, (int, float))
        and isinstance(order, (int, float))
        and isinstance(z, (int, float, complex))
    ):
        return _native.angular_scalar(degree, order, z, kind)
    degrees = np.asarray(degree, dtype=np.float64)
    orders = np.asarray(order, dtype=np.float64)
    arguments = np.asarray(z, dtype=np.complex128)
    arrays = (degrees, orders, arguments)
    broadcast = (
        arrays
        if degrees.shape == orders.shape == arguments.shape
        or (
            degrees.size == orders.size == 1
            and max(degrees.ndim, orders.ndim) <= arguments.ndim
        )
        else np.broadcast_arrays(*arrays)
    )
    return _native.angular(
        (degrees if degrees.size == 1 else broadcast[0]).ravel(),
        (orders if orders.size == 1 else broadcast[1]).ravel(),
        (arguments if arguments.size == 1 else broadcast[2]).ravel(),
        kind,
        broadcast[2].shape,
        arguments.shape,
    )


def wigner(
    degree: ArrayLike,
    row: ArrayLike,
    column: ArrayLike,
    phi: ArrayLike,
    theta: ArrayLike,
    psi: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.WignerContext]:
    """Broadcast Wigner D elements and native pullbacks to all three Euler angles.

    Integer degree/row/column labels stay fixed; angles may be complex. Each
    pullback reduces to its original input shape and owns its forward inputs.
    """
    if (
        isinstance(degree, int)
        and isinstance(row, int)
        and isinstance(column, int)
        and isinstance(phi, (int, float, complex))
        and isinstance(theta, (int, float, complex))
        and isinstance(psi, (int, float, complex))
    ):
        return _native.wigner_scalar(
            (degree, row, column), (complex(phi), complex(theta), complex(psi))
        )
    labels = tuple(np.asarray(v, dtype=np.float64) for v in (degree, row, column))
    if any(
        np.any(~np.isfinite(v) | (v != np.floor(v)) | (np.abs(v) > 260)) for v in labels
    ):
        raise ValueError("Wigner labels must be integers in [-260, 260]")
    angles = tuple(np.asarray(v, dtype=np.complex128) for v in (phi, theta, psi))
    arrays = (*labels, *angles)
    fixed = all(
        v.size == 1 and v.ndim <= angles[1].ndim
        for v in (*labels, angles[0], angles[2])
    )
    broadcast = (
        arrays
        if fixed or all(v.shape == arrays[0].shape for v in arrays)
        else np.broadcast_arrays(*arrays)
    )
    modes = (
        [tuple(int(v.item()) for v in labels)]
        if all(v.size == 1 for v in labels)
        else [
            tuple(int(v) for v in mode)
            for mode in zip(*(v.flat for v in broadcast[:3]), strict=True)
        ]
    )
    return _native.wigner(
        [(mode[0], mode[1], mode[2]) for mode in modes],
        cast(
            "NDArray[np.complex128]", angles[0] if angles[0].size == 1 else broadcast[3]
        ).ravel(),
        cast(
            "NDArray[np.complex128]", angles[1] if angles[1].size == 1 else broadcast[4]
        ).ravel(),
        cast(
            "NDArray[np.complex128]", angles[2] if angles[2].size == 1 else broadcast[5]
        ).ravel(),
        broadcast[4].shape,
        (angles[0].shape, angles[1].shape, angles[2].shape),
    )


def chirality_density(
    ks: ArrayLike, normal: ArrayLike, z: ArrayLike = (0.0, 0.0)
) -> tuple[NDArray[np.complex128], _native.ChiralityContext]:
    """Compact (3, modes) up/down/cross density coefficients and native pullback.

    Multiply by helicity signs, or pair opposite parity modes, to form operators.
    The cross form contracts as Re(down.conj() @ cross @ up).
    Real transverse wavevectors in an xy basis are assumed; ks, normal and z vary.
    """
    start, stop = np.asarray(z, dtype=np.float64)
    return _native.chirality_density(
        np.ascontiguousarray(ks, dtype=np.complex128),
        np.ascontiguousarray(normal, dtype=np.complex128),
        (float(start), float(stop)),
    )


def oriented_chirality(
    transverse: ArrayLike,
    normal: ArrayLike,
    z: ArrayLike = (0.0, 0.0),
    *,
    polarizations: ArrayLike,
    axis: int = 2,
) -> tuple[NDArray[np.complex128], _native.OrientedChiralityContext]:
    """Signed helicity (3, modes) chirality forms for any Cartesian normal.

    Transverse components are real, shape (modes, 2), in cyclic order after axis.
    The native pullback covers transverse/complex normal components and endpoints.
    """
    start, stop = np.asarray(z, dtype=np.float64)
    return _native.oriented_chirality(
        np.asarray(transverse, dtype=np.float64),
        np.asarray(normal, dtype=np.complex128),
        np.asarray(polarizations).tolist(),
        axis,
        (float(start), float(stop)),
    )


def ebcm_qmat(
    radii: ArrayLike,
    slopes: ArrayLike,
    ks: ArrayLike,
    zs: ArrayLike,
    *,
    theta: ArrayLike,
    weights: ArrayLike,
    out: Modes,
    in_: Modes | None = None,
    singular: bool = True,
    legacy: bool = False,
) -> tuple[NDArray[np.complex128], _native.QContext]:
    """Axisymmetric Q matrix and native sampled-radius/slope/medium pullback.

    Polar quadrature nodes and integration weights are held fixed in reverse.
    legacy=True omits the radial area factor, reproducing treams' integral.
    """
    from .ebcm import _modes

    zm, zp = np.asarray(zs, dtype=np.complex128)
    return _native.ebcm_qmat(
        np.column_stack([theta, weights, radii, slopes]).astype(np.float64),
        _modes(out),
        _modes(out if in_ is None else in_),
        np.ascontiguousarray(ks, dtype=np.complex128),
        (complex(zm), complex(zp)),
        singular,
        legacy,
    )


def tmatrix_metric(
    operator: ArrayLike,
    ks: ArrayLike = (1.0, 1.0),
    *,
    polarizations: ArrayLike,
    kind: str,
) -> tuple[float, _native.MetricContext]:
    """Global helicity cd/db/chi and native matrix/real-wavenumber pullback.

    Wavenumbers only affect cd. A zero scattering norm, or zero total absorption
    for cd, is undefined. Chi at zero contrast has a value but no nonzero VJP.
    """
    km, kp = np.asarray(ks, dtype=np.float64)
    return _native.tmatrix_metric(
        np.ascontiguousarray(operator, dtype=np.complex128),
        np.asarray(polarizations).tolist(),
        (float(km), float(kp)),
        kind,
    )


def svdvals(operator: ArrayLike) -> tuple[NDArray[np.float64], _native.SingularContext]:
    """Descending singular values and native matrix VJP using thin singular vectors.

    Repeated positive values require equal weights. Zero singular values require
    zero weights; individual values there are not differentiable.
    """
    return _native.svdvals(np.ascontiguousarray(operator, dtype=np.complex128))


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


def smatrix_tr(
    matrices: ArrayLike,
    incident: ArrayLike,
    ks: ArrayLike,
    zs: ArrayLike,
    q: ArrayLike,
    *,
    modes: Sequence[tuple[int, int]],
    axis: int = 2,
    poltype: str | None = None,
    modetype: str = "up",
    fixed_q: bool = False,
) -> tuple[NDArray[np.float64], _native.TransmissionContext]:
    """Power transmission/reflection with all matrix, illumination and port gradients.

    Incident columns are independent illuminations; output rows are T and R.
    ks has shape (2,2), zs (2,), and q (groups,2). Each static mode is
    (diffraction group index, polarization). Groups must have distinct q values.
    Pullback returns matrices, incident, ks, zs, q. Normal incidence requires
    fixed_q=True because the plane polarization gauge has no direction derivative.
    """
    poltype = _resolve_poltype(poltype)
    if modetype not in ("up", "down"):
        raise ValueError("invalid polarization or propagation direction")
    value, context = _native.smatrix_transmittance(
        np.asarray(matrices, dtype=np.complex128),
        np.asarray(incident, dtype=np.complex128),
        np.asarray(ks, dtype=np.complex128).tolist(),
        np.asarray(zs, dtype=np.complex128).tolist(),
        np.asarray(q, dtype=np.float64).tolist(),
        modes,
        axis,
        poltype == "helicity",
        0 if modetype == "up" else 1,
        fixed_q,
        True,
    )
    assert context is not None
    return value, context


def smatrix_illuminate(
    lower: ArrayLike, upper: ArrayLike, up: ArrayLike, down: ArrayLike
) -> tuple[NDArray[np.complex128], _native.IlluminationContext]:
    """Outgoing up/down and internal up/down coefficients of two adjacent stacks.

    Inputs up/down have shape (modes, illuminations); output shape is
    (4, modes, illuminations). The native solve only computes these right-hand
    sides. Pullback returns lower, upper, up and down cotangents.
    """
    return _native.smatrix_illuminate(
        np.asarray(lower, dtype=np.complex128),
        np.asarray(upper, dtype=np.complex128),
        np.asarray(up, dtype=np.complex128),
        np.asarray(down, dtype=np.complex128),
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
    poltype: str | None = None,
    fixed_q: bool = False,
) -> tuple[NDArray[np.complex128], _native.ChannelsContext]:
    """Incident/emitted, up/down arrays, shaped (2, 2, multipoles, plane modes).

    Emitted arrays are transposed: ``channels[1, side].T`` maps multipoles to
    outgoing plane waves. VJP returns (positions, ks, q, area). At exactly normal
    incidence the azimuth is undefined; set fixed_q for derivatives at fixed incidence.
    """
    poltype = _resolve_poltype(poltype)
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


def particle_cluster(
    local: Sequence[ArrayLike],
    positions: ArrayLike,
    ks: ArrayLike,
    *,
    bases: Sequence[SphericalWaveBasis | CylindricalWaveBasis],
    poltype: str | None = None,
) -> tuple[NDArray[np.complex128], _native.ParticleClusterContext]:
    """Heterogeneous particles; VJP returns (local matrices, positions, ks).

    Local bases can have different cutoffs and mode subsets. They must each use
    one origin. Particles must have non-overlapping enclosing surfaces.
    """
    poltype = _resolve_poltype(poltype)
    if len(local) != len(bases) or not bases:
        raise ValueError("one local matrix and basis required per particle")
    family = type(bases[0])
    modes: list[tuple[int, float, int, int]] = []
    arrays = []
    for particle, (value, basis) in enumerate(zip(local, bases, strict=True)):
        if type(basis) is not family or not basis.isglobal:
            raise ValueError("local particles require global bases of one wave family")
        array = np.asarray(value, dtype=np.complex128)
        if array.shape != (len(basis), len(basis)):
            raise ValueError("local matrix shape must match its basis")
        arrays.append(array)
        modes.extend((particle, degree, order, pol) for _, degree, order, pol in basis)
    km, kp = np.asarray(ks, dtype=np.complex128)
    points = np.asarray(positions, dtype=np.float64).tolist()
    wave_numbers = (complex(km), complex(kp))
    if family is SphericalWaveBasis:
        return _native.particle_cluster(
            arrays,
            [(p, int(degree), m, pol) for p, degree, m, pol in modes],
            points,
            wave_numbers,
            poltype == "helicity",
        )
    return _native.cylindrical_particle_cluster(
        arrays, modes, points, wave_numbers, poltype == "helicity"
    )


def factor_interaction(
    local: ArrayLike, coupling: ArrayLike
) -> _native.InteractionFactor:
    """Own and factor I - T C once, then solve only requested incident columns."""
    return _native.InteractionFactor(
        np.asarray(local, dtype=np.complex128),
        np.asarray(coupling, dtype=np.complex128),
    )


def cluster_factor(
    lmax: int, k0: float, radii: ArrayLike, epsilon: ArrayLike, positions: ArrayLike
) -> _native.InteractionFactor:
    """Native sphere-pair assembly and reusable requested-illumination LU.

    Same homogeneous nonmagnetic vacuum spheres as cluster(). This static factor
    avoids a full local block-diagonal array and full interacting T-matrix. Its
    recorded VJP returns local-block, coupling and incident cotangents; use the
    iterative sphere solver for physical radius/material/position pullbacks.
    """
    return _native.cluster_factor(
        lmax,
        k0,
        np.ascontiguousarray(radii, dtype=np.float64),
        np.ascontiguousarray(epsilon, dtype=np.complex128),
        np.asarray(positions, dtype=np.float64),
    )


def factor_interaction_blocks(
    local: Sequence[ArrayLike], coupling: ArrayLike
) -> _native.InteractionFactor:
    """Reusable factor retaining local particle matrices as separate dense blocks."""
    return _native.InteractionFactor.from_blocks(
        [np.asarray(block, dtype=np.complex128) for block in local],
        np.asarray(coupling, dtype=np.complex128),
    )


def illuminate(
    local: ArrayLike, coupling: ArrayLike, incident: ArrayLike
) -> tuple[NDArray[np.complex128], _native.InteractionIlluminationContext]:
    """Solve (I - T C) scattered = T incident for channel-by-illumination columns.

    The VJP returns (local, coupling, incident) cotangents. No full interacting
    T-matrix is computed. Use factor_interaction to reuse the LU across calls.
    """
    return factor_interaction(local, coupling).record(
        np.asarray(incident, dtype=np.complex128)
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
    poltype: str | None = None,
    singular: bool = False,
) -> tuple[NDArray[np.complex128], _native.ExpansionContext]:
    """Expansion VJP returns (destination positions, source positions, ks).

    Cylindrical ``context.pullback_axial`` adds a fourth gradient array, ordered
    by sorted distinct axial wavenumbers from both bases. Each derivative moves
    all modes with that shared label together; different groups remain distinct.
    """
    poltype = _resolve_poltype(poltype)
    values = np.asarray(ks, dtype=np.complex128)
    if values.shape != (2,):
        raise ValueError("ks must contain negative and positive helicity wavenumbers")
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
    poltype: str | None = None,
    singular: bool = False,
) -> tuple[NDArray[np.complex128], _native.FieldOperatorContext]:
    """Field matrix (samples, 3, modes); VJP returns (points, origins, ks).

    For cylindrical bases, ``context.pullback_axial`` additionally returns real
    per-mode axial-wavenumber gradients as the last array.
    """
    poltype = _resolve_poltype(poltype)
    values = np.asarray(ks, dtype=np.complex128)
    if values.shape != (2,):
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
    poltype: str | None = None,
    singular: bool = False,
) -> tuple[NDArray[np.complex128], _native.FieldContext]:
    """Electric samples (N, 3); VJP returns (coefficients, points, origins, ks).

    Inputs are multipole amplitudes, Cartesian points (N, 3), a multipole basis,
    and negative/positive helicity wavenumbers. The residual uses linear storage.
    For cylindrical bases, ``context.pullback_axial`` additionally returns real
    per-mode axial-wavenumber gradients as the last array.
    """
    poltype = _resolve_poltype(poltype)
    values = np.asarray(ks, dtype=np.complex128)
    if values.shape != (2,):
        raise ValueError("ks must contain negative and positive helicity wavenumbers")
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


def plane_phases(
    points: ArrayLike, vectors: ArrayLike
) -> tuple[NDArray[np.complex128], _native.PlanePhaseContext]:
    """exp(i k.r) for (P, 3) real displacements and (N, 3) complex wavevectors.

    Returns a (P, N) array and an inputs-only native context. Pullback returns
    displacement and complex-wavevector cotangents; axial vectors are supported.
    """
    return _native.plane_phases(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(vectors, dtype=np.complex128),
    )


def plane_field(
    coefficients: ArrayLike | None,
    points: ArrayLike,
    vectors: ArrayLike,
    polarizations: ArrayLike,
    *,
    poltype: str | None = None,
    fixed_vectors: bool = False,
) -> tuple[NDArray[np.complex128], _native.PlaneFieldContext]:
    """Weighted Cartesian plane fields or their full operator when coefficients=None.

    Pullback returns (amplitudes, points, full complex wavevectors). The amplitude
    gradient is empty for an operator. Use fixed_vectors at the polarization axis,
    where a full direction derivative is undefined in the upstream convention.
    """
    poltype = _resolve_poltype(poltype)
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
    poltype: str | None = None,
    fixed_vectors: bool = False,
) -> tuple[NDArray[np.complex128], _native.PlaneExpansionContext]:
    """Regular plane-to-multipole expansion; VJP returns (origins, wavevectors).

    Cylindrical axial components are fixed labels with zero cotangents.
    At axial propagation, fixed_vectors enables origin gradients at fixed incidence.
    """
    poltype = _resolve_poltype(poltype)
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
    poltype: str | None = None,
    fixed_q: bool = False,
) -> tuple[NDArray[np.complex128], _native.ChannelsContext]:
    """Cylindrical incidence/emission channels for a periodic array along x.

    q=(kz,kx) identifies zx-aligned plane modes; up/down refers to +/-y.
    Pullback returns (origins, ks, q, period). Axial kz labels are fixed, so
    q[:,0] cotangents are zero. fixed_q also holds kx constant.
    """
    poltype = _resolve_poltype(poltype)
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
    poltype: str | None = None,
) -> tuple[NDArray[np.complex128], _native.PeriodicConversionContext]:
    """Spherical z-periodic radiation into outgoing cylindrical waves.

    VJP returns (destination origins, source origins, ks, destination kz, period).
    Each cylindrical mode's real kz is differentiable independently; physical
    diffraction orders satisfy kz=kpar+2*pi*n/period.
    """
    poltype = _resolve_poltype(poltype)
    return _native.periodic_conversion(
        list(destination.modes),
        list(source.modes),
        destination.positions.tolist(),
        source.positions.tolist(),
        np.asarray(ks, dtype=np.complex128).tolist(),
        period,
        poltype == "helicity",
    )


def plane_permutation(
    vectors: ArrayLike,
    polarizations: ArrayLike,
    n: int = 1,
    *,
    poltype: str | None = None,
) -> tuple[NDArray[np.complex128], _native.PlanePermutationContext]:
    """Cyclic Cartesian-axis change, with both output polarizations per input mode.

    Returns coefficients of shape (2, modes). The new wavevectors are
    np.roll(vectors, n, axis=1). The native pullback differentiates complex vectors.
    """
    poltype = _resolve_poltype(poltype)
    if n != int(n):
        raise ValueError("number of permutations must be integer")
    return _native.plane_permutation(
        np.asarray(vectors, dtype=np.complex128),
        np.asarray(polarizations, dtype=np.float64),
        int(n) % 3,
        poltype == "helicity",
    )


def coordinates(
    points: ArrayLike, *, kind: str
) -> tuple[NDArray[np.float64], _native.CoordinateContext]:
    """Coordinate conversion with real input VJP; kind is e.g. car2sph.

    The final axis has two polar or three spatial components. Undefined angular
    derivatives at an axis/origin raise unless their output cotangent is zero.
    """
    return _native.coordinates(np.asarray(points, dtype=np.float64), kind)


def vector_coordinates(
    vectors: ArrayLike, points: ArrayLike, *, kind: str
) -> tuple[NDArray[np.complex128], _native.VectorCoordinateContext]:
    """Transform vector components; VJP returns (vectors, source coordinates).

    kind is e.g. car2sph, without the public vector-function's v prefix. Vector
    and point batch dimensions broadcast; gradients return their original shapes.
    """
    vector = np.asarray(vectors, dtype=np.complex128)
    position = np.asarray(points, dtype=np.float64)
    dim = 2 if kind in ("car2pol", "pol2car") else 3
    if (
        not vector.shape
        or not position.shape
        or vector.shape[-1] != dim
        or position.shape[-1] != dim
    ):
        raise ValueError("last axes must match coordinate dimension")
    v, p = np.broadcast_arrays(vector, position)
    return _native.vector_coordinates(
        vector if vector.size == dim else v,
        position if position.size == dim else cast("NDArray[np.float64]", p),
        kind,
        v.shape,
        (vector.shape, position.shape),
    )


def vector_wave(
    *arguments: ArrayLike,
    kind: str,
    degree: ArrayLike = 0,
    order: ArrayLike = 0,
    polarization: ArrayLike = 0,
) -> tuple[NDArray[np.complex128], _native.WaveContext]:
    """Low-level vector wave with native VJPs to every continuous argument.

    kind is a special function name (e.g. vsw_rA). Degree, order and polarization
    are fixed labels. Arguments have the public function's order after removing
    those labels: (kr, theta, phi) for spherical, (theta, phi) for harmonics,
    (kz, krr, phi, z[, k]) for cylindrical and (kx, ky, kz, x, y, z) for plane waves.
    Pullback returns one gradient per argument, reduced to its original shape.
    """
    values = tuple(np.asarray(v, dtype=np.complex128) for v in arguments)
    if (
        values
        and isinstance(degree, (int, np.integer))
        and isinstance(order, (int, np.integer))
        and isinstance(polarization, (int, np.integer))
    ):
        shape = max((v.shape for v in values), key=len)
        if all(
            v.shape == shape or (v.size == 1 and v.ndim <= len(shape)) for v in values
        ):
            return _native.vector_wave(
                kind,
                [(int(degree), int(order), int(polarization))],
                [v.ravel() for v in values],
                shape,
                [v.shape for v in values],
            )
    labels = tuple(
        np.asarray(v, dtype=np.float64) for v in (degree, order, polarization)
    )
    if any(
        np.any(~np.isfinite(v) | (v != np.floor(v)) | (np.abs(v) > 128)) for v in labels
    ):
        raise ValueError("wave labels must be integers in [-128, 128]")
    broadcast = np.broadcast_arrays(*labels, *values)
    modes = (
        [(int(labels[0].item()), int(labels[1].item()), int(labels[2].item()))]
        if all(v.size == 1 for v in labels)
        else [
            (int(ell), int(m), int(p))
            for ell, m, p in zip(*(v.flat for v in broadcast[:3]), strict=True)
        ]
    )
    return _native.vector_wave(
        kind,
        modes,
        [
            (v if v.size == 1 else cast("NDArray[np.complex128]", b)).ravel()
            for v, b in zip(values, broadcast[3:], strict=True)
        ],
        broadcast[0].shape,
        [v.shape for v in values],
    )


def sph_harm(
    theta: ArrayLike,
    phi: ArrayLike,
    *,
    degree: ArrayLike,
    order: ArrayLike,
) -> tuple[NDArray[np.complex128], _native.WaveContext]:
    """Normalized spherical harmonic with native theta and phi pullbacks."""
    return vector_wave(theta, phi, kind="sph_harm", degree=degree, order=order)


def spherical_translation(
    kr: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    *,
    destination: Sequence[ArrayLike],
    source: Sequence[ArrayLike],
    poltype: str | None = None,
    singular: bool = True,
) -> tuple[NDArray[np.complex128], _native.PolarTranslationContext]:
    """Polar translation and (kr, theta, phi) VJPs for fixed (degree, order, pol) labels."""
    poltype = _resolve_poltype(poltype)
    if len(destination) != 3 or len(source) != 3:
        raise ValueError("each mode requires degree, order and polarization")
    arguments = tuple(np.asarray(v, dtype=np.complex128) for v in (kr, theta, phi))
    if all(isinstance(v, (int, np.integer)) for v in (*destination, *source)):
        shape = max((v.shape for v in arguments), key=len)
        if all(
            v.shape == shape or (v.size == 1 and v.ndim <= len(shape))
            for v in arguments
        ):
            return _native.spherical_translation(
                [
                    (
                        cast(
                            "tuple[int, int, int]",
                            tuple(int(v) for v in cast("Sequence[int]", destination)),
                        ),
                        cast(
                            "tuple[int, int, int]",
                            tuple(int(v) for v in cast("Sequence[int]", source)),
                        ),
                    )
                ],
                tuple(v.ravel() for v in arguments),
                poltype == "helicity",
                singular,
                shape,
                (arguments[0].shape, arguments[1].shape, arguments[2].shape),
            )
    labels = tuple(np.asarray(v, dtype=np.float64) for v in (*destination, *source))
    if any(
        np.any(~np.isfinite(v) | (v != np.floor(v)) | (np.abs(v) > 128)) for v in labels
    ):
        raise ValueError("mode labels must be integers in [-128,128]")
    arrays = np.broadcast_arrays(*labels, *arguments)
    rows = (
        [tuple(int(v.item()) for v in labels)]
        if all(v.size == 1 for v in labels)
        else [
            tuple(int(v) for v in row)
            for row in zip(*(v.flat for v in arrays[:6]), strict=True)
        ]
    )
    return _native.spherical_translation(
        [((r[0], r[1], r[2]), (r[3], r[4], r[5])) for r in rows],
        tuple(
            (a if a.size == 1 else cast("NDArray[np.complex128]", b)).ravel()
            for a, b in zip(arguments, arrays[6:], strict=True)
        ),
        poltype == "helicity",
        singular,
        arrays[0].shape,
        (arguments[0].shape, arguments[1].shape, arguments[2].shape),
    )


def cylindrical_translation(
    krr: ArrayLike,
    phi: ArrayLike,
    z: ArrayLike,
    kz: ArrayLike,
    *,
    order: ArrayLike,
    singular: bool = True,
) -> tuple[NDArray[np.complex128], _native.CylindricalTranslationContext]:
    """Cylindrical polar translation; order is source minus destination order.

    Pullback returns (krr, phi, z, kz). The two matching axial wave labels move
    together with kz; changing their equality partition is a discrete operation.
    """
    arguments = tuple(np.asarray(v, dtype=np.complex128) for v in (krr, phi, z, kz))
    if isinstance(order, (int, np.integer)):
        shape = max((v.shape for v in arguments), key=len)
        if all(
            v.shape == shape or (v.size == 1 and v.ndim <= len(shape))
            for v in arguments
        ):
            return _native.cylindrical_translation(
                [int(order)],
                tuple(v.ravel() for v in arguments),
                singular,
                shape,
                (
                    arguments[0].shape,
                    arguments[1].shape,
                    arguments[2].shape,
                    arguments[3].shape,
                ),
            )
    orders = np.asarray(order, dtype=np.float64)
    if np.any(
        ~np.isfinite(orders) | (orders != np.floor(orders)) | (np.abs(orders) > 256)
    ):
        raise ValueError("order differences must be integers in [-256,256]")
    arrays = np.broadcast_arrays(orders, *arguments)
    return _native.cylindrical_translation(
        (orders if orders.size == 1 else arrays[0]).astype(np.int32).ravel().tolist(),
        tuple(
            (a if a.size == 1 else cast("NDArray[np.complex128]", b)).ravel()
            for a, b in zip(arguments, arrays[1:], strict=True)
        ),
        singular,
        arrays[0].shape,
        (
            arguments[0].shape,
            arguments[1].shape,
            arguments[2].shape,
            arguments[3].shape,
        ),
    )


def periodic_from_table(
    values: ArrayLike,
    destination: SphericalWaveBasis,
    source: SphericalWaveBasis | None = None,
    *,
    poltype: str | None = None,
) -> tuple[NDArray[np.complex128], _native.PeriodicTableContext]:
    """Native angular contraction and its pullback to supplied lattice harmonics.

    The table shape is (destination origins, source origins, 1 or 2 wavenumber
    channels, harmonics). The last index is l*l+l+m through the sum of the maximum
    basis degrees. Parity requires one wavenumber channel. The residual stores
    only the sparse angular map; the callback supplying values owns its derivatives.
    """
    poltype = _resolve_poltype(poltype)
    source = destination if source is None else source
    table = np.asarray(values, dtype=np.complex128)
    return _native.periodic_from_table(
        list(destination.modes),
        list(source.modes),
        destination.positions.tolist(),
        source.positions.tolist(),
        poltype == "helicity",
        table,
    )

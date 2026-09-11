"""First-order Advect adapters for native forward/pullback operations.

Install ``treams-rs[advect]``. Each reverse pass consumes its native residual
once. Forward mode, higher derivatives, staging, and checkpointing are unsupported.
A fresh forward call creates a fresh residual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import advect as ad
import numpy as np

from . import coeffs, diff, lattice
from ._operators import _rs_weights

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

    from ._core import CylindricalWaveBasis, SphericalWaveBasis
    from .ebcm import Modes


type _Values = tuple[ArrayLike, ...]
type _Pullback = Callable[[NDArray[np.complex128]], _Values]
type _Forward = Callable[[_Values], tuple[NDArray[np.complex128], _Pullback]]


@ad.primitive(static_argnames=("forward",), residual=True)
def _execute(
    values: _Values, *, forward: _Forward
) -> ad.PrimitiveResult[NDArray[np.complex128]]:
    value, pullback = forward(values)
    # Clearing the holder drops the native context even when a failed trace is kept.
    return ad.PrimitiveResult(value, [pullback], release=list.clear)


@_execute.def_transpose
def _transpose(
    cotangent: ArrayLike,
    primals: _Values,
    _output: ArrayLike,
    residual: list[_Pullback],
    *,
    forward: _Forward,
) -> _Values:
    # Advect and the native core both use dL = Re(vdot(gradient, dx)).
    gradients = residual[0](
        np.ascontiguousarray(cotangent, dtype=np.complex128).reshape(
            np.shape(cotangent)
        )
    )
    return tuple(
        np.asarray(
            gradient if np.iscomplexobj(primal) else np.real(gradient),
            dtype=np.asarray(primal).dtype,
        ).reshape(np.shape(primal))
        for gradient, primal in zip(gradients, primals, strict=True)
    )


def _call(values: _Values, forward: _Forward) -> NDArray[np.complex128]:
    # Normalize containers before the primitive flattens its dynamic leaves.
    return cast(
        "NDArray[np.complex128]",
        _execute(tuple(ad.numpy.asarray(value) for value in values), forward=forward),
    )


def chirality_density(
    ks: ArrayLike, normal: ArrayLike, z: ArrayLike = (0.0, 0.0)
) -> NDArray[np.complex128]:
    """Compact up/down/cross chirality coefficients, with native k/normal/z VJP."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.chirality_density(*values)
        return value, context.pullback

    return _call((ks, normal, z), forward)


def sphere(
    lmax: int,
    k0: ArrayLike,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike | None = None,
    kappa: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Differentiable multilayer/chiral sphere in the helicity basis."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.sphere(lmax, float(np.asarray(values[0])), *values[1:])
        return value, context.pullback

    # Defaults are constants with the input's shape; no boxed input is coerced.
    shape = np.shape(epsilon)
    return _call(
        (
            k0,
            radii,
            epsilon,
            np.ones(shape) if mu is None else mu,
            np.zeros(shape) if kappa is None else kappa,
        ),
        forward,
    )


def cluster(
    lmax: int, k0: ArrayLike, radii: ArrayLike, epsilon: ArrayLike, positions: ArrayLike
) -> NDArray[np.complex128]:
    """Differentiable interacting T-matrix of homogeneous spheres in vacuum."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.cluster(
            lmax, float(np.asarray(values[3])), values[0], values[2], values[1]
        )
        return value, context.pullback

    return _call((radii, positions, epsilon, k0), forward)


def interaction(local: ArrayLike, coupling: ArrayLike) -> NDArray[np.complex128]:
    """Differentiable solve of (I - T C) X = T."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.interaction(values[0], values[1])
        return value, context.pullback

    return _call((local, coupling), forward)


def solve(operator: ArrayLike, rhs: ArrayLike) -> NDArray[np.complex128]:
    """Differentiable A X = B for a matrix B, reusing native pivoted LU in reverse."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.solve(values[0], values[1])
        return value, context.pullback

    return _call((operator, rhs), forward)


def eig(operator: ArrayLike) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Native complex eigensystem with eigenvalue and phase-fixed vector VJPs.

    At repeated eigenvalues only equally weighted eigenvalue sums, without vector
    dependence, have a supported pullback. Individual eigenmodes are undefined.
    """

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        (eigenvalues, eigenvectors), context = diff.eig(values[0])
        return np.vstack((eigenvalues, eigenvectors)), lambda g: (
            context.pullback(g[0], g[1:]),
        )

    packed = _call((operator,), forward)
    return packed[0], packed[1:]


def smatrix_add(lower: ArrayLike, upper: ArrayLike) -> NDArray[np.complex128]:
    """Differentiable Redheffer composition, with arrays shaped (2, 2, n, n)."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.smatrix_add(values[0], values[1])
        return value, context.pullback

    return _call((lower, upper), forward)


def smatrix_illuminate(
    lower: ArrayLike, upper: ArrayLike, up: ArrayLike, down: ArrayLike
) -> NDArray[np.complex128]:
    """Outgoing/internal up/down fields for mode-by-illumination incident arrays."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.smatrix_illuminate(*values)
        return value, context.pullback

    return _call((lower, upper, up, down), forward)


def smatrix_periodic(smats: ArrayLike) -> NDArray[np.complex128]:
    """Native periodic transfer matrix, with a factorization-reusing adjoint."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.smatrix_periodic(values[0])
        return value, lambda g: (context.pullback(g),)

    return _call((smats,), forward)


def bands(
    smats: ArrayLike, period: ArrayLike
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Normal Bloch wavenumbers/vectors, with native S-matrix and period pullbacks."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        (wavenumbers, vectors), context = diff.bands(
            values[0], float(np.asarray(values[1]))
        )
        return np.vstack((wavenumbers, vectors)), lambda g: context.pullback(
            g[0], g[1:]
        )

    packed = _call((smats, period), forward)
    return packed[0], packed[1:]


def smatrix_from_array(
    response: ArrayLike, channels: ArrayLike
) -> NDArray[np.complex128]:
    """Differentiable radiation of an effective multipole response into plane waves."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.smatrix_from_array(values[0], values[1])
        return value, context.pullback

    return _call((response, channels), forward)


def spherical_channels(
    positions: ArrayLike,
    ks: ArrayLike,
    q: ArrayLike,
    area: ArrayLike,
    *,
    basis: SphericalWaveBasis,
    polarizations: ArrayLike,
    poltype: str = "helicity",
    fixed_q: bool = False,
) -> NDArray[np.complex128]:
    """Differentiable spherical incidence/radiation channels.

    fixed_q treats q as a static constant; this supports exactly normal incidence.
    Direction gradients otherwise require nonzero transverse wavevectors.
    """
    from ._core import SphericalWaveBasis

    static_q = np.asarray(q, dtype=np.float64) if fixed_q else None

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        dynamic_basis = SphericalWaveBasis(basis.modes, positions=values[0])
        value, context = diff.spherical_channels(
            dynamic_basis,
            values[1],
            static_q if static_q is not None else values[3],
            polarizations,
            float(np.asarray(values[2])),
            poltype=poltype,
            fixed_q=fixed_q,
        )

        def pullback(g: NDArray[np.complex128]) -> _Values:
            positions, ks, q, area = context.pullback(g)
            return (positions, ks, area) if fixed_q else (positions, ks, area, q)

        return value, pullback

    return _call(
        (positions, ks, area) if fixed_q else (positions, ks, area, q), forward
    )


def fresnel(ks: ArrayLike, kzs: ArrayLike, zs: ArrayLike) -> NDArray[np.complex128]:
    """Differentiable chiral planar-interface coefficients."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = coeffs.fresnel_with_context(*values)
        return value, context.pullback

    return _call((ks, kzs, zs), forward)


def propagation(vectors: ArrayLike, distance: ArrayLike) -> NDArray[np.complex128]:
    """Differentiable propagation for upgoing wavevectors and a Cartesian displacement."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.propagation(values[0], values[1])
        return value, context.pullback

    return _call((vectors, distance), forward)


def mie(
    degree: int, sizes: ArrayLike, epsilon: ArrayLike, mu: ArrayLike, kappa: ArrayLike
) -> NDArray[np.complex128]:
    """Differentiable spherical Mie coefficient matrix."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = coeffs.mie_with_context(degree, *values)
        return value, context.pullback

    return _call((sizes, epsilon, mu, kappa), forward)


def mie_cyl(
    kz: ArrayLike,
    order: int,
    k0: ArrayLike,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike,
    kappa: ArrayLike,
) -> NDArray[np.complex128]:
    """Differentiable cylindrical Mie coefficient matrix."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = coeffs.mie_cyl_with_context(
            float(np.asarray(values[0])),
            order,
            float(np.asarray(values[1])),
            *values[2:],
        )
        return value, context.pullback

    return _call((kz, k0, radii, epsilon, mu, kappa), forward)


def rotation(
    angles: ArrayLike,
    *,
    destination: SphericalWaveBasis | CylindricalWaveBasis,
    source: SphericalWaveBasis | CylindricalWaveBasis | None = None,
) -> NDArray[np.complex128]:
    """Euler-angle derivatives; cylindrical theta must remain fixed at zero."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.rotation(values[0], destination, source)
        return value, lambda g: (np.asarray(context.pullback(g)),)

    return _call((angles,), forward)


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
) -> NDArray[np.complex128]:
    """Axisymmetric surface integral with shape and material adjoints owned by Rust."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.ebcm_qmat(
            *values,
            theta=theta,
            weights=weights,
            out=out,
            in_=in_,
            singular=singular,
            legacy=legacy,
        )
        return value, context.pullback

    return _call((radii, slopes, ks, zs), forward)


def tmatrix_metric(
    operator: ArrayLike,
    ks: ArrayLike = (1.0, 1.0),
    *,
    polarizations: ArrayLike,
    kind: str,
) -> NDArray[np.float64]:
    """Global helicity cd/db/chi, with all derivatives owned by Rust."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.tmatrix_metric(
            values[0], values[1], polarizations=polarizations, kind=kind
        )
        return np.asarray(value, dtype=np.complex128), lambda g: context.pullback(
            float(g.real)
        )

    return ad.numpy.real(_call((operator, ks), forward))


def svdvals(operator: ArrayLike) -> NDArray[np.float64]:
    """Descending singular values with a native first-order matrix pullback."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.svdvals(values[0])
        return value.astype(np.complex128), lambda g: (context.pullback(g.real),)

    return ad.numpy.real(_call((operator,), forward))


def field_operator(
    points: ArrayLike,
    origins: ArrayLike,
    ks: ArrayLike,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    poltype: str = "helicity",
    singular: bool = False,
) -> NDArray[np.complex128]:
    """Full field matrix with native geometry/wavenumber derivatives."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.field_operator(
            values[0],
            type(basis)(basis.modes, values[1]),
            values[2],
            poltype=poltype,
            singular=singular,
        )
        return value, context.pullback

    return _call((points, origins, ks), forward)


def field(
    coefficients: ArrayLike,
    points: ArrayLike,
    origins: ArrayLike,
    ks: ArrayLike,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    poltype: str = "helicity",
    singular: bool = False,
) -> NDArray[np.complex128]:
    """Electric field; differentiable in amplitudes, points, origins and wavenumbers."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        dynamic_basis = type(basis)(basis.modes, positions=values[2])
        value, context = diff.field(
            values[0],
            values[1],
            dynamic_basis,
            values[3],
            poltype=poltype,
            singular=singular,
        )
        return value, context.pullback

    return _call((coefficients, points, origins, ks), forward)


def hfield(
    coefficients: ArrayLike,
    points: ArrayLike,
    origins: ArrayLike,
    ks: ArrayLike,
    impedance: ArrayLike,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    poltype: str = "helicity",
    singular: bool = False,
) -> NDArray[np.complex128]:
    """Magnetic samples, including impedance derivatives through the linear weights."""
    weights = 2 * basis.pol - 1 if poltype == "helicity" else 1
    if poltype == "parity":
        basis = type(basis)(
            [(*mode[:3], 1 - mode[3]) for mode in basis.modes], basis.positions
        )
    coefficients = (
        -1j * ad.numpy.asarray(coefficients) * weights / ad.numpy.asarray(impedance)
    )
    return field(
        coefficients,
        points,
        origins,
        ks,
        basis=basis,
        poltype=poltype,
        singular=singular,
    )


def gfield(
    pol: int,
    coefficients: ArrayLike,
    points: ArrayLike,
    origins: ArrayLike,
    ks: ArrayLike,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    poltype: str = "helicity",
    singular: bool = False,
) -> NDArray[np.complex128]:
    """Weighted G samples and native field pullbacks, with upstream scaling."""
    electric, magnetic = _rs_weights(pol, basis, poltype)
    value = field(
        ad.numpy.asarray(coefficients) * electric,
        points,
        origins,
        ks,
        basis=basis,
        poltype=poltype,
        singular=singular,
    )
    if magnetic:
        value = value + 1j * magnetic * hfield(
            coefficients,
            points,
            origins,
            ks,
            1.0,
            basis=basis,
            poltype=poltype,
            singular=singular,
        )
    return value


def ffield(
    pol: int,
    coefficients: ArrayLike,
    points: ArrayLike,
    origins: ArrayLike,
    ks: ArrayLike,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    poltype: str = "helicity",
    singular: bool = False,
) -> NDArray[np.complex128]:
    """Weighted F samples, differentiating the chiral index weights as well."""
    if poltype == "helicity":
        ks = ad.numpy.asarray(ks)
        coefficients = (
            ad.numpy.asarray(coefficients) * 2 * ks[basis.pol] / ad.numpy.sum(ks)
        )
    return gfield(
        pol,
        coefficients,
        points,
        origins,
        ks,
        basis=basis,
        poltype=poltype,
        singular=singular,
    )


def expansion(
    destination_positions: ArrayLike,
    source_positions: ArrayLike,
    ks: ArrayLike,
    *,
    destination: SphericalWaveBasis | CylindricalWaveBasis,
    source: SphericalWaveBasis | CylindricalWaveBasis,
    poltype: str = "helicity",
    singular: bool = False,
) -> NDArray[np.complex128]:
    """Expansion matrix, differentiable in both origin arrays and wavenumbers."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.expansion(
            type(destination)(destination.modes, positions=values[0]),
            type(source)(source.modes, positions=values[1]),
            values[2],
            poltype=poltype,
            singular=singular,
        )
        return value, context.pullback

    return _call((destination_positions, source_positions, ks), forward)


def cylinder(
    kzs: ArrayLike,
    mmax: int,
    k0: ArrayLike,
    radii: ArrayLike,
    epsilon: ArrayLike,
    mu: ArrayLike | None = None,
    kappa: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Differentiable multilayer/chiral cylinder T-matrix."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.cylinder(
            values[0], mmax, float(np.asarray(values[1])), *values[2:]
        )
        return value, context.pullback

    shape = np.shape(epsilon)
    return _call(
        (
            kzs,
            k0,
            radii,
            epsilon,
            np.ones(shape) if mu is None else mu,
            np.zeros(shape) if kappa is None else kappa,
        ),
        forward,
    )


def lattice_expansion(
    destination_positions: ArrayLike,
    source_positions: ArrayLike,
    ks: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    *,
    destination: SphericalWaveBasis | CylindricalWaveBasis,
    source: SphericalWaveBasis | CylindricalWaveBasis,
    poltype: str = "helicity",
    eta: complex = 0,
) -> NDArray[np.complex128]:
    """Periodic coupling with native VJPs for origins, two ks, Bloch and lattice vectors.

    The Ewald split eta is a numerical constant; its exact physical derivative is zero.
    Cylindrical axial wavenumbers remain fixed basis labels.
    """

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = lattice.expansion_with_context(
            type(destination)(destination.modes, positions=values[0]),
            type(source)(source.modes, positions=values[1]),
            values[2],
            values[4],
            values[3],
            poltype=poltype,
            eta=eta,
        )
        return value, context.pullback

    return _call((destination_positions, source_positions, ks, kpar, a), forward)


def plane_phases(points: ArrayLike, vectors: ArrayLike) -> NDArray[np.complex128]:
    """Plane translation phases with native displacement and wavevector VJPs."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.plane_phases(values[0], values[1])
        return value, context.pullback

    return _call((points, vectors), forward)


def plane_field(
    coefficients: ArrayLike | None,
    points: ArrayLike,
    vectors: ArrayLike,
    *,
    polarizations: ArrayLike,
    poltype: str = "helicity",
    fixed_vectors: bool = False,
) -> NDArray[np.complex128]:
    """Native plane field/operator, differentiable in amplitudes, points and wavevectors.

    fixed_vectors removes the wavevectors from the differentiable inputs.
    """
    dynamic: _Values = (points,) if coefficients is None else (coefficients, points)
    if not fixed_vectors:
        dynamic += (vectors,)

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.plane_field(
            None if coefficients is None else values[0],
            values[0] if coefficients is None else values[1],
            vectors if fixed_vectors else values[-1],
            polarizations,
            poltype=poltype,
            fixed_vectors=fixed_vectors,
        )

        def pullback(g: NDArray[np.complex128]) -> _Values:
            gc, gp, gk = context.pullback(g)
            gradients: _Values = (gp,) if coefficients is None else (gc, gp)
            return gradients if fixed_vectors else (*gradients, gk)

        return value, pullback

    return _call(dynamic, forward)


def plane_expansion(
    origins: ArrayLike,
    vectors: ArrayLike,
    *,
    destination: SphericalWaveBasis | CylindricalWaveBasis,
    polarizations: ArrayLike,
    poltype: str = "helicity",
    fixed_vectors: bool = False,
) -> NDArray[np.complex128]:
    """Plane-to-multipole illumination with native origin and wavevector pullbacks."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.plane_expansion(
            type(destination)(destination.modes, values[0]),
            vectors if fixed_vectors else values[1],
            polarizations,
            poltype=poltype,
            fixed_vectors=fixed_vectors,
        )

        def pullback(g: NDArray[np.complex128]) -> _Values:
            gradients = context.pullback(g)
            return gradients[:1] if fixed_vectors else gradients

        return value, pullback

    return _call((origins,) if fixed_vectors else (origins, vectors), forward)


def cylindrical_channels(
    origins: ArrayLike,
    ks: ArrayLike,
    kx: ArrayLike,
    period: ArrayLike,
    *,
    basis: CylindricalWaveBasis,
    kz_labels: ArrayLike,
    polarizations: ArrayLike,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Differentiable cylindrical radiation, holding axial mode labels fixed."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        q = np.column_stack([kz_labels, values[2]])
        value, context = diff.cylindrical_channels(
            type(basis)(basis.modes, values[0]),
            values[1],
            q,
            polarizations,
            float(np.asarray(values[3])),
            poltype=poltype,
        )

        def pullback(g: NDArray[np.complex128]) -> _Values:
            go, gk, gq, ga = context.pullback(g)
            return go, gk, gq[:, 1], np.asarray(ga)

        return value, pullback

    return _call((origins, ks, kx, period), forward)


def interface(
    ks: ArrayLike,
    zs: ArrayLike,
    q: ArrayLike,
    *,
    alignment: str = "xy",
    fixed_q: bool = False,
) -> NDArray[np.complex128]:
    """Native Cartesian interface matching with its implicit solve adjoint."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.interface(
            values[0],
            values[1],
            q if fixed_q else values[2],
            alignment=alignment,
            fixed_q=fixed_q,
        )

        def pullback(g: NDArray[np.complex128]) -> _Values:
            gradients = context.pullback(g)
            return gradients[:2] if fixed_q else gradients

        return value, pullback

    return _call((ks, zs) if fixed_q else (ks, zs, q), forward)


def layer_stack(
    ks: ArrayLike,
    zs: ArrayLike,
    q: ArrayLike,
    thickness: ArrayLike,
    *,
    alignment: str = "xy",
    fixed_q: bool = False,
) -> NDArray[np.complex128]:
    """Compact native layer stack with linear channel storage and analytic pullback."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.layer_stack(
            values[0],
            values[1],
            q if fixed_q else values[2],
            values[2] if fixed_q else values[3],
            alignment=alignment,
            fixed_q=fixed_q,
        )

        def pullback(g: NDArray[np.complex128]) -> _Values:
            gk, gz, gq, gd = context.pullback(g)
            return (gk, gz, gd) if fixed_q else (gk, gz, gq, gd)

        return value, pullback

    return _call((ks, zs, thickness) if fixed_q else (ks, zs, q, thickness), forward)


def periodic_conversion(
    destination_origins: ArrayLike,
    source_origins: ArrayLike,
    ks: ArrayLike,
    kzs: ArrayLike,
    period: ArrayLike,
    *,
    destination: CylindricalWaveBasis,
    source: SphericalWaveBasis,
    poltype: str = "helicity",
) -> NDArray[np.complex128]:
    """Periodic spherical-to-cylindrical radiation, including moving Fourier labels."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        modes = [
            (p, float(kz), m, pol)
            for (p, _, m, pol), kz in zip(
                destination.modes, np.asarray(values[3], dtype=np.float64), strict=True
            )
        ]
        value, context = diff.periodic_conversion(
            type(destination)(modes, values[0]),
            type(source)(source.modes, values[1]),
            values[2],
            float(np.asarray(values[4])),
            poltype=poltype,
        )
        return value, context.pullback

    return _call((destination_origins, source_origins, ks, kzs, period), forward)

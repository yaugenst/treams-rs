"""Differentiable physical scattering with Advect and native Rust pullbacks.

Use this namespace for all objects inside a differentiated objective::

    import advect as ad
    import advect.numpy as np
    import treams_rs.advect as tr

    def objective(radius):
        sphere = tr.sphere_tmatrix(k0=2.0, lmax=2, radius=radius, material=3.0)
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=2.0)
        return sphere.cross_sections(wave).scattering

    value, gradient = ad.value_and_grad(objective)(0.2)

Use ``ad.grad`` for just the derivative, or ``ad.value_and_grad`` for both.
For multiple parameters use an array (or pytree) argument and unpack it inside
objective; its gradient has the same structure. Build traced geometry using
``advect.numpy.stack`` or ``asarray``; never convert a traced value to float or
plain NumPy. ``tr.Material(epsilon=...)`` accepts traced real/complex parameters.
The same functions work without tracing for value evaluation. Convert to NumPy
or float only after the transform returns, e.g. to write JSON.

``Cluster(..., positions=...).scatter(wave).efield(points)`` includes multiple
scattering; add ``wave.efield(points)`` for the total field. ``slab(...).power``
returns differentiable power fractions. ``tr.PlaneWavePorts`` and
``tr.Lattice`` supply fixed mode geometry in this namespace too. Numerical arrays are exposed as
``.array`` on responses and ``.coefficients`` on waves.

Install ``treams-rs[advect]``. CPU, float64/complex128 and first-order reverse
mode; mode cutoffs, integer labels and topology remain static. Each reverse
pass consumes its native residual once. Forward mode, higher derivatives,
staging and checkpointing are unsupported. Call the transformed objective
again for every optimization step to create fresh residuals. The root
``treams_rs`` namespace is NumPy-only; mixing it into a trace loses derivatives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import advect as ad
import numpy as np

# Physical objects share one implementation; this namespace selects execution.
from . import _framework as _physics
from . import coeffs, diff, lattice
from ._core import CylindricalWaveBasis, PlaneWaveBasisByComp, SphericalWaveBasis
from ._core import CylindricalWaveBasis as CylindricalBasis
from ._core import PlaneWaveBasisByComp as PlaneWavePorts
from ._core import PlaneWaveBasisByUnitVector as PlaneWaveBasis
from ._core import SphericalWaveBasis as SphericalBasis
from ._framework import (
    BandModes as BandModes,
)
from ._framework import (
    Cluster as Cluster,
)
from ._framework import (
    CrossSections as CrossSections,
)
from ._framework import (
    Material as Material,
)
from ._framework import (
    PeriodicResponse as PeriodicResponse,
)
from ._framework import PortWave as PortWave
from ._framework import (
    PowerBalance as PowerBalance,
)
from ._framework import ScatteredPorts as ScatteredPorts
from ._framework import SMatrix as SMatrix
from ._framework import TMatrix as TMatrix
from ._framework import Wave as Wave
from ._lattice import Lattice as Lattice
from ._operators import _rs_weights
from .config import _resolve_poltype

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike, NDArray

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


def bessel(
    z: ArrayLike,
    *,
    order: ArrayLike,
    kind: str = "j",
    spherical: bool = False,
    derivative: bool = False,
) -> NDArray[np.complex128]:
    """Broadcast Bessel values with a native argument VJP; order is held fixed."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.bessel(
            order, values[0], kind=kind, spherical=spherical, derivative=derivative
        )
        return value, lambda g: (context.pullback(g),)

    return _call((z,), forward)


def incgamma(z: ArrayLike, *, n: ArrayLike) -> NDArray[np.complex128]:
    """Upper incomplete gamma with a native argument pullback; n stays fixed."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.incgamma(n, values[0])
        return value, lambda g: (context.pullback(g),)

    return _call((z,), forward)


def intkambe(z: ArrayLike, eta: ArrayLike, *, n: ArrayLike) -> NDArray[np.complex128]:
    """Kambe integral with native z and eta pullbacks; n stays fixed."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.intkambe(n, *values)
        return value, context.pullback

    return _call((z, eta), forward)


def lattice_sum(
    k: ArrayLike,
    kpar: ArrayLike,
    a: ArrayLike,
    r: ArrayLike,
    eta: ArrayLike = 0,
    *,
    dim: int,
    degree: ArrayLike = 0,
    order: ArrayLike = 0,
    spherical: bool = True,
    part: str = "full",
    shell: ArrayLike = 0,
) -> NDArray[np.complex128]:
    """Native lattice-sum pullbacks with fixed wave labels and direct-shell index."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.lattice_sum(
            dim, degree, order, *values, spherical=spherical, part=part, shell=shell
        )
        return value, context.pullback

    return _call((k, kpar, a, r, eta), forward)


def angular(
    z: ArrayLike, *, degree: ArrayLike, order: ArrayLike, kind: str = "legendre"
) -> NDArray[np.complex128]:
    """Integer-degree angular functions with a native argument VJP."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.angular(degree, order, values[0], kind=kind)
        return value, lambda g: (context.pullback(g),)

    return _call((z,), forward)


def wigner(
    phi: ArrayLike,
    theta: ArrayLike,
    psi: ArrayLike,
    *,
    degree: ArrayLike,
    row: ArrayLike,
    column: ArrayLike,
) -> NDArray[np.complex128]:
    """Wigner D elements with native complex Euler-angle pullbacks."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.wigner(degree, row, column, *values)
        return value, context.pullback

    return _call((phi, theta, psi), forward)


def chirality_density(
    ks: ArrayLike, normal: ArrayLike, z: ArrayLike = (0.0, 0.0)
) -> NDArray[np.complex128]:
    """Compact up/down/cross chirality coefficients, with native k/normal/z VJP."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.chirality_density(*values)
        return value, context.pullback

    return _call((ks, normal, z), forward)


def oriented_chirality(
    transverse: ArrayLike,
    normal: ArrayLike,
    z: ArrayLike = (0.0, 0.0),
    *,
    polarizations: ArrayLike,
    axis: int = 2,
) -> NDArray[np.complex128]:
    """Signed helicity chirality forms with native geometry and interval VJP."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.oriented_chirality(
            *values, polarizations=polarizations, axis=axis
        )
        return value, context.pullback

    return _call((transverse, normal, z), forward)


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


def particle_cluster(
    local: Sequence[ArrayLike],
    positions: ArrayLike,
    ks: ArrayLike,
    *,
    bases: Sequence[SphericalWaveBasis | CylindricalWaveBasis],
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Heterogeneous local matrices with native geometry and embedding adjoints."""
    poltype = _resolve_poltype(poltype)

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.particle_cluster(
            values[2:],
            values[0],
            values[1],
            bases=bases,
            poltype=poltype,
        )

        def pullback(g: NDArray[np.complex128]) -> _Values:
            local, positions, ks = context.pullback(g)
            return (positions, ks, *local)

        return value, pullback

    return _call((positions, ks, *local), forward)


def illuminate(
    local: ArrayLike, coupling: ArrayLike, incident: ArrayLike
) -> NDArray[np.complex128]:
    """Differentiate only the requested incident columns of a scattering solve."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.illuminate(values[0], values[1], values[2])
        return value, context.pullback

    return _call((local, coupling, incident), forward)


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
) -> NDArray[np.complex128]:
    """Differentiable T/R rows for illumination columns and complete port geometry."""
    poltype = _resolve_poltype(poltype)
    static_q = np.asarray(q, dtype=np.float64) if fixed_q else None

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.smatrix_tr(
            values[0],
            values[1],
            values[2],
            values[3],
            static_q if static_q is not None else values[4],
            modes=modes,
            axis=axis,
            poltype=poltype,
            modetype=modetype,
            fixed_q=fixed_q,
        )

        def pullback(g: NDArray[np.complex128]) -> _Values:
            result = context.pullback(g)
            return result[:4] if fixed_q else result

        return value.astype(np.complex128), pullback

    return _call(
        (matrices, incident, ks, zs) if fixed_q else (matrices, incident, ks, zs, q),
        forward,
    )


def smatrix_cd(
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
) -> NDArray[np.complex128]:
    """Transmission and total-outgoing-power contrast with complete native port AD.

    A single native power evaluation batches both opposite-polarization
    illuminations. Advect composes only their selection and scalar contrasts.
    """
    poltype = _resolve_poltype(poltype)
    incoming = ad.numpy.asarray(incident)
    if incoming.ndim != 1:
        raise ValueError("CD requires one incident mode vector")
    if poltype == "helicity":
        indices = {mode: i for i, mode in enumerate(modes)}
        try:
            opposite = incoming[[indices[(group, 1 - pol)] for group, pol in modes]]
        except KeyError as error:
            raise ValueError("CD requires both helicities of each direction") from error
    else:
        opposite = incoming * np.array([2 * pol - 1 for _, pol in modes])
    power = smatrix_tr(
        matrices,
        ad.numpy.stack((incoming, opposite), axis=1),
        ks,
        zs,
        q,
        modes=modes,
        axis=axis,
        poltype=poltype,
        modetype=modetype,
        fixed_q=fixed_q,
    )
    total = ad.numpy.sum(power, axis=0)
    denominator = ad.numpy.stack((ad.numpy.sum(power[0]), ad.numpy.sum(total)))
    if ad.numpy.any(denominator == 0):
        raise ValueError(
            "CD is undefined for zero summed transmission or outgoing power"
        )
    return (
        ad.numpy.stack((power[0, 1] - power[0, 0], total[1] - total[0])) / denominator
    )


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
    poltype: str | None = None,
    fixed_q: bool = False,
) -> NDArray[np.complex128]:
    """Differentiable spherical incidence/radiation channels.

    fixed_q treats q as a static constant; this supports exactly normal incidence.
    Direction gradients otherwise require nonzero transverse wavevectors.
    """
    poltype = _resolve_poltype(poltype)

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


def propagation_matrix(
    vectors: ArrayLike, distance: ArrayLike
) -> NDArray[np.complex128]:
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


def _real_kzs(kzs: ArrayLike) -> NDArray[np.float64]:
    axial = np.asarray(kzs)
    if np.iscomplexobj(axial):
        if np.any(axial.imag != 0):
            raise ValueError("kzs must be real")
        axial = axial.real
    return np.asarray(axial, dtype=np.float64)


def _dynamic_basis(
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    origins: ArrayLike,
    kzs: ArrayLike | None,
) -> SphericalWaveBasis | CylindricalWaveBasis:
    if kzs is None:
        return type(basis)(basis.modes, origins)
    if not isinstance(basis, CylindricalWaveBasis):
        raise ValueError("axial derivatives require a cylindrical basis")
    axial = _real_kzs(kzs)
    if axial.shape != (len(basis),):
        raise ValueError("kzs must contain one real axial wavenumber per mode")
    result = CylindricalWaveBasis(
        (
            (p, float(kz), m, pol)
            for (p, _, m, pol), kz in zip(basis.modes, axial, strict=True)
        ),
        origins,
    )
    if len(result) != len(basis):
        raise ValueError("axial wavenumbers must preserve distinct mode labels")
    return result


def field_operator(
    points: ArrayLike,
    origins: ArrayLike,
    ks: ArrayLike,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    poltype: str | None = None,
    singular: bool = False,
    kzs: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Full field matrix; optional per-mode kzs are differentiable for cylinders."""
    poltype = _resolve_poltype(poltype)

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        dynamic_basis = _dynamic_basis(
            basis, values[1], values[3] if kzs is not None else None
        )
        value, context = diff.field_operator(
            values[0], dynamic_basis, values[2], poltype=poltype, singular=singular
        )
        return value, context.pullback if kzs is None else context.pullback_axial

    values = (points, origins, ks) if kzs is None else (points, origins, ks, kzs)
    return _call(values, forward)


def field(
    coefficients: ArrayLike,
    points: ArrayLike,
    origins: ArrayLike,
    ks: ArrayLike,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    poltype: str | None = None,
    singular: bool = False,
    kzs: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Electric samples with native geometry/medium and optional cylindrical kz VJPs.

    For routine scattering prefer ``Cluster(...).scatter(incident).efield(points)``.
    Raw scattered coefficients require ``singular=True`` (outgoing waves);
    ``singular=False`` means regular incident waves. ``ks`` contains medium
    wavenumbers for negative/positive helicity, both k0 in vacuum. Helicity
    labels do not mean opposite propagation directions or opposite signs of k.

    The raw path agrees with the physical wave API::

        import numpy as np
        import treams_rs.advect as tr

        k0 = 2.0
        sphere = tr.sphere_tmatrix(k0=k0, lmax=2, radius=0.2, material=3.0)
        incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
        scattered = sphere.scatter(incident)
        points = [[0.4, 0.1, 0.3]]
        raw = tr.field(scattered.coefficients, points, scattered.basis.positions,
                       [k0, k0], basis=scattered.basis, singular=True)
        np.testing.assert_allclose(raw, scattered.efield(points))
    """
    poltype = _resolve_poltype(poltype)

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        dynamic_basis = _dynamic_basis(
            basis, values[2], values[4] if kzs is not None else None
        )
        value, context = diff.field(
            values[0],
            values[1],
            dynamic_basis,
            values[3],
            poltype=poltype,
            singular=singular,
        )
        return value, context.pullback if kzs is None else context.pullback_axial

    values = (
        (coefficients, points, origins, ks)
        if kzs is None
        else (coefficients, points, origins, ks, kzs)
    )
    return _call(values, forward)


def hfield(
    coefficients: ArrayLike,
    points: ArrayLike,
    origins: ArrayLike,
    ks: ArrayLike,
    impedance: ArrayLike,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    poltype: str | None = None,
    singular: bool = False,
    kzs: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Magnetic samples, including impedance derivatives through the linear weights."""
    poltype = _resolve_poltype(poltype)
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
        kzs=kzs,
    )


def gfield(
    pol: int,
    coefficients: ArrayLike,
    points: ArrayLike,
    origins: ArrayLike,
    ks: ArrayLike,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    poltype: str | None = None,
    singular: bool = False,
    kzs: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Weighted G samples and native field pullbacks, with upstream scaling."""
    poltype = _resolve_poltype(poltype)
    electric, magnetic = _rs_weights(pol, basis, poltype)
    value = field(
        ad.numpy.asarray(coefficients) * electric,
        points,
        origins,
        ks,
        basis=basis,
        poltype=poltype,
        singular=singular,
        kzs=kzs,
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
            kzs=kzs,
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
    poltype: str | None = None,
    singular: bool = False,
    kzs: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Weighted F samples, differentiating the chiral index weights as well."""
    poltype = _resolve_poltype(poltype)
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
        kzs=kzs,
    )


def _group_axial(
    destination: SphericalWaveBasis | CylindricalWaveBasis,
    source: SphericalWaveBasis | CylindricalWaveBasis,
    kzs: ArrayLike | None,
) -> tuple[
    NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.intp] | None
]:
    if kzs is None:
        return None, None, None
    if not isinstance(destination, CylindricalWaveBasis) or not isinstance(
        source, CylindricalWaveBasis
    ):
        raise ValueError("axial expansion derivatives require two cylindrical bases")
    groups = np.unique(np.concatenate((destination.kz, source.kz)))
    axial = _real_kzs(kzs)
    if axial.shape != groups.shape or not np.all(np.isfinite(axial)):
        raise ValueError("kzs must contain one finite real value per axial group")
    if len(np.unique(axial)) != len(groups):
        raise ValueError("axial groups must remain distinct")
    return (
        axial[np.searchsorted(groups, destination.kz)],
        axial[np.searchsorted(groups, source.kz)],
        np.searchsorted(np.sort(axial), axial),
    )


def expansion(
    destination_positions: ArrayLike,
    source_positions: ArrayLike,
    ks: ArrayLike,
    *,
    destination: SphericalWaveBasis | CylindricalWaveBasis,
    source: SphericalWaveBasis | CylindricalWaveBasis,
    poltype: str | None = None,
    singular: bool = False,
    kzs: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Expansion with origin/medium VJPs and optional cylindrical axial groups.

    ``kzs`` corresponds to sorted distinct axial labels of both original bases.
    One value moves the entire matching group in both bases. Values must remain
    distinct; changing which modes couple is a discrete operation.
    """
    poltype = _resolve_poltype(poltype)

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        to_axial, from_axial, order = _group_axial(
            destination, source, values[3] if kzs is not None else None
        )
        value, context = diff.expansion(
            _dynamic_basis(destination, values[0], to_axial),
            _dynamic_basis(source, values[1], from_axial),
            values[2],
            poltype=poltype,
            singular=singular,
        )
        if order is None:
            return value, context.pullback

        def pullback(g: NDArray[np.complex128]) -> _Values:
            to, source, ks, axial = context.pullback_axial(g)
            return to, source, ks, axial[order]

        return value, pullback

    values = (destination_positions, source_positions, ks)
    return _call(values if kzs is None else (*values, kzs), forward)


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
    poltype: str | None = None,
    eta: complex = 0,
    kzs: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Periodic coupling with native VJPs for origins, two ks, Bloch and lattice vectors.

    The Ewald split eta is a numerical constant; its exact physical derivative is zero.
    Optional ``kzs`` moves shared cylindrical axial groups, ordered as in
    ``expansion``. Groups must remain distinct.
    """
    poltype = _resolve_poltype(poltype)

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        to_axial, from_axial, order = _group_axial(
            destination, source, values[5] if kzs is not None else None
        )
        value, context = lattice.expansion_with_context(
            _dynamic_basis(destination, values[0], to_axial),
            _dynamic_basis(source, values[1], from_axial),
            values[2],
            values[4],
            values[3],
            poltype=poltype,
            eta=eta,
        )
        if order is None:
            return value, context.pullback

        def pullback(g: NDArray[np.complex128]) -> _Values:
            to, source, ks, bloch, vectors, axial = context.pullback_axial(g)
            return to, source, ks, bloch, vectors, axial[order]

        return value, pullback

    values = (destination_positions, source_positions, ks, kpar, a)
    return _call(values if kzs is None else (*values, kzs), forward)


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
    poltype: str | None = None,
    fixed_vectors: bool = False,
) -> NDArray[np.complex128]:
    """Native plane field/operator, differentiable in amplitudes, points and wavevectors.

    fixed_vectors removes the wavevectors from the differentiable inputs.
    """
    poltype = _resolve_poltype(poltype)
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
    poltype: str | None = None,
    fixed_vectors: bool = False,
) -> NDArray[np.complex128]:
    """Plane-to-multipole illumination with native origin and wavevector pullbacks."""
    poltype = _resolve_poltype(poltype)

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
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Differentiable cylindrical radiation, holding axial mode labels fixed."""
    poltype = _resolve_poltype(poltype)

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


def interface_coefficients(
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
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Periodic spherical-to-cylindrical radiation, including moving Fourier labels."""
    poltype = _resolve_poltype(poltype)

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


def plane_permutation(
    vectors: ArrayLike,
    *,
    polarizations: ArrayLike,
    n: int = 1,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Cyclic-axis polarization coefficients with native complex-vector pullbacks."""
    poltype = _resolve_poltype(poltype)

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.plane_permutation(
            values[0], polarizations, n, poltype=poltype
        )
        return value, lambda g: (context.pullback(g),)

    return _call((vectors,), forward)


def coordinates(points: ArrayLike, *, kind: str) -> NDArray[np.float64]:
    """Coordinate conversion with native real point pullback."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.coordinates(values[0], kind=kind)
        return value.astype(np.complex128), lambda g: (context.pullback(g.real),)

    return ad.numpy.real(_call((points,), forward))


def vector_coordinates(
    vectors: ArrayLike, points: ArrayLike, *, kind: str
) -> NDArray[np.complex128]:
    """Vector-frame conversion with native vector and source-position pullbacks."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.vector_coordinates(*values, kind=kind)
        return value, context.pullback

    return _call((vectors, points), forward)


def vector_wave(
    *arguments: ArrayLike,
    kind: str,
    degree: ArrayLike = 0,
    order: ArrayLike = 0,
    polarization: ArrayLike = 0,
) -> NDArray[np.complex128]:
    """Native low-level vector waves with all continuous arguments differentiable."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.vector_wave(
            *values, kind=kind, degree=degree, order=order, polarization=polarization
        )
        return value, lambda g: tuple(context.pullback(g))

    return _call(arguments, forward)


def sph_harm(
    theta: ArrayLike,
    phi: ArrayLike,
    *,
    degree: ArrayLike,
    order: ArrayLike,
) -> NDArray[np.complex128]:
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
) -> NDArray[np.complex128]:
    """Polar spherical translation with native argument pullbacks."""
    poltype = _resolve_poltype(poltype)

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.spherical_translation(
            *values,
            destination=destination,
            source=source,
            poltype=poltype,
            singular=singular,
        )
        return value, context.pullback

    return _call((kr, theta, phi), forward)


def cylindrical_translation(
    krr: ArrayLike,
    phi: ArrayLike,
    z: ArrayLike,
    kz: ArrayLike,
    *,
    order: ArrayLike,
    singular: bool = True,
) -> NDArray[np.complex128]:
    """Cylindrical polar translation with an analytic common-axial-label VJP."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.cylindrical_translation(
            *values, order=order, singular=singular
        )
        return value, context.pullback

    return _call((krr, phi, z, kz), forward)


def periodic_from_table(
    values: ArrayLike,
    *,
    destination: SphericalWaveBasis,
    source: SphericalWaveBasis | None = None,
    poltype: str | None = None,
) -> NDArray[np.complex128]:
    """Compose a custom differentiable lattice table with native angular coupling."""
    poltype = _resolve_poltype(poltype)

    def forward(inputs: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.periodic_from_table(
            inputs[0], destination, source, poltype=poltype
        )
        return value, lambda g: (context.pullback(g),)

    return _call((values,), forward)


class PlaneWave(_physics.PlaneWave):
    """Plane illumination using this namespace's explicitly selected framework."""

    def __init__(
        self, direction: Any, polarization: Any, *, k0: Any, medium: Any = 1.0
    ):
        super().__init__(
            direction, polarization, k0=k0, medium=medium, backend=_backend
        )


def smatrix(
    array: Any,
    *,
    basis: PlaneWaveBasisByComp,
    k0: Any,
    negative_medium: Any = 1.0,
    positive_medium: Any = 1.0,
    polarization: str = "helicity",
) -> SMatrix:
    """Wrap scattering blocks (2,2,modes,modes) with explicit exterior media."""
    matrix = _backend.array(array, complex_=True)
    if matrix.shape != (2, 2, len(basis), len(basis)):
        raise ValueError("scattering blocks must have shape (2,2,modes,modes)")
    result = SMatrix(
        matrix,
        basis=basis,
        k0=_backend.array(k0),
        media=(
            _physics._material(positive_medium),
            _physics._material(negative_medium),
        ),
        backend=_backend,
    )
    result.polarization = polarization
    return result


def wave(
    coefficients: Any,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    k0: Any,
    medium: Any = 1.0,
    kind: str = "regular",
    polarization: str = "helicity",
    positions: Any = None,
) -> Wave:
    """Construct a physical multipole wave using this namespace; coefficients have shape (modes,)."""
    if kind not in ("regular", "outgoing"):
        raise ValueError("wave kind must be regular or outgoing")
    array = _backend.array(coefficients, complex_=True)
    if array.shape != (len(basis),):
        raise ValueError("one coefficient is required per basis mode")
    return Wave(
        array,
        basis=basis,
        k0=_backend.array(k0),
        medium=_physics._material(medium),
        backend=_backend,
        positions=positions,
        outgoing=kind == "outgoing",
        polarization=polarization,
    )


def tmatrix(
    array: Any,
    *,
    basis: SphericalWaveBasis | CylindricalWaveBasis,
    k0: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
    positions: Any = None,
) -> TMatrix:
    """Wrap a user response (modes,modes) with fixed labels and dynamic physical metadata."""
    matrix = _backend.array(array, complex_=True)
    if matrix.shape != (len(basis), len(basis)):
        raise ValueError("matrix dimensions must match the basis")
    return TMatrix(
        matrix,
        basis=basis,
        k0=_backend.array(k0),
        medium=_physics._material(medium),
        backend=_backend,
        positions=positions,
        polarization=polarization,
    )


def sphere_tmatrix(
    *,
    k0: Any,
    lmax: int,
    radius: Any,
    material: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
) -> TMatrix:
    """Homogeneous sphere with differentiable geometry, material and frequency."""
    return multilayer_sphere_tmatrix(
        k0=k0,
        lmax=lmax,
        radii=_backend.stack((_backend.array(radius),)),
        materials=(material,),
        medium=medium,
        polarization=polarization,
    )


def multilayer_sphere_tmatrix(
    *,
    k0: Any,
    lmax: int,
    radii: Any,
    materials: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
) -> TMatrix:
    """Concentric layers, one material per radius, with a separate exterior."""
    result = _backend.multilayer(
        k0=k0,
        radii=radii,
        materials=materials,
        medium=medium,
        basis=SphericalWaveBasis.default(lmax),
        polarization="helicity",
    )
    return (
        result if polarization == "helicity" else result.with_polarization(polarization)
    )


def cylinder_tmatrix(
    *,
    k0: Any,
    kz: Any,
    mmax: int,
    radius: Any,
    material: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
) -> TMatrix:
    """Homogeneous cylinder; axial mode labels are fixed configuration."""
    return multilayer_cylinder_tmatrix(
        k0=k0,
        kz=kz,
        mmax=mmax,
        radii=_backend.stack((_backend.array(radius),)),
        materials=(material,),
        medium=medium,
        polarization=polarization,
    )


def multilayer_cylinder_tmatrix(
    *,
    k0: Any,
    kz: Any,
    mmax: int,
    radii: Any,
    materials: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
) -> TMatrix:
    """Concentric cylinders with fixed axial labels and dynamic layers."""
    result = _backend.multilayer(
        k0=k0,
        radii=radii,
        materials=materials,
        medium=medium,
        basis=CylindricalWaveBasis.default(kz, mmax),
        polarization="helicity",
        kz=kz,
        mmax=mmax,
    )
    return (
        result if polarization == "helicity" else result.with_polarization(polarization)
    )


def plane_wave(
    direction: Any, polarization: Any, *, k0: Any, medium: Any = 1.0
) -> PlaneWave:
    """Fixed-direction plane wave with dynamic frequency, medium and amplitudes."""
    return PlaneWave(direction, polarization, k0=k0, medium=medium)


def solve_periodic(
    unit_cell: TMatrix | Cluster, *, lattice: Any, kpar: Any, eta: complex = 0
) -> PeriodicResponse:
    """Solve periodic coupling once, then use response.to_smatrix(ports)."""
    return _physics.solve_periodic(unit_cell, lattice=lattice, kpar=kpar, eta=eta)


def slab(
    *,
    basis: PlaneWaveBasisByComp,
    k0: Any,
    thickness: Any,
    material: Any,
    negative_medium: Any = 1.0,
    positive_medium: Any = 1.0,
    polarization: str = "helicity",
) -> SMatrix:
    """One layer with differentiable geometry and explicit exterior media."""
    return _physics.layer_stack(
        _backend,
        k0=k0,
        basis=basis,
        materials=(negative_medium, material, positive_medium),
        thickness=thickness,
    ).with_polarization(polarization)


def interface(
    *,
    basis: PlaneWaveBasisByComp,
    k0: Any,
    negative_medium: Any,
    positive_medium: Any,
    polarization: str = "helicity",
) -> SMatrix:
    """Interface from negative to positive side of the port normal."""
    return _physics.layer_stack(
        _backend,
        k0=k0,
        basis=basis,
        materials=(negative_medium, positive_medium),
        thickness=[],
    ).with_polarization(polarization)


def multilayer_slab(
    *,
    basis: PlaneWaveBasisByComp,
    k0: Any,
    thicknesses: Any,
    materials: Any,
    negative_medium: Any = 1.0,
    positive_medium: Any = 1.0,
    polarization: str = "helicity",
) -> SMatrix:
    """Interior layers in increasing normal order, one material per thickness."""
    return _physics.layer_stack(
        _backend,
        k0=k0,
        basis=basis,
        materials=(negative_medium, *materials, positive_medium),
        thickness=thicknesses,
    ).with_polarization(polarization)


def propagation(
    *,
    distance: Any,
    basis: PlaneWaveBasisByComp,
    k0: Any,
    medium: Any = 1.0,
    polarization: str = "helicity",
) -> SMatrix:
    """Homogeneous propagation with native distance and wavevector derivatives."""
    return _physics.propagation(
        _backend,
        distance=distance,
        basis=basis,
        k0=k0,
        medium=medium,
        polarization=polarization,
    )


def stack(layers: Any) -> SMatrix:
    """Cascade layers from the negative to positive side of the port normal."""
    if not layers:
        raise ValueError("stack requires at least one layer")
    result = layers[0]
    for layer in layers[1:]:
        result = result.cascade(layer)
    return result


def _physics_call(
    record: Any, shape: tuple[int, ...], *values: Any, real: bool = False
) -> Any:
    def forward(primals: Any) -> Any:
        result, context = record(*primals)
        pullback = context if callable(context) else context.pullback

        def backward(g: Any) -> Any:
            result = pullback(g.real if real else g)
            return result if isinstance(result, tuple) else (result,)

        return np.asarray(result, dtype=np.complex128), backward

    result = _call(values, forward)
    return ad.numpy.real(result) if real else result


_backend = _physics.Backend(ad.numpy, _physics_call)


__all__ = [
    "BandModes",
    "Cluster",
    "CrossSections",
    "CylindricalBasis",
    "Lattice",
    "Material",
    "PeriodicResponse",
    "PlaneWave",
    "PlaneWaveBasis",
    "PlaneWavePorts",
    "PortWave",
    "PowerBalance",
    "SMatrix",
    "ScatteredPorts",
    "SphericalBasis",
    "TMatrix",
    "Wave",
    "angular",
    "bands",
    "bessel",
    "chirality_density",
    "cluster",
    "coordinates",
    "cylinder",
    "cylinder_tmatrix",
    "cylindrical_channels",
    "cylindrical_translation",
    "ebcm_qmat",
    "eig",
    "expansion",
    "ffield",
    "field",
    "field_operator",
    "fresnel",
    "gfield",
    "hfield",
    "illuminate",
    "incgamma",
    "interaction",
    "interface",
    "interface_coefficients",
    "intkambe",
    "lattice_expansion",
    "lattice_sum",
    "layer_stack",
    "mie",
    "mie_cyl",
    "multilayer_cylinder_tmatrix",
    "multilayer_slab",
    "multilayer_sphere_tmatrix",
    "oriented_chirality",
    "particle_cluster",
    "periodic_conversion",
    "periodic_from_table",
    "plane_expansion",
    "plane_field",
    "plane_permutation",
    "plane_phases",
    "plane_wave",
    "propagation",
    "propagation_matrix",
    "rotation",
    "slab",
    "smatrix",
    "smatrix_add",
    "smatrix_cd",
    "smatrix_from_array",
    "smatrix_illuminate",
    "smatrix_periodic",
    "smatrix_tr",
    "solve",
    "solve_periodic",
    "sph_harm",
    "sphere",
    "sphere_tmatrix",
    "spherical_channels",
    "spherical_translation",
    "stack",
    "svdvals",
    "tmatrix",
    "tmatrix_metric",
    "vector_coordinates",
    "vector_wave",
    "wave",
    "wigner",
]

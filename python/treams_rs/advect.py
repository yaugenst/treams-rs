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

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

    from ._core import CylindricalWaveBasis, SphericalWaveBasis


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
    gradients = residual[0](np.ascontiguousarray(cotangent, dtype=np.complex128))
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


def smatrix_add(lower: ArrayLike, upper: ArrayLike) -> NDArray[np.complex128]:
    """Differentiable Redheffer composition, with arrays shaped (2, 2, n, n)."""

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        value, context = diff.smatrix_add(values[0], values[1])
        return value, context.pullback

    return _call((lower, upper), forward)


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


def field(
    coefficients: ArrayLike,
    points: ArrayLike,
    origins: ArrayLike,
    ks: ArrayLike,
    *,
    basis: SphericalWaveBasis,
    poltype: str = "helicity",
    singular: bool = False,
) -> NDArray[np.complex128]:
    """Electric field; differentiable in amplitudes, points, origins and wavenumbers."""
    from ._core import SphericalWaveBasis

    def forward(values: _Values) -> tuple[NDArray[np.complex128], _Pullback]:
        dynamic_basis = SphericalWaveBasis(basis.modes, positions=values[2])
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

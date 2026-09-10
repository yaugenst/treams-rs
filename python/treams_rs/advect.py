"""First-order Advect adapters for native forward/pullback operations.

Install ``treams-rs[advect]``. Each reverse pass consumes its native residual
once. Forward mode, higher derivatives, staging, and checkpointing are unsupported.
A fresh forward call creates a fresh residual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import advect as ad
import numpy as np

from . import coeffs, diff

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray


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

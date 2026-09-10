"""First-order Autograd adapters for native forward/pullback operations.

Install ``treams-rs[autograd]``. The Rust solver remains opaque to Autograd.
Each VJP consumes its native residual once; repeated VJPs and higher derivatives
are not supported. A fresh forward call creates a fresh residual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.builtins as ag_builtins
import numpy as np
from autograd.extend import defvjp, primitive
from autograd.tracer import isbox

from . import coeffs, diff

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray


type _Values = tuple[ArrayLike, ...]
type _Pullback = Callable[[NDArray[np.complex128]], _Values]
type _Forward = Callable[[_Values], tuple[NDArray[np.complex128], _Pullback]]


@primitive
def _execute(
    values: _Values, forward: _Forward, holder: list[_Pullback]
) -> NDArray[np.complex128]:
    value, pullback = forward(values)
    holder.append(pullback)
    return value


def _make_vjp(
    _answer: ArrayLike, values: _Values, _forward: _Forward, holder: list[_Pullback]
) -> Callable[[ArrayLike], _Values]:
    native_pullback = holder.pop()

    def pullback(cotangent: ArrayLike) -> _Values:
        if isbox(cotangent):
            raise NotImplementedError("native adapters support first-order VJPs only")
        native_cotangent = np.ascontiguousarray(np.conj(cotangent), dtype=np.complex128)
        gradients = native_pullback(native_cotangent)
        # Autograd uses the bilinear complex convention; the core uses Hermitian.
        return ag_builtins.tuple(
            np.asarray(
                np.conj(gradient) if np.iscomplexobj(value) else np.real(gradient)
            ).reshape(np.shape(value))
            for gradient, value in zip(gradients, values, strict=True)
        )

    return pullback


defvjp(_execute, _make_vjp)


def _call(values: _Values, forward: _Forward) -> NDArray[np.complex128]:
    if any(isbox(value) and isbox(getattr(value, "_value", None)) for value in values):
        raise NotImplementedError("native adapters support first-order VJPs only")
    holder: list[_Pullback] = []
    # Per-call ownership keeps independent calls and threads separate. Python
    # reference counting drops a residual if tracing or the objective fails.
    return _execute(ag_builtins.tuple(values), forward, holder)


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

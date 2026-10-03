"""Ordinary special functions backed by the existing analytic Rust records."""

# ruff: noqa: E741 - upstream degree argument l

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from . import diff
from ._autodiff_functions import require_no_out


def _apply(
    backend: Any,
    record: Any,
    labels: tuple[Any, ...],
    *values: Any,
    vector: bool = False,
) -> Any:
    from ._dispatch import backend_for

    if backend_for(*labels) is not None:
        raise TypeError(
            "degrees, orders and polarization labels must be NumPy or Python constants"
        )
    arrays = tuple(backend.array(value, complex_=True) for value in values)
    shape = np.broadcast_shapes(
        *(np.shape(label) for label in labels), *(array.shape for array in arrays)
    )
    return backend.apply(record, (*shape, 3) if vector else shape, *arrays)


def bessel(
    backend: Any,
    v: Any,
    z: Any,
    out: Any = None,
    *,
    where: Any = True,
    function: str,
    spherical: bool = False,
    derivative: bool = False,
) -> Any:
    require_no_out(out, where)
    return _apply(
        backend,
        partial(
            diff.bessel,
            v,
            function=function,
            spherical=spherical,
            derivative=derivative,
        ),
        (v,),
        z,
    )


def spherical_bessel(
    backend: Any,
    n: Any,
    z: Any,
    derivative: bool = False,
    *,
    out: Any = None,
    where: Any = True,
    function: str,
) -> Any:
    return bessel(
        backend,
        n,
        z,
        out,
        where=where,
        function=function,
        spherical=True,
        derivative=derivative,
    )


def angular(
    backend: Any,
    n: Any,
    m: Any,
    z: Any,
    out: Any = None,
    *,
    where: Any = True,
    function: str,
) -> Any:
    require_no_out(out, where)
    return _apply(backend, partial(diff.angular, n, m, function=function), (n, m), z)


def lpmv(
    backend: Any, m: Any, n: Any, z: Any, *, out: Any = None, where: Any = True
) -> Any:
    require_no_out(out, where)
    # Normalize nested inputs in their framework before inspecting dtype; NumPy
    # cannot infer the dtype of a container of tracers without coercing them.
    z = backend.asarray(z, dtype=None)
    complex_ = "complex" in str(z.dtype)

    def record(value: Any) -> Any:
        result, context = diff.angular(n, m, value, function="legendre")
        # Match Rust lpmv_real's domain restriction before discarding imaginary
        # values. Even orders and the zero extension remain valid outside [-1, 1].
        if not complex_ and np.any(result.imag != 0):
            raise ValueError(
                "real lpmv requires -1 <= x <= 1 for odd orders |order| <= degree; "
                "pass complex z for the continuation"
            )
        return result, context

    result = _apply(backend, record, (n, m), z)
    # The NumPy function returns real values for real arguments. The native
    # record is complex; projection is a framework operation with its own rule.
    return result if complex_ else backend.xp.real(result)


def incgamma(
    backend: Any, n: Any, z: Any, *, out: Any = None, where: Any = True
) -> Any:
    require_no_out(out, where)
    return _apply(backend, partial(diff.incgamma, n), (n,), z)


def intkambe(
    backend: Any, n: Any, z: Any, eta: Any, *, out: Any = None, where: Any = True
) -> Any:
    require_no_out(out, where)
    return _apply(backend, partial(diff.intkambe, n), (n,), z, eta)


def wignerd(
    backend: Any,
    l: Any,
    m: Any,
    k: Any,
    phi: Any,
    theta: Any,
    psi: Any,
    out: Any = None,
    *,
    where: Any = True,
) -> Any:
    require_no_out(out, where)
    return _apply(backend, partial(diff.wignerd, l, m, k), (l, m, k), phi, theta, psi)


def wignersmalld(
    backend: Any,
    l: Any,
    m: Any,
    k: Any,
    theta: Any,
    out: Any = None,
    *,
    where: Any = True,
) -> Any:
    return wignerd(backend, l, m, k, 0.0, theta, 0.0, out, where=where)


def coordinates(backend: Any, points: Any, out: Any = None, *, function: str) -> Any:
    require_no_out(out)
    value = backend.array(points)
    return backend.apply(
        partial(diff.coordinates, function=function), value.shape, value, real=True
    )


def vector_coordinates(
    backend: Any, vector: Any, points: Any, out: Any = None, *, function: str
) -> Any:
    require_no_out(out)
    vector, point = backend.array(vector, complex_=True), backend.array(points)
    return backend.apply(
        partial(diff.vector_coordinates, function=function),
        np.broadcast_shapes(vector.shape, point.shape),
        vector,
        point,
    )


def sph_harm(
    backend: Any,
    m: Any,
    l: Any,
    phi: Any,
    theta: Any,
    out: Any = None,
    *,
    where: Any = True,
) -> Any:
    require_no_out(out, where)
    return _apply(
        backend, partial(diff.sph_harm, degree=l, order=m), (l, m), theta, phi
    )


def spherical_wave(
    backend: Any,
    l: Any,
    m: Any,
    *args: Any,
    function: str,
    out: Any = None,
    where: Any = True,
) -> Any:
    count = 2 if function.startswith("vsh") else 3
    has_pol = function.endswith("A")
    expected = count + int(has_pol)
    if len(args) == expected + 1 and out is None:
        args, out = args[:-1], args[-1]
    require_no_out(out, where)
    if len(args) != expected:
        raise TypeError(f"{function} expects {expected + 2} input arguments")
    arguments, pol = (args[:-1], args[-1]) if has_pol else (args, 0)
    return _apply(
        backend,
        partial(diff.vector_wave, function=function, degree=l, order=m, pol=pol),
        (l, m, pol),
        *arguments,
        vector=True,
    )


def cylindrical_wave(
    backend: Any,
    kz: Any,
    m: Any,
    *args: Any,
    function: str,
    out: Any = None,
    where: Any = True,
) -> Any:
    count = 3 + int(not function.endswith("M"))
    has_pol = function.endswith("A")
    expected = count + int(has_pol)
    if len(args) == expected + 1 and out is None:
        args, out = args[:-1], args[-1]
    require_no_out(out, where)
    if len(args) != expected:
        raise TypeError(f"{function} expects {expected + 2} input arguments")
    arguments, pol = (args[:-1], args[-1]) if has_pol else (args, 0)
    return _apply(
        backend,
        partial(diff.vector_wave, function=function, order=m, pol=pol),
        (m, pol),
        kz,
        *arguments,
        vector=True,
    )


def plane_wave(
    backend: Any,
    kx: Any,
    ky: Any,
    kz: Any,
    x: Any,
    y: Any,
    z: Any,
    *args: Any,
    function: str,
    out: Any = None,
    pol: Any = None,
) -> Any:
    has_pol = function.endswith("A")
    if pol is not None and not has_pol:
        raise TypeError(f"{function} does not take a polarization argument")
    positional_pol = has_pol and pol is None
    if len(args) == int(positional_pol) + 1 and out is None:
        args, out = args[:-1], args[-1]
    require_no_out(out)
    if len(args) != int(positional_pol):
        raise TypeError(f"{function} expects {6 + int(has_pol)} input arguments")
    pol = args[0] if positional_pol else (0 if pol is None else pol)
    return _apply(
        backend,
        partial(diff.vector_wave, function=function, pol=pol),
        (pol,),
        kx,
        ky,
        kz,
        x,
        y,
        z,
        vector=True,
    )


def spherical_translation(
    backend: Any,
    l: Any,
    m: Any,
    degree: Any,
    order: Any,
    kr: Any,
    theta: Any,
    phi: Any,
    out: Any = None,
    *,
    where: Any = True,
    singular: bool,
    cross: bool,
) -> Any:
    require_no_out(out, where)
    destination, source = (l, m, 0), (degree, order, int(cross))
    return _apply(
        backend,
        partial(
            diff.spherical_translation,
            destination=destination,
            source=source,
            poltype="parity",
            singular=singular,
        ),
        (*destination, *source),
        kr,
        theta,
        phi,
    )


def cylindrical_translation(
    backend: Any,
    kz: Any,
    mu: Any,
    qz: Any,
    m: Any,
    krr: Any,
    phi: Any,
    z: Any,
    *args: Any,
    singular: bool,
    **kwargs: Any,
) -> Any:
    from . import cw

    return cw.translate(kz, mu, 0, qz, m, 0, krr, phi, z, singular, *args, **kwargs)

"""Framework calls for the coefficient namespaces, using existing native records.

Most coefficient records already broadcast. Matrix-only records are evaluated
one element at a time inside a single framework operation: this avoids building
a quadratic matrix merely to extract its diagonal. Their pullbacks stay native.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _modes, _native, diff
from ._autodiff_functions import require_no_out
from ._bases import CylindricalBasis, SphericalBasis
from ._dispatch import backend_for
from ._records import DerivativeContext
from ._saved import ArraySpec, SavedRecord, native_state, saved_record

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._framework_backend import Backend


def _options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    if len(args) > 1:
        raise TypeError("only one positional output argument is supported")
    if args and "out" in kwargs:
        raise TypeError("out was supplied twice")
    require_no_out(
        args[0] if args else kwargs.pop("out", None), kwargs.pop("where", True)
    )
    if kwargs:
        raise TypeError(
            f"unsupported differentiable ufunc keywords: {', '.join(kwargs)}"
        )


def _labels(*values: Any) -> tuple[Any, ...]:
    if backend_for(*values) is not None:
        raise TypeError(
            "discrete mode and polarization labels must be ordinary constants"
        )
    return tuple(np.asarray(value) for value in values)


def _shape(values: tuple[Any, ...], labels: tuple[Any, ...] = ()) -> tuple[int, ...]:
    return np.broadcast_shapes(*(value.shape for value in (*values, *labels)))


def _broadcast(backend: Backend, value: Any, shape: tuple[int, ...]) -> Any:
    # Autograd's broadcast VJP requires equal ranks, unlike NumPy's forward.
    padded = (1,) * (len(shape) - value.ndim) + value.shape
    return backend.xp.broadcast_to(value.reshape(padded), shape)


def _elementwise(
    backend: Backend,
    evaluate: Callable[..., Any],
    values: tuple[Any, ...],
    labels: tuple[Any, ...] = (),
) -> Any:
    """Lift scalar records, broadcasting tangents and reducing input gradients."""
    shape = _shape(values, labels)
    prepared = saved_record(evaluate)
    if prepared is None:
        raise TypeError("coefficient record must declare its saved-state contract")
    child: SavedRecord = prepared

    def scalar_specs(arrays: Any) -> tuple[ArraySpec, ...]:
        return tuple(ArraySpec((), value.dtype) for value in (*arrays, *labels))

    def map_context(contexts: Any, arrays: Any) -> DerivativeContext:
        def pullback(cotangent: Any) -> Any:
            gradients = [np.empty(shape, dtype=np.complex128) for _ in arrays]
            for index, context in contexts:
                for gradient, value in zip(
                    gradients, context.pullback(cotangent[index]), strict=True
                ):
                    gradient[index] = value
            reduced = []
            for gradient, array in zip(gradients, arrays, strict=True):
                padded = (1,) * (len(shape) - array.ndim) + array.shape
                axes = tuple(
                    axis
                    for axis, (a, b) in enumerate(zip(padded, shape, strict=True))
                    if a == 1 and b != 1
                )
                reduced.append(
                    gradient.sum(axis=axes, keepdims=True).reshape(array.shape)
                )
            return tuple(reduced)

        def pushforward(*tangents: Any) -> Any:
            broadcast = [np.broadcast_to(tangent, shape) for tangent in tangents]
            output = np.empty(shape, dtype=np.complex128)
            for index, context in contexts:
                output[index] = context.pushforward(
                    *(value[index] for value in broadcast)
                )
            return output

        return DerivativeContext(pullback, pushforward, (contexts, arrays))

    def record(*arrays: Any) -> Any:
        broadcast = np.broadcast_arrays(*arrays, *labels)
        result = np.empty(shape, dtype=np.complex128)
        contexts = []
        for index in np.ndindex(shape):
            scalar_values = tuple(value[index] for value in broadcast)
            result[index], context = evaluate(*scalar_values)
            contexts.append((index, context))
        return result, map_context(contexts, arrays)

    def state_spec(inputs: tuple[ArraySpec, ...]) -> tuple[ArraySpec, ...]:
        return tuple(
            ArraySpec((*shape, *spec.shape), spec.dtype)
            for spec in child.state_spec(scalar_specs(inputs))
        )

    def save(context: DerivativeContext) -> tuple[Any, ...]:
        contexts, arrays = context.native_context
        states = tuple(
            np.empty((*shape, *spec.shape), dtype=spec.dtype)
            for spec in child.state_spec(scalar_specs(arrays))
        )
        for index, context in contexts:
            for target, value in zip(states, child.save(context), strict=True):
                target[index] = value
        return states

    def restore(states: tuple[Any, ...], *arrays: Any) -> DerivativeContext:
        broadcast = np.broadcast_arrays(*arrays, *labels)
        contexts = [
            (
                index,
                child.restore(
                    tuple(state[index] for state in states),
                    *(value[index] for value in broadcast),
                ),
            )
            for index in np.ndindex(shape)
        ]
        return map_context(contexts, arrays)

    return backend.apply(SavedRecord(record, state_spec, save, restore), shape, *values)


def sw_translate(
    backend: Backend,
    lambda_: Any,
    mu: Any,
    pol: Any,
    l: Any,  # noqa: E741 - public spherical degree label
    m: Any,
    qol: Any,
    kr: Any,
    theta: Any,
    phi: Any,
    poltype: Any = None,
    singular: Any = True,
    *args: Any,
    **kwargs: Any,
) -> Any:
    _options(args, kwargs)
    labels = _labels(lambda_, mu, pol, l, m, qol)
    values = tuple(backend.array(v, complex_=True) for v in (kr, theta, phi))
    shape = _shape(values, labels)

    record = partial(
        diff.spherical_translation,
        destination=labels[:3],
        source=labels[3:],
        poltype=poltype,
        singular=singular,
    )
    return backend.apply(record, shape, *values)


def sw_rotate(
    backend: Backend,
    lambda_: Any,
    mu: Any,
    pol: Any,
    l: Any,  # noqa: E741 - public spherical degree label
    m: Any,
    qol: Any,
    phi: Any,
    theta: Any = 0,
    psi: Any = 0,
    *args: Any,
    **kwargs: Any,
) -> Any:
    _options(args, kwargs)
    labels = _labels(lambda_, mu, pol, l, m, qol)
    values = tuple(backend.array(v) for v in (phi, theta, psi))
    shape = _shape(values, labels)
    values = tuple(_broadcast(backend, value, shape) for value in values)

    def restore(_state: Any, phi: Any, theta: Any, psi: Any) -> DerivativeContext:
        context = _native.wignerd_context(
            *diff._wignerd_inputs(labels[3], labels[1], labels[4], phi, theta, psi)
        )
        mask = (labels[0] == labels[3]) & (labels[2] == labels[5])

        def pullback(g: Any) -> Any:
            return context.pullback(np.where(mask, g, 0))

        def pushforward(*tangents: Any) -> Any:
            return np.where(mask, context.pushforward(*tangents), 0)

        return DerivativeContext(pullback, pushforward)

    def record(phi: Any, theta: Any, psi: Any) -> Any:
        value = _native.sw_rotate(*labels, phi, theta, psi)
        return value, restore((), phi, theta, psi)

    return backend.apply(
        SavedRecord(record, lambda _inputs: (), lambda _context: (), restore),
        shape,
        *values,
    )


def cw_rotate(
    backend: Backend,
    kz: Any,
    mu: Any,
    pol: Any,
    qz: Any,
    m: Any,
    qol: Any,
    phi: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    _options(args, kwargs)
    labels = _labels(mu, pol, m, qol)
    values = (backend.array(phi), backend.array(kz), backend.array(qz))
    shape = _shape(values, labels)
    values = tuple(_broadcast(backend, value, shape) for value in values)

    def restore(_state: Any, phi: Any, kz: Any, qz: Any) -> DerivativeContext:
        mu, pol, m, qol = labels
        context = _native.wignerd_context(
            *diff._wignerd_inputs(np.abs(m), m, m, phi, 0.0, 0.0)
        )
        mask = (kz == qz) & (mu == m) & (pol == qol)

        def pullback(g: Any) -> Any:
            angle, _, _ = context.pullback(np.where(mask, g, 0))
            return angle, np.zeros_like(kz), np.zeros_like(qz)

        def pushforward(phi: Any, _kz: Any, _qz: Any) -> Any:
            return np.where(mask, context.pushforward(phi, 0.0, 0.0), 0)

        return DerivativeContext(pullback, pushforward)

    def record(phi: Any, kz: Any, qz: Any) -> Any:
        mu, pol, m, qol = labels
        value = _native.cw_rotate(kz, mu, pol, qz, m, qol, phi)
        return value, restore((), phi, kz, qz)

    return backend.apply(
        SavedRecord(record, lambda _inputs: (), lambda _context: (), restore),
        shape,
        *values,
    )


def cw_translate(
    backend: Backend,
    kz: Any,
    mu: Any,
    pol: Any,
    qz: Any,
    m: Any,
    qol: Any,
    krr: Any,
    phi: Any,
    z: Any,
    singular: Any = True,
    *args: Any,
    **kwargs: Any,
) -> Any:
    _options(args, kwargs)
    labels = _labels(mu, pol, m, qol)
    values = (
        backend.array(krr, complex_=True),
        backend.array(phi),
        backend.array(z),
        backend.array(kz),
        backend.array(qz),
    )
    shape = _shape(values, labels)
    values = tuple(_broadcast(backend, value, shape) for value in values)

    def restore(
        _state: Any, krr: Any, phi: Any, z: Any, kz: Any, qz: Any
    ) -> DerivativeContext:
        mu, pol, m, qol = labels
        context = _native.cylindrical_translation_context(
            *diff._cylindrical_translation_inputs(
                krr, phi, z, qz, order=m - mu, singular=singular
            )
        )
        mask = (kz == qz) & (pol == qol)

        def pullback(g: Any) -> Any:
            radial, angle, axial, wave = context.pullback(np.where(mask, g, 0))
            # As documented by cw.translate, matching axial labels move together.
            # Credit their shared phase derivative once, to the source qz.
            return radial, angle, axial, np.zeros_like(kz), wave

        def pushforward(krr: Any, phi: Any, z: Any, _kz: Any, qz: Any) -> Any:
            return np.where(mask, context.pushforward(krr, phi, z, qz), 0)

        return DerivativeContext(pullback, pushforward)

    def record(krr: Any, phi: Any, z: Any, kz: Any, qz: Any) -> Any:
        mu, pol, m, qol = labels
        function = _native.cw_translate_s if singular else _native.cw_translate_r
        value = function(kz, mu, pol, qz, m, qol, krr, phi, z)
        return value, restore((), krr, phi, z, kz, qz)

    return backend.apply(
        SavedRecord(record, lambda _inputs: (), lambda _context: (), restore),
        shape,
        *values,
    )


def pw_translate(
    backend: Backend,
    kx: Any,
    ky: Any,
    kz: Any,
    x: Any,
    y: Any,
    z: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    _options(args, kwargs)
    values = (
        *(backend.array(v, complex_=True) for v in (kx, ky, kz)),
        *(backend.array(v) for v in (x, y, z)),
    )

    def map_context(context: Any, *_primals: Any) -> DerivativeContext:
        def pullback(g: Any) -> Any:
            points, vectors = context.pullback(np.array([[g]]))
            return (*vectors[0], *points[0])

        def pushforward(kx: Any, ky: Any, kz: Any, x: Any, y: Any, z: Any) -> Any:
            return context.pushforward([[x, y, z]], [[kx, ky, kz]])[0, 0]

        return DerivativeContext(pullback, pushforward, context)

    @native_state(
        _native.PlanePhasesContext, lambda _inputs: (1, 1), map_context=map_context
    )
    def evaluate(kx: Any, ky: Any, kz: Any, x: Any, y: Any, z: Any) -> Any:
        value, context = diff.plane_phases([[x, y, z]], [[kx, ky, kz]])
        return value[0, 0], map_context(context, kx, ky, kz, x, y, z)

    return _elementwise(backend, evaluate, values)


def pw_to_sw(
    backend: Backend,
    l: Any,  # noqa: E741 - public spherical degree label
    m: Any,
    polsw: Any,
    kx: Any,
    ky: Any,
    kz: Any,
    polpw: Any,
    poltype: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    _options(args, kwargs)
    labels = _labels(l, m, polsw, polpw)
    values = tuple(backend.array(v, complex_=True) for v in (kx, ky, kz))

    def map_context(context: Any, *_primals: Any) -> DerivativeContext:
        def pullback(g: Any) -> Any:
            return tuple(context.pullback(np.array([[g]]))[1][0])

        def pushforward(kx: Any, ky: Any, kz: Any) -> Any:
            return context.pushforward(np.zeros((1, 3)), [[kx, ky, kz]])[0, 0]

        return DerivativeContext(pullback, pushforward, context)

    @native_state(
        _native.PlaneExpansionContext,
        lambda _inputs: (1, 1, 1, False),
        map_context=map_context,
    )
    def evaluate(
        kx: Any,
        ky: Any,
        kz: Any,
        l: Any,  # noqa: E741 - public spherical degree label
        m: Any,
        polsw: Any,
        polpw: Any,
    ) -> Any:
        value, context = diff.plane_expansion(
            SphericalBasis([(l, m, polsw)]), [[kx, ky, kz]], [polpw], poltype=poltype
        )
        return value[0, 0], map_context(context, kx, ky, kz, l, m, polsw, polpw)

    return _elementwise(backend, evaluate, values, labels)


def pw_to_cw(
    backend: Backend,
    kzcw: Any,
    m: Any,
    polcw: Any,
    kx: Any,
    ky: Any,
    kzpw: Any,
    polpw: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    _options(args, kwargs)
    labels = _labels(m, polcw, polpw)
    values = (
        backend.array(kzcw),
        backend.array(kx),
        backend.array(ky, complex_=True),
        backend.array(kzpw),
    )

    def map_context(context: Any, *_primals: Any) -> DerivativeContext:
        if context is None:

            def zero_pullback(_g: Any) -> Any:
                return 0.0, 0.0j, 0.0j, 0.0

            def zero_pushforward(*_tangents: Any) -> Any:
                return 0.0j

            return DerivativeContext(zero_pullback, zero_pushforward)

        def pullback(g: Any) -> Any:
            vector = context.pullback(np.array([[g]]))[1][0]
            return 0.0, vector[0], vector[1], 0.0

        def pushforward(_kzcw: Any, kx: Any, ky: Any, _kzpw: Any) -> Any:
            return context.pushforward(np.zeros((1, 3)), [[kx, ky, 0.0]])[0, 0]

        return DerivativeContext(pullback, pushforward, context)

    def evaluate(
        kzcw: Any, kx: Any, ky: Any, kzpw: Any, m: Any, polcw: Any, polpw: Any
    ) -> Any:
        # The public coefficient uses exact matching; the matrix record permits
        # a few ulps when matching a plane basis to cylindrical labels.
        value = _native.pw_to_cw(kzcw, m, polcw, kx, ky, kzpw, polpw)
        context = None
        if kzcw == kzpw and polcw == polpw:
            _, context = diff.plane_expansion(
                CylindricalBasis([(kzcw, m, polcw)]), [[kx, ky, kzpw]], [polpw]
            )
        return value, map_context(context, kzcw, kx, ky, kzpw, m, polcw, polpw)

    def state_spec(_inputs: tuple[ArraySpec, ...]) -> tuple[ArraySpec, ...]:
        size = 1 + _native.PlaneExpansionContext._state_spec(1, 1, 1, True)
        return (ArraySpec((size,), np.dtype(np.uint8)),)

    def save(context: DerivativeContext) -> tuple[Any, ...]:
        state = np.zeros(state_spec(())[0].shape, dtype=np.uint8)
        if context.native_context is not None:
            state[0] = 1
            state[1:] = context.native_context._state()
        return (state,)

    def restore(state: tuple[Any, ...], *primals: Any) -> DerivativeContext:
        active = state[0][0]
        if active not in (0, 1):
            raise ValueError("invalid plane-expansion state tag")
        context = (
            _native.PlaneExpansionContext._from_state(state[0][1:]) if active else None
        )
        return map_context(context, *primals)

    return _elementwise(
        backend, SavedRecord(evaluate, state_spec, save, restore), values, labels
    )


def pw_permute_xyz(
    backend: Backend,
    kx: Any,
    ky: Any,
    kz: Any,
    p: Any,
    q: Any,
    poltype: Any = None,
    inverse: Any = False,
    *args: Any,
    **kwargs: Any,
) -> Any:
    _options(args, kwargs)
    labels = _labels(p, q)
    values = tuple(backend.array(v, complex_=True) for v in (kx, ky, kz))

    def map_context(
        context: Any, _kx: Any, _ky: Any, _kz: Any, p: Any, _q: Any
    ) -> DerivativeContext:
        def pullback(g: Any) -> Any:
            cotangent = np.zeros((2, 1), dtype=np.complex128)
            cotangent[int(p), 0] = g
            return tuple(context.pullback(cotangent)[0])

        def pushforward(kx: Any, ky: Any, kz: Any) -> Any:
            return context.pushforward([[kx, ky, kz]])[int(p), 0]

        return DerivativeContext(pullback, pushforward, context)

    @native_state(
        _native.PlanePermutationContext, lambda _inputs: (1,), map_context=map_context
    )
    def evaluate(kx: Any, ky: Any, kz: Any, p: Any, q: Any) -> Any:
        if p not in (0, 1):
            raise ValueError("polarization must be 0 or 1")
        value, context = diff.plane_permutation(
            [[kx, ky, kz]], [q], n=2 if inverse else 1, poltype=poltype
        )
        return value[int(p), 0], map_context(context, kx, ky, kz, p, q)

    return _elementwise(backend, evaluate, values, labels)


def sw_periodic_to_cw(
    backend: Backend,
    kz: Any,
    m: Any,
    pol: Any,
    l: Any,  # noqa: E741 - public spherical degree label
    mu: Any,
    qol: Any,
    k: Any,
    area: Any,
    poltype: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    _options(args, kwargs)
    labels = _labels(m, pol, l, mu, qol)
    values = (backend.array(kz), backend.array(k, complex_=True), backend.array(area))

    def map_context(context: Any, *_primals: Any) -> DerivativeContext:
        def pullback(g: Any) -> Any:
            _, _, ks, axial, period = context.pullback(np.array([[g]]))
            return axial[0], ks.sum(), period

        def pushforward(kz: Any, k: Any, area: Any) -> Any:
            return context.pushforward(
                np.zeros((1, 3)), np.zeros((1, 3)), [k, k], [kz], area
            )[0, 0]

        return DerivativeContext(pullback, pushforward, context)

    @native_state(
        _native.PeriodicToCwContext,
        lambda _inputs: (1, 1, 1, 1),
        map_context=map_context,
    )
    def evaluate(
        kz: Any,
        k: Any,
        area: Any,
        m: Any,
        pol: Any,
        l: Any,  # noqa: E741 - public spherical degree label
        mu: Any,
        qol: Any,
    ) -> Any:
        value, context = diff.periodic_to_cw(
            CylindricalBasis([(kz, m, pol)]),
            SphericalBasis([(l, mu, qol)]),
            [k, k],
            area,
            poltype=poltype,
        )

        return value[0, 0], map_context(context, kz, k, area, m, pol, l, mu, qol)

    return _elementwise(backend, evaluate, values, labels)


def cw_to_sw(
    backend: Backend,
    l: Any,  # noqa: E741 - public spherical degree label
    m: Any,
    polsw: Any,
    kz: Any,
    mu: Any,
    polcw: Any,
    k: Any,
    poltype: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    _options(args, kwargs)
    if backend_for(kz) is not None:
        raise NotImplementedError(
            "cw.to_sw gradients in the axial mode label kz are not available"
        )
    labels = _labels(l, m, polsw, kz, mu, polcw)
    values = (backend.array(k, complex_=True),)

    def map_context(context: Any, *_primals: Any) -> DerivativeContext:
        def pullback(g: Any) -> Any:
            return (context.pullback(np.array([[g]]))[2].sum(),)

        def pushforward(k: Any) -> Any:
            return context.pushforward(np.zeros((1, 3)), np.zeros((1, 3)), [k, k])[0, 0]

        return DerivativeContext(pullback, pushforward, context)

    @native_state(
        _native.ExpansionContext,
        lambda _inputs: (1, 1, 1, 1, 2),
        map_context=map_context,
    )
    def evaluate(
        k: Any,
        l: Any,  # noqa: E741 - public spherical degree label
        m: Any,
        polsw: Any,
        kz: Any,
        mu: Any,
        polcw: Any,
    ) -> Any:
        value, context = diff.expansion(
            SphericalBasis([(l, m, polsw)]),
            CylindricalBasis([(kz, mu, polcw)]),
            [k, k],
            poltype=poltype,
        )

        return value[0, 0], map_context(context, k, l, m, polsw, kz, mu, polcw)

    return _elementwise(backend, evaluate, values, labels)


def periodic_to_pw(backend: Backend, *args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "independent wavevector derivatives of periodic_to_pw are not available; "
        "use a periodic T-matrix's plane-wave channels for physical port gradients"
    )


def translate_periodic(
    backend: Backend,
    family: Any,
    ks: Any,
    kpar: Any,
    a: Any,
    rs: Any,
    out: Any,
    in_: Any = None,
    rsin: Any = None,
    *,
    poltype: Any = None,
    eta: Any = 0,
) -> Any:
    _labels(out, out if in_ is None else in_)
    if backend_for(eta) is not None:
        raise TypeError("the Ewald split parameter eta must be a constant")
    positions = backend.array(rs)
    positions = positions.reshape((-1, 3))
    source_positions = (
        positions if rsin is None else backend.array(rsin).reshape((-1, 3))
    )
    ks = _broadcast(backend, backend.array(ks, complex_=True), (2,))
    bloch = backend.array(kpar).reshape(-1)
    cell = backend.array(a).reshape((bloch.shape[0], bloch.shape[0]))
    destination, source = _modes.periodic_bases(
        family, np.zeros(positions.shape), out, in_, np.zeros(source_positions.shape)
    )

    @native_state(
        _native.LatticeExpansionContext,
        lambda inputs: (
            len(destination),
            len(source),
            inputs[0].shape[0],
            inputs[1].shape[0],
            family is CylindricalBasis,
        ),
    )
    def record(rs: Any, rsin: Any, ks: Any, kpar: Any, a: Any) -> Any:
        destination, source = _modes.periodic_bases(family, rs, out, in_, rsin)
        return diff.lattice_expansion(
            destination, source, ks, kpar, a, poltype=poltype, eta=eta
        )

    return backend.apply(
        record,
        (len(destination), len(source)),
        positions,
        source_positions,
        ks,
        bloch,
        cell,
    )


def sw_translate_periodic(
    backend: Backend,
    ks: Any,
    kpar: Any,
    a: Any,
    rs: Any,
    out: Any,
    in_: Any = None,
    rsin: Any = None,
    poltype: Any = None,
    eta: Any = 0,
    func: Any = None,
) -> Any:
    from . import lattice

    if func is not None and func is not lattice.lsumsw:
        raise NotImplementedError(
            "custom lattice callbacks do not provide autodiff records"
        )
    return translate_periodic(
        backend,
        SphericalBasis,
        ks,
        kpar,
        a,
        rs,
        out,
        in_,
        rsin,
        poltype=poltype,
        eta=eta,
    )


def cw_translate_periodic(
    backend: Backend,
    ks: Any,
    kpar: Any,
    a: Any,
    rs: Any,
    out: Any,
    in_: Any = None,
    rsin: Any = None,
    eta: Any = 0,
) -> Any:
    return translate_periodic(
        backend, CylindricalBasis, ks, kpar, a, rs, out, in_, rsin, eta=eta
    )

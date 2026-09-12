"""First-order PyTorch bridges to native CPU operations.

Install ``treams-rs[torch]``. Dynamic tensors must be CPU float64/complex128.
The first backward uses the retained native context; repeated backward recomputes
the native forward. Higher-order AD and torch.func transforms are unsupported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import torch

from . import diff
from ._adapters import execute, gradients, input_array

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

    from ._adapters import Array, Pullback, Record

type Output = torch.Tensor | tuple[torch.Tensor, ...]


def _numpy(value: torch.Tensor) -> Array:
    return value.detach().resolve_conj().resolve_neg().numpy()


class _Execute(torch.autograd.Function):
    @staticmethod
    @override
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any, record: Record, *values: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        snapshots = tuple(v.detach().clone() for v in values)
        primals = tuple(_numpy(v) for v in snapshots)
        outputs, pullback, _ = execute(record, primals)
        ctx.record = record
        ctx.pullback = pullback
        ctx.input_count = len(values)
        # PyTorch checks these tensors' version counters before any backward,
        # including repeats; caller mutation cannot change recomputed primals.
        ctx.save_for_backward(*values, *snapshots)
        return tuple(torch.from_numpy(np.array(v, copy=True)) for v in outputs)

    @staticmethod
    @override
    def backward(
        ctx: Any, *cotangents: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise NotImplementedError(
                "treams-rs PyTorch adapters support first-order VJPs only"
            )
        # Saved tensors validate original version counters and release snapshots
        # with the graph; NumPy aliases cannot alter the owned recomputation data.
        values = tuple(_numpy(v) for v in ctx.saved_tensors[ctx.input_count :])
        pullback = ctx.pullback
        ctx.pullback = cast("Pullback | None", None)
        if pullback is None:
            _, pullback, _ = execute(ctx.record, values)
        result = gradients(
            pullback, tuple(_numpy(g) for g in cotangents), values, conjugate=False
        )
        return (None, *(torch.from_numpy(np.array(g, copy=True)) for g in result))


def wrap(record: Record) -> Callable[..., Output]:
    """Adapt a native forward/context function, holding static options in a closure.

    Native pullbacks must return one gradient per dynamic parameter, in order.
    A single output returns a Tensor; multiple outputs return a flat tuple.
    See docs/adapters.md for native operations with reordered/extra adjoints.
    """

    def operation(*values: torch.Tensor | ArrayLike) -> Output:
        arrays = tuple(
            value
            if isinstance(value, torch.Tensor)
            else torch.from_numpy(input_array(value))
            for value in values
        )
        for array in arrays:
            if array.device.type != "cpu":
                raise ValueError("treams-rs PyTorch adapters require CPU tensors")
            if array.dtype not in (torch.float64, torch.complex128):
                raise TypeError(
                    "native adapters require float64 or complex128 parameters"
                )
        # Structure is discovered during the same native forward, without a
        # second simulation or output signature registry.
        multiple = [False]

        def forward(*primals: Array) -> tuple[Any, Any]:
            output, context = record(*primals)
            multiple[0] = isinstance(output, tuple)
            return output, context

        outputs = cast("tuple[torch.Tensor, ...]", _Execute.apply(forward, *arrays))
        return outputs if multiple[0] else outputs[0]

    return operation


def solve(
    operator: torch.Tensor | ArrayLike, rhs: torch.Tensor | ArrayLike
) -> torch.Tensor:
    """Solve A X = B using native LU and its analytic first-order VJP."""
    return cast("torch.Tensor", wrap(diff.solve)(operator, rhs))


def interaction(
    local: torch.Tensor | ArrayLike, coupling: torch.Tensor | ArrayLike
) -> torch.Tensor:
    """Solve (I - T C) X = T with native matrix pullbacks."""
    return cast("torch.Tensor", wrap(diff.interaction)(local, coupling))


def illuminate(
    local: torch.Tensor | ArrayLike,
    coupling: torch.Tensor | ArrayLike,
    incident: torch.Tensor | ArrayLike,
) -> torch.Tensor:
    """Solve only requested incident columns, with native T/C/incident adjoints."""
    return cast("torch.Tensor", wrap(diff.illuminate)(local, coupling, incident))


def sphere(
    lmax: int,
    k0: torch.Tensor | ArrayLike,
    radii: torch.Tensor | ArrayLike,
    epsilon: torch.Tensor | ArrayLike,
    mu: torch.Tensor | ArrayLike | None = None,
    kappa: torch.Tensor | ArrayLike | None = None,
) -> torch.Tensor:
    """Multilayer chiral sphere with native material, radius and frequency VJPs."""

    def record(k: Array, r: Array, e: Array, m: Array, c: Array) -> tuple[Any, Any]:
        return diff.sphere(lmax, float(k), r, e, m, c)

    return cast(
        "torch.Tensor",
        wrap(record)(
            k0,
            radii,
            epsilon,
            np.ones(np.shape(epsilon)) if mu is None else mu,
            np.zeros(np.shape(epsilon)) if kappa is None else kappa,
        ),
    )


def bessel(
    z: torch.Tensor | ArrayLike,
    *,
    order: ArrayLike,
    kind: str = "j",
    spherical: bool = False,
    derivative: bool = False,
) -> torch.Tensor:
    """Broadcast Bessel values with the order fixed and a native argument VJP."""

    def record(value: Array) -> tuple[Any, Any]:
        return diff.bessel(
            order, value, kind=kind, spherical=spherical, derivative=derivative
        )

    return cast("torch.Tensor", wrap(record)(z))

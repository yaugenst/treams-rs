"""Framework array operations and the framework Material.

Arrays and continuous metadata stay in their framework. Discrete bases stay in
Python; native callbacks rebuild them with the moving positions at execution time.

Derived physics objects (``with_polarization``, ``cascade``, ...) call their
class with explicit keywords: a shared replace helper measured 3-18% slower for
``with_polarization`` and ``cascade``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ._records import Record

import numpy as np

from . import _material, diff
from ._bases import CylindricalBasis, PlaneWavePorts, SphericalBasis
from ._polarization import pol_partners

__all__ = [
    "Backend",
    "Basis",
    "Material",
    "Operation",
    "Recorded",
    "as_material",
    "material_defaults",
    "port_modes",
    "with_zero_metadata",
]

type Basis = SphericalBasis | CylindricalBasis
# What a record returns: its output and a context with ``pullback``, or the
# pullback itself (one gradient per dynamic input).
type Recorded = tuple[Any, Any]


@dataclass(frozen=True)
class Material:
    """Relative permittivity, permeability and chirality, retaining framework values."""

    epsilon: Any = 1.0
    mu: Any = 1.0
    kappa: Any = 0.0


def as_material(value: Any) -> Material:
    """Material from a Material, an (epsilon, mu, kappa) tuple, or epsilon alone.

    A root ``treams_rs.Material`` contributes its three values.
    """
    if isinstance(value, Material):
        return value
    if isinstance(value, _material.Material):
        return Material(value.epsilon, value.mu, value.kappa)
    if isinstance(value, tuple):
        return Material(*value)
    return Material(value)


def material_defaults(epsilon: Any, mu: Any, kappa: Any) -> tuple[Any, Any]:
    """``mu`` and ``kappa``, with vacuum values in the shape of ``epsilon`` for None."""
    # Constants of the input's shape: no framework input is converted.
    shape = np.shape(epsilon)
    return (
        np.ones(shape) if mu is None else mu,
        np.zeros(shape) if kappa is None else kappa,
    )


def with_zero_metadata(
    pullback: Callable[[Any], Sequence[Any]], *metadata: Any
) -> Callable[[Any], tuple[Any, ...]]:
    """``pullback`` followed by zero gradients for the metadata a record compares."""
    return lambda g: (*pullback(g), *(np.zeros_like(value) for value in metadata))


def port_modes(
    basis: PlaneWavePorts,
) -> tuple[list[tuple[float, ...]], tuple[tuple[int, int], ...]]:
    """Distinct transverse wavevectors in first-seen order, and (group, pol) per port."""
    groups: dict[tuple[float, ...], int] = {}
    modes = tuple(
        (groups.setdefault(tuple(q), len(groups)), int(pol))
        for q, pol in zip(basis.components, basis.pol, strict=True)
    )
    return list(groups), modes


class Operation(Protocol):
    """Run one native record as a differentiable framework operation.

    Each adapter module defines one, named ``_operation``. ``shape`` and
    ``real`` declare the output; JAX needs them to build its callback, Advect
    and PyTorch read the output from the native forward.
    """

    def __call__(
        self, record: Record, *values: Any, shape: tuple[int, ...], real: bool = False
    ) -> Any: ...


class Backend:
    """Framework array operations and the one entry point for native records.

    ``xp`` is the framework's NumPy-style namespace (``axis=`` keywords).
    ``operation`` runs a native record in the framework; ``apply`` is its
    checked form. ``asarray(value, dtype=...)`` replaces ``xp.asarray`` where
    the framework needs it, and ``validate`` checks every value converted by
    ``array``.
    """

    def __init__(
        self,
        xp: Any,
        operation: Operation,
        *,
        asarray: Callable[..., Any] | None = None,
        validate: Callable[[Any], None] | None = None,
    ):
        self.xp, self.operation, self.validate = xp, operation, validate
        self.asarray = xp.asarray if asarray is None else asarray

    def change_polarization(
        self, value: Any, basis: Any, source: str, target: str, axes: Sequence[int]
    ) -> Any:
        """``value`` with the polarization change of ``basis`` applied along ``axes``.

        The change matrix of ``changepoltype`` pairs each mode with its partner
        of the other ``pol`` (``_polarization.change_polarization``), so each
        axis costs one gather and two scaled terms instead of a dense product.
        """
        if source == target:
            return value
        if {source, target} != {"helicity", "parity"}:
            raise ValueError("polarization conversion must switch helicity and parity")
        partners = pol_partners(basis)  # raises unless every mode has its partner
        half = np.sqrt(0.5)
        diagonal = np.where(basis.pol == 0, -half, half)
        for axis in axes:
            shape = [1] * value.ndim
            shape[axis] = -1
            partner = value[(slice(None),) * axis + (partners,)]
            value = self.array(diagonal.reshape(shape)) * value + half * partner
        return value

    def require_achiral(self, value: Any, *media: Material) -> Any:
        """Pass ``value`` through; raise unless every medium has zero chirality.

        Parity channels exist only in achiral media. Static chiralities are
        checked at once; framework values are checked in one guard.
        """
        kappas = [medium.kappa for medium in media]
        if all(
            isinstance(k, (int, float, complex, np.number, np.ndarray)) for k in kappas
        ):
            _require_zero_kappa(value, *kappas)
            return value
        return self.guard(
            _require_zero_kappa, value, *(self.array(k, complex_=True) for k in kappas)
        )

    def array(self, value: Any, *, complex_: bool = False) -> Any:
        if self.validate is not None:
            self.validate(value)
        dtype = self.xp.complex128 if complex_ else self.xp.float64
        return self.asarray(value, dtype=dtype)

    def positions(self, basis: Basis, positions: Any = None) -> Any:
        """``positions``, or the positions of ``basis`` for None, as an array."""
        return self.array(basis.positions if positions is None else positions)

    def vector(self, value: Any) -> Any:
        array = self.array(value)
        return self.stack((array,)) if array.ndim == 0 else array.reshape(-1)

    def stack(self, values: Sequence[Any], axis: int = 0) -> Any:
        return self.xp.stack(tuple(values), axis=axis)

    def concat(self, values: Sequence[Any], axis: int = 0) -> Any:
        return self.xp.concatenate(tuple(values), axis=axis)

    def upper_half(self, value: Any) -> Any:
        """Select the root with nonnegative imaginary part (decaying branch)."""
        return self.xp.where(value.imag < 0, -value, value)

    def ks(self, medium: Material, k0: Any) -> Any:
        n = self.xp.sqrt(
            self.array(medium.epsilon, complex_=True)
            * self.array(medium.mu, complex_=True)
        )
        return k0 * self.upper_half(self.stack((n - medium.kappa, n + medium.kappa)))

    def medium_key(self, medium: Material, k0: Any = None) -> Any:
        """Complex [k0,] epsilon, mu and kappa, stacked for guards to compare."""
        values = (medium.epsilon, medium.mu, medium.kappa)
        return self.stack(
            [
                self.array(v, complex_=True)
                for v in (values if k0 is None else (k0, *values))
            ]
        )

    def plane_vectors(self, q: Any, ks: Any, *, positive: bool = True) -> Any:
        """Local (q1, q2, normal) wavevectors, propagating along +/- the normal."""
        kz = self.upper_half(self.xp.sqrt(ks**2 - self.xp.sum(q * q, axis=1)))
        return self.concat(
            (self.array(q, complex_=True), (kz if positive else -kz)[:, None]), axis=1
        )

    def change_port_polarization(
        self,
        value: Any,
        modes: Sequence[tuple[int, int]],
        source: str,
        target: str,
        axes: Sequence[int],
    ) -> Any:
        """``change_polarization`` of plane ports labelled (group, pol)."""
        # The group index stands in for kx (ky = 0). The change only pairs
        # modes whose labels agree, so it never reads the wavevector values.
        labels = PlaneWavePorts([(float(group), 0.0, pol) for group, pol in modes])
        return self.change_polarization(value, labels, source, target, axes)

    def impedance(self, medium: Material) -> Any:
        return self.xp.sqrt(
            self.array(medium.mu, complex_=True)
            / self.array(medium.epsilon, complex_=True)
        )

    def expansion(
        self,
        destination: Basis,
        source: Basis,
        destination_positions: Any,
        source_positions: Any,
        ks: Any,
        *,
        poltype: str,
        singular: bool = False,
    ) -> Any:
        """``diff.expansion`` from ``source`` to ``destination`` at moving positions."""

        def record(destination_at: Any, source_at: Any, ks: Any) -> Any:
            return diff.expansion(
                type(destination)(destination.modes, destination_at),
                type(source)(source.modes, source_at),
                ks,
                poltype=poltype,
                singular=singular,
            )

        return self.apply(
            record,
            (len(destination), len(source)),
            destination_positions,
            source_positions,
            ks,
        )

    def lattice_expansion(
        self,
        basis: Basis,
        positions: Any,
        ks: Any,
        kpar: Any,
        lattice: Any,
        *,
        poltype: str,
        eta: complex,
    ) -> Any:
        """``diff.lattice_expansion`` within ``basis`` at moving positions."""

        def record(destination: Any, source: Any, ks: Any, q: Any, a: Any) -> Any:
            return diff.lattice_expansion(
                type(basis)(basis.modes, destination),
                type(basis)(basis.modes, source),
                ks,
                q,
                a,
                poltype=poltype,
                eta=eta,
            )

        return self.apply(
            record, (len(basis), len(basis)), positions, positions, ks, kpar, lattice
        )

    def apply(
        self, record: Record, shape: tuple[int, ...], *values: Any, real: bool = False
    ) -> Any:
        """Run ``record(*values) -> (output, context_or_pullback)`` as one framework op.

        This is ``operation`` with a check of the declared output. ``values``
        are the dynamic inputs; bases, labels and conventions stay static in the
        record's closure. The pullback returns one gradient per dynamic input,
        in order. Each call runs one native forward: Advect keeps its context
        for the reverse pass, PyTorch keeps it for the first backward and
        records again for a repeated one, and JAX records again in every
        reverse pass.

        ``shape`` is the output shape and ``real`` selects a float64 output (and
        a real cotangent) instead of complex128. All three frameworks check both,
        so a wrong declaration fails everywhere, not only in JAX, which builds
        its callback's output description from them.

        A record has exactly one output. Records with two results, the
        eigenvalues and eigenvectors of ``bands``, stack them with
        ``np.vstack`` and the caller slices them apart. One output keeps the
        declaration to one shape and one dtype per call.

        Checks of metadata that must agree run inside records: under
        ``jax.jit`` that metadata is traced, and its values exist only in the
        native callback. Such a guard gives the metadata zero gradients. The
        guards are:

        - ``require_same_medium``: equal k0 and medium of a response and
          its illumination, or of cluster particles (through ``guard``);
        - ``TMatrix.cross_sections``: a nonabsorbing, propagating exterior and
          a positive flux (through ``guard``);
        - ``SMatrix._incident``: equal k0 and medium, and one port per incident
          wavevector; the record also selects the port amplitudes;
        - ``SMatrix.cascade``: equal wavevectors, k0 and adjacent media, inside
          the ``smatrix_add`` record;
        - ``SMatrix.bands``: equal outer media, inside the ``bands`` record;
        - ``PeriodicResponse.to_smatrix``: ports on lattice diffraction
          orders, inside the channels record.
        """

        def checked(*primals: Any) -> Any:
            output, context = record(*primals)
            if np.shape(output) != shape or np.iscomplexobj(output) == real:
                raise ValueError(
                    f"{getattr(record, '__qualname__', record)}: declared "
                    f"{'real' if real else 'complex'} output of shape {shape}, "
                    f"native returned {np.asarray(output).dtype} of shape "
                    f"{np.shape(output)}"
                )
            return output, context

        return self.operation(checked, *values, shape=shape, real=real)

    def guard(self, check: Callable[..., None], value: Any, *metadata: Any) -> Any:
        """Pass ``value`` through after ``check(value, *metadata)`` in one record.

        ``check`` raises on a mismatch; the metadata gets zero gradients. Each
        guard is one native callback. Guards that also compute something
        (incident ports, cascade, bands, ``to_smatrix``) keep their own
        records; ``apply`` lists them all.
        """

        def record(array: Any, *compared: Any) -> Recorded:
            check(array, *compared)
            return array, with_zero_metadata(lambda g: (g,), *compared)

        return self.apply(record, tuple(value.shape), value, *metadata)

    def require_same(self, other: Backend, message: str) -> None:
        """Raise ``TypeError(message)`` unless ``other`` is this backend."""
        if other is not self:
            raise TypeError(message)

    def require_same_medium(self, value: Any, first: Any, second: Any) -> Any:
        """Pass ``value`` through; raise unless both objects share k0 and medium."""
        first._backend.require_same(
            second._backend, "physical objects must use the same framework namespace"
        )
        return self.guard(
            _require_equal_keys,
            value,
            self.medium_key(first.medium, first.k0),
            self.medium_key(second.medium, second.k0),
        )


def _require_zero_kappa(_value: Any, *kappas: Any) -> None:
    if any(np.any(np.asarray(kappa) != 0) for kappa in kappas):
        raise ValueError("parity polarization requires an achiral embedding medium")


def _require_equal_keys(_value: Any, left: Any, right: Any) -> None:
    if not np.array_equal(left, right):
        raise ValueError("physical operations require matching k0 and embedding medium")

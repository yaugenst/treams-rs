"""Framework S-matrices and layer stacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _native, diff
from ._bases import ALIGNMENT_AXIS
from ._framework_backend import Backend, Material, Recorded, with_zero_metadata
from ._framework_waves import HasPorts, PlaneWave, PortSet, PortWave, Wave
from ._promotion import promote
from ._records import DerivativeContext
from ._results import BandModes, PowerBalance, ScatteredPorts
from ._saved import ArraySpec, SavedRecord, native_state

__all__ = ["SMatrix", "stack"]


@dataclass(frozen=True)
class _IncidentSelection:
    """The fixed matching of an incident wave to diffraction ports."""

    selection: Any
    metadata: tuple[Any, ...]

    def pullback(self, gradient: Any) -> tuple[Any, ...]:
        """Map port gradients to incident coefficients; metadata gradients are zero."""
        return (
            self.selection.T @ gradient,
            *(np.zeros_like(value) for value in self.metadata),
        )

    def pushforward(self, tangent: Any, *metadata: Any) -> Any:
        """Select incident tangents at ports; metadata does not affect the selection."""
        return self.selection @ tangent


def _power_context(context: Any, *_primals: Any) -> DerivativeContext:
    def pullback(gradient: Any) -> Any:
        return context.pullback(np.asarray(gradient, dtype=np.complex128))

    return DerivativeContext(pullback, context.pushforward, native_context=context)


def _cascade_context(
    context: Any, _lower: Any, _higher: Any, *metadata: Any
) -> DerivativeContext:
    return with_zero_metadata(context, *metadata)


def _bands_context(
    context: Any, _array: Any, _period: Any, outer: Any
) -> DerivativeContext:
    def pullback(gradient: Any) -> Any:
        return context.pullback(gradient[0], gradient[1:])

    def pushforward(*tangents: Any) -> Any:
        return np.vstack(context.pushforward(*tangents))

    return with_zero_metadata(
        DerivativeContext(pullback, pushforward, native_context=context), outer
    )


class SMatrix(HasPorts):
    """S-matrix of framework arrays: the four blocks of a planar system.

    The constructors ``smatrix``, ``interface``, ``slab``, ``multilayer_slab``,
    ``propagation`` and ``stack`` of advect, jax and torch build it, and
    ``cascade`` and ``PeriodicResponse.to_smatrix`` return one. The ports and
    the polarization convention are fixed; the blocks, k0 and the media may
    carry gradients.
    """

    array: Any
    """Blocks of shape (2, 2, ports, ports); index 0 is the positive side."""
    k0: Any
    """Vacuum wavenumber."""
    polarization: str
    """Polarization convention, "helicity" or "parity"."""
    ports: PortSet
    """Plane-wave ports of both sides."""

    @property
    def negative_medium(self) -> Material:
        """Exterior on the negative side of the plane normal."""
        return self.media[1]

    @property
    def positive_medium(self) -> Material:
        """Exterior on the positive side of the plane normal."""
        return self.media[0]

    def __init__(
        self,
        array: Any,
        *,
        ports: PortSet,
        k0: Any,
        media: tuple[Material, Material],
        backend: Backend,
        polarization: str = "helicity",
    ):
        if polarization == "parity":
            array = backend.require_achiral(array, *media)
        self.array, self.ports, self.k0 = array, ports, k0
        self.media, self._backend = media, backend
        self.polarization = polarization

    def with_polarization(self, polarization: str) -> SMatrix:
        """Convert the port polarization convention using a static basis matrix."""
        if polarization == self.polarization:
            return self

        return SMatrix(
            self._backend.change_port_polarization(
                self.array, self.ports.modes, self.polarization, polarization, (2, 3)
            ),
            ports=self.ports,
            k0=self.k0,
            media=self.media,
            backend=self._backend,
            polarization=polarization,
        )

    def _incident(self, incident: Any, side: str) -> Any:
        b = self._backend
        incident = promote(incident, b)
        if not isinstance(incident, (PlaneWave, PortWave)):
            if isinstance(incident, Wave):
                raise ValueError(
                    "planar scattering requires a plane wave or plane-port wave"
                )
            return b.array(incident, complex_=True)
        b.require_same(
            incident._backend, "incident wave must use the same framework namespace"
        )
        medium = self.media[1 if side == "negative" else 0]  # (positive, negative)
        axis = ALIGNMENT_AXIS[self.ports.alignment]
        if isinstance(incident, PlaneWave):

            def check_direction(_: Any, direction: Any) -> None:
                if (direction[axis] > 0) != (side == "negative") or direction[
                    axis
                ] == 0:
                    raise ValueError(
                        "plane wave propagates away from the selected incident side"
                    )

            if incident._fixed_direction:
                check_direction(None, incident.direction)
            transverse = incident._vectors()[:, [(axis + 1) % 3, (axis + 2) % 3]]
            if not incident._fixed_direction:
                transverse = b.guard(check_direction, transverse, incident.direction)
            pols = np.array((0, 1))
            coefficients = b.change_port_polarization(
                incident.coefficients,
                ((0, 0), (0, 1)),
                incident.polarization,
                self.polarization,
                (0,),
            )
        else:
            if (
                incident.positive != (side == "negative")
                or incident.ports.alignment != self.ports.alignment
            ):
                raise ValueError(
                    "port wave propagates away from the selected incident side"
                )
            if incident.polarization != self.polarization:
                incident = incident._complete_polarizations().with_polarization(
                    self.polarization
                )
            transverse = incident.ports.transverse_wavevectors[incident.ports.groups]
            pols = incident.ports.pols
            coefficients = incident.coefficients

        groups = self.ports.groups
        matching = self.ports.pols[:, None] == pols

        def record(c: Any, vectors: Any, ports: Any, left: Any, right: Any) -> Recorded:
            if not np.array_equal(left, right):
                raise ValueError(
                    "incident wave requires matching k0 and exterior medium"
                )
            # (modes, incident columns): same polarization and transverse vector.
            close = matching & np.all(
                np.isclose(
                    ports[groups][:, None], vectors[None], atol=1e-12, rtol=1e-12
                ),
                axis=-1,
            )
            unique = np.count_nonzero(close, axis=0) == 1
            if np.any(~unique & (c != 0)):
                raise ValueError(
                    "incident plane wave must match one represented diffraction port"
                )
            selection = (close & unique).astype(np.complex128)
            return selection @ c, _IncidentSelection(
                selection, (vectors, ports, left, right)
            )

        def restore(
            state: tuple[Any, ...], _coefficients: Any, *metadata: Any
        ) -> _IncidentSelection:
            return _IncidentSelection(state[0], metadata)

        prepared = SavedRecord(
            record,
            lambda inputs: (
                ArraySpec(
                    (len(self.ports.modes), inputs[0].shape[0]),
                    np.dtype(np.complex128),
                ),
            ),
            lambda context: (context.selection,),
            restore,
        )
        return b.apply(
            prepared,
            (len(self.ports.modes),),
            coefficients,
            transverse,
            self.ports.transverse_wavevectors,
            b.medium_key(medium, self.k0),
            b.medium_key(incident.medium, incident.k0),
        )

    def scatter(
        self, *, negative: Any = None, positive: Any = None
    ) -> ScatteredPorts[PortWave]:
        """Scatter one coherent illumination supplied on either or both sides."""
        if negative is None and positive is None:
            raise ValueError("supply an incident wave on at least one side")
        b = self._backend
        zero = b.array(np.zeros(len(self.ports.modes)), complex_=True)
        up = zero if negative is None else self._incident(negative, "negative")
        down = zero if positive is None else self._incident(positive, "positive")
        return ScatteredPorts(
            PortWave(
                self.array[0, 0] @ up + self.array[0, 1] @ down,
                system=self,
                positive=True,
            ),
            PortWave(
                self.array[1, 0] @ up + self.array[1, 1] @ down,
                system=self,
                positive=False,
            ),
        )

    def power(self, incident: Any, *, side: str | None = None) -> PowerBalance[Any]:
        """Transmission/reflection for port amplitudes; fixed port wavevectors."""
        incident = promote(incident, self._backend)
        if side is None:
            if isinstance(incident, PlaneWave) and not incident._fixed_direction:
                raise ValueError(
                    "supply side when the incident direction is differentiable"
                )
            axis = ALIGNMENT_AXIS[self.ports.alignment]
            side = (
                ("negative" if incident.direction[axis] > 0 else "positive")
                if isinstance(incident, PlaneWave)
                else ("negative" if incident.positive else "positive")
                if isinstance(incident, PortWave)
                else "negative"
            )
        if side not in ("negative", "positive"):
            raise ValueError("incident side must be negative or positive")
        direction = "up" if side == "negative" else "down"
        b = self._backend
        incoming = self._incident(incident, side).reshape(len(self.ports.modes), -1)

        @native_state(
            _native.SMatrixTrContext,
            lambda inputs: (
                inputs[0].shape[-1],
                inputs[1].shape[1],
                inputs[4].shape[0],
            ),
            map_context=_power_context,
        )
        def record(a: Any, i: Any, ks: Any, zs: Any, q: Any) -> Any:
            value, context = diff.smatrix_tr(
                a,
                i,
                ks,
                zs,
                q,
                modes=self.ports.modes,
                poltype=self.polarization,
                axis=ALIGNMENT_AXIS[self.ports.alignment],
                modetype=direction,
                fixed_q=self.ports.fixed_q,
            )
            return value, _power_context(context)

        power = b.apply(
            record,
            (2, incoming.shape[1]),
            self.array,
            incoming,
            b.stack([b.plane_ks(m, self.k0) for m in self.media]),
            b.stack([b.impedance(m) for m in self.media]),
            self.ports.transverse_wavevectors,
            real=True,
        )
        return PowerBalance(b.xp.squeeze(power[0]), b.xp.squeeze(power[1]))

    def cascade(self, next_layer: SMatrix) -> SMatrix:
        """Compose adjacent systems whose port wavevectors have the same dependence.

        Fixed basis ports cannot be mixed with diffraction-order ports: matching
        their current wavevectors does not match their derivatives. Use a fixed
        basis for both systems, or diffraction orders for both.
        """
        upper = promote(next_layer, self._backend)
        if self.ports.fixed_q != upper.ports.fixed_q:
            raise ValueError(
                "stacked systems cannot mix fixed and diffraction-order ports; "
                "use the same port definition for both systems"
            )
        if (
            self.ports.modes != upper.ports.modes
            or self.ports.alignment != upper.ports.alignment
            or self.polarization != upper.polarization
        ):
            raise ValueError("stacked systems require the same port basis")
        b = self._backend
        b.require_same(
            upper._backend, "stacked systems require one framework namespace"
        )

        @native_state(
            _native.SMatrixAddContext,
            lambda inputs: (inputs[0].shape[-1],),
            map_context=_cascade_context,
        )
        def record(
            lower: Any,
            higher: Any,
            first_q: Any,
            second_q: Any,
            first_k: Any,
            second_k: Any,
            first_medium: Any,
            second_medium: Any,
        ) -> Recorded:
            if not (
                np.array_equal(first_q, second_q)
                and np.array_equal(first_k, second_k)
                and np.array_equal(first_medium, second_medium)
            ):
                raise ValueError(
                    "stacked systems require matching wavevectors, k0 and adjacent medium"
                )
            value, context = diff.smatrix_add(lower, higher)
            return value, _cascade_context(
                context,
                lower,
                higher,
                first_q,
                second_q,
                first_k,
                second_k,
                first_medium,
                second_medium,
            )

        array = b.apply(
            record,
            tuple(self.array.shape),
            self.array,
            upper.array,
            self.ports.transverse_wavevectors,
            upper.ports.transverse_wavevectors,
            b.array(self.k0),
            b.array(upper.k0),
            # media = (positive, negative): this positive side meets the
            # upper system's negative side.
            b.medium_key(self.media[0]),
            b.medium_key(upper.media[1]),
        )
        return SMatrix(
            array,
            ports=self.ports,
            k0=self.k0,
            media=(upper.media[0], self.media[1]),  # (positive, negative)
            backend=b,
            polarization=self.polarization,
        )

    def bands(self, period: Any) -> BandModes[Any]:
        """Principal-branch normal Bloch wavenumbers and right vectors."""
        b = self._backend
        n = 2 * len(self.ports.modes)

        @native_state(
            _native.BandsContext,
            lambda inputs: (inputs[0].shape[-1],),
            map_context=_bands_context,
        )
        def record(a: Any, p: Any, outer: Any) -> Recorded:
            if not np.array_equal(outer[0], outer[1]):
                raise ValueError("bands require equal outer media")
            (values, vectors), context = diff.bands(a, float(p))
            return np.vstack((values, vectors)), _bands_context(context, a, p, outer)

        packed = b.apply(
            record,
            (n + 1, n),
            self.array,
            b.array(period),
            b.stack([b.medium_key(m) for m in self.media]),
        )
        return BandModes(packed[0], packed[1:])


def stack(layers: Any) -> SMatrix:
    """Cascade layers from the negative to positive side of the port normal."""
    if not layers:
        raise ValueError("stack requires at least one layer")
    from ._dispatch import backend_for

    backend = backend_for(layers)
    if backend is not None:
        layers = [promote(layer, backend) for layer in layers]
    result = layers[0]
    for layer in layers[1:]:
        result = result.cascade(layer)
    return result

"""Matrix-free requested illuminations for finite sphere clusters in vacuum.

Only the supplied incident columns are solved. Native restarted GMRES checks
its true residual before returning and raises ValueError on nonconvergence.
Geometry and Mie coefficients are shared across solves and retained pullbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from . import _native

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class Convergence(NamedTuple):
    """True residual certificate for one forward or adjoint column."""

    iterations: int
    residual_norm: float
    rhs_norm: float


class Solution(NamedTuple):
    """Requested scattered multipoles and one convergence report per column."""

    coefficients: NDArray[np.complex128]
    convergence: tuple[Convergence, ...]


class Gradient(NamedTuple):
    """Physical and incident cotangents under the real Hermitian pairing."""

    radii: NDArray[np.float64]
    positions: NDArray[np.float64]
    epsilon: NDArray[np.complex128]
    k0: float
    incident: NDArray[np.complex128]
    convergence: tuple[Convergence, ...]


def _columns(value: ArrayLike) -> tuple[NDArray[np.complex128], bool]:
    array = np.asarray(value, dtype=np.complex128)
    vector = array.ndim == 1
    return (array[:, None] if vector else array), vector


class Pullback:
    """One-use native implicit adjoint; all recorded numerical inputs are owned."""

    def __init__(self, context: _native.IterativeContext, vector: bool) -> None:
        self._context = context
        self._vector = vector

    def pullback(self, cotangent: ArrayLike) -> Gradient:
        """Differentiate the converged solution without differentiating iterations.

        The adjoint uses the forward tolerance and independently checks its true
        residual. A failed solve raises instead of returning an unqualified VJP.
        """
        values, vector = _columns(cotangent)
        if vector != self._vector:
            raise ValueError("cotangent shape does not match forward output")
        radii, positions, epsilon, k0, incident, reports = self._context.pullback(
            values
        )
        return Gradient(
            radii,
            positions,
            epsilon,
            k0,
            incident[:, 0] if vector else incident,
            tuple(Convergence(*report) for report in reports),
        )


class SphereCluster:
    """Reusable matrix-free operator for homogeneous nonmagnetic spheres.

    ``lmax`` defines a common local helicity basis for every sphere. The incident
    vector has ``dimension = N * 2*lmax*(lmax+2)`` entries, ordered by particle,
    then the usual spherical modes. Pass shape ``(dimension, P)`` for P selected
    illuminations. Radius, position and k0 units must be consistent.

    This avoids global dense coupling, local-T, factorization and gradient
    matrices. Pair translations are recomputed at each iteration; convergence
    and speed depend on the physical configuration and multipole truncation.
    """

    def __init__(
        self,
        lmax: int,
        k0: float,
        radii: ArrayLike,
        epsilon: ArrayLike,
        positions: ArrayLike,
    ) -> None:
        self._operator = _native.NativeSphereCluster(
            lmax,
            k0,
            np.asarray(radii, dtype=np.float64),
            np.asarray(epsilon, dtype=np.complex128),
            np.asarray(positions, dtype=np.float64),
        )

    @property
    def dimension(self) -> int:
        """Number of local multipoles across all particles."""
        return self._operator.dimension

    def solve(
        self,
        incident: ArrayLike,
        *,
        rtol: float = 1e-10,
        atol: float = 0.0,
        restart: int = 30,
        max_iterations: int = 300,
    ) -> Solution:
        """Solve requested columns until ||A X - T B|| <= max(atol, rtol ||T B||).

        ``restart`` controls Krylov workspace, independent of the number of
        illuminations. No forward iteration history is retained.
        """
        values, vector = _columns(incident)
        coefficients, reports = self._operator.solve(
            values, rtol=rtol, atol=atol, restart=restart, max_iterations=max_iterations
        )
        return Solution(
            coefficients[:, 0] if vector else coefficients,
            tuple(Convergence(*report) for report in reports),
        )

    def solve_with_pullback(
        self,
        incident: ArrayLike,
        *,
        rtol: float = 1e-10,
        atol: float = 0.0,
        restart: int = 30,
        max_iterations: int = 300,
    ) -> tuple[Solution, Pullback]:
        """Solve and retain only the inputs and requested solution for an adjoint."""
        values, vector = _columns(incident)
        coefficients, context, reports = self._operator.solve_with_pullback(
            values, rtol=rtol, atol=atol, restart=restart, max_iterations=max_iterations
        )
        return (
            Solution(
                coefficients[:, 0] if vector else coefficients,
                tuple(Convergence(*report) for report in reports),
            ),
            Pullback(context, vector),
        )

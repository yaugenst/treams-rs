"""Matrix-free scattering by finite clusters of homogeneous spheres in vacuum.

treams-rs extension. ``SphereCluster`` solves the multiple-scattering equations
with restarted GMRES and never forms the dense interaction matrix, so memory
grows with the number of spheres, not with its square. It solves only the
incident columns you pass. The coupled system is ``A x = T b``: x holds the
scattered coefficients, b the incident coefficients, T the sphere T-matrices,
and A is the multiple-scattering matrix, the identity minus T times the
translations between spheres. Each solve stops when the true residual
``||A x - T b||`` is at most ``max(atol, rtol ||T b||)`` and raises ValueError
otherwise. The solver computes the geometry and the Mie coefficients once and
reuses them for every solve and gradient.

``SphereCluster.record`` returns the solution and an ``IterativeContext``. A
record is a function that returns a value and a context; ``context.pullback(g)``
takes the gradient ``g`` of a real loss with respect to the value and returns
the gradients with respect to the inputs. The context supports repeated
pushforwards and pullbacks using the saved forward solution. For a
few spheres, the dense ``treams_rs.Cluster`` and ``diff.sphere_cluster`` build
the full T-matrix.

Derivative solves use only ``rtol`` so small directions and loss gradients keep
their relative accuracy. If the primal solve used ``rtol=0``, derivatives use
the default ``1e-10`` instead. ``atol`` applies only to primal solves; restart
and iteration limits apply to both. Derivatives are approximate to the solver
accuracy, so tighten ``rtol`` when needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from . import _native
from ._bases import SphericalBasis as _SphericalBasis
from ._material import Material as _Material
from ._waves import PlaneWave as _PlaneWave
from ._waves import Wave as _Wave

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    # Annotation-only names; runtime code uses the private aliases above.
    from ._waves import PlaneWave, Wave

__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_RESTART",
    "DEFAULT_RTOL",
    "Convergence",
    "Gradient",
    "IterativeContext",
    "ScatteringSolution",
    "Solution",
    "SphereCluster",
]

# The GMRES defaults mirror treams_core linalg::GmresOptions::default.
DEFAULT_RTOL = 1e-10
"""Relative residual tolerance, with respect to the norm of the right-hand side."""
DEFAULT_ATOL = 0.0
"""Absolute residual tolerance, useful for very small right-hand sides."""
DEFAULT_RESTART = 30
"""Krylov vectors kept before GMRES restarts."""
DEFAULT_MAX_ITERATIONS = 300
"""Arnoldi iterations allowed per column, over all restarts, before GMRES gives up."""


class Convergence(NamedTuple):
    """GMRES result of one column of a solve or of a pullback.

    ``residual_norm`` is the true residual norm at the end, recomputed from the
    solution; ``rhs_norm`` is the norm of the right-hand side.
    """

    iterations: int
    """Arnoldi iterations GMRES performed, over all restarts."""
    residual_norm: float
    """Norm of the true residual at the end."""
    rhs_norm: float
    """Norm of the right-hand side."""


class Solution(NamedTuple):
    """Scattered multipole coefficients and one ``Convergence`` per column."""

    coefficients: NDArray[np.complex128]
    """Scattered coefficients, with the shape of the incident coefficients."""
    convergence: tuple[Convergence, ...]
    """One report per column."""


class ScatteringSolution(NamedTuple):
    """Scattered (singular) wave and one ``Convergence`` per illumination."""

    wave: Wave
    """Scattered wave in the solver's basis."""
    convergence: tuple[Convergence, ...]
    """One report per illumination."""


class Gradient(NamedTuple):
    """Gradients of a real loss with respect to the inputs of a record.

    The fields follow the argument order of ``SphereCluster`` and ``record``,
    then the convergence of each column the pullback solves. Complex gradients
    follow dL = Re sum(conj(g) dx).
    """

    k0: float
    """Gradient with respect to k0."""
    radii: NDArray[np.float64]
    """Gradients with respect to the radii, shape (N,)."""
    epsilon: NDArray[np.complex128]
    """Gradients with respect to the permittivities, shape (N,)."""
    positions: NDArray[np.float64]
    """Gradients with respect to the positions, shape (N, 3)."""
    incident: NDArray[np.complex128]
    """Gradient with respect to the incident coefficients, with their shape."""
    convergence: tuple[Convergence, ...]
    """One report per column of the conjugate-transposed solve."""


def _columns(value: ArrayLike) -> tuple[NDArray[np.complex128], bool]:
    array = np.asarray(value, dtype=np.complex128)
    vector = array.ndim == 1
    return (array[:, None] if vector else array), vector


def _restore(values: NDArray[np.complex128], vector: bool) -> NDArray[np.complex128]:
    """Undo _columns for a result with one column per illumination."""
    return values[:, 0] if vector else values


def _reports(reports: list[tuple[int, float, float]]) -> tuple[Convergence, ...]:
    return tuple(Convergence(*report) for report in reports)


class IterativeContext:
    """Reusable derivative context of ``SphereCluster.record``.

    It holds copies of every input, so later changes to your arrays leave the
    gradients unchanged.
    """

    def __init__(self, context: _native.IterativeContext, vector: bool) -> None:
        self._context = context
        self._vector = vector

    def pushforward(
        self,
        k0: float,
        radii: ArrayLike,
        epsilon: ArrayLike,
        positions: ArrayLike,
        incident: ArrayLike,
    ) -> Solution:
        """Directional derivative of the converged scattered coefficients.

        Pass one direction for each continuous input, in constructor order
        followed by the incident coefficients. The tangent solves the
        linearized scattering equation using only the forward solver's
        ``rtol`` (or ``1e-10`` when it was zero), ignoring ``atol``.
        Its convergence reports certify that tangent solve; GMRES iterations
        are not differentiated.
        """
        columns, vector = _columns(incident)
        if vector != self._vector:
            raise ValueError("incident tangent shape does not match forward input")
        values, convergence = self._context.pushforward(
            k0, radii, epsilon, positions, columns
        )
        return Solution(_restore(values, vector), _reports(convergence))

    def pullback(self, cotangent: ArrayLike) -> Gradient:
        """Gradients of a real loss with respect to the inputs of ``record``.

        The pullback differentiates the converged solution, not the GMRES
        iterations: it solves the conjugate-transposed system using the same
        relative-only tolerance as ``pushforward`` and checks its true
        residual. A failed solve raises ValueError and returns no gradient.

        Args:
            cotangent: Gradient of the loss with respect to
                ``Solution.coefficients``, with the same shape.
        """
        values, vector = _columns(cotangent)
        if vector != self._vector:
            raise ValueError("cotangent shape does not match forward output")
        k0, radii, epsilon, positions, incident, convergence = self._context.pullback(
            values
        )
        return Gradient(
            k0,
            radii,
            epsilon,
            positions,
            _restore(incident, vector),
            _reports(convergence),
        )


class SphereCluster:
    """Matrix-free multiple-scattering operator of homogeneous nonmagnetic spheres.

    Every sphere uses the local helicity basis ``SphericalBasis.default(lmax)``
    at its position. An incident vector has ``dimension = N * 2*lmax*(lmax+2)``
    entries for N spheres, ordered by sphere, then by mode; pass shape
    ``(dimension, P)`` for P illuminations. Radii, positions and 1/k0 share one
    length unit.

    The operator stores no dense coupling, T-matrix, factorization or gradient
    matrix. It recomputes the pair translations at every GMRES iteration, so
    the run time depends on the geometry, the materials and ``lmax``.

    Args:
        lmax: Largest multipole degree of every sphere.
        k0: Vacuum angular wavenumber.
        radii: Sphere radii, shape (N,).
        epsilon: Relative permittivities, shape (N,).
        positions: Sphere centres, shape (N, 3).
    """

    def __init__(
        self,
        lmax: int,
        k0: float,
        radii: ArrayLike,
        epsilon: ArrayLike,
        positions: ArrayLike,
    ) -> None:
        self._operator = _native.IterativeSphereCluster(
            lmax,
            k0,
            np.asarray(radii, dtype=np.float64),
            np.asarray(epsilon, dtype=np.complex128),
            np.asarray(positions, dtype=np.float64),
        )
        self.k0 = float(k0)
        self.basis = _SphericalBasis.default(
            lmax, nmax=len(np.asarray(radii)), positions=positions
        )

    @property
    def dimension(self) -> int:
        """Number of local multipoles across all particles."""
        return self._operator.dimension

    def solve(
        self,
        incident: ArrayLike,
        *,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        restart: int = DEFAULT_RESTART,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> Solution:
        """Scattered coefficients for the incident columns you pass.

        GMRES stops when ``||A x - T b|| <= max(atol, rtol ||T b||)`` for each
        column and raises ValueError otherwise.

        Up to eight columns share translation evaluations while keeping independent
        Krylov bases and convergence checks. This uses up to eight columns' Krylov
        workspace at once; the interaction matrix is never stored.

        Args:
            incident: Incident coefficients, shape (dimension,) or
                (dimension, P).
            rtol: Relative tolerance on the true residual.
            atol: Absolute tolerance on the true residual.
            restart: Krylov vectors kept before GMRES restarts; the memory per
                column grows with it.
            max_iterations: Arnoldi iterations allowed per column, over all
                restarts. Each restart also applies the operator once to check
                the true residual.
        """
        values, vector = _columns(incident)
        coefficients, reports = self._operator.solve(
            values, rtol=rtol, atol=atol, restart=restart, max_iterations=max_iterations
        )
        return Solution(_restore(coefficients, vector), _reports(reports))

    def scatter(
        self,
        incident: PlaneWave | Wave,
        *,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        restart: int = DEFAULT_RESTART,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> ScatteringSolution:
        """Scatter a plane or multipole wave in vacuum at the solver's k0.

        The incident wave must use helicity polarization. The result holds a
        singular ``Wave`` in ``self.basis``, which evaluates fields, and the
        ``Convergence`` of each illumination. Use ``solve`` for coefficient
        arrays. The tolerances are those of ``solve``.
        """
        if incident.k0 != self.k0 or incident.medium != _Material():
            raise ValueError("incident wave must match the frequency and vacuum medium")
        if incident.polarization != "helicity":
            raise ValueError("iterative scattering requires helicity polarization")
        local = (
            incident.in_basis(self.basis)
            if isinstance(incident, _PlaneWave)
            else incident.in_basis(self.basis, kind="regular")
        )
        result = self.solve(
            local.array,
            rtol=rtol,
            atol=atol,
            restart=restart,
            max_iterations=max_iterations,
        )
        return ScatteringSolution(
            _Wave(
                result.coefficients,
                basis=self.basis,
                k0=self.k0,
                kind="singular",
                polarization="helicity",
            ),
            result.convergence,
        )

    def record(
        self,
        incident: ArrayLike,
        *,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        restart: int = DEFAULT_RESTART,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> tuple[Solution, IterativeContext]:
        """Record of ``solve``: the solution and an ``IterativeContext``.

        The context keeps the inputs and the solution, not the GMRES
        iterations. The arguments are those of ``solve``.
        """
        values, vector = _columns(incident)
        coefficients, context, reports = self._operator.record(
            values, rtol=rtol, atol=atol, restart=restart, max_iterations=max_iterations
        )
        return (
            Solution(_restore(coefficients, vector), _reports(reports)),
            IterativeContext(context, vector),
        )

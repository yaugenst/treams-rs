"""Type stub of ``treams_rs._native``, the compiled Rust extension.

``_native`` is private: the public modules wrap every entry, and users never
import it. The entries follow the binding files of ``crates/treams-py/src`` in
the order of ``lib.rs``; a comment names each file and the treams-core module it
calls. The test hooks come last.

Naming rules:

* ufuncs keep the upstream treams names (``jv``, ``lpmv``, ``tl_vsw_A``). A
  ufunc that a public namespace exposes directly reports the public name as
  ``__name__``.
* Hidden variants behind a public function are named
  ``<namespace>_<upstream function>_<variant>``: ``sw_translate_sh`` serves
  ``sw.translate``.
* ``<ufunc or family>_scalar`` is a fast path for Python scalars that records
  nothing. ``hankel_scalar`` and ``angular_scalar`` serve a family of ufuncs.
* A record returns ``(value, context)`` and takes the name of the
  ``treams_rs.diff`` function it serves; the twin for cylindrical bases starts
  with ``cylindrical_``. Broadcast records end in ``_record``, and their 0-d
  variants in ``_record_scalar``. Solver objects record through a method named
  ``record``.
* A context is named ``<DiffName>Context`` after the diff function that
  returns it. Most cylindrical twins and every ``_record_scalar`` variant share
  the context of their family. Object records use the context of the matching
  diff function (``InteractionFactor.record`` returns an IlluminateContext) or
  a short context named after the class (``IterativeSphereCluster.record``
  returns an IterativeContext).
* Test hooks end in ``_jet``.

A pullback takes the cotangent, the gradient of a real loss with respect to the
value, and returns the gradients with respect to the inputs in the order of the
arguments of the diff function. Gradients follow dL = Re Σ conj(g)·dx. A context
can be used once; a second pullback raises ValueError.
"""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]

# linalg.rs: treams_core::linalg (LU solve, eigensystems, singular values).

class SolveContext:
    """Created by ``diff.solve``. ``pullback(cotangent) -> (operator, rhs)``."""
    def pullback(self, cotangent: ArrayLike) -> tuple[ComplexArray, ComplexArray]: ...

def solve(
    operator: ComplexArray, rhs: ComplexArray
) -> tuple[ComplexArray, SolveContext]: ...

class EigContext:
    """Created by ``diff.eig``. ``pullback(eigenvalues, eigenvectors) -> operator``, one
    cotangent for each output.
    """
    def pullback(
        self, eigenvalues: ArrayLike, eigenvectors: ArrayLike
    ) -> ComplexArray: ...

def eig(operator: ComplexArray) -> tuple[ComplexArray, ComplexArray, EigContext]: ...

class SvdvalsContext:
    """Created by ``diff.svdvals``. ``pullback(cotangent) -> operator``."""
    def pullback(self, cotangent: ArrayLike) -> ComplexArray: ...

def svdvals(operator: ComplexArray) -> tuple[RealArray, SvdvalsContext]: ...

# coordinates.rs: treams_core::special::coordinates.

class CoordinatesContext:
    """Created by ``diff.coordinates``. ``pullback(cotangent) -> points``."""
    def pullback(self, cotangent: RealArray) -> RealArray: ...

def coordinates_record(
    points: RealArray, kind: str
) -> tuple[RealArray, CoordinatesContext]: ...

class VectorCoordinatesContext:
    """Created by ``diff.vector_coordinates``. ``pullback(cotangent) -> (vectors, points)``."""
    def pullback(self, cotangent: ComplexArray) -> tuple[ComplexArray, RealArray]: ...

def vector_coordinates_record(
    vectors: ComplexArray,
    points: RealArray,
    kind: str,
    shape: tuple[int, ...],
    argument_shapes: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[ComplexArray, VectorCoordinatesContext]: ...

# integrals.rs: treams_core::special (incomplete gamma, Kambe integral).

class IncgammaContext:
    """Created by ``diff.incgamma``. ``pullback(cotangent) -> z``."""
    def pullback(self, cotangent: ComplexArray) -> ComplexArray: ...

def incgamma_record(
    degrees: RealArray,
    arguments: ComplexArray,
    shape: Sequence[int],
    argument_shape: Sequence[int],
) -> tuple[ComplexArray, IncgammaContext]: ...
def incgamma_record_scalar(
    n: float, z: complex
) -> tuple[ComplexArray, IncgammaContext]: ...

class IntkambeContext:
    """Created by ``diff.intkambe``. ``pullback(cotangent) -> (z, eta)``."""
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray]: ...

def intkambe_record(
    orders: NDArray[np.int32],
    z: ComplexArray,
    eta: ComplexArray,
    shape: Sequence[int],
    argument_shapes: tuple[Sequence[int], Sequence[int]],
) -> tuple[ComplexArray, IntkambeContext]: ...
def intkambe_record_scalar(
    n: int, z: complex, eta: complex
) -> tuple[ComplexArray, IntkambeContext]: ...

# special.rs: treams_core::special (Bessel, angular and Wigner functions).

class BesselContext:
    """Created by ``diff.bessel``. ``pullback(cotangent) -> z``."""
    def pullback(self, cotangent: ComplexArray) -> ComplexArray: ...

def bessel_record(
    orders: RealArray,
    arguments: ComplexArray,
    function: str,
    spherical: bool,
    derivative: int,
    shape: tuple[int, ...],
    argument_shape: tuple[int, ...],
) -> tuple[ComplexArray, BesselContext]: ...
def bessel_record_scalar(
    order: float, z: complex, function: str, spherical: bool, derivative: int
) -> tuple[ComplexArray, BesselContext]: ...

class AngularContext:
    """Created by ``diff.angular``. ``pullback(cotangent) -> z``."""
    def pullback(self, cotangent: ComplexArray) -> ComplexArray: ...

def angular_record(
    degrees: RealArray,
    orders: RealArray,
    arguments: ComplexArray,
    function: str,
    shape: tuple[int, ...],
    argument_shape: tuple[int, ...],
) -> tuple[ComplexArray, AngularContext]: ...
def angular_record_scalar(
    degree: float, order: float, z: complex, function: str
) -> tuple[ComplexArray, AngularContext]: ...

class WignerdContext:
    """Created by ``diff.wignerd``. ``pullback(cotangent) -> (phi, theta, psi)``."""
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]: ...

def wignerd_record(
    labels: list[tuple[int, int, int]],
    phi: ComplexArray,
    theta: ComplexArray,
    psi: ComplexArray,
    shape: tuple[int, ...],
    argument_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> tuple[ComplexArray, WignerdContext]: ...
def wignerd_record_scalar(
    labels: tuple[int, int, int], angles: tuple[complex, complex, complex]
) -> tuple[ComplexArray, WignerdContext]: ...

# lattice.rs: treams_core::lattice, sw::lattice_expansion and cw::lattice_expansion.

class LatticeSumContext:
    """Created by ``diff.lattice_sum``. ``pullback(cotangent) -> (k, kpar, a, r, eta)``."""
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[ComplexArray, RealArray, RealArray, RealArray, ComplexArray]: ...

def lattice_sum_record(
    spherical: bool,
    dim: int,
    modes: list[tuple[int, int]],
    k: ComplexArray,
    kpar: RealArray,
    a: RealArray,
    r: RealArray,
    eta: ComplexArray,
    part: int,
    shells: list[int],
    shape: tuple[int, ...],
    argument_shapes: list[tuple[int, ...]],
) -> tuple[ComplexArray, LatticeSumContext]: ...

class LatticeExpansionContext:
    """Created by ``diff.lattice_expansion``.

    ``pullback(cotangent) -> (destination_positions, source_positions, ks, kpar, a)``.
    For cylindrical bases, ``pullback_axial(cotangent)`` appends the gradient of
    kzs, the sorted distinct axial wavenumbers.
    """
    def pullback_axial(
        self, cotangent: ArrayLike
    ) -> tuple[RealArray, RealArray, ComplexArray, RealArray, RealArray, RealArray]: ...
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[RealArray, RealArray, ComplexArray, RealArray, RealArray]: ...

def lattice_expansion(
    destination: list[tuple[int, int, int, int]],
    source: list[tuple[int, int, int, int]],
    destination_positions: list[list[float]],
    source_positions: list[list[float]],
    ks: Sequence[complex],
    helicity: bool,
    kpar: list[float],
    a: list[list[float]],
    eta: complex,
) -> tuple[ComplexArray, LatticeExpansionContext]: ...
def cylindrical_lattice_expansion(
    destination: list[tuple[int, float, int, int]],
    source: list[tuple[int, float, int, int]],
    destination_positions: list[list[float]],
    source_positions: list[list[float]],
    ks: Sequence[complex],
    kpar: list[float],
    a: list[list[float]],
    eta: complex,
) -> tuple[ComplexArray, LatticeExpansionContext]: ...

class LatticeExpansionFromTableContext:
    """Created by ``diff.lattice_expansion_from_table``. ``pullback(cotangent) -> values``,
    the gradient of the lattice-sum table.
    """
    def pullback(self, cotangent: ArrayLike) -> ComplexArray: ...

def lattice_expansion_from_table(
    destination: list[tuple[int, int, int, int]],
    source: list[tuple[int, int, int, int]],
    destination_positions: list[list[float]],
    source_positions: list[list[float]],
    helicity: bool,
    table: ComplexArray,
) -> tuple[ComplexArray, LatticeExpansionFromTableContext]: ...
def lattice_cube(dim: int, n: int, edge: bool) -> NDArray[np.int64]: ...
def diffraction_orders(b: RealArray, radius: float) -> NDArray[np.int64]: ...
def first_brillouin(k: RealArray, b: RealArray, dim: int, n: int) -> RealArray: ...

# channels.rs: treams_core::channels.

class SphericalChannelsContext:
    """Created by ``diff.spherical_channels``.
    ``pullback(cotangent) -> (positions, ks, q, area)``.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[RealArray, ComplexArray, RealArray, float]: ...

def spherical_channels(
    modes: list[tuple[int, int, int, int]],
    positions: list[list[float]],
    ks: Sequence[complex],
    q: list[list[float]],
    polarizations: list[int],
    area: float,
    helicity: bool,
    fixed_q: bool,
) -> tuple[ComplexArray, SphericalChannelsContext]: ...

class CylindricalChannelsContext:
    """Created by ``diff.cylindrical_channels``.
    ``pullback(cotangent) -> (positions, ks, q, period)``.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[RealArray, ComplexArray, RealArray, float]: ...

def cylindrical_channels(
    modes: list[tuple[int, float, int, int]],
    positions: list[list[float]],
    ks: Sequence[complex],
    q: list[list[float]],
    polarizations: list[int],
    period: float,
    helicity: bool,
    fixed_q: bool,
) -> tuple[ComplexArray, CylindricalChannelsContext]: ...

# expansion.rs: treams_core::sw and cw (expansion, to_sw_matrix, periodic_to_cw_matrix).

class ExpansionContext:
    """Created by ``diff.expansion`` (through ``expansion``, ``cylindrical_expansion``
    or ``cw_to_sw``).

    ``pullback(cotangent) -> (destination_positions, source_positions, ks)``. For two
    cylindrical bases, ``pullback_axial(cotangent)`` appends the gradient of kzs,
    the sorted distinct axial wavenumbers.
    """
    def pullback_axial(
        self, cotangent: ArrayLike
    ) -> tuple[RealArray, RealArray, ComplexArray, RealArray]: ...
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[RealArray, RealArray, ComplexArray]: ...

def expansion(
    destination: list[tuple[int, int, int, int]],
    source: list[tuple[int, int, int, int]],
    destination_positions: list[list[float]],
    source_positions: list[list[float]],
    ks: Sequence[complex],
    helicity: bool,
    singular: bool,
) -> tuple[ComplexArray, ExpansionContext]: ...
def cylindrical_expansion(
    destination: list[tuple[int, float, int, int]],
    source: list[tuple[int, float, int, int]],
    destination_positions: list[list[float]],
    source_positions: list[list[float]],
    ks: Sequence[complex],
    singular: bool,
) -> tuple[ComplexArray, ExpansionContext]: ...
def cw_to_sw(
    destination: list[tuple[int, int, int, int]],
    source: list[tuple[int, float, int, int]],
    destination_positions: list[list[float]],
    source_positions: list[list[float]],
    ks: Sequence[complex],
    helicity: bool,
) -> tuple[ComplexArray, ExpansionContext]: ...

class PeriodicToCwContext:
    """Created by ``diff.periodic_to_cw``.
    ``pullback(cotangent) -> (destination_positions, source_positions, ks, kz, period)``,
    with one kz per destination mode.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[RealArray, RealArray, ComplexArray, RealArray, float]: ...

def periodic_to_cw(
    destination: list[tuple[int, float, int, int]],
    source: list[tuple[int, int, int, int]],
    destination_positions: list[list[float]],
    source_positions: list[list[float]],
    ks: Sequence[complex],
    period: float,
    helicity: bool,
) -> tuple[ComplexArray, PeriodicToCwContext]: ...

# fields.rs: treams_core::fields.

class FieldContext:
    """Created by ``diff.field`` (through ``field`` or ``cylindrical_field``).

    ``pullback(cotangent) -> (coefficients, points, positions, ks)``. For a
    cylindrical basis, ``pullback_axial(cotangent)`` appends the gradient of kz,
    one per mode.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[
        ComplexArray,
        RealArray,
        RealArray,
        ComplexArray,
    ]: ...
    def pullback_axial(
        self, cotangent: ArrayLike
    ) -> tuple[ComplexArray, RealArray, RealArray, ComplexArray, RealArray]: ...

def field(
    modes: list[tuple[int, int, int, int]],
    positions: list[list[float]],
    coefficients: ComplexArray,
    points: RealArray,
    ks: Sequence[complex],
    helicity: bool,
    singular: bool,
) -> tuple[ComplexArray, FieldContext]: ...
def cylindrical_field(
    modes: list[tuple[int, float, int, int]],
    positions: list[list[float]],
    coefficients: ComplexArray,
    points: RealArray,
    ks: Sequence[complex],
    helicity: bool,
    singular: bool,
) -> tuple[ComplexArray, FieldContext]: ...

class FieldOperatorContext:
    """Created by ``diff.field_operator`` (through ``field_operator`` or
    ``cylindrical_field_operator``).

    ``pullback(cotangent) -> (points, positions, ks)``. For a cylindrical basis,
    ``pullback_axial(cotangent)`` appends the gradient of kz, one per mode.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[RealArray, RealArray, ComplexArray]: ...
    def pullback_axial(
        self, cotangent: ArrayLike
    ) -> tuple[RealArray, RealArray, ComplexArray, RealArray]: ...

def field_operator(
    modes: list[tuple[int, int, int, int]],
    positions: list[list[float]],
    points: RealArray,
    ks: Sequence[complex],
    helicity: bool,
    singular: bool,
) -> tuple[ComplexArray, FieldOperatorContext]: ...
def cylindrical_field_operator(
    modes: list[tuple[int, float, int, int]],
    positions: list[list[float]],
    points: RealArray,
    ks: Sequence[complex],
    helicity: bool,
    singular: bool,
) -> tuple[ComplexArray, FieldOperatorContext]: ...

# plane.rs: treams_core::pw.

def plane_polarization(
    vector: tuple[complex, ...], pol: int, helicity: bool
) -> list[complex]: ...

class PlaneExpansionContext:
    """Created by ``diff.plane_expansion``. ``pullback(cotangent) -> (positions, vectors)``."""
    def pullback(self, cotangent: ArrayLike) -> tuple[RealArray, ComplexArray]: ...

def plane_expansion(
    modes: list[tuple[int, int, int, int]],
    positions: list[list[float]],
    vectors: list[list[complex]],
    polarizations: list[int],
    helicity: bool,
    fixed_vectors: bool,
) -> tuple[ComplexArray, PlaneExpansionContext]: ...
def cylindrical_plane_expansion(
    modes: list[tuple[int, float, int, int]],
    positions: list[list[float]],
    vectors: list[list[complex]],
    polarizations: list[int],
    helicity: bool,
    fixed_vectors: bool,
) -> tuple[ComplexArray, PlaneExpansionContext]: ...

class PlaneFieldContext:
    """Created by ``diff.plane_field``.
    ``pullback(cotangent) -> (coefficients, points, vectors)``; the coefficient
    gradient is empty for a field operator.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[ComplexArray, RealArray, ComplexArray]: ...

def plane_field(
    vectors: ComplexArray,
    polarizations: list[int],
    points: RealArray,
    coefficients: ComplexArray | None,
    helicity: bool,
    fixed_vectors: bool,
) -> tuple[ComplexArray, PlaneFieldContext]: ...

class PlanePhasesContext:
    """Created by ``diff.plane_phases``. ``pullback(cotangent) -> (points, vectors)``."""
    def pullback(self, cotangent: ArrayLike) -> tuple[RealArray, ComplexArray]: ...

def plane_phases(
    points: RealArray, vectors: ComplexArray
) -> tuple[ComplexArray, PlanePhasesContext]: ...

class PlanePermutationContext:
    """Created by ``diff.plane_permutation``. ``pullback(cotangent) -> vectors``."""
    def pullback(self, cotangent: ArrayLike) -> ComplexArray: ...

def plane_permutation(
    vectors: ComplexArray, polarizations: RealArray, turns: int, helicity: bool
) -> tuple[ComplexArray, PlanePermutationContext]: ...

# rotation.rs: treams_core::rotation.

class RotationContext:
    """Created by ``diff.rotation``. ``pullback(cotangent) -> [phi, theta, psi]``, a list
    of three floats.
    """
    def pullback(self, cotangent: ArrayLike) -> list[float]: ...

def rotation(
    destination: list[tuple[int, int, int, int]],
    source: list[tuple[int, int, int, int]],
    destination_positions: list[list[float]],
    source_positions: list[list[float]],
    angles: Sequence[float],
) -> tuple[ComplexArray, RotationContext]: ...
def cylindrical_rotation(
    destination: list[tuple[int, float, int, int]],
    source: list[tuple[int, float, int, int]],
    destination_positions: list[list[float]],
    source_positions: list[list[float]],
    angles: Sequence[float],
) -> tuple[ComplexArray, RotationContext]: ...

# translation.rs: treams_core::sw and cw (translation coefficients of one mode pair).

class SphericalTranslationContext:
    """Created by ``diff.spherical_translation``.
    ``pullback(cotangent) -> (kr, theta, phi)``.
    """
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]: ...

def spherical_translation_record(
    modes: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    arguments: tuple[ComplexArray, ...],
    helicity: bool,
    singular: bool,
    shape: tuple[int, ...],
    argument_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> tuple[ComplexArray, SphericalTranslationContext]: ...

class CylindricalTranslationContext:
    """Created by ``diff.cylindrical_translation``.
    ``pullback(cotangent) -> (krr, phi, z, kz)``.
    """
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[
        ComplexArray,
        ComplexArray,
        ComplexArray,
        ComplexArray,
    ]: ...

def cylindrical_translation_record(
    orders: list[int],
    arguments: tuple[ComplexArray, ...],
    singular: bool,
    shape: tuple[int, ...],
    argument_shapes: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ],
) -> tuple[ComplexArray, CylindricalTranslationContext]: ...

# vectorwaves.rs: treams_core::vectorwaves.

class VectorWaveContext:
    """Created by ``diff.vector_wave`` and ``diff.sph_harm``.
    ``pullback(cotangent)`` returns one gradient per argument, in order.
    """
    def pullback(self, cotangent: ComplexArray) -> tuple[ComplexArray, ...]: ...

def vector_wave_record(
    function: str,
    labels: list[tuple[int, int, int]],
    arguments: list[ComplexArray],
    shape: tuple[int, ...],
    argument_shapes: list[tuple[int, ...]],
) -> tuple[ComplexArray, VectorWaveContext]: ...

# cluster.rs: treams_core::cluster (dense clusters, interaction solves and factors).

class SphereClusterContext:
    """Created by ``diff.sphere_cluster``.
    ``pullback(cotangent) -> (k0, radii, epsilon, positions)``.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[float, RealArray, ComplexArray, RealArray]: ...

def sphere_cluster(
    lmax: int, k0: float, radii: RealArray, epsilon: ComplexArray, positions: RealArray
) -> tuple[ComplexArray, SphereClusterContext]: ...

class InteractionContext:
    """Created by ``diff.interaction``. ``pullback(cotangent) -> (local, coupling)``."""
    def pullback(self, cotangent: ArrayLike) -> tuple[ComplexArray, ComplexArray]: ...

def interaction(
    local: ComplexArray, coupling: ComplexArray
) -> tuple[ComplexArray, InteractionContext]: ...

class ParticleClusterContext:
    """Created by ``diff.particle_cluster``.
    ``pullback(cotangent) -> (local, positions, ks)``, where ``local`` is a list
    with one gradient per particle.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[list[ComplexArray], RealArray, ComplexArray]: ...

def particle_cluster(
    local: list[ComplexArray],
    modes: list[tuple[int, int, int, int]],
    positions: list[list[float]],
    ks: Sequence[complex],
    helicity: bool,
) -> tuple[ComplexArray, ParticleClusterContext]: ...
def cylindrical_particle_cluster(
    local: list[ComplexArray],
    modes: list[tuple[int, float, int, int]],
    positions: list[list[float]],
    ks: Sequence[complex],
    helicity: bool,
) -> tuple[ComplexArray, ParticleClusterContext]: ...

class InteractionFactor:
    """LU factors of I - T C, for solving only the requested incident columns.

    ``diff.factor_interaction`` builds it from a dense T (the constructor),
    ``diff.factor_interaction_blocks`` from separate particle blocks
    (``from_blocks``), and ``diff.sphere_cluster_factor`` from homogeneous spheres.
    ``solve(incident)`` returns the scattered coefficients;
    ``record(incident)`` also returns an IlluminateContext.
    """
    def __init__(self, local: ComplexArray, coupling: ComplexArray) -> None: ...
    @staticmethod
    def from_blocks(
        local: list[ComplexArray], coupling: ComplexArray
    ) -> InteractionFactor: ...
    @property
    def dimension(self) -> int: ...
    def solve(self, incident: ComplexArray) -> ComplexArray: ...
    def record(
        self, incident: ComplexArray
    ) -> tuple[ComplexArray, IlluminateContext]: ...

class IlluminateContext:
    """Created by ``InteractionFactor.record``, which ``diff.illuminate`` calls.

    The factor decides the pullback method when it is built, and the other
    method raises ValueError:

    * dense T (``factor_interaction``, ``illuminate``):
      ``pullback(cotangent) -> (local, coupling, incident)``;
    * blocks (``factor_interaction_blocks``, ``sphere_cluster_factor``):
      ``pullback_blocks(cotangent) -> (local, coupling, incident)``, where
      ``local`` is a list with one gradient per block.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]: ...
    def pullback_blocks(
        self, cotangent: ArrayLike
    ) -> tuple[list[ComplexArray], ComplexArray, ComplexArray]: ...

def sphere_cluster_factor(
    lmax: int, k0: float, radii: RealArray, epsilon: ComplexArray, positions: RealArray
) -> InteractionFactor: ...

# coeffs.rs: treams_core::coeffs.

class MieContext:
    """Created by ``diff.mie``. ``pullback(cotangent) -> (x, epsilon, mu, kappa)``."""
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[RealArray, ComplexArray, ComplexArray, ComplexArray]: ...

def mie(
    l: int,
    sizes: RealArray,
    epsilon: ComplexArray,
    mu: ComplexArray,
    kappa: ComplexArray,
) -> tuple[ComplexArray, MieContext]: ...

class MieCylContext:
    """Created by ``diff.mie_cyl``.
    ``pullback(cotangent) -> (kz, k0, radii, epsilon, mu, kappa)``.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[float, float, RealArray, ComplexArray, ComplexArray, ComplexArray]: ...

def mie_cyl(
    kz: float,
    m: int,
    k0: float,
    radii: RealArray,
    epsilon: ComplexArray,
    mu: ComplexArray,
    kappa: ComplexArray,
) -> tuple[ComplexArray, MieCylContext]: ...

# ebcm.rs: treams_core::ebcm.

class EbcmQmatContext:
    """Created by ``diff.ebcm_qmat``. ``pullback(cotangent) -> (radii, slopes, ks, zs)``."""
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[RealArray, RealArray, ComplexArray, ComplexArray]: ...

def ebcm_qmat(
    samples: RealArray,
    destination: list[tuple[int, int, int]],
    source: list[tuple[int, int, int]],
    ks: ComplexArray,
    zs: Sequence[complex],
    singular: bool,
    radial_area_factor: bool,
) -> tuple[ComplexArray, EbcmQmatContext]: ...

# iterative.rs: treams_core::cluster::IterativeSphereCluster.

# (iterations, residual norm, right-hand-side norm) of one GMRES solve.
type Convergence = tuple[int, float, float]

class IterativeSphereCluster:
    """Matrix-free operator of a sphere cluster, solved with GMRES.

    ``iterative.SphereCluster`` wraps it. ``solve(incident)`` returns the
    scattered coefficients and the convergence of each illumination;
    ``record(incident)`` also returns an IterativeContext before the
    convergence.
    """
    def __init__(
        self,
        lmax: int,
        k0: float,
        radii: RealArray,
        epsilon: ComplexArray,
        positions: RealArray,
    ) -> None: ...
    @property
    def dimension(self) -> int: ...
    def solve(
        self,
        incident: ComplexArray,
        *,
        rtol: float = 1e-10,
        atol: float = 0.0,
        restart: int = 30,
        max_iterations: int = 300,
    ) -> tuple[ComplexArray, list[Convergence]]: ...
    def record(
        self,
        incident: ComplexArray,
        *,
        rtol: float = 1e-10,
        atol: float = 0.0,
        restart: int = 30,
        max_iterations: int = 300,
    ) -> tuple[ComplexArray, IterativeContext, list[Convergence]]: ...

class IterativeContext:
    """Created by ``IterativeSphereCluster.record``, which
    ``iterative.SphereCluster.record`` calls.

    ``pullback(cotangent) -> (k0, radii, epsilon, positions, incident, convergence)``.
    The last item is no gradient: it holds the GMRES convergence of each
    solve of the adjoint (conjugate-transposed) system.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[
        float, RealArray, ComplexArray, RealArray, ComplexArray, list[Convergence]
    ]: ...

# tmatrix.rs: treams_core::tmatrix.

class SphereContext:
    """Created by ``diff.sphere``. ``pullback(cotangent) -> (k0, radii, epsilon, mu, kappa)``."""
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[float, RealArray, ComplexArray, ComplexArray, ComplexArray]: ...

def sphere(
    lmax: int,
    k0: float,
    radii: RealArray,
    epsilon: ComplexArray,
    mu: ComplexArray,
    kappa: ComplexArray,
) -> tuple[ComplexArray, SphereContext]: ...

class CylinderContext:
    """Created by ``diff.cylinder``.
    ``pullback(cotangent) -> (kzs, k0, radii, epsilon, mu, kappa)``.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[
        RealArray,
        float,
        RealArray,
        ComplexArray,
        ComplexArray,
        ComplexArray,
    ]: ...

def cylinder(
    kzs: RealArray,
    mmax: int,
    k0: float,
    radii: RealArray,
    epsilon: ComplexArray,
    mu: ComplexArray,
    kappa: ComplexArray,
) -> tuple[ComplexArray, CylinderContext]: ...

class TMatrixMetricContext:
    """Created by ``diff.tmatrix_metric``. ``pullback(cotangent) -> (operator, ks)`` for
    a float cotangent.
    """
    def pullback(self, cotangent: float) -> tuple[ComplexArray, RealArray]: ...

def tmatrix_metric(
    operator: ComplexArray, polarizations: list[int], ks: Sequence[float], kind: str
) -> tuple[float, TMatrixMetricContext]: ...

# smatrix.rs: treams_core::smatrix.

class SMatrixAddContext:
    """Created by ``diff.smatrix_add``. ``pullback(cotangent) -> (lower, upper)``."""
    def pullback(self, cotangent: ArrayLike) -> tuple[ComplexArray, ComplexArray]: ...

def smatrix_add(
    lower: ComplexArray, upper: ComplexArray
) -> tuple[ComplexArray, SMatrixAddContext]: ...

class SMatrixIlluminateContext:
    """Created by ``diff.smatrix_illuminate``.
    ``pullback(cotangent) -> (lower, upper, up, down)``.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]: ...

def smatrix_illuminate(
    lower: ComplexArray, upper: ComplexArray, up: ComplexArray, down: ComplexArray
) -> tuple[ComplexArray, SMatrixIlluminateContext]: ...
def smatrix_illuminate_value(
    lower: ComplexArray, upper: ComplexArray, up: ComplexArray, down: ComplexArray
) -> ComplexArray: ...

class SMatrixPeriodicContext:
    """Created by ``diff.smatrix_periodic``. ``pullback(cotangent) -> smats``."""
    def pullback(self, cotangent: ArrayLike) -> ComplexArray: ...

def smatrix_periodic(
    smats: ComplexArray,
) -> tuple[ComplexArray, SMatrixPeriodicContext]: ...

class BandsContext:
    """Created by ``diff.bands``. ``pullback(wavenumbers, eigenvectors) -> (smats, period)``,
    one cotangent for each output.
    """
    def pullback(
        self, wavenumbers: ArrayLike, eigenvectors: ArrayLike
    ) -> tuple[ComplexArray, float]: ...

def bands(
    smats: ComplexArray, period: float
) -> tuple[ComplexArray, ComplexArray, BandsContext]: ...

class SMatrixFromArrayContext:
    """Created by ``diff.smatrix_from_array``.
    ``pullback(cotangent) -> (response, channels)``.
    """
    def pullback(self, cotangent: ArrayLike) -> tuple[ComplexArray, ComplexArray]: ...

def smatrix_from_array(
    response: ComplexArray, channels: ComplexArray
) -> tuple[ComplexArray, SMatrixFromArrayContext]: ...

class SMatrixTrContext:
    """Created by ``diff.smatrix_tr``.
    ``pullback(cotangent) -> (matrices, incident, ks, zs, q)``.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray, RealArray]: ...

def smatrix_tr(
    matrices: ComplexArray,
    incident: ComplexArray,
    ks: Sequence[Sequence[complex]],
    zs: Sequence[complex],
    q: Sequence[Sequence[float]],
    modes: Sequence[tuple[int, int]],
    axis: int,
    helicity: bool,
    direction: int,
    fixed_q: bool,
) -> tuple[RealArray, SMatrixTrContext]: ...
def smatrix_tr_value(
    matrices: ComplexArray,
    incident: ComplexArray,
    ks: Sequence[Sequence[complex]],
    zs: Sequence[complex],
    q: Sequence[Sequence[float]],
    modes: Sequence[tuple[int, int]],
    axis: int,
    helicity: bool,
    direction: int,
) -> RealArray: ...

class FresnelContext:
    """Created by ``diff.fresnel``. ``pullback(cotangent) -> (ks, kzs, zs)``."""
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]: ...

def fresnel(
    ks: Sequence[Sequence[complex]],
    kzs: Sequence[Sequence[complex]],
    zs: Sequence[complex],
) -> tuple[ComplexArray, FresnelContext]: ...

class InterfaceCoefficientsContext:
    """Created by ``diff.interface_coefficients``. ``pullback(cotangent) -> (ks, zs, q)``."""
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[ComplexArray, ComplexArray, RealArray]: ...

def interface_coefficients(
    ks: Sequence[Sequence[complex]],
    zs: Sequence[complex],
    q: Sequence[float],
    axis: int,
    fixed_q: bool,
) -> tuple[ComplexArray, InterfaceCoefficientsContext]: ...

class PropagationMatrixContext:
    """Created by ``diff.propagation_matrix``.
    ``pullback(cotangent) -> (vectors, distance)``.
    """
    def pullback(self, cotangent: ArrayLike) -> tuple[ComplexArray, RealArray]: ...

def propagation_matrix(
    vectors: list[list[complex]], distance: Sequence[float]
) -> tuple[ComplexArray, PropagationMatrixContext]: ...

class LayerStackContext:
    """Created by ``diff.layer_stack``. ``pullback(cotangent) -> (ks, zs, q, thickness)``."""
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[ComplexArray, ComplexArray, RealArray, RealArray]: ...

def layer_stack(
    ks: list[list[complex]],
    zs: list[complex],
    q: list[list[float]],
    thickness: list[float],
    axis: int,
    fixed_q: bool,
) -> tuple[ComplexArray, LayerStackContext]: ...

class ChiralityDensityContext:
    """Created by ``diff.chirality_density``. ``pullback(cotangent) -> (ks, normal, z)``."""
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[ComplexArray, ComplexArray, RealArray]: ...

def chirality_density(
    ks: ComplexArray,
    normal: ComplexArray,
    interval: Sequence[float],
) -> tuple[ComplexArray, ChiralityDensityContext]: ...

class OrientedChiralityContext:
    """Created by ``diff.oriented_chirality``.
    ``pullback(cotangent) -> (transverse, normal, z)``.
    """
    def pullback(
        self, cotangent: ArrayLike
    ) -> tuple[RealArray, ComplexArray, RealArray]: ...

def oriented_chirality(
    transverse: RealArray,
    normal: ComplexArray,
    polarizations: list[int],
    axis: int,
    interval: Sequence[float],
) -> tuple[ComplexArray, OrientedChiralityContext]: ...

# ufunc/: the NumPy ufuncs (registry.rs) and their Python-scalar fast paths
# (fast_paths.rs).

# Bessel functions (treams_core::special).

jv: np.ufunc

yv: np.ufunc

hankel1: np.ufunc

hankel2: np.ufunc

jv_d: np.ufunc

yv_d: np.ufunc

hankel1_d: np.ufunc

hankel2_d: np.ufunc

spherical_jn: np.ufunc

spherical_yn: np.ufunc

spherical_hankel1: np.ufunc

spherical_hankel2: np.ufunc

spherical_jn_d: np.ufunc

spherical_yn_d: np.ufunc

spherical_hankel1_d: np.ufunc

spherical_hankel2_d: np.ufunc

def hankel_scalar(order: float, z: complex, first: bool) -> complex: ...

# Angular functions (treams_core::special).

lpmv: np.ufunc

pi_fun: np.ufunc

tau_fun: np.ufunc

def angular_scalar(
    degree: float, order: float, z: complex, function: str
) -> complex: ...
def lpmv_real_scalar(degree: float, order: float, x: float) -> float: ...

# Wigner symbols (treams_core::special).

wignersmalld: np.ufunc

wignerd: np.ufunc

wigner3j: np.ufunc

def wigner3j_scalar(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float: ...

# Incomplete gamma function and Kambe integral (treams_core::special).

incgamma: np.ufunc

intkambe: np.ufunc

def incgamma_scalar(n: float, z: complex) -> complex: ...
def intkambe_scalar(n: int, z: complex, eta: complex) -> complex: ...

# Coordinate transforms (treams_core::special::coordinates).

def car2cyl(points: ArrayLike, *args: object, **kwargs: object) -> RealArray: ...
def car2sph(points: ArrayLike, *args: object, **kwargs: object) -> RealArray: ...
def cyl2car(points: ArrayLike, *args: object, **kwargs: object) -> RealArray: ...
def cyl2sph(points: ArrayLike, *args: object, **kwargs: object) -> RealArray: ...
def sph2car(points: ArrayLike, *args: object, **kwargs: object) -> RealArray: ...
def sph2cyl(points: ArrayLike, *args: object, **kwargs: object) -> RealArray: ...
def car2pol(points: ArrayLike, *args: object, **kwargs: object) -> RealArray: ...
def pol2car(points: ArrayLike, *args: object, **kwargs: object) -> RealArray: ...
def vcar2cyl(
    vector: ArrayLike, points: ArrayLike, *args: object, **kwargs: object
) -> RealArray | ComplexArray: ...
def vcar2sph(
    vector: ArrayLike, points: ArrayLike, *args: object, **kwargs: object
) -> RealArray | ComplexArray: ...
def vcyl2car(
    vector: ArrayLike, points: ArrayLike, *args: object, **kwargs: object
) -> RealArray | ComplexArray: ...
def vcyl2sph(
    vector: ArrayLike, points: ArrayLike, *args: object, **kwargs: object
) -> RealArray | ComplexArray: ...
def vsph2car(
    vector: ArrayLike, points: ArrayLike, *args: object, **kwargs: object
) -> RealArray | ComplexArray: ...
def vsph2cyl(
    vector: ArrayLike, points: ArrayLike, *args: object, **kwargs: object
) -> RealArray | ComplexArray: ...
def vcar2pol(
    vector: ArrayLike, points: ArrayLike, *args: object, **kwargs: object
) -> RealArray | ComplexArray: ...
def vpol2car(
    vector: ArrayLike, points: ArrayLike, *args: object, **kwargs: object
) -> RealArray | ComplexArray: ...

# Vector waves (treams_core::vectorwaves).

sph_harm: np.ufunc

vsh_X: np.ufunc

vsh_Y: np.ufunc

vsh_Z: np.ufunc

vsw_M: np.ufunc

vsw_N: np.ufunc

vsw_A: np.ufunc

vsw_rM: np.ufunc

vsw_rN: np.ufunc

vsw_rA: np.ufunc

vcw_M: np.ufunc

vcw_N: np.ufunc

vcw_A: np.ufunc

vcw_rM: np.ufunc

vcw_rN: np.ufunc

vcw_rA: np.ufunc

def vpw_M(
    kx: ArrayLike,
    ky: ArrayLike,
    kz: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *args: object,
    **kwargs: object,
) -> ComplexArray: ...
def vpw_N(
    kx: ArrayLike,
    ky: ArrayLike,
    kz: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *args: object,
    **kwargs: object,
) -> ComplexArray: ...
def vpw_A(
    kx: ArrayLike,
    ky: ArrayLike,
    kz: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    pol: ArrayLike,
    *args: object,
    **kwargs: object,
) -> ComplexArray: ...

# Translation coefficients of one mode pair (treams_core::sw and cw).

tl_vsw_A: np.ufunc

tl_vsw_B: np.ufunc

tl_vsw_rA: np.ufunc

tl_vsw_rB: np.ufunc

tl_vcw: np.ufunc

tl_vcw_r: np.ufunc

def tl_vcw_scalar(
    kz: float,
    mu: int,
    qz: float,
    m: int,
    kr: complex,
    phi: float,
    z: float,
    singular: bool,
) -> complex: ...

# Rotations and translations behind sw, cw and pw (treams_core::rotation, sw, cw, pw).

sw_rotate: np.ufunc

cw_rotate: np.ufunc

def cw_rotate_scalar(
    kz: float, mu: int, p: int, qz: float, m: int, q: int, phi: float
) -> complex: ...

sw_translate_sh: np.ufunc

sw_translate_rh: np.ufunc

sw_translate_sp: np.ufunc

sw_translate_rp: np.ufunc

cw_translate_s: np.ufunc

cw_translate_r: np.ufunc

def cw_translate_scalar(
    kz: float,
    mu: int,
    p: int,
    qz: float,
    m: int,
    q: int,
    kr: complex,
    phi: float,
    z: float,
    singular: bool,
) -> complex: ...
def pw_translate(
    kx: ArrayLike,
    ky: ArrayLike,
    kz: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *args: object,
    **kwargs: object,
) -> complex | ComplexArray: ...

# Changes of wave family behind sw, cw and pw (treams_core::sw, cw, pw, channels).

pw_to_sw_h: np.ufunc

pw_to_sw_p: np.ufunc

pw_to_cw: np.ufunc

cw_to_sw_h: np.ufunc

cw_to_sw_p: np.ufunc

pw_permute_xyz_h: np.ufunc

pw_permute_xyz_p: np.ufunc

pw_permute_xyz_inverse_h: np.ufunc

pw_permute_xyz_inverse_p: np.ufunc

def pw_permute_xyz_scalar(
    kx: complex, ky: complex, kz: complex, p: int, q: int, helicity: bool, inverse: bool
) -> complex: ...

sw_periodic_to_pw_h: np.ufunc

sw_periodic_to_pw_p: np.ufunc

cw_periodic_to_pw: np.ufunc

sw_periodic_to_cw_h: np.ufunc

sw_periodic_to_cw_p: np.ufunc

# Lattice cells, media and lattice sums (treams_core::lattice, coeffs).

def cell_volume(
    cell: ArrayLike, *args: object, **kwargs: object
) -> float | np.float64 | np.int64 | RealArray | NDArray[np.int64]: ...
def cell_reciprocal(cell: ArrayLike, *args: object, **kwargs: object) -> RealArray: ...

refractive_indices: np.ufunc

wave_vector_z: np.ufunc

first_brillouin_1d: np.ufunc

lsumsw1d: np.ufunc

lsumsw1d_shift: np.ufunc

lsumsw2d: np.ufunc

lsumsw2d_shift: np.ufunc

lsumsw3d: np.ufunc

lsumcw1d: np.ufunc

lsumcw1d_shift: np.ufunc

lsumcw2d: np.ufunc

realsumsw1d: np.ufunc

realsumsw1d_shift: np.ufunc

realsumsw2d: np.ufunc

realsumsw2d_shift: np.ufunc

realsumsw3d: np.ufunc

realsumcw1d: np.ufunc

realsumcw1d_shift: np.ufunc

realsumcw2d: np.ufunc

recsumsw1d: np.ufunc

recsumsw1d_shift: np.ufunc

recsumsw2d: np.ufunc

recsumsw2d_shift: np.ufunc

recsumsw3d: np.ufunc

recsumcw1d: np.ufunc

recsumcw1d_shift: np.ufunc

recsumcw2d: np.ufunc

dsumsw1d: np.ufunc

dsumsw1d_shift: np.ufunc

dsumsw2d: np.ufunc

dsumsw2d_shift: np.ufunc

dsumsw3d: np.ufunc

dsumcw1d: np.ufunc

dsumcw1d_shift: np.ufunc

dsumcw2d: np.ufunc

def dsumcw1d_scalar(
    m: int, k: complex, q: float, a: float, r: float, shell: int
) -> complex: ...
def dsumcw1d_shift_scalar(
    m: int, k: complex, q: float, a: float, r: RealArray, shell: int
) -> complex: ...
def dsumcw2d_scalar(
    m: int, k: complex, q: RealArray, a: RealArray, r: RealArray, shell: int
) -> complex: ...

# testing.rs: hooks for tests and scripts; treams_rs itself calls none of them.

def build_profile() -> str: ...

# Runs `work()` while this thread flushes subnormals to zero.
def _run_flushing_for_tests[T](work: Callable[[], T]) -> T: ...

type Vector = list[complex]

def cartesian_translation_jet(
    destination: tuple[int, int, int],
    source: tuple[int, int, int],
    k: complex,
    position: tuple[float, float, float],
    helicity: bool,
    singular: bool,
) -> tuple[complex, list[complex], complex]: ...
def spherical_wave_jet(
    mode: tuple[int, int, int],
    k: complex,
    position: tuple[float, float, float],
    helicity: bool,
    singular: bool,
) -> tuple[Vector, list[Vector], Vector]: ...
def cylindrical_cartesian_translation_jet(
    destination: tuple[float, int, int],
    source: tuple[float, int, int],
    k: complex,
    position: tuple[float, float, float],
    singular: bool,
) -> tuple[complex, list[complex], complex, complex]: ...
def cylindrical_radial_jet(
    m: int, z: complex, singular: bool
) -> tuple[complex, complex, complex]: ...

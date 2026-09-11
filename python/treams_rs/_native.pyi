import numpy as np
from numpy.typing import NDArray

type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]

class RotationContext:
    def pullback(self, cotangent: ComplexArray) -> list[float]: ...

class FieldOperatorContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, RealArray, ComplexArray]: ...

def field_operator(
    modes: list[tuple[int, int, int, int]],
    origins: list[list[float]],
    points: RealArray,
    ks: tuple[complex, complex],
    helicity: bool,
    outgoing: bool,
) -> tuple[ComplexArray, FieldOperatorContext]: ...
def cylindrical_field_operator(
    modes: list[tuple[int, float, int, int]],
    origins: list[list[float]],
    points: RealArray,
    ks: tuple[complex, complex],
    helicity: bool,
    outgoing: bool,
) -> tuple[ComplexArray, FieldOperatorContext]: ...
def rotation(
    to: list[tuple[int, int, int, int]],
    source: list[tuple[int, int, int, int]],
    to_positions: list[list[float]],
    source_positions: list[list[float]],
    angles: tuple[float, float, float],
) -> tuple[ComplexArray, RotationContext]: ...
def cyl_rotation(
    to: list[tuple[int, float, int, int]],
    source: list[tuple[int, float, int, int]],
    to_positions: list[list[float]],
    source_positions: list[list[float]],
    angles: tuple[float, float, float],
) -> tuple[ComplexArray, RotationContext]: ...

class ArrayContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray]: ...

def smatrix_from_array(
    response: ComplexArray, channels: ComplexArray
) -> tuple[ComplexArray, ArrayContext]: ...

class ChannelsContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, ComplexArray, RealArray, float]: ...

def spherical_channels(
    modes: list[tuple[int, int, int, int]],
    positions: list[list[float]],
    ks: list[complex],
    q: list[list[float]],
    polarizations: list[int],
    area: float,
    helicity: bool,
    fixed_q: bool,
) -> tuple[ComplexArray, ChannelsContext]: ...

class SMatrixContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray]: ...

def smatrix_add(
    lower: ComplexArray, upper: ComplexArray
) -> tuple[ComplexArray, SMatrixContext]: ...

class FresnelContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]: ...

def fresnel(
    ks: list[list[complex]], kz: list[list[complex]], z: list[complex]
) -> tuple[ComplexArray, FresnelContext]: ...

class PropagationContext:
    def pullback(self, cotangent: ComplexArray) -> tuple[ComplexArray, RealArray]: ...

def propagation(
    vectors: list[list[complex]], distance: list[float]
) -> tuple[ComplexArray, PropagationContext]: ...

class MieContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, ComplexArray, ComplexArray, ComplexArray]: ...

def mie(
    l: int,
    sizes: RealArray,
    epsilon: ComplexArray,
    mu: ComplexArray,
    kappa: ComplexArray,
) -> tuple[ComplexArray, MieContext]: ...
def radial(l: int, z: complex, outgoing: bool) -> tuple[complex, complex, complex]: ...

class SphereContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[float, RealArray, ComplexArray, ComplexArray, ComplexArray]: ...

class ClusterContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, RealArray, ComplexArray, float]: ...

class InteractionContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray]: ...

class SolveContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray]: ...

class EigenContext:
    def pullback(
        self, eigenvalues: ComplexArray, eigenvectors: ComplexArray
    ) -> ComplexArray: ...

def linear_solve(
    operator: ComplexArray, rhs: ComplexArray
) -> tuple[ComplexArray, SolveContext]: ...
def eig(operator: ComplexArray) -> tuple[ComplexArray, ComplexArray, EigenContext]: ...

class IlluminationContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]: ...

class SMatrixPeriodicContext:
    def pullback(self, cotangent: ComplexArray) -> ComplexArray: ...

def smatrix_illuminate(
    lower: ComplexArray, upper: ComplexArray, up: ComplexArray, down: ComplexArray
) -> tuple[ComplexArray, IlluminationContext]: ...
def smatrix_periodic(
    smats: ComplexArray,
) -> tuple[ComplexArray, SMatrixPeriodicContext]: ...

class BandContext:
    def pullback(
        self, wavenumbers: ComplexArray, eigenvectors: ComplexArray
    ) -> tuple[ComplexArray, float]: ...

def bands(
    smats: ComplexArray, period: float
) -> tuple[ComplexArray, ComplexArray, BandContext]: ...
def sphere(
    lmax: int,
    k0: float,
    radii: RealArray,
    epsilon: ComplexArray,
    mu: ComplexArray,
    kappa: ComplexArray,
) -> tuple[ComplexArray, SphereContext]: ...
def cluster(
    lmax: int, k0: float, radii: RealArray, epsilon: ComplexArray, positions: RealArray
) -> tuple[ComplexArray, ClusterContext]: ...
def interact(
    local: ComplexArray, coupling: ComplexArray
) -> tuple[ComplexArray, InteractionContext]: ...
def translation(
    to: tuple[int, int, int],
    source: tuple[int, int, int],
    k: complex,
    position: tuple[float, float, float],
    helicity: bool,
    outgoing: bool,
) -> tuple[complex, tuple[complex, complex, complex], complex]: ...

class ExpansionContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, RealArray, ComplexArray]: ...

def expansion(
    to: list[tuple[int, int, int, int]],
    source: list[tuple[int, int, int, int]],
    to_positions: list[list[float]],
    source_positions: list[list[float]],
    ks: tuple[complex, complex],
    helicity: bool,
    outgoing: bool,
) -> tuple[ComplexArray, ExpansionContext]: ...
def cw_to_sw(
    to: list[tuple[int, int, int, int]],
    source: list[tuple[int, float, int, int]],
    to_positions: list[list[float]],
    source_positions: list[list[float]],
    ks: tuple[complex, complex],
    helicity: bool,
) -> tuple[ComplexArray, ExpansionContext]: ...
def build_profile() -> str: ...

class CylinderContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[float, float, RealArray, ComplexArray, ComplexArray, ComplexArray]: ...

def mie_cyl(
    kz: float,
    m: int,
    k0: float,
    radii: RealArray,
    epsilon: ComplexArray,
    mu: ComplexArray,
    kappa: ComplexArray,
) -> tuple[ComplexArray, CylinderContext]: ...
def cylindrical(
    m: int, z: complex, outgoing: bool
) -> tuple[complex, complex, complex]: ...

type Vector = tuple[complex, complex, complex]

def spherical_wave(
    mode: tuple[int, int, int],
    k: complex,
    position: tuple[float, float, float],
    helicity: bool,
    outgoing: bool,
) -> tuple[Vector, tuple[Vector, Vector, Vector], Vector]: ...

class FieldContext:
    def pullback(
        self, cotangent: NDArray[np.complex128]
    ) -> tuple[
        NDArray[np.complex128],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.complex128],
    ]: ...

def field(
    modes: list[tuple[int, int, int, int]],
    origins: list[list[float]],
    coefficients: NDArray[np.complex128],
    points: NDArray[np.float64],
    ks: tuple[complex, complex],
    helicity: bool,
    outgoing: bool,
) -> tuple[NDArray[np.complex128], FieldContext]: ...
def cylindrical_field(
    modes: list[tuple[int, float, int, int]],
    origins: list[list[float]],
    coefficients: ComplexArray,
    points: RealArray,
    ks: tuple[complex, complex],
    helicity: bool,
    outgoing: bool,
) -> tuple[ComplexArray, FieldContext]: ...

class CylinderMatrixContext:
    def pullback(
        self, cotangent: NDArray[np.complex128]
    ) -> tuple[
        NDArray[np.float64],
        float,
        NDArray[np.float64],
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
    ]: ...

def cylinder(
    kzs: NDArray[np.float64],
    mmax: int,
    k0: float,
    radii: NDArray[np.float64],
    epsilon: NDArray[np.complex128],
    mu: NDArray[np.complex128],
    kappa: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], CylinderMatrixContext]: ...
def cyl_expansion(
    to: list[tuple[int, float, int, int]],
    source: list[tuple[int, float, int, int]],
    to_positions: list[list[float]],
    source_positions: list[list[float]],
    ks: tuple[complex, complex],
    outgoing: bool,
) -> tuple[NDArray[np.complex128], ExpansionContext]: ...
def cyl_translation(
    to: tuple[float, int, int],
    source: tuple[float, int, int],
    k: complex,
    position: tuple[float, float, float],
    outgoing: bool,
) -> tuple[complex, tuple[complex, complex, complex], complex, complex]: ...
def plane_to_spherical(
    modes: list[tuple[int, int, int, int]],
    positions: list[list[float]],
    vector: tuple[complex, ...],
    pol: int,
    helicity: bool,
) -> NDArray[np.complex128]: ...
def plane_polarization(
    vector: tuple[complex, ...], pol: int, helicity: bool
) -> tuple[complex, complex, complex]: ...
def plane_to_cylindrical(
    modes: list[tuple[int, float, int, int]],
    positions: list[list[float]],
    vector: tuple[complex, ...],
    pol: int,
) -> NDArray[np.complex128]: ...
def incgamma(n: float, z: complex) -> complex: ...
def intkambe(n: int, z: complex, eta: complex) -> complex: ...
def lattice_sum(
    spherical: bool,
    modes: list[tuple[int, int]],
    k: complex,
    bloch: list[float],
    vectors: list[list[float]],
    shift: tuple[float, float, float],
    eta: complex,
) -> list[complex]: ...
def periodic_expansion(
    to: list[tuple[int, int, int, int]],
    source: list[tuple[int, int, int, int]],
    to_positions: list[list[float]],
    source_positions: list[list[float]],
    ks: tuple[complex, complex],
    helicity: bool,
    bloch: list[float],
    vectors: list[list[float]],
    eta: complex,
) -> tuple[ComplexArray, PeriodicContext]: ...
def periodic_cyl_expansion(
    to: list[tuple[int, float, int, int]],
    source: list[tuple[int, float, int, int]],
    to_positions: list[list[float]],
    source_positions: list[list[float]],
    ks: tuple[complex, complex],
    bloch: list[float],
    vectors: list[list[float]],
    eta: complex,
) -> tuple[ComplexArray, PeriodicContext]: ...
def lattice_derivatives(
    spherical: bool,
    mode: tuple[int, int],
    k: complex,
    bloch: list[float],
    vectors: list[list[float]],
    shift: tuple[float, float, float],
    eta: complex,
) -> tuple[
    complex,
    complex,
    tuple[complex, complex, complex],
    tuple[complex, complex, complex],
    tuple[
        tuple[complex, complex, complex],
        tuple[complex, complex, complex],
        tuple[complex, complex, complex],
    ],
]: ...

class PeriodicContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, RealArray, ComplexArray, RealArray, RealArray]: ...

class PlaneFieldContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, RealArray, ComplexArray]: ...

def plane_field(
    vectors: ComplexArray,
    polarizations: list[int],
    points: RealArray,
    coefficients: ComplexArray | None,
    helicity: bool,
    fixed_vectors: bool,
) -> tuple[ComplexArray, PlaneFieldContext]: ...

class PlaneExpansionContext:
    def pullback(self, cotangent: ComplexArray) -> tuple[RealArray, ComplexArray]: ...

def plane_expansion(
    modes: list[tuple[int, int, int, int]],
    origins: list[list[float]],
    vectors: list[list[complex]],
    polarizations: list[int],
    helicity: bool,
    fixed_vectors: bool,
) -> tuple[ComplexArray, PlaneExpansionContext]: ...
def cylindrical_plane_expansion(
    modes: list[tuple[int, float, int, int]],
    origins: list[list[float]],
    vectors: list[list[complex]],
    polarizations: list[int],
    helicity: bool,
    fixed_vectors: bool,
) -> tuple[ComplexArray, PlaneExpansionContext]: ...
def cylindrical_channels(
    modes: list[tuple[int, float, int, int]],
    positions: list[list[float]],
    ks: list[complex],
    q: list[list[float]],
    polarizations: list[int],
    period: float,
    helicity: bool,
    fixed_q: bool,
) -> tuple[ComplexArray, ChannelsContext]: ...

class InterfaceContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray, RealArray]: ...

def interface(
    ks: list[list[complex]], z: list[complex], q: list[float], axis: int, fixed_q: bool
) -> tuple[ComplexArray, InterfaceContext]: ...

class LayersContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray, RealArray, RealArray]: ...

def layer_stack(
    ks: list[list[complex]],
    zs: list[complex],
    q: list[list[float]],
    thickness: list[float],
    axis: int,
    fixed_q: bool,
) -> tuple[ComplexArray, LayersContext]: ...

class PeriodicConversionContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, RealArray, ComplexArray, RealArray, float]: ...

def periodic_conversion(
    to: list[tuple[int, float, int, int]],
    source: list[tuple[int, int, int, int]],
    to_positions: list[list[float]],
    source_positions: list[list[float]],
    ks: list[complex],
    period: float,
    helicity: bool,
) -> tuple[ComplexArray, PeriodicConversionContext]: ...

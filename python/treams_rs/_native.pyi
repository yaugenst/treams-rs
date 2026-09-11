from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]

class QContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, RealArray, ComplexArray, ComplexArray]: ...

def ebcm_qmat(
    samples: RealArray,
    to: list[tuple[int, int, int]],
    source: list[tuple[int, int, int]],
    ks: ComplexArray,
    zs: tuple[complex, complex],
    singular: bool,
    legacy: bool,
) -> tuple[ComplexArray, QContext]: ...

class MetricContext:
    def pullback(self, cotangent: float) -> tuple[ComplexArray, RealArray]: ...

def tmatrix_metric(
    operator: ComplexArray, polarizations: list[int], ks: tuple[float, float], kind: str
) -> tuple[float, MetricContext]: ...

class SingularContext:
    def pullback(self, cotangent: RealArray) -> ComplexArray: ...

def svdvals(operator: ComplexArray) -> tuple[RealArray, SingularContext]: ...

class RotationContext:
    def pullback(self, cotangent: ComplexArray) -> list[float]: ...

class FieldOperatorContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, RealArray, ComplexArray]: ...
    def pullback_axial(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, RealArray, ComplexArray, RealArray]: ...

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

def hankel_scalar(order: float, z: complex, first: bool) -> complex: ...

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

def bessel_scalar(
    order: float, z: complex, kind: str, spherical: bool, derivative: int
) -> tuple[ComplexArray, BesselContext]: ...

class BesselContext:
    def pullback(self, cotangent: ComplexArray) -> ComplexArray: ...

def bessel(
    orders: RealArray,
    arguments: ComplexArray,
    kind: str,
    spherical: bool,
    derivative: int,
    shape: tuple[int, ...],
    argument_shape: tuple[int, ...],
) -> tuple[ComplexArray, BesselContext]: ...

lpmv: np.ufunc
pi_fun: np.ufunc
tau_fun: np.ufunc

class AngularContext:
    def pullback(self, cotangent: ComplexArray) -> ComplexArray: ...

def angular_value(degree: float, order: float, z: complex, kind: str) -> complex: ...
def angular_scalar(
    degree: float, order: float, z: complex, kind: str
) -> tuple[ComplexArray, AngularContext]: ...
def angular(
    degrees: RealArray,
    orders: RealArray,
    arguments: ComplexArray,
    kind: str,
    shape: tuple[int, ...],
    argument_shape: tuple[int, ...],
) -> tuple[ComplexArray, AngularContext]: ...

class ParticleClusterContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[list[ComplexArray], RealArray, ComplexArray]: ...

def particle_cluster(
    local: list[ComplexArray],
    modes: list[tuple[int, int, int, int]],
    positions: list[list[float]],
    ks: tuple[complex, complex],
    helicity: bool,
) -> tuple[ComplexArray, ParticleClusterContext]: ...
def cylindrical_particle_cluster(
    local: list[ComplexArray],
    modes: list[tuple[int, float, int, int]],
    positions: list[list[float]],
    ks: tuple[complex, complex],
    helicity: bool,
) -> tuple[ComplexArray, ParticleClusterContext]: ...

class ChiralityContext:
    def pullback(
        self, cotangent: NDArray[np.complex128]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64]]: ...

def chirality_density(
    ks: NDArray[np.complex128],
    normal: NDArray[np.complex128],
    interval: tuple[float, float],
) -> tuple[NDArray[np.complex128], ChiralityContext]: ...

class OrientedChiralityContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, ComplexArray, RealArray]: ...

def oriented_chirality(
    transverse: RealArray,
    normal: ComplexArray,
    polarizations: list[int],
    axis: int,
    interval: tuple[float, float],
) -> tuple[ComplexArray, OrientedChiralityContext]: ...

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
def smatrix_illuminate_forward(
    lower: ComplexArray, upper: ComplexArray, up: ComplexArray, down: ComplexArray
) -> ComplexArray: ...

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
    def pullback_axial(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, RealArray, ComplexArray, RealArray]: ...
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
    def pullback_axial(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, RealArray, RealArray, ComplexArray, RealArray]: ...

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
    def pullback_axial(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, RealArray, ComplexArray, RealArray, RealArray, RealArray]: ...
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[RealArray, RealArray, ComplexArray, RealArray, RealArray]: ...

class PlanePhaseContext:
    def pullback(self, cotangent: ComplexArray) -> tuple[RealArray, ComplexArray]: ...

def plane_phases(
    points: RealArray, vectors: ComplexArray
) -> tuple[ComplexArray, PlanePhaseContext]: ...

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

class PlanePermutationContext:
    def pullback(self, cotangent: ComplexArray) -> ComplexArray: ...

def plane_permutation(
    vectors: ComplexArray, polarizations: RealArray, turns: int, helicity: bool
) -> tuple[ComplexArray, PlanePermutationContext]: ...

wignersmalld: np.ufunc
wignerd: np.ufunc
wigner3j: np.ufunc
incgamma_ufunc: np.ufunc
intkambe_ufunc: np.ufunc

class WignerContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]: ...

def wigner(
    labels: list[tuple[int, int, int]],
    phi: ComplexArray,
    theta: ComplexArray,
    psi: ComplexArray,
    shape: tuple[int, ...],
    argument_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> tuple[ComplexArray, WignerContext]: ...
def wigner_scalar(
    labels: tuple[int, int, int], angles: tuple[complex, complex, complex]
) -> tuple[ComplexArray, WignerContext]: ...
def wigner3j_scalar(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float: ...

car2cyl: np.ufunc
car2sph: np.ufunc
cyl2car: np.ufunc
cyl2sph: np.ufunc
sph2car: np.ufunc
sph2cyl: np.ufunc
car2pol: np.ufunc
pol2car: np.ufunc
vcar2cyl: np.ufunc
vcar2sph: np.ufunc
vcyl2car: np.ufunc
vcyl2sph: np.ufunc
vsph2car: np.ufunc
vsph2cyl: np.ufunc
vcar2pol: np.ufunc
vpol2car: np.ufunc

class CoordinateContext:
    def pullback(self, cotangent: RealArray) -> RealArray: ...

class VectorCoordinateContext:
    def pullback(self, cotangent: ComplexArray) -> tuple[ComplexArray, RealArray]: ...

def coordinates(
    points: RealArray, kind: str
) -> tuple[RealArray, CoordinateContext]: ...
def vector_coordinates(
    vectors: ComplexArray,
    points: RealArray,
    kind: str,
    shape: tuple[int, ...],
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[ComplexArray, VectorCoordinateContext]: ...

sph_harm: np.ufunc
vsh_X: np.ufunc  # noqa: N816 - upstream public function name
vsh_Y: np.ufunc  # noqa: N816 - upstream public function name
vsh_Z: np.ufunc  # noqa: N816 - upstream public function name
vsw_M: np.ufunc  # noqa: N816 - upstream public function name
vsw_N: np.ufunc  # noqa: N816 - upstream public function name
vsw_A: np.ufunc  # noqa: N816 - upstream public function name
vsw_rM: np.ufunc  # noqa: N816 - upstream public function name
vsw_rN: np.ufunc  # noqa: N816 - upstream public function name
vsw_rA: np.ufunc  # noqa: N816 - upstream public function name
vcw_M: np.ufunc  # noqa: N816 - upstream public function name
vcw_N: np.ufunc  # noqa: N816 - upstream public function name
vcw_A: np.ufunc  # noqa: N816 - upstream public function name
vcw_rM: np.ufunc  # noqa: N816 - upstream public function name
vcw_rN: np.ufunc  # noqa: N816 - upstream public function name
vcw_rA: np.ufunc  # noqa: N816 - upstream public function name
vpw_M: np.ufunc  # noqa: N816 - upstream public function name
vpw_N: np.ufunc  # noqa: N816 - upstream public function name
vpw_A: np.ufunc  # noqa: N816 - upstream public function name

class WaveContext:
    def pullback(
        self, cotangent: NDArray[np.complex128]
    ) -> list[NDArray[np.complex128]]: ...

def vector_wave(
    kind: str,
    labels: list[tuple[int, int, int]],
    arguments: list[NDArray[np.complex128]],
    shape: tuple[int, ...],
    argument_shapes: list[tuple[int, ...]],
) -> tuple[NDArray[np.complex128], WaveContext]: ...

tl_vsw_A: np.ufunc  # noqa: N816 - upstream public function name
tl_vsw_B: np.ufunc  # noqa: N816 - upstream public function name
tl_vsw_rA: np.ufunc  # noqa: N816 - upstream public function name
tl_vsw_rB: np.ufunc  # noqa: N816 - upstream public function name

class PolarTranslationContext:
    def pullback(
        self, cotangent: NDArray[np.complex128]
    ) -> tuple[
        NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]
    ]: ...

def spherical_translation(
    modes: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    arguments: tuple[NDArray[np.complex128], ...],
    helicity: bool,
    singular: bool,
    shape: tuple[int, ...],
    argument_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> tuple[NDArray[np.complex128], PolarTranslationContext]: ...

tl_vcw: np.ufunc
tl_vcw_r: np.ufunc

class CylindricalTranslationContext:
    def pullback(
        self, cotangent: NDArray[np.complex128]
    ) -> tuple[
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
    ]: ...

def cylindrical_translation(
    orders: list[int],
    arguments: tuple[NDArray[np.complex128], ...],
    singular: bool,
    shape: tuple[int, ...],
    argument_shapes: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ],
) -> tuple[NDArray[np.complex128], CylindricalTranslationContext]: ...

sw_rotate: np.ufunc
cw_rotate: np.ufunc
sw_translate_sh: np.ufunc
sw_translate_rh: np.ufunc
sw_translate_sp: np.ufunc
sw_translate_rp: np.ufunc
cw_translate_s: np.ufunc
cw_translate_r: np.ufunc

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

pw_to_sw_h: np.ufunc
pw_to_sw_p: np.ufunc
pw_to_cw: np.ufunc
cw_to_sw_h: np.ufunc
cw_to_sw_p: np.ufunc
pw_permute_h: np.ufunc
pw_permute_p: np.ufunc
pw_inverse_h: np.ufunc
pw_inverse_p: np.ufunc

sw_to_pw_h: np.ufunc
sw_to_pw_p: np.ufunc
cw_to_pw: np.ufunc
sw_to_cw_h: np.ufunc
sw_to_cw_p: np.ufunc

class GammaContext:
    def pullback(self, cotangent: ComplexArray) -> ComplexArray: ...

class KambeContext:
    def pullback(
        self, cotangent: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray]: ...

def gamma_record(
    degrees: NDArray[np.float64],
    arguments: ComplexArray,
    shape: Sequence[int],
    argument_shape: Sequence[int],
) -> tuple[ComplexArray, GammaContext]: ...
def gamma_record_scalar(n: float, z: complex) -> tuple[ComplexArray, GammaContext]: ...
def kambe_record(
    orders: NDArray[np.int32],
    z: ComplexArray,
    eta: ComplexArray,
    shape: Sequence[int],
    argument_shapes: tuple[Sequence[int], Sequence[int]],
) -> tuple[ComplexArray, KambeContext]: ...
def kambe_record_scalar(
    n: int, z: complex, eta: complex
) -> tuple[ComplexArray, KambeContext]: ...

cell_volume: np.ufunc
cell_reciprocal: np.ufunc
refractive_indices: np.ufunc
wave_vector_z: np.ufunc
first_brillouin_1d: np.ufunc

def lattice_cube(dim: int, n: int, edge: bool) -> NDArray[np.int64]: ...
def diffraction_orders(b: RealArray, radius: float) -> NDArray[np.int64]: ...
def first_brillouin(k: RealArray, b: RealArray, dim: int, n: int) -> RealArray: ...
def cylindrical_translation_scalar(
    kz: float,
    mu: int,
    qz: float,
    m: int,
    kr: complex,
    phi: float,
    z: float,
    singular: bool,
) -> complex: ...
def plane_permutation_scalar(
    kx: complex, ky: complex, kz: complex, p: int, q: int, helicity: bool, inverse: bool
) -> complex: ...

import numpy as np
from numpy.typing import NDArray

type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]

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

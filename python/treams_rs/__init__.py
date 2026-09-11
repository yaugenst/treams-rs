"""T-matrix scattering with a Rust numerical core and native pullbacks."""

from . import coeffs, diff, ebcm, lattice
from ._core import (
    CylindricalWaveBasis,
    Material,
    PlaneWaveBasisByComp,
    PlaneWaveBasisByUnitVector,
    SphericalWaveBasis,
)
from ._operators import (
    bfield,
    dfield,
    efield,
    expand,
    expandlattice,
    ffield,
    gfield,
    hfield,
    rotate,
)
from ._plane import PlaneWave, plane_wave, plane_wave_angle
from ._smatrix import SMatrices, poynting_avg_z
from ._tmatrix import TMatrix, TMatrixC

__all__ = [
    "CylindricalWaveBasis",
    "Material",
    "PlaneWave",
    "PlaneWaveBasisByComp",
    "PlaneWaveBasisByUnitVector",
    "SMatrices",
    "SphericalWaveBasis",
    "TMatrix",
    "TMatrixC",
    "bfield",
    "coeffs",
    "dfield",
    "diff",
    "ebcm",
    "efield",
    "expand",
    "expandlattice",
    "ffield",
    "gfield",
    "hfield",
    "lattice",
    "plane_wave",
    "plane_wave_angle",
    "poynting_avg_z",
    "rotate",
]

"""T-matrix scattering with a Rust numerical core and native pullbacks."""

from . import coeffs, diff, ebcm, lattice, special
from ._core import (
    CylindricalWaveBasis,
    Material,
    PlaneWaveBasisByComp,
    PlaneWaveBasisByUnitVector,
    SphericalWaveBasis,
)
from ._operators import (
    bfield,
    changepoltype,
    dfield,
    efield,
    expand,
    expandlattice,
    ffield,
    gfield,
    hfield,
    permute,
    rotate,
    translate,
)
from ._plane import PlaneWave, plane_wave, plane_wave_angle
from ._smatrix import SMatrices, chirality_density, poynting_avg_z
from ._source import MultipoleWave, cylindrical_wave, spherical_wave
from ._tmatrix import TMatrix, TMatrixC

__all__ = [
    "CylindricalWaveBasis",
    "Material",
    "MultipoleWave",
    "PlaneWave",
    "PlaneWaveBasisByComp",
    "PlaneWaveBasisByUnitVector",
    "SMatrices",
    "SphericalWaveBasis",
    "TMatrix",
    "TMatrixC",
    "bfield",
    "changepoltype",
    "chirality_density",
    "coeffs",
    "cylindrical_wave",
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
    "permute",
    "plane_wave",
    "plane_wave_angle",
    "poynting_avg_z",
    "rotate",
    "special",
    "spherical_wave",
    "translate",
]

"""T-matrix scattering with a Rust numerical core and native pullbacks.

Start with ``support_catalog()`` for installed capabilities, signatures and
pullback contracts, or ``python -m treams_rs --format markdown`` for an offline
reference. Repository readers can start at ``llms.txt`` and ``docs/agents.md``.
Optional framework adapters are explicit submodule imports (advect, jax, torch).
"""

from importlib import import_module
from types import ModuleType

from . import (
    coeffs,
    config,
    cw,
    diff,
    ebcm,
    iterative,
    lattice,
    misc,
    pw,
    special,
    sw,
    testing,
)
from ._array import PhysicsArray
from ._core import (
    CylindricalWaveBasis,
    Material,
    PlaneWaveBasisByComp,
    PlaneWaveBasisByUnitVector,
    SphericalWaveBasis,
)
from ._lattice import Lattice, WaveVector
from ._operators import (
    BField,
    ChangePoltype,
    DField,
    EField,
    Expand,
    ExpandLattice,
    FField,
    FieldOperator,
    GField,
    HField,
    Operator,
    OperatorAttribute,
    Permute,
    Rotate,
    Translate,
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
from ._smatrix import SMatrices, SMatrix, chirality_density, poynting_avg_z
from ._source import MultipoleWave, cylindrical_wave, spherical_wave
from ._tmatrix import TMatrix, TMatrixC
from .support import support_catalog

__all__ = [
    "BField",
    "ChangePoltype",
    "CylindricalWaveBasis",
    "DField",
    "EField",
    "Expand",
    "ExpandLattice",
    "FField",
    "FieldOperator",
    "GField",
    "HField",
    "Lattice",
    "Material",
    "MultipoleWave",
    "Operator",
    "OperatorAttribute",
    "Permute",
    "PhysicsArray",
    "PlaneWave",
    "PlaneWaveBasisByComp",
    "PlaneWaveBasisByUnitVector",
    "Rotate",
    "SMatrices",
    "SMatrix",
    "SphericalWaveBasis",
    "TMatrix",
    "TMatrixC",
    "Translate",
    "WaveVector",
    "bfield",
    "changepoltype",
    "chirality_density",
    "coeffs",
    "config",
    "cw",
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
    "iterative",
    "lattice",
    "misc",
    "permute",
    "plane_wave",
    "plane_wave_angle",
    "poynting_avg_z",
    "pw",
    "rotate",
    "special",
    "spherical_wave",
    "support_catalog",
    "sw",
    "testing",
    "translate",
]


_OPTIONAL_MODULES = {
    "advect": "advect",
    "jax": "jax",
    "torch": "torch",
    "io": "h5py",
    "cuda": None,
}


def __getattr__(name: str) -> ModuleType:
    if name not in _OPTIONAL_MODULES:
        raise AttributeError(f"module treams_rs has no attribute {name!r}")
    try:
        module = import_module(f".{name}", __name__)
    except ModuleNotFoundError as error:
        if error.name != _OPTIONAL_MODULES[name]:
            raise
        raise ModuleNotFoundError(
            f"treams_rs.{name} needs {_OPTIONAL_MODULES[name]}; install the optional extra with "
            f"pip install 'treams-rs[{name}]' in the environment containing your private wheel"
        ) from error
    globals()[name] = module
    return module

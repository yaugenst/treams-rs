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
    cw,
    diff,
    ebcm,
    iterative,
    lattice,
    misc,
    operators,
    pw,
    special,
    sw,
    testing,
)
from ._core import CylindricalWaveBasis as CylindricalBasis
from ._core import Material
from ._core import PlaneWaveBasisByComp as PlaneWavePorts
from ._core import PlaneWaveBasisByUnitVector as PlaneWaveBasis
from ._core import SphericalWaveBasis as SphericalBasis
from ._lattice import Lattice, WaveVector
from ._physics import (
    Cluster,
    PeriodicResponse,
    PeriodicWave,
    ScatteringFactor,
    cylinder_tmatrix,
    interface,
    multilayer_cylinder_tmatrix,
    multilayer_slab,
    multilayer_sphere_tmatrix,
    propagation,
    slab,
    solve_periodic,
    sphere_tmatrix,
    stack,
)
from ._plane import PlaneWave, plane_wave, plane_wave_angle
from ._smatrix import (
    BandModes,
    CircularDichroism,
    PowerBalance,
    ScatteredPorts,
    chirality_density,
    poynting_avg_z,
)
from ._smatrix import SMatrices as SMatrix
from ._smatrix import SMatrix as ScatteringBlock
from ._source import MultipoleWave as Wave
from ._source import cylindrical_wave, spherical_wave
from ._tmatrix import CrossSections, TMatrix
from ._tmatrix import TMatrixC as CylindricalTMatrix
from .support import support_catalog

__all__ = [
    "BandModes",
    "CircularDichroism",
    "Cluster",
    "CrossSections",
    "CylindricalBasis",
    "CylindricalTMatrix",
    "Lattice",
    "Material",
    "PeriodicResponse",
    "PeriodicWave",
    "PlaneWave",
    "PlaneWaveBasis",
    "PlaneWavePorts",
    "PowerBalance",
    "SMatrix",
    "ScatteredPorts",
    "ScatteringBlock",
    "ScatteringFactor",
    "SphericalBasis",
    "TMatrix",
    "Wave",
    "WaveVector",
    "chirality_density",
    "coeffs",
    "cw",
    "cylinder_tmatrix",
    "cylindrical_wave",
    "diff",
    "ebcm",
    "interface",
    "iterative",
    "lattice",
    "misc",
    "multilayer_cylinder_tmatrix",
    "multilayer_slab",
    "multilayer_sphere_tmatrix",
    "operators",
    "plane_wave",
    "plane_wave_angle",
    "poynting_avg_z",
    "propagation",
    "pw",
    "slab",
    "solve_periodic",
    "special",
    "sphere_tmatrix",
    "spherical_wave",
    "stack",
    "support_catalog",
    "sw",
    "testing",
]


_OPTIONAL_MODULES = {
    "advect": "advect",
    "jax": "jax",
    "torch": "torch",
    "io": "h5py",
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

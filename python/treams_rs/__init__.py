"""T-matrix scattering with Rust numerics and analytic first-order derivatives.

Import as ``import treams_rs as tr``. Lengths use any consistent unit;
``k0 = 2*pi/vacuum_wavelength`` uses its inverse. Material numbers are relative
permittivities (epsilon), not refractive indices; use ``Material(epsilon, mu,
kappa)`` for magnetic/chiral media. Passive loss has positive imaginary epsilon.

A complete particle calculation::

    import numpy as np
    import treams_rs as tr

    k0 = 2*np.pi/0.8
    particle = tr.sphere_tmatrix(k0=k0, lmax=3, radius=0.12, material=3+0.02j)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
    cross = particle.cross_sections(incident)
    print(cross.scattering, cross.extinction, cross.absorption)  # areas
    points = [[0.3, 0.1, 0.4]]
    total_e = incident.efield(points) + particle.scatter(incident).efield(points)

Choose the physical workflow:

* ``multilayer_sphere_tmatrix``: increasing radii, one material per finite layer,
  inside-out; exterior passed separately as ``medium``.
* ``cylinder_tmatrix``: infinite z cylinder; specify ``kz`` and ``mmax`` and use
  ``cross_widths(incident)`` for lengths instead of areas.
* ``Cluster(particles, positions=xyz).solve()``: full finite multiple-scattering
  response; ``.scatter(incident)`` directly solves requested illuminations.
* ``solve_periodic(particle, lattice=cell, kpar=[0, 0]).to_smatrix(ports)``:
  periodic response in plane-wave channels, ready to combine with planar layers.
* ``slab`` / ``multilayer_slab`` / ``propagation`` / ``stack``: layers ordered
  negative to positive z. Specify ``negative_medium`` and ``positive_medium``
  for exterior media. ``PlaneWavePorts.default([0, 0])`` gives normal-incidence
  ports. Pass a physical ``plane_wave`` to ``network.power(incident)`` to select
  direction and helicity by name; the incident side follows its propagation.
  The result has named ``transmission``, ``reflection`` and ``absorption``.
  Reverse incidence uses a -z wave without reversing layer order. Raw amplitudes
  follow ``ports.pol`` (1 means positive helicity, 0 means negative); inspect that
  array instead of assuming the order.

Sensitivity and optimization use an explicit framework namespace. With the
``[advect]`` extra installed::

    import advect as ad
    import treams_rs.advect as tr

    def objective(radius):
        particle = tr.sphere_tmatrix(k0=2.0, lmax=2, radius=radius, material=3.0)
        incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=2.0)
        return particle.cross_sections(incident).scattering

    value, derivative = ad.value_and_grad(objective)(0.2)

Construct changing geometry/materials inside the objective and keep traced
values as framework arrays (``advect.numpy``); convert to float/NumPy only after
differentiation. Continuous geometry, material and frequency compose through
native pullbacks; mode cutoffs, integer labels and topology stay fixed. CPU,
first-order reverse mode only. JAX and PyTorch have explicit optional namespaces.
The root NumPy namespace does not trace; ``diff`` exposes manual native VJPs.

Offline help: ``python -m treams_rs`` shows this quickstart;
``python -m treams_rs sphere_tmatrix`` or ``python -m treams_rs advect`` shows
focused contracts; ``python -m treams_rs --search cross`` lists matching names.
``help(tr.sphere_tmatrix)`` also works. ``support_catalog()`` or explicit
``python -m treams_rs --format json`` gives the complete source-derived catalog.
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
    ScatteringBlock,
    SMatrix,
    chirality_density,
    poynting_avg_z,
)
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
        for public_name in __all__:
            value = globals()[public_name]
            if isinstance(value, type) and value.__name__ == name:
                raise AttributeError(
                    f"{name} is exported as treams_rs.{public_name}; use {public_name}"
                )
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

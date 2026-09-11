# /// script
# requires-python = ">=3.12,<3.14"
# ///
"""Run with the clean wheel environment, without the upstream scientific stack."""

import importlib.util

import advect
import advect.numpy as anp
import numpy as np

import treams_rs as tr
from treams_rs import advect as ad

for dependency in ("treams", "scipy", "autograd"):
    assert importlib.util.find_spec(dependency) is None, dependency

cell = np.diag([1.7, 1.8])
basis = tr.PlaneWaveBasisByComp.diffr_orders([0, 0], cell, 4)
sphere = tr.TMatrix.sphere(2, 2.1, 0.2, [3, 1])
array = tr.SMatrices.from_array(sphere, basis, lattice=cell, kpar=[0, 0])
np.testing.assert_allclose(
    sum(array.tr(tr.plane_wave([0, 0, 1], 1, k0=2.1))), 1, atol=1e-10
)

modes = tr.SphericalWaveBasis.default(2)
positions = np.zeros((1, 3))


def reflectance(radius):
    particle = ad.sphere(2, 2.1, anp.reshape(radius, (1,)), [3.0, 1.0])
    coupling = ad.lattice_expansion(
        positions, positions, [2.1, 2.1], [0, 0], cell, destination=modes, source=modes
    )
    response = ad.interaction(particle, coupling)
    channels = ad.spherical_channels(
        positions,
        [2.1, 2.1],
        [[0, 0], [0, 0]],
        np.linalg.det(cell),
        basis=modes,
        polarizations=[1, 0],
        fixed_q=True,
    )
    value = ad.smatrix_from_array(response, channels)[1, 0, :, 0]
    return anp.sum(anp.real(value * anp.conj(value)))


gradient = advect.grad(reflectance)(np.array(0.2))
h = 1e-5
np.testing.assert_allclose(
    gradient, (reflectance(0.2 + h) - reflectance(0.2 - h)) / (2 * h), rtol=1e-7
)
print("Clean wheel: periodic power conservation and complete Advect gradient passed")

cb = tr.CylindricalWaveBasis.default([0.2], 1)
points = np.array([[0.8, 0.3, 0.1]])
coefficients = np.ones(len(cb), complex)
electric = tr.efield(points, basis=cb, k0=1.3)
np.testing.assert_allclose(
    electric @ coefficients,
    tr.diff.field(coefficients, points, cb, [1.3, 1.3])[0],
    rtol=1e-12,
    atol=1e-12,
)


def magnetic_energy(impedance):
    value = ad.hfield(
        coefficients, points, cb.positions, [1.3, 1.3], impedance, basis=cb
    )
    return anp.sum(anp.real(value * anp.conj(value)))


np.testing.assert_allclose(
    advect.grad(magnetic_energy)(np.array(0.8)),
    -2 * magnetic_energy(0.8) / 0.8,
    rtol=1e-12,
)
print("Clean wheel: cylindrical field operator and magnetic impedance gradient passed")


plane = tr.PlaneWaveBasisByComp.default([[0.2, 0.3], [1.5, -0.1]])
vectors = np.column_stack(plane.kvecs(1.3))
operator = tr.efield(points, basis=plane, k0=1.3)
np.testing.assert_allclose(
    operator, tr.diff.plane_field(None, points, vectors, plane.pol)[0], rtol=1e-12
)


def plane_energy(scale):
    field = ad.plane_field(
        np.ones(len(plane)), points / scale, vectors * scale, polarizations=plane.pol
    )
    return anp.sum(anp.real(field * anp.conj(field)))


np.testing.assert_allclose(advect.grad(plane_energy)(np.array(1.0)), 0, atol=1e-12)
print(
    "Clean wheel: plane operator and Advect coordinate/wavevector scale identity passed"
)


cylinder = tr.TMatrixC.cylinder([0.2], 3, 1.3, [0.2], [4, 1])
cports = tr.PlaneWaveBasisByComp.default([[0.2, 0.1]], "zx")
carray = tr.SMatrices.from_array(cylinder, cports, lattice=1.7, kpar=0.1)
np.testing.assert_allclose(sum(carray.tr([1, 0])), 1, atol=2e-10)


def cylindrical_radiation_energy(period):
    channels = ad.cylindrical_channels(
        cylinder.basis.positions,
        [1.3, 1.3],
        [0.1, 0.1],
        period,
        basis=cylinder.basis,
        kz_labels=[0.2, 0.2],
        polarizations=[1, 0],
    )
    radiated = channels[1]
    return anp.sum(anp.real(radiated * anp.conj(radiated)))


np.testing.assert_allclose(
    advect.grad(cylindrical_radiation_energy)(np.array(1.7)),
    -2 * cylindrical_radiation_energy(1.7) / 1.7,
    rtol=1e-12,
)
print(
    "Clean wheel: cylindrical array power and native radiation period gradient passed"
)

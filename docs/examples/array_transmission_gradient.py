# Requires: jax
"""Gradient of the transmission of a sphere array on a slab, checked by differences."""

import jax
import numpy as np

import treams_rs.jax as tr

# The adapter computes in double precision and requires JAX to accept it.
jax.config.update("jax_enable_x64", True)

# The array of the array_spheres example, at a vacuum wavelength of 400 nm.
k0 = 2 * np.pi / 400
kpar = [0, 0.3 * k0]
lattice = tr.Lattice.square(500)
material_sphere = tr.Material(4, 1, 0.05)
material_slab = 3
lmax = 3
ports = tr.PlaneWavePorts.diffr_orders(kpar, lattice, 0.02)
# A plane wave from below, with k_par = kpar.
kz = np.sqrt(k0**2 - kpar[1] ** 2)
direction = [0, kpar[1], kz]


def transmission(radius, thickness):
    """Transmitted fraction of the incident power."""
    sphere = tr.sphere_tmatrix(
        k0=k0, lmax=lmax, radius=radius, material=material_sphere
    )
    array = tr.solve_periodic(sphere, lattice=lattice, kpar=kpar).to_smatrix(ports)
    slab = tr.slab(k0=k0, basis=ports, thickness=thickness, material=material_slab)
    # The sphere centers lie one radius above the slab.
    gap = tr.propagation(k0=k0, basis=ports, distance=radius)
    total = tr.stack([slab, gap, array])
    incident = tr.plane_wave(direction, "positive_helicity", k0=k0)
    return total.power(incident).transmission


parameters = (100.0, 10.0)
names = ("radius", "thickness")
value, gradient = jax.value_and_grad(transmission, argnums=(0, 1))(*parameters)

# Central differences with a step of 1e-3 nm, one parameter at a time.
step = 1e-3
differences = []
for i in range(len(parameters)):
    shift = np.eye(len(parameters))[i] * step
    plus = transmission(*(np.asarray(parameters) + shift))
    minus = transmission(*(np.asarray(parameters) - shift))
    differences.append((plus - minus) / (2 * step))

gradient = np.asarray(gradient)
differences = np.asarray(differences)
np.testing.assert_allclose(gradient, differences, rtol=1e-6, atol=0)

if __name__ == "__main__":
    print(f"Transmission T = {float(value):.10g}")
    print("Derivatives of T in 1/nm")
    print(f"{'parameter':>10}{'value/nm':>10}{'jax.grad':>16}{'central diff.':>16}")
    rows = zip(names, parameters, gradient, differences, strict=True)
    for name, x, exact, approximate in rows:
        print(f"{name:>10}{x:>10.6g}{exact:>16.8g}{approximate:>16.8g}")

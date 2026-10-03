"""A periodic stack of sphere arrays on slabs: its Bloch modes along z."""

import numpy as np

import treams_rs as tr

# Enlarge for a full band diagram: treams uses 200 frequencies.
FREQUENCIES = 6

k0s = 2 * np.pi * np.linspace(1 / 10_000, 1 / 350, FREQUENCIES)
material_slab = 3
thickness = 10
material_sphere = tr.Material(4 + 0.1j, 1, 0.05)
lattice = tr.Lattice.square(500)
radius = 100
lmax = 3
# Period of the stack along z.
az = 210

# One array per frequency: Bloch wavenumbers kz in units of pi / az, kept
# where |Im| < 0.1, that is, for modes that decay slowly along the stack.
res = []
for k0 in k0s:
    kpar = [0, 0]
    sphere = tr.sphere_tmatrix(
        k0=k0, lmax=lmax, radius=radius, material=material_sphere
    )
    spheres = tr.solve_periodic(sphere, lattice=lattice, kpar=kpar)

    ports = tr.PlaneWavePorts.diffr_orders(kpar, lattice, 0.02)
    slab = tr.slab(k0=k0, basis=ports, thickness=thickness, material=material_slab)
    gap = tr.propagation(k0=k0, basis=ports, distance=radius)
    array = spheres.to_smatrix(ports)
    # One period: slab, gap, sphere centers, and the gap to the next slab.
    cell = tr.stack([slab, gap, array, gap])
    x = cell.bands(period=az).wavenumbers * az / np.pi
    # Eigenvalue solvers return the modes in no fixed order; sort them.
    res.append(np.sort_complex(x[np.abs(np.imag(x)) < 0.1]))

if __name__ == "__main__":
    print("Bloch wavenumbers kz az / pi with |Im| < 0.1")
    print(f"{'k0/(2 pi)':>13}{'Re':>13}{'Im':>13}")
    for k0, values in zip(k0s, res, strict=True):
        for value in values:
            print(f"{k0 / (2 * np.pi):>13.6g}{value.real:>13.6g}{value.imag:>13.6g}")

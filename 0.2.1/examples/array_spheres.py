"""A square array of chiral spheres on a thin slab: transmission and reflection."""

import numpy as np

import treams_rs as tr

# Enlarge for a full spectrum: treams uses 100 frequencies.
FREQUENCIES = 5

# The sweep stops short of 1/350 nm^-1, where the diffraction order with
# k_y = 0.3 k0 + 2 pi / 500 starts to propagate and the lattice sum diverges.
k0s = 2 * np.pi * np.linspace(1 / 600, 1 / 350, FREQUENCIES, endpoint=False)
material_slab = 3
thickness = 10
material_sphere = tr.Material(4, 1, 0.05)
lattice = tr.Lattice.square(500)
radius = 100
lmax = 3

# Rows hold (T, R) for each frequency.
power = np.zeros((FREQUENCIES, 2))
for i, k0 in enumerate(k0s):
    kpar = [0, 0.3 * k0]
    sphere = tr.sphere_tmatrix(
        k0=k0, lmax=lmax, radius=radius, material=material_sphere
    )
    spheres = tr.solve_periodic(sphere, lattice=lattice, kpar=kpar)

    # Ports for the diffraction orders kpar + G with |G| <= 0.02 / nm.
    ports = tr.PlaneWavePorts.diffr_orders(kpar, lattice, 0.02)
    slab = tr.slab(k0=k0, basis=ports, thickness=thickness, material=material_slab)
    # The array's S-matrix refers to the plane through the sphere centers,
    # one radius above the slab.
    gap = tr.propagation(k0=k0, basis=ports, distance=radius)
    array = spheres.to_smatrix(ports)
    total = tr.stack([slab, gap, array])

    # A plane wave from below, with k_par = kpar and its electric field along x.
    kz = np.sqrt(k0**2 - kpar[1] ** 2)
    incident = tr.plane_wave(direction=[0, kpar[1], kz], pol=[1, 0, 0], k0=k0)
    power[i] = total.power(incident)

if __name__ == "__main__":
    print("Transmission T and reflection R at oblique incidence")
    print(f"{'k0/(2 pi)':>13}{'T':>13}{'R':>13}")
    for k0, (t, r) in zip(k0s, power, strict=True):
        print(f"{k0 / (2 * np.pi):>13.6g}{t:>13.6g}{r:>13.6g}")

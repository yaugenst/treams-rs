"""A square array of chiral spheres on a slab, built from cylindrical waves."""

import numpy as np

import treams_rs as tr

# Enlarge for a full spectrum: treams uses 100 frequencies.
FREQUENCIES = 5

# The sweep stops short of 1/350 nm^-1, where the diffraction order with
# k_x = 0.3 k0 + 2 pi / 500 starts to propagate and the lattice sum diverges.
k0s = 2 * np.pi * np.linspace(1 / 600, 1 / 350, FREQUENCIES, endpoint=False)
material_slab = 3
thickness = 10
material_sphere = tr.Material(4, 1, 0.05)
pitch = 500
# The array lies in the zx plane: chains of spheres along z repeat along x, and
# the slab and the ports face the y direction.
lattice = tr.Lattice.square(pitch, "zx")
radius = 100
lmax = mmax = 3

# Rows hold (T, R) for each frequency.
power = np.zeros((FREQUENCIES, 2))
for i, k0 in enumerate(k0s):
    # The Bloch vector (k_z, k_x) of the array.
    kpar = [0, 0.3 * k0]

    # A chain of spheres along z, in cylindrical waves.
    sphere = tr.sphere_tmatrix(
        k0=k0, lmax=lmax, radius=radius, material=material_sphere
    )
    spheres = tr.solve_periodic(sphere, lattice=pitch, kpar=kpar[0])
    basis = tr.CylindricalBasis.diffr_orders(kpar[0], mmax, pitch, 0.02)
    spheres_cw = spheres.to_cylindrical(basis)
    # Chains repeated along x make the square array.
    chains = tr.solve_periodic(spheres_cw, lattice=pitch, kpar=kpar[1])

    ports = tr.PlaneWavePorts.diffr_orders(kpar, lattice, 0.02)
    slab = tr.slab(k0=k0, basis=ports, thickness=thickness, material=material_slab)
    # The array's S-matrix refers to the plane through the sphere centers,
    # one radius above the slab.
    gap = tr.propagation(k0=k0, basis=ports, distance=radius)
    array = chains.to_smatrix(ports)
    total = tr.stack([slab, gap, array])

    # A plane wave from below (negative y) with its electric field along z.
    ky = np.sqrt(k0**2 - kpar[1] ** 2)
    incident = tr.plane_wave(direction=[kpar[1], ky, 0], pol=[0, 0, 1], k0=k0)
    power[i] = total.power(incident)

if __name__ == "__main__":
    print("Transmission T and reflection R at oblique incidence")
    print(f"{'k0/(2 pi)':>13}{'T':>13}{'R':>13}")
    for k0, (t, r) in zip(k0s, power, strict=True):
        print(f"{k0 / (2 * np.pi):>13.6g}{t:>13.6g}{r:>13.6g}")

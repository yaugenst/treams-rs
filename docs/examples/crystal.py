"""A simple cubic crystal of spheres: where its lattice modes lie."""

import numpy as np

import treams_rs as tr

# Enlarge for a full spectrum: treams uses 200 frequencies.
FREQUENCIES = 5

k0s = 2 * np.pi * np.linspace(0.01, 1, FREQUENCIES)
material = tr.Material(12.4)
lmax = 3
radius = 0.2
lattice = tr.Lattice.cubic(0.5)
kpar = [0, 0, 0]

# A mode of the crystal exists where the lattice interaction matrix
# I - T C is singular; its smallest singular value dips towards zero there.
res = []
for k0 in k0s:
    sphere = tr.sphere_tmatrix(k0=k0, lmax=lmax, radius=radius, material=material)
    svd = np.linalg.svd(sphere.latticeinteraction(lattice, kpar), compute_uv=False)
    res.append(svd[-1])

if __name__ == "__main__":
    print("Smallest singular value of the lattice interaction matrix")
    print(f"{'k0/(2 pi)':>13}{'value':>13}")
    for k0, value in zip(k0s, res, strict=True):
        print(f"{k0 / (2 * np.pi):>13.6g}{value:>13.6g}")

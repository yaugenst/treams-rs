# Source: https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/array_spheres.py
# Adapted: reduced sizes, sweep ends before the diffraction threshold at 1/350,
# plotting removed.
# MIT license, copyright Dominik Beutel; see LICENSE.treams.
import numpy as np
import treams

k0s = 2 * np.pi * np.linspace(1 / 600, 1 / 350, 5, endpoint=False)
material_slab = 3
thickness = 10
material_sphere = treams.Material(4, 1, 0.05)
lattice = treams.Lattice.square(500)
radius = 100
lmax = 3

tr = np.zeros((len(k0s), 2))
for i, k0 in enumerate(k0s):
    kpar = [0, 0.3 * k0]
    spheres = treams.TMatrix.sphere(
        lmax, k0, radius, [material_sphere, 1]
    ).latticeinteraction.solve(lattice, kpar)

    pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, 0.02)
    plw = treams.plane_wave(kpar, [1, 0, 0], k0=k0, basis=pwb, material=1)
    slab = treams.SMatrices.slab(thickness, pwb, k0, [1, material_slab, 1])
    dist = treams.SMatrices.propagation([0, 0, radius], pwb, k0, 1)
    array = treams.SMatrices.from_array(spheres, pwb)
    total = treams.SMatrices.stack([slab, dist, array])
    tr[i, :] = total.tr(plw)

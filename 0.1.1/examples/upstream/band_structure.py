# Source: https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/band_structure.py
# Adapted: reduced sizes, plotting removed.
# MIT license, copyright Dominik Beutel; see LICENSE.treams.
import numpy as np
import treams

k0s = 2 * np.pi * np.linspace(1 / 10_000, 1 / 350, 6)
material_slab = 3
thickness = 10
material_sphere = treams.Material(4 + 0.1j, 1, 0.05)
lattice = treams.Lattice.square(500)
radius = 100
lmax = 3
az = 210

res = []
for k0 in k0s:
    kpar = [0, 0]
    spheres = treams.TMatrix.sphere(
        lmax, k0, radius, [material_sphere, 1]
    ).latticeinteraction.solve(lattice, kpar)

    pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, 0.02)
    plw = treams.plane_wave(kpar, [1, 0, 0], k0=k0, basis=pwb, material=1)
    slab = treams.SMatrices.slab(thickness, pwb, k0, [1, material_slab, 1])
    dist = treams.SMatrices.propagation([0, 0, radius], pwb, k0, 1)
    array = treams.SMatrices.from_array(spheres, pwb)
    total = treams.SMatrices.stack([slab, dist, array, dist])
    x, _ = total.bands_kz(az)
    x = x * az / np.pi
    sel = x[np.abs(np.imag(x)) < 0.1]
    res.append(sel)

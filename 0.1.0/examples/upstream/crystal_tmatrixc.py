# Source: https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/crystal_tmatrixc.py
# Adapted: reduced sizes, plotting removed.
# MIT license, copyright Dominik Beutel; see LICENSE.treams.
import numpy as np
import treams

k0s = 2 * np.pi * np.linspace(0.01, 1, 5)
materials = [treams.Material(12.4), treams.Material()]
mmax = 3
radius = 0.2
lattice = treams.Lattice.square(0.3)
kpar = [0, 0, 0]

res = []
for k0 in k0s:
    cyl = treams.TMatrixC.cylinder(kpar[2], mmax, k0, radius, materials)
    svd = np.linalg.svd(cyl.latticeinteraction(lattice, kpar[:2]), compute_uv=False)
    res.append(svd[-1])

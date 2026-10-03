# Source: https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/slab.py
# Adapted: reduced sizes, plotting removed.
# MIT license, copyright Dominik Beutel; see LICENSE.treams.
import numpy as np
import treams

k0s = 2 * np.pi * np.linspace(1 / 1000, 1 / 300, 6)
materials = [(), (12.4 + 1j, 1 + 0.1j, 0.5 + 0.05j), (2, 2)]
thickness = 50
tr = np.zeros((2, len(k0s), 2))
for i, k0 in enumerate(k0s):
    pwb = treams.PlaneWaveBasisByComp.default([0, 0.5 * k0])
    slab = treams.SMatrices.slab(thickness, pwb, k0, materials)
    tr[0, i, :] = slab.tr([1, 0])
    tr[1, i, :] = slab.tr([0, 1])

# Source: https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/cylinder_tmatrixc.py
# Adapted: reduced sizes, plotting removed.
# MIT license, copyright Dominik Beutel; see LICENSE.treams.
import numpy as np
import treams

k0s = 2 * np.pi * np.linspace(1 / 6000, 1 / 300, 6)
materials = [treams.Material(16 + 0.5j), treams.Material()]
mmax = 4
radius = 75
kzs = [0]
cylinders = [treams.TMatrixC.cylinder(kzs, mmax, k0, radius, materials) for k0 in k0s]

xw_sca = np.array([tm.xw_sca_avg for tm in cylinders]) / (2 * radius)
xw_ext = np.array([tm.xw_ext_avg for tm in cylinders]) / (2 * radius)

cwb_mmax0 = treams.CylindricalWaveBasis.default(kzs, 0)
cylinders_mmax0 = [tm[cwb_mmax0] for tm in cylinders]
xw_sca_mmax0 = np.array([tm.xw_sca_avg for tm in cylinders_mmax0]) / (2 * radius)
xw_ext_mmax0 = np.array([tm.xw_ext_avg for tm in cylinders_mmax0]) / (2 * radius)

tm = cylinders[-1]
inc = treams.plane_wave([tm.k0, 0, 0], 1, k0=tm.k0, material=tm.material)
sca = tm @ inc.expand(tm.basis)

x = np.linspace(-100, 100, 7)
y = np.linspace(-100, 100, 7)
z = 0
xx, yy = np.meshgrid(x, y, indexing="ij")
zz = np.full_like(xx, z)
grid = np.stack((xx, yy, zz), axis=-1)

intensity = np.zeros_like(xx)
valid = tm.valid_points(grid, [radius])
intensity[~valid] = np.nan
intensity[valid] = 0.5 * np.sum(
    np.abs(inc.efield(grid[valid]) + sca.efield(grid[valid])) ** 2, -1
)

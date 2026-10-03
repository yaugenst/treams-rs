# Source: https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/sphere.py
# Adapted: reduced sizes, plotting removed.
# MIT license, copyright Dominik Beutel; see LICENSE.treams.
import numpy as np
import treams

k0s = 2 * np.pi * np.linspace(1 / 700, 1 / 300, 6)
materials = [treams.Material(16 + 0.5j), treams.Material()]
lmax = 4
radius = 75
spheres = [treams.TMatrix.sphere(lmax, k0, radius, materials) for k0 in k0s]

xs_sca = np.array([tm.xs_sca_avg for tm in spheres]) / (np.pi * radius**2)
xs_ext = np.array([tm.xs_ext_avg for tm in spheres]) / (np.pi * radius**2)

swb_lmax1 = treams.SphericalWaveBasis.default(1)
spheres_lmax1 = [tm[swb_lmax1] for tm in spheres]
xs_sca_lmax1 = np.array([tm.xs_sca_avg for tm in spheres_lmax1]) / (np.pi * radius**2)
xs_ext_lmax1 = np.array([tm.xs_ext_avg for tm in spheres_lmax1]) / (np.pi * radius**2)

tm = spheres[-1]
inc = treams.plane_wave([0, 0, tm.k0], 1, k0=tm.k0, material=tm.material)
sca = tm @ inc.expand(tm.basis)

x = np.linspace(-100, 100, 7)
y = 0
z = np.linspace(-100, 100, 7)
xx, zz = np.meshgrid(x, z, indexing="ij")
yy = np.full_like(xx, y)
grid = np.stack((xx, yy, zz), axis=-1)

intensity = np.zeros_like(xx)
valid = tm.valid_points(grid, [radius])
intensity[~valid] = np.nan
intensity[valid] = 0.5 * np.sum(
    np.abs(inc.efield(grid[valid]) + sca.efield(grid[valid])) ** 2, -1
)

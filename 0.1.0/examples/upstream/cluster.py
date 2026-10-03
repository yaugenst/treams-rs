# Source: https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/cluster.py
# Adapted: reduced sizes, plotting removed.
# MIT license, copyright Dominik Beutel; see LICENSE.treams.
import numpy as np
import treams

k0 = 2 * np.pi / 1000
materials = [treams.Material(16 + 0.5j), treams.Material()]
lmax = 3
radii = [110, 90, 80, 75]
positions = (220 / np.sqrt(24)) * np.array(
    [
        [np.sqrt(8), 0, -1],
        [-np.sqrt(2), np.sqrt(6), -1],
        [-np.sqrt(2), -np.sqrt(6), -1],
        [0, 0, 3],
    ]
)

spheres = [treams.TMatrix.sphere(lmax, k0, r, materials) for r in radii]
tm = treams.TMatrix.cluster(spheres, positions).interaction.solve()

inc = treams.plane_wave([k0, 0, 0], 0, k0=tm.k0, material=tm.material)
sca = tm @ inc.expand(tm.basis)
xs = tm.xs(inc)

x = np.linspace(-300, 300, 7)
y = 0
z = np.linspace(-300, 300, 7)
xx, zz = np.meshgrid(x, z, indexing="ij")
yy = np.full_like(xx, y)
grid = np.stack((xx, yy, zz), axis=-1)

intensity = np.zeros_like(xx)
valid = tm.valid_points(grid, radii)
intensity[~valid] = np.nan
intensity[valid] = 0.5 * np.sum(
    np.abs(inc.efield(grid[valid]) + sca.efield(grid[valid])) ** 2, -1
)

tm_global = tm.expand(treams.SphericalWaveBasis.default(6))
sca = tm_global @ inc.expand(tm_global.basis)

xs = tm_global.xs(inc)

intensity_global = np.zeros_like(xx)
valid = tm_global.valid_points(grid, [260])
intensity_global[~valid] = np.nan
intensity_global[valid] = 0.5 * np.sum(
    np.abs(inc.efield(grid[valid]) + sca.efield(grid[valid])) ** 2, -1
)

inc = treams.plane_wave([0, 0, -k0], 0, k0=tm.k0, material=tm.material)
tm_rotate = tm_global.rotate(0, np.pi / 2)
sca = tm_rotate @ inc.expand(tm_rotate.basis)

xs = tm_rotate.xs(inc)

intensity_global = np.zeros_like(xx)
valid = tm_rotate.valid_points(grid, [260])
intensity_global[~valid] = np.nan
intensity_global[valid] = 0.5 * np.sum(
    np.abs(inc.efield(grid[valid]) + sca.efield(grid[valid])) ** 2, -1
)

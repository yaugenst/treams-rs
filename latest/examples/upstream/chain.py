# Source: https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/chain.py
# Adapted: reduced sizes, plotting removed.
# MIT license, copyright Dominik Beutel; see LICENSE.treams.
import numpy as np
import treams

k0 = 2 * np.pi / 700
materials = [treams.Material(-16.5 + 1j), treams.Material()]
lmax = 3
radii = [75, 75]
positions = [[-30, 0, -75], [30, 0, 75]]
lattice = 300
kz = 0

spheres = [treams.TMatrix.sphere(lmax, k0, r, materials) for r in radii]
chain = treams.TMatrix.cluster(spheres, positions).latticeinteraction.solve(lattice, kz)

inc = treams.plane_wave([k0, 0, 0], [0, 0, 1], k0=chain.k0, material=chain.material)
sca = chain @ inc.expand(chain.basis)

x = np.linspace(-150, 150, 5)
y = 0
z = np.linspace(-150, 150, 5)
xx, zz = np.meshgrid(x, z, indexing="ij")
yy = np.full_like(xx, y)
grid = np.stack((xx, yy, zz), axis=-1)

ez = np.zeros_like(xx)
valid = chain.valid_points(grid, radii)
vals = []
for r in grid[valid]:
    swb = treams.SphericalWaveBasis.default(1, positions=[r])
    field = sca.expandlattice(basis=swb).efield(r)
    vals.append(np.real(inc.efield(r)[2] + field[2]))
ez[valid] = vals
ez[~valid] = np.nan

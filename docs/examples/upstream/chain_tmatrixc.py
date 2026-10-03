# Source: https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/chain_tmatrixc.py
# Adapted: reduced sizes, plotting and the copies of neighboring cells removed.
# MIT license, copyright Dominik Beutel; see LICENSE.treams.
import numpy as np
import treams

k0 = 2 * np.pi / 700
materials = [treams.Material(-16.5 + 1j), treams.Material()]
lmax = mmax = 3
radii = [75, 65]
positions = [[-30, 0, -75], [30, 0, 75]]
lattice = treams.Lattice(300)
kz = 0.005

spheres = [treams.TMatrix.sphere(lmax, k0, r, materials) for r in radii]
chain = treams.TMatrix.cluster(spheres, positions).latticeinteraction.solve(lattice, kz)

bmax = 3.1 * lattice.reciprocal
cwb = treams.CylindricalWaveBasis.diffr_orders(kz, mmax, lattice, bmax, 2, positions)
chain_tmc = treams.TMatrixC.from_array(chain, cwb)

inc = treams.plane_wave(
    [np.sqrt(k0**2 - kz**2), 0, kz],
    [np.sqrt(0.5), np.sqrt(0.5)],
    k0=chain.k0,
    material=chain.material,
)
sca = chain_tmc @ inc.expand(chain_tmc.basis)

x = np.linspace(-300, 300, 7)
y = 0
z = np.linspace(-150, 150, 5)
xx, zz = np.meshgrid(x, z, indexing="ij")
yy = np.full_like(xx, y)
grid = np.stack((xx, yy, zz), axis=-1)

ez = np.zeros_like(xx, complex)
valid = chain_tmc.valid_points(grid, radii)
ez[valid] = (inc.efield(grid[valid]) + sca.efield(grid[valid]))[..., 2]
ez[~valid] = np.nan
sca = chain @ inc.expand(chain.basis)
valid = ~valid & chain.valid_points(grid, radii)
vals = []
for r in grid[valid]:
    swb = treams.SphericalWaveBasis.default(1, positions=[r])
    field = sca.expandlattice(basis=swb).efield(r)
    vals.append(inc.efield(r)[2] + field[2])
ez[valid] = vals

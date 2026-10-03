# Source: https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/cluster_tmatrixc.py
# Adapted: reduced sizes, NaN inside the cylinders, plotting and the copies of
# neighboring cells removed.
# MIT license, copyright Dominik Beutel; see LICENSE.treams.
import numpy as np
import treams

k0 = 2 * np.pi / 700
materials = [treams.Material(-16.5 + 1j), treams.Material()]
lmax = mmax = 3
radii = [65, 55]
positions = [[-40, -50, 0], [40, 50, 0]]
lattice = treams.Lattice(300)
kz = 0.005

sphere = treams.TMatrix.sphere(lmax, k0, radii[0], materials)
chain = sphere.latticeinteraction.solve(lattice, kz)
bmax = 3.1 * lattice.reciprocal
cwb = treams.CylindricalWaveBasis.diffr_orders(kz, mmax, lattice, bmax)
chain_tmc = treams.TMatrixC.from_array(chain, cwb)

cylinder = treams.TMatrixC.cylinder(np.unique(cwb.kz), mmax, k0, radii[1], materials)

cluster = treams.TMatrixC.cluster([chain_tmc, cylinder], positions).interaction.solve()
inc = treams.plane_wave(
    [np.sqrt(k0**2 - kz**2), 0, kz],
    [np.sqrt(0.5), np.sqrt(0.5)],
    k0=chain.k0,
    material=chain.material,
)
sca = cluster @ inc.expand(cluster.basis)

x = np.linspace(-300, 300, 7)
y = 0
z = np.linspace(-150, 150, 5)
xx, zz = np.meshgrid(x, z, indexing="ij")
yy = np.full_like(xx, y)
grid = np.stack((xx, yy, zz), axis=-1)

ez = np.zeros_like(xx, complex)
valid = cluster.valid_points(grid, radii)
ez[valid] = (inc.efield(grid[valid]) + sca.efield(grid[valid]))[..., 2]
ez[~valid] = np.nan

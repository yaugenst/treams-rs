# Source: https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/grating_tmatrixc.py
# Adapted: reduced sizes, plotting removed.
# MIT license, copyright Dominik Beutel; see LICENSE.treams.
import numpy as np
import treams

k0 = 2 * np.pi / 700
materials = [treams.Material(-16.5 + 1j), treams.Material()]
lmax = mmax = 3
radii = [65, 55]
positions = [[-40, -50, 0], [40, 50, 0]]
lattice_z = treams.Lattice(300)
lattice_x = treams.Lattice(300, "x")
kz = 0.005

sphere = treams.TMatrix.sphere(lmax, k0, radii[0], materials)
chain = sphere.latticeinteraction.solve(lattice_z, kz)
bmax = 1.1 * lattice_z.reciprocal
cwb = treams.CylindricalWaveBasis.diffr_orders(kz, mmax, lattice_z, bmax)
kzs = np.unique(cwb.kz)
chain_tmc = treams.TMatrixC.from_array(chain, cwb)
cylinder = treams.TMatrixC.cylinder(kzs, mmax, k0, radii[1], materials)

cluster = treams.TMatrixC.cluster(
    [chain_tmc, cylinder], positions
).latticeinteraction.solve(lattice_x, 0)
inc = treams.plane_wave(
    [0, np.sqrt(k0**2 - kz**2), kz],
    [np.sqrt(0.5), -np.sqrt(0.5)],
    k0=chain.k0,
    material=chain.material,
)
sca = cluster @ inc.expand(cluster.basis)

x = 0
y = np.linspace(-150, 150, 4)
z = np.linspace(-150, 150, 4)
yy, zz = np.meshgrid(y, z, indexing="ij")
xx = np.full_like(yy, x)
grid = np.stack((xx, yy, zz), axis=-1)

ex = np.zeros_like(xx)
valid = cluster.valid_points(grid, radii)
vals = []
for r in grid[valid]:
    cwb = treams.CylindricalWaveBasis.default(kzs, 1, positions=[r])
    field = sca.expandlattice(basis=cwb).efield(r)
    vals.append(np.real(inc.efield(r)[0] + field[0]))
ex[valid] = vals
ex[~valid] = np.nan

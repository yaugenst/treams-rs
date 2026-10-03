"""A sphere chain next to a cylinder: the near field of the coupled pair."""

import numpy as np

import treams_rs as tr

# Enlarge for a full map: treams uses a 61 x 31 grid.
GRID_POINTS = (7, 5)

k0 = 2 * np.pi / 700
material = tr.Material(-16.5 + 1j)
lmax = mmax = 3
radii = [65, 55]
positions = [[-40, -50, 0], [40, 50, 0]]
lattice = tr.Lattice(300)
kz = 0.005

# A chain of spheres along z, expressed in cylindrical waves.
sphere = tr.sphere_tmatrix(k0=k0, lmax=lmax, radius=radii[0], material=material)
chain = tr.solve_periodic(sphere, lattice=lattice, kpar=kz)
bmax = 3.1 * lattice.reciprocal
basis = tr.CylindricalBasis.diffr_orders(kz, mmax, lattice, bmax)
chain_tmc = chain.to_cylindrical(basis)

# The cylinder needs the same axial wavenumbers as the chain to couple to it.
cylinder = tr.cylinder_tmatrix(
    k0=k0, kz=np.unique(basis.kz), mmax=mmax, radius=radii[1], material=material
)
cluster = tr.Cluster([chain_tmc, cylinder], positions=positions).solve()

inc = tr.plane_wave(
    direction=[np.sqrt(k0**2 - kz**2), 0, kz],
    pol=[np.sqrt(0.5), np.sqrt(0.5)],
    k0=k0,
)
sca = cluster.scatter(inc)

x = np.linspace(-300, 300, GRID_POINTS[0])
z = np.linspace(-150, 150, GRID_POINTS[1])
xx, zz = np.meshgrid(x, z, indexing="ij")
grid = np.stack((xx, np.zeros_like(xx), zz), axis=-1)

# Points inside either cylinder get NaN: the expansion holds only outside them.
ez = np.full_like(xx, np.nan, dtype=complex)
valid = cluster.valid_points(grid, radii)
ez[valid] = (inc.efield(grid[valid]) + sca.efield(grid[valid]))[..., 2]

if __name__ == "__main__":
    print("Real part of E_z at y = 0 (rows: x, columns: z)")
    for row in ez:
        print("".join(f"{v.real:>13.6g}" for v in row))

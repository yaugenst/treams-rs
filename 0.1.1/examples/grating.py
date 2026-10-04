"""A grating of sphere chains and cylinders: the near field in one period."""

import numpy as np

import treams_rs as tr

# Enlarge for a full map: treams uses a 16 x 16 grid.
GRID_POINTS = 4

k0 = 2 * np.pi / 700
material = tr.Material(-16.5 + 1j)
lmax = mmax = 3
radii = [65, 55]
positions = [[-40, -50, 0], [40, 50, 0]]
lattice_z = tr.Lattice(300)
lattice_x = tr.Lattice(300, "x")
kz = 0.005

# A chain of spheres along z in cylindrical waves, as in the cylindrical
# cluster example, with the axial orders |n| <= 1.1.
sphere = tr.sphere_tmatrix(k0=k0, lmax=lmax, radius=radii[0], material=material)
chain = tr.solve_periodic(sphere, lattice=lattice_z, kpar=kz)
bmax = 1.1 * lattice_z.reciprocal
basis = tr.CylindricalBasis.diffr_orders(kz, mmax, lattice_z, bmax)
kzs = np.unique(basis.kz)
chain_tmc = chain.to_cylindrical(basis)
cylinder = tr.cylinder_tmatrix(
    k0=k0, kz=kzs, mmax=mmax, radius=radii[1], material=material
)

# The chain and the cylinder repeat every 300 nm along x.
grating = tr.solve_periodic(
    tr.Cluster([chain_tmc, cylinder], positions=positions), lattice=lattice_x, kpar=0
)
inc = tr.plane_wave(
    direction=[0, np.sqrt(k0**2 - kz**2), kz],
    pol=[np.sqrt(0.5), -np.sqrt(0.5)],
    k0=k0,
)
sca = grating.scatter(inc)

y = np.linspace(-150, 150, GRID_POINTS)
z = np.linspace(-150, 150, GRID_POINTS)
yy, zz = np.meshgrid(y, z, indexing="ij")
grid = np.stack((np.zeros_like(yy), yy, zz), axis=-1)

# Points inside a cylinder of the reference cell get NaN, as in treams.
distances = np.linalg.norm(grid[..., None, :2] - np.array(positions)[:, :2], axis=-1)
valid = np.all(distances > np.array(radii), axis=-1)

# The waves of all cells, expanded in regular cylindrical waves about each
# point, give the scattered field at that point.
ex = np.full_like(yy, np.nan)
ex[valid] = [
    np.real(
        inc.efield(r)[0]
        + sca.in_basis(
            tr.CylindricalBasis.default(kzs, 1, positions=[r]), kind="regular"
        ).efield(r)[0]
    )
    for r in grid[valid]
]

if __name__ == "__main__":
    print("Real part of E_x at x = 0 (rows: y, columns: z)")
    for row in ex:
        print("".join(f"{v:>13.6g}" for v in row))

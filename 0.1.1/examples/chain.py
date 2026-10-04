"""A chain of sphere pairs, periodic along z: the near field of the array."""

import numpy as np

import treams_rs as tr

# Enlarge for a full map: treams uses a 31 x 31 grid.
GRID_POINTS = 5

k0 = 2 * np.pi / 700
material = tr.Material(-16.5 + 1j)
lmax = 3
radii = [75, 75]
positions = [[-30, 0, -75], [30, 0, 75]]
lattice = 300
kz = 0

spheres = [
    tr.sphere_tmatrix(k0=k0, lmax=lmax, radius=r, material=material) for r in radii
]
# A one-dimensional lattice is periodic along z.
chain = tr.solve_periodic(
    tr.Cluster(spheres, positions=positions), lattice=lattice, kpar=kz
)

inc = tr.plane_wave(direction=[1, 0, 0], pol=[0, 0, 1], k0=k0)
sca = chain.scatter(inc)

x = np.linspace(-150, 150, GRID_POINTS)
z = np.linspace(-150, 150, GRID_POINTS)
xx, zz = np.meshgrid(x, z, indexing="ij")
grid = np.stack((xx, np.zeros_like(xx), zz), axis=-1)

# Points inside a sphere of the reference cell get NaN, as in treams.
distances = np.linalg.norm(grid[..., None, :] - np.array(positions), axis=-1)
valid = np.all(distances > np.array(radii), axis=-1)

# The waves of all cells, expanded in regular dipole waves about each point,
# give the scattered field at that point.
ez = np.full_like(xx, np.nan)
ez[valid] = [
    np.real(
        inc.efield(r)[2]
        + sca.in_basis(
            tr.SphericalBasis.default(1, positions=[r]), kind="regular"
        ).efield(r)[2]
    )
    for r in grid[valid]
]

if __name__ == "__main__":
    print("Real part of E_z at y = 0 (rows: x, columns: z)")
    for row in ez:
        print("".join(f"{v:>13.6g}" for v in row))

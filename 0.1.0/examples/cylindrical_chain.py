"""A chain of sphere pairs along z, described by cylindrical waves: its near field."""

import numpy as np

import treams_rs as tr

# Enlarge for a full map: treams uses a 61 x 31 grid.
GRID_POINTS = (7, 5)

k0 = 2 * np.pi / 700
material = tr.Material(-16.5 + 1j)
lmax = mmax = 3
radii = [75, 65]
positions = [[-30, 0, -75], [30, 0, 75]]
lattice = tr.Lattice(300)
kz = 0.005

spheres = [
    tr.sphere_tmatrix(k0=k0, lmax=lmax, radius=r, material=material) for r in radii
]
chain = tr.solve_periodic(
    tr.Cluster(spheres, positions=positions), lattice=lattice, kpar=kz
)

# Cylindrical waves about one axis through each sphere, for the axial
# wavenumbers kz + 2 pi n / 300 with |n| <= 3.1, the diffraction orders of the
# chain. Each axis collects the waves of its own sphere.
bmax = 3.1 * lattice.reciprocal
basis = tr.CylindricalBasis.diffr_orders(kz, mmax, lattice, bmax, 2, positions)
chain_tmc = chain.to_cylindrical(basis)

inc = tr.plane_wave(
    direction=[np.sqrt(k0**2 - kz**2), 0, kz],
    pol=[np.sqrt(0.5), np.sqrt(0.5)],
    k0=k0,
)

x = np.linspace(-300, 300, GRID_POINTS[0])
z = np.linspace(-150, 150, GRID_POINTS[1])
xx, zz = np.meshgrid(x, z, indexing="ij")
grid = np.stack((xx, np.zeros_like(xx), zz), axis=-1)
sca_cells = chain.scatter(inc)


def lattice_field(points):
    """E_z from the waves of all cells, in regular dipole waves about each point."""
    return np.array(
        [
            inc.efield(r)[2]
            + sca_cells.in_basis(
                tr.SphericalBasis.default(1, positions=[r]), kind="regular"
            ).efield(r)[2]
            for r in points
        ]
    )


# Outside the cylinders that enclose the spheres, the cylindrical waves give
# the field.
ez = np.full_like(xx, np.nan, dtype=complex)
outside = chain_tmc.valid_points(grid, radii)
sca = chain_tmc.scatter(inc)
ez[outside] = (inc.efield(grid[outside]) + sca.efield(grid[outside]))[..., 2]

# Between the spheres, the spherical waves of all cells give the field.
distances = np.linalg.norm(grid[..., None, :] - np.array(positions), axis=-1)
between = ~outside & np.all(distances > np.array(radii), axis=-1)
ez[between] = lattice_field(grid[between])

# Outside the cylinders both descriptions hold: compare them there, row by row.
rows = np.all(outside, axis=1)
difference = np.abs(lattice_field(grid[rows].reshape(-1, 3)) - ez[rows].ravel())
difference = difference.reshape(-1, GRID_POINTS[1]).max(axis=1)

if __name__ == "__main__":
    print("Real part of E_z at y = 0 (rows: x, columns: z)")
    for row in ez:
        print("".join(f"{v.real:>13.6g}" for v in row))
    print()
    print("Largest |E_z| difference of the two descriptions outside the cylinders")
    for xi, value in zip(x[rows], difference, strict=True):
        print(f"x = {xi:>6.0f} nm: {value:.2e}")

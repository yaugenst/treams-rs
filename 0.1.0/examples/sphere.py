"""Extinction and scattering spectrum of a sphere and the intensity around it."""

import numpy as np

import treams_rs as tr

# Enlarge these for full sweeps: treams uses 200 frequencies and a 101 x 101 grid.
FREQUENCIES = 6
GRID_POINTS = 7

k0s = 2 * np.pi * np.linspace(1 / 700, 1 / 300, FREQUENCIES)
material = tr.Material(16 + 0.5j)
lmax = 4
radius = 75
spheres = [
    tr.sphere_tmatrix(k0=k0, lmax=lmax, radius=radius, material=material) for k0 in k0s
]

# Efficiencies are cross sections divided by the geometric cross section.
area = np.pi * radius**2
averages = [tm.average_cross_sections for tm in spheres]
xs_sca = np.array([avg.scattering for avg in averages]) / area
xs_ext = np.array([avg.extinction for avg in averages]) / area

# Keep only the dipole modes (l = 1) of the same T-matrices.
dipoles = tr.SphericalBasis.default(1)
averages_lmax1 = [tm.select(dipoles).average_cross_sections for tm in spheres]
xs_sca_lmax1 = np.array([avg.scattering for avg in averages_lmax1]) / area
xs_ext_lmax1 = np.array([avg.extinction for avg in averages_lmax1]) / area

# Total field of a plane wave at the highest frequency, in the plane y = 0.
tm = spheres[-1]
inc = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=tm.k0)
sca = tm.scatter(inc)

x = np.linspace(-100, 100, GRID_POINTS)
z = np.linspace(-100, 100, GRID_POINTS)
xx, zz = np.meshgrid(x, z, indexing="ij")
grid = np.stack((xx, np.zeros_like(xx), zz), axis=-1)

# Points inside the sphere get NaN: the expansion holds only outside it.
intensity = np.full_like(xx, np.nan)
valid = tm.valid_points(grid, [radius])
total = inc.efield(grid[valid]) + sca.efield(grid[valid])
intensity[valid] = 0.5 * np.sum(np.abs(total) ** 2, axis=-1)

if __name__ == "__main__":
    print("Efficiencies (cross section / geometric cross section)")
    headers = ("frequency/THz", "extinction", "scattering", "ext. l=1", "sca. l=1")
    print("".join(f"{h:>15}" for h in headers))
    frequencies = k0s * 299792.458 / (2 * np.pi)
    columns = (frequencies, xs_ext, xs_sca, xs_ext_lmax1, xs_sca_lmax1)
    for row in zip(*columns, strict=True):
        print("".join(f"{v:>15.6g}" for v in row))
    print("\nIntensity at y = 0 (rows: x, columns: z)")
    for row in intensity:
        print("".join(f"{v:>11.6g}" for v in row))

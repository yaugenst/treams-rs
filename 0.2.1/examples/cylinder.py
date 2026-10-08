"""Cross widths of an infinite cylinder and the intensity around it."""

import numpy as np

import treams_rs as tr

# Enlarge these for full sweeps: treams uses 200 frequencies and a 101 x 101 grid.
FREQUENCIES = 6
GRID_POINTS = 7

k0s = 2 * np.pi * np.linspace(1 / 6000, 1 / 300, FREQUENCIES)
material = tr.Material(16 + 0.5j)
mmax = 4
radius = 75
# Light that travels in the xy plane has the axial wavenumber kz = 0.
kzs = [0]
cylinders = [
    tr.cylinder_tmatrix(k0=k0, kz=kzs, mmax=mmax, radius=radius, material=material)
    for k0 in k0s
]

# Efficiencies are cross widths divided by the diameter.
averages = [tm.average_cross_widths for tm in cylinders]
xw_sca = np.array([avg.scattering for avg in averages]) / (2 * radius)
xw_ext = np.array([avg.extinction for avg in averages]) / (2 * radius)

# Keep only the modes of order m = 0 of the same T-matrices.
order_zero = tr.CylindricalBasis.default(kzs, 0)
averages_mmax0 = [tm.select(order_zero).average_cross_widths for tm in cylinders]
xw_sca_mmax0 = np.array([avg.scattering for avg in averages_mmax0]) / (2 * radius)
xw_ext_mmax0 = np.array([avg.extinction for avg in averages_mmax0]) / (2 * radius)

# Total field of a plane wave at the highest frequency, in the plane z = 0.
tm = cylinders[-1]
inc = tr.plane_wave(direction=[1, 0, 0], pol="positive_helicity", k0=tm.k0)
sca = tm.scatter(inc)

x = np.linspace(-100, 100, GRID_POINTS)
y = np.linspace(-100, 100, GRID_POINTS)
xx, yy = np.meshgrid(x, y, indexing="ij")
grid = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)

# Points inside the cylinder get NaN: the expansion holds only outside it.
intensity = np.full_like(xx, np.nan)
valid = tm.valid_points(grid, [radius])
total = inc.efield(grid[valid]) + sca.efield(grid[valid])
intensity[valid] = 0.5 * np.sum(np.abs(total) ** 2, axis=-1)

if __name__ == "__main__":
    print("Efficiencies (cross width / diameter)")
    headers = ("frequency/THz", "extinction", "scattering", "ext. m=0", "sca. m=0")
    print("".join(f"{h:>15}" for h in headers))
    frequencies = k0s * 299792.458 / (2 * np.pi)
    columns = (frequencies, xw_ext, xw_sca, xw_ext_mmax0, xw_sca_mmax0)
    for row in zip(*columns, strict=True):
        print("".join(f"{v:>15.6g}" for v in row))
    print("\nIntensity at z = 0 (rows: x, columns: y)")
    for row in intensity:
        print("".join(f"{v:>11.6g}" for v in row))

"""Four spheres coupled by multiple scattering: cross sections and near fields."""

import numpy as np

import treams_rs as tr

# Enlarge for a full map: treams uses a 101 x 101 grid.
GRID_POINTS = 7

k0 = 2 * np.pi / 1000
material = tr.Material(16 + 0.5j)
lmax = 3
radii = [110, 90, 80, 75]
positions = (220 / np.sqrt(24)) * np.array(
    [
        [np.sqrt(8), 0, -1],
        [-np.sqrt(2), np.sqrt(6), -1],
        [-np.sqrt(2), -np.sqrt(6), -1],
        [0, 0, 3],
    ]
)

spheres = [
    tr.sphere_tmatrix(k0=k0, lmax=lmax, radius=r, material=material) for r in radii
]
# Cluster.solve couples the spheres and returns one T-matrix with an
# expansion center at every sphere.
tm = tr.Cluster(spheres, positions=positions).solve()

inc = tr.plane_wave(direction=[1, 0, 0], pol="negative_helicity", k0=k0)
xs = tm.cross_sections(inc)

x = np.linspace(-300, 300, GRID_POINTS)
z = np.linspace(-300, 300, GRID_POINTS)
xx, zz = np.meshgrid(x, z, indexing="ij")
grid = np.stack((xx, np.zeros_like(xx), zz), axis=-1)


def intensity_map(tm, inc, radii):
    """Total-field intensity on the grid; NaN where the expansion does not hold."""
    sca = tm.scatter(inc)
    intensity = np.full_like(xx, np.nan)
    valid = tm.valid_points(grid, radii)
    total = inc.efield(grid[valid]) + sca.efield(grid[valid])
    intensity[valid] = 0.5 * np.sum(np.abs(total) ** 2, axis=-1)
    return intensity


intensity = intensity_map(tm, inc, radii)

# One expansion about the origin up to l = 6. Its fields hold only outside
# the sphere of radius 260 that encloses the cluster.
tm_global = tm.in_basis(tr.SphericalBasis.default(6))
xs_global = tm_global.cross_sections(inc)
intensity_global = intensity_map(tm_global, inc, [260])

# Rotating the cluster by 90 degrees about y and illuminating it along -z
# reproduces the first illumination in the rotated frame.
inc_rotate = tr.plane_wave(direction=[0, 0, -1], pol="negative_helicity", k0=k0)
tm_rotate = tm_global.rotate(0, np.pi / 2)
xs_rotate = tm_rotate.cross_sections(inc_rotate)
intensity_rotate = intensity_map(tm_rotate, inc_rotate, [260])

if __name__ == "__main__":
    print("Cross sections (nm^2)")
    headers = ("expansion", "scattering", "extinction", "absorption")
    print("".join(f"{h:>13}" for h in headers))
    for label, result in [("local", xs), ("global", xs_global), ("rotated", xs_rotate)]:
        print(
            f"{label:>13}"
            + "".join(f"{v:>13.6g}" for v in (*result, result.absorption))
        )
    maps = [
        ("local", intensity),
        ("global", intensity_global),
        ("rotated", intensity_rotate),
    ]
    for label, values in maps:
        print(f"\nIntensity at y = 0, {label} expansion (rows: x, columns: z)")
        for row in values:
            print("".join(f"{v:>11.6g}" for v in row))

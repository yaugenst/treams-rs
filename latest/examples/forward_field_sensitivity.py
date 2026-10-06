# Requires: jax
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.1,<3", "jax>=0.10,<0.12", "treams-rs"]
# ///
"""How does the scattered field change with wavelength and sphere radius?"""

import jax
import numpy as np

import treams_rs as tr

# Keep double precision for the comparison with small finite differences.
jax.config.update("jax_enable_x64", True)

GRID_POINTS = 5


def make_problem(grid_points=GRID_POINTS):
    """Return the field calculation and starting [k0 (1/nm), radius (nm)]."""
    positions = np.array([[-160, 0, 0], [160, 0, 0], [0, 250, 0]], dtype=float)
    x, y = np.meshgrid(
        np.linspace(-320, 320, grid_points),
        np.linspace(-200, 400, grid_points),
        indexing="ij",
    )
    # Every observation point is outside the spheres, on a plane 350 nm above them.
    points = np.stack((x, y, np.full_like(x, 350)), axis=-1)
    initial = np.array([2 * np.pi / 650, 80.0])

    def field_map(parameters):
        k0, radius = parameters
        sphere = tr.sphere_tmatrix(k0=k0, lmax=2, radius=radius, material=3.2 + 0.05j)
        cluster = tr.Cluster([sphere, sphere, sphere], positions=positions)
        incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
        return cluster.scatter(incident).efield(points)

    return field_map, initial


def main():
    field_map, parameters = make_problem()
    field = np.asarray(field_map(parameters))
    # The last axis labels k0 and radius. Every grid point and electric-field
    # component gets both derivatives in this one call.
    derivatives = np.asarray(jax.jacfwd(field_map)(parameters))

    # Changing wavelength changes k0 = 2*pi/wavelength.
    wavelength, radius = 2 * np.pi / parameters[0], parameters[1]
    wavelength_derivative = derivatives[..., 0] * (-2 * np.pi / wavelength**2)
    radius_derivative = derivatives[..., 1]

    # Check both entire derivative maps, using steps in their own units.
    for parameter, step in enumerate((1e-7, 1e-3)):
        shift = np.eye(2)[parameter] * step
        difference = (
            np.asarray(field_map(parameters + shift))
            - np.asarray(field_map(parameters - shift))
        ) / (2 * step)
        np.testing.assert_allclose(
            derivatives[..., parameter], difference, rtol=2e-6, atol=1e-9
        )

    center = GRID_POINTS // 2
    print(f"Three spheres: radius {radius:g} nm, wavelength {wavelength:g} nm")
    print(f"Field map: {GRID_POINTS} x {GRID_POINTS} points, three complex components")
    print("At the middle grid point, scattered electric field and changes per nm")
    print(
        f"{'E':>3} {'field (real, imag)':>23} {'d/d wavelength':>23} {'d/d radius':>23}"
    )
    for component, label in enumerate("xyz"):
        values = (
            field[center, center, component],
            wavelength_derivative[center, center, component],
            radius_derivative[center, center, component],
        )
        print(
            f"{label:>3} "
            + " ".join(f"{value.real:>11.4g}{value.imag:>11.4g}j" for value in values)
        )
    print("Both complete derivative maps agree with central differences.")


if __name__ == "__main__":
    main()

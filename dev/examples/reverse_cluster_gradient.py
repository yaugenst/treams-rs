# Requires: jax
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.1,<3", "jax>=0.10,<0.12", "treams-rs"]
# ///
"""Which sphere radii or positions increase the scattering cross section?"""

import jax
import jax.numpy as jnp
import numpy as np

import treams_rs as tr

# Keep double precision for the comparison with small finite differences.
jax.config.update("jax_enable_x64", True)


def make_problem(particles=8):
    """Return a scalar objective and (radius, x, y, z) parameters, all in nm."""
    # A slightly irregular stack of squares, with separated dielectric spheres.
    index = np.arange(particles)
    positions = np.column_stack(
        (180 * (index % 2), 190 * ((index // 2) % 2), 200 * (index // 4))
    ).astype(float)
    positions += np.column_stack(
        (9 * np.sin(index), 7 * np.cos(index), 11 * np.sin(2 * index))
    )
    positions -= positions.mean(axis=0)
    initial = np.column_stack((45 + 3 * (index % 4), positions))

    k0 = 2 * np.pi / 700  # vacuum wavelength: 700 nm
    incident = tr.plane_wave([0.2, -0.1, 1], "positive_helicity", k0=k0)

    def scattering(parameters):
        """Cluster scattering cross section in nm², for fixed illumination."""
        # Use the same calculation for the gradient and the difference check.
        parameters = jnp.asarray(parameters)
        spheres = [
            tr.sphere_tmatrix(
                k0=k0, lmax=1, radius=parameters[i, 0], material=4 + 0.02j
            )
            for i in range(particles)
        ]
        response = tr.Cluster(spheres, positions=parameters[:, 1:]).solve()
        return response.cross_sections(incident).scattering

    return scattering, initial


def main():
    scattering, initial = make_problem()
    value, gradient = jax.value_and_grad(scattering)(jnp.asarray(initial))
    gradient = np.asarray(gradient)

    # Perturb every radius and position together, along a unit direction.
    direction = np.sin(np.arange(initial.size) + 0.5).reshape(initial.shape)
    direction /= np.linalg.norm(direction)
    step = 1e-3  # nm
    difference = (
        scattering(initial + step * direction) - scattering(initial - step * direction)
    ) / (2 * step)
    projection = np.sum(gradient * direction)
    np.testing.assert_allclose(projection, difference, rtol=1e-6, atol=1e-9)

    print(f"Scattering cross section: {float(value):.8f} nm^2")
    print(f"One reverse pass: {initial.size} radius and position derivatives")
    print("Gradient in nm (cross section in nm^2, parameters in nm)")
    print(f"{'sphere':>7}{'radius':>13}{'x':>13}{'y':>13}{'z':>13}")
    for i, row in enumerate(gradient):
        print(f"{i:>7}" + "".join(f"{entry:>13.6f}" for entry in row))
    print(f"Joint directional derivative: {projection:.8f} nm")
    print(f"Central difference:           {float(difference):.8f} nm")


if __name__ == "__main__":
    main()

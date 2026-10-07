# Requires: jax
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.1,<3", "jax>=0.10,<0.12", "treams-rs"]
# ///
"""Which sphere radii explain the intensities measured around a cluster?"""

import jax
import jax.numpy as jnp
import numpy as np

import treams_rs as tr

# Keep double precision for the comparison with small finite differences.
jax.config.update("jax_enable_x64", True)

STEPS = 6


def make_problem(particles=8, detectors_per_cone=24):
    """Return the residuals of the fit, the true radii and the starting radii in nm."""
    # An irregular ring of dielectric spheres; their positions are known.
    index = np.arange(particles)
    angle = 2 * np.pi * index / particles
    positions = np.column_stack(
        (
            220 * np.cos(angle) + 30 * np.sin(3 * index),
            220 * np.sin(angle) + 25 * np.cos(5 * index),
            40 * np.sin(2 * index),
        )
    )
    true_radii = 50 + 8 * np.sin(1.7 * index + 0.3)

    # Two cones of detectors 5 um away, at 60 and 125 degrees from the +z axis.
    polar, azimuth = np.meshgrid(
        np.radians([60.0, 125.0]),
        np.linspace(0, 2 * np.pi, detectors_per_cone, endpoint=False),
        indexing="ij",
    )
    sin = np.sin(polar)
    directions = np.stack((sin * np.cos(azimuth), sin * np.sin(azimuth), np.cos(polar)))
    points = 5000 * directions.reshape(3, -1).T
    k0 = 2 * np.pi / 600  # vacuum wavelength: 600 nm
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)

    @jax.jit
    def intensities(radii):
        """Scattered intensity |E|^2 at every detector."""
        spheres = [
            tr.sphere_tmatrix(k0=k0, lmax=1, radius=radii[i], material=6 + 0.05j)
            for i in range(particles)
        ]
        cluster = tr.Cluster(spheres, positions=positions)
        field = cluster.scatter(incident).efield(points)
        return jnp.sum(jnp.abs(field) ** 2, axis=-1)

    # Synthetic, noise-free measurements of the true cluster.
    measured = np.asarray(intensities(jnp.asarray(true_radii)))

    def residuals(radii):
        """Relative misfit of every predicted intensity."""
        return intensities(radii) / measured - 1

    return residuals, true_radii, np.full(particles, 50.0)


def gauss_newton_operator(residuals, radii):
    """Return the residuals r, the gradient J^T r and v -> J^T J v at radii."""
    # One native forward calculation; its saved contexts serve every product.
    value, forward = jax.linearize(residuals, jnp.asarray(radii))
    reverse = jax.linear_transpose(forward, jnp.asarray(radii))

    def product(v):
        # Forward mode carries v to the intensities, J v. Reverse mode carries
        # that change back to the radii, J^T (J v). J itself is never formed.
        (result,) = reverse(forward(jnp.asarray(v)))
        return np.asarray(result)

    (gradient,) = reverse(value)
    return np.asarray(value), np.asarray(gradient), product


def conjugate_gradient(product, right, rtol=1e-7):
    """Solve product(x) = right for a symmetric positive definite product."""
    x = np.zeros_like(right)
    residual = right.copy()
    direction = residual.copy()
    norm = residual @ residual
    for count in range(1, 4 * right.size + 1):
        image = product(direction)
        alpha = norm / (direction @ image)
        x += alpha * direction
        residual -= alpha * image
        new_norm = residual @ residual
        if np.sqrt(new_norm) <= rtol * np.linalg.norm(right):
            return x, count
        direction = residual + (new_norm / norm) * direction
        norm = new_norm
    raise RuntimeError("conjugate gradients did not converge")


def main():
    residuals, true_radii, radii = make_problem()

    # Check J^T J v at the start against a Jacobian from central differences.
    _, _, product = gauss_newton_operator(residuals, radii)
    step = 1e-3  # nm
    columns = []
    for e in np.eye(radii.size):
        plus, minus = residuals(radii + step * e), residuals(radii - step * e)
        columns.append(np.asarray(plus - minus) / (2 * step))
    jacobian = np.column_stack(columns)
    direction = np.sin(np.arange(radii.size) + 0.5)
    np.testing.assert_allclose(
        product(direction), jacobian.T @ (jacobian @ direction), rtol=1e-6
    )

    print(f"Fit {radii.size} sphere radii to {len(jacobian)} detector intensities")
    print(f"{'step':>5}{'misfit':>14}{'largest error/nm':>18}{'products':>10}")
    for i in range(STEPS):
        value, gradient, product = gauss_newton_operator(residuals, radii)
        # The Gauss-Newton step s solves (J^T J) s = -J^T r, using products only.
        update, count = conjugate_gradient(product, -gradient)
        misfit = 0.5 * value @ value
        error = np.abs(radii - true_radii).max()
        print(f"{i:>5}{misfit:>14.6e}{error:>18.6e}{count:>10}")
        radii = radii + update

    np.testing.assert_allclose(radii, true_radii, rtol=0, atol=1e-9)
    print(f"After {STEPS} steps every radius is within 1e-9 nm of the true radius.")
    print("Fitted radii in nm:", " ".join(f"{radius:.6f}" for radius in radii))


if __name__ == "__main__":
    main()

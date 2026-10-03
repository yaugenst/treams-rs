# Requires: advect
"""Gradient ascent on the radius of a dielectric sphere to maximize its scattering."""

import advect
import numpy as np

import treams_rs.advect as tr
from treams_rs.testing import check_gradient

STEPS = 5

# Lengths are in nm: light of 600 nm wavelength meets a sphere of refractive index 2.
k0 = 2 * np.pi / 600
material = 4  # permittivity 4, refractive index 2
lmax = 6
start = 190.0
# The gradient has units of 1/nm, so the step size has units of nm^2.
step_size = 50.0


def efficiency(radius):
    """Scattering cross section divided by the geometric cross section."""
    sphere = tr.sphere_tmatrix(k0=k0, lmax=lmax, radius=radius, material=material)
    # A sphere scatters every plane wave alike, so one wave gives the average.
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
    return sphere.cross_sections(incident).scattering / (np.pi * radius**2)


# Compare the gradient with central differences before using it.
check_gradient(
    efficiency, advect.grad(efficiency), np.asarray(start), step=1e-3, rtol=1e-6, atol=0
)

# Rows hold (radius, efficiency, gradient) before each step and after the last.
history = np.zeros((STEPS + 1, 3))
radius = start
for i in range(STEPS + 1):
    value, gradient = advect.value_and_grad(efficiency)(np.asarray(radius))
    history[i] = radius, value, gradient
    radius = radius + step_size * gradient

if __name__ == "__main__":
    print("Gradient ascent on the scattering efficiency Q")
    print(f"{'step':>6}{'radius/nm':>14}{'Q':>14}{'dQ/dr in 1/nm':>16}")
    for i, (r, q, dq) in enumerate(history):
        print(f"{i:>6}{r:>14.8g}{q:>14.8g}{dq:>16.6g}")

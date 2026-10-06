"""A chiral slab between vacuum and a dense medium: transmission and reflection."""

import numpy as np

import treams_rs as tr

# Enlarge for a full spectrum: treams uses 50 frequencies.
FREQUENCIES = 6

k0s = 2 * np.pi * np.linspace(1 / 1000, 1 / 300, FREQUENCIES)
material = tr.Material(12.4 + 1j, 1 + 0.1j, 0.5 + 0.05j)
thickness = 50

# power[0] holds (T, R) for positive helicity, power[1] for negative helicity.
power = np.zeros((2, FREQUENCIES, 2))
for i, k0 in enumerate(k0s):
    # One direction with k_y = k0 / 2; its channels list positive helicity first.
    ports = tr.PlaneWavePorts.default([0, 0.5 * k0])
    slab = tr.slab(
        k0=k0,
        basis=ports,
        thickness=thickness,
        material=material,
        negative_medium=1,
        positive_medium=tr.Material(2, 2),
    )
    # Amplitudes per channel: light enters from the vacuum side.
    power[0, i] = slab.power([1, 0])
    power[1, i] = slab.power([0, 1])

if __name__ == "__main__":
    print("Transmission T and reflection R for each helicity")
    print(f"{'k0/(2 pi)':>13}{'T+':>13}{'R+':>13}{'T-':>13}{'R-':>13}")
    for k0, (plus, minus) in zip(k0s, power.transpose(1, 0, 2), strict=True):
        values = "".join(f"{v:>13.6g}" for v in (*plus, *minus))
        print(f"{k0 / (2 * np.pi):>13.6g}{values}")

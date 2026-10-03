# Planar layers and S-matrices

An `SMatrix` is the scattering matrix of a planar two-port network: it maps the
waves arriving at both sides to the waves leaving both sides. The sides are
named after the z axis of the ports' basis:

- the **negative** side, below (z < 0), with medium `negative_medium`;
- the **positive** side, above, with medium `positive_medium`.

!!! note "SMatrix in treams"

    treams calls this whole network `SMatrices`, and calls one of its four
    blocks an `SMatrix`. In treams-rs one block is a `ScatteringBlock`. See
    [Coming from treams](../coming-from-treams/index.md).

## Building a network

| Function | Builds |
| --- | --- |
| `interface(basis=, k0=, negative_medium=, positive_medium=)` | one flat interface |
| `slab(basis=, k0=, thickness=, material=)` | one homogeneous layer |
| `multilayer_slab(basis=, k0=, thicknesses=, materials=)` | layers from the negative to the positive side |
| `propagation(distance=, basis=, k0=)` | a gap of homogeneous medium |
| `PeriodicResponse.to_smatrix(ports)` | a periodic array; see [Periodic arrays](periodic.md) |
| `stack([first, second, ...])` | the networks in order from the negative to the positive side |

The basis is a `PlaneWavePorts`: `PlaneWavePorts.default([0, 0])` holds the
two pol values at normal incidence; `PlaneWavePorts.diffr_orders` holds the
diffraction orders of a lattice.

```python
import numpy as np
import treams_rs as tr

ports = tr.PlaneWavePorts.default([0, 0])
glass = tr.interface(basis=ports, k0=1.0, negative_medium=1, positive_medium=2.25)
incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.0)
power = glass.power(incident)
# Fresnel reflectance at normal incidence from n = 1 to n = 1.5.
np.testing.assert_allclose(power.reflection, ((1 - 1.5) / (1 + 1.5)) ** 2)
two = tr.multilayer_slab(
    basis=ports, k0=1.0, thicknesses=[0.3, 0.4], materials=[2.25, 4]
)
layers = [
    tr.slab(basis=ports, k0=1.0, thickness=d, material=e)
    for d, e in ((0.3, 2.25), (0.4, 4))
]
np.testing.assert_allclose(two.array, tr.stack(layers).array, atol=1e-15)
```

## Results

| Method | Returns |
| --- | --- |
| `power(incident)` | `PowerBalance(transmission, reflection)` as fractions of the incident power |
| `scatter(negative=..., positive=...)` | `ScatteredPorts(negative, positive)`: the waves leaving each side |
| `block("positive", "negative")` | one `ScatteringBlock`: the waves leaving the first side for waves arriving from the second; this one is the transmission upwards |
| `circular_dichroism(incident)` | the contrasts of transmission and total outgoing power between the two helicities |
| `bands(period=...)` | `BandModes(wavenumbers, eigenvectors)` of the network repeated along z |
| `transfer_matrix()` | the transfer matrix of one period |

`power` and `scatter` read the incident side from the direction of the wave: a
wave with a positive z component arrives from the negative side. A wave from
above uses `direction=[0, 0, -1]` and the positive medium; the layer order stays
the same.

```python
import numpy as np
import treams_rs as tr

ports = tr.PlaneWavePorts.default([0, 0])
glass = tr.interface(basis=ports, k0=1.0, negative_medium=1, positive_medium=2.25)
transmission = np.asarray(glass.block("positive", "negative"))
# The field transmission coefficient 2 n1 / (n1 + n2), the same for both pols.
np.testing.assert_allclose(transmission, 0.8 * np.eye(2), atol=1e-15)
from_above = tr.plane_wave(
    direction=[0, 0, -1], pol="positive_helicity", k0=1.0, medium=2.25
)
np.testing.assert_allclose(glass.power(from_above).reflection, 0.04)
layer = tr.slab(basis=ports, k0=1.0, thickness=0.5, material=2.25)
bands = layer.bands(period=0.5)
# A period filled with one material carries waves with kz = +-n k0 = +-1.5.
np.testing.assert_allclose(np.sort(bands.wavenumbers.real), [-1.5, -1.5, 1.5, 1.5])
```

`cascade` joins two networks like `stack` does for a list. The treams method
names `add`, `illuminate`, `tr`, `cd`, `periodic` and `bands_kz` call these
methods; the [name map](../coming-from-treams/names.md) lists them.

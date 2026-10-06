# Changes across a field map

How does the scattered field above three spheres change when all three grow
slightly, or when the illumination wavelength changes? This example computes
both changes at every point of a 5 × 5 grid. It keeps the real and imaginary
parts of all three electric-field components, so changes in phase remain
available as well as changes in amplitude.

The spheres have radius 80 nm and permittivity 3.2 + 0.05j. A circularly
polarized plane wave of wavelength 650 nm illuminates them along +z, and the
grid lies 350 nm above their centers. Only two inputs vary: the common radius
and the vacuum wavenumber `k0 = 2*pi/wavelength`. The script converts the latter
derivative into a change per nanometer of wavelength for the printed table.

`jax.jacfwd(field_map)` asks for every output derivative with respect to both
inputs. This is **forward mode**: follow an input change through the
calculation to every field value. With two inputs and many outputs, only two
input directions are needed. The returned array has shape `(5, 5, 3, 2)`:
grid positions, electric-field component, and input parameter `(k0, radius)`.

<!-- fmt: off -->

```python
--8<-- "docs/examples/forward_field_sensitivity.py"
```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/forward_field_sensitivity.txt"
```

## Using the result

`derivatives[..., 1]` tells you how every complex field value changes per
nanometer added to all three sphere radii. For a small radius change `dr`,
`field + derivatives[..., 1] * dr` is its first-order prediction. The example
checks both entire derivative maps against central differences before
printing a few entries.

The incident wave and the spheres receive the same changing `k0`, so the
wavelength derivative includes both the excitation and the scattering.
The permittivity stays fixed; this example does not include material dispersion.
Lengths are in nanometers, the grid stays fixed, and the displayed field is
the **scattered** field. Add `incident.efield(points)` inside `field_map` if
you want the total field instead.

The small grid and `lmax=2` keep the example quick. Increase the grid density
and check convergence in `lmax` for a calculation you intend to use. The
derivative checks establish agreement for this chosen discretization.

When many particle parameters control a single quantity, the
[cluster gradient example](reverse_cluster_gradient.md) uses reverse mode.
[Choosing a differentiation mode](../differentiation/choosing-a-mode.md)
compares both modes on these same physical questions, including a runnable
timing comparison.

---
description: How changing eight sphere radii and their positions affects one scattering cross section, with a gradient from JAX.
---

# Cluster scattering gradient

Which sphere radii or positions would increase the scattering cross section?
Eight dielectric spheres scatter a circularly polarized plane wave at a vacuum
wavelength of 700 nm. Their radii range from 45 to 54 nm; their relative
permittivity is `4 + 0.02j`. The objective is the scattering cross section:
the scattered power divided by the incident intensity, in nm².

`jax.value_and_grad` computes this scalar and its derivatives with respect to
all 32 parameters: one radius and three position coordinates per sphere.
This is a natural **reverse-mode** problem: start with the one output of
interest and work back to every input. Forward mode would follow 32 input
directions to recover the same full gradient.

The script also changes all the radii and positions slightly, then checks
that the gradient predicts the resulting change in cross section. The
prediction and a central difference agree to `1e-6` relative tolerance.

<!-- fmt: off -->

```python no-exec
--8<-- "docs/examples/reverse_cluster_gradient.py"
```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/reverse_cluster_gradient.txt"
```

## How it works

- Rows of `parameters` hold `(radius, x, y, z)` in nm. JAX tracks the radius
  through `sphere_tmatrix` and the positions through `Cluster`, including
  multiple scattering, the incident phase, and interference in the outgoing
  power.
- `Cluster.solve()` constructs the coupled T-matrix; `cross_sections` turns it
  into one scattering cross section. Its gradient includes repeated scattering
  between the spheres.
- `jax_enable_x64` retains double precision for the finite-difference check.
  The returned gradient has units of nm: cross section in nm² differentiated
  with respect to a length in nm.
- The material, incident wave, and multipole truncation remain fixed. The
  small `lmax=1` dipole model keeps the example compact. Check convergence in
  `lmax` before using this calculation for quantitative design; derivative
  agreement only validates differentiation of the chosen model.

A positive `gradient[i, 0]` means that slightly increasing sphere `i`'s radius
raises the cross section, with every other parameter held fixed. The other
three entries describe translations along x, y and z. For a proposed small
change to all the parameters, `(gradient * change).sum()` predicts the change
in cross section. The central-difference check tests exactly this prediction.

The [field sensitivity example](forward_field_sensitivity.md) asks for changes
across a whole map and uses forward mode. [Choosing a differentiation
mode](../differentiation/choosing-a-mode.md) compares the times to obtain the
same result with either mode. See
[framework adapters](../differentiation/frameworks.md) for the differentiation
interfaces and [gradient checks](../differentiation/gradient-checks.md) for more
validation methods.

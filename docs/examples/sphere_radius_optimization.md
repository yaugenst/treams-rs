---
description: Five gradient-ascent steps on the radius of a lossless dielectric sphere maximize its scattering efficiency.
---

# Sphere radius optimization

A lossless sphere of refractive index 2 sits in vacuum, lit at a wavelength of
600 nm. Its scattering efficiency Q, the scattering cross section divided by
π r², depends on the radius r. Starting at r = 190 nm, five steps of gradient
ascent, r → r + 50 nm² · dQ/dr, reach the maximum Q = 5.759 at r = 201.75 nm.
The derivative dQ/dr comes from Advect through the analytic derivatives of
treams-rs; before the first step, the script checks it against central
differences. treams computes no derivatives, so this example has no treams
version.

<!-- fmt: off -->

```python no-exec
--8<-- "docs/examples/sphere_radius_optimization.py"
```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/sphere_radius_optimization.txt"
```

## How it works

- `treams_rs` selects Advect from the traced radius, so
  `advect.value_and_grad(efficiency)` returns Q and dQ/dr. The constant incident
  wave composes with the differentiated sphere automatically.
  [Framework adapters](../differentiation/frameworks.md) lists the JAX, PyTorch
  and HIPS Autograd equivalents.
- Rust computes each derivative analytically: a pullback maps the gradient with
  respect to an output, here the cross section, back to the inputs, here the
  radius. [Differentiation](../differentiation/index.md) describes these
  pullbacks.
- `check_gradient` compares dQ/dr with central differences of step 1e-3 nm
  in a random direction and raises an error when they differ by more than
  1e-6 relative; here they differ by about 1e-9. [Gradient checks](../differentiation/gradient-checks.md)
  lists its options.

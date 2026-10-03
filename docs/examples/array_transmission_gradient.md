---
description: jax.grad of the transmission of a sphere array on a slab with respect to the sphere radius and slab thickness, next to central differences.
---

# Array transmission gradient

The structure is that of the [array example](array_spheres.md): a square
array of chiral spheres with a 500 nm period on a 10 nm slab of permittivity 3.
A circularly polarized plane wave of 400 nm wavelength comes from below, with
k_y = 0.3 k0. Its helicity, the sign of its spin along its direction of
travel, is positive. The script computes the transmitted power fraction T = 0.844 and
its derivatives with respect to the sphere radius and the slab thickness with
`jax.grad`. Central differences with a step of 1e-3 nm agree with both
derivatives to 8 digits; the script stops with an error if they differ by more
than 1e-6 relative. treams computes no derivatives, so this example has no
treams version.

<!-- fmt: off -->

```python no-exec
--8<-- "docs/examples/array_transmission_gradient.py"
```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/array_transmission_gradient.txt"
```

## How it works

- `treams_rs` selects JAX from the traced parameters, so `jax.grad`
  differentiates the whole chain: the
  sphere T-matrix, the lattice sum, the slab, the propagation between them and
  the transmitted power. [Framework adapters](../differentiation/frameworks.md)
  lists what each framework supports.
- The radius enters twice: through the sphere T-matrix and through the gap
  between the slab and the plane of sphere centers.
- This example enables `jax_enable_x64` to retain double precision when
  comparing small finite differences. JAX's default single precision also
  works, with lower-precision outputs and gradients.
- The lattice, the Bloch vector `kpar`, the ports and `lmax` stay fixed.
  To differentiate with respect to the lattice, convert with integer
  diffraction orders, as described in
  [periodic geometry](../differentiation/frameworks.md#periodic-geometry).
- Rust computes each derivative analytically: a pullback maps the gradient with
  respect to an output back to the inputs.
  [Differentiation](../differentiation/index.md) describes these pullbacks, and
  [gradient checks](../differentiation/gradient-checks.md) shows how to compare
  them with finite differences.

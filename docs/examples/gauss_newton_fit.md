---
description: Fit eight sphere radii to measured intensities with Gauss–Newton steps built from forward- and reverse-mode products.
---

# Gauss–Newton fit

Which sphere radii explain the measured scattering? Eight dielectric spheres
sit at known positions, for example from an electron micrograph. Their radii
are unknown. A circularly polarized plane wave of vacuum wavelength 600 nm
illuminates them along +z, and 48 detectors 5 µm away record the scattered
intensity on two cones, 60° and 125° from the +z axis. The script fits the
radii to these intensities.

The fit minimizes the misfit `f = ½ Σ r_i²`, where each residual `r_i` is the
relative difference between a predicted and a measured intensity. The Hessian
of `f` is `JᵀJ + Σ r_i ∇²r_i`, with `J` the Jacobian of the residuals with
respect to the radii. **Gauss–Newton** keeps only `JᵀJ`, which needs first
derivatives alone. The neglected term vanishes where the fit is exact, so
near such a fit the steps converge quadratically, like Newton's method.

The script never forms `J`. It needs only products `JᵀJ v`, and each one
combines both differentiation modes:

- **Forward mode** carries a change `v` of the radii to the intensities: `J v`.
- **Reverse mode** carries that change back to the radii: `Jᵀ (J v)`.

`jax.linearize` runs the scattering calculation once and keeps the native
state of its records; `jax.linear_transpose` turns the forward map into the
reverse one. Every product then reuses that saved state, without a new
scattering calculation. Conjugate gradients solve `(JᵀJ) s = −Jᵀ r` for the
step `s` with these products alone.

<!-- fmt: off -->

```python no-exec
--8<-- "docs/examples/gauss_newton_fit.py"
```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/gauss_newton_fit.txt"
```

Each row shows the fit before a step and the number of products its
conjugate-gradient solve used. From step 3 on, each largest radius error is
about 0.06/nm times the square of the previous one: 0.40, 8.6e-3, 4.6e-6 nm.
This quadratic convergence is what Gauss–Newton gives near an exact fit.

## How it works

- `residuals` returns relative intensity differences, so intensities of every
  size count alike. The measurements come from the same model at the true
  radii, without noise: the fit can become exact, and the example can check the
  recovered radii. With measured data, the misfit stops at the noise level and
  the convergence near the fit is slower, because the neglected term no longer
  vanishes.
- The first step overshoots: the largest radius error grows, although the
  misfit falls. The steps here are plain Gauss–Newton steps. For real data,
  add damping or a trust region. Levenberg–Marquardt adds `λ v` to each
  product. SciPy's `least_squares` with `method="trf"` and
  `tr_solver="lsmr"` accepts `J` as a `scipy.sparse.linalg.LinearOperator`
  whose `matvec` is the forward map and whose `rmatvec` is the reverse map.
- Before fitting, the script compares one product `JᵀJ v` with a Jacobian
  built from central differences, to `1e-6` relative tolerance.
- Conjugate gradients need at most eight products for eight radii in exact
  arithmetic; rounding adds one here. At this size, building `J` with
  `jax.jacfwd` takes eight tangents and would cost about the same. Products pay
  off for many parameters: `J` is never stored, and the solve can stop after
  fewer products than there are parameters when an approximate step still
  lowers the misfit.
- The positions, the material and the multipole truncation stay fixed. The
  small `lmax=1` dipole model keeps the example quick; check convergence in
  `lmax` before fitting real measurements.

## Not a Hessian

A true Hessian–vector product differentiates the gradient again, usually
forward-over-reverse. treams-rs derivatives are first order, so that is
[unsupported](../differentiation/index.md#rules). Gauss–Newton combines two
first-order derivatives at fixed radii instead, and needs no second
derivative of the scattering calculation.
[Framework adapters](../differentiation/frameworks.md#use-both-modes-together)
shows the same product in JAX and PyTorch on a smaller function.
[Choosing a differentiation mode](../differentiation/choosing-a-mode.md)
compares forward and reverse mode for a single derivative.

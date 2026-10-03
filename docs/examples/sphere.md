---
description: Averaged extinction and scattering spectra of a high-index sphere, and the intensity of the total field around it.
---

# Single sphere

A sphere with permittivity 16 + 0.5i and radius 75 nm has electric and magnetic
multipole resonances between 430 and 1000 THz. The spectrum shows the
extinction and scattering efficiencies, the cross sections averaged over all
directions and polarizations of the incident light and divided by the
geometric cross section πr². Keeping only the dipole modes (l = 1) shows which
resonances higher multipoles cause. A plane wave at 1000 THz then gives the
intensity of the total field around the sphere.

<!-- fmt: off -->

=== "treams-rs"

    ```python no-exec
    --8<-- "docs/examples/sphere.py"
    ```

=== "treams 0.4.5"

    ```python no-exec
    --8<-- "docs/examples/upstream/sphere.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/sphere.txt"
```

## Differences from the treams version

- `sphere_tmatrix` takes keywords and the sphere's `material`; the exterior
  `medium` is vacuum unless given. treams takes one list of materials, ending
  with the exterior.
- `average_cross_sections` returns named `scattering` and `extinction` values;
  treams reads `xs_sca_avg` and `xs_ext_avg`.
- `select(SphericalBasis.default(1))` keeps the dipole modes; treams indexes
  with `tm[swb_lmax1]`.
- The script gives `plane_wave` a unit `direction`, which does not depend on
  k0; treams passes the wave vector `[0, 0, k0]`, which treams-rs also accepts.
- `scatter` returns the scattered wave; treams writes
  `tm @ inc.expand(tm.basis)`.
- The intensity map starts as NaN, so points inside the sphere stay NaN.

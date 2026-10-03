---
description: Averaged extinction and scattering cross widths of an infinite cylinder, and the intensity of the total field around it.
---

# Single cylinder

An infinite cylinder along z with permittivity 16 + 0.5i and radius 75 nm
scatters light that travels in the xy plane, so the axial wavenumber kz is 0.
The spectrum shows the extinction and scattering efficiencies: the cross
widths averaged over all directions in the xy plane and both polarizations,
divided by the diameter. A cross width is the cross section per unit length of
the cylinder. Keeping only the modes of order m = 0 shows which resonances the
higher orders cause. A plane wave at 1000 THz then gives the intensity of the
total field around the cylinder.

<!-- fmt: off -->

=== "treams-rs"

    ```python no-exec
    --8<-- "docs/examples/cylinder.py"
    ```

=== "treams 0.4.5"

    ```python no-exec
    --8<-- "docs/examples/upstream/cylinder_tmatrixc.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/cylinder.txt"
```

## Differences from the treams version

- `cylinder_tmatrix` takes keywords and the cylinder's `material`; the exterior
  `medium` is vacuum unless given. treams takes one list of materials, ending
  with the exterior.
- `average_cross_widths` returns named `scattering` and `extinction` values;
  treams reads `xw_sca_avg` and `xw_ext_avg`.
- `select(CylindricalBasis.default(kzs, 0))` keeps the modes of order 0;
  treams indexes with `tm[cwb_mmax0]`.
- `scatter` returns the scattered wave; treams writes
  `tm @ inc.expand(tm.basis)`.
- The intensity map starts as NaN, so points inside the cylinder stay NaN.

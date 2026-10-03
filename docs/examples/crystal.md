---
description: Smallest singular value of the lattice interaction matrix of a simple cubic crystal of spheres.
---

# Photonic crystal

Spheres with permittivity 12.4 and radius 0.2 on a simple cubic lattice with
period 0.5 form a photonic crystal. The lattice interaction matrix is I - T C,
where T is the T-matrix of one sphere and C expands the waves scattered by all
other spheres about it. The crystal has a mode with zero Bloch wave vector
where this matrix is singular, so dips of its smallest singular value mark the
mode frequencies. Lengths have no unit; the frequency column is k0 / 2π, the
inverse vacuum wavelength.

<!-- fmt: off -->

=== "treams-rs"

    ```python no-exec
    --8<-- "docs/examples/crystal.py"
    ```

=== "treams 0.4.5"

    ```python no-exec
    --8<-- "docs/examples/upstream/crystal.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/crystal.txt"
```

## Differences from the treams version

- `latticeinteraction(lattice, kpar)` keeps its treams name: treams-rs has no
  other name for the lattice interaction matrix. `solve_periodic` solves the
  same system when the response itself is needed.
- `sphere_tmatrix` takes keywords and the sphere's `material`; the exterior
  `medium` is vacuum unless given.

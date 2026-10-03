# Cylinder crystal

Infinite cylinders with permittivity 12.4 and radius 0.2 stand on a square
lattice with period 0.3 in the xy plane. The lattice interaction matrix is
I - T C, where T is the cylindrical T-matrix of one cylinder and C expands the
waves scattered by all other cylinders about it. The crystal has a mode with
zero Bloch wave vector where this matrix is singular, so dips of its smallest
singular value mark the mode frequencies. Light travels in the xy plane
(kz = 0). Lengths have no unit; the frequency column is k0 / 2π, the inverse
vacuum wavelength.

<!-- fmt: off -->

=== "treams-rs"

    ```python
    --8<-- "docs/examples/cylindrical_crystal.py"
    ```

=== "treams 0.4.5"

    ```python
    --8<-- "docs/examples/upstream/crystal_tmatrixc.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/cylindrical_crystal.txt"
```

## Differences from the treams version

- `latticeinteraction(lattice, kpar)` is a treams member that treams-rs keeps
  under its treams name, as in the [photonic crystal example](crystal.md).
  `solve_periodic` solves the same system when the response itself is needed.
- `cylinder_tmatrix` takes keywords and the cylinder's `material`; the exterior
  `medium` is vacuum unless given.

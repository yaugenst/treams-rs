# Grating

The chain of spheres and the cylinder from the
[chain and cylinder example](cylindrical_cluster.md) repeat every 300 nm
along x and form a grating. Here the chain uses the three axial wavenumbers
kz + 2πn / 300 nm with |n| ≤ 1. A plane wave with kz = 0.005 nm⁻¹ travels
almost along y. The field at a point outside the cylinders comes from
expanding the waves of all cells in regular cylindrical waves about that
point. The map shows the real part of E_x in the plane x = 0; the rows
y = ±50 nm lie inside the cylinders.

<!-- fmt: off -->

=== "treams-rs"

    ```python
    --8<-- "docs/examples/grating.py"
    ```

=== "treams 0.4.5"

    ```python
    --8<-- "docs/examples/upstream/grating_tmatrixc.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/grating.txt"
```

## Differences from the treams version

- `solve_periodic(Cluster(...), lattice=Lattice(300, "x"), kpar=0)` returns a
  periodic response that keeps its lattice and Bloch vector; treams writes
  `TMatrixC.cluster(...).latticeinteraction.solve(lattice_x, 0)`.
- `in_basis(basis, kind="regular")` expands the waves of all cells about a
  point; treams writes `sca.expandlattice(basis=cwb)`.
- The periodic response has no `valid_points`, so the script finds the points
  outside the cylinders of the unit cell with NumPy.
- The results agree with treams to a relative 1.3e-8. treams sums the
  lattice of cylinders with an automatic Ewald split (the parameter that
  divides the sum into a real-space and a reciprocal-space part) that leaves
  errors of about 5e-9. With a larger explicit split, treams agrees with
  treams-rs to 1e-12. One explicit split cannot serve this example, because
  its axial orders include both propagating and decaying waves; see
  [differences](../coming-from-treams/differences.md#numerical-accuracy-limitations).

# Grid of spheres

The pair of metal spheres from the [chain example](chain.md) forms the unit cell
of a square array with a period of 300 nm in the xy plane. A plane wave hits
the array at normal incidence, traveling along z with its electric field
along -x. The field at a point outside the spheres comes from expanding the
waves of all cells in regular dipole waves about that point. The map shows the
real part of E_z; at x = ±75 nm it vanishes by symmetry, and the printed values
of about 1e-17 are rounding errors.

<!-- fmt: off -->

=== "treams-rs"

    ```python
    --8<-- "docs/examples/grid.py"
    ```

=== "treams 0.4.5"

    ```python
    --8<-- "docs/examples/upstream/grid.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/grid.txt"
```

## Differences from the treams version

- `solve_periodic(Cluster(...), lattice=Lattice.square(300), kpar=[0, 0])`
  returns a periodic response that keeps its lattice and Bloch vector; treams
  writes `TMatrix.cluster(...).latticeinteraction.solve(lattice, kpar)`.
- `in_basis(basis, kind="regular")` expands the waves of all cells about a
  point; treams writes `sca.expandlattice(basis=swb)`.
- The periodic response has no `valid_points`, so the script finds the points
  outside the spheres of the unit cell with NumPy.

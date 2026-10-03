---
description: Near field of a chain of sphere pairs that repeats every 300 nm along z.
---

# Chain of spheres

Two metal spheres with permittivity -16.5 + 1i form the unit cell of a chain
that repeats every 300 nm along z. A plane wave travels along x with its
electric field along z, so the Bloch wave vector along the chain is zero.
`solve_periodic` solves the interaction of all cells at once. The scattered
field at a point outside the spheres comes from expanding the waves of all
cells in regular dipole waves about that point. The map shows the real part of
E_z.

<!-- fmt: off -->

=== "treams-rs"

    ```python no-exec
    --8<-- "docs/examples/chain.py"
    ```

=== "treams 0.4.5"

    ```python no-exec
    --8<-- "docs/examples/upstream/chain.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/chain.txt"
```

## Differences from the treams version

- `solve_periodic(Cluster(...), lattice=300, kpar=0)` returns a periodic
  response that keeps its lattice and Bloch vector; treams writes
  `TMatrix.cluster(...).latticeinteraction.solve(lattice, kz)` and returns a
  T-matrix.
- `scatter` returns the waves of all cells. `in_basis(basis, kind="regular")`
  expands them about a point; treams writes `sca.expandlattice(basis=swb)`.
- The periodic response has no `valid_points`, so the script finds the points
  outside the spheres of the unit cell with NumPy, by the same rule as treams.

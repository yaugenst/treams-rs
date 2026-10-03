---
description: Near field of a chain of spheres and an infinite cylinder, coupled through their cylindrical T-matrices.
---

# Chain and cylinder

A chain of metal spheres along z and an infinite cylinder stand side by side
and couple. The chain has spheres of radius 65 nm every 300 nm, with its axis
at (x, y) = (-40, -50) nm; the cylinder has radius 55 nm, with its axis at
(40, 50) nm. Both have permittivity -16.5 + 1i. As in the
[chain example](cylindrical_chain.md), `to_cylindrical` describes the chain by
cylindrical waves. The cylinder uses the same seven axial wavenumbers, because
waves with different kz do not couple. `Cluster(...).solve()` then gives the
T-matrix of the pair. The map shows the real part of E_z for a plane wave with
kz = 0.005 nm⁻¹; the row x = 0 lies inside the chain's cylinder.

<!-- fmt: off -->

=== "treams-rs"

    ```python no-exec
    --8<-- "docs/examples/cylindrical_cluster.py"
    ```

=== "treams 0.4.5"

    ```python no-exec
    --8<-- "docs/examples/upstream/cluster_tmatrixc.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/cylindrical_cluster.txt"
```

## Differences from the treams version

- `Cluster([chain_tmc, cylinder], positions=positions)` holds the uncoupled
  T-matrices and `solve()` couples them; treams writes
  `TMatrixC.cluster(...).interaction.solve()`.
- `to_cylindrical(basis)` converts the solved chain; treams writes
  `TMatrixC.from_array(chain, cwb)`.
- `cylinder_tmatrix(kz=...)` takes the axial wavenumbers as a keyword; treams
  takes them as the first argument of `TMatrixC.cylinder`.
- Points inside a cylinder get NaN in both scripts. The treams version leaves
  them at zero and repeats the map for the neighboring cells; both scripts here
  stop at one cell.

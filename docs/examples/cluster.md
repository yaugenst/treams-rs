---
description: Cross sections and near fields of four coupled spheres from local, global and rotated T-matrices.
---

# Cluster of spheres

Four high-index spheres with radii from 75 to 110 nm sit at the corners of a
tetrahedron. Multiple scattering couples them: the T-matrix of the cluster
includes every interaction and keeps one expansion center per sphere.
Expanding it about the origin up to l = 6 gives one global T-matrix, valid
outside the sphere of radius 260 nm that encloses the cluster. Rotating the
cluster by 90 degrees about y and turning the incident wave with it leaves the
cross sections unchanged and rotates the intensity map.

<!-- fmt: off -->

=== "treams-rs"

    ```python no-exec
    --8<-- "docs/examples/cluster.py"
    ```

=== "treams 0.4.5"

    ```python no-exec
    --8<-- "docs/examples/upstream/cluster.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/cluster.txt"
```

## Differences from the treams version

- `Cluster(spheres, positions=...)` holds the uncoupled spheres and `solve()`
  couples them; treams writes `TMatrix.cluster(...).interaction.solve()`.
- `cross_sections` returns named `scattering`, `extinction` and `absorption`
  values; treams `xs` returns the tuple `(scattering, extinction)`.
- `in_basis` expands the T-matrix about new centers; treams calls it `expand`.
- Each result has its own name (`xs_global`, `intensity_rotate`, ...);
  treams reuses `xs`, `sca` and `intensity_global`.
- One function, `intensity_map`, computes all three maps.

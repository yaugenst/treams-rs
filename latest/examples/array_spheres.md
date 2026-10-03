# Array of spheres on a slab

Spheres with radius 100 nm, permittivity 4 and chirality parameter 0.05 form a
square array with a period of 500 nm. They rest on a 10 nm slab with
permittivity 3, so the plane of their centers lies 100 nm above the slab. A
plane wave hits the structure from below at 17.5 degrees, with
k_y = 0.3 k0 and its electric field along x. The S-matrix of the whole stack
gives the transmitted and reflected power. Lengths are in nm; the frequency
column is k0 / 2π, the inverse vacuum wavelength in 1/nm.

<!-- fmt: off -->

=== "treams-rs"

    ```python
    --8<-- "docs/examples/array_spheres.py"
    ```

=== "treams 0.4.5"

    ```python
    --8<-- "docs/examples/upstream/array_spheres.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/array_spheres.txt"
```

## Differences from the treams version

- `solve_periodic(sphere, lattice=lattice, kpar=kpar)` solves the array once;
  treams writes `TMatrix.sphere(...).latticeinteraction.solve(lattice, kpar)`.
- `to_smatrix(ports)` turns the solved array into an S-matrix; treams writes
  `SMatrices.from_array(spheres, pwb)`.
- `propagation(distance=radius)` takes the distance along z; treams takes the
  displacement `[0, 0, radius]`, which treams-rs also accepts.
- `plane_wave` takes the full propagation direction. treams takes the
  parallel wave vector together with the plane-wave basis of the ports.
- Both scripts stop the sweep before 1/350 nm⁻¹. There the diffraction order
  with k_y = 0.3 k0 + 2π / 500 nm turns from decaying to propagating, and the
  lattice sum diverges. treams-rs raises a `ValueError` at that frequency;
  treams returns a finite number.

# Array of spheres from cylindrical waves

Spheres with radius 100 nm, permittivity 4 and chirality parameter 0.05
repeat every 500 nm along z and form a chain. `to_cylindrical` describes the
chain by cylindrical waves, and chains repeated every 500 nm along x make a
square array. The array lies in the zx plane, on a 10 nm slab with
permittivity 3 that faces the y direction. A plane wave hits the structure
from below at 17.5 degrees, with k_x = 0.3 k0 and its electric field along z.
This is the structure of the [array example](array_spheres.md), turned so that
the chains run along z. The transmission and reflection differ from that
example by at most 5e-6, because three axial orders describe each chain.
Lengths are in nm; the frequency column is k0 / 2π, the inverse vacuum
wavelength in 1/nm.

<!-- fmt: off -->

=== "treams-rs"

    ```python
    --8<-- "docs/examples/cylindrical_array.py"
    ```

=== "treams 0.4.5"

    ```python
    --8<-- "docs/examples/upstream/array_spheres_tmatrixc.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/cylindrical_array.txt"
```

## Differences from the treams version

- treams keeps the ports in the xy plane and turns the basis internally in
  `SMatrices.from_array`. treams-rs needs ports in the zx plane for an array
  of cylinders, so the script states the turned geometry:
  `Lattice.square(pitch, "zx")`, the direction `[0.3 k0, k_y, 0]` and the
  field along z. treams gives the parallel wave vector `[0, 0.3 k0]` in the xy
  plane and the field along x.
- `solve_periodic` solves both the chain and the array of chains; treams
  writes `latticeinteraction.solve(pitch, kpar)` twice.
- `to_cylindrical(basis)` and `to_smatrix(ports)` convert the solved arrays;
  treams writes `TMatrixC.from_array` and `SMatrices.from_array`.
- Both scripts stop the sweep before 1/350 nm⁻¹, where the diffraction order
  with k_x = 0.3 k0 + 2π / 500 nm starts to propagate and the lattice sum
  diverges.
- The results agree with treams to a relative 7e-9. treams sums the row of
  chains with an automatic Ewald split (the parameter that divides the sum
  into a real-space and a reciprocal-space part) that leaves errors of about
  8e-9. With an explicit split of 0.5 to 1, treams agrees with treams-rs to
  1e-14. One explicit split cannot serve the lower frequencies, where the
  axial orders include decaying waves; see
  [differences](../coming-from-treams/differences.md#numerical-accuracy-limitations).

# Band structure

The unit of the [array example](array_spheres.md), with lossy spheres of
permittivity 4 + 0.1i, repeats along z with a period of 210 nm: the 10 nm
slab, a 100 nm gap, the plane of sphere centers and another 100 nm gap. A
Bloch mode of this stack repeats from one period to the next up to a factor
exp(i kz a), where kz is its Bloch wavenumber and a = 210 nm the period.
`bands` finds these wavenumbers from the S-matrix of one period. The table
lists kz a / π at normal incidence for the modes with |Im| < 0.1, that is,
for the modes that decay slowly along the stack. The modes come in pairs
±kz, one for each direction. Lengths are in nm; the frequency column is
k0 / 2π, the inverse vacuum wavelength in 1/nm.

<!-- fmt: off -->

=== "treams-rs"

    ```python
    --8<-- "docs/examples/band_structure.py"
    ```

=== "treams 0.4.5"

    ```python
    --8<-- "docs/examples/upstream/band_structure.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/band_structure.txt"
```

## Differences from the treams version

- `bands(period=az)` returns the named fields `wavenumbers` and
  `eigenvectors`; treams' `bands_kz(az)` returns a pair. treams-rs also has
  `bands_kz`.
- The script sorts the modes of each frequency, so the table has a fixed order.
  The two codes return the eigenvalues in different orders.
- The treams version also builds a plane wave that it never uses; the
  treams-rs script leaves it out.

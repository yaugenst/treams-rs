# Chiral slab

A 50 nm slab with permittivity 12.4 + 1i, permeability 1 + 0.1i and
chirality parameter 0.5 + 0.05i lies between vacuum below and a medium with
permittivity 2 and permeability 2 above. Light comes from the vacuum side at
30 degrees, with k_y = k0 / 2. The chirality makes the slab transmit the two
helicities differently, by up to 0.07 in the table. Lengths are in nm; the
frequency column is k0 / 2π, the inverse vacuum wavelength in 1/nm.

<!-- fmt: off -->

=== "treams-rs"

    ```python
    --8<-- "docs/examples/slab.py"
    ```

=== "treams 0.4.5"

    ```python
    --8<-- "docs/examples/upstream/slab.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/slab.txt"
```

## Differences from the treams version

- `slab` names the media on both sides: `negative_medium` below the slab and
  `positive_medium` above it. treams takes one list of three materials.
- `SMatrix` is the S-matrix of the whole two-port network, which treams calls
  `SMatrices`. `PlaneWavePorts` is treams' `PlaneWaveBasisByComp`.
- `power(amplitudes)` returns the named fractions `transmission` and
  `reflection`, plus `absorption`; treams' `tr` returns a pair.
- The results are called `power`, because `tr` names the treams-rs module.

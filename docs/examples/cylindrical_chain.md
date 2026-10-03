---
description: Near field of a chain of sphere pairs along z, from cylindrical waves outside the chain's cylinders and spherical waves between the spheres.
---

# Chain of spheres in cylindrical waves

Two metal spheres with permittivity -16.5 + 1i repeat every 300 nm along z:
one of radius 75 nm at (-30, 0, -75) nm and one of radius 65 nm at
(30, 0, 75) nm. `to_cylindrical` turns the solved chain into a cylindrical
T-matrix with one axis through each sphere, for the seven axial wavenumbers
kz + 2πn / 300 nm with |n| ≤ 3. Each axis collects the waves of its own
sphere. A plane wave with kz = 0.005 nm⁻¹ travels almost along x, with equal
amplitudes of both helicities. Outside the two cylinders around the axes, the
cylindrical waves give the field. Between the spheres, the script expands the
waves of all cells in regular dipole waves about each point, as in the
[chain example](chain.md). The map shows the real part of E_z.

Outside the cylinders, both descriptions give the same field to 5e-9 at
x = ±300 nm and to 0.03 at x = 100 nm, 5 nm outside the cylinder of the 65 nm
sphere. There the seven axial orders with |n| ≤ 3 converge slowly; with
|n| ≤ 6 the difference drops to 0.004.

<!-- fmt: off -->

=== "treams-rs"

    ```python no-exec
    --8<-- "docs/examples/cylindrical_chain.py"
    ```

=== "treams 0.4.5"

    ```python no-exec
    --8<-- "docs/examples/upstream/chain_tmatrixc.py"
    ```

<!-- fmt: on -->

## Output

```text
--8<-- "docs/examples/output/cylindrical_chain.txt"
```

## Differences from the treams version

- `solve_periodic(tr.Cluster(spheres, positions=positions), lattice=lattice, kpar=kz)`
  solves the chain once; treams writes
  `TMatrix.cluster(spheres, positions).latticeinteraction.solve(lattice, kz)`.
- `to_cylindrical(basis)` converts the solved chain; treams writes
  `TMatrixC.from_array(chain, cwb)`.
- `in_basis(basis, kind="regular")` expands the waves of all cells about a
  point; treams writes `sca.expandlattice(basis=swb)`.
- The periodic response has no `valid_points`, so the script finds the points
  outside the spheres with NumPy.
- The treams-rs script also evaluates the spherical waves outside the
  cylinders and prints how far the two descriptions differ there.
- The treams version repeats the map for the neighboring cells with the Bloch
  phase; both scripts here stop at one cell.

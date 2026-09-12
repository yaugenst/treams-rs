# Upstream size and rewrite scope

Inventory of tfp-photonics/treams at
`1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39` (2026-08-24), counting physical
lines in `src/treams` and excluding notebooks, documentation and generated C:

| Source | Files | Lines |
| --- | ---: | ---: |
| Python | 13 | 8,714 |
| Cython `.pyx` | 20 | 13,666 |
| Cython declarations `.pxd` | 13 | 329 |
| Total package source | 46 | 22,709 |
| Tests | 16 | 6,005 |

The tests contain 730 test definitions. Three repetitive ufunc/gufunc wrapper
files account for 6,505 source lines, so the numerical algorithm surface is
smaller than the raw total suggests. Still, the project is substantially broader
than Mie scattering: spherical/cylindrical/plane waves, multilayer and chiral
materials, translation and rotation algebra, periodic Ewald sums in several
dimensions, S-matrix composition, fields, EBCM, observables and basis metadata.

The main difficulty is numerical convention and conditioning, not source volume.
Complex square-root branches, helicity ordering, normalization, small-argument
limits, multipole cancellation and lattice convergence all require independent
qualification. The Rust implementation preserves supported Python constructors and numerical
results through PyO3. Its API uses explicit arrays to avoid silently carrying physical metadata
through arbitrary NumPy operations. The legacy ndarray annotation engine is
outside this explicit-object API; supported workflow differences are listed in
[implementation status](status.md).

Native pullbacks own the numerical solver calculation. Framework adapters then
compose a user's objective and translate cotangent conventions. Having a native
pullback does not automatically provide higher-order derivatives, batching or
accelerator support. Those remain separate contracts.

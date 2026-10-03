# Lattice-sum tests

This suite checks lattice sums and periodic wave expansions against
symmetries, derivative identities, independent calculations and recorded
reference values.

[mod.rs](mod.rs) defines properties and their input ranges.
[strategies.rs](strategies.rs) generates inputs; [ewald.rs](ewald.rs) describes
sum cases and evaluates their parts. [checks.rs](checks.rs) checks convergence,
branches, translations and derivatives. [periodic.rs](periodic.rs) checks
periodic expansions and their gradients. [tables.rs](tables.rs) holds selected
cases; high-precision data also comes from
[references](../../../references/README.md).

The tests compare direct shells, Ewald parts and spectral series where their
domains overlap. Run this suite with
`cargo test -p treams-core properties::lattice`.

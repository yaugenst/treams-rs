# Physical and numerical identities

These tests check what the public Rust calculations promise: physical laws,
analytic identities and gradients. Files are grouped by domain; the larger
[lattice suite](lattice/README.md) has its own directory.

Each property draws inputs and passes them to a function that states and
checks the identity. [test_support.rs](../test_support.rs) supplies shared
input generators, comparisons, finite differences and case limits. Recorded
failure cases and invalid-input checks use ordinary tests.

Tests of a particular numerical method stay beside its implementation.
Run both groups with `cargo test -p treams-core`; see the
[testing guide](../../../../docs/development/testing.md) for conventions and
focused commands.

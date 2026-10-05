//! Physical, analytic and adjoint identities of the public operations, one module per
//! domain: `special`, `linalg`, `lattice`, `waves`, `plane`, `smatrix` and `tmatrix`.
//! Each domain is one file, except `lattice`, a directory: its `mod.rs` holds the
//! properties, and the files beside it hold their checks, strategies, helpers and
//! tables of pinned sums.
//!
//! The crate has two tiers of tests:
//!
//! - A module's inline `tests` module checks how that module computes its results:
//!   fast paths against reference paths, tables and plans, dispatch thresholds, error
//!   paths, accuracy against reference tables, and private helpers.
//! - A domain module here checks what the public operations of its domain promise:
//!   physical laws, analytic identities, and pullbacks against finite differences and
//!   adjoint pairings.
//!
//! Both tiers need crate-private items and `cfg(test)` hooks, so the crate has no
//! `tests/` directory. The Python suite in the repository's `tests/` mirrors these
//! domains.
//!
//! Each domain module starts with one `proptest!` block per case budget
//! ([`crate::test_support::DEFAULT_CASES`] and its siblings). Each property there only
//! draws inputs and calls a documented, rustfmt-formatted
//! `check_*(..) -> Result<(), TestCaseError>` function, which states and checks the
//! identity: below the blocks, or for `lattice` in `checks.rs` and `periodic.rs`. Plain
//! `#[test]` functions cover recorded inputs and error paths. New test names are
//! sentences, `<subject>_<verb>_<claim>`, such as `regular_translations_compose`.
//! Shared pairings, strategies, finite differences and assertions live in
//! [`crate::test_support`].

mod cluster_forward;
mod lattice;
mod linalg;
mod plane;
mod smatrix;
mod special;
mod tmatrix;
mod waves;

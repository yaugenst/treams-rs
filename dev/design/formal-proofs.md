# Formal proofs

Lean 4 and Mathlib proofs cover six kernels of `treams-core`:

- diffraction-order enumeration;
- lattice shells;
- the index of the translation table;
- the degree loop of spherical translations and its Wigner 3j selection rules;
- LU equilibration;
- the pullback of the requested-illumination solve. A pullback maps the
  gradient of a loss with respect to an output to the gradients with respect
  to the inputs.

Each Lean file restates a Rust algorithm as a Lean definition and proves
properties of that definition over exact integers and reals. The proofs do not
cover floating-point rounding. The Lean sources live in
[`formal/`](https://github.com/yaugenst/treams-rs/tree/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/formal).

## Rust items and their models

Rust files are relative to `crates/treams-core/src/`.

| Rust item | Rust file | Lean file | Main theorems | Result |
| --- | --- | --- | --- | --- |
| `lattice::diffraction_orders` | [`lattice/geometry.rs`](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/crates/treams-core/src/lattice/geometry.rs) | [DiffractionOrders](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/formal/Formal/DiffractionOrders.lean) | `mem_orders`, `nodup_orders` | The output holds exactly the orders with `\|m b0 + n b1\| <= radius`, each once, for every nonsingular lattice. |
| `lattice::cube`, `lattice::geometry::visit_cube` | [`lattice/geometry.rs`](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/crates/treams-core/src/lattice/geometry.rs) | [Shells](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/formal/Formal/Shells.lean) | `mem_shell`, `mem_visit_full` | Edge shell `n` holds exactly the points of Chebyshev norm `n`; the full cube holds every point of norm at most `n`. |
| `lattice::shells::Shells::sum` | [`lattice/shells.rs`](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/crates/treams-core/src/lattice/shells.rs) | [Shells](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/formal/Formal/Shells.lean) | `shells_partition` | Shells `0..N` visit every point of Chebyshev norm below `N` exactly once. |
| `lattice::direct::direct_shell` | [`lattice/direct.rs`](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/crates/treams-core/src/lattice/direct.rs) | [Shells](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/formal/Formal/Shells.lean) | `mem_shell` | Shell `n` of the direct sum holds exactly the points of Chebyshev norm `n`. |
| `sw::plan::TranslationPlan`, `sw::plan::harmonics` | [`sw/plan.rs`](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/crates/treams-core/src/sw/plan.rs) | [Harmonics](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/formal/Formal/Harmonics.lean) | `table_index` | Index `p * p + p + m` is in bounds and names `(p, m)`. |
| `special::SolidTable::visit` | [`special/harmonics.rs`](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/crates/treams-core/src/special/harmonics.rs) | [Harmonics](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/formal/Formal/Harmonics.lean) | `table_index` | The solid-harmonic tables use the same index `p * p + p + m`. |
| `sw::coupling::Coupling::new`, `sw::coupling::degrees`, `sw::terms`, `sw::coupling::tl_vsw_term` | [`sw/coupling.rs`](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/crates/treams-core/src/sw/coupling.rs) | [SelectionRules](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/formal/Formal/SelectionRules.lean) | `mem_termDegrees`, `skipped_degree_vanishes`, `visited_term_index` | The degree loop visits exactly the degrees that pass the guard of `tl_vsw_term`. Every skipped degree has a zero coefficient under the Racah formula. Every visited degree indexes the translation table in bounds. |
| `special::wigner3j`, `special::Wigner3jRow` | [`special/wigner.rs`](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/crates/treams-core/src/special/wigner.rs) | [SelectionRules](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/formal/Formal/SelectionRules.lean) | `threeJ_eq_zero`, `threeJ_zero_odd` | The Wigner 3j symbol is zero wherever `wigner3j` returns zero early, and `(j1 j2 j3; 0 0 0)` is zero when `j1 + j2 + j3` is odd. |
| `cluster::InteractionFactor::record`, `cluster::IlluminateResidual::pullback` | [`cluster/interaction.rs`](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/crates/treams-core/src/cluster/interaction.rs) | [ImplicitAdjoint](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/formal/Formal/ImplicitAdjoint.lean) | `pullback_correct` | For invertible `I - T C`, the adjoint solve and the three returned gradients give the exact derivative of `Re tr(Gᴴ X)` in every direction `(dT, dC, da)`. |
| `linalg::equilibrate`, `linalg::Lu::new`, `Lu::solve_in_place`, `Lu::solve_adjoint_in_place` | [`linalg/mod.rs`](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/crates/treams-core/src/linalg/mod.rs) | [Equilibration](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/formal/Formal/Equilibration.lean) | `solve_eq`, `solveAdjoint_eq`, `transposed_branch`, `illumination_uses_exact_solves` | For any nonzero real scales, the scaled forward and adjoint solves equal `A⁻¹ b` and `Aᴴ⁻¹ g`. The transposed branch returns the correctly swapped scales, so the pullback theorem covers the equilibrated path. |

In the pullback row, `T` is the local T-matrix, `C` the coupling between
particles, `a` the incident fields, `X` the scattered fields and `G` the
gradient of the loss with respect to `X`. `IlluminateResidual` is what
`record` saves for the pullback.

## Why the diffraction-order code clamps at zero

Rounding can put a row of orders just beyond the cutoff even when it contains
an order on the cutoff circle. This occurs for `(±1, 0)` on a 0.3 by 0.51
rectangular lattice with the cutoff at `|G10|`.
The square-root argument `(radius - distance) (radius + distance)`
is then slightly negative. The Rust code clamps it at zero, as `Real.sqrt` does
in the model, so the row keeps its orders and the `hypot` test decides each one.
`mem_orders` holds for the model because of this clamp. The regression test
`diffraction_cutoff_on_an_order_keeps_it` in `lattice/geometry.rs` puts the
cutoff on an order of square, rectangular and hexagonal lattices and checks
that the order stays.

## Tests that tie the models to the Rust code

A hand-written model can drift from its Rust source. `lake env lean --run
Golden.lean` evaluates the executable models and writes
[`formal/golden/`](https://github.com/yaugenst/treams-rs/tree/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/formal/golden). Three Rust tests compare the Rust
functions with these files exactly, including the order of the output:

| Rust test | Rust file | Rust function | Lean definition | Golden file |
| --- | --- | --- | --- | --- |
| `cube_matches_lean_model` | `lattice/geometry.rs` | `lattice::cube` | `Shells.cube` | `cube.txt` |
| `degrees_match_lean_model` | `sw/coupling.rs` | `sw::coupling::degrees` | `SelectionRules.termDegrees` | `degrees.txt` |
| `harmonics_match_lean_model` | `sw/plan.rs` | `sw::plan::harmonics` | `Harmonics.table` | `harmonics.txt` |

Each test also requires the file to list exactly the cases from `Golden.lean`
in the same order, so a duplicated or missing line fails even when the
line count is unchanged. `cargo test` runs these tests in CI, and
`just formal` fails when `golden/` no longer matches the Lean models. A change
on either side therefore fails a check. For example, shifting the lower bound
of the degree loop by one would drop terms without an error; it fails
`degrees_match_lean_model`.

Two more tests in `sw/coupling.rs` tie the floating-point code to the
selection rules on the same cases:

- `skipped_degrees_have_zero_coefficients` is the Rust counterpart of
  `skipped_degree_vanishes`: `tl_vsw_term` returns exactly zero at every
  degree that `degrees` skips.
- `terms_use_admitted_degrees_and_table_indices` checks that `sw::terms` emits
  only admitted degrees, with order `m - mu` and an in-bounds table index.

## What the proofs do not cover

- **Rounding.** Every proof works over exact integers and reals. A rounding
  error in the Rust code, such as the diffraction-order case above, lies
  outside the theorems; Rust tests cover it.
- **Equivalence on all inputs.** The golden-file tests compare outputs only on
  the enumerated cases; they do not prove that the Rust function equals the
  Lean definition.
- **Models without an executable counterpart.** The diffraction-order, pullback
  and equilibration models have no golden file. A reviewer compares them with
  the Rust code by reading both.

## Running the proofs

Install [elan](https://github.com/leanprover/elan), then run `just formal` from
the repository root. It downloads the prebuilt Mathlib cache, checks every
proof with warnings as errors, and checks `golden/`. Lean reports an unproved
step (`sorry`) as a warning, so the build fails on it. The
[Formal workflow](https://github.com/yaugenst/treams-rs/blob/8bbcb87aaeaccd9293fc8c09eecb8727c37c4970/.github/workflows/formal.yml) runs `just formal` when
`formal/`, the `justfile`, the workflow itself or a Rust file in the table above
changes; `tests/scripts/test_repository.py` keeps that list equal to the table.

After changing one of the Rust items above, update its Lean definition and the
golden files, and run `just formal`.

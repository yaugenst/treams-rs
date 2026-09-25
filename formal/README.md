# Formal proofs

Lean 4 and Mathlib proofs about six kernels in `treams-core`. Each file under
[`Formal/`](Formal) restates a Rust algorithm as a Lean definition and proves
properties of that definition over exact integers and reals. None of these proofs
cover floating-point rounding.

The `experimental/aeneas` branch holds an experiment that instead proves
a property of `visit_cube` as Charon and Aeneas translate it from the Rust source.

| Rust source | Lean file | Main theorems | Result |
| --- | --- | --- | --- |
| `geometry::diffraction_orders` | [DiffractionOrders](Formal/DiffractionOrders.lean) | `mem_orders`, `nodup_orders` | The output holds exactly the orders with `\|m b0 + n b1\| <= radius`, each once, for every nonsingular lattice. |
| `geometry::visit_cube`, `lattice::shell_sum` | [Shells](Formal/Shells.lean) | `mem_shell`, `mem_visit_full`, `shells_partition` | Edge shell `n` holds exactly the points of Chebyshev norm `n`. Shells `0..N` visit every point of norm below `N` exactly once. |
| `TranslationPlan` table | [Harmonics](Formal/Harmonics.lean) | `table_index` | Index `p * p + p + m` is in bounds and names `(p, m)`. |
| `waves::terms`, `helper`, `angular::wigner3j` | [SelectionRules](Formal/SelectionRules.lean) | `mem_termDegrees`, `threeJ_zero_odd`, `skipped_degree_vanishes`, `visited_term_index` | The degree loop visits exactly the degrees that pass `helper`'s guard. Every skipped degree has a zero coefficient under the Racah formula. Every visited degree indexes the plan table in bounds. |
| `illumination::Residual::pullback` | [ImplicitAdjoint](Formal/ImplicitAdjoint.lean) | `pullback_correct` | For invertible `I - T C`, the adjoint solve and the three returned gradients give the exact derivative of `Re tr(Gᴴ X)` in every direction `(dT, dC, da)`. |
| `linalg::equilibrate`, `Lu::solve_in_place`, `Lu::solve_adjoint_in_place` | [Equilibration](Formal/Equilibration.lean) | `solve_eq`, `solveAdjoint_eq`, `transposed_branch`, `illumination_uses_exact_solves` | For any nonzero real scales, the scaled forward and adjoint solves equal `A⁻¹ b` and `Aᴴ⁻¹ g`. The transposed branch returns the correctly swapped scales, so the pullback theorem covers the equilibrated path. |

The diffraction-order proof exposed a rounding bug. Before a fix, a row whose distance
rounded above the cutoff was skipped. That dropped orders whose `hypot` equals the
cutoff: on a 0.3 by 0.51 rectangular lattice with the cutoff at `|G10|`, the orders
`(±1, 0)` were missing. The Rust code now clamps the square-root argument at zero, as
`Real.sqrt` does in the model. `diffraction_cutoff_on_an_order_keeps_it` in
`geometry.rs` is the regression test.

## Keeping models and code aligned

A hand-written model can drift from its Rust source. `lake exe golden` evaluates the
executable models and writes [`golden/`](golden). Three Rust tests compare the Rust
functions against these files exactly, including order:
- `cube_matches_lean_model` checks `geometry::cube` against `Shells.cube`;
- `degrees_match_lean_model` checks `waves::degrees` against `SelectionRules.termDegrees`;
- `harmonics_match_lean_model` checks `translation_plan::harmonics` against `Harmonics.table`.

`cargo test` runs them on every CI run. `just formal` also fails when `golden/` no
longer matches the Lean models. Changing either side therefore fails a check.
Shifting the degree loop's lower bound by one, which would silently drop terms, fails
`degrees_match_lean_model`.

The drift checks compare outputs only on the sampled inputs; they don't prove
equivalence. The diffraction-order, illumination and equilibration models have no
executable counterpart and are checked only by review.

## Build

Install [elan](https://github.com/leanprover/elan), then run `just formal` from the
repository root. It downloads the prebuilt Mathlib cache, checks every proof with
warnings as errors, and checks `golden/`. A `sorry` produces a warning, so the build
fails on any unproved step. The [Formal workflow](../.github/workflows/formal.yml)
runs `just formal` when this directory or a modeled Rust file changes.

After changing one of the Rust functions above, update its Lean definition and
reference files and run `just formal`.

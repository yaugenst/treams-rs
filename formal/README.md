# Formal proofs

Lean 4 and Mathlib proofs of properties of five kernels in `treams-core`. Each file
restates a Rust algorithm as a Lean definition and proves properties of that
definition over exact integers and reals. The proofs say nothing about floating-point
rounding, and nothing checks automatically that a Lean definition still matches its
Rust source.

| Rust source | Lean file | Main theorems | Result |
| --- | --- | --- | --- |
| `geometry::diffraction_orders` | [DiffractionOrders](Formal/DiffractionOrders.lean) | `mem_orders`, `nodup_orders` | The output holds exactly the orders with `\|m b0 + n b1\| <= radius`, each once, for every nonsingular lattice. |
| `geometry::visit_cube`, `lattice::shell_sum` | [Shells](Formal/Shells.lean) | `mem_shell`, `mem_visit_full`, `shells_partition` | Edge shell `n` holds exactly the points of Chebyshev norm `n`. Shells `0..N` visit every point of norm below `N` exactly once. |
| `TranslationPlan` table | [Harmonics](Formal/Harmonics.lean) | `table_index` | Index `p * p + p + m` is in bounds and names `(p, m)`. |
| `waves::terms`, `helper`, `angular::wigner3j` | [SelectionRules](Formal/SelectionRules.lean) | `mem_termDegrees`, `threeJ_zero_odd`, `skipped_degree_vanishes`, `visited_term_index` | The degree loop visits exactly the degrees that pass `helper`'s guard. Every skipped degree has a zero coefficient under the Racah formula. Every visited degree indexes the plan table in bounds. |
| `illumination::Residual::pullback` | [ImplicitAdjoint](Formal/ImplicitAdjoint.lean) | `pullback_correct` | For invertible `I - T C`, the adjoint solve and the three returned gradients give the exact derivative of `Re tr(Gᴴ X)` in every direction `(dT, dC, da)`. |

The diffraction-order model uses `Real.sqrt`, which is zero for negative arguments.
The Rust code clamps the same argument. Before that clamp, a row whose distance
rounded above the cutoff was skipped, dropping orders whose `hypot` equals the
cutoff. For example, a 0.3 by 0.51 rectangular lattice with the cutoff at `|G10|`
dropped `(±1, 0)`. `diffraction_cutoff_on_an_order_keeps_it` in `geometry.rs` is the
regression test.

## Model boundaries

- The illumination model uses a dense local matrix and square incident fields. A
  call with fewer columns is the same case with zero columns added. The Rust
  restriction of `adjoint * responseᴴ` to diagonal blocks is not modeled.
- `iterative::IterativeResidual::pullback` contracts the same adjoint, incident and
  response quantities per particle block and pair. The chain rule into Mie and
  translation derivatives and the GMRES tolerance are not modeled.
- The translation-coefficient formula, the `wigner3j` recurrence and Ewald
  convergence are not modeled.
- The Lean definitions were checked once against Rust output. `cube` matches
  `lattice_cube` exactly, including order, for dimensions 1 to 3 and `n <= 4`.
  Every degree that `terms` emits for `l, λ <= 5` lies in the Lean loop.

## Build

Install [elan](https://github.com/leanprover/elan), then run `just formal` from the
repository root. It downloads the prebuilt Mathlib cache and checks every proof. The
Lean toolchain is pinned in `lean-toolchain`, and Mathlib in `lake-manifest.json`. The
build prints a warning for any unproved `sorry`. `#print axioms` on the main theorems
lists only `propext`, `Classical.choice` and `Quot.sound`.

After changing one of the Rust functions above, update its Lean definition and run
`just formal`.

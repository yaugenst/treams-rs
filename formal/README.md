# Formal proofs

Lean 4 and Mathlib proofs about six kernels in `treams-core`, in two kinds.

- **Hand-written models.** Each file under [`Formal/`](Formal) restates a Rust
  algorithm as a Lean definition and proves properties of that definition over exact
  integers and reals.
- **Translated Rust.** The project under [`aeneas/`](aeneas) proves a property of the
  Lean code that Charon and Aeneas generate from the Rust source of `visit_cube`.

None of these proofs cover floating-point rounding.

| Rust source | Lean file | Main theorems | Result |
| --- | --- | --- | --- |
| `geometry::diffraction_orders` | [DiffractionOrders](Formal/DiffractionOrders.lean) | `mem_orders`, `nodup_orders` | The output holds exactly the orders with `\|m b0 + n b1\| <= radius`, each once, for every nonsingular lattice. |
| `geometry::visit_cube`, `lattice::shell_sum` | [Shells](Formal/Shells.lean) | `mem_shell`, `mem_visit_full`, `shells_partition` | Edge shell `n` holds exactly the points of Chebyshev norm `n`. Shells `0..N` visit every point of norm below `N` exactly once. |
| `TranslationPlan` table | [Harmonics](Formal/Harmonics.lean) | `table_index` | Index `p * p + p + m` is in bounds and names `(p, m)`. |
| `waves::terms`, `helper`, `angular::wigner3j` | [SelectionRules](Formal/SelectionRules.lean) | `mem_termDegrees`, `threeJ_zero_odd`, `skipped_degree_vanishes`, `visited_term_index` | The degree loop visits exactly the degrees that pass `helper`'s guard. Every skipped degree has a zero coefficient under the Racah formula. Every visited degree indexes the plan table in bounds. |
| `illumination::Residual::pullback` | [ImplicitAdjoint](Formal/ImplicitAdjoint.lean) | `pullback_correct` | For invertible `I - T C`, the adjoint solve and the three returned gradients give the exact derivative of `Re tr(Gᴴ X)` in every direction `(dT, dC, da)`. |
| `linalg::equilibrate`, `Lu::solve_in_place`, `Lu::solve_adjoint_in_place` | [Equilibration](Formal/Equilibration.lean) | `solve_eq`, `solveAdjoint_eq`, `transposed_branch`, `illumination_uses_exact_solves` | For any nonzero real scales, the scaled forward and adjoint solves equal `A⁻¹ b` and `Aᴴ⁻¹ g`. The transposed branch returns the correctly swapped scales, so the pullback theorem covers the equilibrated path. |
| `geometry::visit_cube`, translated | [VisitCubeProof](aeneas/TreamsAeneas/VisitCubeProof.lean) | `append_spec`, `visit_cube_records`, `visit_cube_ok` | The translated Rust recursion returns `Ok` and records exactly the points of `Shells.cube`, padded with zeros to three coordinates. This holds for dimensions up to 3 and every `n` from 0 to `i64::MAX`, with no overflow or panic. |

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

## The Aeneas proof

[Charon](https://github.com/AeneasVerif/charon) and
[Aeneas](https://github.com/AeneasVerif/aeneas) translate `visit_cube` from the Rust
source into [`aeneas/VisitCube/Funs.lean`](aeneas/VisitCube/Funs.lean). These tools
handle only a narrow part of this repository:

- Aeneas rejects floating point, so it cannot translate the other kernels.
- Its default loop translation produced Lean that fails to compile. The recursive
  `append` calls itself from inside loops, and Lean cannot prove the `loop`
  combinator monotone. `-loops-to-rec` avoids the problem.
- It left four standard-library functions as axioms: array iteration, `i64::abs`,
  `RangeInclusive::contains` and `String::from`. They are defined in
  [`FunsExternal.lean`](aeneas/VisitCube/FunsExternal.lean) and
  [`TypesExternal.lean`](aeneas/VisitCube/TypesExternal.lean). Those definitions
  model the Rust standard library and are trusted.
- The proof uses a visitor that records each point; the real callers' closures are
  not translated.

The proof also trusts Charon, Aeneas and Aeneas's Lean library. `append_spec` and
`visit_cube_records` depend only on the standard Lean axioms. `visit_cube_ok` also
depends on the native evaluation axiom, which the generated definition uses for a
string literal.

Aeneas pins Lean v4.31.0, so the project has its own toolchain and Mathlib. It also
carries a copy of `Range.lean` and `Shells.lean` with one lemma name adjusted. To
regenerate the translation, run Aeneas `63afc34` with its pinned Charon
`62585970` from `crates/treams-core`:

```sh
charon cargo --preset=aeneas --start-from 'crate::geometry::visit_cube' --dest-file visit_cube.llbc
aeneas -backend lean visit_cube.llbc -dest out -split-files -loops-to-rec
```

Copy `Types.lean` and `Funs.lean` from `out` into `aeneas/VisitCube`. Nothing checks
automatically that the generated files match the current Rust source.

## Build

Install [elan](https://github.com/leanprover/elan). From the repository root:
- `just formal` downloads the prebuilt Mathlib cache, checks every proof in
  `Formal/` with warnings as errors, and checks `golden/`;
- `just formal-aeneas` checks the Aeneas proof. The first run also compiles Aeneas's
  Lean library.

A `sorry` produces a warning, so the build fails on any unproved step. The
[Formal workflow](../.github/workflows/formal.yml) runs `just formal` when this
directory or a modeled Rust file changes. The Aeneas check is not in CI.

After changing one of the Rust functions above, update its Lean definition and
reference files and run `just formal`. After changing `visit_cube`, regenerate the
translation and run `just formal-aeneas`.

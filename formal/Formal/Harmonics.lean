import Formal.Range

/-!
# Harmonic table indices

`TranslationPlan` in `crates/treams-core/src/translation_plan.rs` stores each coupling
term as `index = p * p + p + m` into a table built by

```rust
for p in 0..=self.order { for m in -p..=p { table.push(harmonic(p, m, ..)) } }
```

The same `(l, m)` order builds the lattice normalization and periodic tables.
-/

namespace Treams.Harmonics

open Treams

/-- The `(p, m)` order of the Rust table. -/
def table (order : ℕ) : List (ℤ × ℤ) :=
  (List.range (order + 1)).flatMap fun p : ℕ => (irange (-p) p).map fun m => ((p : ℤ), m)

theorem length_irange (a b : ℤ) : (irange a b).length = (b + 1 - a).toNat := by
  simp [irange]

theorem getElem?_irange {a b : ℤ} {i : ℕ} (hi : i < (b + 1 - a).toNat) :
    (irange a b)[i]? = some (a + i) := by
  simp [irange, hi]

theorem length_table (order : ℕ) : (table order).length = (order + 1) ^ 2 := by
  induction order with
  | zero => simp [table, length_irange]
  | succ k ih =>
    rw [table, List.range_succ, List.flatMap_append, List.length_append, ← table, ih]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.length_map,
      length_irange]
    push_cast
    have : ((k : ℤ) + 1 + 1 - -((k : ℤ) + 1)).toNat = 2 * k + 3 := by omega
    rw [this]; ring

/-- The table position of `(p, -p + j)` for `j ≤ 2p`. -/
theorem table_offset {order p j : ℕ} (hp : p ≤ order) (hj : j ≤ 2 * p) :
    (table order)[p * p + j]? = some ((p : ℤ), -(p : ℤ) + j) := by
  induction order with
  | zero =>
    obtain rfl : p = 0 := by omega
    obtain rfl : j = 0 := by omega
    simp [table, irange]
  | succ k ih =>
    rw [table, List.range_succ, List.flatMap_append, ← table]
    rcases Nat.lt_or_ge p (k + 1) with hlt | hge
    · rw [List.getElem?_append_left (by rw [length_table]; nlinarith), ih (by omega)]
    · obtain rfl : p = k + 1 := by omega
      rw [List.getElem?_append_right (by rw [length_table]; nlinarith), length_table]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.getElem?_map]
      rw [show (k + 1) * (k + 1) + j - (k + 1) ^ 2 = j by ring_nf; omega,
        getElem?_irange (by omega)]
      simp

/-- Every valid `(p, m)` sits at `p * p + p + m`, so the Rust index is in bounds. -/
theorem table_index {order p : ℕ} {m : ℤ} (hp : p ≤ order) (hm : m.natAbs ≤ p) :
    (p * p + p + m).toNat < (order + 1) ^ 2 ∧
      (table order)[(p * p + p + m).toNat]? = some ((p : ℤ), m) := by
  obtain ⟨j, hj, rfl⟩ : ∃ j : ℕ, j ≤ 2 * p ∧ m = -(p : ℤ) + j := ⟨(m + p).toNat, by omega, by omega⟩
  have hidx : ((p : ℤ) * p + p + (-p + j)).toNat = p * p + j := by
    rw [show (p : ℤ) * p + p + (-p + j) = ((p * p + j : ℕ) : ℤ) by push_cast; ring]
    exact Int.toNat_natCast _
  rw [hidx]
  refine ⟨by nlinarith, table_offset hp hj⟩

end Treams.Harmonics

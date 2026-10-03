import Formal.Harmonics

/-!
# Selection-rule pruning in spherical translations

`Coupling::new` in `crates/treams-core/src/sw/coupling.rs`, which both `sw::terms` and
`TranslationPlan::between` use, iterates the degrees of `degrees`,

```rust
let start = l + lambda - i32::from(cross);
let end = (lambda - l).abs().saturating_add(i32::from(cross)).max((m - mu).abs());
for p in (end..=start).rev().step_by(2) { .. tl_vsw_term(l, m, lambda, -mu, p, p - i32::from(cross), rows) .. }
```

and `tl_vsw_term` returns zero early unless both Wigner 3j symbols it multiplies,
`(l λ p; m -μ μ-m)` and `(l λ q; 0 0 0)` with `q = p - cross`, can be nonzero. It reads
them from `Wigner3jRow`s, which equal `special::wigner3j` bit for bit.

We show that the loop visits exactly the `p` that pass `tl_vsw_term`'s guard, and that every
skipped `p` has a zero coefficient under the Racah formula. The parity rule for
`(l λ q; 0 0 0)` is the only case that is not a definitional zero.
-/

namespace Treams.SelectionRules

/-- `(stop..=start).rev().step_by(2)`. -/
def stepDown (start stop : ℤ) : List ℤ :=
  (List.range ((start - stop) / 2 + 1).toNat).map fun i : ℕ => start - 2 * (i : ℤ)

theorem mem_stepDown {start stop p : ℤ} :
    p ∈ stepDown start stop ↔ stop ≤ p ∧ p ≤ start ∧ (start - p) % 2 = 0 := by
  unfold stepDown
  rw [List.mem_map]
  constructor
  · rintro ⟨i, hi, rfl⟩
    have := List.mem_range.mp hi
    omega
  · intro h
    exact ⟨((start - p) / 2).toNat, List.mem_range.mpr (by omega), by omega⟩

/-- The degrees `p` visited by `terms` for source `(l, m)`, destination `(λ, μ)`. -/
def termDegrees (l m lam mu c : ℤ) : List ℤ :=
  stepDown (l + lam - c) (max (|lam - l| + c) |m - mu|)

/-- `tl_vsw_term(l, m, lambda, mu, p, q, rows)` passes its early-return guard. -/
def helperPasses (l m lam mu p q : ℤ) : Prop :=
  ¬(p < max |m + mu| |l - lam| ∨ p > l + lam ∨ q < |l - lam| ∨ q > l + lam ∨
    (q + l + lam) % 2 ≠ 0)

/-- The loop is exactly `tl_vsw_term`'s guard: it skips no admissible degree and visits no
degree that `tl_vsw_term` would reject. -/
theorem mem_termDegrees {l m lam mu c p : ℤ} (hc : c = 0 ∨ c = 1) :
    p ∈ termDegrees l m lam mu c ↔ helperPasses l m lam (-mu) p (p - c) := by
  simp only [termDegrees, helperPasses, mem_stepDown, Int.abs_eq_natAbs]
  omega

/-! ## Wigner 3j symbols -/

noncomputable def fact (n : ℤ) : ℝ := (n.toNat.factorial : ℝ)

/-- `1 / n!`, zero for negative `n`: the Racah-sum convention. -/
noncomputable def invFact (n : ℤ) : ℝ := if n < 0 then 0 else (n.toNat.factorial : ℝ)⁻¹

/-- The Racah sum of `(j1 j2 j3; m1 m2 m3)`. -/
noncomputable def racahSum (j1 j2 j3 m1 m2 : ℤ) : ℝ :=
  ∑ k ∈ Finset.range (j1 + j2 - j3 + 1).toNat,
    (-1 : ℝ) ^ k * invFact k * invFact (j3 - j2 + k + m1) * invFact (j3 - j1 + k - m2) *
      invFact (j1 + j2 - j3 - k) * invFact (j1 - k - m1) * invFact (j2 - k + m2)

/-- The Wigner 3j symbol by the Racah formula. The zero conditions are those of the
early return in `wigner3j` (`crates/treams-core/src/special/wigner.rs`). -/
noncomputable def threeJ (j1 j2 j3 m1 m2 m3 : ℤ) : ℝ :=
  open Classical in
  if j1 < 0 ∨ j2 < 0 ∨ j3 < 0 ∨ j3 < |j1 - j2| ∨ j3 > j1 + j2 ∨ |m1| > j1 ∨ |m2| > j2 ∨
      |m3| > j3 ∨ m1 + m2 + m3 ≠ 0 then 0
  else
    (-1 : ℝ) ^ (j1 - j2 - m3) *
      √(fact (j1 + j2 - j3) * fact (j1 - j2 + j3) * fact (-j1 + j2 + j3) /
        fact (j1 + j2 + j3 + 1)) *
      √(fact (j1 + m1) * fact (j1 - m1) * fact (j2 + m2) * fact (j2 - m2) * fact (j3 + m3) *
        fact (j3 - m3)) *
      racahSum j1 j2 j3 m1 m2

theorem neg_one_pow_sub {a k : ℕ} (h : k ≤ a) : (-1 : ℝ) ^ (a - k) = (-1) ^ a * (-1) ^ k := by
  obtain ⟨c, rfl⟩ : ∃ c, a = k + c := ⟨a - k, by omega⟩
  rw [Nat.add_sub_cancel_left, pow_add, mul_comm ((-1 : ℝ) ^ k), mul_assoc, ← mul_pow]
  simp

/-- Reflecting the summation index `k ↦ (j1 + j2 - j3) - k` permutes the six factorials. -/
theorem racahSum_reflect {j1 j2 j3 : ℤ} (h : 0 ≤ j1 + j2 - j3) :
    racahSum j1 j2 j3 0 0 = (-1 : ℝ) ^ (j1 + j2 - j3).toNat * racahSum j1 j2 j3 0 0 := by
  set a := (j1 + j2 - j3).toNat with ha
  have ha' : (a : ℤ) = j1 + j2 - j3 := by omega
  unfold racahSum
  simp only [add_zero, sub_zero]
  rw [show (j1 + j2 - j3 + 1).toNat = a + 1 by omega, Finset.mul_sum]
  conv_lhs => rw [← Finset.sum_range_reflect]
  refine Finset.sum_congr rfl fun k hk => ?_
  have hk := Finset.mem_range.mp hk
  rw [show a + 1 - 1 - k = a - k by omega]
  have hcast : ((a - k : ℕ) : ℤ) = a - k := by omega
  simp only [hcast]
  rw [show j3 - j2 + (a - k) = j1 - k by omega, show j3 - j1 + (a - k) = j2 - k by omega,
    show j1 + j2 - j3 - (a - k) = (k : ℤ) by omega, show j1 - (a - k) = j3 - j2 + k by omega,
    show j2 - (a - k) = j3 - j1 + k by omega, show (a : ℤ) - k = j1 + j2 - j3 - k by omega]
  rw [neg_one_pow_sub (by omega)]
  ring

/-- `(j1 j2 j3; 0 0 0)` vanishes when `j1 + j2 + j3` is odd. -/
theorem threeJ_zero_odd {j1 j2 j3 : ℤ} (hodd : (j1 + j2 + j3) % 2 ≠ 0) :
    threeJ j1 j2 j3 0 0 0 = 0 := by
  unfold threeJ
  split_ifs with hguard
  · rfl
  have htri : 0 ≤ j1 + j2 - j3 := by
    simp only [Int.abs_eq_natAbs] at hguard
    omega
  have hsum : racahSum j1 j2 j3 0 0 = 0 := by
    have hr := racahSum_reflect htri
    rw [Odd.neg_one_pow (Nat.odd_iff.mpr (by omega))] at hr
    linarith
  rw [hsum, mul_zero]

theorem threeJ_eq_zero {j1 j2 j3 m1 m2 m3 : ℤ}
    (h : j1 < 0 ∨ j2 < 0 ∨ j3 < 0 ∨ j3 < |j1 - j2| ∨ j3 > j1 + j2 ∨ |m1| > j1 ∨ |m2| > j2 ∨
      |m3| > j3 ∨ m1 + m2 + m3 ≠ 0) :
    threeJ j1 j2 j3 m1 m2 m3 = 0 := by
  unfold threeJ; exact ite_eq_left h

/-- Every degree the loop skips has a zero translation coefficient. -/
theorem skipped_degree_vanishes {l m lam mu c p : ℤ} (hc : c = 0 ∨ c = 1)
    (hp : p ∉ termDegrees l m lam mu c) :
    threeJ l lam p m (-mu) (mu - m) * threeJ l lam (p - c) 0 0 0 = 0 := by
  rw [mem_termDegrees hc, helperPasses, not_not] at hp
  simp only [Int.abs_eq_natAbs] at hp
  by_cases hparity : (p - c + l + lam) % 2 ≠ 0
  · rw [threeJ_zero_odd (by omega), mul_zero]
  by_cases hfirst : p < max (m + -mu).natAbs (l - lam).natAbs ∨ p > l + lam
  · rw [threeJ_eq_zero (by simp only [Int.abs_eq_natAbs]; omega), zero_mul]
  · rw [threeJ_eq_zero (j3 := p - c) (by simp only [Int.abs_eq_natAbs]; omega), mul_zero]

/-- Every degree the loop visits indexes the `TranslationPlan` table in bounds, at its own
`(p, m - μ)`, when the plan order is at least `l + λ`. -/
theorem visited_term_index {l m lam mu c p : ℤ} {order : ℕ} (hc : c = 0 ∨ c = 1)
    (horder : l + lam ≤ order) (hp : p ∈ termDegrees l m lam mu c) :
    (p * p + p + (m - mu)).toNat < (order + 1) ^ 2 ∧
      (Harmonics.table order)[(p * p + p + (m - mu)).toNat]? = some (p, m - mu) := by
  simp only [termDegrees, mem_stepDown, Int.abs_eq_natAbs] at hp
  obtain ⟨q, rfl⟩ : ∃ q : ℕ, p = q := ⟨p.toNat, by omega⟩
  exact Harmonics.table_index (by omega) (by omega)

end Treams.SelectionRules

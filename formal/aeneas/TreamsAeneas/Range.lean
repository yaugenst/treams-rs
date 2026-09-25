import Mathlib

/-! # Inclusive integer ranges, as Rust's `a..=b` -/

namespace Treams

/-- Inclusive integer range `a..=b`. -/
def irange (a b : ℤ) : List ℤ :=
  List.map (fun i : ℕ => a + (i : ℤ)) (List.range (b + 1 - a).toNat)

theorem mem_irange {a b x : ℤ} : x ∈ irange a b ↔ a ≤ x ∧ x ≤ b := by
  unfold irange
  rw [List.mem_map]
  constructor
  · rintro ⟨i, hi, rfl⟩
    have := List.mem_range.mp hi
    omega
  · rintro ⟨h1, h2⟩
    exact ⟨(x - a).toNat, List.mem_range.mpr (by omega), by omega⟩

theorem nodup_irange (a b : ℤ) : (irange a b).Nodup :=
  List.Nodup.map (fun i j h => by simpa using h) List.nodup_range

end Treams

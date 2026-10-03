import Formal.Range

/-!
# Diffraction-order enumeration

A real-arithmetic model of `diffraction_orders` in
`crates/treams-core/src/lattice/geometry.rs`. Reciprocal vectors are `b0 = (b00, b01)` and
`b1 = (b10, b11)`; an order `(m, n)` lies at `m b0 + n b1`.

`Real.sqrt` returns zero for negative arguments, so `half` models the Rust code
with the square-root argument clamped at zero and no separate row-distance skip.
-/

namespace Treams.DiffractionOrders

open Treams

structure Basis where
  b00 : ℝ
  b01 : ℝ
  b10 : ℝ
  b11 : ℝ

variable (B : Basis) (R : ℝ)

noncomputable def det : ℝ := B.b00 * B.b11 - B.b01 * B.b10
noncomputable def len : ℝ := √(B.b10 ^ 2 + B.b11 ^ 2)

/-- `x.hypot(y) <= radius` for the order's reciprocal vector. -/
def inside (m n : ℤ) : Prop :=
  √((m * B.b00 + n * B.b10) ^ 2 + (m * B.b01 + n * B.b11) ^ 2) ≤ R

/-- The orders emitted for one value of `n` in row `m`. -/
noncomputable def emit (m n : ℤ) : List (ℤ × ℤ) :=
  open Classical in
  if m = 0 ∧ n ≤ 0 then [] else if inside B R m n then [(m, n), (-m, -n)] else []

/-- One `m` iteration of the Rust loop. -/
noncomputable def row (m : ℤ) : List (ℤ × ℤ) :=
  let distance := |m * det B / len B|
  let center := -m * (B.b00 * (B.b10 / len B) + B.b01 * (B.b11 / len B)) / len B
  let half := √((R - distance) * (R + distance)) / len B
  let lower := ⌊center - half⌋ - 1
  let upper := ⌈center + half⌉ + 1
  (irange (max lower 0) upper).flatMap (emit B R m) ++
    (irange lower (min upper (-1))).reverse.flatMap (emit B R m)

/-- The flat Rust output `[0, 0, m, n, -m, -n, ...]`, as a list of pairs. -/
noncomputable def orders : List (ℤ × ℤ) :=
  open Classical in
  if R < 0 then [] else (0, 0) :: (irange 0 ⌈R * len B / |det B|⌉).flatMap (row B R)

/-! ## Geometry -/

variable {B R}

theorem inside_neg {m n : ℤ} : inside B R (-m) (-n) ↔ inside B R m n := by
  unfold inside; push_cast; ring_nf

theorem len_pos (hdet : det B ≠ 0) : 0 < len B := by
  unfold len
  apply Real.sqrt_pos.mpr
  by_contra h
  have h10 : B.b10 = 0 := by nlinarith [sq_nonneg B.b10, sq_nonneg B.b11]
  have h11 : B.b11 = 0 := by nlinarith [sq_nonneg B.b10, sq_nonneg B.b11]
  exact hdet (by simp [det, h10, h11])

theorem len_sq : len B ^ 2 = B.b10 ^ 2 + B.b11 ^ 2 := by
  unfold len; rw [Real.sq_sqrt (by positivity)]

/-- The Lagrange identity split into the row distance and the projection on `b1`. -/
theorem lagrange (m n : ℤ) :
    (m * det B) ^ 2 + (m * (B.b00 * B.b10 + B.b01 * B.b11) + n * len B ^ 2) ^ 2 =
      ((m * B.b00 + n * B.b10) ^ 2 + (m * B.b01 + n * B.b11) ^ 2) * len B ^ 2 := by
  rw [len_sq]; unfold det; ring

/-- Every inside order with `m ≥ 0` lies within row `m`'s rounded `n` window. -/
theorem window (hdet : det B ≠ 0) {m n : ℤ} (h : inside B R m n) :
    let distance := |m * det B / len B|
    let center := -m * (B.b00 * (B.b10 / len B) + B.b01 * (B.b11 / len B)) / len B
    let half := √((R - distance) * (R + distance)) / len B
    ⌊center - half⌋ - 1 ≤ n ∧ n ≤ ⌈center + half⌉ + 1 := by
  intro distance center half
  have hL := len_pos hdet
  set dot := m * (B.b00 * B.b10 + B.b01 * B.b11) + n * len B ^ 2
  set norm2 := (m * B.b00 + n * B.b10) ^ 2 + (m * B.b01 + n * B.b11) ^ 2
  have hR : 0 ≤ R := le_trans (Real.sqrt_nonneg _) h
  have hn2 : norm2 ≤ R ^ 2 := by
    calc norm2 = √norm2 ^ 2 := (Real.sq_sqrt (by positivity)).symm
      _ ≤ R ^ 2 := by gcongr; exact h
  have hdist : distance ^ 2 = (m * det B) ^ 2 / len B ^ 2 := by
    simp only [distance, sq_abs, div_pow]
  -- (R - d)(R + d) ≥ (dot / L)^2
  have hkey : (dot / len B) ^ 2 ≤ (R - distance) * (R + distance) := by
    have hl := lagrange (B := B) m n
    rw [div_pow, show (R - distance) * (R + distance) = R ^ 2 - distance ^ 2 by ring, hdist,
      div_le_iff₀ (by positivity), sub_mul, div_mul_cancel₀ _ (by positivity)]
    nlinarith [hl, mul_le_mul_of_nonneg_right hn2 (sq_nonneg (len B))]
  have hhalf : |dot| / len B ^ 2 ≤ half := by
    have := Real.sqrt_le_sqrt hkey
    rw [Real.sqrt_sq_eq_abs, abs_div, abs_of_pos hL] at this
    simp only [half, pow_two, ← div_div]
    exact div_le_div_of_nonneg_right this hL.le
  have hcenter : (n : ℝ) - center = dot / len B ^ 2 := by
    simp only [center, dot]
    field_simp
    ring
  have hle : |(n : ℝ) - center| ≤ half := by
    rw [hcenter, abs_div, abs_of_pos (pow_pos hL 2)]; exact hhalf
  replace hle := abs_le.mp hle
  constructor
  · have : ⌊center - half⌋ ≤ n := by
      rw [← Int.floor_intCast (R := ℝ) n]; exact Int.floor_le_floor (by linarith [hle.1])
    omega
  · have : n ≤ ⌈center + half⌉ := by
      rw [← Int.ceil_intCast (R := ℝ) n]; exact Int.ceil_le_ceil (by linarith [hle.2])
    omega

/-- Every inside order with `m ≥ 0` lies below the row bound. -/
theorem row_bound (hdet : det B ≠ 0) {m n : ℤ} (hm : 0 ≤ m) (h : inside B R m n) :
    m ≤ ⌈R * len B / |det B|⌉ := by
  have hL := len_pos hdet
  have hd : 0 < |det B| := abs_pos.mpr hdet
  have hl := lagrange (B := B) m n
  set norm2 := (m * B.b00 + n * B.b10) ^ 2 + (m * B.b01 + n * B.b11) ^ 2
  have hsq : (m * det B) ^ 2 ≤ (R * len B) ^ 2 := by
    have hn2 : norm2 ≤ R ^ 2 := by
      calc norm2 = √norm2 ^ 2 := (Real.sq_sqrt (by positivity)).symm
        _ ≤ R ^ 2 := by gcongr; exact h
    nlinarith [sq_nonneg (m * (B.b00 * B.b10 + B.b01 * B.b11) + n * len B ^ 2),
      mul_le_mul_of_nonneg_right hn2 (sq_nonneg (len B))]
  have hR : 0 ≤ R := le_trans (Real.sqrt_nonneg _) h
  have habs : (m : ℝ) * |det B| ≤ R * len B := by
    have := sq_le_sq.mp hsq
    rwa [abs_mul, abs_of_nonneg (by exact_mod_cast hm : (0 : ℝ) ≤ m),
      abs_of_nonneg (mul_nonneg hR hL.le)] at this
  have : (m : ℝ) ≤ R * len B / |det B| := by rw [le_div_iff₀ hd]; exact habs
  exact Int.cast_le.mp (le_trans this (Int.le_ceil _))

/-! ## Enumeration -/

theorem mem_emit {m n : ℤ} {e : ℤ × ℤ} :
    e ∈ emit B R m n ↔ ¬(m = 0 ∧ n ≤ 0) ∧ inside B R m n ∧ (e = (m, n) ∨ e = (-m, -n)) := by
  unfold emit; split_ifs <;> simp_all

theorem mem_row {m : ℤ} {e : ℤ × ℤ} (h : e ∈ row B R m) :
    ∃ n, ¬(m = 0 ∧ n ≤ 0) ∧ inside B R m n ∧ (e = (m, n) ∨ e = (-m, -n)) := by
  simp only [row, List.mem_append, List.mem_flatMap, List.mem_reverse] at h
  rcases h with ⟨n, -, h⟩ | ⟨n, -, h⟩ <;> exact ⟨n, mem_emit.mp h⟩

theorem pair_mem_row (hdet : det B ≠ 0) {m n : ℤ} (hn : ¬(m = 0 ∧ n ≤ 0))
    (h : inside B R m n) {e : ℤ × ℤ} (he : e = (m, n) ∨ e = (-m, -n)) : e ∈ row B R m := by
  obtain ⟨hlo, hhi⟩ := window hdet h
  have he : e ∈ emit B R m n := mem_emit.mpr ⟨hn, h, he⟩
  simp only [row, List.mem_append, List.mem_flatMap, List.mem_reverse]
  by_cases h0 : 0 ≤ n
  · exact Or.inl ⟨n, mem_irange.mpr ⟨by omega, hhi⟩, he⟩
  · exact Or.inr ⟨n, mem_irange.mpr ⟨hlo, by omega⟩, he⟩

/-- Completeness and soundness: the output holds exactly the orders inside the circle. -/
theorem mem_orders (hdet : det B ≠ 0) (hR : 0 ≤ R) (m n : ℤ) :
    (m, n) ∈ orders B R ↔ inside B R m n := by
  simp only [orders, not_lt.mpr hR, ite_false, List.mem_cons, List.mem_flatMap]
  constructor
  · rintro (h | ⟨k, -, h⟩)
    · simp only [Prod.mk.injEq] at h
      simp [inside, h.1, h.2, hR]
    · obtain ⟨j, -, hin, he | he⟩ := mem_row h <;> simp only [Prod.mk.injEq] at he
      · rwa [he.1, he.2]
      · rw [he.1, he.2, inside_neg]; exact hin
  · intro h
    by_cases h00 : m = 0 ∧ n = 0
    · left; rw [h00.1, h00.2]
    right
    by_cases hpos : 0 ≤ m ∧ ¬(m = 0 ∧ n ≤ 0)
    · exact ⟨m, mem_irange.mpr ⟨hpos.1, row_bound hdet hpos.1 h⟩,
        pair_mem_row hdet hpos.2 h (Or.inl rfl)⟩
    · have h' : inside B R (-m) (-n) := inside_neg.mpr h
      have hn : ¬(-m = 0 ∧ -n ≤ 0) := by omega
      exact ⟨-m, mem_irange.mpr ⟨by omega, row_bound hdet (by omega) h'⟩,
        pair_mem_row hdet hn h' (Or.inr (by simp))⟩

theorem nodup_flatMap_emit {m : ℤ} {ns : List ℤ} (h : ns.Nodup) :
    (ns.flatMap (emit B R m)).Nodup := by
  rw [List.nodup_flatMap]
  refine ⟨fun n _ => ?_, h.imp fun {n n'} hne => ?_⟩
  · unfold emit
    split_ifs with h1 h2
    · exact List.nodup_nil
    · simp only [List.nodup_cons, List.mem_cons, Prod.mk.injEq, List.not_mem_nil,
        List.nodup_nil, or_false, not_false_eq_true, and_true]
      omega
    · exact List.nodup_nil
  · intro e h1 h2
    obtain ⟨hn1, -, e1⟩ := mem_emit.mp h1
    obtain ⟨hn2, -, e2⟩ := mem_emit.mp h2
    rcases e1 with rfl | rfl <;> rcases e2 with h | h <;> simp only [Prod.mk.injEq] at h <;> omega

theorem nodup_row {m : ℤ} : (row B R m).Nodup := by
  unfold row
  rw [← List.flatMap_append]
  refine nodup_flatMap_emit
    ((nodup_irange _ _).append (List.nodup_reverse.mpr (nodup_irange _ _)) ?_)
  intro a h1 h2
  have := (mem_irange.mp h1).1
  have := (mem_irange.mp (List.mem_reverse.mp h2)).2
  omega

/-- Every order appears once. -/
theorem nodup_orders : (orders B R).Nodup := by
  unfold orders
  split_ifs
  · exact List.nodup_nil
  refine List.nodup_cons.mpr ⟨?_, ?_⟩
  · simp only [List.mem_flatMap, not_exists, not_and]
    intro k _ h
    obtain ⟨n, hn, -, he | he⟩ := mem_row h <;> simp only [Prod.mk.injEq] at he <;> omega
  · rw [List.nodup_flatMap]
    refine ⟨fun _ _ => nodup_row, (nodup_irange _ _).imp_of_mem ?_⟩
    intro a b ha hb hab e h1 h2
    have ha := (mem_irange.mp ha).1
    have hb := (mem_irange.mp hb).1
    obtain ⟨n, hn, -, e1⟩ := mem_row h1
    obtain ⟨n', hn', -, e2⟩ := mem_row h2
    rcases e1 with rfl | rfl <;> rcases e2 with h | h <;> simp only [Prod.mk.injEq] at h <;> omega

end Treams.DiffractionOrders

import Formal.Range

/-!
# Lattice shells

A model of `visit_cube` in `crates/treams-core/src/lattice/geometry.rs`. `visit n edge k
boundary` is the recursive `append` with `k` axes left to fill; a point lists the
current axis first. The Rust condition `axis + 1 == dim` is `k = 0` here.

`Shells::sum` in `lattice/shells.rs` visits `visit n true dim false` for `n = 0, 1, ...`;
`direct_shell` in `lattice/direct.rs` visits one such shell.
-/

namespace Treams.Shells

open Treams

def visit (n : ℤ) (edge : Bool) : ℕ → Bool → List (List ℤ)
  | 0, _ => [[]]
  | k + 1, boundary =>
    if edge && !boundary && k == 0 && 0 < n then
      [-n, n].flatMap fun v => (visit n edge k true).map (v :: ·)
    else
      (irange (-n) n).flatMap fun v => (visit n edge k (boundary || decide (|v| = n))).map (v :: ·)

/-- Points of the shell with Chebyshev radius `n`, or of the whole cube when `edge = false`. -/
def cube (dim : ℕ) (n : ℤ) (edge : Bool) : List (List ℤ) := visit n edge dim false

theorem mem_visit_edge {n : ℤ} (hn : 0 ≤ n) (k : ℕ) (b : Bool) (x : List ℤ) :
    x ∈ visit n true (k + 1) b ↔
      x.length = k + 1 ∧ (∀ v ∈ x, |v| ≤ n) ∧ (b = true ∨ ∃ v ∈ x, |v| = n) := by
  induction k generalizing b x with
  | zero =>
    by_cases h : b = false ∧ 0 < n
    · obtain ⟨rfl, hpos⟩ := h
      simp only [visit, hpos, List.flatMap_cons, List.flatMap_nil, List.map_cons, List.map_nil,
        Bool.not_false, Bool.and_self, BEq.rfl, decide_true, ↓reduceIte, List.append_nil,
        List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil, or_false, zero_add,
        Bool.false_eq_true, false_or]
      constructor
      · rintro (rfl | rfl) <;> simp [abs_of_pos hpos]
      · rintro ⟨hlen, hall, v, hv, habs⟩
        match x, hlen with
        | [w], _ =>
          simp only [List.mem_singleton] at hv; subst hv
          rcases abs_eq hn |>.mp habs with h | h <;> simp [h]
    · simp only [visit]
      rw [ite_eq_right_of_eq_false _ _ (by simpa [not_and_or, Bool.not_eq_false] using h)]
      simp only [List.mem_flatMap, List.mem_map, List.mem_singleton, mem_irange]
      constructor
      · rintro ⟨v, hv, _, rfl, rfl⟩
        refine ⟨rfl, by simpa [abs_le] using hv, ?_⟩
        rcases b with _ | _
        · have hn0 : n = 0 := by simp at h; omega
          have hv0 : v = 0 := by omega
          exact Or.inr ⟨v, by simp, by simp [hv0, hn0]⟩
        · exact Or.inl rfl
      · rintro ⟨hlen, hall, hb⟩
        match x, hlen with
        | [w], _ => exact ⟨w, abs_le.mp (hall w (by simp)), [], rfl, rfl⟩
  | succ k ih =>
    simp only [visit]
    rw [ite_eq_right_of_eq_false _ _ (by simp)]
    simp only [List.mem_flatMap, List.mem_map, mem_irange]
    constructor
    · rintro ⟨v, hv, y, hy, rfl⟩
      obtain ⟨hlen, hall, hb⟩ := (ih _ y).mp hy
      refine ⟨by simp [hlen], ?_, ?_⟩
      · intro w hw
        rcases List.mem_cons.mp hw with rfl | hw
        · exact abs_le.mpr hv
        · exact hall w hw
      · rcases hb with hb | ⟨w, hw, hw'⟩
        · rcases b with _ | _ <;> simp_all
        · exact Or.inr ⟨w, List.mem_cons_of_mem _ hw, hw'⟩
    · rintro ⟨hlen, hall, hb⟩
      match x, hlen with
      | v :: y, hlen =>
        refine ⟨v, abs_le.mp (hall v (by simp)), y, (ih _ y).mpr ⟨by simpa using hlen,
          fun w hw => hall w (List.mem_cons_of_mem _ hw), ?_⟩, rfl⟩
        rcases hb with hb | ⟨w, hw, hw'⟩
        · simp [hb]
        · rcases List.mem_cons.mp hw with rfl | hw
          · simp [hw']
          · exact Or.inr ⟨w, hw, hw'⟩

theorem mem_visit_full {n : ℤ} (k : ℕ) (b : Bool) (x : List ℤ) :
    x ∈ visit n false k b ↔ x.length = k ∧ ∀ v ∈ x, |v| ≤ n := by
  induction k generalizing b x with
  | zero => cases x <;> simp [visit]
  | succ k ih =>
    simp only [visit, Bool.false_and, Bool.false_eq_true, ite_false, List.mem_flatMap,
      List.mem_map, mem_irange, ih]
    constructor
    · rintro ⟨v, hv, y, ⟨hlen, hall⟩, rfl⟩
      refine ⟨by simp [hlen], fun w hw => ?_⟩
      rcases List.mem_cons.mp hw with rfl | hw
      · exact abs_le.mpr hv
      · exact hall w hw
    · rintro ⟨hlen, hall⟩
      match x, hlen with
      | v :: y, hlen =>
        exact ⟨v, abs_le.mp (hall v (by simp)), y,
          ⟨by simpa using hlen, fun w hw => hall w (List.mem_cons_of_mem _ hw)⟩, rfl⟩

theorem nodup_visit (n : ℤ) (edge : Bool) : ∀ k b, (visit n edge k b).Nodup
  | 0, _ => by simp [visit]
  | k + 1, b => by
    simp only [visit]
    split_ifs with h
    · simp only [Bool.and_eq_true, decide_eq_true_eq] at h
      rw [List.nodup_flatMap]
      refine ⟨fun v _ => (nodup_visit n edge k true).map List.cons_injective, ?_⟩
      simp only [List.pairwise_cons, List.mem_singleton, forall_eq, List.not_mem_nil,
        IsEmpty.forall_iff, implies_true, List.Pairwise.nil, and_true]
      intro x h1 h2
      obtain ⟨_, _, rfl⟩ := List.mem_map.mp h1
      obtain ⟨_, _, h⟩ := List.mem_map.mp h2
      have := (List.cons.inj h).1
      omega
    · rw [List.nodup_flatMap]
      refine ⟨fun v _ => (nodup_visit n edge k _).map List.cons_injective,
        (nodup_irange _ _).imp fun hne x h1 h2 => ?_⟩
      obtain ⟨_, _, rfl⟩ := List.mem_map.mp h1
      obtain ⟨_, _, h⟩ := List.mem_map.mp h2
      exact hne (List.cons.inj h).1.symm

/-- A shell holds exactly the points of Chebyshev norm `n`. -/
theorem mem_shell {dim : ℕ} (hdim : 0 < dim) {n : ℤ} (hn : 0 ≤ n) (x : List ℤ) :
    x ∈ cube dim n true ↔ x.length = dim ∧ (∀ v ∈ x, |v| ≤ n) ∧ ∃ v ∈ x, |v| = n := by
  obtain ⟨k, rfl⟩ : ∃ k, dim = k + 1 := ⟨dim - 1, by omega⟩
  simp [cube, mem_visit_edge hn]

/-- `Shells::sum` with radii `0..maximum` adds every point of Chebyshev norm below `maximum`
exactly once. -/
theorem shells_partition {dim : ℕ} (hdim : 0 < dim) (maximum : ℕ) :
    let all := (List.range maximum).flatMap fun r : ℕ => cube dim (r : ℤ) true
    all.Nodup ∧ ∀ x, x ∈ all ↔ x.length = dim ∧ ∀ v ∈ x, |v| < maximum := by
  refine ⟨?_, fun x => ?_⟩
  · rw [List.nodup_flatMap]
    refine ⟨fun r _ => nodup_visit _ _ _ _, List.nodup_range.imp fun {r s} hne x h1 h2 => ?_⟩
    obtain ⟨_, h1, v, hv, hv'⟩ := (mem_shell hdim (by positivity) x).mp h1
    obtain ⟨_, h2, w, hw, hw'⟩ := (mem_shell hdim (by positivity) x).mp h2
    have := h1 w hw
    have := h2 v hv
    omega
  · simp only [List.mem_flatMap, List.mem_range]
    constructor
    · rintro ⟨r, hr, h⟩
      obtain ⟨hlen, hall, -⟩ := (mem_shell hdim (by positivity) x).mp h
      exact ⟨hlen, fun v hv => by have := hall v hv; omega⟩
    · rintro ⟨hlen, hall⟩
      have hne : x.toFinset.Nonempty := by
        rw [List.toFinset_nonempty_iff]; rintro rfl; simp at hlen; omega
      obtain ⟨v, hv, hmax⟩ := Finset.exists_max_image x.toFinset (fun v : ℤ => |v|) hne
      rw [List.mem_toFinset] at hv
      have hlt := hall v hv
      have h0 : 0 ≤ |v| := abs_nonneg v
      refine ⟨|v|.toNat, by omega, (mem_shell hdim (by positivity) x).mpr
        ⟨hlen, fun w hw => ?_, v, hv, ?_⟩⟩
      · have := hmax w (List.mem_toFinset.mpr hw); omega
      · omega

end Treams.Shells

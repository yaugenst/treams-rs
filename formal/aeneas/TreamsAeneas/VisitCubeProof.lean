import VisitCube
import TreamsAeneas.Shells

open Aeneas Aeneas.Std Result treams_core

namespace Treams.AeneasVisitCube

abbrev Point := Aeneas.Std.Array Std.I64 3#usize

/-- Values an `i64` inclusive range has left to yield. -/
def rem (r : core.ops.range.RangeInclusive Std.I64) : List ℤ :=
  if r.exhausted then [] else irange r.start.val r.«end».val

theorem irange_nil {a b : ℤ} (h : b < a) : irange a b = [] := by
  simp [irange]; omega

theorem irange_cons {a b : ℤ} (h : a ≤ b) : irange a b = a :: irange (a + 1) b := by
  unfold irange
  rw [show (b + 1 - a).toNat = (b + 1 - (a + 1)).toNat + 1 by omega, List.range_succ_eq_map]
  simp only [List.map_cons, List.map_map]
  congr 1
  · simp
  · apply List.map_congr_left; intro i _; simp; ring

abbrev next := core.ops.range.RangeInclusive.Insts.CoreIterTraitsIteratorIterator.next
  core.iter.range.StepI64

theorem next_rem (r : core.ops.range.RangeInclusive Std.I64) :
    match rem r with
    | [] => next r = ok (none, r)
    | v :: vs => ∃ r', next r = ok (some r.start, r') ∧ r.start.val = v ∧ rem r' = vs := by
  unfold next core.ops.range.RangeInclusive.Insts.CoreIterTraitsIteratorIterator.next
    core.ops.range.RangeInclusive.is_empty
  simp only [core.iter.range.IScalarStep, core.iter.range.StepI64, liftFun2, liftFun1,
    core.cmp.impls.PartialOrdI64.le, core.cmp.impls.PartialOrdI64.lt,
    core.clone.impls.CloneI64.clone, core.iter.range.IScalarStep.forward_checked]
  by_cases hex : r.exhausted = true
  · simp [rem, hex]
  have hex' : r.exhausted = false := by simpa using hex
  rcases lt_trichotomy r.«end».val r.start.val with hlt | heq | hgt
  · have : rem r = [] := by simp [rem, hex', irange_nil hlt]
    rw [this]
    simp [hex', show ¬ r.start.val ≤ r.«end».val by omega]
  · have : rem r = [r.start.val] := by
      simp [rem, hex', irange_cons (le_of_eq heq.symm), irange_nil (show r.«end».val < r.start.val + 1 by omega)]
    rw [this]
    refine ⟨{ start := r.start, «end» := r.«end», exhausted := true }, ?_, rfl, by simp [rem]⟩
    simp [hex', show r.start.val ≤ r.«end».val by omega, show ¬ r.start.val < r.«end».val by omega]
  · have : rem r = r.start.val :: irange (r.start.val + 1) r.«end».val := by
      simp [rem, hex', irange_cons hgt.le]
    rw [this]
    have hlt : r.start.val < I64.max := by
      have := r.«end».hBounds; scalar_tac
    have hb : -2 ^ (IScalarTy.I64.numBits - 1) ≤ r.start.val + 1 ∧
        r.start.val + 1 < 2 ^ (IScalarTy.I64.numBits - 1) := by
      have := r.start.hBounds; scalar_tac
    let r' : core.ops.range.RangeInclusive Std.I64 :=
      ⟨IScalar.ofIntCore (r.start.val + 1) hb, r.«end», r.exhausted⟩
    refine ⟨r', ?_, rfl, ?_⟩
    · simp [hex', hgt.le, hgt, hlt, r']
    · simp [rem, r', hex']

/-- A visitor that records every point it receives. -/
def recorder : core.ops.function.FnMut (List Point) Point (core.result.Result Unit treams_core.Error) where
  FnOnceInst := ⟨fun _ _ => ok (.Ok ())⟩
  call_mut := fun s p => ok (.Ok (), s ++ [p])

def ints (p : Point) : List ℤ := p.val.map (·.val)

theorem length_ints (p : Point) : (ints p).length = 3 := by
  simp [ints]

/-- Recorded points: each suffix spliced between the fixed prefix and tail of `p`. -/
def spliced (p : List ℤ) (axis dim : ℕ) (sufs : List (List ℤ)) : List (List ℤ) :=
  sufs.map fun suf => p.take axis ++ suf ++ p.drop dim

/-- `append` succeeds, keeps the coordinates outside `axis..dim`, and records `sufs`. -/
def Post (point : Point) (axis dim : ℕ) (sufs : List (List ℤ)) (s : List Point)
    (res : core.result.Result Unit treams_core.Error × Point × List Point) : Prop :=
  res.1 = .Ok () ∧ (ints res.2.1).take axis = (ints point).take axis ∧
    (ints res.2.1).drop dim = (ints point).drop dim ∧
    res.2.2.map ints = s.map ints ++ spliced (ints point) axis dim sufs

theorem ints_set (p : Point) (i : Usize) (x : Std.I64) :
    ints (p.set i x) = (ints p).set i.val x.val := by
  simp [ints, Aeneas.Std.Array.set, List.map_set]

theorem take_set_succ {l : List ℤ} {i : ℕ} {x : ℤ} (h : i < l.length) :
    (l.set i x).take (i + 1) = l.take i ++ [x] := by
  rw [List.take_add_one, List.getElem?_set_self h]
  simp [List.take_set_of_le (le_refl i)]

theorem splice_step {p : List ℤ} {axis dim : ℕ} {v : ℤ} (hax : axis < dim)
    (hdim : dim ≤ p.length) (sufs : List (List ℤ)) :
    spliced (p.set axis v) (axis + 1) dim sufs = spliced p axis dim (sufs.map (v :: ·)) := by
  simp only [spliced, List.map_map]
  apply List.map_congr_left
  intro suf _
  simp only [Function.comp, take_set_succ (show axis < p.length by omega),
    List.drop_set_of_lt (hnm := hax)]
  simp

theorem succ_axis {axis : Usize} (h : axis.val < 3) :
    ∃ i : Usize, axis + 1#usize = ok i ∧ i.val = axis.val + 1 := by
  have := WP.spec_imp_exists (UScalar.add_spec (x := axis) (y := 1#usize) (by scalar_tac))
  simpa using this

theorem update_ok (point : Point) {axis : Usize} (h : axis.val < 3) (x : Std.I64) :
    Aeneas.Std.Array.update point axis x = ok (point.set axis x) := by
  have := WP.spec_imp_exists (Aeneas.Std.Array.update_spec point axis x (by simpa using h))
  obtain ⟨y, hy, rfl⟩ := this
  exact hy

theorem neg_ok (x : Std.I64) (h : I64.min < x.val) : ∃ y, (-. x) = ok y ∧ y.val = -x.val := by
  have := WP.spec_imp_exists (IScalar.neg_step x (by scalar_tac))
  have h' : ∃ y, IScalar.neg x = ok y ∧ y.val = -x.val := by simpa using this
  exact h'

theorem abs_ok (x : Std.I64) (h : I64.min < x.val) :
    ∃ y, core.num.I64.abs x = ok y ∧ y.val = |x.val| := by
  unfold core.num.I64.abs
  split_ifs with hneg
  · obtain ⟨y, hy, hv⟩ := neg_ok x h
    exact ⟨y, hy, by rw [hv, abs_of_neg hneg]⟩
  · exact ⟨x, rfl, (abs_of_nonneg (not_lt.mp hneg)).symm⟩

theorem decide_eq_val {a n : Std.I64} : decide (a = n) = decide (a.val = n.val) := by
  simp [IScalar.eq_equiv]

/-- The recursive call's contract at the next axis, for fixed `edge`. -/
def Next (dim : Usize) (axis : ℕ) (n : Std.I64) (edge : Bool) (k : ℕ) : Prop :=
  ∀ (point : Point) (i : Usize) (boundary : Bool) (s : List Point), i.val = axis + 1 →
    ∃ res, geometry.visit_cube.append recorder point dim i n edge boundary s = ok res ∧
      Post point (axis + 1) dim.val (Shells.visit n.val edge k boundary) s res

theorem post_step {point : Point} {axis dim : ℕ} {v : Std.I64} {hi : Usize}
    (hax : axis < dim) (hdim : dim ≤ 3) {S T : List (List ℤ)} {s : List Point} {r1 r2}
    (h1 : Post (point.set hi v) (axis + 1) dim S s r1) (hhi : hi.val = axis)
    (h2 : Post r1.2.1 axis dim T r1.2.2 r2) :
    Post point axis dim (S.map (v.val :: ·) ++ T) s r2 := by
  obtain ⟨-, t1, d1, s1⟩ := h1
  obtain ⟨o2, t2, d2, s2⟩ := h2
  have hlen := length_ints point
  rw [ints_set, hhi] at t1 d1 s1
  have t1' : (ints r1.2.1).take axis = (ints point).take axis := by
    have := congrArg (List.take axis) t1
    simp only [List.take_take, min_eq_left (Nat.le_succ axis)] at this
    rw [this, List.take_set_of_le (le_refl axis)]
  have d1' : (ints r1.2.1).drop dim = (ints point).drop dim := by
    rw [d1, List.drop_set_of_lt (hnm := hax)]
  refine ⟨o2, t2.trans t1', d2.trans d1', ?_⟩
  rw [s2, s1, splice_step hax (by omega)]
  simp only [spliced, List.map_append, List.append_assoc, t1', d1']

theorem loop0_spec {dim axis : Usize} {n : Std.I64} {k : ℕ} (hax : axis.val < dim.val)
    (hdim : dim.val ≤ 3) (IH : Next dim axis.val n true k) :
    ∀ (vs : List ℤ) (iter) (point : Point) (s : List Point), rem iter = vs →
      ∃ res, geometry.visit_cube.append_loop0 recorder iter point dim axis n s = ok res ∧
        Post point axis.val dim.val
          (vs.flatMap fun v => (Shells.visit n.val true k true).map (v :: ·)) s res := by
  intro vs
  induction vs with
  | nil =>
    intro iter point s h
    have hn := next_rem iter
    rw [h] at hn
    rw [geometry.visit_cube.append_loop0]
    simp only [next] at hn
    simp only [hn, bind_tc_ok]
    exact ⟨_, rfl, rfl, rfl, rfl, by simp [spliced]⟩
  | cons v vs ihv =>
    intro iter point s h
    have hn := next_rem iter
    rw [h] at hn
    obtain ⟨iter', hn, hv, hrem⟩ := hn
    obtain ⟨i, hi, hival⟩ := succ_axis (axis := axis) (by omega)
    obtain ⟨r1, hcall, hpost1⟩ := IH (point.set axis iter.start) i true s hival
    obtain ⟨r2, hloop, hpost2⟩ := ihv iter' r1.2.1 r1.2.2 hrem
    refine ⟨r2, ?_, ?_⟩
    · rw [geometry.visit_cube.append_loop0]
      simp only [next] at hn
      obtain ⟨o1, p1, s1⟩ := r1
      obtain ⟨ho1, -⟩ := hpost1
      simp only at ho1
      subst ho1
      simpa [hn, update_ok point (show axis.val < 3 by omega), hi, hcall,
        core.result.Result.Insts.CoreOpsTry.branch] using hloop
    · rw [List.flatMap_cons]
      rw [← hv]
      exact post_step hax hdim hpost1 rfl hpost2

theorem loop1_spec {dim axis : Usize} {n : Std.I64} {k : ℕ} (hax : axis.val < dim.val)
    (hdim : dim.val ≤ 3) (IH : Next dim axis.val n true k) :
    ∀ (iter : List Std.I64) (point : Point) (s : List Point),
      ∃ res, geometry.visit_cube.append_loop1 recorder iter point dim axis n s = ok res ∧
        Post point axis.val dim.val
          ((iter.map (·.val)).flatMap fun v => (Shells.visit n.val true k true).map (v :: ·))
          s res := by
  intro iter
  induction iter with
  | nil =>
    intro point s
    rw [geometry.visit_cube.append_loop1]
    simp only [core.array.iter.IntoIter.Insts.CoreIterTraitsIteratorIterator.next, bind_tc_ok]
    exact ⟨_, rfl, rfl, rfl, rfl, by simp [spliced]⟩
  | cons x xs ihx =>
    intro point s
    obtain ⟨i, hi, hival⟩ := succ_axis (axis := axis) (by omega)
    obtain ⟨r1, hcall, hpost1⟩ := IH (point.set axis x) i true s hival
    obtain ⟨r2, hloop, hpost2⟩ := ihx r1.2.1 r1.2.2
    refine ⟨r2, ?_, ?_⟩
    · rw [geometry.visit_cube.append_loop1]
      obtain ⟨o1, p1, s1⟩ := r1
      obtain ⟨ho1, -⟩ := hpost1
      simp only at ho1
      subst ho1
      simpa [core.array.iter.IntoIter.Insts.CoreIterTraitsIteratorIterator.next,
        update_ok point (show axis.val < 3 by omega), hi, hcall,
        core.result.Result.Insts.CoreOpsTry.branch] using hloop
    · rw [List.map_cons, List.flatMap_cons]
      exact post_step hax hdim hpost1 rfl hpost2

theorem loop2_spec {dim axis : Usize} {n : Std.I64} {k : ℕ} (hax : axis.val < dim.val)
    (hdim : dim.val ≤ 3) (IH : Next dim axis.val n true k) :
    ∀ (vs : List ℤ) (iter) (point : Point) (s : List Point), rem iter = vs →
      (∀ v ∈ vs, I64.min < v) →
      ∃ res, geometry.visit_cube.append_loop2 recorder iter point dim axis n s = ok res ∧
        Post point axis.val dim.val
          (vs.flatMap fun v => (Shells.visit n.val true k (decide (|v| = n.val))).map (v :: ·))
          s res := by
  intro vs
  induction vs with
  | nil =>
    intro iter point s h _
    have hn := next_rem iter
    rw [h] at hn
    rw [geometry.visit_cube.append_loop2]
    simp only [next] at hn
    simp only [hn, bind_tc_ok]
    exact ⟨_, rfl, rfl, rfl, rfl, by simp [spliced]⟩
  | cons v vs ihv =>
    intro iter point s h hmin
    have hn := next_rem iter
    rw [h] at hn
    obtain ⟨iter', hn, hv, hrem⟩ := hn
    obtain ⟨i, hi, hival⟩ := succ_axis (axis := axis) (by omega)
    obtain ⟨a, ha, haval⟩ := abs_ok iter.start (by rw [hv]; exact hmin v (by simp))
    obtain ⟨r1, hcall, hpost1⟩ := IH (point.set axis iter.start) i (decide (a = n)) s hival
    obtain ⟨r2, hloop, hpost2⟩ := ihv iter' r1.2.1 r1.2.2 hrem
      (fun w hw => hmin w (List.mem_cons_of_mem _ hw))
    have hb : decide (a = n) = decide (|v| = n.val) := by rw [decide_eq_val, haval, hv]
    refine ⟨r2, ?_, ?_⟩
    · rw [geometry.visit_cube.append_loop2]
      simp only [next] at hn
      obtain ⟨o1, p1, s1⟩ := r1
      obtain ⟨ho1, -⟩ := hpost1
      simp only at ho1
      subst ho1
      simpa [hn, update_ok point (show axis.val < 3 by omega), hi, ha, hcall,
        core.result.Result.Insts.CoreOpsTry.branch] using hloop
    · rw [List.flatMap_cons, ← hb, ← hv]
      exact post_step hax hdim hpost1 rfl hpost2

theorem loop3_spec {dim axis : Usize} {n : Std.I64} {k : ℕ} (hax : axis.val < dim.val)
    (hdim : dim.val ≤ 3) (IH : Next dim axis.val n true k) :
    ∀ (vs : List ℤ) (iter) (point : Point) (s : List Point), rem iter = vs →
      (∀ v ∈ vs, I64.min < v) →
      ∃ res, geometry.visit_cube.append_loop3 recorder iter point dim axis n s = ok res ∧
        Post point axis.val dim.val
          (vs.flatMap fun v => (Shells.visit n.val true k (decide (|v| = n.val))).map (v :: ·))
          s res := by
  intro vs
  induction vs with
  | nil =>
    intro iter point s h _
    have hn := next_rem iter
    rw [h] at hn
    rw [geometry.visit_cube.append_loop3]
    simp only [next] at hn
    simp only [hn, bind_tc_ok]
    exact ⟨_, rfl, rfl, rfl, rfl, by simp [spliced]⟩
  | cons v vs ihv =>
    intro iter point s h hmin
    have hn := next_rem iter
    rw [h] at hn
    obtain ⟨iter', hn, hv, hrem⟩ := hn
    obtain ⟨i, hi, hival⟩ := succ_axis (axis := axis) (by omega)
    obtain ⟨a, ha, haval⟩ := abs_ok iter.start (by rw [hv]; exact hmin v (by simp))
    obtain ⟨r1, hcall, hpost1⟩ := IH (point.set axis iter.start) i (decide (a = n)) s hival
    obtain ⟨r2, hloop, hpost2⟩ := ihv iter' r1.2.1 r1.2.2 hrem
      (fun w hw => hmin w (List.mem_cons_of_mem _ hw))
    have hb : decide (a = n) = decide (|v| = n.val) := by rw [decide_eq_val, haval, hv]
    refine ⟨r2, ?_, ?_⟩
    · rw [geometry.visit_cube.append_loop3]
      simp only [next] at hn
      obtain ⟨o1, p1, s1⟩ := r1
      obtain ⟨ho1, -⟩ := hpost1
      simp only at ho1
      subst ho1
      simpa [hn, update_ok point (show axis.val < 3 by omega), hi, ha, hcall,
        core.result.Result.Insts.CoreOpsTry.branch] using hloop
    · rw [List.flatMap_cons, ← hb, ← hv]
      exact post_step hax hdim hpost1 rfl hpost2

theorem loop4_spec {dim axis : Usize} {n : Std.I64} {k : ℕ} (hax : axis.val < dim.val)
    (hdim : dim.val ≤ 3) (IH : Next dim axis.val n false k) (b : Bool) :
    ∀ (vs : List ℤ) (iter) (point : Point) (s : List Point), rem iter = vs →
      (∀ v ∈ vs, I64.min < v) →
      ∃ res, geometry.visit_cube.append_loop4 recorder iter point dim axis n b s = ok res ∧
        Post point axis.val dim.val
          (vs.flatMap fun v =>
            (Shells.visit n.val false k (b || decide (|v| = n.val))).map (v :: ·)) s res := by
  intro vs
  induction vs with
  | nil =>
    intro iter point s h _
    have hn := next_rem iter
    rw [h] at hn
    rw [geometry.visit_cube.append_loop4]
    simp only [next] at hn
    simp only [hn, bind_tc_ok]
    exact ⟨_, rfl, rfl, rfl, rfl, by simp [spliced]⟩
  | cons v vs ihv =>
    intro iter point s h hmin
    have hn := next_rem iter
    rw [h] at hn
    obtain ⟨iter', hn, hv, hrem⟩ := hn
    obtain ⟨i, hi, hival⟩ := succ_axis (axis := axis) (by omega)
    obtain ⟨a, ha, haval⟩ := abs_ok iter.start (by rw [hv]; exact hmin v (by simp))
    have hb : decide (a = n) = decide (|v| = n.val) := by rw [decide_eq_val, haval, hv]
    obtain ⟨r1, hcall, hpost1⟩ :=
      IH (point.set axis iter.start) i (b || decide (a = n)) s hival
    obtain ⟨r2, hloop, hpost2⟩ := ihv iter' r1.2.1 r1.2.2 hrem
      (fun w hw => hmin w (List.mem_cons_of_mem _ hw))
    refine ⟨r2, ?_, ?_⟩
    · rw [geometry.visit_cube.append_loop4]
      simp only [next] at hn
      obtain ⟨o1, p1, s1⟩ := r1
      obtain ⟨ho1, -⟩ := hpost1
      simp only at ho1
      subst ho1
      cases b
      · simp only [Bool.false_or] at hcall
        simpa [hn, update_ok point (show axis.val < 3 by omega), hi, ha, hcall,
          core.result.Result.Insts.CoreOpsTry.branch] using hloop
      · simp only [Bool.true_or] at hcall
        simpa [hn, update_ok point (show axis.val < 3 by omega), hi, hcall,
          core.result.Result.Insts.CoreOpsTry.branch] using hloop
    · rw [List.flatMap_cons, ← hb, ← hv]
      exact post_step hax hdim hpost1 rfl hpost2

theorem irange_min {n : Std.I64} (hn : 0 ≤ n.val) : ∀ v ∈ irange (-n.val) n.val, I64.min < v := by
  intro v hv
  have := (mem_irange.mp hv).1
  have := n.hBounds
  scalar_tac

/-- The Rust `append` records exactly the points of the Lean shell model `Shells.visit`,
spliced into the coordinates it leaves fixed, and returns `Ok`. -/
theorem append_spec : ∀ (k : ℕ) (point : Point) (dim axis : Usize) (n : Std.I64)
    (edge boundary : Bool) (s : List Point), dim.val ≤ 3 → axis.val + k = dim.val → 0 ≤ n.val →
    ∃ res, geometry.visit_cube.append recorder point dim axis n edge boundary s = ok res ∧
      Post point axis.val dim.val (Shells.visit n.val edge k boundary) s res := by
  intro k
  induction k with
  | zero =>
    intro point dim axis n edge boundary s hdim hk hn
    have had : axis = dim := by scalar_tac
    subst had
    rw [geometry.visit_cube.append]
    simp only [↓reduceIte, recorder, bind_tc_ok]
    refine ⟨_, rfl, rfl, rfl, rfl, ?_⟩
    simp [spliced, Shells.visit]
  | succ k ih =>
    intro point dim axis n edge boundary s hdim hk hn
    have hax : axis.val < dim.val := by omega
    have hne : ¬ axis = dim := by intro h; subst h; omega
    have hnext : ∀ e, Next dim axis.val n e k := by
      intro e point' i b s' hi
      have := ih point' dim i n e b s' hdim (by omega) hn
      rw [hi] at this
      exact this
    obtain ⟨m, hm, hmval⟩ := neg_ok n (by have := n.hBounds; scalar_tac)
    have hrem : rem ⟨m, n, false⟩ = irange (-n.val) n.val := by simp [rem, hmval]
    have hmin := irange_min hn
    rw [geometry.visit_cube.append]
    simp only [hne, ↓reduceIte, hm, bind_tc_ok, core.ops.range.RangeInclusive.new]
    rw [Shells.visit]
    cases edge
    · simpa using loop4_spec hax hdim (hnext false) boundary _ _ point s hrem hmin
    cases boundary
    · obtain ⟨i, hi, hival⟩ := succ_axis (axis := axis) (by omega)
      simp only [hi, bind_tc_ok]
      by_cases hk0 : k = 0
      · subst hk0
        have hid : i = dim := by scalar_tac
        simp only [hid, ↓reduceIte]
        by_cases hpos : 0 < n.val
        · have hgt : n > 0#i64 := by scalar_tac
          simp only [hgt, ↓reduceIte, bind_tc_ok,
            Array.Insts.CoreIterTraitsCollectIntoIteratorTIntoIter.into_iter]
          have := loop1_spec hax hdim (hnext true) [m, n] point s
          simpa [hpos, hmval] using this
        · have hgt : ¬ n > 0#i64 := by scalar_tac
          simp only [hgt, ↓reduceIte]
          simpa [hpos] using loop2_spec hax hdim (hnext true) _ _ point s hrem hmin
      · have hid : ¬ i = dim := by scalar_tac
        simp only [hid, ↓reduceIte]
        simpa [hk0] using loop3_spec hax hdim (hnext true) _ _ point s hrem hmin
    · simpa using loop0_spec hax hdim (hnext true) _ _ point s hrem

/-- Pad a point of the `dim`-dimensional model with the zero coordinates Rust keeps. -/
def pad (dim : ℕ) (x : List ℤ) : List ℤ := x ++ List.replicate (3 - dim) 0

/-- From the zero point, the Rust recursion records exactly the Lean shell `Shells.cube`. -/
theorem visit_cube_records (dim : Usize) (n : Std.I64) (edge : Bool) (hdim : dim.val ≤ 3)
    (hn : 0 ≤ n.val) :
    ∃ res, geometry.visit_cube.append recorder (Aeneas.Std.Array.repeat 3#usize 0#i64) dim 0#usize
        n edge false [] = ok res ∧ res.1 = .Ok () ∧
      res.2.2.map ints = (Shells.cube dim.val n.val edge).map (pad dim.val) := by
  obtain ⟨res, hres, ho, -, -, hs⟩ := append_spec dim.val _ dim 0#usize n edge false [] hdim
    (by simp) hn
  refine ⟨res, hres, ho, ?_⟩
  rw [hs]
  simp only [spliced, Shells.cube, ints, Aeneas.Std.Array.repeat]
  simp only [List.map_nil, List.nil_append]
  apply List.map_congr_left
  intro a _
  simp only [pad]
  rcases (show dim.val = 0 ∨ dim.val = 1 ∨ dim.val = 2 ∨ dim.val = 3 by omega) with
    h | h | h | h <;> simp [h]

/-- `visit_cube` itself accepts every valid shell request and returns `Ok`. -/
theorem visit_cube_ok (dim : Usize) (n : Std.I64) (edge : Bool)
    (hdim : 1 ≤ dim.val ∧ dim.val ≤ 3) (hn : 0 ≤ n.val) :
    geometry.visit_cube recorder dim n edge [] = ok (.Ok ()) := by
  obtain ⟨res, hres, ho, -⟩ := visit_cube_records dim n edge hdim.2 hn
  obtain ⟨o, p, s⟩ := res
  simp only at ho
  subst ho
  rw [geometry.visit_cube]
  simp [core.ops.range.RangeInclusive.new, core.ops.range.RangeInclusive.contains,
    core.cmp.impls.PartialOrdUsize.le, liftFun2, hdim.1, hdim.2,
    show ¬ n.val < 0 by omega, hres]

theorem pad_injective (d : ℕ) : Function.Injective (pad d) :=
  fun _ _ h => List.append_cancel_right h

/-- The padded shells `0, ..., maximum - 1`, which `visit_cube_records` identifies with the Rust
runs, record every lattice point of Chebyshev norm below `maximum` exactly once. -/
theorem padded_shells_partition {dim : ℕ} (hdim : 0 < dim) (maximum : ℕ) :
    let recorded := (List.range maximum).flatMap fun r : ℕ =>
      (Shells.cube dim (r : ℤ) true).map (pad dim)
    recorded.Nodup ∧
      ∀ x, x ∈ recorded ↔ ∃ y, x = pad dim y ∧ y.length = dim ∧ ∀ v ∈ y, |v| < maximum := by
  obtain ⟨hnodup, hmem⟩ := Shells.shells_partition hdim maximum
  have hflat : ((List.range maximum).flatMap fun r : ℕ =>
      (Shells.cube dim (r : ℤ) true).map (pad dim)) =
      ((List.range maximum).flatMap fun r : ℕ => Shells.cube dim (r : ℤ) true).map (pad dim) := by
    simp [List.map_flatMap]
  refine ⟨?_, fun x => ?_⟩
  · rw [hflat]; exact hnodup.map (pad_injective dim)
  · rw [hflat, List.mem_map]
    constructor
    · rintro ⟨y, hy, rfl⟩; exact ⟨y, rfl, (hmem y).mp hy⟩
    · rintro ⟨y, rfl, h⟩; exact ⟨y, (hmem y).mpr h, rfl⟩

end Treams.AeneasVisitCube

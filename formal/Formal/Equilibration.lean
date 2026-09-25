import Formal.ImplicitAdjoint

/-!
# Equilibrated LU solves

A model of `equilibrate` and `Lu::solve_in_place` / `Lu::solve_adjoint_in_place` in
`crates/treams-core/src/linalg.rs`. Equilibration factors `Â = diag(row) A diag(column)`
with real scales. The forward solve scales the right-hand side by `row`, solves with `Â`,
then scales by `column`; the adjoint solve uses `column`, `Âᴴ`, then `row`.
The transposed branch equilibrates `Aᵀ` and swaps the returned scales.
-/

namespace Treams.Equilibration

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- A real diagonal scale as a complex matrix. -/
noncomputable def scale (s : n → ℝ) : Matrix n n ℂ := diagonal fun i => (s i : ℂ)

/-- The equilibrated operator that `Lu::new` factors. -/
noncomputable def equilibrated (row column : n → ℝ) (A : Matrix n n ℂ) : Matrix n n ℂ :=
  scale row * A * scale column

noncomputable def solve (row column : n → ℝ) (A b : Matrix n n ℂ) : Matrix n n ℂ :=
  scale column * ((equilibrated row column A)⁻¹ * (scale row * b))

noncomputable def solveAdjoint (row column : n → ℝ) (A g : Matrix n n ℂ) : Matrix n n ℂ :=
  scale row * ((equilibrated row column A)ᴴ⁻¹ * (scale column * g))

theorem scale_mul_scale_inv {s : n → ℝ} (hs : ∀ i, s i ≠ 0) :
    scale s * scale (fun i => (s i)⁻¹) = 1 := by
  rw [scale, scale, diagonal_mul_diagonal, ← diagonal_one]
  congr 1
  funext i
  have : (s i : ℂ) ≠ 0 := by exact_mod_cast hs i
  push_cast
  exact mul_inv_cancel₀ this

theorem inv_scale {s : n → ℝ} (hs : ∀ i, s i ≠ 0) : (scale s)⁻¹ = scale fun i => (s i)⁻¹ :=
  inv_eq_right_inv (scale_mul_scale_inv hs)

theorem scale_mul_inv {s : n → ℝ} (hs : ∀ i, s i ≠ 0) : scale s * (scale s)⁻¹ = 1 := by
  rw [inv_scale hs, scale_mul_scale_inv hs]

theorem inv_mul_scale {s : n → ℝ} (hs : ∀ i, s i ≠ 0) : (scale s)⁻¹ * scale s = 1 := by
  rw [inv_scale hs, scale, scale, diagonal_mul_diagonal, ← diagonal_one]
  congr 1
  funext i
  have : (s i : ℂ) ≠ 0 := by exact_mod_cast hs i
  push_cast
  exact inv_mul_cancel₀ this

omit [Fintype n] in
theorem scale_conjTranspose (s : n → ℝ) : (scale s)ᴴ = scale s := by
  simp [scale, diagonal_conjTranspose]

/-- The equilibrated forward solve is the exact solve of `A x = b`. -/
theorem solve_eq {row column : n → ℝ} (hr : ∀ i, row i ≠ 0) (hc : ∀ i, column i ≠ 0)
    (A b : Matrix n n ℂ) : solve row column A b = A⁻¹ * b := by
  unfold solve equilibrated
  rw [Matrix.mul_inv_rev, Matrix.mul_inv_rev]
  simp only [← Matrix.mul_assoc]
  rw [scale_mul_inv hc, Matrix.one_mul, Matrix.mul_assoc _ (scale row)⁻¹, inv_mul_scale hr,
    Matrix.mul_one]

/-- The equilibrated adjoint solve is the exact solve of `Aᴴ y = g`. -/
theorem solveAdjoint_eq {row column : n → ℝ} (hr : ∀ i, row i ≠ 0) (hc : ∀ i, column i ≠ 0)
    (A g : Matrix n n ℂ) : solveAdjoint row column A g = Aᴴ⁻¹ * g := by
  unfold solveAdjoint equilibrated
  rw [conjTranspose_mul, conjTranspose_mul, scale_conjTranspose, scale_conjTranspose,
    Matrix.mul_inv_rev, Matrix.mul_inv_rev]
  simp only [← Matrix.mul_assoc]
  rw [scale_mul_inv hr, Matrix.one_mul, Matrix.mul_assoc _ (scale column)⁻¹,
    inv_mul_scale hc, Matrix.mul_one]

/-- The transposed branch: equilibrating `Aᵀ` with `(row, column)` and transposing back
equilibrates `A` with the swapped scales that `equilibrate` returns. -/
theorem transposed_branch (row column : n → ℝ) (A : Matrix n n ℂ) :
    (equilibrated row column Aᵀ)ᵀ = equilibrated column row A := by
  simp [equilibrated, scale, transpose_mul, Matrix.mul_assoc]

/-- The equilibrated LU path computes the forward solve and the adjoint of the
requested-illumination model, so `ImplicitAdjoint.pullback_correct` covers it. -/
theorem illumination_uses_exact_solves {row column : n → ℝ} (hr : ∀ i, row i ≠ 0)
    (hc : ∀ i, column i ≠ 0) (T C a G : Matrix n n ℂ) :
    solve row column (1 - T * C) (T * a) = ImplicitAdjoint.solution T C a ∧
      solveAdjoint row column (1 - T * C) G = ImplicitAdjoint.adjoint T C G := by
  refine ⟨?_, solveAdjoint_eq hr hc _ _⟩
  rw [solve_eq hr hc, ImplicitAdjoint.solution, Matrix.mul_assoc]

end Treams.Equilibration

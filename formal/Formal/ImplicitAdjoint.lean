import Mathlib

/-!
# Pullback of the requested-illumination solve

A model of `Factor::record` and `Residual::pullback` in
`crates/treams-core/src/illumination.rs` for a dense local matrix `T`, coupling `C` and
incident fields `a`. The forward solve is `X = (I - T C)⁻¹ T a`; the pullback computes

* `adjoint = (I - T C)ᴴ⁻¹ G` (`solve_adjoint_in_place`),
* `incident = Tᴴ adjoint` (`local.apply(&adjoint, true)`),
* `local = adjoint (a + C X)ᴴ`,
* `coupling = incident Xᴴ` (`product_adjoint_right`).

The native pairing is `Re(sum(conj(g) * x)) = Re tr(gᴴ x)`. We prove that these
gradients give the derivative of `t ↦ Re tr(Gᴴ X(T + t dT, C + t dC, a + t da))`.
Incident fields are square here; a Rust call with fewer columns is this case with zero
columns padded into `a` and `G`.
-/

namespace Treams.ImplicitAdjoint

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

attribute [local instance] Matrix.linftyOpNormedRing Matrix.linftyOpNormedAlgebra

/-- `Re(sum(conj(g) * x))`. -/
noncomputable def pairing (G X : Matrix n n ℂ) : ℝ := (trace (Gᴴ * X)).re

noncomputable def solution (T C a : Matrix n n ℂ) : Matrix n n ℂ := (1 - T * C)⁻¹ * T * a

noncomputable def adjoint (T C G : Matrix n n ℂ) : Matrix n n ℂ := (1 - T * C)ᴴ⁻¹ * G

noncomputable def gradIncident (T C G : Matrix n n ℂ) : Matrix n n ℂ := Tᴴ * adjoint T C G

noncomputable def gradLocal (T C a G : Matrix n n ℂ) : Matrix n n ℂ :=
  adjoint T C G * (a + C * solution T C a)ᴴ

noncomputable def gradCoupling (T C a G : Matrix n n ℂ) : Matrix n n ℂ :=
  gradIncident T C G * (solution T C a)ᴴ

/-- The pairing with a fixed cotangent as a continuous real-linear map. -/
noncomputable def pairingCLM (G : Matrix n n ℂ) : Matrix n n ℂ →L[ℝ] ℝ :=
  LinearMap.toContinuousLinearMap
    { toFun := pairing G
      map_add' := fun X Y => by simp [pairing, Matrix.mul_add, trace_add]
      map_smul' := fun r X => by simp [pairing, trace_smul] }

theorem hasDerivAt_line (M dM : Matrix n n ℂ) :
    HasDerivAt (fun t : ℝ => M + t • dM) dM 0 := by
  have := ((hasDerivAt_id (0 : ℝ)).smul_const dM).const_add M
  simp only [id, one_smul] at this
  exact this

/-- The linearized forward solve: `dX = (I - T C)⁻¹ (dT (a + C X) + T dC X + T da)`. -/
theorem hasDerivAt_solution (T C a dT dC da : Matrix n n ℂ) (h : IsUnit (1 - T * C)) :
    HasDerivAt (fun t : ℝ => solution (T + t • dT) (C + t • dC) (a + t • da))
      ((1 - T * C)⁻¹ * (dT * (a + C * solution T C a) + T * dC * solution T C a + T * da)) 0 := by
  have hA :
      HasDerivAt (fun t : ℝ => 1 - (T + t • dT) * (C + t • dC)) (0 - (dT * C + T * dC)) 0 := by
    have := (hasDerivAt_line T dT).mul (hasDerivAt_line C dC)
    simp only [zero_smul, add_zero] at this
    exact (hasDerivAt_const (0 : ℝ) (1 : Matrix n n ℂ)).sub this
  have hinv := HasFDerivAt.comp_hasDerivAt_of_eq (hl := hasFDerivAt_ringInverse (𝕜 := ℝ) h.unit)
    (hf := hA) (hy := by simp)
  have hX := (hinv.mul (hasDerivAt_line T dT)).mul (hasDerivAt_line a da)
  have hfun : (fun t : ℝ => solution (T + t • dT) (C + t • dC) (a + t • da)) =
      ((Ring.inverse ∘ fun t : ℝ => 1 - (T + t • dT) * (C + t • dC)) * fun t => T + t • dT) *
        fun t => a + t • da := by
    funext t
    simp [solution, Matrix.nonsing_inv_eq_ringInverse]
  rw [hfun]
  refine hX.congr_deriv ?_
  simp only [Pi.mul_apply, Function.comp, zero_smul, add_zero,
    ← Matrix.nonsing_inv_eq_ringInverse, _root_.neg_apply,
    ContinuousLinearMap.mulLeftRight_apply, solution]
  rw [Matrix.coe_units_inv, IsUnit.unit_spec]
  simp only [Matrix.mul_add, Matrix.add_mul, Matrix.mul_assoc, zero_sub, Matrix.mul_neg,
    Matrix.neg_mul, neg_neg]
  abel

/-- The adjoint identity: pairing the linearized solve with `G` equals pairing the three
Rust gradients with their parameter directions. -/
theorem adjoint_identity (T C a G dT dC da : Matrix n n ℂ) :
    pairing G ((1 - T * C)⁻¹ * (dT * (a + C * solution T C a) + T * dC * solution T C a +
        T * da)) =
      pairing (gradLocal T C a G) dT + pairing (gradCoupling T C a G) dC +
        pairing (gradIncident T C G) da := by
  have hl : (adjoint T C G)ᴴ = Gᴴ * (1 - T * C)⁻¹ := by
    simp [adjoint, conjTranspose_mul, conjTranspose_nonsing_inv]
  simp only [pairing, gradLocal, gradCoupling, gradIncident, conjTranspose_mul,
    conjTranspose_conjTranspose, Matrix.mul_assoc, hl]
  rw [trace_mul_comm (a + C * solution T C a), trace_mul_comm (solution T C a)]
  simp only [Matrix.mul_add, Matrix.mul_assoc, trace_add, Complex.add_re]

/-- The Rust pullback returns the derivative of the paired forward solve. -/
theorem pullback_correct (T C a G dT dC da : Matrix n n ℂ) (h : IsUnit (1 - T * C)) :
    HasDerivAt (fun t : ℝ => pairing G (solution (T + t • dT) (C + t • dC) (a + t • da)))
      (pairing (gradLocal T C a G) dT + pairing (gradCoupling T C a G) dC +
        pairing (gradIncident T C G) da) 0 := by
  rw [← adjoint_identity]
  exact (pairingCLM G).hasFDerivAt.comp_hasDerivAt (0 : ℝ) (hasDerivAt_solution T C a dT dC da h)

end Treams.ImplicitAdjoint

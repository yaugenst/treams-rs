//! The internal fields between two adjacent stacks: a certified Krylov or pivoted LU
//! solve of `(I - L1 U2) up = rhs`, kept for the pullback.
//!
//! Upstream: the `numpy.linalg.solve` inside `treams.SMatrices.illuminate` with a
//! second S-matrix. The Krylov branch and its contraction test are treams-rs
//! extensions.
//!
//! `L1` is the reflection of the lower stack for waves incident from the positive side
//! (above) and `U2` the reflection of the upper stack for waves incident from the
//! negative side (below), so `up` holds the upgoing fields between the stacks.

mod saved;

use faer::MatRef;
use nalgebra::DMatrix;

use crate::{
    Complex, Error, Result,
    linalg::{GmresOptions, Lu, gmres, norm, product_views, view, view_mut},
    numerics::finite,
};

/// The internal operator `I - lower upper`.
fn internal_operator(lower: MatRef<'_, Complex>, upper: MatRef<'_, Complex>) -> DMatrix<Complex> {
    let mut value = -product_views(lower, upper);
    for i in 0..value.nrows() {
        value[(i, i)] += 1.0;
    }
    value
}

/// What the pullback reuses to solve the adjoint system.
///
/// A certified contraction is invertible, including for zero or dependent right-hand
/// sides, so its Krylov solve needs no factorization. Every other operator keeps the
/// pivoted LU and its singular-system behavior.
#[derive(Debug)]
enum InternalFactor {
    /// Apply the two reflections already saved by the caller, without forming `L U`.
    Reflections,
    /// The operator itself, for Krylov adjoint solves with a dense LU fallback.
    Krylov(DMatrix<Complex>),
    /// The pivoted LU factorization of the operator.
    Lu(Lu),
}

/// The forward solution of the internal system and what its adjoint solve reuses.
#[derive(Debug)]
pub(super) struct InternalSolve {
    /// Solution of the internal system for the forward right-hand sides.
    pub(super) value: DMatrix<Complex>,
    factor: InternalFactor,
}

/// Whether `||L U||_inf < 1` holds for the operator `I - L U`, which proves it invertible.
///
/// Each row sum of `|re| + |im|` over the entries of `L U` rounds upward, so the sums
/// bound the infinity norm from above: rounding can make the test fail, never pass.
fn is_contraction(operator: &DMatrix<Complex>) -> bool {
    let n = operator.nrows();
    let mut rows = vec![0.0_f64; n];
    for (j, column) in operator.as_slice().chunks_exact(n).enumerate() {
        for (i, (&z, row)) in column.iter().zip(&mut rows).enumerate() {
            let delta = if i == j {
                Complex::new(1.0, 0.0) - z
            } else {
                z
            };
            // |re|+|im| bounds the complex modulus. Upward-rounded positive
            // additions keep this an upper bound rather than a norm estimate.
            *row = (*row + (delta.re.abs() + delta.im.abs()).next_up()).next_up();
        }
    }
    rows.iter().all(|&sum| sum < 1.0)
}

/// Positive row sums of `(|re A| + |im A|) weights`, rounded upward.
fn absolute_row_sums(matrix: MatRef<'_, Complex>, weights: &[f64]) -> Vec<f64> {
    let accumulate = |sum: &mut f64, z: Complex, weight: f64| {
        // Exact zeros need no rounding. Preserving them also avoids multiplying
        // artificial subnormals throughout sparse reflection blocks stored densely.
        if z != Complex::default() && weight != 0.0 {
            let term = ((z.re.abs() + z.im.abs()).next_up() * weight).next_up();
            *sum = (*sum + term).next_up();
        }
    };
    let mut sums = vec![0.0_f64; matrix.nrows()];
    if matrix.col_stride() == 1 {
        for (row, sum) in matrix.row_iter().zip(&mut sums) {
            for (&z, &weight) in row.iter().zip(weights) {
                accumulate(sum, z, weight);
            }
        }
    } else {
        for (column, &weight) in matrix.col_iter().zip(weights) {
            for (&z, sum) in column.iter().zip(&mut sums) {
                accumulate(sum, z, weight);
            }
        }
    }
    sums
}

/// Certify `||L U||_inf < 1` using `|L| |U| 1`, in quadratic work.
///
/// The entrywise absolute-value bound ignores cancellations, so failure says
/// nothing about invertibility. Such systems retain the dense-product test below.
fn reflections_contract(lower: MatRef<'_, Complex>, upper: MatRef<'_, Complex>) -> bool {
    let upper_rows = absolute_row_sums(upper, &vec![1.0; upper.ncols()]);
    absolute_row_sums(lower, &upper_rows)
        .iter()
        .all(|&sum| sum < 1.0)
}

/// Apply `I - L U`, or its adjoint, using only thin products.
fn reflection_action(
    lower: MatRef<'_, Complex>,
    upper: MatRef<'_, Complex>,
    x: MatRef<'_, Complex>,
    adjoint: bool,
) -> DMatrix<Complex> {
    let mut result = if adjoint {
        let inner = product_views(lower.adjoint(), x);
        product_views(upper.adjoint(), view(&inner))
    } else {
        let inner = product_views(upper, x);
        product_views(lower, view(&inner))
    };
    for j in 0..result.ncols() {
        for i in 0..result.nrows() {
            result[(i, j)] = x[(i, j)] - result[(i, j)];
        }
    }
    result
}

/// The unrestarted Krylov solve of the operator, or of its adjoint, for a thin batch, or
/// `None` when it fails numerically and the caller should factor the operator instead.
fn internal_iteration(
    rhs: &DMatrix<Complex>,
    apply: impl Fn(MatRef<'_, Complex>) -> DMatrix<Complex>,
) -> Result<Option<DMatrix<Complex>>> {
    // One Krylov iteration applies the operator to the entire thin batch.
    // Its total residual is bounded by the smallest nonzero column's tolerance,
    // which also certifies every individual illumination at 8 epsilon.
    let column_norm = |j| norm(rhs.column(j).as_slice());
    let minimum = (0..rhs.ncols())
        .map(column_norm)
        .filter(|&v| v > 0.0)
        .fold(f64::INFINITY, f64::min);
    if rhs.iter().all(|&z| z == Complex::default()) {
        return Ok(Some(DMatrix::zeros(rhs.nrows(), rhs.ncols())));
    }
    if !minimum.is_finite() {
        return Ok(None);
    }
    let options = GmresOptions {
        rtol: 0.0,
        atol: 8.0 * f64::EPSILON * minimum,
        restart: KRYLOV_STEPS,
        max_iterations: KRYLOV_STEPS,
    };
    match gmres(rhs.as_slice(), options, |x| {
        let x_view = MatRef::from_column_major_slice(x, rhs.nrows(), rhs.ncols());
        let result = apply(x_view);
        Ok(Vec::from(result.data))
    }) {
        Ok((answer, _)) => Ok(Some(DMatrix::from_vec(rhs.nrows(), rhs.ncols(), answer))),
        // Memory and input errors propagate instead of reaching a larger dense allocation.
        Err(error) if error.is_numerical() => Ok(None),
        Err(error) => Err(error),
    }
}

/// Rows from which a certified contraction tries the bounded Krylov solve before
/// falling back to a dense LU factorization.
const ITERATIVE_MIN_ROWS: usize = 512;

/// Most right-hand sides for which the Krylov solve is tried; wider batches factor.
const ITERATIVE_MAX_COLUMNS: usize = 8;

/// Krylov iterations, without restart, of one internal solve before the dense fallback.
const KRYLOV_STEPS: usize = 16;

impl InternalSolve {
    pub(super) fn new(
        lower: MatRef<'_, Complex>,
        upper: MatRef<'_, Complex>,
        rhs: DMatrix<Complex>,
    ) -> Result<Self> {
        Self::with_threshold(lower, upper, rhs, ITERATIVE_MIN_ROWS)
    }

    /// The pivoted LU solve, the branch [`Self::new`] takes whenever the Krylov
    /// conditions fail.
    pub(super) fn direct(
        lower: MatRef<'_, Complex>,
        upper: MatRef<'_, Complex>,
        rhs: DMatrix<Complex>,
    ) -> Result<Self> {
        if rhs.iter().any(|&z| !finite(z)) {
            return Err(Error::Singular);
        }
        Self::lu(internal_operator(lower, upper), rhs)
    }

    /// [`Self::new`] with a lower row threshold, which tests use to reach every
    /// dispatch branch with small systems.
    fn with_threshold(
        lower: MatRef<'_, Complex>,
        upper: MatRef<'_, Complex>,
        rhs: DMatrix<Complex>,
        min_rows: usize,
    ) -> Result<Self> {
        if rhs.iter().any(|&z| !finite(z)) {
            return Err(Error::Singular);
        }
        let thin = rhs.nrows() >= min_rows && rhs.ncols() <= ITERATIVE_MAX_COLUMNS;
        let factored = thin && reflections_contract(lower, upper);
        if factored
            && let Some(value) =
                internal_iteration(&rhs, |x| reflection_action(lower, upper, x, false))?
        {
            return Ok(Self {
                value,
                factor: InternalFactor::Reflections,
            });
        }
        let operator = internal_operator(lower, upper);
        if thin
            && !factored
            && is_contraction(&operator)
            && let Some(value) = internal_iteration(&rhs, |x| product_views(view(&operator), x))?
        {
            return Ok(Self {
                value,
                factor: InternalFactor::Krylov(operator),
            });
        }
        Self::lu(operator, rhs)
    }

    /// Factor the operator and solve for `rhs` in place.
    fn lu(operator: DMatrix<Complex>, mut rhs: DMatrix<Complex>) -> Result<Self> {
        let lu = Lu::new(operator)?;
        lu.solve_in_place(view_mut(&mut rhs))?;
        if rhs.iter().any(|&z| !finite(z)) {
            return Err(Error::Singular);
        }
        Ok(Self {
            value: rhs,
            factor: InternalFactor::Lu(lu),
        })
    }

    /// Solve the adjoint system for `rhs`, reusing the forward solve.
    /// `lower` and `upper` must be the reflection pair used by that forward solve.
    pub(super) fn solve_adjoint(
        &self,
        lower: MatRef<'_, Complex>,
        upper: MatRef<'_, Complex>,
        rhs: DMatrix<Complex>,
    ) -> Result<AdjointSolve<'_>> {
        let adjoint = self.solve_rhs(lower, upper, rhs, true)?;
        Ok(AdjointSolve {
            adjoint,
            value: &self.value,
        })
    }

    /// Solve a tangent right-hand side with the recorded forward operator.
    pub(super) fn solve_forward(
        &self,
        lower: MatRef<'_, Complex>,
        upper: MatRef<'_, Complex>,
        rhs: DMatrix<Complex>,
    ) -> Result<DMatrix<Complex>> {
        self.solve_rhs(lower, upper, rhs, false)
    }

    /// Reuse the recorded solve strategy for either derivative direction.
    fn solve_rhs(
        &self,
        lower: MatRef<'_, Complex>,
        upper: MatRef<'_, Complex>,
        mut rhs: DMatrix<Complex>,
        adjoint: bool,
    ) -> Result<DMatrix<Complex>> {
        let fallback;
        let lu = match &self.factor {
            InternalFactor::Lu(lu) => lu,
            InternalFactor::Reflections => {
                if let Some(value) =
                    internal_iteration(&rhs, |x| reflection_action(lower, upper, x, adjoint))?
                {
                    return Ok(value);
                }
                fallback = Lu::new(internal_operator(lower, upper))?;
                &fallback
            }
            InternalFactor::Krylov(operator) => {
                if let Some(value) = internal_iteration(&rhs, |x| {
                    let operator = view(operator);
                    if adjoint {
                        product_views(operator.adjoint(), x)
                    } else {
                        product_views(operator, x)
                    }
                })? {
                    return Ok(value);
                }
                fallback = Lu::new(operator.clone())?;
                &fallback
            }
        };
        if adjoint {
            lu.solve_adjoint_in_place(view_mut(&mut rhs))?;
        } else {
            lu.solve_in_place(view_mut(&mut rhs))?;
        }
        if rhs.iter().any(|&z| !finite(z)) {
            return Err(Error::Singular);
        }
        Ok(rhs)
    }
}

/// The adjoint solution of an internal system, beside its borrowed forward solution.
#[derive(Debug)]
pub(super) struct AdjointSolve<'a> {
    /// Solution of the adjoint system `(I - L U)^H x = rhs`.
    pub(super) adjoint: DMatrix<Complex>,
    /// Solution of the forward system, borrowed from the [`InternalSolve`].
    pub(super) value: &'a DMatrix<Complex>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg;
    use crate::test_support::{patterned, prop_assert_close, re_dot};
    use proptest::{prelude::*, test_runner::TestCaseError};
    use std::result::Result;

    /// The complex Hermitian pairing `Σ conj(a) b`, from its real and imaginary parts.
    fn pairing(a: &DMatrix<Complex>, b: &DMatrix<Complex>) -> Complex {
        let rotated = a * Complex::i();
        Complex::new(re_dot(a.iter(), b.iter()), re_dot(rotated.iter(), b.iter()))
    }

    /// For a weak `n`-by-`n` reflection pair, the Krylov solves of the operator and of
    /// its adjoint match the dense LU solves and pair as adjoints:
    /// `<g, A^-1 rhs> = <A^-H g, rhs>`.
    fn check_krylov_matches_dense_solve(
        n: usize,
        p: usize,
        seed: f64,
    ) -> Result<(), TestCaseError> {
        let weight = Complex::new(0.1 / f64::from(u32::try_from(n).unwrap()).sqrt(), 0.0);
        let lower = patterned(n, n, seed) * weight;
        let upper = patterned(n, n, seed + 0.4) * weight;
        let rhs = patterned(n, p, seed + 0.2);
        let g = patterned(n, p, seed + 0.7);
        let operator = internal_operator(view(&lower), view(&upper));
        let dense = linalg::solve_owned(operator.clone(), rhs.clone()).unwrap();
        prop_assert!(is_contraction(&operator));
        let value = internal_iteration(&rhs, |x| product_views(view(&operator), x)).unwrap();
        let adjoint =
            internal_iteration(&g, |x| product_views(view(&operator).adjoint(), x)).unwrap();
        let (value, adjoint) = (value.unwrap(), adjoint.unwrap());
        prop_assert_close!(&value, dense.value(), 2e-14 * dense.value().norm());
        let dense_adjoint = dense.adjoint_rhs(g.clone()).unwrap();
        prop_assert_close!(&adjoint, &dense_adjoint, 2e-14 * g.norm());
        let factored = internal_iteration(&rhs, |x| {
            reflection_action(view(&lower), view(&upper), x, false)
        })
        .unwrap()
        .unwrap();
        let factored_adjoint = internal_iteration(&g, |x| {
            reflection_action(view(&lower), view(&upper), x, true)
        })
        .unwrap()
        .unwrap();
        prop_assert_close!(&factored, &value, 2e-14 * value.norm());
        prop_assert_close!(&factored_adjoint, &adjoint, 2e-14 * g.norm());
        prop_assert_close!(
            pairing(&g, &value),
            pairing(&adjoint, &rhs),
            2e-13 * rhs.norm() * g.norm()
        );
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]
        #[test]
        fn reflection_actions_match_dense_solve_and_adjoint(
            n in 3_usize..12, p in 1_usize..4, seed in -2.0_f64..2.0,
        ) {
            check_krylov_matches_dense_solve(n, p, seed)?;
        }
    }

    #[test]
    fn production_threshold_solves_thin_contractions_iteratively() -> Result<(), TestCaseError> {
        // Pin the production threshold from both sides: the same thin contraction
        // iterates at `ITERATIVE_MIN_ROWS` rows and factors one row below it.
        let coefficient = Complex::new(1.0, 0.0) - Complex::new(0.1, 0.03).powi(2);
        for (n, iterates) in [(ITERATIVE_MIN_ROWS, true), (ITERATIVE_MIN_ROWS - 1, false)] {
            let weak = DMatrix::identity(n, n) * Complex::new(0.1, 0.03);
            let rhs = patterned(n, 2, 0.4);
            let solve = InternalSolve::new(view(&weak), view(&weak), rhs.clone()).unwrap();
            prop_assert_eq!(
                matches!(solve.factor, InternalFactor::Reflections),
                iterates
            );
            prop_assert_close!(&solve.value * coefficient, &rhs, 2e-13 * rhs.norm());
            let tangent = solve
                .solve_forward(view(&weak), view(&weak), rhs.clone())
                .unwrap();
            prop_assert_close!(tangent * coefficient, &rhs, 2e-13 * rhs.norm());
            let gradient = solve
                .solve_adjoint(view(&weak), view(&weak), rhs.clone())
                .unwrap()
                .adjoint;
            prop_assert_close!(gradient * coefficient.conj(), &rhs, 2e-13 * rhs.norm());
        }
        Ok(())
    }

    #[test]
    fn cancellation_contractions_keep_the_dense_product_path() -> Result<(), TestCaseError> {
        let n = 24;
        let lower = DMatrix::from_fn(n, n, |_, j| {
            Complex::new(if j % 2 == 0 { 0.1 } else { -0.1 }, 0.0)
        });
        let upper = DMatrix::from_element(n, n, Complex::new(0.1, 0.0));
        let rhs = patterned(n, 2, 0.4);
        prop_assert!(!reflections_contract(view(&lower), view(&upper)));
        let operator = internal_operator(view(&lower), view(&upper));
        prop_assert!(is_contraction(&operator));
        let solve =
            InternalSolve::with_threshold(view(&lower), view(&upper), rhs.clone(), n).unwrap();
        prop_assert!(matches!(solve.factor, InternalFactor::Krylov(_)));
        prop_assert_close!(&operator * &solve.value, &rhs, 2e-13 * rhs.norm());
        let tangent = solve
            .solve_forward(view(&lower), view(&upper), rhs.clone())
            .unwrap();
        prop_assert_close!(&operator * tangent, &rhs, 2e-13 * rhs.norm());
        let adjoint = solve
            .solve_adjoint(view(&lower), view(&upper), rhs.clone())
            .unwrap()
            .adjoint;
        prop_assert_close!(operator.adjoint() * adjoint, &rhs, 2e-13 * rhs.norm());
        Ok(())
    }

    #[test]
    fn reflection_certificate_handles_layouts_and_rejects_unit_gain() -> Result<(), TestCaseError> {
        let n = 12;
        let lower = patterned(n, n, 0.4) * Complex::new(0.02, 0.0);
        let upper = patterned(n, n, 0.9) * Complex::new(0.02, 0.0);
        let lower_rows = lower.transpose();
        let upper_rows = upper.transpose();
        for lower in [view(&lower), view(&lower_rows).transpose()] {
            for upper in [view(&upper), view(&upper_rows).transpose()] {
                prop_assert!(reflections_contract(lower, upper));
                prop_assert!(is_contraction(&internal_operator(lower, upper)));
            }
        }
        let identity = DMatrix::identity(n, n);
        prop_assert!(!reflections_contract(view(&identity), view(&identity)));
        // Reject overflow and NaN rather than certifying from an underestimated sum.
        let huge = &identity * Complex::new(f64::MAX, 0.0);
        prop_assert!(!reflections_contract(view(&huge), view(&huge)));
        Ok(())
    }

    #[test]
    fn thin_large_solve_skips_lu_and_keeps_dense_fallback() -> Result<(), TestCaseError> {
        // Every dispatch branch at a lowered threshold; the production threshold is
        // pinned by `production_threshold_solves_thin_contractions_iteratively`.
        // The cyclic reflection below has n distinct eigenvalues on a circle, so the
        // Krylov trial cannot converge only while n exceeds its iteration budget.
        let (n, min_rows) = (24, 16);
        prop_assert!(n > KRYLOV_STEPS);
        let solve = |lower: &DMatrix<Complex>, upper: &DMatrix<Complex>, rhs, min_rows| {
            InternalSolve::with_threshold(view(lower), view(upper), rhs, min_rows)
        };
        let identity = DMatrix::identity(n, n);
        let weak = &identity * Complex::new(0.1, 0.03);
        let rhs = patterned(n, 2, 0.4);
        let iterative = solve(&weak, &weak, rhs.clone(), min_rows).unwrap();
        prop_assert!(matches!(iterative.factor, InternalFactor::Reflections));
        let coefficient = Complex::new(1.0, 0.0) - Complex::new(0.1, 0.03).powi(2);
        prop_assert_close!(&iterative.value * coefficient, &rhs, 2e-13 * rhs.norm());
        let gradient = iterative
            .solve_adjoint(view(&weak), view(&weak), rhs.clone())
            .unwrap()
            .adjoint;
        prop_assert_close!(gradient * coefficient.conj(), &rhs, 2e-13 * rhs.norm());

        // The threshold is inclusive, and wide batches or short systems use LU.
        let at = solve(&weak, &weak, rhs.clone(), n).unwrap();
        prop_assert!(matches!(at.factor, InternalFactor::Reflections));
        let below = solve(&weak, &weak, rhs.clone(), n + 1).unwrap();
        prop_assert!(matches!(below.factor, InternalFactor::Lu(_)));
        prop_assert_close!(&below.value, &at.value, 2e-13 * rhs.norm());
        let wide = patterned(n, ITERATIVE_MAX_COLUMNS + 1, 0.4);
        let batch = solve(&weak, &weak, wide.clone(), min_rows).unwrap();
        prop_assert!(matches!(batch.factor, InternalFactor::Lu(_)));
        prop_assert_close!(&batch.value * coefficient, &wide, 2e-13 * wide.norm());

        // A strongly reflecting cyclic system needs more iterations than the bounded
        // Krylov trial allows, so it takes the direct solve.
        let mut cyclic = DMatrix::zeros(n, n);
        for i in 0..n {
            cyclic[(i, (i + 1) % n)] = Complex::new(0.95, 0.0);
        }
        let dense = solve(&cyclic, &identity, rhs.clone(), min_rows).unwrap();
        prop_assert!(matches!(dense.factor, InternalFactor::Lu(_)));
        let operator = internal_operator(view(&cyclic), view(&identity));
        prop_assert!(is_contraction(&operator));
        prop_assert_close!(&operator * &dense.value, &rhs, 2e-13 * rhs.norm());
        let eigenvector = DMatrix::from_element(n, 2, Complex::new(1.0, 0.0));
        let easy_forward = solve(&cyclic, &identity, eigenvector, min_rows).unwrap();
        prop_assert!(matches!(easy_forward.factor, InternalFactor::Reflections));
        // A tangent direction can require the same direct fallback as an adjoint.
        let tangent = easy_forward
            .solve_forward(view(&cyclic), view(&identity), rhs.clone())
            .unwrap();
        prop_assert_close!(&operator * tangent, &rhs, 2e-13 * rhs.norm());
        // A forward eigenvector can converge immediately while an unrelated
        // adjoint RHS needs the direct fallback. Certify that path separately.
        let reverse = easy_forward
            .solve_adjoint(view(&cyclic), view(&identity), rhs.clone())
            .unwrap()
            .adjoint;
        prop_assert_close!(operator.adjoint() * reverse, &rhs, 2e-13 * rhs.norm());

        // Singular systems stay errors, including consistent and zero right-hand sides.
        prop_assert!(solve(&identity, &identity, rhs, min_rows).is_err());
        let mut partial = DMatrix::zeros(n, n);
        partial[(0, 0)] = Complex::new(1.0, 0.0);
        let mut in_range = DMatrix::zeros(n, 1);
        in_range[(1, 0)] = Complex::new(1.0, 0.0);
        prop_assert!(solve(&partial, &identity, in_range, min_rows).is_err());
        prop_assert!(solve(&identity, &identity, DMatrix::zeros(n, 1), min_rows).is_err());
        Ok(())
    }
}

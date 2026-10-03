//! Dense linear solves and general complex eigensystems with analytic pullbacks, and
//! restarted GMRES. treams-rs extension.
//!
//! Two libraries share the linear algebra, by one rule:
//!
//! - nalgebra's `DMatrix` stores every dense matrix.
//! - faer factors and multiplies dense complex matrices of any size, through this
//!   module: LU solves ([`solve`], `Lu`), singular values ([`svdvals`]), eigensystems
//!   ([`eig`]) and products (`product` and its variants). `view` and `view_mut` lend a
//!   `DMatrix` to faer without a copy.
//! - Real symmetric eigenproblems use nalgebra's `SymmetricEigen`: `special::wigner`
//!   diagonalizes the tridiagonal generator of the small-d matrix, of size 2l + 1.
//! - Fixed-size matrices of at most 4 x 4 stay in nalgebra: the 3 x 3 inverse of
//!   `lattice::reciprocal`, the closed-form 2 x 2 inverse of `coeffs::mie`, the 4 x 4
//!   and 2 x 2 inverses of `coeffs::mie_cyl` and the 4 x 4 inverse of
//!   `smatrix::interface`.
#![allow(clippy::indexing_slicing)] // Validated matrix dimensions and eigenvector pivots.

mod gmres;

pub use gmres::{Convergence, GmresOptions};
pub(crate) use gmres::{gmres, norm};

use faer::{
    Accum, Conj, MatMut, MatRef, Spec,
    dyn_stack::{MemBuffer, MemStack},
    linalg::{
        lu::partial_pivoting::{factor, solve as lu_solve},
        matmul::matmul,
        solvers::{Eigen, Svd},
    },
    perm::Perm,
    traits::Conjugate,
};
use nalgebra::DMatrix;

use crate::{
    Complex, Error, Result,
    numerics::{finite, ratio},
};

/// Column-major faer view of a nalgebra matrix.
pub(crate) fn view(matrix: &DMatrix<Complex>) -> MatRef<'_, Complex> {
    MatRef::from_column_major_slice(matrix.as_slice(), matrix.nrows(), matrix.ncols())
}

/// Mutable column-major faer view of a nalgebra matrix.
pub(crate) fn view_mut(matrix: &mut DMatrix<Complex>) -> MatMut<'_, Complex> {
    let (rows, cols) = matrix.shape();
    MatMut::from_column_major_slice_mut(matrix.as_mut_slice(), rows, cols)
}

/// `out = left right` with the global parallelism. Pass `.adjoint()` views for
/// conjugate-transposed factors; faer conjugates them inside the product kernel.
pub(crate) fn product_into<L, R>(
    out: MatMut<'_, Complex>,
    left: MatRef<'_, L>,
    right: MatRef<'_, R>,
) where
    L: Conjugate<Canonical = Complex>,
    R: Conjugate<Canonical = Complex>,
{
    matmul(
        out,
        Accum::Replace,
        left,
        right,
        Complex::new(1.0, 0.0),
        faer::get_global_parallelism(),
    );
}

/// `left right` in a new matrix; see [`product_into`].
pub(crate) fn product_views<L, R>(left: MatRef<'_, L>, right: MatRef<'_, R>) -> DMatrix<Complex>
where
    L: Conjugate<Canonical = Complex>,
    R: Conjugate<Canonical = Complex>,
{
    let mut result = DMatrix::zeros(left.nrows(), right.ncols());
    product_into(view_mut(&mut result), left, right);
    result
}

/// `left right`.
pub(crate) fn product(left: &DMatrix<Complex>, right: &DMatrix<Complex>) -> DMatrix<Complex> {
    product_views(view(left), view(right))
}

/// `left rightᴴ`.
pub(crate) fn product_adjoint_right(
    left: &DMatrix<Complex>,
    right: &DMatrix<Complex>,
) -> DMatrix<Complex> {
    product_views(view(left), view(right).adjoint())
}

/// `out = left rightᴴ`, reusing the allocation of `out`.
pub(crate) fn product_adjoint_right_into(
    out: &mut DMatrix<Complex>,
    left: &DMatrix<Complex>,
    right: &DMatrix<Complex>,
) {
    product_into(view_mut(out), view(left), view(right).adjoint());
}

/// Positive diagonal scales for the equivalent operator `R A C`.
///
/// A normal solve scales its right-hand side by `row` and its solution by
/// `column`. A conjugate-transpose solve uses the opposite order. These scales
/// are numerical coordinates, not additional inputs to the implicit derivative.
#[derive(Clone, Debug)]
pub(crate) struct Equilibration {
    /// Left (equation) scale.
    pub(crate) row: Vec<f64>,
    /// Right (unknown) scale.
    pub(crate) column: Vec<f64>,
}

/// Equilibrate severely unbalanced finite square operators in place.
///
/// Small-argument, high-order multipole operators can have unit diagonal and
/// off-diagonal entries exceeding `1e20`. Their unscaled LU loses accuracy even
/// when the equivalent equilibrated system is well conditioned. The scan that
/// rejects non-finite entries also measures the spread of entry sizes. An operator
/// whose nonzero entries, or whose diagonal entries, all lie within a factor of 10 of
/// the largest entry stays unchanged.
pub(crate) fn equilibrate(operator: &mut DMatrix<Complex>) -> Result<Option<Equilibration>> {
    let n = operator.nrows();
    if n == 0 || !operator.is_square() {
        return Err(Error::InvalidInput(
            "require a finite nonempty square operator".into(),
        ));
    }
    // LAPACK xLAQGE uses 0.1 for row/column scale ratios. Waiting until
    // sqrt(epsilon) loses observable digits in moderately unbalanced multipole
    // systems even though a change of coordinates makes them well conditioned.
    let relative_floor = 0.1;
    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0_f64;
    for &z in operator.iter() {
        if !finite(z) {
            return Err(Error::InvalidInput(
                "require a finite nonempty square operator".into(),
            ));
        }
        let magnitude = z.re.abs().max(z.im.abs());
        if magnitude > 0.0 {
            minimum = minimum.min(magnitude);
        }
        maximum = maximum.max(magnitude);
    }
    if maximum > 0.0
        && (minimum / maximum >= relative_floor
            || (0..n).all(|i| {
                let z = operator[(i, i)];
                z.re.abs().max(z.im.abs()) / maximum >= relative_floor
            }))
    {
        return Ok(None);
    }
    // Only potentially unbalanced operators allocate scratch or need another
    // pass. Isolated tiny entries do not by themselves require equilibration.
    let mut row = vec![0.0_f64; n];
    let mut column_min = f64::INFINITY;
    for values in operator.as_slice().chunks_exact(n) {
        let mut column_max = 0.0_f64;
        for (row_max, &z) in row.iter_mut().zip(values) {
            let magnitude = z.re.abs().max(z.im.abs());
            *row_max = row_max.max(magnitude);
            column_max = column_max.max(magnitude);
        }
        column_min = column_min.min(column_max);
    }
    let row_min = row.iter().copied().fold(f64::INFINITY, f64::min);
    if row_min == 0.0 || column_min == 0.0 {
        return Err(Error::Singular);
    }
    if row_min / maximum >= relative_floor && column_min / maximum >= relative_floor {
        return Ok(None);
    }
    if column_min < row_min {
        // Scale the more disparate axis first. Otherwise normalizing large rows
        // can underflow an entire small column before it gets its own scale.
        // Transposition swaps the two measured spreads, so this recurses once.
        operator.transpose_mut();
        let result = equilibrate(operator);
        operator.transpose_mut();
        return result.map(|scales| {
            scales.map(|scales| Equilibration {
                row: scales.column,
                column: scales.row,
            })
        });
    }
    let inverse = |value: f64| {
        value
            .clamp(f64::MIN_POSITIVE, f64::MIN_POSITIVE.recip())
            .recip()
    };
    for value in &mut row {
        *value = inverse(*value);
    }
    let mut column = Vec::with_capacity(n);
    for values in operator.as_mut_slice().chunks_exact_mut(n) {
        let mut column_max = 0.0_f64;
        for (value, &scale) in values.iter_mut().zip(&row) {
            *value *= scale;
            column_max = column_max.max(value.re.abs().max(value.im.abs()));
        }
        let scale = inverse(column_max);
        column.push(scale);
        for value in values {
            *value *= scale;
        }
    }
    Ok(Some(Equilibration { row, column }))
}

fn scale_rows(mut matrix: MatMut<'_, Complex>, scales: &[f64]) {
    for j in 0..matrix.ncols() {
        for (value, &scale) in matrix.as_mut().col_mut(j).iter_mut().zip(scales) {
            *value *= scale;
        }
    }
}

/// The faer workers of an LU factorization or solve with `rows` rows and `columns`
/// right-hand sides, on a pool of `budget` threads.
///
/// Recursive LU and triangular solves split the work into panels, one task each. A
/// pool of up to four threads runs whole: the panels stay wide enough for four
/// workers. Larger pools get one worker per 512 rows up to four, one per 2048 rows
/// beyond that, and at most one per 16 right-hand sides, because narrower panels cost
/// more in task overhead than they gain. The constants are tuned up to 8192 rows;
/// `benches/lu_scheduling.rs` times a factorization and its solves for any worker
/// count, to check them on other hardware.
fn lu_threads(rows: usize, columns: usize, budget: usize) -> usize {
    if budget <= 4 {
        return budget;
    }
    (rows / 512)
        .min(4)
        .max(rows / 2048)
        .min(columns / 16)
        .clamp(1, budget)
}

fn lu_parallelism(rows: usize, columns: usize) -> faer::Par {
    let configured = faer::get_global_parallelism();
    if configured.degree() == 1 {
        return faer::Par::Seq;
    }
    let budget = configured.degree().min(rayon::current_num_threads());
    match lu_threads(rows, columns, budget) {
        1 => faer::Par::Seq,
        threads => faer::Par::rayon(threads),
    }
}

/// Packed pivoted LU: overwrite the operator and share its triangular storage.
#[derive(Clone, Debug)]
pub(crate) struct Lu {
    factors: DMatrix<Complex>,
    permutation: Perm<usize>,
    equilibration: Option<Equilibration>,
}

impl Lu {
    pub(crate) fn new(mut factors: DMatrix<Complex>) -> Result<Self> {
        let n = factors.nrows();
        let equilibration = equilibrate(&mut factors)?;
        let par = lu_parallelism(n, n);
        let mut forward = vec![0usize; n];
        let mut inverse = vec![0usize; n];
        factor::lu_in_place(
            view_mut(&mut factors),
            &mut forward,
            &mut inverse,
            par,
            MemStack::new(&mut MemBuffer::new(factor::lu_in_place_scratch::<
                usize,
                Complex,
            >(
                n, n, par, Spec::default()
            ))),
            Spec::default(),
        );
        if factors.diagonal().iter().any(|&z| z == Complex::default()) {
            return Err(Error::Singular);
        }
        Ok(Self {
            factors,
            equilibration,
            permutation: Perm::new_checked(
                forward.into_boxed_slice(),
                inverse.into_boxed_slice(),
                n,
            ),
        })
    }

    /// Solve `A X = B` in place, with `A` the operator given to [`Lu::new`].
    pub(crate) fn solve_in_place(&self, rhs: MatMut<'_, Complex>) {
        self.solve_with(rhs, false);
    }

    /// Solve `Aᴴ X = B` in place.
    pub(crate) fn solve_adjoint_in_place(&self, rhs: MatMut<'_, Complex>) {
        self.solve_with(rhs, true);
    }

    // The factors hold `R A C`. A forward solve scales by the row scales, solves
    // and scales by the column scales; an adjoint solve uses the reverse order.
    fn solve_with(&self, mut rhs: MatMut<'_, Complex>, adjoint: bool) {
        let (n, columns) = (self.factors.nrows(), rhs.ncols());
        let par = lu_parallelism(n, columns);
        let scales = self.equilibration.as_ref().map(|scales| {
            if adjoint {
                (&scales.column, &scales.row)
            } else {
                (&scales.row, &scales.column)
            }
        });
        if let Some((before, _)) = scales {
            scale_rows(rhs.as_mut(), before);
        }
        let (factors, permutation) = (view(&self.factors), self.permutation.as_ref());
        if adjoint {
            lu_solve::solve_transpose_in_place_with_conj(
                factors,
                factors,
                permutation,
                Conj::Yes,
                rhs.as_mut(),
                par,
                MemStack::new(&mut MemBuffer::new(
                    lu_solve::solve_transpose_in_place_scratch::<usize, Complex>(n, columns, par),
                )),
            );
        } else {
            lu_solve::solve_in_place(
                factors,
                factors,
                permutation,
                rhs.as_mut(),
                par,
                MemStack::new(&mut MemBuffer::new(lu_solve::solve_in_place_scratch::<
                    usize,
                    Complex,
                >(n, columns, par))),
            );
        }
        if let Some((_, after)) = scales {
            scale_rows(rhs, after);
        }
    }
}

/// What [`solve`] saves for its pullback: the LU factors and the solution. The pullback
/// solves the adjoint system with the same factors.
#[derive(Clone, Debug)]
pub struct SolveResidual {
    lu: Lu,
    value: DMatrix<Complex>,
}

/// Solve A X = B with pivoted LU; B is a matrix of right-hand sides.
pub fn solve(operator: &DMatrix<Complex>, rhs: DMatrix<Complex>) -> Result<SolveResidual> {
    solve_owned(operator.clone(), rhs)
}

/// Solve while reusing the owned operator buffer for its LU factors.
pub fn solve_owned(operator: DMatrix<Complex>, mut rhs: DMatrix<Complex>) -> Result<SolveResidual> {
    let n = operator.nrows();
    if n == 0
        || !operator.is_square()
        || rhs.nrows() != n
        || rhs.ncols() == 0
        || rhs.iter().any(|&z| !finite(z))
    {
        return Err(Error::InvalidInput(
            "require a finite nonempty square operator and matching right-hand sides".into(),
        ));
    }
    let lu = Lu::new(operator)?;
    lu.solve_in_place(view_mut(&mut rhs));
    if rhs.iter().any(|&z| !finite(z)) {
        return Err(Error::Singular);
    }
    Ok(SolveResidual { lu, value: rhs })
}

/// What [`svdvals`] saves for its pullback: the thin singular vectors and the singular
/// values.
#[derive(Debug)]
pub struct SvdvalsResidual {
    decomposition: Svd<Complex>,
    values: Vec<f64>,
}

/// Singular values of a finite nonempty rectangular complex matrix.
pub fn svdvals(operator: &DMatrix<Complex>) -> Result<SvdvalsResidual> {
    if operator.is_empty() || operator.iter().any(|&z| !finite(z)) {
        return Err(Error::InvalidInput(
            "SVD requires a finite nonempty matrix".into(),
        ));
    }
    let scale = operator.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let scale = if scale == 0.0 { 1.0 } else { scale };
    let decomposition = Svd::new_thin(view(&operator.map(|z| z / scale)))
        .map_err(|err| Error::NotConverged(format!("singular-value decomposition: {err:?}")))?;
    let values: Vec<_> = (0..operator.nrows().min(operator.ncols()))
        .map(|i| decomposition.S()[i].re * scale)
        .collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFinite("non-finite singular values".into()));
    }
    Ok(SvdvalsResidual {
        decomposition,
        values,
    })
}

impl SvdvalsResidual {
    /// The singular values, in descending order, which the pullback reads.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Equal weights are supported at repeated positive values; zero values
    /// require zero weights because the singular value itself is not smooth there.
    pub fn pullback(self, cotangent: &[f64]) -> Result<DMatrix<Complex>> {
        if cotangent.len() != self.values.len() || cotangent.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidInput(
                "invalid singular-value cotangent".into(),
            ));
        }
        let tolerance = 64.0 * f64::EPSILON * self.values[0];
        for i in 0..cotangent.len() {
            if self.values[i] == 0.0 && cotangent[i] != 0.0 {
                return Err(Error::InvalidInput(
                    "nonzero singular-value weight at zero is not differentiable".into(),
                ));
            }
            if i > 0
                && (self.values[i - 1] - self.values[i]).abs() <= tolerance
                && (cotangent[i - 1] - cotangent[i]).abs()
                    > 64.0 * f64::EPSILON * cotangent[i - 1].abs().max(cotangent[i].abs())
            {
                return Err(Error::InvalidInput(
                    "repeated singular values require equal cotangent weights".into(),
                ));
            }
        }
        let u = self.decomposition.U();
        let weighted =
            DMatrix::from_fn(u.nrows(), cotangent.len(), |i, j| u[(i, j)] * cotangent[j]);
        Ok(product_views(
            view(&weighted),
            self.decomposition.V().adjoint(),
        ))
    }
}

/// Cotangents of the inputs of [`solve`] and [`solve_owned`], in the order of their
/// arguments.
#[derive(Clone, Debug)]
pub struct SolveGradient {
    /// Cotangent of the operator `A`.
    pub operator: DMatrix<Complex>,
    /// Cotangent of the right-hand sides `B`.
    pub rhs: DMatrix<Complex>,
}

impl SolveResidual {
    /// The solution `X` of `A X = B`, one column per right-hand side, which the pullback
    /// reads.
    #[must_use]
    pub const fn value(&self) -> &DMatrix<Complex> {
        &self.value
    }

    /// The shape of the solution: unknowns and right-hand sides.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.value.shape()
    }

    pub(crate) fn adjoint_rhs(&self, mut cotangent: DMatrix<Complex>) -> Result<DMatrix<Complex>> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid linear-solve cotangent".into()));
        }
        self.lu.solve_adjoint_in_place(view_mut(&mut cotangent));
        Ok(cotangent)
    }

    /// Return operator and right-hand-side cotangents, reusing the forward LU.
    pub fn pullback(self, cotangent: DMatrix<Complex>) -> Result<SolveGradient> {
        let rhs = self.adjoint_rhs(cotangent)?;
        Ok(SolveGradient {
            operator: -product_adjoint_right(&rhs, &self.value),
            rhs,
        })
    }
}

/// What [`eig`] saves for its pullback: the right eigensystem, with unit vectors whose
/// largest component is real and positive.
#[derive(Debug)]
pub struct EigResidual {
    values: Vec<Complex>,
    vectors: DMatrix<Complex>,
    pivots: Vec<usize>,
    scale: f64,
}

/// General complex eigendecomposition. Repeated eigenvalues are allowed in forward.
pub fn eig(operator: &DMatrix<Complex>) -> Result<EigResidual> {
    let n = operator.nrows();
    if n == 0 || !operator.is_square() || operator.iter().any(|&z| !finite(z)) {
        return Err(Error::InvalidInput(
            "eigensystem requires a finite nonempty square matrix".into(),
        ));
    }
    let scale = operator.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let scale = if scale == 0.0 { 1.0 } else { scale };
    let decomposition = Eigen::new(view(&operator.map(|z| z / scale)))
        .map_err(|err| Error::NotConverged(format!("eigendecomposition: {err:?}")))?;
    let values: Vec<_> = (0..n).map(|i| decomposition.S()[i] * scale).collect();
    let mut vectors = DMatrix::from_fn(n, n, |i, j| decomposition.U()[(i, j)]);
    let mut pivots = Vec::with_capacity(n);
    for mut column in vectors.column_iter_mut() {
        let pivot = (0..n)
            .max_by(|&i, &j| column[i].norm_sqr().total_cmp(&column[j].norm_sqr()))
            .ok_or(Error::Singular)?;
        let phase = column[pivot] / column[pivot].norm();
        let normalization = phase * column.norm();
        column /= normalization;
        pivots.push(pivot);
    }
    if values.iter().chain(vectors.iter()).any(|&z| !finite(z)) {
        return Err(Error::NonFinite("non-finite eigensystem".into()));
    }
    Ok(EigResidual {
        values,
        vectors,
        pivots,
        scale,
    })
}

impl EigResidual {
    /// The eigenvalues in the solver's order, which the pullback reads.
    #[must_use]
    pub fn values(&self) -> &[Complex] {
        &self.values
    }

    /// The right eigenvectors, one column per eigenvalue, which the pullback reads.
    #[must_use]
    pub const fn vectors(&self) -> &DMatrix<Complex> {
        &self.vectors
    }

    /// The operator gradient from the eigenvalue and eigenvector cotangents, with the
    /// phase of each vector fixed at its pivot.
    ///
    /// Repeated eigenvalues support equal eigenvalue weights and zero vector
    /// cotangents within each repeated group. Individual modes there have no pullback.
    pub fn pullback(
        self,
        values: &[Complex],
        mut vectors: DMatrix<Complex>,
    ) -> Result<DMatrix<Complex>> {
        let n = self.values.len();
        if values.len() != n
            || vectors.shape() != (n, n)
            || values.iter().chain(vectors.iter()).any(|&z| !finite(z))
        {
            return Err(Error::InvalidInput("invalid eigensystem cotangent".into()));
        }
        let vector_active: Vec<_> = vectors
            .column_iter()
            .map(|v| v.iter().any(|&z| z != Complex::default()))
            .collect();
        if !vector_active.iter().any(|&active| active) && values.iter().all(|&g| g == values[0]) {
            // Sum of all eigenvalues is trace(A), including defective matrices.
            return Ok(DMatrix::identity(n, n) * values[0]);
        }
        // Project out normalization and pivot-phase changes before pairing the
        // cotangent with the off-diagonal eigenvector derivative.
        for j in 0..n {
            let inner = vectors.column(j).dotc(&self.vectors.column(j));
            let pivot = self.pivots[j];
            let pivot_size = self.vectors[(pivot, j)].norm();
            if inner.im.abs() > 64.0 * f64::EPSILON * vectors.column(j).norm()
                && (0..n).any(|i| {
                    i != pivot
                        && (pivot_size - self.vectors[(i, j)].norm()).abs()
                            <= 64.0 * f64::EPSILON * pivot_size
                })
            {
                return Err(Error::InvalidInput(
                    "phase-dependent eigenvector pullback requires a unique largest component"
                        .into(),
                ));
            }
            for i in 0..n {
                vectors[(i, j)] -= inner.re * self.vectors[(i, j)];
            }
            vectors[(pivot, j)] += Complex::i() * inner.im / self.vectors[(pivot, j)].re;
        }
        let mut g = product_views(view(&self.vectors).adjoint(), view(&vectors));
        for j in 0..n {
            for i in 0..n {
                if i == j {
                    g[(i, j)] = values[i];
                    continue;
                }
                let gap = self.values[j] - self.values[i];
                if gap.norm() <= 64.0 * f64::EPSILON * self.scale {
                    let weight_scale = values[i].norm().max(values[j].norm());
                    if vector_active[i]
                        || vector_active[j]
                        || (values[i] - values[j]).norm() > 64.0 * f64::EPSILON * weight_scale
                    {
                        return Err(Error::InvalidInput(
                            "individual eigenmodes have no pullback at repeated eigenvalues".into(),
                        ));
                    }
                    g[(i, j)] = Complex::default();
                } else {
                    g[(i, j)] = ratio(g[(i, j)], gap.conj());
                }
            }
        }
        let mut result = product_adjoint_right(&g, &self.vectors);
        let lu = Lu::new(self.vectors)?;
        lu.solve_adjoint_in_place(view_mut(&mut result));
        if result.iter().any(|&z| !finite(z)) {
            return Err(Error::Singular);
        }
        Ok(result)
    }
}

/// LU scheduling, equilibration and the solve pullback.
#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;
    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::{SolveGradient, equilibrate, lu_threads, solve_owned};
    use crate::{
        Complex,
        test_support::{ALGEBRA_CASES, complex_matrix, prop_assert_close},
    };

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn lu_scheduling_respects_worker_and_rhs_budgets(
            rows in 1_usize..100_000,
            columns in 1_usize..100_000,
            budget in 1_usize..128,
        ) {
            check_lu_scheduling(rows, columns, budget)?;
        }
    }

    /// The worker count stays within the pool, uses a pool of up to four threads whole,
    /// and runs on one worker below 32 right-hand sides on larger pools.
    fn check_lu_scheduling(
        rows: usize,
        columns: usize,
        budget: usize,
    ) -> Result<(), TestCaseError> {
        let threads = lu_threads(rows, columns, budget);
        prop_assert!((1..=budget).contains(&threads));
        if budget <= 4 {
            prop_assert_eq!(threads, budget);
        } else if columns < 32 {
            prop_assert_eq!(threads, 1);
        }
        Ok(())
    }

    /// Power-of-two equation (row) and unknown (column) units of a linear system.
    #[derive(Clone, Debug)]
    enum Units {
        /// Independent row exponents.
        Rows(Vec<i32>),
        /// Independent column exponents.
        Columns(Vec<i32>),
        /// Rows alternating `2^∓e` and columns `2^±e/2`, or with the two spreads
        /// exchanged, which makes the columns the more disparate axis.
        Alternating { exponent: i32, transposed: bool },
    }

    impl Units {
        /// Row and column scales `(d_r, d_c)` of `A = d_r H d_c`.
        fn scales(&self, n: usize) -> (Vec<f64>, Vec<f64>) {
            let power = |e: i32| 2.0_f64.powi(e);
            let unit = vec![1.0; n];
            match self {
                Self::Rows(e) => (e.iter().map(|&e| power(e)).collect(), unit),
                Self::Columns(e) => (unit, e.iter().map(|&e| power(e)).collect()),
                &Self::Alternating {
                    exponent,
                    transposed,
                } => {
                    let sign = |i: usize| if i.is_multiple_of(2) { -1 } else { 1 };
                    let (wide, narrow) = (exponent, exponent / 2);
                    let (row, column) = if transposed {
                        (narrow, wide)
                    } else {
                        (wide, narrow)
                    };
                    (
                        (0..n).map(|i| power(sign(i) * row)).collect(),
                        (0..n).map(|j| power(-sign(j) * column)).collect(),
                    )
                }
            }
        }
    }

    fn system() -> impl Strategy<Value = (DMatrix<Complex>, Units, DMatrix<Complex>)> {
        (1_usize..=40, 1_usize..=3).prop_flat_map(|(n, columns)| {
            let exponents = || prop::collection::vec(-400_i32..=400, n);
            (
                complex_matrix(n, n, 0.5),
                Just((0..n).collect::<Vec<_>>()).prop_shuffle(),
                prop_oneof![
                    exponents().prop_map(Units::Rows),
                    exponents().prop_map(Units::Columns),
                    (0_i32..=400, any::<bool>()).prop_map(|(exponent, transposed)| {
                        Units::Alternating {
                            exponent,
                            transposed,
                        }
                    }),
                ],
                complex_matrix(n, columns, 1.0),
            )
                .prop_map(move |(random, rows, units, y)| {
                    // `random + n I` with permuted rows, so pivoting must reorder them.
                    let size = Complex::from(f64::from(u32::try_from(n).unwrap()));
                    let mut h = random + DMatrix::identity(n, n) * size;
                    h = DMatrix::from_fn(n, n, |i, j| h[(rows[i], j)]);
                    (h, units, y)
                })
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn solve_and_adjoint_are_invariant_to_equation_and_unknown_units(
            (h, units, y) in system(),
        ) {
            check_units(&h, &units, &y)?;
        }
    }

    /// For `A = d_r H d_c` with a well-conditioned `H`, the solve of `A X = d_r H Y`
    /// gives `d_c X = Y` and the adjoint solve of `Aᴴ Z = d_c Hᴴ Y` gives `d_r Z = Y`,
    /// whatever the units: equilibration removes them in both the direct and the
    /// transposed branch. The returned scales reproduce the equilibrated operator,
    /// and the operator cotangent is `-Z Xᴴ`, so scaling `A` and `B` jointly leaves
    /// the loss unchanged.
    fn check_units(
        h: &DMatrix<Complex>,
        units: &Units,
        y: &DMatrix<Complex>,
    ) -> Result<(), TestCaseError> {
        let n = h.nrows();
        let (row, column) = units.scales(n);
        let operator = DMatrix::from_fn(n, n, |i, j| h[(i, j)] * row[i] * column[j]);
        let hy = h * y;
        let rhs = DMatrix::from_fn(n, y.ncols(), |i, j| hy[(i, j)] * row[i]);
        let tolerance = 1e-13 * y.norm();
        let residual = solve_owned(operator.clone(), rhs.clone()).unwrap();
        let unknowns = DMatrix::from_fn(n, y.ncols(), |i, j| residual.value[(i, j)] * column[i]);
        prop_assert_close!(&unknowns, y, tolerance);
        let hz = h.adjoint() * y;
        let cotangent = DMatrix::from_fn(n, y.ncols(), |i, j| hz[(i, j)] * column[i]);
        let adjoint = residual.adjoint_rhs(cotangent.clone()).unwrap();
        let equations = DMatrix::from_fn(n, y.ncols(), |i, j| adjoint[(i, j)] * row[i]);
        prop_assert_close!(&equations, y, tolerance);

        let SolveGradient {
            operator: ga,
            rhs: gb,
        } = residual.clone().pullback(cotangent).unwrap();
        prop_assert_eq!(&gb, &adjoint);
        for i in 0..n {
            for j in 0..n {
                let terms = (0..y.ncols()).map(|p| (adjoint[(i, p)], residual.value[(j, p)]));
                let expected: Complex = terms.clone().map(|(z, x)| -z * x.conj()).sum();
                let scale: f64 = terms.map(|(z, x)| z.norm() * x.norm()).sum();
                prop_assert_close!(ga[(i, j)], expected, 1e-14 * scale, "entry ({}, {})", i, j);
            }
        }
        // Scaling A and B jointly leaves X unchanged: Re⟨ga, A⟩ + Re⟨gb, B⟩ = 0.
        let pairs = ga.iter().zip(&operator).chain(gb.iter().zip(&rhs));
        let magnitude: f64 = pairs.clone().map(|(g, x)| g.norm() * x.norm()).sum();
        let joint: f64 = pairs.map(|(g, x)| (g.conj() * x).re).sum();
        prop_assert_close!(joint, 0.0, 1e-13 * magnitude);

        // Equilibration in the direct or transposed branch.
        let Units::Alternating {
            exponent,
            transposed,
        } = *units
        else {
            return Ok(());
        };
        // Past 2^32 the spreads, not the random entries of H, decide the branch.
        if n < 2 || exponent < 32 {
            return Ok(());
        }
        let magnitude = |z: Complex| z.re.abs().max(z.im.abs());
        let row_min = operator
            .row_iter()
            .map(|r| r.iter().copied().map(magnitude).fold(0.0, f64::max))
            .fold(f64::INFINITY, f64::min);
        let column_min = operator
            .column_iter()
            .map(|c| c.iter().copied().map(magnitude).fold(0.0, f64::max))
            .fold(f64::INFINITY, f64::min);
        prop_assert_eq!(column_min < row_min, transposed);
        let mut equilibrated = operator.clone();
        let scales = equilibrate(&mut equilibrated).unwrap().unwrap();
        for i in 0..n {
            for j in 0..n {
                let expected = operator[(i, j)] * scales.row[i] * scales.column[j];
                let actual = equilibrated[(i, j)];
                let tolerance = 4.0 * f64::EPSILON * expected.norm() + f64::MIN_POSITIVE;
                prop_assert_close!(actual, expected, tolerance, "entry ({}, {})", i, j);
            }
        }
        let mut transpose = operator.transpose();
        let swapped = equilibrate(&mut transpose).unwrap().unwrap();
        prop_assert_eq!(&scales.row, &swapped.column);
        prop_assert_eq!(&scales.column, &swapped.row);
        Ok(())
    }

    #[test]
    fn ordinary_operator_is_not_modified() {
        let mut operator = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex::new(2.0, 1.0),
                Complex::new(0.5, -0.25),
                Complex::new(-0.25, 0.5),
                Complex::new(3.0, -1.0),
            ],
        );
        let original = operator.clone();
        assert!(equilibrate(&mut operator).unwrap().is_none());
        assert_eq!(operator, original);
        operator[(0, 1)] = Complex::from(2.0_f64.powi(-100));
        assert!(equilibrate(&mut operator).unwrap().is_none());
        operator *= Complex::from(2.0_f64.powi(-600));
        assert!(equilibrate(&mut operator).unwrap().is_none());
    }

    #[test]
    fn extreme_column_units_do_not_underflow_during_row_scaling() {
        let high = 2.0_f64.powi(600);
        let low = high.recip();
        let operator = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex::from(high),
                Complex::from(low),
                Complex::from(high),
                Complex::from(2.0 * low),
            ],
        );
        let rhs = DMatrix::from_column_slice(2, 1, &[Complex::from(1.0), Complex::from(2.0)]);
        let residual = solve_owned(operator, rhs).unwrap();
        assert_eq!(residual.value[(0, 0)], Complex::default());
        assert_eq!(residual.value[(1, 0)], Complex::from(high));
        // A^H [1, -1] = [0, -low], with both exact adjoint coordinates finite.
        let cotangent =
            DMatrix::from_column_slice(2, 1, &[Complex::default(), Complex::from(-low)]);
        let adjoint = residual.adjoint_rhs(cotangent).unwrap();
        assert_eq!(adjoint[(0, 0)], Complex::from(1.0));
        assert_eq!(adjoint[(1, 0)], Complex::from(-1.0));
    }
}

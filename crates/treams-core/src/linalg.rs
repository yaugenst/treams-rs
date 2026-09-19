//! Dense linear solves and general complex eigensystems with native pullbacks.
#![allow(clippy::indexing_slicing)] // Validated matrix dimensions and eigenvector pivots.

use faer::{
    Conj, MatMut, Spec,
    dyn_stack::{MemBuffer, MemStack},
    linalg::{
        lu::partial_pivoting::{factor, solve as lu_solve},
        solvers::{Eigen, Svd},
    },
    perm::Perm,
};
use nalgebra::DMatrix;

use crate::{
    Complex, Error, Result, finite,
    interaction::{product_adjoint_left, product_adjoint_right, view, view_mut},
    ratio,
};

/// Positive diagonal scales for the equivalent operator `R A C`.
///
/// A normal solve scales its right-hand side by `row` and its solution by
/// `column`. A conjugate-transpose solve uses the opposite order. These scales
/// are numerical coordinates, not additional inputs to the implicit derivative.
#[derive(Clone, Debug)]
pub struct Equilibration {
    /// Left (equation) scale.
    pub row: Vec<f64>,
    /// Right (unknown) scale.
    pub column: Vec<f64>,
}

/// Equilibrate severely unbalanced finite square operators in place.
///
/// Small-argument, high-order multipole operators can have unit diagonal and
/// off-diagonal entries exceeding `1e20`. Their unscaled LU loses accuracy even
/// when the equivalent equilibrated system is well conditioned. The existing
/// validation scan also measures scale spread; ordinary operators are untouched.
pub fn equilibrate(operator: &mut DMatrix<Complex>) -> Result<Option<Equilibration>> {
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
        for (i, &scale) in scales.iter().enumerate() {
            matrix[(i, j)] *= scale;
        }
    }
}

// Recursive LU and triangular solves schedule many narrow panels. On large
// Rayon pools their task overhead dominates long before all workers are useful.
// Keep the established <=4-worker profile; scale larger pools by panel width.
fn lu_threads(rows: usize, columns: usize, budget: usize) -> usize {
    if budget <= 4 {
        return budget;
    }
    // ponytail: calibrated through 8192 complex128 rows on Ryzen 9950X;
    // allow more workers above that range as the per-panel work grows.
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

    pub(crate) fn solve_in_place(&self, mut rhs: MatMut<'_, Complex>) {
        let columns = rhs.ncols();
        let par = lu_parallelism(self.factors.nrows(), columns);
        if let Some(scales) = &self.equilibration {
            scale_rows(rhs.as_mut(), &scales.row);
        }
        lu_solve::solve_in_place(
            view(&self.factors),
            view(&self.factors),
            self.permutation.as_ref(),
            rhs.as_mut(),
            par,
            MemStack::new(&mut MemBuffer::new(lu_solve::solve_in_place_scratch::<
                usize,
                Complex,
            >(
                self.factors.nrows(), columns, par
            ))),
        );
        if let Some(scales) = &self.equilibration {
            scale_rows(rhs, &scales.column);
        }
    }

    pub(crate) fn solve_adjoint_in_place(&self, mut rhs: MatMut<'_, Complex>) {
        let columns = rhs.ncols();
        let par = lu_parallelism(self.factors.nrows(), columns);
        if let Some(scales) = &self.equilibration {
            scale_rows(rhs.as_mut(), &scales.column);
        }
        lu_solve::solve_transpose_in_place_with_conj(
            view(&self.factors),
            view(&self.factors),
            self.permutation.as_ref(),
            Conj::Yes,
            rhs.as_mut(),
            par,
            MemStack::new(&mut MemBuffer::new(
                lu_solve::solve_transpose_in_place_scratch::<usize, Complex>(
                    self.factors.nrows(),
                    columns,
                    par,
                ),
            )),
        );
        if let Some(scales) = &self.equilibration {
            scale_rows(rhs, &scales.row);
        }
    }
}

/// Factorization and solution retained for a linear solve's implicit adjoint.
#[derive(Clone, Debug)]
pub struct SolveResidual {
    lu: Lu,
    /// Solution of A X = B, with one or more right-hand sides.
    pub value: DMatrix<Complex>,
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

/// Thin singular vectors retained for a singular-value pullback.
#[derive(Debug)]
pub struct SingularResidual {
    decomposition: Svd<Complex>,
    /// Singular values, in descending order.
    pub values: Vec<f64>,
}

/// Singular values of a finite nonempty rectangular complex matrix.
pub fn svdvals(operator: &DMatrix<Complex>) -> Result<SingularResidual> {
    if operator.is_empty() || operator.iter().any(|&z| !finite(z)) {
        return Err(Error::InvalidInput(
            "SVD requires a finite nonempty matrix".into(),
        ));
    }
    let scale = operator.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let scale = if scale == 0.0 { 1.0 } else { scale };
    let decomposition = Svd::new_thin(view(&operator.map(|z| z / scale)))
        .map_err(|err| Error::SpecialFunction(format!("singular-value decomposition: {err:?}")))?;
    let values: Vec<_> = (0..operator.nrows().min(operator.ncols()))
        .map(|i| decomposition.S()[i].re * scale)
        .collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err(Error::SpecialFunction("non-finite singular values".into()));
    }
    Ok(SingularResidual {
        decomposition,
        values,
    })
}

impl SingularResidual {
    /// Equal weights are supported at repeated positive values; zero values
    /// require zero weights because the singular value itself is not smooth there.
    pub fn pullback(self, g: &[f64]) -> Result<DMatrix<Complex>> {
        if g.len() != self.values.len() || g.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidInput(
                "invalid singular-value cotangent".into(),
            ));
        }
        let tolerance = 64.0 * f64::EPSILON * self.values[0];
        for i in 0..g.len() {
            if self.values[i] == 0.0 && g[i] != 0.0 {
                return Err(Error::InvalidInput(
                    "nonzero singular-value weight at zero is not differentiable".into(),
                ));
            }
            if i > 0
                && (self.values[i - 1] - self.values[i]).abs() <= tolerance
                && (g[i - 1] - g[i]).abs() > 64.0 * f64::EPSILON * g[i - 1].abs().max(g[i].abs())
            {
                return Err(Error::InvalidInput(
                    "repeated singular values require equal cotangent weights".into(),
                ));
            }
        }
        let u = self.decomposition.U();
        let v = self.decomposition.V();
        let weighted = DMatrix::from_fn(u.nrows(), g.len(), |i, j| u[(i, j)] * g[j]);
        let mut result = DMatrix::zeros(u.nrows(), v.nrows());
        faer::linalg::matmul::matmul_with_conj(
            view_mut(&mut result),
            faer::Accum::Replace,
            view(&weighted),
            Conj::No,
            v.transpose(),
            Conj::Yes,
            Complex::new(1.0, 0.0),
            faer::get_global_parallelism(),
        );
        Ok(result)
    }
}

impl SolveResidual {
    pub(crate) fn adjoint_rhs(&self, mut g: DMatrix<Complex>) -> Result<DMatrix<Complex>> {
        if g.shape() != self.value.shape() || g.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid linear-solve cotangent".into()));
        }
        self.lu.solve_adjoint_in_place(view_mut(&mut g));
        Ok(g)
    }

    /// Return operator and right-hand-side cotangents, reusing the forward LU.
    pub fn pullback(self, g: DMatrix<Complex>) -> Result<(DMatrix<Complex>, DMatrix<Complex>)> {
        let g = self.adjoint_rhs(g)?;
        Ok((-product_adjoint_right(&g, &self.value), g))
    }
}

/// Right eigensystem, with unit vectors whose largest component is real positive.
#[derive(Debug)]
pub struct EigenResidual {
    /// Eigenvalues in the solver's order.
    pub values: Vec<Complex>,
    /// Corresponding right eigenvectors, stored in columns.
    pub vectors: DMatrix<Complex>,
    pivots: Vec<usize>,
    scale: f64,
}

/// General complex eigendecomposition. Repeated eigenvalues are allowed in forward.
pub fn eig(operator: &DMatrix<Complex>) -> Result<EigenResidual> {
    let n = operator.nrows();
    if n == 0 || !operator.is_square() || operator.iter().any(|&z| !finite(z)) {
        return Err(Error::InvalidInput(
            "eigensystem requires a finite nonempty square matrix".into(),
        ));
    }
    let scale = operator.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let scale = if scale == 0.0 { 1.0 } else { scale };
    let decomposition = Eigen::new(view(&operator.map(|z| z / scale)))
        .map_err(|err| Error::SpecialFunction(format!("eigendecomposition: {err:?}")))?;
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
        return Err(Error::SpecialFunction("non-finite eigensystem".into()));
    }
    Ok(EigenResidual {
        values,
        vectors,
        pivots,
        scale,
    })
}

impl EigenResidual {
    /// Contract eigenvalue and eigenvector cotangents under the fixed pivot phase.
    ///
    /// Repeated eigenvalues support equal eigenvalue weights and zero vector
    /// cotangents within each repeated group. Individual modes there have no VJP.
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
        // Project out normalization and pivot-phase changes before contracting
        // the off-diagonal eigenvector derivative.
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
        let mut g = product_adjoint_left(&self.vectors, &vectors);
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

#[cfg(test)]
mod conditioning_tests {
    #![allow(clippy::unwrap_used)] // proptest retains failing scaled systems.
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn lu_scheduling_respects_worker_and_rhs_budgets(
            rows in 1_usize..100_000,
            columns in 1_usize..100_000,
            budget in 1_usize..128,
        ) {
            let threads = lu_threads(rows, columns, budget);
            prop_assert!((1..=budget).contains(&threads));
            if budget <= 4 {
                prop_assert_eq!(threads, budget);
            } else if columns < 32 {
                prop_assert_eq!(threads, 1);
            }
        }
    }

    proptest! {
        #[test]
        fn solve_and_adjoint_are_invariant_to_equation_and_unknown_units(
            exponent in 1_i32..400,
            a in -0.5_f64..0.5,
            b in -0.5_f64..0.5,
        ) {
            let h = DMatrix::from_row_slice(2, 2, &[
                Complex::new(2.0, 1.0), Complex::new(0.5, -0.25),
                Complex::new(-0.25, 0.5), Complex::new(3.0, -1.0),
            ]);
            let row = [2.0_f64.powi(-exponent), 2.0_f64.powi(exponent)];
            let column = [2.0_f64.powi(exponent / 2), 2.0_f64.powi(-exponent / 2)];
            let operator = DMatrix::from_fn(2, 2, |i, j| h[(i, j)] / row[i] / column[j]);
            let y = DMatrix::from_row_slice(2, 2, &[
                Complex::new(1.0 + a, b), Complex::new(a, 1.0 + b),
                Complex::new(-1.0 + b, a), Complex::new(1.0 + b, -a),
            ]);
            let hy = &h * &y;
            let rhs = DMatrix::from_fn(2, 2, |i, j| hy[(i, j)] / row[i]);
            let residual = solve_owned(operator, rhs).unwrap();
            for j in 0..2 {
                for i in 0..2 {
                    prop_assert!((residual.value[(i, j)] / column[i] - y[(i, j)]).norm() < 2e-14);
                }
            }
            let hz = h.adjoint() * &y;
            let cotangent = DMatrix::from_fn(2, 2, |i, j| hz[(i, j)] / column[i]);
            let adjoint = residual.adjoint_rhs(cotangent).unwrap();
            for j in 0..2 {
                for i in 0..2 {
                    prop_assert!((adjoint[(i, j)] / row[i] - y[(i, j)]).norm() < 2e-14);
                }
            }
        }
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

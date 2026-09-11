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

/// Packed pivoted LU: overwrite the operator and share its triangular storage.
#[derive(Clone, Debug)]
pub(crate) struct Lu {
    factors: DMatrix<Complex>,
    permutation: Perm<usize>,
}

impl Lu {
    pub(crate) fn new(mut factors: DMatrix<Complex>) -> Result<Self> {
        let n = factors.nrows();
        if n == 0 || !factors.is_square() || factors.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput(
                "require a finite nonempty square operator".into(),
            ));
        }
        let par = faer::get_global_parallelism();
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
            permutation: Perm::new_checked(
                forward.into_boxed_slice(),
                inverse.into_boxed_slice(),
                n,
            ),
        })
    }

    pub(crate) fn solve_in_place(&self, rhs: MatMut<'_, Complex>) {
        let par = faer::get_global_parallelism();
        let columns = rhs.ncols();
        lu_solve::solve_in_place(
            view(&self.factors),
            view(&self.factors),
            self.permutation.as_ref(),
            rhs,
            par,
            MemStack::new(&mut MemBuffer::new(lu_solve::solve_in_place_scratch::<
                usize,
                Complex,
            >(
                self.factors.nrows(), columns, par
            ))),
        );
    }

    pub(crate) fn solve_adjoint_in_place(&self, rhs: MatMut<'_, Complex>) {
        let par = faer::get_global_parallelism();
        let columns = rhs.ncols();
        lu_solve::solve_transpose_in_place_with_conj(
            view(&self.factors),
            view(&self.factors),
            self.permutation.as_ref(),
            Conj::Yes,
            rhs,
            par,
            MemStack::new(&mut MemBuffer::new(
                lu_solve::solve_transpose_in_place_scratch::<usize, Complex>(
                    self.factors.nrows(),
                    columns,
                    par,
                ),
            )),
        );
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

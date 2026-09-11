//! Dense multiple-scattering solve with a factorization-reusing pullback.

use faer::{
    Accum, Conj, MatMut, MatRef,
    linalg::matmul::{matmul, matmul_with_conj},
};
use nalgebra::DMatrix;

use crate::{Complex, Error, Result, finite, linalg::Lu};

pub(crate) fn view(matrix: &DMatrix<Complex>) -> MatRef<'_, Complex> {
    MatRef::from_column_major_slice(matrix.as_slice(), matrix.nrows(), matrix.ncols())
}
pub(crate) fn view_mut(matrix: &mut DMatrix<Complex>) -> MatMut<'_, Complex> {
    let (rows, cols) = matrix.shape();
    MatMut::from_column_major_slice_mut(matrix.as_mut_slice(), rows, cols)
}
pub(crate) fn product(left: &DMatrix<Complex>, right: &DMatrix<Complex>) -> DMatrix<Complex> {
    product_views(view(left), view(right))
}
pub(crate) fn product_views(
    left: MatRef<'_, Complex>,
    right: MatRef<'_, Complex>,
) -> DMatrix<Complex> {
    let mut result = DMatrix::zeros(left.nrows(), right.ncols());
    matmul(
        view_mut(&mut result),
        Accum::Replace,
        left,
        right,
        Complex::new(1.0, 0.0),
        faer::get_global_parallelism(),
    );
    result
}
pub(crate) fn product_adjoint_left(
    left: &DMatrix<Complex>,
    right: &DMatrix<Complex>,
) -> DMatrix<Complex> {
    product_adjoint_left_view(view(left), view(right))
}
pub(crate) fn product_adjoint_left_view(
    left: MatRef<'_, Complex>,
    right: MatRef<'_, Complex>,
) -> DMatrix<Complex> {
    let mut result = DMatrix::zeros(left.ncols(), right.ncols());
    matmul_with_conj(
        view_mut(&mut result),
        Accum::Replace,
        left.transpose(),
        Conj::Yes,
        right,
        Conj::No,
        Complex::new(1.0, 0.0),
        faer::get_global_parallelism(),
    );
    result
}
pub(crate) fn product_adjoint_right(
    left: &DMatrix<Complex>,
    right: &DMatrix<Complex>,
) -> DMatrix<Complex> {
    let mut result = DMatrix::zeros(left.nrows(), right.nrows());
    product_adjoint_right_into(&mut result, left, right);
    result
}
pub(crate) fn product_adjoint_right_into(
    result: &mut DMatrix<Complex>,
    left: &DMatrix<Complex>,
    right: &DMatrix<Complex>,
) {
    matmul_with_conj(
        view_mut(result),
        Accum::Replace,
        view(left),
        Conj::No,
        view(right).transpose(),
        Conj::Yes,
        Complex::new(1.0, 0.0),
        faer::get_global_parallelism(),
    );
}

#[derive(Clone, Debug)]
enum LocalMatrix {
    Dense(DMatrix<Complex>),
    Blocks(Vec<DMatrix<Complex>>),
}
impl LocalMatrix {
    fn dimension(&self) -> usize {
        match self {
            Self::Dense(m) => m.nrows(),
            Self::Blocks(blocks) => blocks.iter().map(DMatrix::nrows).sum(),
        }
    }
    fn dense(&self) -> DMatrix<Complex> {
        match self {
            Self::Dense(m) => m.clone(),
            Self::Blocks(blocks) => {
                let mut result = DMatrix::zeros(self.dimension(), self.dimension());
                let mut offset = 0;
                for block in blocks {
                    result
                        .view_mut((offset, offset), block.shape())
                        .copy_from(block);
                    offset += block.nrows();
                }
                result
            }
        }
    }
    fn apply(&self, right: &DMatrix<Complex>, adjoint: bool) -> DMatrix<Complex> {
        let mut result = DMatrix::zeros(self.dimension(), right.ncols());
        let blocks = match self {
            Self::Dense(m) => std::slice::from_ref(m),
            Self::Blocks(blocks) => blocks.as_slice(),
        };
        let mut offset = 0;
        for block in blocks {
            let target = view_mut(&mut result).subrows_mut(offset, block.nrows());
            let rhs = view(right).subrows(offset, block.nrows());
            if adjoint {
                matmul(
                    target,
                    Accum::Replace,
                    view(block).adjoint(),
                    rhs,
                    Complex::new(1.0, 0.0),
                    faer::get_global_parallelism(),
                );
            } else {
                matmul(
                    target,
                    Accum::Replace,
                    view(block),
                    rhs,
                    Complex::new(1.0, 0.0),
                    faer::get_global_parallelism(),
                );
            }
            offset += block.nrows();
        }
        result
    }
}

/// Native residual of `(I - T C) X = T`.
#[derive(Clone, Debug)]
pub struct InteractionResidual {
    local: LocalMatrix,
    coupling: DMatrix<Complex>,
    lu: Lu,
    /// Interacting T-matrix.
    pub value: DMatrix<Complex>,
}

/// Solve a multiple-scattering system, retaining its LU factorization.
pub fn forward(local: DMatrix<Complex>, coupling: DMatrix<Complex>) -> Result<InteractionResidual> {
    if !local.is_square()
        || local.nrows() == 0
        || local.shape() != coupling.shape()
        || local.iter().chain(coupling.iter()).any(|z| !finite(*z))
    {
        return Err(Error::InvalidInput(
            "local and coupling must be finite equally sized nonempty square matrices".into(),
        ));
    }
    factor(LocalMatrix::Dense(local), coupling)
}

/// Solve a cluster without storing or multiplying the zero off-diagonal local blocks.
pub(crate) fn forward_blocks(
    blocks: Vec<DMatrix<Complex>>,
    coupling: DMatrix<Complex>,
) -> Result<InteractionResidual> {
    if blocks.is_empty()
        || blocks
            .iter()
            .any(|m| !m.is_square() || m.nrows() == 0 || m.iter().any(|z| !finite(*z)))
    {
        return Err(Error::InvalidInput(
            "local blocks must be finite nonempty square matrices".into(),
        ));
    }
    let local = LocalMatrix::Blocks(blocks);
    if coupling.shape() != (local.dimension(), local.dimension())
        || coupling.iter().any(|z| !finite(*z))
    {
        return Err(Error::InvalidInput(
            "invalid cluster coupling matrix".into(),
        ));
    }
    factor(local, coupling)
}

fn factor(local: LocalMatrix, coupling: DMatrix<Complex>) -> Result<InteractionResidual> {
    let mut operator = -local.apply(&coupling, false);
    operator.set_diagonal(&(operator.diagonal().add_scalar(Complex::new(1.0, 0.0))));
    let lu = Lu::new(operator)?;
    let mut value = local.dense();
    lu.solve_in_place(view_mut(&mut value));
    if value.iter().any(|z| !finite(*z)) {
        return Err(Error::Singular);
    }
    Ok(InteractionResidual {
        local,
        coupling,
        lu,
        value,
    })
}

impl InteractionResidual {
    /// Input cotangents `(T_bar, C_bar)` under the real Hermitian pairing.
    pub fn pullback(
        self,
        cotangent: &DMatrix<Complex>,
    ) -> Result<(DMatrix<Complex>, DMatrix<Complex>)> {
        self.pullback_with(cotangent, product_adjoint_right)
    }

    /// Return only the local diagonal-block cotangents of a block-diagonal input.
    pub fn pullback_blocks(
        self,
        cotangent: &DMatrix<Complex>,
    ) -> Result<(Vec<DMatrix<Complex>>, DMatrix<Complex>)> {
        let LocalMatrix::Blocks(blocks) = &self.local else {
            return Err(Error::InvalidInput(
                "solve was not constructed from local blocks".into(),
            ));
        };
        let sizes: Vec<_> = blocks.iter().map(DMatrix::nrows).collect();
        self.pullback_with(cotangent, |adjoint, response| {
            let mut offset = 0;
            sizes
                .into_iter()
                .map(|n| {
                    let mut block = DMatrix::zeros(n, n);
                    matmul(
                        view_mut(&mut block),
                        Accum::Replace,
                        view(adjoint).subrows(offset, n),
                        view(response).subrows(offset, n).adjoint(),
                        Complex::new(1.0, 0.0),
                        faer::get_global_parallelism(),
                    );
                    offset += n;
                    block
                })
                .collect()
        })
    }

    fn pullback_with<T>(
        self,
        cotangent: &DMatrix<Complex>,
        local_gradient: impl FnOnce(&DMatrix<Complex>, &DMatrix<Complex>) -> T,
    ) -> Result<(T, DMatrix<Complex>)> {
        if cotangent.shape() != self.value.shape() || cotangent.iter().any(|z| !finite(*z)) {
            return Err(Error::InvalidInput(
                "invalid interacting-matrix cotangent".into(),
            ));
        }
        let mut adjoint = cotangent.clone();
        self.lu.solve_adjoint_in_place(view_mut(&mut adjoint));
        let mut response = product(&self.coupling, &self.value);
        response.set_diagonal(&(response.diagonal().add_scalar(Complex::new(1.0, 0.0))));
        let local = local_gradient(&adjoint, &response);
        drop(response);
        let coupling = product_adjoint_right(&self.local.apply(&adjoint, true), &self.value);
        Ok((local, coupling))
    }
}

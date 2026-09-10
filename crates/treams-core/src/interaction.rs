//! Dense multiple-scattering solve with a factorization-reusing pullback.

use faer::{
    Accum, MatMut, MatRef,
    linalg::{
        matmul::matmul,
        solvers::{PartialPivLu, Solve},
    },
};
use nalgebra::DMatrix;

use crate::{Complex, Error, Result, finite};

pub(crate) fn view(matrix: &DMatrix<Complex>) -> MatRef<'_, Complex> {
    MatRef::from_column_major_slice(matrix.as_slice(), matrix.nrows(), matrix.ncols())
}
pub(crate) fn view_mut(matrix: &mut DMatrix<Complex>) -> MatMut<'_, Complex> {
    let (rows, cols) = matrix.shape();
    MatMut::from_column_major_slice_mut(matrix.as_mut_slice(), rows, cols)
}
pub(crate) fn product(left: &DMatrix<Complex>, right: &DMatrix<Complex>) -> DMatrix<Complex> {
    let mut result = DMatrix::zeros(left.nrows(), right.ncols());
    matmul(
        view_mut(&mut result),
        Accum::Replace,
        view(left),
        view(right),
        Complex::new(1.0, 0.0),
        faer::get_global_parallelism(),
    );
    result
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
    lu: PartialPivLu<Complex>,
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
    let lu = PartialPivLu::new(view(&operator));
    if (0..local.dimension()).any(|i| lu.U()[(i, i)].norm_sqr() == 0.0) {
        return Err(Error::Singular);
    }
    drop(operator);
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
        if cotangent.shape() != self.value.shape() || cotangent.iter().any(|z| !finite(*z)) {
            return Err(Error::InvalidInput(
                "invalid interacting-matrix cotangent".into(),
            ));
        }
        let mut adjoint = cotangent.clone();
        self.lu.solve_adjoint_in_place(view_mut(&mut adjoint));
        let mut response = product(&self.coupling, &self.value);
        response.set_diagonal(&(response.diagonal().add_scalar(Complex::new(1.0, 0.0))));
        let local = product(&adjoint, &response.adjoint());
        let coupling = product(&self.local.apply(&adjoint, true), &self.value.adjoint());
        Ok((local, coupling))
    }
}

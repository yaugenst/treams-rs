//! Redheffer composition of two adjacent S-matrices.
//!
//! Upstream: `treams.SMatrices.add`.

use std::ops::AddAssign;

use faer::MatRef;
use nalgebra::DMatrix;

use super::{
    Blocks, checked_dimension, dimension,
    solve::{AdjointSolve, InternalSolve},
};
use crate::{
    Complex, Error, Result,
    linalg::{product, product_adjoint_right_into, product_views, view, view_mut},
};

/// What [`add`] saves for its pullback: four of the eight input blocks, the internal
/// fields and their factorization.
///
/// The pullback reads only blocks 1 and 3 of the lower S-matrix and blocks 0 and 2
/// of the upper one; the other four input blocks are released after the forward pass.
#[derive(Debug)]
pub struct AddResidual {
    /// Lower blocks 1 and 3: reflection of waves incident from the positive side (above)
    /// and downward transmission.
    lower: [DMatrix<Complex>; 2],
    /// Upper blocks 0 and 2: upward transmission and reflection of waves incident from
    /// the negative side (below).
    upper: [DMatrix<Complex>; 2],
    /// Upgoing internal fields `(I - L1 U2)^-1 [L0, L1 U3]` and their factorization.
    solve: InternalSolve,
    /// Downgoing internal fields `U2 up + [0, U3]`.
    down: DMatrix<Complex>,
}

/// Place `upper` above `lower`, eliminating their internal incident fields.
///
/// Upstream: `treams.SMatrices.add`, called as `lower.add(upper)`.
pub fn add(lower: Blocks, upper: Blocks) -> Result<(Blocks, AddResidual)> {
    let n = dimension(&lower)?;
    if dimension(&upper)? != n {
        return Err(Error::InvalidInput("S matrix dimensions must match".into()));
    }
    let [l0, l1, mut l2, l3] = lower;
    let [u0, mut u1, u2, u3] = upper;
    let mut rhs = DMatrix::zeros(n, 2 * n);
    rhs.columns_mut(0, n).copy_from(&l0);
    rhs.columns_mut(n, n).copy_from(&product(&l1, &u3));
    // The Krylov solve needs at least ITERATIVE_MIN_ROWS = 512 rows and at most
    // ITERATIVE_MAX_COLUMNS = 8 right-hand sides, so 2n columns never qualify and
    // the direct solve gives the same bits as `InternalSolve::new`.
    let solve = InternalSolve::direct(view(&l1), view(&u2), rhs)?;
    let up = view(&solve.value);
    let mut down = product_views(view(&u2), up);
    down.columns_mut(n, n).add_assign(&u3);
    let down_view = view(&down);
    let value = [
        product_views(view(&u0), up.subcols(0, n)),
        {
            accumulate_product(&mut u1, view(&u0), up.subcols(n, n));
            u1
        },
        {
            accumulate_product(&mut l2, view(&l3), down_view.subcols(0, n));
            l2
        },
        product_views(view(&l3), down_view.subcols(n, n)),
    ];
    dimension(&value).map_err(|_| Error::Singular)?;
    Ok((
        value,
        AddResidual {
            lower: [l1, l3],
            upper: [u0, u2],
            solve,
            down,
        },
    ))
}

/// `target += left * right`.
fn accumulate_product(
    target: &mut DMatrix<Complex>,
    left: MatRef<'_, Complex>,
    right: MatRef<'_, Complex>,
) {
    let (m, n, k) = (target.nrows(), target.ncols(), left.ncols());
    crate::threads::product(m, n, k, |par| {
        faer::linalg::matmul::matmul(
            view_mut(target),
            faer::Accum::Add,
            left,
            right,
            Complex::new(1.0, 0.0),
            par,
        );
    });
}

/// Cotangents of the inputs of [`add`], in the order of its arguments.
#[derive(Debug)]
pub struct AddGradient {
    /// Cotangents of the blocks of the lower S-matrix.
    pub lower: Blocks,
    /// Cotangents of the blocks of the upper S-matrix.
    pub upper: Blocks,
}

impl AddResidual {
    /// The shape `(n, n)` of each scattering block, with `n` modes in each direction.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.down.nrows(), self.down.nrows())
    }

    /// Input cotangents under `dL = Re(sum(conj(g) * dx))`; consumes the residual.
    pub fn pullback(self, cotangent: &Blocks) -> Result<AddGradient> {
        let n = checked_dimension(
            cotangent,
            self.down.nrows(),
            "invalid S matrix cotangent shape",
        )?;
        let Self {
            lower: [mut reflection, mut transmission],
            upper: [mut top_transmission, mut bottom_reflection],
            solve,
            down: down_fields,
        } = self;
        let mut top = DMatrix::zeros(n, 2 * n);
        top.columns_mut(0, n).copy_from(&cotangent[0]);
        top.columns_mut(n, n).copy_from(&cotangent[1]);
        let mut bottom = DMatrix::zeros(n, 2 * n);
        bottom.columns_mut(0, n).copy_from(&cotangent[2]);
        bottom.columns_mut(n, n).copy_from(&cotangent[3]);
        let down = product_views(view(&transmission).adjoint(), view(&bottom));
        let adjoint = product_views(view(&top_transmission).adjoint(), view(&top))
            + product_views(view(&bottom_reflection).adjoint(), view(&down));
        let AdjointSolve { adjoint, value: up } = solve.solve_adjoint(adjoint)?;
        let incident = down + product_views(view(&reflection).adjoint(), view(&adjoint));
        // The saved input blocks are read for the last time above; overwrite them with
        // their gradients.
        product_adjoint_right_into(&mut reflection, &adjoint, &down_fields);
        product_adjoint_right_into(&mut transmission, &bottom, &down_fields);
        product_adjoint_right_into(&mut top_transmission, &top, &up);
        product_adjoint_right_into(&mut bottom_reflection, &incident, &up);
        let lower = [
            adjoint.columns(0, n).into_owned(),
            reflection,
            cotangent[2].clone(),
            transmission,
        ];
        let upper = [
            top_transmission,
            cotangent[1].clone(),
            bottom_reflection,
            incident.columns(n, n).into_owned(),
        ];
        Ok(AddGradient { lower, upper })
    }
}

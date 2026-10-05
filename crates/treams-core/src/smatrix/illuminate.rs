//! Illumination of a pair of adjacent stacks: the outgoing fields and the internal
//! fields between them, with the pullback of both.
//!
//! Upstream: `treams.SMatrices.illuminate` with a second S-matrix (`smat=`).

mod saved;

use faer::MatRef;
use nalgebra::DMatrix;
use rayon::prelude::*;

use super::{
    Blocks, StoredBlock, any_nonfinite, checked_dimension,
    solve::{AdjointSolve, InternalSolve},
};
use crate::{
    Complex, Error, Result,
    linalg::{product, product_adjoint_right, product_views, view},
    numerics::finite,
};

/// What [`illuminate`] saves for its pullback: copies of the blocks of both stacks, the
/// incoming fields, the internal-field solve and the downgoing internal fields.
///
/// The solve keeps a dense LU factorization unless it solved iteratively. The iterative
/// (Krylov) solve needs at least 512 rows, at most 8 illumination columns, an internal
/// operator that is a proven contraction, and convergence within 16 steps.
#[derive(Debug)]
pub struct IlluminateResidual {
    lower: [StoredBlock; 4],
    upper: [StoredBlock; 4],
    incoming: [DMatrix<Complex>; 2],
    solve: InternalSolve,
    down: DMatrix<Complex>,
}

/// Rows from which the eight recorded input blocks are copied concurrently, when
/// the Rayon pool has more than one thread.
const PARALLEL_SNAPSHOT_ROWS: usize = 128;

/// Rows from which a one-thread pool copies the recorded input blocks on its
/// worker rather than on the calling thread.
const WORKER_SNAPSHOT_ROWS: usize = 512;

/// Rows from which the finite check of the illumination inputs scans the ten
/// matrices in parallel.
const PARALLEL_SCAN_ROWS: usize = 256;

/// Illuminate a pair of stacks. Returns outgoing up/down and internal up/down fields;
/// each input and output matrix has one column per independent illumination.
///
/// The residual keeps copies of the borrowed blocks, in their memory order, for the
/// pullback.
///
/// Upstream: `treams.SMatrices.illuminate`, called as `lower.illuminate(illu, illu2, smat=upper)`.
pub fn illuminate(
    lower: [MatRef<'_, Complex>; 4],
    upper: [MatRef<'_, Complex>; 4],
    incoming: [DMatrix<Complex>; 2],
) -> Result<([DMatrix<Complex>; 4], IlluminateResidual)> {
    // The pages of fresh copies fault and fill at memory speed; copy the eight
    // blocks concurrently before the solve takes the workers.
    let store = |blocks: [MatRef<'_, Complex>; 4]| {
        let [a, b, c, d] = blocks;
        let ((a, b), (c, d)) = crate::threads::join(
            || crate::threads::join(|| StoredBlock::copy_view(a), || StoredBlock::copy_view(b)),
            || crate::threads::join(|| StoredBlock::copy_view(c), || StoredBlock::copy_view(d)),
        );
        [a, b, c, d]
    };
    // A one-thread pool gains nothing from concurrency. Large recorded snapshots
    // are allocated on its worker, alongside the numerical work; small snapshots
    // avoid dispatching to the worker just to copy their inputs.
    let threshold = if crate::threads::current_num_threads() > 1 {
        PARALLEL_SNAPSHOT_ROWS
    } else {
        WORKER_SNAPSHOT_ROWS
    };
    let (lower, upper) = if lower[0].nrows() >= threshold {
        crate::threads::join(|| store(lower), || store(upper))
    } else {
        (
            lower.map(StoredBlock::copy_view),
            upper.map(StoredBlock::copy_view),
        )
    };
    let (top, bottom, down, solve) = illumination_fields(
        lower.each_ref().map(StoredBlock::view),
        upper.each_ref().map(StoredBlock::view),
        incoming.each_ref().map(view),
    )?;
    let value = [top, bottom, solve.value.clone(), down.clone()];
    Ok((
        value,
        IlluminateResidual {
            lower,
            upper,
            incoming,
            solve,
            down,
        },
    ))
}

/// The fields of [`illuminate`] without the data its pullback needs.
/// Inputs can be row- or column-major; only the operator and thin fields are owned.
///
/// Upstream: `treams.SMatrices.illuminate`, called as
/// `lower.illuminate(illu, illu2, smat=upper)`.
pub fn illuminate_value(
    lower: [MatRef<'_, Complex>; 4],
    upper: [MatRef<'_, Complex>; 4],
    incoming: [MatRef<'_, Complex>; 2],
) -> Result<[DMatrix<Complex>; 4]> {
    let (top, bottom, down, solve) = illumination_fields(lower, upper, incoming)?;
    Ok([top, bottom, solve.value, down])
}

/// Outgoing up, outgoing down and internal down fields, and the internal up solve.
type InternalFields = (
    DMatrix<Complex>,
    DMatrix<Complex>,
    DMatrix<Complex>,
    InternalSolve,
);

/// Validate the inputs and compute every field of [`illuminate`].
fn illumination_fields(
    lower: [MatRef<'_, Complex>; 4],
    upper: [MatRef<'_, Complex>; 4],
    incoming: [MatRef<'_, Complex>; 2],
) -> Result<InternalFields> {
    let n = lower[0].nrows();
    let p = incoming[0].ncols();
    let nonfinite = |a: &MatRef<'_, Complex>| any_nonfinite(*a);
    if n == 0
        || p == 0
        || lower.iter().chain(&upper).any(|a| a.shape() != (n, n))
        || incoming.iter().any(|a| a.shape() != (n, p))
        || if n >= PARALLEL_SCAN_ROWS {
            crate::threads::install(|| {
                lower
                    .par_iter()
                    .chain(upper.par_iter())
                    .chain(incoming.par_iter())
                    .any(nonfinite)
            })
        } else {
            lower.iter().chain(&upper).chain(&incoming).any(nonfinite)
        }
    {
        return Err(Error::InvalidInput(
            "require matching finite S matrices and mode-by-illumination inputs".into(),
        ));
    }
    let direct = product_views(upper[3], incoming[1]);
    let rhs = product_views(lower[0], incoming[0]) + product_views(lower[1], view(&direct));
    let solve = InternalSolve::new(lower[1], upper[2], rhs)?;
    let down = product_views(upper[2], view(&solve.value)) + direct;
    let top = product_views(upper[0], view(&solve.value)) + product_views(upper[1], incoming[1]);
    let bottom = product_views(lower[2], incoming[0]) + product_views(lower[3], view(&down));
    if top
        .iter()
        .chain(bottom.iter())
        .chain(down.iter())
        .any(|&z| !finite(z))
    {
        return Err(Error::Singular);
    }
    Ok((top, bottom, down, solve))
}

/// Cotangents of the inputs of [`illuminate`], in the order of its arguments.
#[derive(Debug)]
pub struct IlluminateGradient {
    /// Cotangents of the blocks of the lower S-matrix.
    pub lower: Blocks,
    /// Cotangents of the blocks of the upper S-matrix.
    pub upper: Blocks,
    /// Cotangents of the incoming up and down amplitudes.
    pub incoming: [DMatrix<Complex>; 2],
}

impl IlluminateResidual {
    /// Mode and independent-illumination counts.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.solve.value.shape()
    }

    /// Propagate stack and incident-field tangents through the recorded solve.
    /// Products remain mode-by-illumination arrays for a thin illumination batch.
    pub fn pushforward(
        &self,
        lower: &Blocks,
        upper: &Blocks,
        incoming: &[DMatrix<Complex>; 2],
    ) -> Result<[DMatrix<Complex>; 4]> {
        let (n, _) = self.shape();
        checked_dimension(lower, n, "invalid lower illumination S matrix tangent")?;
        checked_dimension(upper, n, "invalid upper illumination S matrix tangent")?;
        if incoming
            .iter()
            .any(|a| a.shape() != self.shape() || a.iter().any(|&z| !finite(z)))
        {
            return Err(Error::InvalidInput(
                "invalid illumination input tangent".into(),
            ));
        }
        let up = &self.solve.value;
        let direct = product(&upper[2], up)
            + product(&upper[3], &self.incoming[1])
            + product_views(self.upper[3].view(), view(&incoming[1]));
        let rhs = product(&lower[0], &self.incoming[0])
            + product_views(self.lower[0].view(), view(&incoming[0]))
            + product(&lower[1], &self.down)
            + product_views(self.lower[1].view(), view(&direct));
        let tangent_up =
            self.solve
                .solve_forward(self.lower[1].view(), self.upper[2].view(), rhs)?;
        let tangent_down = direct + product_views(self.upper[2].view(), view(&tangent_up));
        let top = product(&upper[0], up)
            + product_views(self.upper[0].view(), view(&tangent_up))
            + product(&upper[1], &self.incoming[1])
            + product_views(self.upper[1].view(), view(&incoming[1]));
        let bottom = product(&lower[2], &self.incoming[0])
            + product_views(self.lower[2].view(), view(&incoming[0]))
            + product(&lower[3], &self.down)
            + product_views(self.lower[3].view(), view(&tangent_down));
        Ok([top, bottom, tangent_up, tangent_down])
    }

    /// Return lower/upper S matrices and incoming up/down amplitude cotangents.
    pub fn pullback(&self, cotangent: &[DMatrix<Complex>; 4]) -> Result<IlluminateGradient> {
        if cotangent
            .iter()
            .any(|a| a.shape() != self.shape() || a.iter().any(|&z| !finite(z)))
        {
            return Err(Error::InvalidInput(
                "invalid field coefficient cotangent".into(),
            ));
        }
        let down =
            &cotangent[3] + product_views(self.lower[3].view().adjoint(), view(&cotangent[1]));
        let up = &cotangent[2]
            + product_views(self.upper[0].view().adjoint(), view(&cotangent[0]))
            + product_views(self.upper[2].view().adjoint(), view(&down));
        let AdjointSolve {
            adjoint: rhs,
            value: internal_up,
        } = self
            .solve
            .solve_adjoint(self.lower[1].view(), self.upper[2].view(), up)?;
        let direct = down + product_views(self.lower[1].view().adjoint(), view(&rhs));
        let incoming = [
            product_views(self.lower[2].view().adjoint(), view(&cotangent[1]))
                + product_views(self.lower[0].view().adjoint(), view(&rhs)),
            product_views(self.upper[1].view().adjoint(), view(&cotangent[0]))
                + product_views(self.upper[3].view().adjoint(), view(&direct)),
        ];
        let lower = [
            product_adjoint_right(&rhs, &self.incoming[0]),
            product_adjoint_right(&rhs, &self.down),
            product_adjoint_right(&cotangent[1], &self.incoming[0]),
            product_adjoint_right(&cotangent[1], &self.down),
        ];
        let upper = [
            product_adjoint_right(&cotangent[0], internal_up),
            product_adjoint_right(&cotangent[0], &self.incoming[1]),
            product_adjoint_right(&direct, internal_up),
            product_adjoint_right(&direct, &self.incoming[1]),
        ];
        Ok(IlluminateGradient {
            lower,
            upper,
            incoming,
        })
    }
}

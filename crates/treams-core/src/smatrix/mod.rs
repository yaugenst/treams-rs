//! Planar scattering matrices of plane-wave modes and their analytic pullbacks.
//!
//! Covers Redheffer composition, chiral Fresnel and Cartesian interfaces,
//! propagation, multilayer stacks, internal illumination between stacks, periodic
//! transfer matrices and Bloch bands, radiation of periodic multipole arrays, port
//! transmittance and reflectance, and chirality-density forms.
//!
//! Upstream: `treams.SMatrices`, `treams.coeffs.fresnel` and `treams.chirality_density`
//! (see [Upstream counterparts](#upstream-counterparts)).
//!
//! # Sides and directions
//!
//! - The *negative side* lies below the structure (`z < 0` for a z normal); in [`add`]
//!   it holds the lower operand. The *positive side* lies above and holds the upper
//!   operand.
//! - Direction 0 is *up*, towards the positive side; direction 1 is *down*.
//! - [`Blocks`] index outgoing, then incident direction: `[S00, S01, S10, S11]`.
//! - [`fresnel`], [`interface()`] and [`layer_stack`] take their media in the order
//!   (negative side, positive side).
//! - [`tr()`] numbers its ports the other way: port 0 is the positive side, as in the
//!   material order `(above, below)` of upstream `SMatrices`.
//!
//! # Upstream counterparts
//!
//! | treams-rs | treams |
//! |---|---|
//! | [`add`] | `SMatrices.add` |
//! | [`periodic()`], [`bands`] | `SMatrices.periodic`, `SMatrices.bands_kz` |
//! | [`from_array`] | `SMatrices.from_array` |
//! | [`illuminate()`], [`illuminate_value`] | `SMatrices.illuminate` |
//! | [`tr()`], [`tr_value`] | `SMatrices.tr` |
//! | [`fresnel`] | `coeffs.fresnel` |
//! | [`interface()`], [`propagation`] | `SMatrices.interface`, `SMatrices.propagation` |
//! | [`layer_stack`] | `SMatrices.stack` of interfaces and propagations |
//! | [`chirality_density`], [`oriented_chirality`] | `chirality_density` |
//! | [`Blocks`] | the four `SMatrix` blocks of one `SMatrices` |
#![allow(clippy::indexing_slicing)] // Four blocks, validated equal matrix dimensions.

mod array;
mod chirality;
mod compose;
mod illuminate;
mod interface;
mod layers;
mod periodic;
mod solve;
mod tr;

use faer::MatRef;
use nalgebra::DMatrix;

pub use array::{Channels, FromArrayGradient, FromArrayResidual, from_array};
pub use chirality::{
    ChiralityDensityGradient, ChiralityDensityResidual, OrientedChiralityGradient,
    OrientedChiralityResidual, chirality_density, oriented_chirality,
};
pub use compose::{AddGradient, AddResidual, add};
pub use illuminate::{IlluminateGradient, IlluminateResidual, illuminate, illuminate_value};
pub use interface::{
    FresnelGradient, FresnelResidual, InterfaceGradient, InterfaceResidual, PropagationGradient,
    PropagationResidual, fresnel, interface, propagation,
};
pub use layers::{LayerStackGradient, LayerStackResidual, layer_stack};
pub use periodic::{BandsGradient, BandsResidual, PeriodicResidual, bands, periodic};
pub use tr::{TrGradient, TrPorts, TrResidual, tr, tr_value};

use crate::{Complex, Error, Result, linalg::view, numerics::finite};

/// The four scattering blocks of one S-matrix.
///
/// These are the four `ScatteringBlock`s of one `treams_rs.SMatrix`, each `S[out][in]`
/// indexed by propagation direction (0 = up, 1 = down).
///
/// The order is `[S00, S01, S10, S11]`: upward transmission, reflection of waves
/// incident from the positive side (above), reflection of waves incident from the
/// negative side (below), and downward transmission.
///
/// Upstream: the four `SMatrix` blocks of one `treams.SMatrices`.
pub type Blocks = [DMatrix<Complex>; 4];

/// Owned square block that preserves its input memory order.
#[derive(Debug)]
struct StoredBlock {
    values: DMatrix<Complex>,
    row_major: bool,
}

/// The matrix in its memory order: the transpose of row-major storage, which then
/// reads column by column, and whether it was transposed.
fn storage_order(matrix: MatRef<'_, Complex>) -> (MatRef<'_, Complex>, bool) {
    if matrix.row_stride() != 1 && matrix.col_stride() == 1 {
        (matrix.transpose(), true)
    } else {
        (matrix, false)
    }
}

/// Whether any entry is not finite, scanning contiguous storage without early exit
/// so the scan vectorizes.
fn any_nonfinite(matrix: MatRef<'_, Complex>) -> bool {
    let (matrix, _) = storage_order(matrix);
    if let Some(contiguous) = matrix.try_as_col_major() {
        (0..matrix.ncols()).any(|j| {
            contiguous
                .col(j)
                .as_slice()
                .iter()
                .fold(false, |bad, &z| bad | !finite(z))
        })
    } else {
        (0..matrix.ncols()).any(|j| (0..matrix.nrows()).any(|i| !finite(matrix[(i, j)])))
    }
}

impl StoredBlock {
    /// Copy `matrix` column by column in its memory order.
    fn copy_view(matrix: MatRef<'_, Complex>) -> Self {
        let (matrix, row_major) = storage_order(matrix);
        let mut values = Vec::with_capacity(matrix.nrows() * matrix.ncols());
        if let Some(contiguous) = matrix.try_as_col_major() {
            for j in 0..matrix.ncols() {
                values.extend_from_slice(contiguous.col(j).as_slice());
            }
        } else {
            for j in 0..matrix.ncols() {
                for i in 0..matrix.nrows() {
                    values.push(matrix[(i, j)]);
                }
            }
        }
        Self {
            values: DMatrix::from_vec(matrix.nrows(), matrix.ncols(), values),
            row_major,
        }
    }

    /// The copied matrix, transposed back when it was row-major.
    fn view(&self) -> MatRef<'_, Complex> {
        let data = view(&self.values);
        if self.row_major {
            data.transpose()
        } else {
            data
        }
    }

    /// The storage, for reuse as a gradient buffer of the same shape.
    fn into_buffer(self) -> DMatrix<Complex> {
        self.values
    }
}

/// The common dimension `n` of four finite, nonempty `n`-by-`n` blocks.
fn dimension(blocks: &Blocks) -> Result<usize> {
    let n = blocks[0].nrows();
    if n == 0
        || blocks
            .iter()
            .any(|m| m.shape() != (n, n) || m.iter().any(|&z| !finite(z)))
    {
        return Err(Error::InvalidInput(
            "S matrices require four finite, equally sized nonempty square blocks".into(),
        ));
    }
    Ok(n)
}

/// The dimension of cotangent blocks, which must match the forward `expected` one.
fn checked_dimension(g: &Blocks, expected: usize, message: &str) -> Result<usize> {
    if dimension(g)? != expected {
        return Err(Error::InvalidInput(message.into()));
    }
    Ok(expected)
}

/// Scattering blocks of `n` unscattered modes: identity transmission, no reflection.
pub(crate) fn identity_blocks(n: usize) -> Blocks {
    std::array::from_fn(|b| {
        if b == 0 || b == 3 {
            DMatrix::identity(n, n)
        } else {
            DMatrix::zeros(n, n)
        }
    })
}

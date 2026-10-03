//! Periodic transfer matrices of an S-matrix and the Bloch bands of its repetition.
//!
//! Upstream: `treams.SMatrices.periodic` and `treams.SMatrices.bands_kz`.

use nalgebra::DMatrix;

use super::{Blocks, dimension};
use crate::{
    Complex, Error, Result,
    linalg::{
        self, SolveGradient, SolveResidual, product, product_adjoint_right, product_views, view,
    },
    numerics::{finite, ratio},
};

/// What [`periodic()`] saves for its pullback: the upper rows of the transfer matrix, the
/// reflection block `S10` and the solve with `S11`.
#[derive(Debug)]
pub struct PeriodicResidual {
    top: DMatrix<Complex>,
    reflection: DMatrix<Complex>,
    solve: SolveResidual,
}

/// Convert scattering blocks to the transfer matrix used for periodic bands.
///
/// Upstream: `treams.SMatrices.periodic`.
pub fn periodic(blocks: Blocks) -> Result<(DMatrix<Complex>, PeriodicResidual)> {
    let n = dimension(&blocks)?;
    let [up, reflection_up, reflection_down, down] = blocks;
    let mut top = DMatrix::zeros(n, 2 * n);
    top.columns_mut(0, n).copy_from(&up);
    top.columns_mut(n, n).copy_from(&reflection_up);
    let mut rhs = -product(&reflection_down, &top);
    for i in 0..n {
        rhs[(i, n + i)] += 1.0;
    }
    let solve = linalg::solve_owned(down, rhs)?;
    let mut value = DMatrix::zeros(2 * n, 2 * n);
    value.rows_mut(0, n).copy_from(&top);
    value.rows_mut(n, n).copy_from(solve.value());
    Ok((
        value,
        PeriodicResidual {
            top,
            reflection: reflection_down,
            solve,
        },
    ))
}

impl PeriodicResidual {
    /// The shape of the transfer matrix: both propagation directions of every
    /// plane-wave mode, `2 n` by `2 n`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (2 * self.top.nrows(), 2 * self.top.nrows())
    }

    /// Pull back a transfer-matrix cotangent into all four scattering blocks.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<Blocks> {
        let n = self.top.nrows();
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput(
                "invalid periodic transfer cotangent".into(),
            ));
        }
        let SolveGradient {
            operator: down,
            rhs,
        } = self.solve.pullback(cotangent.rows(n, n).into_owned())?;
        let reflection = -product_adjoint_right(&rhs, &self.top);
        let top =
            cotangent.rows(0, n) - product_views(view(&self.reflection).adjoint(), view(&rhs));
        Ok([
            top.columns(0, n).into_owned(),
            top.columns(n, n).into_owned(),
            reflection,
            down,
        ])
    }
}

/// What [`bands`] saves for its pullback: the residuals of the transfer matrix and of
/// its eigensystem, the period and the Bloch wavenumbers.
#[derive(Debug)]
pub struct BandsResidual {
    periodic: PeriodicResidual,
    eigen: linalg::EigResidual,
    period: f64,
    wavenumbers: Vec<Complex>,
}

/// Bloch modes of a periodically repeated S matrix with positive repeat distance.
///
/// Upstream: `treams.SMatrices.bands_kz(az)` with `az = period`.
pub fn bands(blocks: Blocks, period: f64) -> Result<BandsResidual> {
    if !period.is_finite() || period <= 0.0 {
        return Err(Error::InvalidInput(
            "band period must be finite and positive".into(),
        ));
    }
    let (transfer, periodic) = periodic(blocks)?;
    let eigen = linalg::eig(&transfer)?;
    let wavenumbers: Vec<_> = eigen
        .values()
        .iter()
        .map(|v| -Complex::i() * v.ln() / period)
        .collect();
    if wavenumbers.iter().any(|&z| !finite(z)) {
        return Err(Error::NonFinite(
            "zero or non-finite Bloch multiplier".into(),
        ));
    }
    Ok(BandsResidual {
        periodic,
        eigen,
        period,
        wavenumbers,
    })
}

/// Cotangents of the inputs of [`bands`], in the order of its arguments.
#[derive(Debug)]
pub struct BandsGradient {
    /// Cotangents of the four scattering blocks.
    pub blocks: Blocks,
    /// Cotangent of the repeat distance.
    pub period: f64,
}

impl BandsResidual {
    /// Normal Bloch wavenumbers from the principal complex logarithm, which the
    /// pullback reads.
    #[must_use]
    pub fn wavenumbers(&self) -> &[Complex] {
        &self.wavenumbers
    }

    /// Unit right Bloch eigenvectors, with the largest component real positive.
    #[must_use]
    pub const fn vectors(&self) -> &DMatrix<Complex> {
        self.eigen.vectors()
    }

    /// Return S-matrix and repeat-distance cotangents; logarithm branch is fixed.
    pub fn pullback(
        self,
        wavenumbers: &[Complex],
        vectors: DMatrix<Complex>,
    ) -> Result<BandsGradient> {
        if wavenumbers.len() != self.wavenumbers.len() || wavenumbers.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput(
                "invalid Bloch wavenumber cotangent".into(),
            ));
        }
        let period = -wavenumbers
            .iter()
            .zip(&self.wavenumbers)
            .map(|(g, k)| (g.conj() * k).re)
            .sum::<f64>()
            / self.period;
        let values: Vec<_> = wavenumbers
            .iter()
            .zip(self.eigen.values())
            .map(|(g, l)| g * ratio(-Complex::i() / self.period, *l).conj())
            .collect();
        let matrix = self.eigen.pullback(&values, vectors)?;
        Ok(BandsGradient {
            blocks: self.periodic.pullback(&matrix)?,
            period,
        })
    }
}

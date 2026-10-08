//! Circular dichroism, duality breaking and electromagnetic chirality of helicity T-matrices.
//!
//! Upstream: `treams.TMatrix.cd`, `treams.TMatrix.db` and `treams.TMatrix.chi`.
#![allow(clippy::indexing_slicing)] // Validated square matrix, polarizations and helicity groups.

mod saved;

use nalgebra::DMatrix;

use crate::linalg::SvdvalsResidual;
use crate::{Complex, Error, Result, numerics::finite};

/// Normalized measures of a global helicity T-matrix.
#[derive(Clone, Copy, Debug)]
pub enum Metric {
    /// Absorption circular dichroism in a real embedding medium. Upstream:
    /// `treams.TMatrix.cd`.
    CircularDichroism,
    /// Fraction of scattering that changes helicity. Upstream: `treams.TMatrix.db`.
    DualityBreaking,
    /// Electromagnetic chirality from the singular values of the four helicity blocks.
    /// Upstream: `treams.TMatrix.chi`.
    Chirality,
}

/// What [`metric`] saves for its pullback: the gradient itself, since the value is a
/// real scalar; no singular vectors are kept.
#[derive(Debug)]
pub struct MetricResidual {
    /// The matrix gradient, or why it is undefined: chirality may be zero where its
    /// derivative is undefined.
    gradient: Result<DMatrix<Complex>>,
    ks_gradient: [f64; 2],
    dimension: usize,
}

/// Gradients of the inputs of [`metric`], in the order of its arguments.
#[derive(Clone, Debug)]
pub struct MetricGradient {
    /// Gradient of the helicity T-matrix.
    pub matrix: DMatrix<Complex>,
    /// Gradients of the real embedding wavenumbers; zero except for circular
    /// dichroism.
    pub ks: [f64; 2],
}

impl MetricResidual {
    /// The shape of the recorded T-matrix.
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.dimension, self.dimension)
    }

    /// The real metric tangent, from the saved scalar gradient and changes of the
    /// T-matrix and embedding wavenumbers, using the real complex inner product.
    pub fn pushforward(&self, matrix: &DMatrix<Complex>, ks: [f64; 2]) -> Result<f64> {
        if matrix.shape() != (self.dimension, self.dimension)
            || matrix.iter().any(|&z| !finite(z))
            || ks.iter().any(|k| !k.is_finite())
        {
            return Err(Error::InvalidInput(
                "metric tangents must be finite and match the recorded inputs".into(),
            ));
        }
        if matrix.iter().all(|&z| z == Complex::default()) && ks.iter().all(|&k| k == 0.0) {
            return Ok(0.0);
        }
        Ok(self
            .gradient
            .as_ref()
            .map_err(Clone::clone)?
            .dotc(matrix)
            .re
            + self
                .ks_gradient
                .iter()
                .zip(ks)
                .map(|(g, d)| g * d)
                .sum::<f64>())
    }

    /// Gradients of the matrix and the embedding wavenumbers from `cotangent`, the
    /// gradient of a real loss with respect to the metric.
    ///
    /// The pullback scales the saved gradient on the calling thread, so the Rayon pool
    /// does not change it.
    pub fn pullback(&self, cotangent: f64) -> Result<MetricGradient> {
        if !cotangent.is_finite() {
            return Err(Error::InvalidInput(
                "metric cotangent must be finite".into(),
            ));
        }
        if cotangent == 0.0 {
            return Ok(MetricGradient {
                matrix: DMatrix::zeros(self.dimension, self.dimension),
                ks: [0.0; 2],
            });
        }
        Ok(MetricGradient {
            matrix: self
                .gradient
                .as_ref()
                .map_err(Clone::clone)?
                .map(|z| z * cotangent),
            ks: self.ks_gradient.map(|k| k * cotangent),
        })
    }
}

/// One [`Metric`] of a global helicity T-matrix.
///
/// `pol` is the polarization index of each row and `ks` holds the negative and positive
/// helicity wavenumbers of the embedding, which enter circular dichroism only.
///
/// Upstream: `treams.TMatrix.cd`, `treams.TMatrix.db` and `treams.TMatrix.chi`.
/// Differences: circular dichroism requires positive real wavenumbers where treams
/// requires a real material; the other two scale the matrix to a largest entry of 1
/// first; the chirality gradient is an error at zero helicity contrast.
pub fn metric(
    matrix: &DMatrix<Complex>,
    pol: &[u8],
    ks: [f64; 2],
    kind: Metric,
) -> Result<(f64, MetricResidual)> {
    let dimension = matrix.nrows();
    if dimension == 0
        || !matrix.is_square()
        || pol.len() != dimension
        || pol.iter().any(|&p| p > 1)
        || matrix.iter().any(|&z| !finite(z))
        || ks.iter().any(|k| !k.is_finite() || *k <= 0.0)
    {
        return Err(Error::InvalidInput("require a finite square helicity matrix, matching polarizations and positive real wavenumbers".into()));
    }
    if matches!(kind, Metric::CircularDichroism) {
        return circular_dichroism(matrix, pol, ks);
    }
    // Normalized metrics are invariant to a common matrix scale. Work at unit
    // scale so norm squares do not overflow or underflow for weak scattering.
    let scale = matrix.iter().map(|z| z.norm()).fold(0.0, f64::max);
    if scale == 0.0 {
        return Err(Error::InvalidInput(
            "normalized metric requires nonzero scattering".into(),
        ));
    }
    let matrix = matrix.map(|z| z / scale);
    let (value, gradient) = if matches!(kind, Metric::DualityBreaking) {
        let (value, gradient) = duality_breaking(&matrix, pol);
        (value, Ok(gradient))
    } else {
        chirality(&matrix, pol)?
    };
    Ok((
        value,
        MetricResidual {
            gradient: gradient.map(|g| g.map(|z| z / scale)),
            ks_gradient: [0.0; 2],
            dimension,
        },
    ))
}

/// Absorption circular dichroism with its matrix and wavenumber gradients.
fn circular_dichroism(
    matrix: &DMatrix<Complex>,
    pol: &[u8],
    ks: [f64; 2],
) -> Result<(f64, MetricResidual)> {
    let dimension = matrix.nrows();
    let weights: Vec<_> = pol.iter().map(|&p| ks[usize::from(p)].powi(-2)).collect();
    let mut absorption = [0.0; 2];
    for j in 0..dimension {
        absorption[usize::from(pol[j])] -= matrix[(j, j)].re * weights[j]
            + (0..dimension)
                .map(|i| matrix[(i, j)].norm_sqr() * weights[i])
                .sum::<f64>();
    }
    let total = absorption.iter().sum::<f64>();
    if total == 0.0 || !total.is_finite() {
        return Err(Error::InvalidInput(
            "circular dichroism requires nonzero total absorption".into(),
        ));
    }
    let value = (absorption[1] - absorption[0]) / total;
    let g = [
        -2.0 * absorption[1] / total / total,
        2.0 * absorption[0] / total / total,
    ];
    let mut gradient = DMatrix::from_fn(dimension, dimension, |i, j| {
        -2.0 * g[usize::from(pol[j])] * weights[i] * matrix[(i, j)]
    });
    let mut gks = [0.0; 2];
    for i in 0..dimension {
        let p = usize::from(pol[i]);
        gradient[(i, i)] -= g[p] * weights[i];
        gks[p] += 2.0 * weights[i] / ks[p]
            * (g[p] * matrix[(i, i)].re
                + (0..dimension)
                    .map(|j| g[usize::from(pol[j])] * matrix[(i, j)].norm_sqr())
                    .sum::<f64>());
    }
    Ok((
        value,
        MetricResidual {
            gradient: Ok(gradient),
            ks_gradient: gks,
            dimension,
        },
    ))
}

/// The fraction of scattering that flips helicity and its gradient, for a matrix at
/// unit scale.
fn duality_breaking(matrix: &DMatrix<Complex>, pol: &[u8]) -> (f64, DMatrix<Complex>) {
    let dimension = matrix.nrows();
    let norm = matrix.norm_squared();
    let flipped = (0..dimension)
        .flat_map(|j| (0..dimension).map(move |i| (i, j)))
        .filter(|&(i, j)| pol[i] != pol[j])
        .map(|(i, j)| matrix[(i, j)].norm_sqr())
        .sum::<f64>();
    let value = flipped / norm;
    let gradient = DMatrix::from_fn(dimension, dimension, |i, j| {
        matrix[(i, j)] * (2.0 * (f64::from(pol[i] != pol[j]) - value) / norm)
    });
    (value, gradient)
}

/// One helicity block of the chirality metric and its place in the singular-value
/// differences.
#[derive(Clone, Copy)]
struct HelicityBlock {
    /// Helicity of the rows.
    row: usize,
    /// Helicity of the columns.
    col: usize,
    /// The difference of spectra the block enters: 0 is `spectrum(1, 1) -
    /// spectrum(0, 0)`, 1 is `spectrum(1, 0) - spectrum(0, 1)`.
    difference: usize,
    /// The sign of the block's spectrum in that difference.
    sign: f64,
}

/// The four helicity blocks, ordered so that block `3 - p` minus block `p` is
/// difference `p`, as treams pairs them in `TMatrix.chi`.
const CHIRALITY_BLOCKS: [HelicityBlock; 4] = [
    HelicityBlock {
        row: 0,
        col: 0,
        difference: 0,
        sign: -1.0,
    },
    HelicityBlock {
        row: 0,
        col: 1,
        difference: 1,
        sign: -1.0,
    },
    HelicityBlock {
        row: 1,
        col: 0,
        difference: 1,
        sign: 1.0,
    },
    HelicityBlock {
        row: 1,
        col: 1,
        difference: 0,
        sign: 1.0,
    },
];

/// Electromagnetic chirality from the singular values of the four helicity blocks,
/// for a matrix at unit scale. Its gradient is undefined at zero contrast.
fn chirality(matrix: &DMatrix<Complex>, pol: &[u8]) -> Result<(f64, Result<DMatrix<Complex>>)> {
    let indices: [Vec<_>; 2] = std::array::from_fn(|p| {
        pol.iter()
            .enumerate()
            .filter_map(|(i, &q)| (usize::from(q) == p).then_some(i))
            .collect()
    });
    let count = indices[0].len();
    if count == 0 || count != indices[1].len() {
        return Err(Error::InvalidInput(
            "chirality requires equally sized nonempty helicity groups".into(),
        ));
    }
    let block = |row: usize, col: usize| {
        DMatrix::from_fn(count, count, |i, j| {
            matrix[(indices[row][i], indices[col][j])]
        })
    };
    let spectra = CHIRALITY_BLOCKS
        .iter()
        .map(|&HelicityBlock { row, col, .. }| crate::linalg::svdvals(&block(row, col)))
        .collect::<Result<Vec<_>>>()?;
    let differences: [Vec<_>; 2] = std::array::from_fn(|p| {
        spectra[3 - p]
            .values()
            .iter()
            .zip(spectra[p].values())
            .map(|(a, b)| a - b)
            .collect()
    });
    let contrast = differences
        .iter()
        .flatten()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    let value = contrast / matrix.norm_squared().sqrt();
    // The value stays available where the gradient is undefined.
    let gradient = if contrast == 0.0 {
        Err(Error::Derivative(
            crate::DerivativeError::ZeroChiralityContrast,
        ))
    } else {
        chirality_gradient(matrix, &indices, spectra, &differences, contrast)
    };
    Ok((value, gradient))
}

/// The chirality gradient through the singular values of the helicity blocks.
fn chirality_gradient(
    matrix: &DMatrix<Complex>,
    indices: &[Vec<usize>; 2],
    spectra: Vec<SvdvalsResidual>,
    differences: &[Vec<f64>; 2],
    contrast: f64,
) -> Result<DMatrix<Complex>> {
    let norm = matrix.norm_squared();
    let value = contrast / norm.sqrt();
    let mut gradient = matrix.map(|z| -value * z / norm);
    for (
        residual,
        &HelicityBlock {
            row,
            col,
            difference,
            sign,
        },
    ) in spectra.into_iter().zip(&CHIRALITY_BLOCKS)
    {
        let weights: Vec<_> = differences[difference]
            .iter()
            .map(|x| sign * x / contrast / norm.sqrt())
            .collect();
        let block = residual.pullback(&weights)?;
        for (j, &target_col) in indices[col].iter().enumerate() {
            for (i, &target_row) in indices[row].iter().enumerate() {
                gradient[(target_row, target_col)] += block[(i, j)];
            }
        }
    }
    Ok(gradient)
}

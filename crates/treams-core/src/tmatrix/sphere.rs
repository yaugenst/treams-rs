//! T-matrices of multilayer chiral spheres from their Mie coefficients.
//!
//! Upstream: `treams.TMatrix.sphere`.
#![allow(clippy::indexing_slicing)] // Degree indices of the 2 x 2 helicity blocks.

use nalgebra::DMatrix;

use crate::coeffs::{
    LayerGradient, Material, Matrix2, MieGradient, MieResidual, mie, to_mode_order,
};
use crate::sw;
use crate::{Complex, Error, Result, numerics::finite};

/// What [`sphere`] saves for its pullback: the Mie residual of each degree.
#[derive(Clone, Debug)]
pub struct SphereResidual {
    k0: f64,
    radii: Vec<f64>,
    degrees: Vec<MieResidual>,
}

/// Gradients of the inputs of [`sphere`], in the order of its arguments.
#[derive(Clone, Debug)]
pub struct SphereGradient {
    /// Gradient of the vacuum wavenumber.
    pub k0: f64,
    /// Gradients of the radii and the materials.
    pub layers: LayerGradient,
}

/// The T-matrix of a concentric multilayer chiral sphere, in the order of [`sw::modes`].
///
/// `radii` are the boundary radii, from inner to outer, and `materials` has one more
/// entry, the embedding medium last. Each degree `l` contributes the Mie coefficients
/// [`mie`] at the size parameters `k0 r`, repeated on its `2 l + 1` orders.
///
/// Upstream: `treams.TMatrix.sphere`. Differences: the helicity convention only.
pub fn sphere(
    lmax: u32,
    k0: f64,
    radii: &[f64],
    materials: &[Material],
) -> Result<(DMatrix<Complex>, SphereResidual)> {
    if !k0.is_finite() || k0 <= 0.0 {
        return Err(Error::InvalidInput("k0 must be finite and positive".into()));
    }
    let dimension = sw::modes(lmax)?.len();
    let sizes: Vec<_> = radii.iter().map(|r| k0 * r).collect();
    let degrees = (1..=lmax)
        .map(|l| mie(l, &sizes, materials))
        .collect::<Result<Vec<_>>>()?;
    let mut value = DMatrix::zeros(dimension, dimension);
    for (block, degree) in block_degrees(degrees.len()).enumerate() {
        value
            .fixed_view_mut::<2, 2>(2 * block, 2 * block)
            .copy_from(&to_mode_order(degrees[degree].value()));
    }
    Ok((
        value,
        SphereResidual {
            k0,
            radii: radii.to_vec(),
            degrees,
        },
    ))
}

/// The degree index `l - 1` of each 2 x 2 diagonal helicity block of a T-matrix in
/// [`sw::modes`] order with `count` degrees, which has one block per order `m` of each
/// degree `l`.
pub(crate) fn block_degrees(count: usize) -> impl Iterator<Item = usize> {
    (0..count).flat_map(|degree| std::iter::repeat_n(degree, 2 * degree + 3))
}

impl SphereResidual {
    /// The shape of the T-matrix: `2 lmax (lmax + 2)` modes, as [`sw::modes`] lists them.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        let lmax = self.degrees.len();
        (2 * lmax * (lmax + 2), 2 * lmax * (lmax + 2))
    }

    /// Gradients of `k0`, the radii and the materials from `cotangent`, the gradient of
    /// a real loss with respect to the T-matrix.
    ///
    /// The pullback runs on the calling thread and adds the degrees in order, so the
    /// Rayon pool does not change it.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<SphereGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|z| !finite(*z)) {
            return Err(Error::InvalidInput(
                "invalid spherical T-matrix cotangent".into(),
            ));
        }
        let mut blocks = vec![Matrix2::zeros(); self.degrees.len()];
        for (block, degree) in block_degrees(self.degrees.len()).enumerate() {
            let g = cotangent
                .fixed_view::<2, 2>(2 * block, 2 * block)
                .into_owned();
            blocks[degree] += to_mode_order(&g);
        }
        let mut sum = MieGradient::zeros(self.radii.len());
        for (residual, g) in self.degrees.into_iter().zip(blocks) {
            sum.accumulate(&residual.pullback(&g)?);
        }
        // The size parameters k0 r carry both the radius and the k0 gradients. The
        // matrix-free cluster applies this chain rule per degree instead
        // (cluster::IterativeSphereCluster::local_gradients), which rounds differently.
        let k0 = sum.sizes.iter().zip(&self.radii).map(|(g, r)| g * r).sum();
        let radii = sum.sizes.iter().map(|g| self.k0 * g).collect();
        Ok(SphereGradient {
            k0,
            layers: LayerGradient {
                radii,
                epsilon: sum.epsilon,
                mu: sum.mu,
                kappa: sum.kappa,
            },
        })
    }
}

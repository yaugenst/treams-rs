//! Isotropic, optionally chiral materials and the validation of concentric layers.
//!
//! Upstream: `treams.Material` and `treams.misc.refractive_index`.
#![allow(clippy::indexing_slicing)] // Adjacent boundaries from `windows(2)`.

use std::ops::AddAssign;

use crate::{Complex, Error, Result, numerics::finite};

/// Isotropic reciprocal material, optionally chiral.
///
/// Upstream: `treams.Material`. Differences: [`index`](Self::index) keeps the
/// principal square root, as `treams.coeffs.mie` does, where `Material.n` flips its
/// sign for a negative imaginary part.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// Relative permittivity.
    pub epsilon: Complex,
    /// Relative permeability.
    pub mu: Complex,
    /// Chirality parameter.
    pub kappa: Complex,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            epsilon: Complex::new(1.0, 0.0),
            mu: Complex::new(1.0, 0.0),
            kappa: Complex::default(),
        }
    }
}

impl Material {
    /// The refractive index `sqrt(epsilon mu)` on the principal branch.
    #[must_use]
    pub fn index(self) -> Complex {
        crate::numerics::complex_sqrt(self.epsilon * self.mu)
    }
    /// The negative and positive helicity indices `sqrt(epsilon mu) -+ kappa`, each
    /// negated if its imaginary part is negative.
    ///
    /// Upstream: `treams.misc.refractive_index`.
    #[must_use]
    pub fn indices(self) -> [Complex; 2] {
        let n = self.index();
        [n - self.kappa, n + self.kappa].map(|v| if v.im < 0.0 { -v } else { v })
    }
    /// The relative impedance `sqrt(mu / epsilon)` on the principal branch.
    ///
    /// Upstream: `treams.Material.impedance`.
    #[must_use]
    pub fn impedance(self) -> Complex {
        crate::numerics::complex_sqrt(self.mu / self.epsilon)
    }
    /// Derivatives of the index, impedance and chirality with respect to epsilon, mu
    /// and kappa, in that order.
    pub(super) fn tangents(self) -> [MaterialTangent; 3] {
        let (n, impedance) = (self.index(), self.impedance());
        let zero = Complex::default();
        [
            MaterialTangent {
                index: self.mu / (2.0 * n),
                impedance: -impedance / (2.0 * self.epsilon),
                kappa: zero,
            },
            MaterialTangent {
                index: self.epsilon / (2.0 * n),
                impedance: impedance / (2.0 * self.mu),
                kappa: zero,
            },
            MaterialTangent {
                index: zero,
                impedance: zero,
                kappa: Complex::new(1.0, 0.0),
            },
        ]
    }
}

/// Negative- and positive-helicity refractive indices `sqrt(epsilon mu) -+ kappa` of a
/// real material.
///
/// A negative product `epsilon mu` gives NaN, as `numpy.sqrt` of a negative float does.
/// Complex materials use [`Material::indices`].
///
/// Upstream: `treams.misc.refractive_index` with real arguments.
#[inline]
#[must_use]
pub fn real_refractive_indices(epsilon: f64, mu: f64, kappa: f64) -> [f64; 2] {
    let n = (epsilon * mu).sqrt();
    [n - kappa, n + kappa]
}

/// Gradients of concentric layers: the radii of their boundaries and their media.
#[derive(Clone, Debug)]
pub struct LayerGradient {
    /// Radius gradients, from inner to outer boundary.
    pub radii: Vec<f64>,
    /// Permittivity gradients, including the embedding medium.
    pub epsilon: Vec<Complex>,
    /// Permeability gradients, including the embedding medium.
    pub mu: Vec<Complex>,
    /// Chirality gradients, including the embedding medium.
    pub kappa: Vec<Complex>,
}

impl LayerGradient {
    /// Zero gradients of `boundaries` radii and `boundaries + 1` media.
    pub(crate) fn zeros(boundaries: usize) -> Self {
        let media = vec![Complex::default(); boundaries + 1];
        Self {
            radii: vec![0.0; boundaries],
            epsilon: media.clone(),
            mu: media.clone(),
            kappa: media,
        }
    }

    /// Add the gradient of the same layers along another cotangent.
    pub(crate) fn accumulate(&mut self, other: &Self) {
        add_entries(&mut self.radii, &other.radii);
        add_entries(&mut self.epsilon, &other.epsilon);
        add_entries(&mut self.mu, &other.mu);
        add_entries(&mut self.kappa, &other.kappa);
    }
}

/// `sums[i] += values[i]` for every entry.
pub(super) fn add_entries<T: Copy + AddAssign>(sums: &mut [T], values: &[T]) {
    for (sum, &value) in sums.iter_mut().zip(values) {
        *sum += value;
    }
}

/// Derivatives of a material's index, impedance and chirality along one parameter.
#[derive(Clone, Copy, Debug)]
pub(super) struct MaterialTangent {
    pub(super) index: Complex,
    pub(super) impedance: Complex,
    pub(super) kappa: Complex,
}

/// Shared material and concentric-boundary validation.
pub(crate) fn validate_layers(sizes: &[f64], materials: &[Material]) -> Result<()> {
    if sizes.is_empty() || materials.len() != sizes.len() + 1 {
        return Err(Error::InvalidInput(
            "require one more material than radii".into(),
        ));
    }
    if sizes.iter().any(|x| !x.is_finite() || *x <= 0.0) || sizes.windows(2).any(|x| x[0] >= x[1]) {
        return Err(Error::InvalidInput(
            "radii and size parameters must be finite, positive and strictly increasing".into(),
        ));
    }
    for material in materials {
        if !finite(material.epsilon)
            || !finite(material.mu)
            || !finite(material.kappa)
            || material.epsilon.norm_sqr() == 0.0
            || material.mu.norm_sqr() == 0.0
        {
            return Err(Error::InvalidInput(
                "require finite materials with nonzero epsilon and mu".into(),
            ));
        }
    }
    Ok(())
}

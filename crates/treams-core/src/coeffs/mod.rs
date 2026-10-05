//! Mie coefficients of multilayer chiral spheres and cylinders, and the materials they
//! take.
//!
//! Upstream: `treams.coeffs`.
//!
//! | treams-rs | treams |
//! |---|---|
//! | [`mie`](fn@mie) | `coeffs.mie` |
//! | [`mie_cyl`](fn@mie_cyl) | `coeffs.mie_cyl` |
//! | [`smatrix::fresnel`](crate::smatrix::fresnel) | `coeffs.fresnel` |
//! | [`Material`] | `treams.Material` |
//! | [`Material::indices`], [`real_refractive_indices`] | `misc.refractive_index` |
//!
//! Each pullback runs on the calling thread and adds its terms in a fixed order, so
//! the Rayon pool does not change the gradients.

mod material;
mod mie;
mod mie_cyl;

pub use material::{LayerGradient, Material, real_refractive_indices};
pub(crate) use material::{validate_layer_tangents, validate_layers};
pub(crate) use mie::to_mode_order;
pub use mie::{MieGradient, MieResidual, mie};
pub use mie_cyl::{MieCylGradient, MieCylResidual, mie_cyl};

use nalgebra::SMatrix;

use crate::Complex;

/// A 2 x 2 matrix over the two polarizations.
///
/// [`mie`](fn@mie) and [`mie_cyl`](fn@mie_cyl) use helicity order (negative, positive), as
/// `treams.coeffs.mie` does. T-matrix blocks swap both axes into mode order, which lists
/// polarization index 1 first.
pub type Matrix2 = SMatrix<Complex, 2, 2>;
pub(crate) type Matrix4 = SMatrix<Complex, 4, 4>;
pub(crate) type Matrix42 = SMatrix<Complex, 4, 2>;

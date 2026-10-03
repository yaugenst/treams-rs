//! T-matrices of single spheres and cylinders, and three helicity measures of a
//! T-matrix.
//!
//! Upstream: `treams.TMatrix` and `treams.TMatrixC`.
//!
//! | treams-rs | treams | Differences |
//! |---|---|---|
//! | [`sphere()`] | `TMatrix.sphere` | helicity convention only; the Python layer converts to parity |
//! | [`cylinder()`] | `TMatrixC.cylinder` | a block `(kz, m)` reuses the boundary solve of its mirror `(-kz, -m)` when both are requested; the two solves agree to rounding |
//! | [`metric()`] with [`Metric::CircularDichroism`] | `TMatrix.cd` | requires positive real wavenumbers instead of a real material |
//! | [`metric()`] with [`Metric::DualityBreaking`] | `TMatrix.db` | scales the matrix to a largest entry of 1 first, so tiny or huge matrices do not underflow or overflow |
//! | [`metric()`] with [`Metric::Chirality`] | `TMatrix.chi` | the same scaling; the gradient is an error where the helicity contrast is zero |
//!
//! [`sphere()`] and [`cylinder()`] list the modes as treams does: within each degree and
//! order (or axial wavenumber and order), polarization index 1 (positive helicity)
//! comes before 0. The Mie coefficients of [`coeffs`](crate::coeffs) use the reverse
//! order (negative, positive), so both axes of every 2 x 2 block swap when it is
//! placed. [`metric()`] takes the polarization index of each row in `pol`.

mod cylinder;
mod metric;
mod sphere;

pub use cylinder::{CylinderGradient, CylinderResidual, cylinder};
pub use metric::{Metric, MetricGradient, MetricResidual, metric};
pub(crate) use sphere::block_degrees;
pub use sphere::{SphereGradient, SphereResidual, sphere};

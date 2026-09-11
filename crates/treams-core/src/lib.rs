//! Electromagnetic T-matrix kernels and analytic pullbacks, independent of Python.

pub mod angular;
pub mod basis;
pub mod channels;
pub mod coeffs;
pub mod conversion;
pub mod coordinates;
pub mod cylinder;
pub mod cylwaves;
pub mod ebcm;
pub mod fields;
pub mod geometry;
pub mod integrals;
pub mod interaction;
mod jet;
pub mod lattice;
pub mod layers;
pub mod linalg;
pub mod plane;
pub mod polar;
pub mod rotation;
pub mod smatrix;
pub mod special;
pub mod tmatrix;
mod translation_plan;
pub mod vectorwaves;
pub mod waves;

// Algebraic principal square root avoids atan2/sin/cos in num-complex's polar
// implementation. Scale extreme finite inputs; preserve signed-zero branch sides.
#[inline]
pub(crate) fn complex_sqrt(z: Complex) -> Complex {
    if z.im == 0.0 || !finite(z) {
        return z.sqrt();
    }
    let scale = z.re.abs().max(z.im.abs());
    if !(1e-150..=1e150).contains(&scale) {
        return complex_sqrt(z / scale) * scale.sqrt();
    }
    let dominant = ((z.re.mul_add(z.re, z.im * z.im).sqrt() + z.re.abs()) * 0.5).sqrt();
    if z.re >= 0.0 {
        Complex::new(dominant, z.im / (2.0 * dominant))
    } else {
        Complex::new(z.im.abs() / (2.0 * dominant), dominant.copysign(z.im))
    }
}

#[cfg(test)]
mod properties;

/// Complex double precision used throughout the numerical core.
pub type Complex = num_complex::Complex64;

/// A numerical failure or invalid physical input.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The input does not satisfy the operation's contract.
    #[error("{0}")]
    InvalidInput(String),
    /// A special-function evaluation failed.
    #[error("special-function evaluation failed: {0}")]
    SpecialFunction(String),
    /// The scattering system is singular at the supplied parameters.
    #[error("singular scattering system")]
    Singular,
}

/// Numerical result with a typed error.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn finite(z: Complex) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

// Avoid squaring an unscaled complex denominator (overflow/underflow).
pub(crate) fn ratio(numerator: Complex, denominator: Complex) -> Complex {
    let scale = denominator.re.abs().max(denominator.im.abs());
    (numerator / scale) / (denominator / scale)
}

#[cfg(test)]
mod ratio_tests {
    use super::*;
    #[test]
    fn principal_square_root_preserves_scales_and_branch_sides() {
        for exponent in [-300, -155, -149, 0, 149, 155, 300] {
            for re in [-0.7, 0.0, 0.7] {
                for im in [-0.4, -0.0, 0.0, 0.4] {
                    let z = Complex::new(re, im) * 10.0_f64.powi(exponent);
                    let actual = complex_sqrt(z);
                    let expected = z.sqrt();
                    assert!((actual - expected).norm() <= 3e-14 * expected.norm());
                    assert_eq!(actual.im.is_sign_negative(), expected.im.is_sign_negative());
                    assert!((actual * actual - z).norm() <= 3e-14 * z.norm());
                }
            }
        }
        for im in [-1e-300, 1e-300] {
            let z = Complex::new(-1.0, im);
            let root = complex_sqrt(z);
            assert!((root * root - z).im.abs() < 1e-313);
            assert!(root.re > 0.0);
        }
    }
    #[test]
    fn complex_division_preserves_extreme_scales() {
        let scales = [-300, -155, -149, -1, 0, 1, 149, 155, 300];
        let a = Complex::new(0.3, -0.2);
        let b = Complex::new(0.4, 0.1);
        for n in scales {
            for d in scales {
                let difference: i32 = n - d;
                if difference.abs() > 300 {
                    continue;
                }
                let expected = (a / b) * 10.0_f64.powi(difference);
                let actual = ratio(a * 10.0_f64.powi(n), b * 10.0_f64.powi(d));
                assert!(
                    (actual - expected).norm() <= 3e-14 * expected.norm(),
                    "{n}/{d}: {actual} != {expected}"
                );
            }
        }
    }
}

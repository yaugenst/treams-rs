//! Numerical support without physics: complex helpers, forward-mode jets, broadcasting,
//! fallible allocation and the parallel policy. treams-rs extension.

pub(crate) mod broadcast;
pub(crate) mod jet;
mod memory;
pub(crate) mod parallel;

pub(crate) use jet::Jet;
pub(crate) use memory::{filled, reserve, zeros};

use crate::Complex;

/// The Euler-Mascheroni constant.
pub(crate) const EULER: f64 = std::f64::consts::EULER_GAMMA;

/// The principal square root of `z`, equal to `z.sqrt()` up to rounding.
///
/// The algebraic form avoids the `atan2`, `sin` and `cos` of num-complex's polar form.
/// Inputs whose larger part lies outside `[1e-150, 1e150]` are scaled first, so the
/// squares neither overflow nor underflow. The sign of a zero imaginary part picks the
/// side of the branch cut on the negative real axis: `complex_sqrt(-1 - 0i) = -i`.
#[inline]
pub(crate) fn complex_sqrt(z: Complex) -> Complex {
    if z.im == 0.0 || !finite(z) {
        return z.sqrt();
    }
    let scale = z.re.abs().max(z.im.abs());
    if !(1e-150..=1e150).contains(&scale) {
        return complex_sqrt(z / scale) * scale.sqrt();
    }
    let norm = z.re.mul_add(z.re, z.im * z.im).sqrt();
    let dominant = norm.midpoint(z.re.abs()).sqrt();
    if z.re >= 0.0 {
        Complex::new(dominant, z.im / (2.0 * dominant))
    } else {
        Complex::new(z.im.abs() / (2.0 * dominant), dominant.copysign(z.im))
    }
}

/// `(-1)^n` as a float, for the sign factors of degrees and orders.
#[inline]
pub(crate) fn parity(n: i32) -> f64 {
    if n % 2 == 0 { 1.0 } else { -1.0 }
}

/// The bits of a real mode label, such as an axial wavenumber, as an exact hash key.
/// Adding zero turns -0.0 into +0.0, so the two zeros, which compare equal, share a key.
pub(crate) fn label_bits(x: f64) -> u64 {
    (x + 0.0).to_bits()
}

/// Whether both parts of `z` are finite.
pub(crate) fn finite(z: Complex) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

/// `numerator / denominator` without overflow or underflow in the denominator's square.
///
/// num-complex divides by `|denominator|^2`, which overflows from about `1e154` and
/// underflows below about `1e-154`. Dividing both operands by the larger part of the
/// denominator first keeps that square near one.
pub(crate) fn ratio(numerator: Complex, denominator: Complex) -> Complex {
    let scale = denominator.re.abs().max(denominator.im.abs());
    (numerator / scale) / (denominator / scale)
}

/// Extreme scales and branch sides of the complex helpers.
#[cfg(test)]
mod tests {
    use super::{complex_sqrt, ratio};
    use crate::Complex;

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

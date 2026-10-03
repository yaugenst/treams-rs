//! Double-double arithmetic: reals `hi + lo` with `|lo|` at most half an ulp of `hi`, and
//! complex numbers with such parts.
//!
//! [`SmallSplitKambe`](super::SmallSplitKambe) recurs the coefficients `c^j / j!` of its
//! series in this arithmetic and rounds them to double at the end. treams-rs extension.

use crate::Complex;

/// A double-double real `hi + lo`, with `|lo|` at most half an ulp of `hi`.
#[derive(Clone, Copy, Debug)]
struct Double {
    hi: f64,
    lo: f64,
}

impl Double {
    /// `hi + lo` as a double-double.
    fn normalized(hi: f64, lo: f64) -> Self {
        let sum = hi + lo;
        Self {
            hi: sum,
            lo: lo - (sum - hi),
        }
    }

    /// The exact product of two doubles.
    fn product(a: f64, b: f64) -> Self {
        let hi = a * b;
        Self {
            hi,
            lo: a.mul_add(b, -hi),
        }
    }

    /// The exact sum of two doubles.
    fn sum(a: f64, b: f64) -> Self {
        let hi = a + b;
        let b_part = hi - a;
        Self {
            hi,
            lo: (a - (hi - b_part)) + (b - b_part),
        }
    }

    /// The quotient `self / divisor`, with one correction step of the double quotient.
    fn divide(self, divisor: Self) -> Self {
        let quotient = self.hi / divisor.hi;
        let remainder = self - Self::from(quotient) * divisor;
        Self::normalized(quotient, remainder.hi / divisor.hi)
    }
}

impl From<f64> for Double {
    fn from(hi: f64) -> Self {
        Self { hi, lo: 0.0 }
    }
}

impl std::ops::Add for Double {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let sum = Self::sum(self.hi, other.hi);
        Self::normalized(sum.hi, sum.lo + self.lo + other.lo)
    }
}

impl std::ops::Neg for Double {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl std::ops::Sub for Double {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + -other
    }
}

impl std::ops::Mul for Double {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let product = Self::product(self.hi, other.hi);
        let lo = self
            .hi
            .mul_add(other.lo, self.lo.mul_add(other.hi, product.lo));
        Self::normalized(product.hi, lo)
    }
}

/// A complex number with double-double parts.
#[derive(Clone, Copy, Debug)]
pub(super) struct DoubleComplex {
    re: Double,
    im: Double,
}

impl DoubleComplex {
    /// `1 / (2 eta^2)` in double-double arithmetic.
    pub(super) fn half_inverse_square(eta: Complex) -> Self {
        let (re, im) = (Double::from(eta.re), Double::from(eta.im));
        // eta^2 = u + i v, and 1 / (2 eta^2) = (u - i v) / (2 (u^2 + v^2)).
        let (u, v) = (re * re - im * im, Double::from(2.0) * re * im);
        let scale = Double::from(2.0) * (u * u + v * v);
        Self {
            re: u.divide(scale),
            im: (-v).divide(scale),
        }
    }

    /// The quotient of each part by a double.
    pub(super) fn divide(self, divisor: f64) -> Self {
        Self {
            re: self.re.divide(Double::from(divisor)),
            im: self.im.divide(Double::from(divisor)),
        }
    }

    /// The nearest double-precision complex number.
    pub(super) fn rounded(self) -> Complex {
        Complex::new(self.re.hi, self.im.hi)
    }
}

impl From<Complex> for DoubleComplex {
    fn from(value: Complex) -> Self {
        Self {
            re: Double::from(value.re),
            im: Double::from(value.im),
        }
    }
}

impl std::ops::Mul for DoubleComplex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

//! First-order forward-mode jets: a value with its derivatives along `N` directions,
//! propagated through numerical kernels by local chain rules. `N = 0` compiles out all
//! derivative storage and arithmetic in forward evaluations. treams-rs extension.
#![allow(clippy::suspicious_arithmetic_impl, clippy::indexing_slicing)] // Product and quotient rules; indices run over the `N` derivative slots.

use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{Complex, numerics::ratio};

/// A complex value and its first derivatives with respect to `N` inputs.
///
/// Kernels that are generic in `N` serve forward evaluation and differentiation alike.
/// With `N = 0` the derivative array is empty, so a forward evaluation stores and
/// computes no derivatives. Operations with a real or complex scalar skip the zero
/// derivative terms of a constant jet. Adding or subtracting a scalar still gives the
/// value a constant jet gives, signed zeros included: the sign of a zero imaginary part
/// selects the side of a branch cut downstream. `special::RadialJet` differs: it holds
/// the first two derivatives in one variable.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Jet<const N: usize> {
    /// The function value.
    pub(crate) value: Complex,
    /// The derivative with respect to each of the `N` inputs.
    pub(crate) derivative: [Complex; N],
}
impl<const N: usize> Jet<N> {
    /// A value that depends on no input: every derivative is zero.
    pub(crate) fn constant(value: impl Into<Complex>) -> Self {
        Self {
            value: value.into(),
            derivative: [Complex::default(); N],
        }
    }
    /// Input number `index` itself: derivative one in that direction, zero in the others.
    /// An `index` of `N` or more gives a constant.
    pub(crate) fn variable(value: impl Into<Complex>, index: usize) -> Self {
        let mut result = Self::constant(value);
        if let Some(d) = result.derivative.get_mut(index) {
            *d = Complex::new(1.0, 0.0);
        }
        result
    }
    /// The chain rule: the jet of `f(self)` from `value = f(x)` and `derivative = f'(x)`
    /// at `x = self.value`. Each derivative of `self` is multiplied by `f'(x)`.
    pub(crate) fn chain(self, value: Complex, derivative: Complex) -> Self {
        Self {
            value,
            derivative: self.derivative.map(|g| g * derivative),
        }
    }
    /// Integer power `self^n`.
    pub(crate) fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::constant(1.0);
        }
        if N == 0 {
            return Self::constant(self.value.powi(n));
        }
        self.chain(self.value.powi(n), f64::from(n) * self.value.powi(n - 1))
    }
    /// Complex exponential.
    pub(crate) fn exp(self) -> Self {
        let value = self.value.exp();
        self.chain(value, value)
    }
    /// Principal square root, with the branch of `numerics::complex_sqrt`.
    pub(crate) fn sqrt(self) -> Self {
        let root = crate::numerics::complex_sqrt(self.value);
        self.chain(root, 0.5 / root)
    }
    /// The largest modulus of the value and the derivatives.
    pub(crate) fn norm(self) -> f64 {
        self.derivative
            .iter()
            .map(|g| g.norm())
            .fold(self.value.norm(), f64::max)
    }
    /// Whether the value and every derivative are finite.
    pub(crate) fn finite(self) -> bool {
        crate::numerics::finite(self.value)
            && self.derivative.iter().all(|&g| crate::numerics::finite(g))
    }
    /// Whether the value and every derivative are zero, without the moduli of `norm`.
    pub(crate) fn is_zero(self) -> bool {
        let zero = Complex::default();
        self.value == zero && self.derivative.iter().all(|&g| g == zero)
    }
}
impl Jet<3> {
    /// The chain rule through three intermediate variables: `self` holds `f` and its
    /// derivatives with respect to `x_b`, `inner[b]` holds `x_b` and its derivatives with
    /// respect to `N` inputs. Returns the value of `f` with the derivatives
    /// `sum_b df/dx_b dx_b/dt_a`, summed in the order `b = 0, 1, 2`.
    pub(crate) fn compose<const N: usize>(self, inner: [Jet<N>; 3]) -> Jet<N> {
        Jet {
            value: self.value,
            derivative: std::array::from_fn(|a| {
                (0..3)
                    .map(|b| self.derivative[b] * inner[b].derivative[a])
                    .sum()
            }),
        }
    }
}
impl<const N: usize> Default for Jet<N> {
    fn default() -> Self {
        Self::constant(0.0)
    }
}
impl<const N: usize> Add for Jet<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            derivative: std::array::from_fn(|i| self.derivative[i] + rhs.derivative[i]),
        }
    }
}
impl<const N: usize> Sub for Jet<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            derivative: std::array::from_fn(|i| self.derivative[i] - rhs.derivative[i]),
        }
    }
}
impl<const N: usize> Mul for Jet<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            derivative: std::array::from_fn(|i| {
                self.derivative[i] * rhs.value + self.value * rhs.derivative[i]
            }),
        }
    }
}
impl<const N: usize> Div for Jet<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let value = ratio(self.value, rhs.value);
        Self {
            value,
            derivative: std::array::from_fn(|i| {
                ratio(self.derivative[i] - value * rhs.derivative[i], rhs.value)
            }),
        }
    }
}
impl<const N: usize> Neg for Jet<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            derivative: self.derivative.map(|g| -g),
        }
    }
}
impl<const N: usize> Sum for Jet<N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), Add::add)
    }
}
// A scalar acts as a constant jet without evaluating its zero derivative terms. Values
// combine with the scalar as a complex number, exactly as with a constant jet:
// num-complex's real-scalar addition keeps a -0.0 imaginary part that a constant's
// +0.0 would turn into +0.0, and that sign selects branch sides downstream.
macro_rules! scalar_ops {
    ($scalar:ty) => {
        impl<const N: usize> Add<$scalar> for Jet<N> {
            type Output = Self;
            fn add(self, rhs: $scalar) -> Self {
                Self {
                    value: self.value + Complex::from(rhs),
                    derivative: self.derivative,
                }
            }
        }
        impl<const N: usize> Add<Jet<N>> for $scalar {
            type Output = Jet<N>;
            fn add(self, rhs: Jet<N>) -> Jet<N> {
                Jet {
                    value: Complex::from(self) + rhs.value,
                    derivative: rhs.derivative,
                }
            }
        }
        impl<const N: usize> Sub<$scalar> for Jet<N> {
            type Output = Self;
            fn sub(self, rhs: $scalar) -> Self {
                Self {
                    value: self.value - Complex::from(rhs),
                    derivative: self.derivative,
                }
            }
        }
        impl<const N: usize> Sub<Jet<N>> for $scalar {
            type Output = Jet<N>;
            fn sub(self, rhs: Jet<N>) -> Jet<N> {
                Jet {
                    value: Complex::from(self) - rhs.value,
                    derivative: rhs.derivative.map(|g| -g),
                }
            }
        }
        impl<const N: usize> Mul<$scalar> for Jet<N> {
            type Output = Self;
            fn mul(self, rhs: $scalar) -> Self {
                Self {
                    value: self.value * rhs,
                    derivative: self.derivative.map(|g| g * rhs),
                }
            }
        }
        impl<const N: usize> Mul<Jet<N>> for $scalar {
            type Output = Jet<N>;
            fn mul(self, rhs: Jet<N>) -> Jet<N> {
                rhs * self
            }
        }
        impl<const N: usize> Div<Jet<N>> for $scalar {
            type Output = Jet<N>;
            fn div(self, rhs: Jet<N>) -> Jet<N> {
                Jet::constant(self) / rhs
            }
        }
    };
}
macro_rules! assignments {
    ($trait:ident, $method:ident, $optrait:ident, $method2:ident) => {
        impl<const N: usize, Rhs> $trait<Rhs> for Jet<N>
        where
            Self: $optrait<Rhs, Output = Self>,
        {
            fn $method(&mut self, rhs: Rhs) {
                *self = self.$method2(rhs);
            }
        }
    };
}
scalar_ops!(f64);
scalar_ops!(Complex);
// Real scaling must stay a real division: complex division can square a
// subnormal denominator to zero when normalizing a nearly axial wavevector.
impl<const N: usize> Div<f64> for Jet<N> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self {
            value: self.value / rhs,
            derivative: self.derivative.map(|g| g / rhs),
        }
    }
}
impl<const N: usize> Div<Complex> for Jet<N> {
    type Output = Self;
    fn div(self, rhs: Complex) -> Self {
        Self {
            value: ratio(self.value, rhs),
            derivative: self.derivative.map(|g| ratio(g, rhs)),
        }
    }
}
assignments!(AddAssign, add_assign, Add, add);
assignments!(SubAssign, sub_assign, Sub, sub);
assignments!(MulAssign, mul_assign, Mul, mul);

/// Algebraic identities of jet arithmetic and the chain rule against closed forms.
#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::Jet;
    use crate::{
        Complex,
        numerics::complex_sqrt,
        test_support::{ALGEBRA_CASES, DEFAULT_CASES, complex, log_polar, prop_assert_close},
    };

    /// Two-direction jets with value modulus in [0.1, 10] and derivatives in [-5, 5]^2.
    fn jet() -> impl Strategy<Value = Jet<2>> {
        (log_polar(-1.0..1.0, -PI..PI), complex(5.0), complex(5.0)).prop_map(|(value, a, b)| Jet {
            value,
            derivative: [a, b],
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn scalar_operations_equal_constant_jets(
            a in jet(), b in jet(), re in -100.0_f64..100.0, im in -100.0_f64..100.0,
        ) {
            check_scalar_operations(a, b, re, Complex::new(re, im))?;
        }

        #[test]
        fn jet_operations_invert(a in jet(), b in jet(), n in -4_i32..=6) {
            check_inverses(a, b, n)?;
        }

        #[test]
        fn chain_rule_matches_closed_form(
            z in log_polar(-0.52..0.48, -PI..PI), w in complex(2.0), n in -4_i32..=4,
        ) {
            check_chain_rule(z, w, n)?;
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

        #[test]
        fn local_quotient_common_scale(
            x in -0.3_f64..0.3, exponent in prop_oneof![Just(-300), Just(0), Just(300)],
        ) {
            check_local_quotient_common_scale(Complex::new(0.7, x), 10_f64.powi(exponent))?;
        }
    }

    fn assert_equal(
        actual: Jet<2>,
        expected: Jet<2>,
        operation: &str,
    ) -> Result<(), TestCaseError> {
        prop_assert_eq!(actual.value, expected.value, "{}", operation);
        prop_assert_eq!(actual.derivative, expected.derivative, "{}", operation);
        Ok(())
    }

    /// Real and complex scalars act as constant jets, from either side, in the binary
    /// and assignment forms; sums and linear combinations are exact.
    fn check_scalar_operations(
        a: Jet<2>,
        b: Jet<2>,
        real: f64,
        scalar: Complex,
    ) -> Result<(), TestCaseError> {
        let (r, c) = (Jet::constant(real), Jet::constant(scalar));
        for (actual, expected, operation) in [
            (a + real, a + r, "jet + real"),
            (real + a, r + a, "real + jet"),
            (a - real, a - r, "jet - real"),
            (real - a, r - a, "real - jet"),
            (a * real, a * r, "jet * real"),
            (real * a, r * a, "real * jet"),
            (a / real, a / r, "jet / real"),
            (a + scalar, a + c, "jet + complex"),
            (scalar + a, c + a, "complex + jet"),
            (a - scalar, a - c, "jet - complex"),
            (scalar - a, c - a, "complex - jet"),
            (a * scalar, a * c, "jet * complex"),
            (scalar * a, c * a, "complex * jet"),
            (a / scalar, a / c, "jet / complex"),
            ([a, b].into_iter().sum(), a + b, "sum"),
        ] {
            assert_equal(actual, expected, operation)?;
        }
        let mut assigned = a;
        assigned *= real;
        assigned += scalar;
        assigned -= b;
        assert_equal(assigned, a * real + scalar - b, "assignments")?;
        let x = Jet::<2>::variable(a.value, 0);
        let y = Jet::<2>::variable(b.value, 1);
        let combination = scalar * x + y;
        prop_assert_eq!(combination.derivative, [scalar, Complex::new(1.0, 0.0)]);
        Ok(())
    }

    /// Division undoes multiplication, the square root squares back off the branch cut,
    /// `exp(a) exp(-a)` is the constant 1, and integer powers are repeated products.
    fn check_inverses(a: Jet<2>, b: Jet<2>, n: i32) -> Result<(), TestCaseError> {
        let one = Jet::<2>::constant(1.0);
        // Rounding in the quotient rule grows with |b'| / |b| relative to a.
        let scale = a.norm() * (1.0 + b.norm() / b.value.norm());
        prop_assert_close!(((a * b) / b).value, a.value, 1e-14 * scale);
        prop_assert_close!(((a * b) / b).derivative, a.derivative, 1e-14 * scale);
        let root = if a.value.re > 0.0 { a } else { -a }.sqrt();
        let square = root * root;
        let expected = if a.value.re > 0.0 { a } else { -a };
        prop_assert_close!(square.value, expected.value, 1e-14 * a.norm());
        prop_assert_close!(square.derivative, expected.derivative, 1e-14 * a.norm());
        let product = a.exp() * (-a).exp();
        prop_assert_close!(product.value, one.value, 1e-14);
        prop_assert_close!(product.derivative, one.derivative, 1e-14 * a.norm());
        let repeated = (0..n.abs()).fold(one, |power, _| power * a);
        let expected = if n < 0 { 1.0 / repeated } else { repeated };
        let power = a.powi(n);
        let scale = expected.norm() * f64::from(1 + n.abs());
        prop_assert_close!(power.value, expected.value, 1e-14 * scale, "n = {}", n);
        prop_assert_close!(
            power.derivative,
            expected.derivative,
            1e-14 * scale,
            "n = {}",
            n
        );
        Ok(())
    }

    /// The derivatives of `sqrt(x^n e^x / (1 + x^2)) - 3/x + y x` in both variables equal
    /// their closed forms; the square root's branch follows `complex_sqrt`.
    fn check_chain_rule(z: Complex, w: Complex, n: i32) -> Result<(), TestCaseError> {
        let i = Complex::i();
        prop_assume!((z - i).norm() > 0.2 && (z + i).norm() > 0.2);
        let (x, y) = (Jet::<2>::variable(z, 0), Jet::<2>::variable(w, 1));
        let f = ((x.powi(n) * x.exp()) / (1.0 + x * x)).sqrt() - 3.0 / x + y * x;
        let g = z.powi(n) * z.exp() / (1.0 + z * z);
        let root = complex_sqrt(g);
        let dg = g * (f64::from(n) / z + 1.0 - 2.0 * z / (1.0 + z * z));
        let expected = [dg / (2.0 * root) + 3.0 / (z * z) + w, z];
        let scale = (dg / root).norm() + 3.0 / z.norm_sqr() + w.norm() + z.norm();
        prop_assert_close!(f.value, root - 3.0 / z + w * z, 1e-13 * scale);
        prop_assert_close!(f.derivative, expected, 1e-13 * scale);
        Ok(())
    }

    /// The quotient rule is exact when numerator and denominator share an extreme scale.
    fn check_local_quotient_common_scale(a: Complex, scale: f64) -> Result<(), TestCaseError> {
        let b = Complex::new(1.3, 0.4);
        let result = Jet::<2>::variable(a * scale, 0) / Jet::<2>::variable(b * scale, 1);
        prop_assert_close!(result.value, a / b, 1e-14);
        prop_assert_close!(
            result.derivative[0] * scale,
            Complex::new(1.0, 0.0) / b,
            1e-14
        );
        prop_assert_close!(result.derivative[1] * scale, -a / b / b, 1e-14);
        Ok(())
    }
}

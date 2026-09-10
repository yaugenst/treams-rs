//! Fixed-size local first-order chain rules for numerical kernels.
//! N=0 compiles out all derivative storage and arithmetic in forward evaluations.
#![allow(clippy::suspicious_arithmetic_impl, clippy::indexing_slicing)] // Product and quotient rules.

use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::Complex;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Jet<const N: usize> {
    pub(crate) value: Complex,
    pub(crate) derivative: [Complex; N],
}
impl<const N: usize> Jet<N> {
    pub(crate) fn constant(value: impl Into<Complex>) -> Self {
        Self {
            value: value.into(),
            derivative: [Complex::default(); N],
        }
    }
    pub(crate) fn variable(value: impl Into<Complex>, index: usize) -> Self {
        let mut result = Self::constant(value);
        if let Some(d) = result.derivative.get_mut(index) {
            *d = Complex::new(1.0, 0.0);
        }
        result
    }
    pub(crate) fn map(self, value: Complex, derivative: Complex) -> Self {
        Self {
            value,
            derivative: self.derivative.map(|g| g * derivative),
        }
    }
    pub(crate) fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::constant(1.0);
        }
        self.map(self.value.powi(n), f64::from(n) * self.value.powi(n - 1))
    }
    pub(crate) fn exp(self) -> Self {
        self.map(self.value.exp(), self.value.exp())
    }
    pub(crate) fn sqrt(self) -> Self {
        self.map(self.value.sqrt(), 0.5 / self.value.sqrt())
    }
    pub(crate) fn norm(self) -> f64 {
        self.derivative
            .iter()
            .map(|g| g.norm())
            .fold(self.value.norm(), f64::max)
    }
    pub(crate) fn finite(self) -> bool {
        crate::finite(self.value) && self.derivative.iter().all(|&g| crate::finite(g))
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
        Self {
            value: self.value / rhs.value,
            derivative: std::array::from_fn(|i| {
                (self.derivative[i] - self.value / rhs.value * rhs.derivative[i]) / rhs.value
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
macro_rules! scalar_ops {
    ($trait:ident, $method:ident, $op:tt, $scalar:ty) => {
        impl<const N: usize> $trait<$scalar> for Jet<N> {
            type Output = Self;
            fn $method(self, rhs: $scalar) -> Self { self $op Self::constant(rhs) }
        }
        impl<const N: usize> $trait<Jet<N>> for $scalar {
            type Output = Jet<N>;
            fn $method(self, rhs: Jet<N>) -> Jet<N> { Jet::constant(self) $op rhs }
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
scalar_ops!(Add, add, +, f64);
scalar_ops!(Sub, sub, -, f64);
scalar_ops!(Mul, mul, *, f64);
scalar_ops!(Div, div, /, f64);
scalar_ops!(Add, add, +, Complex);
scalar_ops!(Sub, sub, -, Complex);
scalar_ops!(Mul, mul, *, Complex);
scalar_ops!(Div, div, /, Complex);
assignments!(AddAssign, add_assign, Add, add);
assignments!(SubAssign, sub_assign, Sub, sub);
assignments!(MulAssign, mul_assign, Mul, mul);

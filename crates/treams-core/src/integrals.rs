//! Incomplete gamma and Kambe integrals used by the Ewald lattice sums.
//! Recurrences follow DLMF 8.8 and Kambe (1968), as in treams.
#![allow(clippy::float_cmp, clippy::while_float)] // Half-integer degrees step exactly by one.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use errorfunctions::ComplexErrorFunctions;

use crate::{Complex, Error, Result, finite};

const EULER: f64 = 0.577_215_664_901_532_9;

// DLMF 8.9.2, evaluated by modified Lentz iteration. Unlike downward recurrence,
// this retains relative accuracy for negative degree and large positive argument.
fn gamma_fraction(n: f64, z: Complex) -> Complex {
    let mut b = z + 1.0 - n;
    let mut c = Complex::new(1e300, 0.0);
    let mut d = 1.0 / b;
    let mut h = d;
    for j in 1..=1000 {
        let j = f64::from(j);
        let a = -j * (j - n);
        b += 2.0;
        d = b + a * d;
        c = b + a / c;
        if d.norm() < 1e-300 {
            d = Complex::new(1e-300, 0.0);
        }
        if c.norm() < 1e-300 {
            c = Complex::new(1e-300, 0.0);
        }
        d = 1.0 / d;
        let delta = c * d;
        h *= delta;
        if (delta - 1.0).norm() < 3e-15 {
            break;
        }
    }
    (n * z.ln() - z).exp() * h
}

fn exp1(z: Complex) -> Complex {
    if z.norm() >= 5.0 && !(z.re < -2.0 * z.im.abs() && z.norm() < 40.0) {
        let mut result = gamma_fraction(0.0, z);
        if z.re < 0.0 && z.im == 0.0 {
            result.im = -PI.copysign(z.im);
        }
        return result;
    }
    // E1(z) = -gamma - log(z) - sum (-z)^j/(j j!).
    let mut term = -z;
    let mut sum = term;
    for j in 2..=500 {
        let j = f64::from(j);
        term *= -z / j;
        let change = term / j;
        sum += change;
        if change.norm() <= 2e-16 * sum.norm() {
            break;
        }
    }
    -EULER - z.ln() - sum
}

/// Upper incomplete gamma, with integer or half-integer `n` and a signed branch cut.
pub fn incgamma(n: f64, z: Complex) -> Result<Complex> {
    if !n.is_finite() || (2.0 * n).fract() != 0.0 || n.abs() > 128.0 || !finite(z) {
        return Err(Error::InvalidInput(
            "gamma degree must be an integer or half-integer in [-128, 128], and z finite".into(),
        ));
    }
    Ok(gamma(n, z))
}

pub(crate) fn gamma(n: f64, z: Complex) -> Complex {
    if z == Complex::default() {
        return Complex::new(
            if n <= 0.0 {
                f64::INFINITY
            } else {
                libm::tgamma(n)
            },
            0.0,
        );
    }
    if z.re > 4.0 && z.norm() > n.abs() + 2.0 {
        return gamma_fraction(n, z);
    }
    let base = if n.fract() == 0.0 { 0.0 } else { 0.5 };
    let mut value = if base == 0.0 {
        exp1(z)
    } else {
        PI.sqrt() * z.sqrt().erfc()
    };
    let exponential = (-z).exp();
    let mut degree = base;
    while degree < n {
        value = degree * value + z.powf(degree) * exponential;
        degree += 1.0;
    }
    while degree > n {
        degree -= 1.0;
        value = (value - z.powf(degree) * exponential) / degree;
    }
    value
}

fn odd_base(n: i32, z: Complex, eta: Complex) -> Complex {
    let mult = 0.25 * z * z;
    let argument = 0.5 * z * z * eta * eta;
    let mut coefficient = if n == -3 {
        mult
    } else {
        Complex::new(0.5, 0.0)
    };
    let mut sum = Complex::default();
    for j in 0..200 {
        let degree = if n == -3 { -1 - j } else { -j };
        let term = coefficient * gamma(f64::from(degree), argument);
        sum += term;
        if j > 3 && term.norm() <= 2e-16 * sum.norm() {
            break;
        }
        coefficient *= mult / f64::from(j + 1);
    }
    sum
}

fn even_base(n: i32, z: Complex, eta: Complex) -> Complex {
    let z = if z.re < 0.0 { -z } else { z };
    let plus = ((z * eta - Complex::i() / eta) * FRAC_1_SQRT_2).erfc() * (-Complex::i() * z).exp();
    let minus = ((z * eta + Complex::i() / eta) * FRAC_1_SQRT_2).erfc() * (Complex::i() * z).exp();
    let scale = 0.5 * PI.sqrt() * FRAC_1_SQRT_2;
    if n == -2 {
        -Complex::i() * scale * (plus - minus)
    } else {
        scale * (plus + minus) / z
    }
}

/// `I_n(z, eta) = integral_eta^infinity t^n exp(-z^2 t^2/2 + 1/(2t^2)) dt`.
pub fn intkambe(n: i32, z: Complex, eta: Complex) -> Result<Complex> {
    if !(-260..=260).contains(&n) || !finite(z) || !finite(eta) {
        return Err(Error::InvalidInput(
            "Kambe order must be in [-260, 260] and arguments finite".into(),
        ));
    }
    Ok(kambe(n, z, eta))
}

pub(crate) fn kambe(n: i32, z: Complex, eta: Complex) -> Complex {
    if eta == Complex::default() || (z == Complex::default() && n > -2) {
        return Complex::new(f64::INFINITY, 0.0);
    }
    let exponential = (0.5 * (1.0 / (eta * eta) - z * z * eta * eta)).exp();
    let odd = n % 2 != 0;
    let base = if odd { -3 } else { -2 };
    let mut lower = if odd {
        if z == Complex::default() {
            exponential - 1.0
        } else {
            odd_base(-3, z, eta)
        }
    } else {
        even_base(-2, z, eta)
    };
    if n == base {
        return lower;
    }
    if n < base {
        let mut upper = if z == Complex::default() {
            Complex::default()
        } else if odd {
            odd_base(-1, z, eta)
        } else {
            even_base(0, z, eta)
        };
        let mut order = base - 2;
        while order >= n {
            let value =
                f64::from(order + 3) * lower - z * z * upper + eta.powi(order + 3) * exponential;
            upper = lower;
            lower = value;
            order -= 2;
        }
        return lower;
    }
    let mut upper = if odd {
        odd_base(-1, z, eta)
    } else {
        even_base(0, z, eta)
    };
    let mut order = base + 4;
    while order <= n {
        let value =
            (f64::from(order - 1) * upper - lower + eta.powi(order - 1) * exponential) / (z * z);
        lower = upper;
        upper = value;
        order += 2;
    }
    upper
}

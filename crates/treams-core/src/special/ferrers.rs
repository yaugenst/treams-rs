//! Real-degree Ferrers functions: degree recurrence and convergent endpoint series.
//! Adapted from scipy/xsf `specfun::lpmv/lpmv0` (BSD-3-Clause; see LICENSE.xsf).
//!
//! Upstream: `treams.special.lpmv` at a non-integer degree and a real argument, which
//! scipy evaluates with the same xsf routines.
//!
//! A Ferrers function is the associated Legendre function `P_v^m(x)` on `-1 < x <= 1`.
//! The series about `x = 1` is a hypergeometric series in `(1 - x) / 2`; the series
//! about `x = -1` adds the logarithmic terms of the singular endpoint. [`ENDPOINT_SPLIT`]
//! chooses between them, and recurrences in the order or the degree carry larger labels.
#![allow(clippy::float_cmp, clippy::cast_possible_truncation)] // Validated bounded degrees and exact endpoints.
use crate::{Error, MAX_DEGREE, Result, numerics::EULER};
use std::f64::consts::PI;

/// The argument that splits `[-1, 1]` between the two endpoint series: arguments below
/// it take the series about the nearer endpoint `x = -1`, the others the series about
/// `x = 1`.
const ENDPOINT_SPLIT: f64 = -0.35;

/// The most terms either endpoint series sums.
const MAX_SERIES_TERMS: i32 = 200;

/// The series about `x = 1` sums more than this many terms before its stopping test
/// applies.
const MIN_SERIES_TERMS: i32 = 12;

/// An endpoint series stops once a term falls below this fraction of the sum, about one
/// ulp.
const SERIES_TOLERANCE: f64 = 2e-16;

/// The digamma function `psi(x)` for `x > 0`: the recurrence `psi(x) = psi(x + 1) - 1/x`
/// shifts `x` to at least 10, where the asymptotic series
/// `ln x - 1/(2x) - sum_k B_2k / (2k x^2k)` with the Bernoulli numbers `B_2` to `B_16`
/// reaches double precision. The coefficients run from `k = 8` (`3617/8160`) down to
/// `k = 1` (`1/12`), for Horner's rule in `1/x^2`.
#[allow(clippy::while_float)] // At most ten exact unit increments into the asymptotic range.
fn digamma(mut x: f64) -> f64 {
    let mut correction = 0.0;
    while x < 10.0 {
        correction -= 1.0 / x;
        x += 1.0;
    }
    let inverse2 = 1.0 / (x * x);
    let series = [
        0.443_259_803_921_568_6,
        -1.0 / 12.0,
        691.0 / 32760.0,
        -1.0 / 132.0,
        1.0 / 240.0,
        -1.0 / 252.0,
        1.0 / 120.0,
        -1.0 / 12.0,
    ]
    .into_iter()
    .fold(0.0, |sum, a| sum * inverse2 + a);
    x.ln() - 0.5 / x + inverse2 * series + correction
}

/// `P_v^m(x)` from the endpoint series, scaled by `sin(theta)^(-|m|)` near `x = 1` and by
/// `sin(theta)^|m|` near `x = -1`. The prefactor combines its factorial factors one
/// order at a time, so no intermediate overflows.
fn base_factor(v: f64, order: i32, x: f64) -> f64 {
    let m = order.abs();
    let weight = if x < ENDPOINT_SPLIT {
        (1.0 - x) * (1.0 + x)
    } else {
        1.0
    };
    let prefactor = (0..m).fold(1.0, |p, j| {
        let j = f64::from(j);
        if order < 0 {
            -p * weight / (2.0 * (j + 1.0))
        } else {
            p * (v - j) * (v + j + 1.0) * weight / (2.0 * (j + 1.0))
        }
    });
    let mf = f64::from(m);
    if x >= ENDPOINT_SPLIT {
        let mut term = 1.0;
        let mut sum = 1.0;
        for k in 1..=MAX_SERIES_TERMS {
            let kf = f64::from(k);
            term *= 0.5 * (-v + mf + kf - 1.0) * (v + mf + kf) * (1.0 - x) / (kf * (mf + kf));
            sum += term;
            if k > MIN_SERIES_TERMS && term.abs() <= SERIES_TOLERANCE * sum.abs() {
                break;
            }
        }
        return if m % 2 == 0 {
            prefactor * sum
        } else {
            -prefactor * sum
        };
    }
    // Analytic continuation in (1+x)/2. The regular and logarithmic terms
    // converge rapidly near the singular endpoint instead of summing at t~1.
    let reduced = v - v.round();
    let sine = (reduced * PI).sin() / PI
        * if (v.round() as i32) % 2 == 0 {
            1.0
        } else {
            -1.0
        };
    let singular = if m == 0 {
        0.0
    } else {
        let scale = (1..=m).fold(1.0, |p, j| {
            let j = f64::from(j);
            let factor = if order < 0 {
                -j / ((v - j + 1.0) * (v + j))
            } else {
                j
            };
            p * factor * (1.0 - x)
        });
        let mut term = 1.0;
        let mut sum = 1.0;
        for k in 1..m {
            let k = f64::from(k);
            term *= 0.5 * (-v + k - 1.0) * (v + k) * (1.0 + x) / (k * (k - mf));
            sum += term;
        }
        -sine * scale * sum / mf
    };
    let pa = 2.0 * (digamma(v) + EULER) + PI / (PI * reduced).tan() + 1.0 / v;
    let harmonic = |q: f64| (q * q + v * v) / (q * (q - v) * (q + v));
    let log = (0.5 * (1.0 + x)).ln();
    let mut sum = pa + (1..=m).map(|j| harmonic(f64::from(j))).sum::<f64>() - 1.0 / (mf - v) + log;
    let mut term = 1.0;
    let mut cumulative = 0.0;
    for k in 1..=MAX_SERIES_TERMS {
        let kf = f64::from(k);
        term *= 0.5 * (-v + mf + kf - 1.0) * (v + mf + kf) * (1.0 + x) / (kf * (kf + mf));
        cumulative += 1.0 / (kf * (kf - v) * (kf + v));
        let h = pa
            + (1..=m).map(|j| harmonic(f64::from(k + j))).sum::<f64>()
            + 2.0 * v * v * cumulative
            - 1.0 / (mf + kf - v)
            + log;
        let change = h * term;
        sum += change;
        if change.abs() <= SERIES_TOLERANCE * sum.abs() {
            break;
        }
    }
    singular + sum * sine * prefactor
}

/// The scaled `P_v^m(x)` of [`base_factor`]. Near `x = -1` large orders recur upward
/// in the order from orders 0 and ±1, unless the degree is within 1e-8 of an integer;
/// elsewhere large degrees recur upward in the degree (DLMF 14.10.3) from two
/// [`base_factor`] values of degrees `frac(v) + |m|` and `frac(v) + |m| + 1`.
fn regular_factor(v: f64, m: i32, x: f64) -> f64 {
    let absolute = m.abs();
    if x < ENDPOINT_SPLIT
        && absolute >= 2
        && v > f64::from(absolute + 1)
        && (v - v.round()).abs() > 1e-8
    {
        let mut previous = regular_factor(v, 0, x);
        let mut current = regular_factor(v, m.signum(), x);
        for order in 1..absolute {
            let j = f64::from(order);
            let next = if m > 0 {
                -2.0 * j * x * current - (v + j) * (v - j + 1.0) * (1.0 - x) * (1.0 + x) * previous
            } else {
                (2.0 * j * x * current - (1.0 - x) * (1.0 + x) * previous)
                    / ((v - j) * (v + j + 1.0))
            };
            previous = current;
            current = next;
        }
        return current;
    }
    if x < ENDPOINT_SPLIT && (v + f64::from(m.abs()) + 1.0).powi(2) * (1.0 + x) <= 4.0 {
        return base_factor(v, m, x);
    }
    let integer = v as i32;
    if integer <= 2 || integer <= m.abs() {
        return base_factor(v, m, x);
    }
    let fraction = v - f64::from(integer);
    let mut lower = base_factor(fraction + f64::from(m.abs()), m, x);
    let mut upper = base_factor(fraction + f64::from(m.abs() + 1), m, x);
    for j in m.abs() + 2..=integer {
        let degree = fraction + f64::from(j);
        let next = ((2.0 * degree - 1.0) * x * upper - (degree - 1.0 + f64::from(m)) * lower)
            / (degree - f64::from(m));
        lower = upper;
        upper = next;
    }
    upper
}

/// `value root^exponent`, multiplied factor by factor where the power alone would
/// underflow or overflow.
fn scaled_power(value: f64, root: f64, exponent: i32) -> f64 {
    let multiplier = if exponent < 0 { 1.0 / root } else { root };
    let power = multiplier.powi(exponent.abs());
    if value == 0.0 {
        return 0.0;
    }
    if power.is_normal() {
        return value * power;
    }
    (0..exponent.abs()).fold(value, |v, _| v * multiplier)
}

/// The largest `|m|` of [`ferrers_real_degree`], two above its largest degree. Orders
/// above the degree are valid; the angular functions pass `|m| <= degree`.
const MAX_REAL_DEGREE_ORDER: i32 = MAX_DEGREE + 2;

/// Real-degree Ferrers function `P_v^m(x)` (degree `v` not an integer) and, with
/// `DERIVATIVE`, its analytic derivative in `x`.
///
/// Upstream: `treams.special.lpmv(m, v, x)` at a non-integer degree. Differences:
/// degrees above 128, orders above 130 and `x` outside `(-1, 1]` give an error.
pub(crate) fn ferrers_real_degree<const DERIVATIVE: bool>(
    v: f64,
    m: i32,
    x: f64,
) -> Result<(f64, f64)> {
    if !(0.0..=f64::from(MAX_DEGREE)).contains(&v)
        || v.fract() == 0.0
        || m.unsigned_abs() > MAX_REAL_DEGREE_ORDER.unsigned_abs()
        || !(-1.0..=1.0).contains(&x)
        || x == -1.0
    {
        return Err(Error::InvalidInput(
            // 128 is MAX_DEGREE and 130 is MAX_REAL_DEGREE_ORDER.
            "noninteger real degree in (0,128], |order|<=130 and -1<x<=1 required".into(),
        ));
    }
    if v < 1e-150 && m == 0 {
        return Ok((1.0, if DERIVATIVE { v / (1.0 + x) } else { 0.0 }));
    }
    let absolute = m.abs();
    let w = (1.0 - x) * (1.0 + x);
    let root = w.sqrt();
    let regular = regular_factor(v, m, x);
    let exponent = if x < ENDPOINT_SPLIT {
        -absolute
    } else {
        absolute
    };
    let value = if x == 1.0 && absolute > 0 {
        0.0
    } else {
        scaled_power(regular, root, exponent)
    };
    let derivative = if DERIVATIVE {
        let adjacent = if m < 0 {
            (v + f64::from(m)) * (v - f64::from(m) + 1.0) * regular_factor(v, m - 1, x)
        } else {
            -regular_factor(v, m + 1, x)
        };
        if x < ENDPOINT_SPLIT {
            scaled_power(
                adjacent - f64::from(absolute) * x * regular,
                root,
                -absolute - 2,
            )
        } else if absolute == 0 {
            adjacent
        } else if x == 1.0 {
            match absolute {
                1 => f64::NAN,
                2 => -2.0 * regular,
                _ => 0.0,
            }
        } else {
            scaled_power(
                w * adjacent - f64::from(absolute) * x * regular,
                root,
                absolute - 2,
            )
        }
    } else {
        0.0
    };
    if !value.is_finite() || !derivative.is_finite() {
        return Err(Error::SpecialFunction(
            "non-finite real-degree Legendre result or derivative".into(),
        ));
    }
    Ok((value, derivative))
}

//! The upper incomplete gamma function `Gamma(n, z)` of integer and half-integer degree.
//!
//! [`upper_gamma`] takes whichever of the continued fraction, power series, Kummer sum and
//! degree recurrence (DLMF 8) keeps relative accuracy at the argument.
//! [`ScaledGammaLadder`] steps through the scaled values `Gamma(d, a) / a^d` at
//! descending degrees, which the Kambe series sums.
//!
//! Upstream: `incgamma` of `treams/special/_integrals.pyx`, which runs the degree
//! recurrence only. The choice of method and the scaled ladder are treams-rs extensions.

use std::f64::consts::PI;

use errorfunctions::ComplexErrorFunctions;

use crate::{
    Complex,
    numerics::{EULER, finite},
};

/// `Gamma(n, z)` from DLMF 8.9.2 by modified Lentz iteration, which keeps relative
/// accuracy for negative degree and large `|z|` away from the negative real axis, and
/// whether it converged.
fn gamma_fraction(n: f64, z: Complex) -> (Complex, bool) {
    let (fraction, converged) = continued_fraction(n, z);
    ((n * z.ln() - z).exp() * fraction, converged)
}

/// The continued fraction `h` of DLMF 8.9.2, `Gamma(n, z) = z^n e^-z h` on the
/// principal branch, with whether it converged.
fn continued_fraction(n: f64, z: Complex) -> (Complex, bool) {
    let mut b = z + 1.0 - n;
    let mut c = Complex::new(1e300, 0.0);
    let mut d = 1.0 / b;
    let mut h = d;
    let mut converged = false;
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
            converged = true;
            break;
        }
    }
    (h, converged)
}

/// The exponential integral `E1(z) = Gamma(0, z)`, the start of the integer-degree
/// recurrence. The power series cancels by about `e^(|z| + Re z)`, while the continued
/// fraction converges slowly next to the negative real axis; beyond `|z| = 40` the
/// fraction converges there too.
fn exp1(z: Complex) -> Complex {
    if z.norm() + z.re > 4.0 || z.norm() >= 40.0 {
        let (mut result, _) = gamma_fraction(0.0, z);
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

/// Largest growth of the relative rounding error accepted from one method without
/// trying another (one digit).
pub(super) const RECURRENCE_GROWTH: f64 = 10.0;
/// Bound on `|z| + Re z` of the parabola around the negative real axis inside which
/// the power series terms share a phase: their cancellation stays below `e`.
const SERIES_PARABOLA: f64 = 1.0;
/// Largest `|z|` at which [`gamma_series`] is used. It needs about `2|z|` terms of size
/// up to `e^|z|`; with the power `z^n` split in two factors every intermediate stays
/// representable for `|n| <=` [`MAX_DEGREE`](crate::MAX_DEGREE).
const GAMMA_SERIES_RADIUS: f64 = 600.0;

/// The upper incomplete gamma function `Gamma(n, z)` of integer or half-integer `n`.
///
/// It takes the continued fraction where `Re z > 4` and `|z| > |n| + 2`; the power
/// series or the continued fraction where a downward recurrence would lose accuracy;
/// otherwise the degree recurrence from `E1` or `erfc`, or the Kummer sum or the power
/// series where their error bounds are smaller.
pub(crate) fn upper_gamma(n: f64, z: Complex) -> Complex {
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
    let modulus = z.norm();
    if z.re > 4.0 && modulus > n.abs() + 2.0 {
        return gamma_fraction(n, z).0;
    }
    if n == 2.0 {
        // DLMF 8.4.8: factor before multiplying by exp(-z). Adding exp(-z)
        // and z*exp(-z) instead loses relative accuracy at the zero z = -1.
        return (1.0 + z) * (-z).exp();
    }
    let base = if n.fract() == 0.0 { 0.0 } else { 0.5 };
    let series = modulus <= GAMMA_SERIES_RADIUS;
    // A downward step subtracts the leading asymptotic term z^d e^-z, losing a
    // factor |z| / |d| when |z| exceeds the degree. There the series (near the
    // cut) or the continued fraction (elsewhere) keeps full relative accuracy.
    if downward_unstable(n, base, modulus) {
        if series && modulus + z.re <= SERIES_PARABOLA {
            return gamma_series(n, z).0;
        }
        if let (value, true) = gamma_fraction(n, z) {
            return value;
        }
    }
    // An upward step cancels where Gamma(n, z) falls far below the terms z^d e^-z it
    // sums, by up to about e^(|z| - Re z): at large positive degree away from the
    // positive real axis. Two sums of the lower function gamma(n, z) = Gamma(n) -
    // Gamma(n, z) avoid that loss, and the smallest error bound wins. For |z| < n the
    // Kummer form has falling terms at any phase of z. The power series has terms of
    // one phase near the negative real axis; elsewhere they cancel by about
    // e^(|z| + Re z), and it is only tried when that is the smaller loss.
    let (mut value, mut error) = gamma_recurrence(n, z, base, modulus);
    // |value| is at least its larger part: most values skip the `hypot`.
    let larger = value.re.abs().max(value.im.abs());
    if error > RECURRENCE_GROWTH * larger && error > RECURRENCE_GROWTH * value.norm() {
        if n > 0.0 && modulus < n {
            let kummer = gamma_kummer(n, z);
            if kummer.1 < error {
                (value, error) = kummer;
            }
        }
        if series && (modulus + z.re).exp() * value.norm() < error {
            let (value_series, error_series) = gamma_series(n, z);
            if error_series < error {
                return value_series;
            }
        }
    }
    value
}

/// The Kummer sum of DLMF 8.7.1 with 8.2.7,
/// `Gamma(a, z) = Gamma(a) - z^a e^-z sum_k z^k / (a (a+1) ... (a+k))`, whose terms fall
/// at any phase of `z` for `0 < |z| < a`, with the magnitude of the terms as the
/// absolute error bound in units of epsilon.
fn gamma_kummer(n: f64, z: Complex) -> (Complex, f64) {
    let mut term = Complex::new(n.recip(), 0.0);
    let (mut sum, mut magnitude) = (term, term.l1_norm());
    let mut k = 0.0;
    // The terms fall at least like n^k / (n+1)...(n+k), below 1e-17 of the first
    // within 9 sqrt(n) + 40 terms.
    while k < 9.0 * n.sqrt() + 40.0 && term.norm() > 1e-17 * sum.norm() {
        k += 1.0;
        term *= z / (n + k);
        sum += term;
        magnitude += term.l1_norm();
    }
    // |z|^a < a^a and |e^-z| < e^a stay representable for the degrees used (a < 131),
    // and their product only overflows with gamma(a, z).
    let exponential = (-z).exp();
    let power = half_power(z, n);
    let complete = libm::tgamma(n);
    (
        complete - power * (exponential * sum),
        complete + power.norm() * (exponential.norm() * magnitude),
    )
}

/// Whether the rounding error of the downward recurrence from `base` to `n` grows
/// beyond [`RECURRENCE_GROWTH`]: the step to degree `d` amplifies it by
/// `max(1, |z|/|d|)`.
fn downward_unstable(n: f64, base: f64, radius: f64) -> bool {
    let mut growth = 1.0;
    let mut degree = base - 1.0;
    while degree >= n {
        growth *= (radius / degree.abs()).max(1.0);
        if growth > RECURRENCE_GROWTH {
            return true;
        }
        degree -= 1.0;
    }
    false
}

/// The degree recurrence DLMF 8.8.2 from `E1` (integer degree) or `sqrt(pi) erfc(sqrt z)`
/// (half-integer degree), with powers of `z` updated by one multiplication per step
/// (integer degrees stay real on the real axis), and a first-order bound on the absolute
/// error in units of epsilon from the start value and each endpoint term. `modulus` is
/// `|z|`.
fn gamma_recurrence(n: f64, z: Complex, base: f64, modulus: f64) -> (Complex, f64) {
    let (mut value, mut power, ulps) = if base == 0.0 {
        (exp1(z), Complex::new(1.0, 0.0), 1.0)
    } else {
        // erfc forms e^-z from the rounded square of sqrt z, losing about |z| ulps.
        let root = crate::numerics::complex_sqrt(z);
        (PI.sqrt() * root.erfc(), root, 1.0 + modulus)
    };
    let mut error = ulps * value.l1_norm();
    let exponential = (-z).exp();
    let mut degree = base;
    while degree < n {
        let endpoint = power * exponential;
        value = degree * value + endpoint;
        error = degree * error + endpoint.l1_norm();
        power *= z;
        degree += 1.0;
    }
    let inverse = inverse(z);
    while degree > n {
        degree -= 1.0;
        power *= inverse;
        let endpoint = power * exponential;
        value = (value - endpoint) / degree;
        error = (error + endpoint.l1_norm()) / degree.abs();
    }
    (value, error)
}

/// The power series of DLMF 8.7.3, `Gamma(a,z) = Gamma(a) - z^a sum_k (-z)^k / (k! (a+k))`,
/// whose terms share a phase near the negative real axis; for `a = -m`, DLMF 8.4.15
/// replaces the `Gamma(a)` pole pair by `(-1)^m (psi(m+1) - ln z) / m!`. The principal
/// logarithm and the signed zero of `Im z` select the side of the cut; the magnitude of
/// the terms bounds the error.
fn gamma_series(n: f64, z: Complex) -> (Complex, f64) {
    let radius = z.norm();
    let mut term = Complex::new(1.0, 0.0);
    let (mut sum, mut magnitude) = (Complex::default(), 0.0);
    let mut k = 0.0;
    while k <= 2.0 * radius + 200.0 {
        if n + k != 0.0 {
            let scaled = term / (n + k);
            sum += scaled;
            magnitude += scaled.l1_norm();
        }
        // Beyond k = 2|z| the tail is below the current term.
        if k > 2.0 * radius && term.norm() <= 1e-17 * sum.norm() {
            break;
        }
        k += 1.0;
        term *= -z / k;
    }
    // |z|^n may leave the normal range of f64 where z^n sum does not, as at n = -128
    // and |z| = 500 (1e-345 and 1e217); z^n then enters in two factors.
    let power = half_power(z, n);
    let (sum, magnitude) = if (1e-300..1e300).contains(&power.l1_norm()) {
        (power * sum, power.l1_norm() * magnitude)
    } else {
        let half = (0.5 * n).trunc();
        let [outer, inner] = [half, n - half].map(|degree| half_power(z, degree));
        (
            outer * (inner * sum),
            outer.l1_norm() * (inner.l1_norm() * magnitude),
        )
    };
    if n.fract() != 0.0 || n > 0.0 {
        let complete = libm::tgamma(n);
        return (complete - sum, complete.abs() + magnitude);
    }
    let (mut digamma, mut scale, mut j) = (-EULER, 1.0, 1.0);
    while j <= -n {
        digamma += 1.0 / j;
        scale /= -j;
        j += 1.0;
    }
    let logarithm = z.ln();
    (
        scale * (digamma - logarithm) - sum,
        scale.abs() * (digamma.abs() + logarithm.norm()) + magnitude,
    )
}

/// `z^degree` on the principal branch for an integer or half-integer degree. Integer
/// degrees stay exactly real on the real axis, unlike `exp(degree ln z)`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Bounded integral degrees.
fn half_power(z: Complex, degree: f64) -> Complex {
    let floor = degree.floor();
    let power = if floor < 0.0 {
        inverse(z).powu((-floor) as u32)
    } else {
        z.powu(floor as u32)
    };
    if degree == floor {
        power
    } else {
        power * crate::numerics::complex_sqrt(z)
    }
}

/// `1/z` without squaring `|z|`: `num_complex`'s `inv()` divides by `|z|^2`, which loses
/// precision below `|z| = 1e-154` and underflows to NaN results below `1e-162`.
fn inverse(z: Complex) -> Complex {
    crate::numerics::ratio(Complex::new(1.0, 0.0), z)
}

/// Whether `1e-290 <= |x| < 1e290`, from the larger component of a finite `x` unless
/// `|x|` lies near either end (`|x|` is at most `sqrt 2` times that component).
fn normal(x: Complex) -> bool {
    let larger = x.re.abs().max(x.im.abs());
    (finite(x) && (2e-290..7e289).contains(&larger)) || (1e-290..1e290).contains(&x.norm())
}

/// Scaled incomplete gamma functions `R(d) = Gamma(d, a) / a^d` of one argument at
/// descending degrees `d = n, n-1, ...`: downward by `R(d) = (a R(d+1) - e^-a) / d`,
/// which is stable once `|d| >= |a|`, and above that upward from one [`upper_gamma`]
/// evaluation up to [`LADDER_RUN`] degrees lower. At `a = 0` every value is the limit
/// `-1/d`. `reflected` selects the branch `-sqrt(a)` for half-integer degrees.
#[derive(Clone, Debug)]
pub(crate) struct ScaledGammaLadder {
    degree: f64,
    argument: Complex,
    /// `|a|`, which every step compares with the degree.
    modulus: f64,
    exponential: Complex,
    reflected: bool,
    /// `R` one degree higher, once one value exists.
    upper: Option<Complex>,
    /// Values of the next degrees from the upward recurrence; the last one is next.
    pending: Vec<Complex>,
}

/// Degrees per upward run of [`ScaledGammaLadder`]. Short runs keep the seed far
/// below `|a|` in degree magnitude, where the continued fraction converges quickly,
/// and avoid evaluating degrees that a short series never reaches.
const LADDER_RUN: f64 = 16.0;

impl ScaledGammaLadder {
    /// A ladder whose first value is `R(n)`.
    pub(crate) fn new(n: f64, argument: Complex, reflected: bool) -> Self {
        Self {
            degree: n,
            argument,
            modulus: argument.norm(),
            exponential: (-argument).exp(),
            reflected,
            upper: None,
            pending: Vec::new(),
        }
    }

    /// The next value: `R(n)` first, then one degree lower each call.
    pub(crate) fn next_lower(&mut self) -> Complex {
        let stable = self.degree.abs() >= self.modulus;
        let value = if self.argument == Complex::default() {
            // The limit needs no recurrence, which R(0) = infinity would spoil.
            self.seed(self.degree)
        } else {
            match (self.pending.pop(), self.upper) {
                (Some(value), _) => value,
                // |degree| >= |z| controls propagation, but does not prevent
                // cancellation at a zero. Restart at the factored degree-two
                // value instead of subtracting nearly equal degree-three terms.
                (None, _) if stable && self.degree == 2.0 => self.seed(self.degree),
                (None, Some(upper)) if stable => {
                    (self.argument * upper - self.exponential) / self.degree
                }
                (None, None) if stable => self.run_down(),
                _ => self.run_up(),
            }
        };
        self.upper = Some(value);
        self.degree -= 1.0;
        value
    }

    /// `R` on the selected branch from one gamma evaluation, or its limit at `a = 0`.
    /// Where `Gamma(d, a)` or `a^d` leave the normal range of `f64` (near `|d| = |a|` from
    /// `|a|` of about 120, at far evanescent orders of the lattice sums) `R = e^-a h`
    /// comes from the continued fraction `h` itself; the division scales the denominator
    /// first.
    fn seed(&self, degree: f64) -> Complex {
        if self.argument == Complex::default() {
            let limit = if degree < 0.0 {
                -1.0 / degree
            } else {
                f64::INFINITY
            };
            return Complex::new(limit, 0.0);
        }
        let power = half_power(self.argument, degree);
        let mut value = upper_gamma(degree, self.argument);
        if self.reflected {
            value -= 2.0 * libm::tgamma(degree);
        }
        // The factored Gamma(2, -1) zero is exact, not an underflowed seed.
        // Keep it out of the continued fraction's tiny-denominator clamps.
        let representable = (normal(value)
            || (degree == 2.0 && self.argument == Complex::new(-1.0, 0.0)))
            && normal(power);
        if !(self.reflected || representable)
            && let (fraction, true) = continued_fraction(degree, self.argument)
        {
            return (-self.argument).exp() * fraction;
        }
        crate::numerics::ratio(value, power)
    }

    /// `R` at the current, stable degree from the degree nearest zero (at most -1/2)
    /// below which every downward step is stable, so that tiny `|a|` never meets `a^d` at
    /// a large one.
    fn run_down(&self) -> Complex {
        let start = (1.0 - self.modulus).min(-0.5);
        let mut degree = self.degree + (start - self.degree).floor().max(0.0);
        let mut value = self.seed(degree);
        while degree > self.degree {
            degree -= 1.0;
            value = (self.argument * value - self.exponential) / degree;
        }
        value
    }

    /// `R` at the current degree by the upward recurrence `R(d+1) = (d R(d) + e^-a) / a`,
    /// whose error shrinks by `|d| / |a|` per step, from one evaluation at the lowest
    /// degree with `|d| < |a|` (or the current one), queuing the values in between.
    fn run_up(&mut self) -> Complex {
        let steps = ((self.degree + self.modulus).ceil() - 1.0).clamp(0.0, LADDER_RUN);
        let mut degree = self.degree - steps;
        let mut value = self.seed(degree);
        while degree < self.degree {
            self.pending.push(value);
            value = (degree * value + self.exponential) / self.argument;
            degree += 1.0;
        }
        value
    }
}

/// The argument derivative `dGamma(n, z)/dz = -z^(n-1) e^-z`, with its limits `-1`
/// (`n = 1`) and `0` (`n > 1`) at `z = 0`.
pub(super) fn gamma_derivative(n: f64, z: Complex) -> Complex {
    if z == Complex::default() && n >= 1.0 {
        Complex::new(if n == 1.0 { -1.0 } else { 0.0 }, 0.0)
    } else {
        -((n - 1.0) * z.ln() - z).exp()
    }
}

#[cfg(test)]
mod tests {
    //! `upper_gamma` and its scaled ladder against reference tables, DLMF identities and
    //! quadrature; the lattice sums built on them are checked in `properties/lattice/`.

    use std::f64::consts::PI;

    use super::{ScaledGammaLadder, half_power, upper_gamma};
    use crate::{
        Complex,
        numerics::finite,
        special::integrals::{incgamma, incgamma_array, reference},
        test_support::{ALGEBRA_CASES, complex, gauss_legendre, log_polar, prop_assert_close},
    };
    use proptest::{prelude::*, test_runner::TestCaseError};

    /// The accuracy of `incgamma` against 40-digit mpmath values in
    /// `references/incgamma.txt`: every regime (continued fraction, power series,
    /// degree recurrence), both sides of the branch cut and `|z|` up to 63.
    #[test]
    fn incgamma_matches_reference_table() {
        let cases = reference(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/references/incgamma.txt"
        )));
        assert_eq!(cases.len(), 1494);
        for (key, expected) in cases {
            let (n, z) = (key[0], Complex::new(key[1], key[2]));
            let actual = incgamma(n, z).unwrap();
            // Positive integer degrees have zeros; measure them on the scale of
            // their leading term z^(n-1) e^-z.
            let scale = expected.norm()
                + if n >= 1.0 {
                    (half_power(z, n - 1.0) * (-z).exp()).norm()
                } else {
                    0.0
                };
            assert!(
                (actual - expected).norm() <= 1e-13 * scale,
                "Gamma({n}, {z}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn integer_degrees_are_exact_on_the_negative_real_axis() {
        for x in [0.5, 3.0, 9.0, 36.0] {
            for side in [0.0, -0.0] {
                let z = Complex::new(-x, side);
                for n in 1..=12 {
                    // Gamma(n, z) = (n-1)! e^-z sum_k<n z^k/k! is entire and real here.
                    assert_eq!(incgamma(f64::from(n), z).unwrap().im, 0.0);
                }
                for m in 0..=12_i32 {
                    // The cut carries Im Gamma(-m, -x +- i0) = -+ pi (-1)^m / m!.
                    let factorial: f64 = (1..=m).map(f64::from).product();
                    let expected =
                        -PI.copysign(side) * f64::from((-1_i32).pow(m.unsigned_abs())) / factorial;
                    let actual = incgamma(f64::from(-m), z).unwrap().im;
                    assert!(
                        (actual - expected).abs() <= 1e-14 * expected.abs(),
                        "{m} {z}"
                    );
                }
            }
        }
    }

    /// Negative degrees at `|z|` below 1e-154, where squaring `|z|` to invert it
    /// underflows: each part within 1e-13 of its 500-digit mpmath reference (the real
    /// parts are 1e-100 to 1e-158 of the modulus).
    #[test]
    fn incgamma_at_tiny_arguments() {
        for (key, expected) in reference(TINY_ARGUMENTS) {
            let (n, z) = (key[0], Complex::new(key[1], key[2]));
            let actual = incgamma(n, z).unwrap();
            assert!(
                (actual.re - expected.re).abs() <= 1e-13 * expected.re.abs()
                    && (actual.im - expected.im).abs() <= 1e-13 * expected.im.abs(),
                "Gamma({n}, {z}) = {actual}, expected {expected}"
            );
        }
    }

    /// `n re(z) im(z): re im` of [`incgamma_at_tiny_arguments`].
    const TINY_ARGUMENTS: &str = "
        -0.5 1e-200 0.0: 2e100 0.0
        -0.5 -5e-201 0.0: -3.544907701811032 -2.82842712474619e100
        -1.5 0.0 1e-160: -4.7140452079103165e239 -4.7140452079103165e239
        -1.0 0.0 1e-160: -368.8363992141458 -1e160
        -1.0 3e-170 4e-170: 1.2e169 -1.6e169";

    /// Large degrees against 60-digit mpmath references to 1e-13: near the negative real
    /// axis, where `z^n` leaves the range of `f64` at negative degree and the upward
    /// recurrence cancels at positive degree, and off it with `|z| < n`, where the Kummer
    /// sum takes over.
    #[test]
    fn incgamma_at_large_degrees() {
        for (key, expected) in reference(LARGE_DEGREES) {
            let (n, z) = (key[0], Complex::new(key[1], key[2]));
            let actual = incgamma(n, z).unwrap();
            assert!(
                (actual - expected).norm() <= 1e-13 * expected.norm(),
                "Gamma({n}, {z}) = {actual}, expected {expected}"
            );
        }
    }

    /// `n re(z) im(z): re im` of [`incgamma_at_large_degrees`].
    const LARGE_DEGREES: &str = "
        -128.0 -500.0 0.0: -1.2885957841150845e-131 -3.7795814367747375e-197
        -127.5 -500.0 5.0: 1.542698211604819e-130 2.407001953488497e-130
        -120.0 -550.0 -20.0: 2.2305931400450102e-93 -2.606522282697957e-94
        -118.0 -500.0 0.0: -1.2252201215779826e-104 -3.901330675138805e-170
        -128.0 -300.0 0.0: -9.678079515916463e-190 -8.146851106928248e-216
        128.0 -40.0 0.0: -1.624661197110015e220 0.0
        100.0 -30.0 -0.0: -4.2347612953807334e158 0.0
        125.0 -33.8 5.2: 1.507213843157233e207 1.308006804208391e204
        128.0 -5.128 45.35: 2.805917215158467e213 -2.1505509958846317e212
        100.5 0.1129 -25.538: 9.320963104082716e156 4.155006301443276e138
        120.0 1.0 15.0: 5.574585761207606e196 -4.1455631801194624e138
        64.5 -20.0 20.0: -2.2838716862512102e100 -6.047806419221159e99";

    /// Recorded cancellation near Gamma(2, -1) and a reflected half-integer branch.
    #[test]
    fn gamma_ladder_regressions() -> Result<(), TestCaseError> {
        check_gamma_ladder(3.0, Complex::new(-0.999_044_462_484_192_2, 0.0), false)?;
        check_gamma_ladder(
            0.5,
            Complex::new(-9.001_950_249_594_67, 7.370_858_461_664_222_5),
            true,
        )
    }

    /// Independent mpmath values, including the exact zero and perturbations
    /// too small for either recurrence to preserve relative accuracy unaided.
    #[test]
    fn gamma_and_ladder_keep_accuracy_near_degree_two_zero() {
        for (key, expected) in reference(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/references/incgamma_zero.txt"
        ))) {
            let (degree, z) = (key[0], Complex::new(key[1], key[2]));
            let tolerance = 1e-13 * expected.norm() + f64::MIN_POSITIVE;
            let actual = incgamma(degree, z).unwrap();
            assert!(
                (actual - expected).norm() <= tolerance,
                "Gamma({degree}, {z}) = {actual}, expected {expected}"
            );
            for top in [3, 16] {
                let mut ladder = ScaledGammaLadder::new(f64::from(top), z, false);
                for n in (1..=top).rev() {
                    let actual = ladder.next_lower() * half_power(z, f64::from(n));
                    if f64::from(n) == degree {
                        assert!(
                            (actual - expected).norm() <= tolerance,
                            "ladder({top}, {degree}, {z}) = {actual}, expected {expected}"
                        );
                        break;
                    }
                }
            }
        }
    }

    /// Gamma(2, -1) / (-1)^2 is exactly zero, on either signed-zero side.
    #[test]
    fn scaled_gamma_ladder_preserves_exact_degree_two_zero() {
        for imaginary in [0.0, -0.0] {
            let z = Complex::new(-1.0, imaginary);
            for top in [2, 3, 16] {
                let mut ladder = ScaledGammaLadder::new(f64::from(top), z, false);
                for _ in 2..top {
                    ladder.next_lower();
                }
                assert_eq!(ladder.next_lower(), Complex::default(), "top={top}, z={z}");
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn gamma_recurrence_and_reflection(
            twice_n in -32_i32..=32,
            re in -60.0_f64..60.0,
            im in prop_oneof![Just(0.0), Just(-0.0), -20.0_f64..20.0],
            g in complex(1.0),
        ) {
            check_gamma(0.5 * f64::from(twice_n), Complex::new(re, im), g)?;
        }

        #[test]
        fn gamma_recurrence_near_the_cut(
            twice_n in -256_i32..=254,
            x in (-2.0_f64..600_f64.log10()).prop_map(|e| 10_f64.powf(e)),
            parabola in prop_oneof![Just(0.0), 0.0_f64..1.0],
            side in prop_oneof![Just(1.0), Just(-1.0)],
        ) {
            // |z| + Re z = parabola, or the cut itself from either side.
            let im = side * (parabola * (parabola + 2.0 * x)).sqrt();
            check_gamma_relations(0.5 * f64::from(twice_n), Complex::new(-x, im))?;
        }

        #[test]
        fn gamma_difference_is_an_integral(
            twice_n in 1_i32..=256,
            ratio in 0.05_f64..1.5,
            phase in -PI..PI,
            step in prop_oneof![Just(-0.25), Just(0.25)],
        ) {
            let n = 0.5 * f64::from(twice_n);
            let z = Complex::from_polar((ratio * n).max(1.0), phase);
            check_gamma_integral(n, z, Complex::new(step, 0.0))?;
        }

        #[test]
        fn gamma_ladder_matches_fresh_values(
            twice_n in -8_i32..=8,
            z in prop_oneof![
                (-40.0_f64..40.0, prop_oneof![Just(0.0), -10.0_f64..10.0])
                    .prop_map(|(re, im)| Complex::new(re, im)),
                // Small |z|, where an upward run from |d| >= |z| would amplify rounding.
                log_polar(-12.0..0.0, -PI..PI),
            ],
            reflected: bool,
        ) {
            check_gamma_ladder(0.5 * f64::from(twice_n), z, reflected)?;
        }
    }

    /// `Gamma(n+1, z) = n Gamma(n, z) + z^n e^-z` relative to its largest term, which
    /// ties together the continued-fraction, series and recurrence regimes and their
    /// switches; Schwarz reflection `Gamma(n, conj z) = conj Gamma(n, z)` including the
    /// signed-zero side of the cut; and the argument pullback `g conj(-z^(n-1) e^-z)`.
    fn check_gamma(n: f64, z: Complex, g: Complex) -> Result<(), TestCaseError> {
        check_gamma_relations(n, z)?;
        let (_, residual) = incgamma_array(vec![n], vec![z]).unwrap();
        let derivative = -half_power(z, n - 1.0) * (-z).exp();
        let gradient = residual.pullback(&[g]).unwrap();
        prop_assert_close!(
            gradient[0],
            g * derivative.conj(),
            1e-13 * (g * derivative).norm()
        );
        Ok(())
    }

    /// The degree recurrence and Schwarz reflection of [`check_gamma`], where the
    /// values and the endpoint term are representable (`z^n` alone may not be).
    fn check_gamma_relations(n: f64, z: Complex) -> Result<(), TestCaseError> {
        prop_assume!(z != Complex::default());
        let (value, upper) = (incgamma(n, z).unwrap(), incgamma(n + 1.0, z).unwrap());
        let half = (0.5 * n).trunc();
        let endpoint = half_power(z, half) * (half_power(z, n - half) * (-z).exp());
        prop_assume!(finite(value) && finite(upper) && finite(endpoint));
        prop_assume!(endpoint.norm() > 1e-290);
        let scale = upper.norm().max((n * value).norm()).max(endpoint.norm());
        prop_assert_close!(upper - n * value, endpoint, 1e-13 * scale);
        let reflected = incgamma(n, z.conj()).unwrap();
        prop_assert_close!(reflected, value.conj(), 1e-15 * value.norm());
        Ok(())
    }

    /// `Gamma(n, z) - Gamma(n, z + h)` is the integral of `t^(n-1) e^-t` along the
    /// segment, by Gauss-Legendre quadrature: an identity no method of `upper_gamma`
    /// builds on, unlike the degree recurrence, which the upward recurrence satisfies
    /// even where it has lost every digit. The segment keeps to one side of the branch
    /// cut, and the error is measured against the values and the largest integrand, as
    /// in the reference table test.
    fn check_gamma_integral(n: f64, z: Complex, h: Complex) -> Result<(), TestCaseError> {
        let (start, end) = (incgamma(n, z).unwrap(), incgamma(n, z + h).unwrap());
        let mut integral = Complex::default();
        let mut largest = 0.0_f64;
        for (node, weight) in gauss_legendre(32) {
            let t = z + 1.0_f64.midpoint(node) * h;
            let integrand = half_power(t, n - 1.0) * (-t).exp();
            integral += 0.5 * weight * h * integrand;
            largest = largest.max(integrand.norm());
        }
        prop_assume!(finite(start) && finite(end) && finite(integral));
        let scale = start.norm() + end.norm() + largest;
        prop_assert_close!(start - end, integral, 3e-13 * scale);
        Ok(())
    }

    /// Each ladder value is the scaled incomplete gamma function `Gamma(d, z) / z^d`
    /// one degree lower, on the branch `-sqrt(z)` for reflected half-integer degrees,
    /// where `Gamma(d, z)` becomes `2 Gamma(d) - Gamma(d, z)` (measured on the scale of
    /// its terms, which cancel).
    fn check_gamma_ladder(n: f64, z: Complex, reflected: bool) -> Result<(), TestCaseError> {
        prop_assume!(z != Complex::default());
        let reflected = reflected && n.fract() != 0.0;
        let mut ladder = ScaledGammaLadder::new(n, z, reflected);
        for step in 0..60 {
            let degree = n - f64::from(step);
            let value = upper_gamma(degree, z);
            let (expected, power, scale) = if reflected {
                let complete = 2.0 * libm::tgamma(degree);
                (
                    complete - value,
                    -half_power(z, degree),
                    complete.abs() + value.norm(),
                )
            } else {
                (value, half_power(z, degree), value.norm())
            };
            if !finite(expected * power) {
                // |z|^d leaves the range of f64 at large negative d for small |z|.
                break;
            }
            prop_assert_close!(
                ladder.next_lower() * power,
                expected,
                1e-13 * scale,
                "degree {degree}"
            );
        }
        Ok(())
    }
}

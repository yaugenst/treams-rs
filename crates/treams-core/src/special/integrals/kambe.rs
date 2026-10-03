//! Kambe integrals `I_n(z, eta) = integral_eta^infinity t^n exp(-z^2 t^2/2 + 1/(2t^2)) dt`
//! with first-order bounds on their rounding errors.
//!
//! [`kambe()`] runs the Kambe (1968) recurrence up from a base pair: closed forms for even
//! orders, the series for odd ones. Below the pair it takes the series over scaled
//! incomplete gamma functions or the downward recurrence, whichever bounds the error more
//! tightly. [`EvenKambe`] carries the even orders of one reciprocal point of a lattice
//! sum. [`SmallSplitKambe`] gives the integrals of the real-space terms of a lattice sum
//! at splits below every automatic one.
//!
//! Upstream: `intkambe` of `treams/special/_integrals.pyx`. The rounding bounds, the
//! choice below the base pair and the small-split series are treams-rs extensions.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use errorfunctions::ComplexErrorFunctions;

use super::{
    double::DoubleComplex,
    gamma::{RECURRENCE_GROWTH, ScaledGammaLadder},
};
use crate::{Complex, numerics::finite};

/// The argument `a = (z eta)^2 / 2` of the incomplete gamma functions of the Kambe
/// series, as the series evaluates it. Odd orders take the principal branch of `ln a`,
/// which the 1D spherical lattice sums continue to another sheet where they need it.
pub(crate) fn kambe_argument(z: Complex, eta: Complex) -> Complex {
    0.5 * (z * z) * eta * eta
}

/// Largest `|c|` that sets the length of the Kambe series. Its terms `c^j / j!` peak
/// at about `e^|c| / sqrt(2 pi |c|)`, which overflows beyond `|c| = 715`: no larger
/// `|c|` gives a finite sum.
const SERIES_PEAK_LIMIT: f64 = 1000.0;

/// Kambe integrals by expanding `exp(1/(2t^2))`: with `a = (z eta)^2 / 2`,
/// `c = 1 / (2 eta^2)` and `R(d) = Gamma(d, a) / a^d` (`-1/d` as `a -> 0`),
/// `I_n = eta^(n+1)/2 sum_j c^j/j! R((n+1)/2 - j)`, whose terms cancel by up to
/// `e^(|c| - Re c)` for complex `eta`. Orders of one parity share one ladder of `R`, each
/// with the magnitude of its terms as its error bound. Even orders take the branch
/// `sqrt(a) = z eta / sqrt 2` with `Re z >= 0`.
fn kambe_series<const K: usize>(orders: [i32; K], z: Complex, eta: Complex) -> [(Complex, f64); K] {
    kambe_series_with(orders, z, eta, None)
}

/// The number of terms of the Kambe series at `c = 1 / (2 eta^2)` after which each order
/// stops: its terms peak near `j = |c|` and soon fall below the rounding of the sum.
fn series_terms(ratio: Complex) -> f64 {
    200.0 + 3.0 * ratio.norm().min(SERIES_PEAK_LIMIT)
}

/// [`kambe_series`] with the coefficients `c^j/j!` from `table`, or recurred term by term.
fn kambe_series_with<const K: usize>(
    orders: [i32; K],
    z: Complex,
    eta: Complex,
    table: Option<&[Complex]>,
) -> [(Complex, f64); K] {
    struct Partial {
        start: i32,
        coefficient: Complex,
        sum: Complex,
        magnitude: f64,
        converged: bool,
    }
    let top = orders.into_iter().max().unwrap_or_default();
    let even = top % 2 == 0;
    let z = if even && z.re < 0.0 { -z } else { z };
    let square = z * z;
    let argument = kambe_argument(z, eta);
    let reflected =
        even && (crate::numerics::complex_sqrt(square * eta * eta) * (z * eta).conj()).re < 0.0;
    let ratio = 0.5 / (eta * eta);
    let peak = ratio.norm();
    let last = series_terms(ratio);
    let mut ladder = ScaledGammaLadder::new(0.5 * f64::from(top + 1), argument, reflected);
    // Order n takes its first term (top - n) / 2 ladder steps down.
    let mut partials = orders.map(|n| Partial {
        start: (top - n) / 2,
        coefficient: Complex::new(1.0, 0.0),
        sum: Complex::default(),
        magnitude: 0.0,
        converged: false,
    });
    let mut step = 0;
    loop {
        let value = ladder.next_lower();
        for partial in partials
            .iter_mut()
            .filter(|partial| partial.start <= step && !partial.converged)
        {
            let index = step - partial.start;
            let j = f64::from(index);
            if let Some(table) = table {
                partial.coefficient = usize::try_from(index)
                    .ok()
                    .and_then(|index| table.get(index))
                    .copied()
                    .unwrap_or_default();
            } else if j > 0.0 {
                partial.coefficient *= ratio / j;
            }
            let term = partial.coefficient * value;
            let modulus = term.l1_norm();
            partial.sum += term;
            partial.magnitude += modulus;
            // With the shared coefficients of the lattice sums the test compares `|Re| +
            // |Im|`, at most `sqrt 2` from the modulus, which takes no square roots.
            let small = if table.is_some() {
                modulus <= 1.4e-16 * partial.sum.l1_norm()
            } else {
                term.norm() <= 2e-16 * partial.sum.norm()
            };
            partial.converged = (j > 3.0 && j > peak && small) || j >= last || !finite(partial.sum);
        }
        if partials.iter().all(|partial| partial.converged) {
            break;
        }
        step += 1;
    }
    std::array::from_fn(|i| {
        let prefactor = 0.5 * eta.powi(orders[i] + 1);
        (
            prefactor * partials[i].sum,
            prefactor.norm() * partials[i].magnitude,
        )
    })
}

/// Orders below the base pair from the series, which cancels by about `e^(|c| - Re c)`
/// for complex `eta`, or the downward recurrence from the base pair (as in treams),
/// accurate while `|1/eta^2|` exceeds about `|n|` and `|z|`: the smaller error bound
/// wins. A series whose predicted loss stays within [`RECURRENCE_GROWTH`] is kept
/// outright; otherwise odd orders sum their base pair in the same pass, and even orders
/// keep a recurrence that grows by at most [`RECURRENCE_GROWTH`] and try the series only
/// where its predicted loss is below [`RECURRENCE_GROWTH`] times the recurrence's growth.
fn kambe_below_base(n: i32, z: Complex, eta: Complex) -> (Complex, f64) {
    let ratio = 0.5 / (eta * eta);
    let odd = n % 2 != 0;
    let cancellation = ratio.norm() - ratio.re;
    if cancellation > RECURRENCE_GROWTH.ln() {
        if odd {
            let [series, lower, upper] = kambe_series([n, -3, -1], z, eta);
            return more_accurate(series, kambe_downward(n, z, eta, [lower, upper]));
        }
        let recurred = kambe_downward(n, z, eta, even_pair(z, eta));
        let growth = recurred.1 / recurred.0.norm();
        if growth <= RECURRENCE_GROWTH || cancellation.exp() >= RECURRENCE_GROWTH * growth {
            return recurred;
        }
        let [series] = kambe_series([n], z, eta);
        return more_accurate(series, recurred);
    }
    let [series] = kambe_series([n], z, eta);
    if series.1 <= RECURRENCE_GROWTH * series.0.norm() {
        return series;
    }
    more_accurate(series, kambe_downward(n, z, eta, base_pair(odd, z, eta)))
}

/// The value with the smaller error bound, unless the series overflowed.
fn more_accurate(series: (Complex, f64), recurred: (Complex, f64)) -> (Complex, f64) {
    if recurred.1 < series.1 || !finite(series.0) {
        recurred
    } else {
        series
    }
}

/// The Kambe (1968) recurrence `I_n = (n+3) I_(n+2) - z^2 I_(n+4) + eta^(n+3) e^(...)`
/// run downward from the base pair, with the first-order absolute error bound in units
/// of epsilon propagated from the pair and each endpoint term.
fn kambe_downward(n: i32, z: Complex, eta: Complex, pair: [(Complex, f64); 2]) -> (Complex, f64) {
    let [(mut lower, mut lower_error), (mut upper, mut upper_error)] = pair;
    // At z = 0 the order above the base diverges, but its coefficient z^2 vanishes.
    if z == Complex::default() {
        (upper, upper_error) = (Complex::default(), 0.0);
    }
    let square = z * z;
    let exponential = (0.5 * (1.0 / (eta * eta) - square * eta * eta)).exp();
    let mut order = if n % 2 == 0 { -4 } else { -5 };
    while order >= n {
        let endpoint = eta.powi(order + 3) * exponential;
        let value = f64::from(order + 3) * lower - square * upper + endpoint;
        let error = f64::from(order + 3).abs() * lower_error
            + square.norm() * upper_error
            + endpoint.l1_norm();
        (upper, upper_error) = (lower, lower_error);
        (lower, lower_error) = (value, error);
        order -= 2;
    }
    (lower, lower_error)
}

/// The base pair `I_-3, I_-1` (odd, by the series) or `I_-2, I_0` (even, in closed form),
/// with error bounds in units of epsilon.
fn base_pair(odd: bool, z: Complex, eta: Complex) -> [(Complex, f64); 2] {
    if odd {
        kambe_series([-3, -1], z, eta)
    } else {
        even_pair(z, eta)
    }
}

/// `I_-2` and `I_0` in closed form by two complementary error functions, with the
/// magnitude of their two terms as error bounds in units of epsilon.
fn even_pair(z: Complex, eta: Complex) -> [(Complex, f64); 2] {
    let (terms, z) = even_terms(z, eta);
    let [lower, upper] = even_values(terms, z);
    let magnitude = even_scale() * (terms[0].norm() + terms[1].norm());
    [(lower, magnitude), (upper, magnitude / z.norm())]
}

/// The factor `sqrt(pi / 8)` of the closed form of the even base pair.
fn even_scale() -> f64 {
    0.5 * PI.sqrt() * FRAC_1_SQRT_2
}

/// The two terms of the closed form of the even base pair, and the argument with
/// `Re z >= 0` they take (even orders are even in `z`).
fn even_terms(z: Complex, eta: Complex) -> ([Complex; 2], Complex) {
    let z = if z.re < 0.0 { -z } else { z };
    let plus = ((z * eta - Complex::i() / eta) * FRAC_1_SQRT_2).erfc() * (-Complex::i() * z).exp();
    let minus = ((z * eta + Complex::i() / eta) * FRAC_1_SQRT_2).erfc() * (Complex::i() * z).exp();
    ([plus, minus], z)
}

/// `I_-2` and `I_0` from the terms of [`even_terms`].
fn even_values([plus, minus]: [Complex; 2], z: Complex) -> [Complex; 2] {
    let scale = even_scale();
    [
        -Complex::i() * scale * (plus - minus),
        scale * (plus + minus) / z,
    ]
}

/// Even Kambe orders `I_2, I_0, I_-2, I_-4, ...` of one argument from the closed-form pair
/// `I_-2, I_0` and the Kambe recurrence, one step per order, as treams evaluates them:
/// the orders of one reciprocal point of a 2D spherical or 1D cylindrical lattice sum,
/// whose correlated errors those sums cancel (see `lattice::Reduced`).
#[derive(Clone, Debug)]
pub(crate) struct EvenKambe {
    /// The argument `z` with `Re z >= 0` that the pair takes.
    argument: Complex,
    square: Complex,
    eta: Complex,
    exponential: Complex,
    /// The moduli of the two terms of the closed-form pair ([`even_terms`]).
    terms: [f64; 2],
    /// `I_0, I_-2, ...` as far as requested.
    values: Vec<Complex>,
}

impl EvenKambe {
    /// The chain of `I_n(z, eta)`, starting from the pair.
    pub(crate) fn new(z: Complex, eta: Complex) -> Self {
        let (terms, argument) = even_terms(z, eta);
        let [lower, upper] = even_values(terms, argument);
        let square = z * z;
        Self {
            argument,
            square,
            eta,
            exponential: (0.5 * (1.0 / (eta * eta) - square * eta * eta)).exp(),
            terms: terms.map(Complex::norm),
            values: vec![upper, lower],
        }
    }

    /// `I_n` for even `n <= 0` on its own, with a first-order bound on its absolute
    /// rounding error in units of epsilon ([`kambe_with_bound`]): below the base pair from
    /// the series where that bounds it more tightly than the recurrence.
    pub(crate) fn single(&self, n: i32) -> (Complex, f64) {
        kambe_with_bound(n, self.argument, self.eta)
    }

    /// A first-order bound, in units of epsilon, on the absolute rounding error of
    /// `sum_j weights[j] I_(-2j)` from this chain (which holds the orders down to the last
    /// weight), and the sum of the moduli of its terms: one ulp of each term of the pair
    /// and of each product and endpoint term of the recurrence, carried to the sum by the
    /// adjoint of the recurrence, which sees the correlation of the errors.
    pub(crate) fn sum_bounds(&self, weights: &[Complex]) -> (f64, f64) {
        let values = &self.values;
        let moduli = weights
            .iter()
            .zip(values)
            .map(|(weight, value)| weight.l1_norm() * value.l1_norm())
            .sum();
        let (mut next, mut after) = (Complex::default(), Complex::default());
        let mut bound = 0.0;
        for (j, &weight) in weights.iter().enumerate().rev() {
            // The derivative of the sum in I_(-2j) through every order below it: I_(-2j-2)
            // takes (1 - 2j) I_(-2j) from j >= 1, I_(-2j-4) takes -z^2 I_(-2j).
            let order = i32::try_from(j).unwrap_or(i32::MAX / 2);
            let step = if j >= 1 {
                f64::from(1 - 2 * order)
            } else {
                0.0
            };
            let adjoint = weight + step * next - self.square * after;
            if let (Some(&lower), Some(&upper)) =
                (values.get(j.wrapping_sub(1)), values.get(j.wrapping_sub(2)))
                && j >= 2
            {
                let endpoint = self.eta.powi(3 - 2 * order) * self.exponential;
                let rounding = (f64::from(3 - 2 * order) * lower).l1_norm()
                    + (self.square * upper).l1_norm()
                    + endpoint.l1_norm();
                bound += adjoint.l1_norm() * rounding;
            }
            (after, next) = (next, adjoint);
        }
        // I_0 = s (p + m) / z and I_-2 = -i s (p - m) from the terms p, m of the pair.
        let (zero, lower) = (next, after);
        let scale = even_scale();
        let sum = zero / self.argument;
        let difference = Complex::i() * lower;
        let pair = scale
            * (self.terms[0] * (sum - difference).l1_norm()
                + self.terms[1] * (sum + difference).l1_norm());
        (bound + pair, moduli)
    }

    /// `I_n` for even `n <= 2`.
    pub(crate) fn get(&mut self, n: i32) -> Complex {
        debug_assert!(n <= 2 && n % 2 == 0, "even Kambe order {n} above 2");
        if n > 0 {
            // One upward step: I_2 = (I_0 - I_-2 + eta e^(...)) / z^2.
            return (self.values[0] - self.values[1] + self.eta * self.exponential) / self.square;
        }
        let index = usize::try_from(-n / 2).unwrap_or_default();
        let mut order = -2 * i32::try_from(self.values.len()).unwrap_or(i32::MAX / 2);
        while order >= n {
            let length = self.values.len();
            let (upper, lower) = (self.values[length - 2], self.values[length - 1]);
            self.values.push(
                f64::from(order + 3) * lower - self.square * upper
                    + self.eta.powi(order + 3) * self.exponential,
            );
            order -= 2;
        }
        self.values[index]
    }
}

/// The Kambe integral `I_n(z, eta)` without input checks.
///
/// The Kambe (1968) recurrence, as in treams, runs upward from the base pair
/// `I_-3, I_-1` (odd) or `I_-2, I_0` (even); orders below it come from the series or the
/// downward recurrence, whichever amplifies rounding less. Orders above -2 diverge at
/// `z = 0`, all at `eta = 0`.
pub(crate) fn kambe(n: i32, z: Complex, eta: Complex) -> Complex {
    let zero = Complex::default();
    if eta == zero || (z == zero && n > -2) {
        return Complex::new(f64::INFINITY, 0.0);
    }
    let odd = n % 2 != 0;
    let base = if odd { -3 } else { -2 };
    // Below the base pair the bounds choose between the series and the recurrence.
    if n < base || (odd && n <= -1) {
        return kambe_with_bound(n, z, eta).0;
    }
    // Above it the values of `kambe_with_bound`, without the moduli of their bounds.
    let pair = if odd {
        kambe_series([-3, -1], z, eta).map(|(value, _)| (value, 0.0))
    } else {
        let (terms, argument) = even_terms(z, eta);
        let values = even_values(terms, argument);
        // I_-2 alone skips the division of I_0.
        if n == base {
            return values[0];
        }
        values.map(|value| (value, 0.0))
    };
    let [(lower, _), (upper, _)] = pair;
    if n <= base + 2 {
        return if n == base { lower } else { upper };
    }
    kambe_upward::<false>(n, z, eta, pair, 1.0).0
}

/// [`kambe`] with a first-order bound on the absolute rounding error of its value in
/// units of epsilon: from the series for `I_-3` and `I_-1`, below the base pair from
/// the series or the downward recurrence that `kambe` takes, and above the base pair
/// propagated through the upward recurrence. Requires nonzero `eta`, and nonzero `z`
/// from order -1 up.
pub(crate) fn kambe_with_bound(n: i32, z: Complex, eta: Complex) -> (Complex, f64) {
    let odd = n % 2 != 0;
    let base = if odd { -3 } else { -2 };
    if n < base {
        return kambe_below_base(n, z, eta);
    }
    if odd && n <= -1 {
        let [series] = kambe_series([n], z, eta);
        return series;
    }
    let [lower, upper] = base_pair(odd, z, eta);
    if n <= base + 2 {
        return if n == base { lower } else { upper };
    }
    kambe_upward::<true>(n, z, eta, [lower, upper], 1.0)
}

/// Kambe integrals `I_n(z, eta)` of the real-space terms of an Ewald sum at a split below
/// every automatic one ([`Self::get`]), with the coefficients `c^j / j!` of their series
/// at `c = 1 / (2 eta^2)`, which every integral at that split shares. There the
/// closed-form pair and its recurrence ([`kambe_with_bound`]) cancel by up to `e^Re(c)`
/// (each error function loses about `|a|^2` ulps to `exp(-a^2)` at its argument
/// `a = (z eta -+ i / eta) / sqrt 2`), while the series sums terms of one phase for real
/// `z` and `eta`; its coefficients are rounded from double-double arithmetic, since
/// recurred in double they carry the rounding of `c` into every integral alike.
#[derive(Clone, Debug)]
pub(crate) struct SmallSplitKambe {
    eta: Complex,
    /// `|c| - Re c`, the cancellation of the series.
    cancellation: f64,
    coefficients: Vec<Complex>,
}

impl SmallSplitKambe {
    /// The coefficients of the series at the split `eta`, as many as it may take.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // At most 3201.
    pub(crate) fn new(eta: Complex) -> Self {
        let ratio = 0.5 / (eta * eta);
        let terms = series_terms(ratio) as usize + 1;
        let c = DoubleComplex::half_inverse_square(eta);
        let mut coefficient = DoubleComplex::from(Complex::new(1.0, 0.0));
        let coefficients = (0..terms)
            .map(|j| {
                if j > 0 {
                    #[allow(clippy::cast_precision_loss)] // At most 3201.
                    let j = j as f64;
                    coefficient = (coefficient * c).divide(j);
                }
                coefficient.rounded()
            })
            .collect();
        Self {
            eta,
            cancellation: ratio.norm() - ratio.re,
            coefficients,
        }
    }

    /// `I_n(z, eta)` of `orders` (from -1 up), each with a first-order bound on its
    /// absolute rounding error in units of epsilon that counts the conditioning of the
    /// elementary functions it takes: from the closed forms where the bound of the first
    /// order stays within [`CLOSED_ULPS`] of its value or within `floor` (an error the
    /// caller can afford), or where the series would cancel as much; otherwise from the
    /// series where it is finite and bounds the first order at least as tightly.
    pub(crate) fn get<const K: usize>(
        &self,
        orders: [i32; K],
        z: Complex,
        floor: f64,
    ) -> [(Complex, f64); K] {
        let eta = self.eta;
        // The ladder of the series takes `exp(-a)` at `a = (z eta)^2 / 2`, which loses
        // about `|a|` ulps, as the gamma function that seeds it.
        let ulps = SERIES_ULPS + kambe_argument(z, eta).norm();
        let closed = orders.map(|n| self.closed(n, z, ulps));
        let modulus = closed[0].0.norm();
        // The series cancels by about `e^(|c| - Re c)` relative to its terms.
        if finite(closed[0].0)
            && (closed[0].1 <= floor.max(CLOSED_ULPS * modulus)
                || self.cancellation.exp() * ulps * modulus >= closed[0].1)
        {
            return closed;
        }
        let series = kambe_series_with(orders, z, eta, Some(&self.coefficients))
            .map(|(value, bound)| (value, ulps * bound));
        if finite(series[0].0) && (series[0].1 <= closed[0].1 || !finite(closed[0].0)) {
            series
        } else {
            closed
        }
    }

    /// [`kambe_with_bound`] from the closed-form pair (the series pair for odd orders) and
    /// the upward recurrence, with the bounds of the pair and of the endpoint terms scaled
    /// by the conditioning of the functions they take, and `ulps` those of the series.
    fn closed(&self, n: i32, z: Complex, ulps: f64) -> (Complex, f64) {
        let eta = self.eta;
        let odd = n % 2 != 0;
        let pair = if odd {
            kambe_series_with([-3, -1], z, eta, Some(&self.coefficients))
                .map(|(value, bound)| (value, ulps * bound))
        } else {
            let (terms, argument) = even_terms(z, eta);
            let [lower, upper] = even_values(terms, argument);
            // The error function at `(z eta -+ i / eta) / sqrt 2` of each term.
            let ulps =
                |sign: f64| 1.0 + 0.5 * (argument * eta + sign * Complex::i() / eta).norm_sqr();
            let magnitude = even_scale()
                * terms[0]
                    .norm()
                    .mul_add(ulps(-1.0), terms[1].norm() * ulps(1.0));
            [(lower, magnitude), (upper, magnitude / argument.norm())]
        };
        let base = if odd { -3 } else { -2 };
        debug_assert!(n >= base, "Kambe order {n} below the base pair");
        if n <= base + 2 {
            return pair[usize::from(n != base)];
        }
        let exponent = 0.5 * (1.0 / (eta * eta) - z * z * eta * eta);
        kambe_upward::<true>(n, z, eta, pair, 1.0 + exponent.norm())
    }
}

/// Bound of the closed forms of [`SmallSplitKambe`], relative to their value in ulps,
/// up to which they are taken without the series.
const CLOSED_ULPS: f64 = 64.0;

/// Relative rounding error, in ulps, of the terms of the Kambe series of
/// [`SmallSplitKambe`] beyond that of the exponential of their gamma functions: up to
/// 22 against 40-digit quadrature (orders up to 28).
const SERIES_ULPS: f64 = 32.0;

/// The Kambe recurrence run upward from the base pair to order `n`, with error bounds in
/// units of epsilon where `BOUND` asks for them (the endpoint terms count
/// `endpoint_ulps`).
fn kambe_upward<const BOUND: bool>(
    n: i32,
    z: Complex,
    eta: Complex,
    pair: [(Complex, f64); 2],
    endpoint_ulps: f64,
) -> (Complex, f64) {
    let [(mut lower, mut lower_error), (mut upper, mut upper_error)] = pair;
    let exponential = (0.5 * (1.0 / (eta * eta) - z * z * eta * eta)).exp();
    let square = if BOUND { (z * z).norm() } else { 0.0 };
    let mut order = if n % 2 == 0 { 2 } else { 1 };
    while order <= n {
        let endpoint = eta.powi(order - 1) * exponential;
        let value = (f64::from(order - 1) * upper - lower + endpoint) / (z * z);
        let error = if BOUND {
            (f64::from(order - 1).abs() * upper_error
                + lower_error
                + endpoint_ulps * endpoint.l1_norm())
                / square
                + value.l1_norm()
        } else {
            0.0
        };
        (lower, lower_error) = (upper, upper_error);
        (upper, upper_error) = (value, error);
        order += 2;
    }
    (upper, upper_error)
}

#[cfg(test)]
mod tests {
    //! Kambe integrals against quadrature references and their recurrence; the lattice
    //! sums built on them are checked in `properties/lattice/`.

    use super::{
        SmallSplitKambe, base_pair, even_pair, kambe, kambe_downward, kambe_series,
        kambe_with_bound,
    };
    use crate::{
        Complex,
        numerics::finite,
        special::integrals::{intkambe, intkambe_array, reference},
        test_support::{ALGEBRA_CASES, central, complex, prop_assert_close},
    };
    use proptest::{prelude::*, test_runner::TestCaseError};

    /// The accuracy of `intkambe` against 40-digit quadrature of its defining
    /// integral (`references/kambe.txt`), for orders below, at and above the base pair.
    #[test]
    fn intkambe_matches_quadrature() {
        let cases = reference(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/references/kambe.txt"
        )));
        assert_eq!(cases.len(), 181);
        for (key, expected) in cases {
            #[allow(clippy::cast_possible_truncation)] // Integral orders.
            let n = key[0] as i32;
            let (z, eta) = (Complex::new(key[1], key[2]), Complex::new(key[3], 0.0));
            let actual = intkambe(n, z, eta).unwrap();
            assert!(
                (actual - expected).norm() <= 1e-13 * expected.norm(),
                "I_{n}({z}, {eta}) = {actual}, expected {expected}"
            );
        }
    }

    /// Rounding error of a sum or recurrence in units of epsilon times its first-order
    /// bound, which counts each input and step once. Calibrated: with up to about 200
    /// series terms the factor stays at or below 80 on the 19584 values that
    /// `references/kambe_lattice.txt` is drawn from.
    const SUM_ROUNDING: f64 = 256.0;

    /// The accuracy of `intkambe` at the arguments of the lattice sums, x = sqrt(-2 v w^2)
    /// and eta = -i/w with |w| in [0.5, 6] (`references/kambe_lattice.txt`), where the
    /// series cancels by up to e^(w^2): each value is within 1e-13 plus [`SUM_ROUNDING`]
    /// ulps of the smaller of the series and recurrence bounds and of the bound
    /// `kambe_with_bound` reports, and even orders, which the closed-form pair and a
    /// stable recurrence carry, within 1e-11. Odd orders lose what their base pair loses
    /// to the series, up to all digits of tiny values at w = 6.
    #[test]
    fn intkambe_at_lattice_arguments() {
        let cases = reference(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/references/kambe_lattice.txt"
        )));
        assert_eq!(cases.len(), 1249);
        for (key, expected) in cases {
            #[allow(clippy::cast_possible_truncation)] // Integral orders.
            let n = key[0] as i32;
            let (z, eta) = (Complex::new(key[1], key[2]), Complex::new(key[3], key[4]));
            let actual = intkambe(n, z, eta).unwrap();
            let odd = n % 2 != 0;
            let [series] = kambe_series([n], z, eta);
            let bound = if n < -3 {
                series
                    .1
                    .min(kambe_downward(n, z, eta, base_pair(odd, z, eta)).1)
            } else if odd {
                series.1
            } else {
                even_pair(z, eta)[usize::from(n == 0)].1
            };
            let error = (actual - expected).norm();
            assert!(
                error <= 1e-13 * expected.norm() + SUM_ROUNDING * f64::EPSILON * bound,
                "I_{n}({z}, {eta}) = {actual}, expected {expected}, bound {bound}"
            );
            assert!(
                n % 2 != 0 || error <= 1e-11 * expected.norm(),
                "I_{n}({z}, {eta}) = {actual}, expected {expected}"
            );
            // The bound the 1D spherical lattice sums accumulate covers the same error.
            let (value, reported) = kambe_with_bound(n, z, eta);
            assert_eq!(value, actual, "I_{n}({z}, {eta})");
            assert!(
                error <= 1e-13 * expected.norm() + SUM_ROUNDING * f64::EPSILON * reported,
                "I_{n}({z}, {eta}) = {actual}, expected {expected}, bound {reported}"
            );
        }
    }

    /// Small `z`, where `Gamma(d, (z eta)^2 / 2)` overflows at the orders of the Kambe
    /// series while its scaled form stays finite: within 1e-13 of 80-digit references;
    /// and at `z = 0`, where the scaled terms are exactly `1 / (j - d)`, the closed form
    /// `I_-3(0, eta) = e^(1/(2 eta^2)) - 1`.
    #[test]
    fn kambe_series_at_small_arguments() {
        for (key, expected) in reference(SMALL_ARGUMENTS) {
            #[allow(clippy::cast_possible_truncation)] // Integral orders.
            let n = key[0] as i32;
            let (z, eta) = (Complex::new(key[1], key[2]), Complex::new(key[3], key[4]));
            let actual = intkambe(n, z, eta).unwrap();
            assert!(
                (actual - expected).norm() <= 1e-13 * expected.norm(),
                "I_{n}({z}, {eta}) = {actual}, expected {expected}"
            );
        }
        let eta = Complex::new(0.9, 0.2);
        let expected = (0.5 / (eta * eta)).exp() - 1.0;
        let actual = intkambe(-3, Complex::default(), eta).unwrap();
        assert!((actual - expected).norm() <= 1e-15 * expected.norm());
    }

    /// `n re(z) im(z) re(eta) im(eta): re im` of [`kambe_series_at_small_arguments`].
    const SMALL_ARGUMENTS: &str = "
        -11 1e-8 0.0 1.0 0.0: 0.1520560731298582 0.0
        -4 1e-12 1e-12 0.0 -1.4285714285714286: 6.468012052193277e-25 -0.09890802845085557
        -60 1e-6 0.0 0.8 0.0: 18842.496102322668 0.0";

    /// Far evanescent orders of the 1D lattice sums, `I_n(i X, i E)` with `E = -1/w`,
    /// where the gamma ladder of the series reaches `|d|` near `a = (X E)^2 / 2`, 84 to
    /// 288 here, and its seed leaves the normal range of `f64`: each value is finite and
    /// within its reported bound of a 40-digit quadrature, which grows like `e^(w^2)` with
    /// the cancellation of the series: four digits at `w = 5`, none at 8 and 12.
    #[test]
    fn kambe_series_at_far_evanescent_orders() {
        for (key, expected) in reference(FAR_EVANESCENT_ORDERS) {
            #[allow(clippy::cast_possible_truncation)] // Integral orders.
            let n = key[0] as i32;
            let (x, eta) = (Complex::new(key[1], key[2]), Complex::new(key[3], key[4]));
            let actual = intkambe(n, x, eta).unwrap();
            let (value, bound) = kambe_with_bound(n, x, eta);
            assert_eq!(value, actual, "I_{n}({x}, {eta})");
            let error = (actual - expected).norm();
            let tolerance = 1e-13 * expected.norm() + SUM_ROUNDING * f64::EPSILON * bound;
            assert!(
                finite(actual) && error <= tolerance,
                "I_{n}({x}, {eta}) = {actual}, expected {expected}, bound {bound}"
            );
            if eta.im == -0.2 {
                assert!(error <= 1e-4 * expected.norm(), "I_{n}({x}, {eta})");
            } else {
                // The bound admits no digits.
                assert!(
                    f64::EPSILON * bound >= expected.norm(),
                    "I_{n}({x}, {eta}): bound {bound}"
                );
            }
        }
    }

    /// `n re(z) im(z) re(eta) im(eta): re im` of
    /// [`kambe_series_at_far_evanescent_orders`].
    const FAR_EVANESCENT_ORDERS: &str = "
        1 0.0 65.0 0.0 -0.2: -2.0660582937881563e-46 0.0
        -3 0.0 65.0 0.0 -0.2: -1.2571235926483948e-43 0.0
        -1 0.0 72.75 0.0 -0.2: 2.1076570255627118e-54 0.0
        -5 0.0 72.75 0.0 -0.2: 1.2903462855893798e-51 0.0
        -3 0.0 74.0 0.0 -0.2: -1.2811250838191011e-54 0.0
        -9 0.0 74.0 0.0 -0.2: 1.9436884363619354e-50 0.0
        -5 0.0 100.0 0.0 -0.2: 8.455104242709188e-93 0.0
        1 0.0 100.0 0.0 -0.2: -5.497277311462082e-97 0.0
        -9 0.0 104.0 0.0 -0.125: 3.6443751390988474e-46 0.0
        -1 0.0 104.0 0.0 -0.125: 2.3262288907012024e-53 0.0
        1 0.0 116.4 0.0 -0.125: -1.4178141658789873e-64 0.0
        -3 0.0 116.4 0.0 -0.125: -5.65917325634384e-61 0.0
        -1 0.0 118.4 0.0 -0.125: 2.1799896234970922e-64 0.0
        -5 0.0 118.4 0.0 -0.125: 8.713921867089539e-61 0.0
        -3 0.0 160.0 0.0 -0.125: -3.292074055899649e-102 0.0
        -9 0.0 160.0 0.0 -0.125: 8.481758744830575e-97 0.0
        -5 0.0 156.0 0.0 -0.08333333333333333: 5.401514253857145e-66 0.0
        1 0.0 156.0 0.0 -0.08333333333333333: -2.0476729683120954e-72 0.0
        -9 0.0 174.60000000000002 0.0 -0.08333333333333333: 2.9610339101407296e-71 0.0
        -1 0.0 174.60000000000002 0.0 -0.08333333333333333: 7.549716659238215e-80 0.0
        1 0.0 177.60000000000002 0.0 -0.08333333333333333: -1.254520496883189e-83 0.0
        -3 0.0 177.60000000000002 0.0 -0.08333333333333333: -2.4885065995274332e-79 0.0
        -1 0.0 240.0 0.0 -0.08333333333333333: 2.8622298120865214e-121 0.0
        -5 0.0 240.0 0.0 -0.08333333333333333: 5.8465689416040285e-117 0.0";

    /// The Kambe integrals of the real-space terms at splits below every automatic one,
    /// where the closed forms and the recurrence lose up to 1.6e-3 (real), 3e6 ulps
    /// (rotated split) and 1e-8 (lossy `z`): within 32 ulps of 50-digit quadrature and
    /// within their reported bounds.
    #[test]
    fn small_split_integrals_keep_their_digits() {
        for (key, expected) in reference(SMALL_SPLITS) {
            #[allow(clippy::cast_possible_truncation)] // Integral orders.
            let n = key[0] as i32;
            let (z, eta) = (Complex::new(key[1], key[2]), Complex::new(key[3], key[4]));
            let [(value, bound)] = SmallSplitKambe::new(eta).get([n], z, 0.0);
            let error = (value - expected).norm();
            assert!(
                error <= 32.0 * f64::EPSILON * expected.norm() && error <= f64::EPSILON * bound,
                "I_{n}({z}, {eta}) = {value}, expected {expected}, bound {bound}"
            );
        }
    }

    /// `n re(z) im(z) re(eta) im(eta): re im` of [`small_split_integrals_keep_their_digits`].
    const SMALL_SPLITS: &str = "
        10 0.51 0.0 0.135 0.0: 1979290.8095660476 0.0
        14 1.19 0.0 0.135 0.0: 13164.26182468963 0.0
        0 0.21 0.0 0.137 0.0: 1013302395.488461 0.0
        2 0.17 0.0 0.135 0.0: 40717456.33548243 0.0
        13 0.65 0.2 0.17 0.0: -5355697.287078753 8848712.085330464
        8 1.59 0.13 0.166 -0.0513: 1.8142292945819555 -1.5315951220636088";

    /// Small `eta` makes `|c| = 1 / (2 |eta|^2)` large, and the series needs about `|c|`
    /// terms. Up to `|c| = 715` they stay representable: `I_-3(0, eta) = e^c - 1` holds
    /// at `|c| = 707`. Beyond, the terms and the integrals (about `e^|c|`) overflow, and
    /// each value must come back non-finite within a bounded number of terms (a watchdog).
    #[test]
    fn kambe_series_at_small_eta() {
        let eta = Complex::new(0.0266, 0.0);
        let expected = (0.5 / (eta * eta)).exp() - 1.0;
        let actual = intkambe(-3, Complex::default(), eta).unwrap();
        assert!((actual - expected).norm() <= 1e-14 * expected.norm());
        let cases = [
            (-1, 1.0, 1e-5),
            (-4, 1.0, 1e-8),
            (-3, 0.0, 1e-6),
            (3, 1.0, 1e-6),
        ];
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let values = cases.map(|(n, z, eta)| {
                intkambe(n, Complex::new(z, 0.0), Complex::new(eta, 0.0)).unwrap()
            });
            sender.send(values).unwrap();
        });
        let values = receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the Kambe series must end within a bounded number of terms");
        for ((n, z, eta), value) in cases.into_iter().zip(values) {
            assert!(!finite(value), "I_{n}({z}, {eta}) = {value}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn kambe_series_and_recurrence(
            n in -16_i32..=4,
            z in complex(3.0),
            (radius, phase) in (0.4_f64..3.0, -3.0_f64..1.5),
        ) {
            check_kambe_series(n, z, Complex::from_polar(radius, phase))?;
        }

        #[test]
        fn kambe_argument_derivatives_and_reflection(
            n in -8_i32..8, x in 0.5_f64..2.0, eta in 0.8_f64..1.8,
        ) {
            check_kambe(n, Complex::new(x, 0.1), Complex::new(eta, 0.03))?;
        }
    }

    /// The gamma series below the base pair continues the closed-form even base pair
    /// (a check of its branch sign), one pass of it over several orders gives each as
    /// its own pass does, and every order obeys the Kambe recurrence
    /// `I_n - (n+3) I_(n+2) + z^2 I_(n+4) = eta^(n+3) e^((1/eta^2 - z^2 eta^2)/2)`
    /// relative to its largest term.
    fn check_kambe_series(n: i32, z: Complex, eta: Complex) -> Result<(), TestCaseError> {
        prop_assume!(z.norm() > 1e-3);
        // The closed form cancels between its two erfc terms by up to a factor of
        // about 10 here, and the series by the ratio of its term magnitudes to its sum
        // (above 5000 at some complex eta); a wrong branch would miss by O(1). Both
        // orders come from one pass of the series.
        let pairs = kambe_series([-2, 0], z, eta)
            .into_iter()
            .zip(even_pair(z, eta));
        for (order, ((series, magnitude), (expected, _))) in [-2, 0].into_iter().zip(pairs) {
            prop_assert_close!(
                series,
                expected,
                1e-12 * expected.norm() + SUM_ROUNDING * f64::EPSILON * magnitude,
                "order {}",
                order
            );
        }
        if n % 2 != 0 && n < -3 {
            let orders = [n, -3, -1];
            let single = orders.map(|order| kambe_series([order], z, eta)[0].0);
            for ((series, magnitude), expected) in
                kambe_series(orders, z, eta).into_iter().zip(single)
            {
                prop_assert_close!(
                    series,
                    expected,
                    1e-13 * expected.norm() + SUM_ROUNDING * f64::EPSILON * magnitude
                );
            }
        }
        let [value, middle, upper] = [n, n + 2, n + 4].map(|order| kambe(order, z, eta));
        let endpoint = eta.powi(n + 3) * (0.5 * (1.0 / (eta * eta) - z * z * eta * eta)).exp();
        let terms = [value, f64::from(n + 3) * middle, z * z * upper, endpoint];
        let scale = terms.iter().map(|t| t.norm()).fold(0.0, f64::max);
        // Even orders from -2 up rest on the closed-form base pair (see above).
        let tolerance = if n % 2 == 0 && n + 4 >= -2 {
            1e-12
        } else {
            1e-13
        };
        prop_assert_close!(value - terms[1] + terms[2], endpoint, tolerance * scale);
        Ok(())
    }

    /// Kambe argument pullbacks match central differences in z and eta, and the
    /// integral obeys Schwarz reflection in both arguments.
    fn check_kambe(n: i32, z: Complex, eta: Complex) -> Result<(), TestCaseError> {
        let g = Complex::new(0.7, -0.2);
        let (_, residual) = intkambe_array(vec![n], [vec![z], vec![eta]]).unwrap();
        let gradient = residual.pullback(&[g]).unwrap();
        let delta = Complex::new(0.3, 0.4);
        let differences = [
            central(1e-5, |h| intkambe(n, z + h * delta, eta).unwrap()),
            central(1e-5, |h| intkambe(n, z, eta + h * delta).unwrap()),
        ];
        for (gradient, difference) in gradient.iter().zip(differences) {
            prop_assert_close!(
                (gradient[0].conj() * delta).re,
                (g.conj() * difference).re,
                3e-6 * (1.0 + difference.norm())
            );
        }
        let value = intkambe(n, z, eta).unwrap();
        let reflected = intkambe(n, z.conj(), eta.conj()).unwrap();
        prop_assert_close!(reflected, value.conj(), 1e-15 * value.norm());
        Ok(())
    }
}

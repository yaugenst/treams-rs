//! Reduced integrals `F_n(v, t)` of the reciprocal terms.
//!
//! Upstream: `_redincgamma` and `_redintkambe` of `lattice/_esum.pyx`. The series in `t`,
//! the continuation to other sheets and the rounding bounds are treams-rs extensions.

use std::f64::consts::TAU;

use super::{
    accuracy::Rounding,
    sheets::{Sheet, lower_side},
};
use crate::{
    Complex,
    numerics::Jet,
    special::{self, Bessel, EvenKambe, ScaledGammaLadder, log_factorial},
};

/// `F_n(v) = Gamma(n, v) / (-v)^n` (`2n = twice_n`), the `w = 0` limit of
/// [`reduced_kambe`].
///
/// # Branch
///
/// On the principal branches, or continued to the argument `sheet` of `v`, which callers
/// pass for `Im k >= 0`: there the outgoing sums take the phase `v^n / (-v)^n =
/// e^(-i pi n)` of the lower side of the real axis.
///
/// # Formula
///
/// A turn `delta` of the sheet adds `-delta v^-n / (-n)!` to integer `n <= 0`.
/// Half-integer orders change sign above the real axis and lose `2 Gamma(n) / (-v)^n`
/// after an odd number of turns (DLMF 8.2.10). That term solves the homogeneous degree
/// recurrence, so the continued `F_n` keep their derivative identities and recurrences.
pub(super) fn reduced_gamma(twice_n: i32, value: Complex, sheet: Option<Sheet>) -> Complex {
    gamma_limit(
        twice_n,
        value,
        sheet.map(|sheet| sheet.turns(lower_side(value))),
    )
}

/// [`reduced_gamma`] with the whole `turns` from the principal argument of `v` (on the
/// lower side of the real axis) to its sheet, found once for many orders.
fn gamma_limit(twice_n: i32, value: Complex, turns: Option<f64>) -> Complex {
    let z = lower_side(value);
    let gamma = special::upper_gamma(0.5 * f64::from(twice_n), z) / reduced_power(twice_n, z);
    on_sheet(twice_n, value, gamma, turns)
}

/// `(-z)^n` of [`reduced_gamma`], with the principal root for half-integer `n`.
fn reduced_power(twice_n: i32, z: Complex) -> Complex {
    let power = (-z).powi(twice_n.div_euclid(2));
    if twice_n % 2 == 0 {
        power
    } else {
        power * crate::numerics::complex_sqrt(-z)
    }
}

/// `z^n / (-z)^n` on the principal branches, for `z` with a signed imaginary part:
/// `(-1)^n`, or for half-integer `n` `(-1)^floor(n)` times `-i` on and below the real
/// axis and `i` above it.
fn principal_phase(twice_n: i32, z: Complex) -> Complex {
    let sign = if twice_n.div_euclid(2) % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    if twice_n % 2 == 0 {
        Complex::new(sign, 0.0)
    } else if z.im > 0.0 {
        Complex::new(0.0, sign)
    } else {
        Complex::new(0.0, -sign)
    }
}

/// `F_n(v)` of [`reduced_gamma`] from its value `gamma` on the principal branches,
/// continued by `turns` (see [`gamma_limit`]) to its sheet.
fn on_sheet(twice_n: i32, value: Complex, mut gamma: Complex, turns: Option<f64>) -> Complex {
    let Some(turns) = turns else {
        return gamma;
    };
    let z = lower_side(value);
    if twice_n % 2 != 0 {
        // Whole turns: an odd number has an odd half (none, without a `trunc` call,
        // on most sheets).
        if turns != 0.0 && (0.5 * turns).fract() != 0.0 {
            gamma -= 2.0 * libm::tgamma(0.5 * f64::from(twice_n)) / reduced_power(twice_n, z);
        }
        if z.im > 0.0 {
            gamma = -gamma;
        }
    } else if twice_n <= 0 && turns != 0.0 {
        let n = -twice_n / 2;
        gamma -= Complex::new(0.0, TAU * turns) * value.powi(n) * (-log_factorial(n)).exp();
    }
    gamma
}

/// The argument `x = sqrt(-2 v) w` (`Re x >= 0`) and split `eta = -i / w` of the Kambe
/// integrals with `F_n(v, w^2) = 2 I_(2n-1)(x, eta) w^2n`, for `Re w >= 0`.
fn kambe_arguments(value: Complex, w: Complex) -> (Complex, Complex) {
    let mut x = (-2.0 * lower_side(value) * w * w).sqrt();
    // The outgoing limit approaches the real x-axis from the upper half plane.
    if x.im == 0.0 {
        x.im = 1e-100;
    }
    (x, -Complex::i() / w)
}

/// `F_n(v, w^2)` of integer `n` (the odd Kambe orders of 1D spherical sums) from Kambe
/// integrals, with a first-order bound on its absolute rounding error in units of
/// epsilon.
///
/// # Branch
///
/// The Kambe series takes the principal `ln a` of its argument `a` (`= v` up to
/// rounding). Where the `sheet` of `v` ([`split_sheet`]) differs by `delta`, the result
/// gains `-delta x^-n J_n(x) w^2n` with `x^2 = -2 v w^2` (DLMF 8.2.10 in each term). That
/// term is an entire function of `v` and `w^2`, so the jets of the continued `F_n` stay
/// exact.
///
/// [`split_sheet`]: super::sheets::split_sheet
pub(super) fn reduced_kambe(
    twice_n: i32,
    value: Complex,
    w: Complex,
    sheet: Option<Sheet>,
) -> (Complex, f64) {
    let w = if w.re < 0.0 { -w } else { w };
    let (x, eta) = kambe_arguments(value, w);
    let (mut integral, mut bound) = special::kambe_with_bound(twice_n - 1, x, eta);
    // The change delta of ln v from the principal branch of the series to the sheet.
    let turns = sheet.map_or(0.0, |sheet| sheet.turns(special::kambe_argument(x, eta)));
    if turns != 0.0 {
        let delta = Complex::new(0.0, TAU * turns);
        // The integral I_(2n-1) = F_n / (2 w^2n) changes by -delta/2 x^-n J_n(x).
        let jump = -0.5 * delta * regular_bessel(twice_n / 2, x);
        integral += jump;
        bound += (8.0 + x.norm()) * jump.norm();
    }
    let power = w.powi(twice_n);
    (2.0 * integral * power, 2.0 * bound * power.norm())
}

/// `x^-n J_n(x)`, entire in `x^2`, or NaN where AMOS fails.
fn regular_bessel(n: i32, x: Complex) -> Complex {
    let order = n.unsigned_abs();
    let Ok(bessel) = special::bessel(f64::from(order), x, Bessel::J, false, 0) else {
        return Complex::new(f64::NAN, f64::NAN);
    };
    if n >= 0 {
        crate::numerics::ratio(bessel, x.powu(order))
    } else {
        // x^|n| J_-|n| = (-1)^n x^|n| J_|n|.
        let sign = if order.is_multiple_of(2) { 1.0 } else { -1.0 };
        sign * x.powu(order) * bessel
    }
}

/// Largest `|t| = |w|^2` at which [`Reduced`] sums every integer order of `F_n(v, t)`
/// from its series in `t`: the Kambe integrals take `eta = -i / w`, whose powers leave
/// the range of `f64` from about `|w| = 1e-50`, and the series terms alternate at most
/// like those of `e^-|t|`, losing a few ulps up to [`SERIES_REACH`].
const SERIES_T: f64 = 1e-2;

/// Largest `|t|` at which [`Reduced`] takes the series for the integer orders below the
/// base pair of the Kambe recurrence (`2n < -2`), which need a series of gamma functions
/// of their own each, while the series in `t` takes all orders from one ladder.
const SERIES_REACH: f64 = 1.0;

/// Largest `|t|` at which [`Reduced`] sums the half-integer orders from their series in
/// `t` (which loses up to `e^|t|` ulps) instead of the Kambe chain (whose correlated
/// errors the sums cancel less the smaller `|t|`).
///
/// Calibrated against 30-digit references: both keep 2D spherical and 1D cylindrical
/// sums within 1e-13 of `max(|S|, 1)` for `|t|` from 0.3 to 1; the chain alone loses
/// 2e-12 at `|t| = 1e-3` and the series alone 1e-12 at `|t| = 3`.
pub(crate) const HALF_INTEGER_SERIES_T: f64 = 1.0;

/// Terms of the series in `t` of [`Reduced`] at most; `|t| <=` [`SERIES_REACH`] stops
/// within 20.
const SERIES_TERMS: i32 = 64;

/// Reduced integrals `F_n(v, t)` of one reciprocal point, as jets.
///
/// # Formula
///
/// `F_n(v, t) = (-1)^n int_1^inf s^(n-1) exp(-v s - t / (2s)) ds`, entire in `t`, with
/// `dF_n/dv = F_(n+1)` and `dF_n/dt = F_(n-1) / 2`; `t = w^2` keeps them regular on the
/// lattice plane and axis. Callers request orders in decreasing steps, so each value is
/// evaluated once and then serves as a derivative of its neighbours.
///
/// # Paths
///
/// Near the plane or axis `F_n` is the series `sum_j (t/2)^j / j! F_(n-j)(v, 0)` of the
/// limits [`reduced_gamma`] on their sheet, which tends to the limit as `t -> 0`:
///
/// - half-integer `n` (2D spherical and 1D cylindrical sums) up to
///   [`HALF_INTEGER_SERIES_T`];
/// - integer `n` (1D spherical sums) up to [`SERIES_T`], and their low orders up to
///   [`SERIES_REACH`].
///
/// Farther off, the integer orders come from [`reduced_kambe`] with their rounding
/// bounds, and the half-integer ones from one chain of Kambe integrals ([`EvenKambe`]).
/// The errors of the chain are correlated across orders and cancel in the sums (taken
/// order by order, each to a few ulps, they lose 1.4e-12 of `max(|S|, 1)`). At small
/// splits the chain carries the rounding of its pair along the homogeneous solutions of
/// the recurrence instead, and each diffraction order takes whichever of the chain and
/// the single integrals bounds the sum more tightly ([`Self::prefers_single`]).
pub(super) struct Reduced<const N: usize> {
    v: Jet<N>,
    t: Jet<N>,
    w: Complex,
    /// The sheet of `v` for [`reduced_kambe`] and the chain, and the turns to it for the
    /// limits once they are needed.
    sheet: Option<Sheet>,
    turns: std::cell::OnceCell<Option<f64>>,
    /// Whether `t` carries a derivative.
    varies: bool,
    /// The two most recent `(2n, F_n, error bound of F_n)`.
    known: [Option<(i32, Complex, f64)>; 2],
    /// The limits `F_d(v, 0)` at `2d = top, top - 2, ..., next + 2`, from the first
    /// order requested down, with the ladder of `Gamma(d, v) / v^d` that continues them.
    limits: Vec<Complex>,
    ladder: Option<ScaledGammaLadder>,
    top: i32,
    next: i32,
    /// The chain of the half-integer orders, with the factor that turns its Kambe
    /// integrals into `F_n / w^2n`.
    even_kambe: Option<(EvenKambe, f64)>,
    /// The weights of `F_(1/2 - j)` in the value of the sum, recorded ([`Self::weigh`])
    /// while the chain serves them.
    weights: Vec<Complex>,
    /// Whether the half-integer orders take the Kambe integrals of the chain one by one
    /// ([`Self::prefers_single`]), and those it evaluated, `I_0, I_-2, ...`.
    single: bool,
    singles: Vec<Complex>,
}

impl<const N: usize> Reduced<N> {
    pub(super) fn new(v: Jet<N>, t: Jet<N>, sheet: Option<Sheet>) -> Self {
        Self {
            v,
            t,
            w: t.value.sqrt(),
            sheet,
            turns: std::cell::OnceCell::new(),
            varies: t.derivative.iter().any(|&d| d != Complex::default()),
            known: [None; 2],
            limits: Vec::new(),
            ladder: None,
            top: 0,
            next: 0,
            even_kambe: None,
            weights: Vec::new(),
            single: false,
            singles: Vec::new(),
        }
    }

    /// Records `weight`, the factor of the value of `F_n` in the sum (`2n = twice_n`),
    /// for [`Self::prefers_single`] where the chain serves it.
    pub(super) fn weigh(&mut self, twice_n: i32, weight: Complex) {
        if self.single || self.t.value.norm() <= HALF_INTEGER_SERIES_T || twice_n % 2 == 0 {
            return;
        }
        let Ok(j) = usize::try_from((1 - twice_n) / 2) else {
            return;
        };
        if self.weights.len() <= j {
            self.weights.resize(j + 1, Complex::default());
        }
        self.weights[j] += weight;
    }

    /// Whether the half-integer orders bound the sum they were weighed in
    /// ([`Self::weigh`]) more tightly one by one ([`EvenKambe::single`]) than from the
    /// chain ([`EvenKambe::sum_bounds`]), which they then take from here on. The single
    /// integrals are evaluated only where the chain's bound exceeds the moduli of the
    /// terms.
    pub(super) fn prefers_single(&mut self) -> bool {
        let Some((even_kambe, factor)) = &self.even_kambe else {
            return false;
        };
        if self.single || self.weights.is_empty() {
            return false;
        }
        let w = if self.w.re < 0.0 { -self.w } else { self.w };
        let mut power = *factor * w;
        let square = w * w;
        let weights: Vec<Complex> = self
            .weights
            .iter()
            .map(|&weight| {
                let scaled = weight * power;
                power /= square;
                scaled
            })
            .collect();
        let (recurrence, moduli) = even_kambe.sum_bounds(&weights);
        // A bound that overflowed to NaN keeps the chain.
        if recurrence.is_nan() || recurrence <= moduli {
            return false;
        }
        let singles: Vec<(Complex, f64)> = (0..)
            .zip(&weights)
            .map(|(j, _)| even_kambe.single(-2 * j))
            .collect();
        let single: f64 = weights
            .iter()
            .zip(&singles)
            .map(|(weight, (_, bound))| weight.l1_norm() * bound)
            .sum();
        // So does a bound of the single integrals that overflowed.
        if single.is_nan() || single >= recurrence {
            return false;
        }
        self.single = true;
        self.singles = singles.into_iter().map(|(value, _)| value).collect();
        self.known = [None; 2];
        true
    }

    /// Half-integer `F_n = 2 I_(2n-1)(x, eta) w^2n` from the chain of the even Kambe
    /// orders, whose series takes the root `sqrt(v) = x eta / sqrt 2` (`Re x >= 0`). Where
    /// that is not the root `|v|^(1/2) e^(i sheet / 2)` of the sheet the chain takes the
    /// split `-eta`: the even `I_(2n-1)` are odd in `eta` on a fixed root.
    pub(super) fn by_even_recurrence(&mut self, twice_n: i32) -> Complex {
        let w = if self.w.re < 0.0 { -self.w } else { self.w };
        let (value, sheet) = (self.v.value, self.sheet);
        let (even_kambe, factor) = self.even_kambe.get_or_insert_with(|| {
            let (x, eta) = kambe_arguments(value, w);
            if sheet.is_some_and(|sheet| sheet.opposes(x * eta, value)) {
                (EvenKambe::new(x, -eta), -2.0)
            } else {
                (EvenKambe::new(x, eta), 2.0)
            }
        });
        *factor * even_kambe.get(twice_n - 1) * w.powi(twice_n)
    }

    /// The turns from the principal argument of `v` to its sheet, for [`gamma_limit`].
    fn turns(&self) -> Option<f64> {
        *self.turns.get_or_init(|| {
            self.sheet
                .map(|sheet| sheet.turns(lower_side(self.v.value)))
        })
    }

    fn value(&mut self, twice_n: i32) -> (Complex, f64) {
        if let Some(&(_, value, bound)) = self.known.iter().flatten().find(|(n, ..)| *n == twice_n)
        {
            return (value, bound);
        }
        let (value, bound) = if self.t.value == Complex::default() {
            // The limit itself: on the lattice plane or axis, where few orders take a
            // gamma function each.
            (gamma_limit(twice_n, self.v.value, self.turns()), 0.0)
        } else if twice_n % 2 != 0 {
            // Half-integer orders keep their relative accuracy and carry no bound.
            if self.t.value.norm() <= HALF_INTEGER_SERIES_T {
                (self.series(twice_n).0, 0.0)
            } else if let (true, Some((even_kambe, factor))) = (self.single, &self.even_kambe) {
                let w = if self.w.re < 0.0 { -self.w } else { self.w };
                let integral = usize::try_from((1 - twice_n) / 2)
                    .ok()
                    .and_then(|j| self.singles.get(j).copied())
                    .unwrap_or_else(|| even_kambe.single(twice_n - 1).0);
                (*factor * integral * w.powi(twice_n), 0.0)
            } else {
                (self.by_even_recurrence(twice_n), 0.0)
            }
        } else if self.t.value.norm() <= if twice_n < -2 { SERIES_REACH } else { SERIES_T } {
            self.series(twice_n)
        } else {
            reduced_kambe(twice_n, self.v.value, self.w, self.sheet)
        };
        self.known = [Some((twice_n, value, bound)), self.known[0]];
        (value, bound)
    }

    /// `F_n(v, t)` from its series in `t`, with the sum of the moduli of its terms as its
    /// rounding bound in units of epsilon. The limits `F_(n-j)(v, 0)` stay within a small
    /// factor of each other, so once the coefficients `(t/2)^j / j!` halve per term, the
    /// first term below a quarter ulp of the moduli bounds the rest.
    pub(super) fn series(&mut self, twice_n: i32) -> (Complex, f64) {
        let half = 0.5 * self.t.value;
        let (mut sum, mut magnitude) = (Complex::default(), 0.0);
        let mut coefficient = Complex::new(1.0, 0.0);
        for j in 0..SERIES_TERMS {
            if j > 0 {
                coefficient *= half / f64::from(j);
                if coefficient == Complex::default() {
                    break;
                }
            }
            let term = coefficient * self.limit(twice_n - 2 * j);
            sum += term;
            magnitude += term.l1_norm();
            if j > 0
                && f64::from(j) >= 2.0 * half.norm()
                && term.l1_norm() <= 0.25 * f64::EPSILON * magnitude
            {
                break;
            }
        }
        (sum, magnitude)
    }

    /// `F_d(v, 0)` of [`reduced_gamma`] for the series, once per order from the first one
    /// requested down: `Gamma(d, v) / v^d` of a [`ScaledGammaLadder`] times `v^d / (-v)^d`.
    fn limit(&mut self, twice_d: i32) -> Complex {
        if self.limits.is_empty() {
            (self.top, self.next) = (twice_d, twice_d);
        }
        let Ok(index) = usize::try_from((self.top - twice_d) / 2) else {
            return gamma_limit(twice_d, self.v.value, self.turns());
        };
        let z = lower_side(self.v.value);
        let (top, turns) = (self.top, self.turns());
        let ladder = self
            .ladder
            .get_or_insert_with(|| ScaledGammaLadder::new(0.5 * f64::from(top), z, false));
        while self.next >= twice_d {
            let principal = ladder.next_lower() * principal_phase(self.next, z);
            self.limits
                .push(on_sheet(self.next, self.v.value, principal, turns));
            self.next -= 2;
        }
        self.limits[index]
    }

    /// `F_n` as a jet, with the error bounds of its value and derivatives from those of
    /// the orders [`reduced_kambe`] returns.
    pub(super) fn get(&mut self, twice_n: i32) -> (Jet<N>, Rounding<N>) {
        if N == 0 {
            let (value, bound) = self.value(twice_n);
            return (Jet::constant(value), Rounding::of(bound));
        }
        let (dz, dz_bound) = self.value(twice_n + 2);
        let (value, bound) = self.value(twice_n);
        let (z, t) = (self.v.derivative, self.t.derivative);
        // Only odd orders carry bounds; the others skip the moduli.
        let (derivative, derivative_bound) = if self.varies {
            let (dt, dt_bound) = self.value(twice_n - 2);
            let dt = 0.5 * dt;
            (
                std::array::from_fn(|i| dz * z[i] + dt * t[i]),
                if dz_bound == 0.0 && dt_bound == 0.0 {
                    [0.0; N]
                } else {
                    std::array::from_fn(|i| {
                        dz_bound * z[i].l1_norm() + 0.5 * dt_bound * t[i].l1_norm()
                    })
                },
            )
        } else if dz_bound == 0.0 {
            (z.map(|z| dz * z), [0.0; N])
        } else {
            (z.map(|z| dz * z), z.map(|z| dz_bound * z.l1_norm()))
        };
        (
            Jet { value, derivative },
            Rounding {
                value: bound,
                derivative: derivative_bound,
            },
        )
    }
}

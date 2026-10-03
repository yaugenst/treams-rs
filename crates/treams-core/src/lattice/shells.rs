//! Sums over cube shells and when they stop.
//!
//! Upstream: the shell loops of `realsum*` and `recsum*` in `lattice/_esum.pyx`. The
//! relative tolerance, the shell limits that grow with the terms and the compensated
//! sums are treams-rs extensions.

use std::f64::consts::PI;

use super::{
    accuracy::{Rounding, TERM_ULPS, below_automatic},
    cell::{BlochLattice, Reduction},
    geometry::visit_cube,
    wave::Family,
};
use crate::{Complex, Error, Result, numerics::Jet};

/// `Jet::norm`, the largest modulus of the value and derivatives, from one square root
/// of the largest squared modulus instead of a `hypot` per component; `Jet::norm` when
/// the squares overflow.
pub(super) fn largest_modulus<const N: usize>(jet: &Jet<N>) -> f64 {
    let square = jet
        .derivative
        .iter()
        .fold(jet.value.norm_sqr(), |square, d| square.max(d.norm_sqr()));
    if square.is_finite() {
        square.sqrt()
    } else {
        jet.norm()
    }
}

/// Largest part of the scale of a shell sum that each of its last two shells may add
/// before the sum stops.
pub(crate) const SHELL_TOLERANCE: f64 = 2e-13;

/// One Ewald part summed over cube shells of integer points ([`Shells::sum`]), which
/// [`Shells::extend`] continues. Below every automatic split ([`below_automatic`]), where
/// the parts cancel, a part that stops at [`SHELL_TOLERANCE`] of its own sum leaves that
/// much in the complete sum, so both parts continue to [`SHELL_TOLERANCE`] of the
/// complete sum, and each part keeps the rounding errors of its additions
/// ([`add_exactly`]): next to a lattice point at a zero Bloch vector millions of
/// real-space terms add in phase far below an ulp of it.
///
/// Calibrated: continuing the parts this way at every split costs the other splits 5%
/// and gains them 1e-14 of `max(|S|, 1)`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Shells<const N: usize> {
    /// The sum, or with `error` its leading part.
    pub(super) sum: Jet<N>,
    /// The rounding errors of the additions to `sum` since the last shell, where they
    /// are kept (`cancelling`); each shell adds them to `sum`.
    error: Jet<N>,
    /// Whether the part may cancel against the other, below every automatic split: it
    /// then keeps the rounding errors of its additions and counts `bound`.
    cancelling: bool,
    /// The rounding bounds of the value and each derivative that its terms predict, in
    /// units of epsilon: [`TERM_ULPS`] ulps of the moduli of each term, plus the bounds
    /// the terms add beyond their relative accuracy.
    pub(super) bound: Rounding<N>,
    /// Shells summed so far.
    count: i32,
    /// Shells to sum before the convergence test may stop the sum ([`peak_shells`]).
    floor: i32,
    /// The largest moduli of the value and derivatives that the last two shells added.
    last: [f64; 2],
    /// The same of what each of the last two shells added in all, where their terms
    /// cancel.
    net: [f64; 2],
    /// Whether the sum stopped at its shell limit on `net` instead of `last` (see
    /// [`Shells::sum`]).
    pub(super) settled: bool,
}

impl<const N: usize> Shells<N> {
    /// Sum `term` over cube shells until two consecutive shells add less than
    /// [`SHELL_TOLERANCE`] times `max(|sum|, 1)`, but not before `floor` shells
    /// ([`peak_shells`]), failing after `maximum` shells; where `cancelling`, keep the
    /// rounding errors of the additions and count their bounds ([`Self::bound`]). With
    /// `settle` a sum at the limit whose last two shells add less than that in all
    /// ([`Self::settled`]) is returned for the caller to verify: at `|k eta|` far below the
    /// automatic split the real-space terms reach beyond the limit while their Bloch
    /// phases cancel them within each shell.
    pub(super) fn sum(
        dim: usize,
        [maximum, floor]: [i32; 2],
        term: &mut impl FnMut([i64; 3], &mut Rounding<N>) -> Result<Jet<N>>,
        settle: bool,
        cancelling: bool,
    ) -> Result<Self> {
        let mut shells = Self {
            sum: Jet::default(),
            error: Jet::default(),
            cancelling,
            bound: Rounding::default(),
            count: 0,
            floor,
            last: [f64::INFINITY; 2],
            net: [f64::INFINITY; 2],
            settled: false,
        };
        if shells.extend(dim, maximum, term, None)? {
            return Ok(shells);
        }
        let limit = SHELL_TOLERANCE * shells.sum.norm().max(1.0);
        if settle && shells.net.iter().all(|&net| net <= limit) {
            shells.settled = true;
            return Ok(shells);
        }
        Err(not_converged())
    }

    /// The moduli of the last two shells, or for a sum that settled what they added in
    /// all: a bound on what the shells beyond would add.
    pub(super) fn tail(&self) -> f64 {
        if self.settled {
            self.net.iter().sum()
        } else {
            self.last.iter().sum()
        }
    }

    /// Adds `value` to a cancelling sum with the rounding error of that addition, and
    /// counts [`TERM_ULPS`] of its moduli in [`Self::bound`]. Kept out of line, like
    /// [`small_split_kambe`], for the sums at other splits.
    ///
    /// [`small_split_kambe`]: super::real::small_split_kambe
    #[inline(never)]
    fn add_compensated(&mut self, value: &Jet<N>) {
        add_jet_exactly(&mut self.sum, &mut self.error, value);
        self.bound += Rounding::relative(value, TERM_ULPS);
    }

    /// Add shells until the last two add less than [`SHELL_TOLERANCE`] times `scale`, or
    /// without one each less than that of `max(|sum|, 1)` as it was then, from
    /// [`Self::floor`] shells on, and whether they did within `maximum` shells.
    pub(super) fn extend(
        &mut self,
        dim: usize,
        maximum: i32,
        term: &mut impl FnMut([i64; 3], &mut Rounding<N>) -> Result<Jet<N>>,
        scale: Option<f64>,
    ) -> Result<bool> {
        let tolerance =
            |sum: &Jet<N>| SHELL_TOLERANCE * scale.unwrap_or_else(|| sum.norm().max(1.0));
        let limit = tolerance(&self.sum);
        let mut quiet = self.last.iter().rev().take_while(|&&m| m <= limit).count();
        while quiet < 2 || self.count < self.floor {
            if self.count >= maximum {
                return Ok(false);
            }
            let mut shell = 0.0;
            let mut net = Jet::default();
            visit_cube(dim, i64::from(self.count), true, |point| {
                let value = term(point, &mut self.bound)?;
                if !value.finite() {
                    return Err(Error::NonFinite(
                        "non-finite Ewald summand; change the split parameter or reduce the order"
                            .into(),
                    ));
                }
                if self.cancelling {
                    self.add_compensated(&value);
                } else {
                    self.sum += value;
                }
                net += value;
                shell += largest_modulus(&value);
                Ok(())
            })?;
            if self.cancelling {
                let error = std::mem::take(&mut self.error);
                add_jet_exactly(&mut self.sum, &mut self.error, &error);
            }
            self.count += 1;
            self.last = [self.last[1], shell];
            self.net = [self.net[1], largest_modulus(&net)];
            quiet = if shell <= tolerance(&self.sum) {
                quiet + 1
            } else {
                0
            };
        }
        Ok(true)
    }
}

/// Adds `x` to `sum` and the rounding error of that addition, which Knuth's two-sum
/// finds exactly, to `error`.
fn add_exactly(sum: &mut f64, error: &mut f64, x: f64) {
    let total = *sum + x;
    let back = total - *sum;
    *error += (*sum - (total - back)) + (x - back);
    *sum = total;
}

/// [`add_exactly`] for the real and imaginary parts of a jet's value and derivatives.
fn add_jet_exactly<const N: usize>(sum: &mut Jet<N>, error: &mut Jet<N>, x: &Jet<N>) {
    let add = |sum: &mut Complex, error: &mut Complex, x: Complex| {
        add_exactly(&mut sum.re, &mut error.re, x.re);
        add_exactly(&mut sum.im, &mut error.im, x.im);
    };
    add(&mut sum.value, &mut error.value, x.value);
    for ((sum, error), &x) in sum
        .derivative
        .iter_mut()
        .zip(&mut error.derivative)
        .zip(&x.derivative)
    {
        add(sum, error, x);
    }
}

/// The message of a shell sum that does not converge within its limit.
pub(super) const NOT_CONVERGED: &str = "Ewald sum did not converge within the shell limit";

/// The failure of a shell sum that does not converge within its limit.
pub(super) fn not_converged() -> Error {
    Error::NotConverged(NOT_CONVERGED.into())
}

/// Whether `error` is the failure of [`not_converged`] rather than one of the
/// cancellation failures that share its variant.
pub(super) fn is_shell_limit(error: &Error) -> bool {
    matches!(error, Error::NotConverged(text) if text == NOT_CONVERGED)
}

/// The shell limits of [`real_shells`] and [`reciprocal_shells`] reach every term whose
/// Gaussian factor exceeds `e^-TERM_DECAY`.
const TERM_DECAY: f64 = 50.0;

/// Cube shells a sum may visit before it fails: 200, 32 and 16 in one, two and three
/// dimensions.
const fn shells(dim: usize) -> i32 {
    match dim {
        1 => 200,
        2 => 32,
        _ => 16,
    }
}

/// Cube shells the reciprocal sum may visit, as `[fail, extend]`.
///
/// # Formula
///
/// The terms fall like `exp(-Re(1 / (k eta)^2) |q + G|^2 / 2)`, below `exp(-50)` beyond
/// `|q + G| = sqrt(100 / Re(1 / (k eta)^2))`. That is `10 |k eta|` for real `k eta`,
/// within `|k| sqrt(1 + 100 |eta|^2)`, and grows as `k eta` turns off the real axis.
///
/// # Limits
///
/// - A sum fails once it passes `fail` shells, which reach the larger of the two radii
///   plus `|q|`.
/// - At small splits a converged sum continues toward the tolerance of the complete sum
///   and keeps what it leaves out as its tail. It stops at `extend` shells, which reach
///   `|k| sqrt(1 + 100 |eta|^2) + |q|` alone, so the larger radius changes no sum that
///   converges within the smaller one.
/// - A 1D spherical sum fails at `extend` shells and takes its spectral series instead.
///
/// At large `|k| L` either radius needs more than the fixed limit.
pub(super) fn reciprocal_shells(
    lattice: &BlochLattice,
    k: Complex,
    eta: Complex,
    kpar: [f64; 3],
) -> [i32; 2] {
    let within = |radius| {
        shells_within(
            lattice.dim,
            radius,
            &lattice.direct,
            lattice.reciprocal_reduction.as_ref(),
        )
    };
    let split = k * eta;
    let decay = (split * split).inv().re;
    let radius = k.norm() * eta.norm_sqr().mul_add(2.0 * TERM_DECAY, 1.0).sqrt();
    let extend = within(radius + norm(kpar));
    // Off the real axis the terms fall slower: 0.27 times as fast with `k eta` at 37 degrees.
    let turned = if decay > 0.0 {
        (2.0 * TERM_DECAY / decay).sqrt()
    } else {
        0.0
    };
    if turned > radius {
        [within(turned + norm(kpar)), extend]
    } else {
        [extend, extend]
    }
}

/// Cube shells the real-space sum may visit.
///
/// # Formula
///
/// The terms fall like `exp(Re(1 / (2 eta^2)) - Re((k eta)^2) |R|^2 / 2)` from the
/// nearest ones, below `exp(-50)` beyond `|R| = sqrt(2 (50 + Re(1 / (2 eta^2))) /
/// Re((k eta)^2))`. Splits with `|k eta|` well below the automatic one need more than the
/// fixed limit to reach it.
///
/// # Why
///
/// Where `(k eta)^2` lies more than 45 degrees off the real axis
/// (`Re((k eta)^2) < |Im((k eta)^2)|`), the phase of the terms turns faster than they
/// fall, and their far shells cancel. There only 1D and 2D sums at small splits, which
/// check the [`cancellation`] of their parts, take the grown limit.
///
/// Calibrated: with the grown limit, 3D sums take up to 20 s, and a 1D cylindrical sum
/// above the automatic splits comes back 2e-5 of `max(|S|, 1)` off.
///
/// [`cancellation`]: super::accuracy::cancellation
pub(super) fn real_shells(lattice: &BlochLattice, k: Complex, eta: Complex) -> i32 {
    let split = k * eta;
    let square = split * split;
    let decay = square.re;
    let turned = decay < square.im.abs();
    if decay > 0.0 && (!turned || (lattice.dim < 3 && below_automatic(eta))) {
        let growth = (0.5 / (eta * eta)).re.max(0.0);
        let radius = (2.0 * (TERM_DECAY + growth) / decay).sqrt();
        shells_within(
            lattice.dim,
            radius,
            &lattice.reciprocal,
            lattice.direct_reduction.as_ref(),
        )
    } else {
        shells(lattice.dim)
    }
}

/// The Euclidean norm of a lattice-frame vector.
pub(super) fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Cube shells of a sum over a lattice, reduced by `reduction`, whose dual rows are
/// `dual` (`a_i . b_j = 2 pi delta_ij`): [`shells`], or where reaching the lattice
/// vectors within `radius` needs more, that many plus two, up to four times the base.
/// Index `i` of a lattice vector `x` is `c_i . x / 2 pi` for the dual rows `c` of the
/// reduced basis.
#[allow(clippy::cast_possible_truncation)] // Bounded by four times the base limit.
fn shells_within(
    dim: usize,
    radius: f64,
    dual: &[[f64; 3]; 3],
    reduction: Option<&Reduction>,
) -> i32 {
    let base = shells(dim);
    let needed = largest_index(dim, radius, dual, reduction).ceil() + 2.0;
    if needed > f64::from(base) {
        needed.min(f64::from(4 * base)) as i32
    } else {
        base
    }
}

/// The largest shell index of the lattice vectors within `radius` of a lattice reduced
/// by `reduction`, whose dual rows are `dual` (see [`shells_within`]).
fn largest_index(
    dim: usize,
    radius: f64,
    dual: &[[f64; 3]; 3],
    reduction: Option<&Reduction>,
) -> f64 {
    let spacing = (0..dim)
        .map(|i| {
            norm(reduction.map_or(dual[i], |reduction| {
                std::array::from_fn(|j| (0..dim).map(|k| reduction.dual[i][k] * dual[k][j]).sum())
            }))
        })
        .fold(0.0, f64::max);
    radius * spacing / (2.0 * PI)
}

/// Cube shells that an Ewald part of `wave` on a lattice of dimension `dim` visits
/// before its convergence test may stop it ([`Shells::sum`]).
///
/// These are the shells with a lattice vector within `offset` (the reduced shift or Bloch
/// vector, the origin of the part) plus the radius where the moduli of its far terms
/// peak, on a lattice reduced by `reduction` with the dual rows `dual`. `0` where the
/// terms do not fall (`decay <= 0`): both parts diverge there.
///
/// # Formula
///
/// Far from the origin each term is the Gaussian `exp(-decay x^2 / 2)` of its distance
/// `x` times a polynomial of degree `l` (`l - |m|` along a chain, `|m|` for cylindrical
/// waves) in real and reciprocal space, less 2 for the endpoint of its Kambe integral or
/// `1 / v`. A shell holds about `x^(dim - 1)` terms, and jets up to `x^2` more, so the
/// moduli of a shell peak at `x^2 = power / decay` and fall monotonically beyond.
///
/// # Why
///
/// Before the peak the moduli can fall and rise again, as the real-space terms of lossy
/// `k` at splits turned close to 90 degrees and the reciprocal terms of high degree do;
/// without the floor such sums stop on their first shells.
#[allow(clippy::cast_possible_truncation)] // Clamped to the range of i32.
pub(super) fn peak_shells(
    wave: Family,
    dim: usize,
    jet: bool,
    decay: f64,
    offset: f64,
    (dual, reduction): (&[[f64; 3]; 3], Option<&Reduction>),
) -> i32 {
    // Terms that do not fall have no peak, nor does a split that makes `decay` NaN.
    if decay.is_nan() || decay <= 0.0 {
        return 0;
    }
    let degree = match wave {
        Family::Spherical { l, m } if dim == 1 => l - m.abs(),
        Family::Spherical { l, .. } => l,
        Family::Cylindrical { m } => m.abs(),
    };
    let derivatives = if jet { 2 } else { 0 };
    let power = degree - 2 + i32::try_from(dim).unwrap_or(3) - 1 + derivatives;
    if power <= 0 {
        return 0;
    }
    let radius = (f64::from(power) / decay).sqrt() + offset;
    let shells = largest_index(dim, radius, dual, reduction).ceil() + 1.0;
    shells.min(f64::from(i32::MAX)) as i32
}

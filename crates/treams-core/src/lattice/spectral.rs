//! The spectral series of 1D spherical sums off the axis, and when to take it.
//!
//! treams-rs extension: treams has only the Ewald sum.

use std::f64::consts::PI;

use super::{
    accuracy::{MAX_LOSS, Rounding, derivative_scale},
    reciprocal::THRESHOLD_DISTANCE,
    shells::largest_modulus,
    wave::azimuthal,
};
use crate::{
    Complex, Error, Result,
    numerics::Jet,
    special::{self, harmonic_normalization},
};

/// Diffraction orders the spectral series of a 1D spherical sum may take on each side
/// of zero before it is not used.
const SERIES_SHELLS: f64 = 512.0;

/// [`spectral_sw1d`] is used only where its terms fall by `e^-SERIES_DECAY` past their
/// peak within [`SERIES_SHELLS`] orders.
const SERIES_DECAY: f64 = 45.0;

/// Part of the summed moduli below which a shell of [`spectral_sw1d`] counts as quiet;
/// the series stops after two quiet shells past the peak.
const QUIET_SHELL: f64 = 1e-18;

/// The spectral series of a 1D spherical sum along z off the axis, with first-order
/// bounds on its rounding errors in units of epsilon.
///
/// # Formula
///
/// By Poisson's formula over the chain,
/// `S = (-i)^l i^|m| (-1)^m N_lm pi / (k a) ((x ± i y) / (k rho^2))^|m|
/// sum_q F_l^m(q / k) u_q^|m| H_|m|(u_q) exp(-i q z)` over the diffraction orders
/// `q = kpar + g reciprocal`, with `u_q = k_q rho`, `k_q = sqrt(k^2 - q^2)` and the
/// Legendre factor `F_l^m = P_l^m / sin^|m|` continued with `sin = k_q / k`.
///
/// # Branch
///
/// `Im k_q >= 0`: the root of the Ewald reciprocal sums at every split (see
/// [`split_sheet`]).
///
/// # Why
///
/// Evanescent orders fall like `(|q| / |k|)^l exp(-|q| rho)`, so the series converges
/// fast off the axis, where the Ewald sum cancels. The value stops on its own moduli
/// and the derivatives on the largest of all, both at [`QUIET_SHELL`] of them, so
/// value-only and jet evaluations return the same value and bound.
///
/// `None` on the axis, for `Re k <= 0`, next to a diffraction threshold, beyond
/// [`SERIES_SHELLS`] orders on either side or at a non-finite term.
///
/// [`split_sheet`]: super::sheets::split_sheet
pub(super) fn spectral_sw1d<const N: usize>(
    (l, m): (i32, i32),
    k: Jet<N>,
    r: [Jet<N>; 3],
    kpar: Jet<N>,
    reciprocal: Jet<N>,
    period: Jet<N>,
) -> Option<(Jet<N>, Rounding<N>)> {
    let rho = (r[0] * r[0] + r[1] * r[1]).sqrt();
    let spacing = reciprocal.value.norm();
    if rho.value.re <= 0.0 || k.value.re <= 0.0 || spacing <= 0.0 {
        return None;
    }
    let order = m.unsigned_abs();
    // Terms fall monotonically once |q| exceeds `peak`; the orders to reach e^-45 past it.
    let peak = k.value.norm() + f64::from(l + 1) / rho.value.re;
    let reach = kpar.value.norm() + peak + SERIES_DECAY / rho.value.re;
    if reach / spacing > SERIES_SHELLS {
        return None;
    }
    let i = Complex::i();
    let mut sum = Jet::default();
    let mut bound = Rounding::default();
    // Summed moduli and consecutive quiet shells of the value and of every component,
    // and the value with its bound once its own terms are quiet.
    let (mut magnitude, mut quiet) = ([0.0; 2], [0; 2]);
    let mut value = None;
    for shell in 0_i32.. {
        let mut shell_magnitude = [0.0; 2];
        for g in [shell, -shell]
            .into_iter()
            .take(if shell == 0 { 1 } else { 2 })
        {
            let q = kpar + f64::from(g) * reciprocal;
            let square = k * k - q * q;
            let mut root = square.value;
            if root.im == 0.0 {
                // An evanescent order of real k takes the +0 side.
                root.im = 0.0;
            }
            // Next to a diffraction threshold the Ewald sum reports it instead.
            if root.norm() <= THRESHOLD_DISTANCE * (k.value * k.value).norm() {
                return None;
            }
            let mut root = crate::numerics::complex_sqrt(root);
            if root.im < 0.0 {
                root = -root;
            }
            let argument = square.chain(root, 0.5 / root) * rho;
            // u^|m| H_|m|(u), whose derivative u^|m| H_(|m|-1)(u) the product rule
            // would form by cancellation next to thresholds.
            let [below, hankel] = special::hankel_below(order, argument.value).ok()?;
            let power = argument.value.powu(order);
            let radial = argument.chain(power * hankel, power * below);
            let axial = q * r[2];
            let term = special::legendre_factor(l, m, q / k) * radial * (-i * axial).exp();
            if !term.finite() {
                return None;
            }
            // The Hankel function and the phase lose |argument| and |q z| ulps, the
            // Legendre recurrence about l.
            let ulps = 8.0 + f64::from(l) + argument.value.norm() + axial.value.norm();
            bound += Rounding::relative(&term, ulps);
            sum += term;
            shell_magnitude[0] += term.value.norm();
            if N > 0 {
                shell_magnitude[1] += largest_modulus(&term);
            }
        }
        let past = f64::from(shell) * spacing - kpar.value.norm() > peak;
        for ((total, count), added) in magnitude.iter_mut().zip(&mut quiet).zip(shell_magnitude) {
            *total += added;
            *count = if past && added <= QUIET_SHELL * *total {
                *count + 1
            } else {
                0
            };
        }
        if value.is_none() && quiet[0] >= 2 {
            value = Some((sum.value, bound.value));
        }
        if let Some((complete, complete_bound)) = value
            && (N == 0 || quiet[1] >= 2)
        {
            sum.value = complete;
            bound.value = complete_bound;
            break;
        }
        if f64::from(shell) > SERIES_SHELLS {
            return None;
        }
    }
    let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
    let order = m.abs();
    let outer = (-i).powi(l) * i.powi(order) * (sign * harmonic_normalization(l, m) * PI)
        / (k * period)
        * (azimuthal(r[0], r[1], m) / (k * rho * rho)).powi(order);
    Some((outer * sum, bound.times(&outer)))
}

/// The `w = |k| rho |eta|` from which a 1D spherical sum tries its spectral series
/// before the Ewald sum, whose reciprocal part then cancels by `e^(w^2)` or more.
pub(super) const SPECTRAL_FIRST_W: f64 = 2.5;

/// Rounding error of the real-space part of a 1D spherical sum relative to its moduli,
/// in ulps: its Kambe integrals of even order keep their relative accuracy.
pub(super) const REAL_ULPS: f64 = 64.0;

/// Relative rounding error, on the scale of `max(|S|, 1)`, from which a component of
/// a 1D spherical Ewald sum is taken from its spectral series instead, if that bounds
/// it more tightly. Derivatives count on the scale `|dS| + max(|S|, 1) s` with
/// `s = |k|` for length derivatives and `1 / |k|` for inverse-length ones.
pub(super) const SPECTRAL_SWITCH_LOSS: f64 = 1e-12;

/// A 1D spherical Ewald sum `ewald` with the rounding bounds `rounding` of its components
/// (the value's with the predicted loss of the [`cancellation`] of its parts), completed
/// from the spectral series: each component whose bound exceeds [`SPECTRAL_SWITCH_LOSS`]
/// comes from the series where that bounds it more tightly, and a failed sum is replaced
/// by the series. The choice of the value reads only values and value bounds, as in
/// value-only evaluations. Fails where the value's bound still exceeds [`MAX_LOSS`].
///
/// [`cancellation`]: super::accuracy::cancellation
pub(super) fn prefer_spectral_sw1d<const N: usize>(
    ewald: Result<Jet<N>>,
    mut rounding: Rounding<N>,
    series: impl FnOnce() -> Option<(Jet<N>, Rounding<N>)>,
    k: Complex,
) -> Result<Jet<N>> {
    let accurate =
        |value: Complex, bound: f64| f64::EPSILON * bound <= MAX_LOSS * value.norm().max(1.0);
    let mut sum = match ewald {
        Ok(sum) => sum,
        // Non-finite summands, failed special functions or no convergence of either part.
        Err(error) if error.is_numerical() => {
            return match series() {
                Some((series, bound)) if accurate(series.value, bound.value) => Ok(series),
                _ => Err(error),
            };
        }
        Err(error) => return Err(error),
    };
    // A bound that overflowed to infinity or NaN counts as lost.
    let exceeds = |bound: f64, tolerance: f64| bound.is_nan() || f64::EPSILON * bound > tolerance;
    let tighter =
        |series: f64, ewald: f64| series.is_finite() && (ewald.is_nan() || series < ewald);
    let scale = sum.value.norm().max(1.0);
    let value_lost = exceeds(rounding.value, SPECTRAL_SWITCH_LOSS * scale);
    let lost: [bool; N] = std::array::from_fn(|i| {
        exceeds(
            rounding.derivative[i],
            SPECTRAL_SWITCH_LOSS
                * scale.mul_add(derivative_scale(i, 1, k), sum.derivative[i].norm()),
        )
    });
    if (value_lost || lost.contains(&true))
        && let Some((series, bound)) = series()
    {
        if value_lost && tighter(bound.value, rounding.value) {
            sum.value = series.value;
            rounding.value = bound.value;
        }
        for (i, lost) in lost.into_iter().enumerate() {
            if lost && tighter(bound.derivative[i], rounding.derivative[i]) {
                sum.derivative[i] = series.derivative[i];
            }
        }
    }
    if !accurate(sum.value, rounding.value) {
        return Err(Error::NotConverged(
            "Ewald sum lost its accuracy to cancelling Kambe integrals; reduce the split parameter"
                .into(),
        ));
    }
    Ok(sum)
}

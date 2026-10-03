//! Continuation of the reciprocal integrals to the sheet of the outgoing sum.
//!
//! treams-rs extension: treams takes the principal branches.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use super::accuracy::below_automatic;
use crate::{Complex, special};

/// `z` with a zero imaginary part made `-0.0`, which puts it on the lower side of a
/// branch cut along the real axis.
pub(super) fn lower_side(mut z: Complex) -> Complex {
    if z.im == 0.0 {
        z.im = -0.0;
    }
    z
}

/// Angle beyond `|arg((k eta)^2)|` by which a diffraction order's `k_q^2` must stay off
/// the positive real axis for [`split_sheet`] to take the principal argument of `v`
/// unevaluated: far above the rounding of that angle (1e-2 next to a threshold).
const SHEET_MARGIN: f64 = 0.5;

/// The split of an Ewald sum as the sheets of its diffraction orders need it
/// ([`split_sheet`]).
#[derive(Clone, Copy, Debug)]
pub(super) struct Split {
    /// `k^2`.
    square: Complex,
    /// `arg((k eta)^2)`, which turns every sheet.
    turn: f64,
    /// `tan^2(|arg((k eta)^2)| + SHEET_MARGIN)` below `pi / 2`, within which a `k_q^2`
    /// needs the argument of its sheet.
    near: Option<f64>,
    /// Whether the split lies below every automatic one ([`below_automatic`]).
    pub(super) small: bool,
}

impl Split {
    pub(super) fn new(k: Complex, eta: Complex) -> Self {
        let split = k * eta;
        let turn = (split * split).arg();
        let angle = turn.abs() + SHEET_MARGIN;
        Self {
            square: k * k,
            turn,
            near: (angle < FRAC_PI_2).then(|| angle.tan().powi(2)),
            small: below_automatic(eta),
        }
    }
}

/// The sheet of `v` of one diffraction order ([`split_sheet`]).
#[derive(Clone, Copy, Debug)]
pub(super) enum Sheet {
    /// The principal argument of `v`.
    Principal,
    /// The argument of `v` on the sheet, which differs from the principal one by whole
    /// turns up to rounding.
    Argument(f64),
}

impl Sheet {
    /// The whole turns from the principal argument of `value` (`= v` up to rounding) to
    /// the sheet.
    pub(super) fn turns(self, value: Complex) -> f64 {
        match self {
            Self::Principal => 0.0,
            Self::Argument(sheet) => sheet_turns(sheet, value),
        }
    }

    /// Whether `root`, a root of `v` up to rounding, is the opposite of the root
    /// `|v|^(1/2) e^(i arg / 2)` of the sheet's argument `arg`.
    pub(super) fn opposes(self, root: Complex, value: Complex) -> bool {
        let direction = match self {
            Self::Principal => crate::numerics::complex_sqrt(lower_side(value)).conj(),
            Self::Argument(sheet) => Complex::from_polar(1.0, -0.5 * sheet),
        };
        (root * direction).re < 0.0
    }
}

/// The sheet of `v = (q^2 - k^2) / (2 k^2 eta^2)` of a diffraction order of squared
/// transverse modulus `q2` that continues the Ewald sum from the automatic split to
/// `split`.
///
/// # Formula
///
/// The argument `arg(k_q^2) - pi - arg((k eta)^2)` with `arg(k_q^2)` in `[0, 2 pi)`.
///
/// # Branch
///
/// It takes the root `k_q = sqrt(k^2 - q^2)` with `Im k_q >= 0` of the spectral series
/// ([`spectral_sw1d`]). Orders farther off the positive real axis than
/// `|arg((k eta)^2)| +` [`SHEET_MARGIN`] take the principal argument.
///
/// # Why
///
/// The principal `ln v` misses the sheet where `k_q^2` lies between the positive real
/// axis and `(k eta)^2` (propagating orders of real `k` at splits rotated to
/// `Im(k eta) > 0`). It leaves the side of the cut to rounding where `v` lies on it
/// (`q = 0` for complex `k` and a real split).
///
/// [`spectral_sw1d`]: super::spectral::spectral_sw1d
pub(super) fn split_sheet(split: &Split, q2: Complex) -> Sheet {
    let square = split.square - q2;
    let near = |tan2: f64| square.re > 0.0 && square.im * square.im <= tan2 * square.re * square.re;
    if split.near.is_some_and(|tan2| !near(tan2)) {
        return Sheet::Principal;
    }
    let turn = if square.im == 0.0 {
        if square.re > 0.0 { 0.0 } else { PI }
    } else {
        square.arg().rem_euclid(TAU)
    };
    Sheet::Argument(turn - PI - split.turn)
}

/// The whole turns from the principal argument of `value` to `sheet`, which differ by
/// a multiple of `2 pi` up to rounding.
pub(super) fn sheet_turns(sheet: f64, value: Complex) -> f64 {
    ((sheet - value.arg()) / TAU).round()
}

/// The sheet `arg v = 2 arg k - pi - arg((k eta)^2)` of `v = -1 / (2 eta^2)` for
/// [`self_term`]: `-pi` continues the outgoing side `Im v = -0` of real `k` and splits,
/// and `k` takes `arg k = pi` on the negative real axis for either sign of zero, as the
/// AMOS outgoing waves do.
///
/// [`self_term`]: super::reciprocal::self_term
pub(super) fn self_sheet(k: Complex, eta: Complex) -> f64 {
    let k = if k.im == 0.0 {
        Complex::new(k.re, 0.0)
    } else {
        k
    };
    2.0 * k.arg() - PI - special::kambe_argument(k, eta).arg()
}

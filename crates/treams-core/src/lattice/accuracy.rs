//! Rounding-error bounds, and the failure of sums whose Ewald parts cancel.
//!
//! treams-rs extension: treams returns the sum of its parts without a bound.

use std::f64::consts::PI;

use super::{
    inputs::{K, KPAR},
    shells::Shells,
};
use crate::{Complex, Error, Result, numerics::Jet};

/// First-order bounds on the absolute rounding errors of a jet's value and of each of
/// its derivatives, in units of epsilon.
#[derive(Clone, Copy, Debug)]
pub(super) struct Rounding<const N: usize> {
    pub(super) value: f64,
    pub(super) derivative: [f64; N],
}

impl<const N: usize> Default for Rounding<N> {
    fn default() -> Self {
        Self::of(0.0)
    }
}

impl<const N: usize> Rounding<N> {
    /// A bound on a value only, whose derivatives are exact.
    pub(super) const fn of(value: f64) -> Self {
        Self {
            value,
            derivative: [0.0; N],
        }
    }

    /// The bounds of `exact * inexact`, where `self` bounds `inexact`. Bounds take the
    /// cheaper `|Re| + |Im|` for moduli, at most `sqrt 2` larger.
    pub(super) fn times(self, exact: &Jet<N>) -> Self {
        if self.value == 0.0 && self.derivative.iter().all(|&d| d == 0.0) {
            return self;
        }
        let scale = exact.value.l1_norm();
        Self {
            value: scale * self.value,
            derivative: std::array::from_fn(|i| {
                scale * self.derivative[i] + exact.derivative[i].l1_norm() * self.value
            }),
        }
    }

    /// The moduli of a jet's value and derivatives times `factor`: a bound for a jet
    /// computed with a relative error of `factor` ulps in each component.
    pub(super) fn relative(jet: &Jet<N>, factor: f64) -> Self {
        Self {
            value: factor * jet.value.l1_norm(),
            derivative: jet.derivative.map(|d| factor * d.l1_norm()),
        }
    }

    /// The bounds of `phase * inexact` for a `phase` of modulus 1, where `self` bounds
    /// `inexact`: its value keeps its bound, which the derivatives of the phase carry
    /// into theirs.
    fn turned(self, phase: &Jet<N>) -> Self {
        Self {
            value: self.value,
            derivative: std::array::from_fn(|i| {
                phase.derivative[i]
                    .norm()
                    .mul_add(self.value, self.derivative[i])
            }),
        }
    }

    /// The larger bound of each component.
    pub(super) fn max(self, other: Self) -> Self {
        Self {
            value: self.value.max(other.value),
            derivative: std::array::from_fn(|i| self.derivative[i].max(other.derivative[i])),
        }
    }
}

impl<const N: usize> std::ops::AddAssign for Rounding<N> {
    fn add_assign(&mut self, other: Self) {
        self.value += other.value;
        for (bound, other) in self.derivative.iter_mut().zip(other.derivative) {
            *bound += other;
        }
    }
}

/// Largest rounding error bound of a 1D spherical sum relative to `max(|S|, 1)`, and
/// largest loss of any sum to the [`cancellation`] of its Ewald parts relative to the
/// scale of a component, before it fails.
///
/// Calibrated: the bounds exceed the errors by a median factor of 17 to 87, and fall
/// short of them by at most a factor of 4.5.
pub(super) const MAX_LOSS: f64 = 1e-3;

/// Relative rounding error of every Ewald term and of the self term, in ulps, that
/// [`cancellation`] counts.
pub(super) const TERM_ULPS: f64 = 8.0;

/// The largest `Re(1 / (2 eta^2))` of the automatic split ([`resolve_split`]): `16 / pi`
/// from `|k L| = 8` on, with a relative margin of 1e-12 for its rounding.
///
/// It sets the growth `e^Re(1 / (2 eta^2))` of the Ewald parts. Only splits beyond it,
/// the small splits:
///
/// - check the [`cancellation`] of their parts;
/// - take the real-space integrals of [`SmallSplitKambe`];
/// - continue both parts to the complete sum ([`Shells`]);
/// - consider single Kambe integrals ([`Reduced::prefers_single`]).
///
/// The other splits keep their values and cost.
///
/// [`resolve_split`]: super::cell::resolve_split
/// [`SmallSplitKambe`]: crate::special::SmallSplitKambe
/// [`Reduced::prefers_single`]: super::reduced::Reduced::prefers_single
const AUTOMATIC_GROWTH: f64 = 16.0 / PI * (1.0 + 1e-12);

/// Whether the split `eta` lies below every automatic one ([`AUTOMATIC_GROWTH`]).
pub(super) fn below_automatic(eta: Complex) -> bool {
    (0.5 / (eta * eta)).re > AUTOMATIC_GROWTH
}

/// The components of a sum that [`cancellation`] checks: the value unless the sum
/// vanishes by symmetry ([`vanishes`]), which returns it as an exact zero, and each
/// derivative that such a sum does not return as zero either, with the scale `s` of
/// its slot ([`derivative_scale`]).
///
/// [`vanishes`]: super::ewald::vanishes
pub(super) struct Checked<const N: usize> {
    pub(super) value: bool,
    pub(super) units: [Option<f64>; N],
}

/// The scale `s` of the derivative in jet slot `slot` of a sum with `dim` lattice
/// dimensions and the wavenumber `k`, on which it counts `|dS| + max(|S|, 1) s`: `1 / |k|`
/// for the wavenumber and the Bloch vector, `|k|` for the shift and the lattice vectors.
pub(super) fn derivative_scale(slot: usize, dim: usize, k: Complex) -> f64 {
    if slot == K || (KPAR..KPAR + dim).contains(&slot) {
        1.0 / k.norm()
    } else {
        k.norm()
    }
}

/// The predicted absolute rounding errors, in units of epsilon, of the value and
/// derivatives of `sum` (`phase` times the Ewald `parts` and the self term `other`), or a
/// failure where they cancel beyond use: where the error of a `checked` component plus
/// the truncation `tail` of parts that did not converge to the complete sum
/// ([`Shells::tail`]) exceeds [`MAX_LOSS`] of its scale, `max(|S|, 1)` for the value
/// and `|dS| + max(|S|, 1) s` for a derivative (`S = 0` for a sum that vanishes by
/// symmetry). Each component predicts the bounds of the real-space Kambe integrals
/// ([`SmallSplitKambe`]) plus [`TERM_ULPS`] times the moduli of the terms and of the self
/// term, and an ulp of each compensated part ([`Shells`]).
///
/// Calibrated against the automatic split: the prediction exceeds the error of every sum
/// more than 1e-11 off by a median factor of 45 at real and 87 at rotated splits (at
/// least 4), and no sum that passes is more than 1.2e-4 off (not counting 1D spherical
/// sums off the axis, see [`prefer_spectral_sw1d`]).
///
/// [`SmallSplitKambe`]: crate::special::SmallSplitKambe
/// [`prefer_spectral_sw1d`]: super::spectral::prefer_spectral_sw1d
pub(super) fn cancellation<const N: usize>(
    parts: &[&Shells<N>],
    other: &Jet<N>,
    tail: f64,
    phase: &Jet<N>,
    sum: &Jet<N>,
    checked: &Checked<N>,
) -> Result<Rounding<N>> {
    let mut rounding = Rounding::relative(other, TERM_ULPS);
    for part in parts {
        rounding += part.bound;
        rounding += Rounding::relative(&part.sum, 1.0);
    }
    // The phase of the shift into the cell has modulus 1 (a real Bloch vector).
    let rounding = rounding.turned(phase);
    // A sum whose value is not checked is returned as an exact zero, whatever rounding
    // its parts leave in it: its scale is 1.
    let scale = if checked.value {
        sum.value.norm().max(1.0)
    } else {
        1.0
    };
    // A bound that overflowed to NaN counts as lost.
    let loss = |bound: f64, scale: f64| f64::EPSILON.mul_add(bound, tail) / scale;
    let lost = |loss: f64| loss.is_nan() || loss > MAX_LOSS;
    let value = loss(rounding.value, scale);
    if checked.value && lost(value) {
        return Err(too_small(value, false));
    }
    for ((&bound, derivative), unit) in rounding
        .derivative
        .iter()
        .zip(&sum.derivative)
        .zip(checked.units)
    {
        if let Some(unit) = unit {
            let derivative = loss(bound, scale.mul_add(unit, derivative.norm()));
            if lost(derivative) {
                return Err(too_small(derivative, true));
            }
        }
    }
    Ok(rounding)
}

/// The failure of a sum whose cancelling Ewald parts lose `loss` of `max(|S|, 1)`, or
/// of a `derivative` on its scale `|dS| + max(|S|, 1) s`.
pub(super) fn too_small(loss: f64, derivative: bool) -> Error {
    let scale = if derivative {
        "of |dS| + max(|S|, 1) s in a derivative"
    } else {
        "of max(|S|, 1)"
    };
    Error::NotConverged(format!(
        "Ewald split too small: its cancelling parts lose about {loss:.0e} {scale}; \
         use a larger split (eta = 0 selects one)"
    ))
}

/// The loss beyond which a complete sum below every automatic split fails early, while
/// its real-space terms still add up.
///
/// The loss is relative to the scale of a component of the sum at the automatic split,
/// as [`cancellation`] counts it. The failure comes before the far shells, where the
/// Kambe series of the terms grow with `1 / eta^2`. The predicted loss only grows with
/// the terms, so the complete sum would fail too unless it came back more than a scale
/// off the automatic one. The automatic sum (the jet, for a jet) is evaluated once,
/// where a loss exceeds this of 1 for the value or of `s` for a derivative; a sum whose
/// automatic split fails continues.
///
/// Calibrated: without the early failure, a 3D sum at the split 0.03 takes 29 s to fail
/// at the shell limit.
pub(super) const EARLY_FAILURE_LOSS: f64 = 2.0 * MAX_LOSS;

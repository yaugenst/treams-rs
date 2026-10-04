//! Real-space terms of the Ewald sum.
//!
//! Upstream: `_realsw*` and `_realcw*` of `lattice/_esum.pyx`. The bounded Kambe
//! integrals at small splits are a treams-rs extension.

use std::f64::consts::PI;

use super::{
    accuracy::Rounding,
    wave::{Family, azimuthal, negative_order_sign, solid_jet},
};
use crate::{
    Complex,
    numerics::Jet,
    special::{self, SmallSplitKambe, harmonic_normalization},
};

/// How the real-space terms take their Kambe integrals ([`kambe_jet`]).
#[derive(Debug)]
pub(super) enum RealKambe<'a, const N: usize> {
    /// The integrals of the recurrences, without bounds.
    Plain,
    /// Below every automatic split ([`below_automatic`]): the integrals of
    /// [`SmallSplitKambe`], which keep their closed forms below the rounding `largest`
    /// of the largest term so far in each component (see [`kambe_floor`]), storing the
    /// bounds [`cancellation`] reads in `bound`.
    ///
    /// [`below_automatic`]: super::accuracy::below_automatic
    /// [`cancellation`]: super::accuracy::cancellation
    SmallSplit {
        kambe: &'a SmallSplitKambe,
        largest: &'a Rounding<N>,
        bound: &'a mut Rounding<N>,
    },
    /// Only the `eta` derivative, in slot 0, for the split derivative.
    EtaDerivative,
}

/// Kambe integral of order `n` at `z` of a real-space term with the factor `factor`, with
/// its `z` derivative `-z I_(n+2)`, or its `eta` derivative alone.
fn kambe_jet<const N: usize>(
    n: i32,
    z: Jet<N>,
    eta: Complex,
    mode: RealKambe<'_, N>,
    factor: &Jet<N>,
) -> Jet<N> {
    let (value, next) = match mode {
        RealKambe::EtaDerivative => {
            let derivative =
                -eta.powi(n) * (0.5 * (1.0 / (eta * eta) - z.value * z.value * eta * eta)).exp();
            let mut jet = Jet::default();
            if let Some(slot) = jet.derivative.first_mut() {
                *slot = derivative;
            }
            return jet;
        }
        RealKambe::Plain => {
            let next = (N > 0).then(|| special::kambe(n + 2, z.value, eta));
            (special::kambe(n, z.value, eta), next)
        }
        RealKambe::SmallSplit {
            kambe,
            largest,
            bound,
        } => small_split_kambe(n, &z, kambe, largest, bound, factor),
    };
    let derivative = next.map_or_else(Complex::default, |next| -z.value * next);
    z.chain(value, derivative)
}

/// The radial integral shared by all spherical orders of one degree at a point.
/// Used only above the small-split cancellation regime, where the integral does not
/// depend on the harmonic's rounding budget.
pub(super) fn spherical_radial<const N: usize>(
    l: i32,
    k: Jet<N>,
    r: [Jet<N>; 3],
    eta: Complex,
) -> Jet<N> {
    let radius = r.into_iter().map(|r| r * r).sum::<Jet<N>>().sqrt();
    kambe_jet(2 * l, k * radius, eta, RealKambe::Plain, &Jet::default())
}

/// A spherical real-space term from its already evaluated radial integral.
pub(super) fn spherical_from_radial<const N: usize>(
    l: i32,
    m: i32,
    k: Jet<N>,
    r: [Jet<N>; 3],
    radial: Jet<N>,
) -> Jet<N> {
    let harmonic = solid_jet(l, m, r) * harmonic_normalization(l, m);
    let factor = -Complex::i() * (2.0 / PI).sqrt() * k.powi(l) * harmonic;
    factor * radial
}

/// `I_n` and, for jets, `I_(n+2)` at `z` in [`RealKambe::SmallSplit`], storing their
/// bounds in the term with the factor `factor` in `bound`.
///
/// Kept out of line, like [`Shells::add_compensated`], so the term loop at the other
/// splits stays small. Calibrated: inlined, it slows 3D jets at the automatic split by
/// 1% to 2.7%.
///
/// [`Shells::add_compensated`]: super::shells::Shells::add_compensated
#[inline(never)]
pub(super) fn small_split_kambe<const N: usize>(
    n: i32,
    z: &Jet<N>,
    kambe: &SmallSplitKambe,
    largest: &Rounding<N>,
    bound: &mut Rounding<N>,
    factor: &Jet<N>,
) -> (Complex, Option<Complex>) {
    let floor = kambe_floor(largest, factor);
    let (value, value_bound, next) = if N == 0 {
        let [(value, value_bound)] = kambe.get([n], z.value, floor);
        (value, value_bound, None)
    } else {
        let [(value, value_bound), next] = kambe.get([n, n + 2], z.value, floor);
        (value, value_bound, Some(next))
    };
    let (next_bound, modulus) = (next.map_or(0.0, |(_, bound)| bound), z.value.l1_norm());
    *bound = Rounding {
        value: value_bound,
        derivative: z.derivative.map(|d| next_bound * modulus * d.l1_norm()),
    }
    .times(factor);
    (value, next.map(|(next, _)| next))
}

/// The error, in units of epsilon, up to which [`SmallSplitKambe::get`] keeps the closed
/// forms of the Kambe integral of a real-space term with the factor `factor`: what leaves
/// less in each component of the term than the rounding `largest` of the largest term so
/// far, carried by the value of the factor or, where that vanishes (off-axis orders on a
/// 1D spherical axis, odd `l + m` in a 2D plane), by its derivatives.
fn kambe_floor<const N: usize>(largest: &Rounding<N>, factor: &Jet<N>) -> f64 {
    let scale = factor.value.l1_norm();
    if scale > 0.0 {
        return largest.value / scale;
    }
    largest
        .derivative
        .iter()
        .zip(&factor.derivative)
        .filter(|&(_, &d)| d != Complex::default())
        .map(|(largest, d)| largest / d.l1_norm())
        .fold(f64::INFINITY, f64::min)
}

/// One real-space summand at the displacement `r = -shift - R`.
///
/// # Formula
///
/// Spherical: `-i sqrt(2 / pi) k^l |r|^l Y_lm(r / |r|) I_(2l)(k |r|, eta)`, where
/// `|r|^l Y_lm` is `harmonic_normalization(l, m)` times the solid harmonic. Cylindrical:
/// `-2 i / pi (-1)^m (k (x ± i y))^|m| I_(2|m| - 1)(k |r|, eta)`, with the sign `(-1)^m`
/// and `x - i y` for negative `m` only.
///
/// [`RealKambe::SmallSplit`] stores first-order bounds on the absolute rounding errors
/// that its Kambe integral leaves in its value and derivatives (see [`kambe_jet`]).
pub(super) fn real_term<const N: usize>(
    wave: Family,
    k: Jet<N>,
    r: [Jet<N>; 3],
    eta: Complex,
    mode: RealKambe<'_, N>,
) -> Jet<N> {
    let radius = r.into_iter().map(|r| r * r).sum::<Jet<N>>().sqrt();
    let (factor, order) = match wave {
        Family::Spherical { l, m } => {
            let harmonic = solid_jet(l, m, r) * harmonic_normalization(l, m);
            (
                -Complex::i() * (2.0 / PI).sqrt() * k.powi(l) * harmonic,
                2 * l,
            )
        }
        Family::Cylindrical { m } => {
            let (xy, sign) = (azimuthal(r[0], r[1], m), negative_order_sign(m));
            (
                -2.0 * Complex::i() / PI * sign * (k * xy).powi(m.abs()),
                2 * m.abs() - 1,
            )
        }
    };
    factor * kambe_jet(order, k * radius, eta, mode, &factor)
}

//! Degree terms `(p, weight)` of the spherical translation coefficients, from the
//! Wigner 3j couplings of a source and a destination mode.
//!
//! Upstream: `treams.special._tl_vsw_helper` and the degree sums of `tl_vsw_A` and
//! `tl_vsw_B`.
#![allow(clippy::indexing_slicing)] // Fixed indices of the two Wigner 3j rows.

use super::Mode;
use crate::Complex;
use crate::special::{Wigner3jRow, helicity_sign};

/// Degree-`p` coefficient of the coupling of source `(l, m)` to destination
/// `(lambda, -mu)` with `q = p - cross`. Zero unless both Wigner 3j symbols it multiplies,
/// `(l lambda p; m mu -m-mu)` from `rows[0] = Wigner3jRow::new(l, lambda, m, mu)` and
/// `(l lambda q; 0 0 0)` from `rows[1] = Wigner3jRow::new(l, lambda, 0, 0)`, can be
/// nonzero.
///
/// Upstream: `treams.special._tl_vsw_helper`.
fn tl_vsw_term(
    l: i32,
    m: i32,
    lambda: i32,
    mu: i32,
    p: i32,
    q: i32,
    rows: [&Wigner3jRow; 2],
) -> Complex {
    if p < (m + mu).abs().max((l - lambda).abs())
        || p > l + lambda
        || q < (l - lambda).abs()
        || q > l + lambda
        || (q + l + lambda) % 2 != 0
    {
        return Complex::default();
    }
    f64::from(2 * p + 1)
        * Complex::i().powi(lambda - l + p)
        * crate::special::factorial_ratio_sqrt(p, m + mu)
        * rows[0].get(p)
        * rows[1].get(q)
}

/// Which of the two coupling kinds of a [`Coupling`] a polarization pair uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Kinds {
    /// The `tl_vsw_A` terms.
    same: bool,
    /// The `tl_vsw_B` terms.
    cross: bool,
}

impl Kinds {
    /// Both kinds, for plans that serve every polarization pair.
    pub(crate) const BOTH: Self = Self {
        same: true,
        cross: true,
    };

    /// The kinds that couple source polarization `from` to destination polarization `to`.
    /// The parity basis couples equal polarizations through `same` and opposite ones
    /// through `cross`; the helicity basis couples equal helicities through both.
    const fn of(to: u8, from: u8, helicity: bool) -> Self {
        let equal = to == from;
        Self {
            same: equal,
            cross: if helicity { equal } else { !equal },
        }
    }
}

/// The degree terms `(p, weight)` coupling a source `(l, m)` to a destination
/// `(lambda, mu)`, for all four polarization pairs. The parity basis couples equal
/// polarizations through `same` and opposite ones through `cross`; the helicity basis
/// couples equal helicities through both, with `cross` signed by `2 pol - 1`.
///
/// `same` holds the terms of `treams.special.tl_vsw_A`, `cross` those of `tl_vsw_B`.
#[derive(Debug)]
pub(crate) struct Coupling {
    same: Vec<(i32, Complex)>,
    cross: Vec<(i32, Complex)>,
}

impl Coupling {
    /// Evaluates the `same` and `cross` terms selected by `kinds`; `zeros` is
    /// `Wigner3jRow::new(l, lambda, 0, 0)`, shared by all orders.
    pub(crate) fn new(
        (l, m): (i32, i32),
        (lambda, mu): (i32, i32),
        zeros: &Wigner3jRow,
        kinds: Kinds,
    ) -> Self {
        let orders = Wigner3jRow::new(l, lambda, m, -mu);
        let rows = [&orders, zeros];
        let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
        let pref = 0.5
            * sign
            * (f64::from((2 * l + 1) * (2 * lambda + 1))
                / (f64::from(l * (l + 1)) * f64::from(lambda * (lambda + 1))))
            .sqrt();
        let [same, cross] = [false, true].map(|cross| {
            if !(if cross { kinds.cross } else { kinds.same }) {
                return Vec::new();
            }
            degrees(l, m, lambda, mu, cross)
                .filter_map(|p| {
                    let factor = if cross {
                        (f64::from(l + lambda + 1 + p)
                            * f64::from(l + lambda + 1 - p)
                            * f64::from(p + lambda - l)
                            * f64::from(p - lambda + l))
                        .sqrt()
                    } else {
                        f64::from(l * (l + 1) + lambda * (lambda + 1) - p * (p + 1))
                    };
                    let weight = pref
                        * factor
                        * tl_vsw_term(l, m, lambda, -mu, p, p - i32::from(cross), rows);
                    (weight.norm_sqr() > 0.0).then_some((p, weight))
                })
                .collect()
        });
        Self { same, cross }
    }

    /// Terms `(p, weight)` from source polarization `from` to destination polarization
    /// `to`, with same-polarization terms first.
    pub(crate) fn terms(
        &self,
        to: u8,
        from: u8,
        helicity: bool,
    ) -> impl Iterator<Item = (i32, Complex)> + '_ {
        let kinds = Kinds::of(to, from, helicity);
        let sign = if helicity { helicity_sign(from) } else { 1.0 };
        let same = if kinds.same { &self.same[..] } else { &[] };
        let cross = if kinds.cross { &self.cross[..] } else { &[] };
        same.iter()
            .copied()
            .chain(cross.iter().map(move |&(p, weight)| (p, sign * weight)))
    }
}

/// Degrees `p` that `tl_vsw_term` admits for source `(l, m)` and destination
/// `(lambda, mu)`, descending in steps of two. `formal/Formal/SelectionRules.lean` proves
/// this range.
pub(crate) fn degrees(
    l: i32,
    m: i32,
    lambda: i32,
    mu: i32,
    cross: bool,
) -> impl ExactSizeIterator<Item = i32> {
    let start = l + lambda - i32::from(cross);
    let end = (lambda - l)
        .abs()
        .saturating_add(i32::from(cross))
        .max((m - mu).abs());
    // `(end..=start).rev().step_by(2)`, whose length is known.
    (0..(start - end + 2).max(0) / 2).map(move |k| start - 2 * k)
}

/// The most [`terms`] from `from` to `to`: one per admitted degree of each selected
/// kind. Terms of zero weight are left out, so there can be fewer.
pub(crate) fn term_bound(to: Mode, from: Mode, helicity: bool) -> usize {
    let Kinds { same, cross } = Kinds::of(to.pol, from.pol, helicity);
    let count = |kind| degrees(from.l, from.m, to.l, to.m, kind).len();
    usize::from(same) * count(false) + usize::from(cross) * count(true)
}

/// Terms `(p, m - mu, weight)` of the translation from source `(l, m)` to destination
/// `(lambda, mu)`: `sum weight z_p(k r) P_p^(m - mu) e^(i (m - mu) phi)`.
pub(crate) fn terms(to: Mode, from: Mode, helicity: bool) -> Vec<(i32, i32, Complex)> {
    let kinds = Kinds::of(to.pol, from.pol, helicity);
    if !kinds.same && !kinds.cross {
        return Vec::new();
    }
    let zeros = Wigner3jRow::new(from.l, to.l, 0, 0);
    Coupling::new((from.l, from.m), (to.l, to.m), &zeros, kinds)
        .terms(to.pol, from.pol, helicity)
        .map(|(p, weight)| (p, from.m - to.m, weight))
        .collect()
}

#[cfg(test)]
mod tests {
    //! The degree selection rules of the coupling terms against the Lean model.
    //! Identities of the translations themselves are in `properties/waves.rs`.

    use super::{Complex, Mode, Wigner3jRow, degrees, term_bound, terms, tl_vsw_term};
    use crate::test_support::table;

    /// The cases of `formal/Golden.lean`'s `degrees`, in its order: `[l, m, lambda, mu, cross]`.
    fn lean_degree_cases() -> Vec<[i32; 5]> {
        (1..=5)
            .flat_map(|l| (1..=5).map(move |lambda| (l, lambda)))
            .flat_map(|(l, lambda)| (-l..=l).map(move |m| (l, lambda, m)))
            .flat_map(|(l, lambda, m)| (-lambda..=lambda).map(move |mu| (l, lambda, m, mu)))
            .flat_map(|(l, lambda, m, mu)| [0, 1].map(|cross| [l, m, lambda, mu, cross]))
            .collect()
    }

    #[test]
    fn degrees_match_lean_model() {
        // `just formal` keeps this file equal to `Treams.SelectionRules.termDegrees`.
        let cases = table::<i32, i32>(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../formal/golden/degrees.txt"
        )));
        let grid = lean_degree_cases();
        assert_eq!(cases.len(), grid.len());
        for ((key, expected), case) in cases.iter().zip(grid) {
            assert_eq!(key, &case);
            let [l, m, lambda, mu, cross] = case;
            let actual: Vec<_> = degrees(l, m, lambda, mu, cross == 1).collect();
            assert_eq!(&actual, expected, "{key:?}");
        }
    }

    #[test]
    fn skipped_degrees_have_zero_coefficients() {
        // Rust counterpart of `SelectionRules.skipped_degree_vanishes`: `terms` evaluates
        // `tl_vsw_term(l, m, lambda, -mu, p, p - cross, rows)` only at `degrees`, so every
        // other degree must be rejected by `tl_vsw_term` itself.
        for [l, m, lambda, mu, cross] in lean_degree_cases() {
            let visited: Vec<_> = degrees(l, m, lambda, mu, cross == 1).collect();
            let rows = [
                &Wigner3jRow::new(l, lambda, m, -mu),
                &Wigner3jRow::new(l, lambda, 0, 0),
            ];
            for p in (0..=l + lambda + 1).filter(|p| !visited.contains(p)) {
                let coefficient = tl_vsw_term(l, m, lambda, -mu, p, p - cross, rows);
                assert_eq!(
                    coefficient,
                    Complex::default(),
                    "{l} {m} {lambda} {mu} {cross}: {p}"
                );
            }
        }
    }

    #[test]
    fn terms_use_admitted_degrees_and_table_indices() {
        // Same- and cross-polarization couplings in both bases (`pol` only flips signs).
        let polarizations = [(true, 1, 1), (false, 0, 0), (false, 0, 1)];
        for [l, m, lambda, mu, _] in lean_degree_cases().into_iter().step_by(2) {
            for (helicity, from_pol, to_pol) in polarizations {
                let from = Mode {
                    l,
                    m,
                    pol: from_pol,
                };
                let to = Mode {
                    l: lambda,
                    m: mu,
                    pol: to_pol,
                };
                let admitted = |p, cross| degrees(l, m, lambda, mu, cross).any(|q| q == p);
                let terms = terms(to, from, helicity);
                assert!(terms.len() <= term_bound(to, from, helicity));
                for (p, order, _) in terms {
                    let case = format!("{from:?} -> {to:?}, helicity {helicity}: {p} {order}");
                    assert_eq!(order, m - mu, "{case}");
                    let expected = if helicity {
                        admitted(p, false) || admitted(p, true)
                    } else {
                        admitted(p, from_pol != to_pol)
                    };
                    assert!(expected, "{case}");
                    // `Harmonics.table_index`: `p * p + p + order` lies in the plan table.
                    assert!(order.abs() <= p && p <= l + lambda, "{case}");
                }
            }
        }
    }
}

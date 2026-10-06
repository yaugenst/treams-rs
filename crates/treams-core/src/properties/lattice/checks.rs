//! Checks of the Ewald sums and the assertions and helpers they share.

use std::f64::consts::{PI, TAU};

use proptest::{prelude::*, test_runner::TestCaseError};

use super::ewald::{Ewald, c, cw, diagonal, embed, leading, sw};
use crate::{
    Complex,
    lattice::{
        self, BlochLattice, Derivatives, SHELL_TOLERANCE, SumPart, derivatives_part, probes,
    },
    numerics::parity,
    special::{self, Radial},
    test_support::{central, dot, prop_assert_close},
};

/// Largest error of a 1D spherical sum relative to its scale in `check_chain`: a
/// component stays with the Ewald sum while its rounding bound stays within
/// `lattice::SPECTRAL_SWITCH_LOSS` = 1e-12 of `max(|S|, 1)`, and its errors exceeded that
/// bound by factors up to 4.5; the plain Ewald references and the series agree to 1e-11.
const CHAIN_TOLERANCE: f64 = 2e-11;

/// Unsupported wavenumbers fail as invalid input in every Ewald part and jet, at
/// both the given and automatic splits (part jets require an explicit split). Keep
/// these draws in the existing strategies so their regression seeds still replay.
pub(super) fn check_rejected_wavenumber(sum: &Ewald) -> Result<(), TestCaseError> {
    let message = if sum.k.im < 0.0 {
        "Im(k) >= 0"
    } else {
        "Re(k) >= 0"
    };
    for eta in [sum.eta, Complex::default()] {
        let sum = sum.at(eta);
        for part in [SumPart::Full, SumPart::Real, SumPart::Reciprocal] {
            let jet_eta = if eta == Complex::default() && !matches!(part, SumPart::Full) {
                c(1.0, 0.0)
            } else {
                eta
            };
            let jet = derivatives_part(sum.wave, sum.k, &sum.lattice(), sum.r, jet_eta, part);
            for result in [sum.part(part), jet.map(|d| d.value)] {
                prop_assert!(
                    matches!(&result, Err(crate::Error::InvalidInput(error)) if error.contains(message)),
                    "{sum:?} {part:?}: {result:?}"
                );
            }
        }
    }
    Ok(())
}

/// A 1D spherical sum against the plain Ewald sum at a split of modulus
/// `min(1.5 / (|k| rho), 1.5)` along `direction` (along `1 / k` without one), where that
/// loses at most `e^(w^2)` and `e^(1 / (2 |eta|^2))`, about 10 and 23 ulps of its terms.
/// With a `direction`, the spectral series matches it in value and every derivative
/// within 1e-11 of their scales plus twice its own rounding bound; without, the sum at
/// the automatic split, from whichever of the two each component comes, within
/// `CHAIN_TOLERANCE` of them, and its value matches the value-only sum within 4 ulps of
/// `max(|S|, 1)`. Gain wavenumbers must instead be rejected. Draws next to a diffraction
/// threshold are skipped ([`skip_threshold_band`]).
pub(super) fn check_chain(sum: &Ewald, direction: Option<Complex>) -> Result<(), TestCaseError> {
    skip_threshold_band(sum)?;
    let (k, rho) = (sum.k, sum.r[0].hypot(sum.r[1]));
    let size = (1.5 / (k.norm() * rho)).min(1.5);
    let eta = direction.map_or_else(|| size * k.norm() / k, |direction| size * direction);
    if sum.k.im < 0.0 {
        return check_rejected_wavenumber(&sum.at(eta));
    }
    let expected = probes::ewald_only(|| sum.at(eta).try_derivatives()).unwrap();
    let scales = component_scales(&expected, k);
    let lattice::Family::Spherical { l, m } = sum.wave else {
        panic!("chain sums are spherical");
    };
    let (actual, bounds) = if direction.is_some() {
        probes::spectral_sw1d_derivatives((l, m), k, &sum.lattice(), sum.r).unwrap()
    } else {
        let actual = sum.derivatives();
        // Only on the spectral series are the value-only sum and the jet's value the same
        // by construction (`spectral_sw1d`). The Ewald shells of both run the same
        // arithmetic but stop once two add less than `SHELL_TOLERANCE` of the largest
        // modulus of what they sum, the value alone or every component, so the two may
        // sum different shells: 7 and 9 for a degree-23 chain, an ulp of Re S apart.
        // 3 of 2e6 draws differed, all within an ulp, and none of 2e6 moved toward a
        // threshold short of the band of `skip_threshold_band` by an ulp of `max(|S|, 1)`.
        let tolerance = 4.0 * f64::EPSILON * actual.value.norm().max(1.0);
        prop_assert_close!(sum.sum(), actual.value, tolerance, "value-only");
        (actual, value_only(Complex::default()))
    };
    let pairs = components(&actual).into_iter().zip(components(&expected));
    for (i, ((a, e), (scale, bound))) in pairs
        .zip(scales.into_iter().zip(components(&bounds)))
        .enumerate()
    {
        let tolerance = if direction.is_some() {
            1e-11 * scale + 2.0 * f64::EPSILON * bound.re
        } else {
            CHAIN_TOLERANCE * scale
        };
        prop_assert_close!(a, e, tolerance, "component {}", i);
    }
    Ok(())
}

/// A degree-0 chain sum at the automatic split against `spectral_chain_sum` within
/// `CHAIN_TOLERANCE` of `max(|S|, 1)`, skipped next to a diffraction threshold
/// ([`skip_threshold_band`]).
pub(super) fn check_off_axis_chain(
    k: f64,
    period: f64,
    kpar: f64,
    r: [f64; 3],
) -> Result<(), TestCaseError> {
    let sum = Ewald::chain(sw(0, 0), period, kpar, r, c(k, 0.0), Complex::default());
    skip_threshold_band(&sum)?;
    // With the automatic split, `w = sqrt(2 pi) rho / L max(|k| L / 8, 1)`.
    let w = TAU.sqrt() * r[0].hypot(r[1]) / period * (k * period / 8.0).max(1.0);
    let (actual, expected) = (sum.sum(), spectral_chain_sum(k, period, kpar, r));
    prop_assert!(
        (actual - expected).norm() <= CHAIN_TOLERANCE * expected.norm().max(1.0),
        "w = {w}: {actual} against {expected}"
    );
    Ok(())
}

/// Rejects a chain draw next to a diffraction threshold, where `eps` times its
/// [`conditioning`] exceeds 1e-12, so that every check of a property skips it. There the
/// rounding of the order at the threshold reaches every sum the checks compare (at other
/// splits, shifts and scales, and by other paths and series): draws moved toward
/// thresholds failed them from 2.6e-12 on, where two orders reach a threshold together
/// at an explicit split, and otherwise from 2e-11 on. 4.6e-4 to 1.5e-3 of the draws of
/// each chain property lie in the band.
pub(super) fn skip_threshold_band(sum: &Ewald) -> Result<(), TestCaseError> {
    prop_assume!(f64::EPSILON * conditioning(sum) <= 1e-12);
    Ok(())
}

/// The conditioning of `sum` next to a diffraction threshold: the largest rounding of
/// `k_q^2` over `|k_q^2|` of its orders ([`order_gaps`]), the relative error that the
/// rounded `k_q^2` of an order carries into its terms.
fn conditioning(sum: &Ewald) -> f64 {
    let gaps = order_gaps(sum, order_reach(sum));
    gaps.iter()
        .map(|(x, rounding)| rounding / x.norm())
        .fold(0.0, f64::max)
}

/// `k_q^2 = k^2 - |q + G|^2` for the reciprocal vectors `G = n . b` of `sum` with
/// `|n_i| <= reach`, which vanishes at the threshold of its diffraction order, and what
/// it rounds by in units of `eps`: `|k|^2` for the squares, the constant of
/// `eps k^2 / |k_q^2|` in "Near thresholds" of `docs/validation/numerical-limits.md`,
/// plus `2 |q + G| (|q| + |G|)`, this test's extension for the rounded `q` and `G`.
fn order_gaps(sum: &Ewald, reach: i32) -> Vec<(Complex, f64)> {
    let b = sum.reciprocal();
    let length = |v: [f64; 3]| dot(v, v).sqrt();
    cells(sum.dim, reach)
        .map(|n| {
            let g: [f64; 3] = std::array::from_fn(|j| (0..3).map(|i| n[i] * b[i][j]).sum());
            let shifted: [f64; 3] = std::array::from_fn(|j| sum.kpar[j] + g[j]);
            let rounding =
                sum.k.norm_sqr() + 2.0 * length(shifted) * (length(sum.kpar) + length(g));
            (sum.k * sum.k - dot(shifted, shifted), rounding)
        })
        .collect()
}

/// The `reach` of [`order_gaps`] that holds every `G` with `|q + G| <= |k| + 1`, as
/// `n_i = a_i . G / 2 pi`.
fn order_reach(sum: &Ewald) -> i32 {
    let length = |v: [f64; 3]| dot(v, v).sqrt();
    let widest = sum.rows.map(length).into_iter().fold(0.0, f64::max);
    let radius = sum.k.norm() + length(sum.kpar) + 1.0;
    #[allow(clippy::cast_possible_truncation)] // A few cells.
    let reach = (radius * widest / TAU).ceil() as i32;
    reach
}

/// The integer points of the cube `[-reach, reach]^dim`, zero beyond `dim`.
fn cells(dim: usize, reach: i32) -> impl Iterator<Item = [f64; 3]> {
    let span = move |i: usize| if i < dim { -reach..=reach } else { 0..=0 };
    span(0).flat_map(move |a| {
        span(1).flat_map(move |b| span(2).map(move |c| [a, b, c].map(f64::from)))
    })
}

/// The degree-0 sum of a chain along z from its spectral series,
/// `-i / (k sqrt(4 pi)) i pi / a sum_m H_0(rho k_m) e^(-i q_m z)` with
/// `q_m = kpar + 2 pi m / a` and `k_m = sqrt(k^2 - q_m^2)`, `Im k_m >= 0`, whose
/// evanescent terms fall like `e^(-2 pi |m| rho / a)` off the axis.
pub(super) fn spectral_chain_sum(k: f64, period: f64, kpar: f64, r: [f64; 3]) -> Complex {
    let rho = r[0].hypot(r[1]);
    let turn = TAU / period;
    // Beyond the propagating orders, 40 / (rho turn) more reach e^-40.
    #[allow(clippy::cast_possible_truncation)] // A few hundred orders at most.
    let reach = ((k + kpar.abs()) / turn + 40.0 / (rho * turn)).ceil() as i32 + 1;
    let total: Complex = (-reach..=reach)
        .map(|m| {
            let q = kpar + turn * f64::from(m);
            let radial = crate::numerics::complex_sqrt(c(k * k - q * q, 0.0));
            special::cylindrical_radial(0, rho * radial, Radial::Singular)
                .unwrap()
                .value
                * Complex::from_polar(1.0, -q * r[2])
        })
        .sum();
    -Complex::i() / (k * (4.0 * PI).sqrt()) * (Complex::i() * PI / period) * total
}

/// The reduced integral `F_n(v, t)` of one reciprocal point, of squared transverse
/// modulus `q2` at `|t| = |k s eta|^2 = modulus`, from its series in `t` and from Kambe
/// integrals agrees to 128 ulps of the moduli of the series terms plus the Kambe bound
/// (at most 98 measured), or to 1e-13 of those moduli plus the value.
pub(super) fn check_reduced_integrals(
    twice_n: i32,
    k: Complex,
    q2: f64,
    eta: Complex,
    modulus: f64,
) -> Result<(), TestCaseError> {
    let distance = modulus.sqrt() / (k * eta).norm();
    let [(series, magnitude), (kambe, bound)] =
        probes::reduced_paths(twice_n, k, q2, eta, distance);
    let tolerance =
        (128.0 * f64::EPSILON * (magnitude + bound)).min(1e-13 * (magnitude + kambe.norm()));
    prop_assert_close!(series, kambe, tolerance);
    Ok(())
}

/// Below every automatic split a sum (or with `jet` its jet) keeps each component within
/// `MAX_LOSS` = 1e-3 of its scale of the automatic split (see
/// [`prop_assert_jets_on_their_scales`]), or fails with "split too small" or "did not
/// converge"; sums whose automatic split fails are skipped.
pub(super) fn check_small_split(case: &Ewald, jet: bool) -> Result<(), TestCaseError> {
    let evaluate = |sum: &Ewald| {
        if jet {
            sum.try_derivatives()
        } else {
            sum.try_sum().map(value_only)
        }
    };
    let expected = evaluate(&case.at(Complex::default()));
    prop_assume!(expected.is_ok());
    match evaluate(case) {
        Ok(actual) => prop_assert_jets_on_their_scales(&actual, &expected.unwrap(), case.k, 1e-3),
        result => {
            let messages = ["split too small", "did not converge"];
            prop_assert!(fails_with(&result, &messages), "{case:?}: {result:?}");
            Ok(())
        }
    }
}

/// Complete sums and jets below every automatic split that fail early, once the loss
/// their real-space terms predict for a component exceeds twice `MAX_LOSS` of its scale
/// in the automatic sum or jet (`EARLY_FAILURE_LOSS`), fail without the early failure
/// (`probes::late_failure`) too, or come back more than a scale off (see
/// [`off_by_its_scale`]); every other sum and jet is the same with and without it.
pub(super) fn check_early_failure(case: &Ewald) -> Result<(), TestCaseError> {
    for jet in [false, true] {
        let evaluate = |sum: &Ewald| {
            if jet {
                sum.try_derivatives()
            } else {
                sum.try_sum().map(value_only)
            }
        };
        let expected = evaluate(&case.at(Complex::default()));
        if !jet {
            prop_assume!(expected.is_ok());
        }
        let Ok(expected) = expected else {
            return Ok(());
        };
        let early = evaluate(case);
        let late = probes::late_failure(|| evaluate(case));
        match (early, late) {
            (Ok(early), late) => prop_assert!(
                matches!(&late, Ok(late) if components(late) == components(&early)),
                "{case:?}: {late:?}"
            ),
            (Err(error), Ok(late)) => prop_assert!(
                off_by_its_scale(&late, &expected, case.k),
                "{case:?}: {late:?} against {expected:?} fails early: {error}"
            ),
            (Err(error), Err(_)) => {
                prop_assert!(error.is_numerical(), "{case:?}: {error}");
            }
        }
    }
    Ok(())
}

/// A sum that vanishes by symmetry, or its Ewald `part`, is exactly zero, and at the
/// automatic split the complete sum moved half a lattice vector off by a small step `d`
/// within the lattice frame is `d` times the derivatives there (at a lattice point the
/// sums exclude the image there, which the moved sums take up).
pub(super) fn check_vanishing(
    sum: &Ewald,
    part: SumPart,
    direction: [f64; 3],
) -> Result<(), TestCaseError> {
    let value = sum.part(part);
    prop_assert!(
        matches!(value, Ok(v) if v == Complex::default()),
        "{sum:?} {part:?}: {value:?}"
    );
    let (axes, reciprocal) = (sum.axes(), sum.reciprocal());
    let at_point = (0..sum.dim).all(|i| {
        let coordinate = (0..sum.dim)
            .map(|j| reciprocal[i][j] * sum.r[axes[j]])
            .sum::<f64>()
            / TAU;
        (coordinate - coordinate.round()).abs() < 0.25
    });
    if sum.eta != Complex::default() || !matches!(part, SumPart::Full) || at_point {
        return Ok(());
    }
    let step = 1e-7;
    let mut d = [0.0; 3];
    for &axis in &axes[..sum.dim] {
        d[axis] = step * direction[axis];
    }
    let gradient = sum.derivatives().shift;
    let linear: Complex = gradient.iter().zip(d).map(|(g, d)| g * d).sum();
    let slope = gradient.iter().map(|g| g.norm()).fold(1.0, f64::max);
    let moved = Ewald {
        r: std::array::from_fn(|i| sum.r[i] + d[i]),
        ..sum.clone()
    };
    let actual = moved.part(part).unwrap();
    prop_assert!(
        (actual - linear).norm() <= 1e-5 * step * slope,
        "{sum:?}: {actual} against {linear}"
    );
    Ok(())
}

/// The value and every derivative of a sum with `Im k >= 0.8` equal those of its direct
/// shells, summed until two consecutive shells change no component by more than
/// `1e-17 (1 + |S| + |component|)`, to `tolerance (1 + |S| + |component|)` plus 32 ulps
/// of the moduli of the shell terms, which the direct sum cancels, plus 4 ulps of the
/// sum over the images `R` of `|r - R| |grad w(r - R)|`, which the rounding of their
/// unit vectors carries in; returns the direct sum and that last allowance.
pub(super) fn check_direct_jets(
    sum: &Ewald,
    tolerance: f64,
) -> Result<(Complex, f64), TestCaseError> {
    let lattice = sum.lattice();
    let (mut total, mut moduli, mut quiet) = (vec![Complex::default(); 16], [0.0; 16], 0);
    let mut sensitivity = 0.0;
    for shell in 0..120 {
        let part = SumPart::Direct(shell);
        let terms = derivatives_part(sum.wave, sum.k, &lattice, sum.r, sum.eta, part).unwrap();
        let scale = 1.0 + total[0].norm();
        let mut changed = false;
        for ((total, modulus), term) in total.iter_mut().zip(&mut moduli).zip(components(&terms)) {
            *total += term;
            *modulus += term.norm();
            changed |= term.norm() > 1e-17 * (scale + total.norm());
        }
        sensitivity += image_sensitivity(sum, &lattice, shell);
        quiet = if changed { 0 } else { quiet + 1 };
        if quiet == 2 {
            break;
        }
    }
    prop_assert_eq!(quiet, 2, "direct shells did not converge");
    let actual = components(&sum.derivatives());
    let scale = 1.0 + total[0].norm();
    let rounding = 4.0 * f64::EPSILON * sensitivity;
    for (i, ((a, e), modulus)) in actual.iter().zip(&total).zip(moduli).enumerate() {
        let tolerance = tolerance * (scale + e.norm()) + 32.0 * f64::EPSILON * modulus + rounding;
        prop_assert_close!(*a, *e, tolerance, "direct, component {i}");
    }
    Ok((total[0], rounding))
}

/// The sum of `|r - R| |grad w(r - R)|` over the images `R` of a direct shell: what the
/// images change by when their unit vectors move by one ulp. Next to a zero of its
/// angular factor an image is far more sensitive than its modulus shows: the image 0.16
/// from the shift of a degree-9 sum, 8e-6 in `cos theta` from a zero, changes by 1.4e-11
/// of itself per ulp of `cos theta`, and the Ewald and direct paths come back 4.4e-12
/// and 5.2e-13 off mpmath.
fn image_sensitivity(sum: &Ewald, lattice: &BlochLattice, shell: i64) -> f64 {
    let (dim, axes) = (sum.dim, sum.axes());
    let points = lattice::cube(dim, shell, true).unwrap();
    points
        .chunks_exact(dim)
        .map(|n| {
            // The image `R` of `w(-r - R)` is the lone image of shell 0 at the shift `r + R`.
            let n = n.iter().map(|&n| f64::from(i32::try_from(n).unwrap()));
            let mut shift = sum.r;
            for (row, n) in sum.rows.iter().zip(n) {
                for (&axis, x) in axes.iter().zip(row).take(dim) {
                    shift[axis] += n * x;
                }
            }
            let image = SumPart::Direct(0);
            let term = derivatives_part(sum.wave, sum.k, lattice, shift, sum.eta, image).unwrap();
            let gradient = term.shift.iter().map(Complex::norm_sqr).sum::<f64>().sqrt();
            shift.iter().map(|x| x * x).sum::<f64>().sqrt() * gradient
        })
        .sum()
}

/// A sum moved off its lattice plane or axis by `distance` along its `normal` matches
/// the sum on it plus its first-order change, in value and every derivative. The
/// tolerance is 1e-12 of `1 + |S| + |component|`. At an explicit split it adds 64 ulps
/// of the largest component modulus within the real part and within the reciprocal
/// part. The value-only path matches the jet's value within what the shells they leave
/// out may add: each stops its parts once two shells add less than `SHELL_TOLERANCE` of the
/// largest modulus of what it sums, the value alone or every component. So the values
/// may differ by about twice that of the largest component, beyond a rounding of 1e-13
/// of `1 + |S|`.
pub(super) fn check_tiny_normal_shift(
    sum: &Ewald,
    normal: [f64; 3],
    distance: f64,
) -> Result<(), TestCaseError> {
    let moved = |distance: f64| {
        let mut r = sum.r;
        for (r, normal) in r.iter_mut().zip(normal) {
            if normal != 0.0 {
                *r = distance * normal;
            }
        }
        Ewald { r, ..sum.clone() }
    };
    let shifted = moved(distance);
    let off = shifted.derivatives();
    let (off_jet, mut on_jet) = (components(&off), components(&sum.derivatives()));
    // A degree-9 sum 0.18 from an image changes by 1.2e17 per unit length off the plane,
    // so 1e-20 off it moves the sum by 1.2e-3 and `dS/dk` by 1.5e-2. The expected jet
    // adds that first-order change. Its slope is the Richardson extrapolation
    // `(4 D(h) - D(2h)) / 3` of central differences at h = 1e-6 and 2e-6. For degrees
    // up to 12 at 0.13 from an image, its relative error stays below 1e-17; a single
    // central difference at 1e-6 is 2e-9 off, up to 960 times the tolerance.
    if distance != 0.0 {
        let slope = |h: f64| {
            let (up, down) = (moved(h).derivatives(), moved(-h).derivatives());
            components(&up)
                .into_iter()
                .zip(components(&down))
                .map(|(up, down)| (up - down) / (2.0 * h))
                .collect::<Vec<_>>()
        };
        let (near, far) = (slope(1e-6), slope(2e-6));
        for ((on, near), far) in on_jet.iter_mut().zip(near).zip(far) {
            *on += distance * (4.0 * near - far) / 3.0;
        }
    }
    // At an explicit split the Ewald parts can cancel by thousands (2.7e3 in `dS/dk` of a
    // degree-7 sum at 2.6 times the automatic split), and the reciprocal paths on and off
    // the plane round them apart by up to 8 ulps of their moduli (each within 15 ulps of
    // mpmath); the automatic split keeps the parts near the sum. The terms of one
    // component can also cancel within a part: those of `dS/da_0x` of a degree-9 sum at
    // 3.3 times the automatic split add up to 9e3 in modulus in mpmath for a reciprocal
    // part of 66, and the paths on and off the plane come back 5.6e-11 apart. The largest
    // component of a part (2e3 there) sets the scale of its terms.
    let moduli: f64 = if sum.eta == Complex::default() {
        0.0
    } else {
        [SumPart::Real, SumPart::Reciprocal]
            .into_iter()
            .map(|part| {
                let part = components(&sum.part_derivatives(part));
                part.iter().map(|c| c.norm()).fold(0.0, f64::max)
            })
            .sum()
    };
    let scale = 1.0 + on_jet[0].norm();
    for (i, (a, e)) in off_jet.iter().zip(&on_jet).enumerate() {
        let tolerance = 1e-12 * (scale + e.norm()) + 64.0 * f64::EPSILON * moduli;
        prop_assert_close!(*a, *e, tolerance, "normal shift, component {i}");
    }
    let largest = off_jet.iter().map(|c| c.norm()).fold(1.0, f64::max);
    let tolerance = (4.0 * SHELL_TOLERANCE).mul_add(largest, 1e-13 * (1.0 + off.value.norm()));
    prop_assert_close!(shifted.sum(), off.value, tolerance);
    Ok(())
}

/// A sum at a lattice point, which excludes the image there, continues the sums shifted
/// off it (see [`check_lattice_point_limit`]), equals the absolutely convergent image sum
/// where that is cheap (see [`check_direct_sum`]) and keeps the exact derivative
/// identities (see [`check_ewald_derivative_identities`]). Unsupported wavenumbers
/// must instead be rejected, including when symmetry would make the sum zero.
pub(super) fn check_lattice_point(sum: &Ewald, direction: [f64; 3]) -> Result<(), TestCaseError> {
    if sum.k.im < 0.0 || (matches!(sum.wave, lattice::Family::Spherical { .. }) && sum.k.re < 0.0) {
        return check_rejected_wavenumber(sum);
    }
    check_lattice_point_limit(sum, direction)?;
    check_direct_sum(sum)?;
    check_ewald_derivative_identities(sum)
}

/// A degree-0 sum at a lattice point is the limit of the sums shifted off it by `r` less
/// the image there (the direct shell 0): their mean at `r = +-eps u` matches it to
/// `O(eps^2)` at the same split, within 1e-8 of `max(|S|, 1)` (at most 1.4e-9 in 3000
/// draws of [`lattice_point`]), which fails where its self term takes another sheet of
/// its incomplete gamma function. (The image, of size `1 / (k eps)` at degree 0, cancels
/// a few digits; at higher degrees it would cancel all.)
///
/// [`lattice_point`]: super::strategies::lattice_point
fn check_lattice_point_limit(sum: &Ewald, direction: [f64; 3]) -> Result<(), TestCaseError> {
    let (degree, cylindrical) = match sum.wave {
        lattice::Family::Spherical { l, .. } => (l, false),
        lattice::Family::Cylindrical { m } => (m, true),
    };
    let direction = [
        direction[0],
        direction[1],
        if cylindrical { 0.0 } else { direction[2] },
    ];
    let length = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    if degree != 0 || length < 0.1 {
        return Ok(());
    }
    let eps = 3e-5;
    let regular = |sign: f64| {
        let moved = Ewald {
            r: direction.map(|x| sign * eps * x / length),
            ..sum.clone()
        };
        moved.part(SumPart::Full).unwrap() - moved.part(SumPart::Direct(0)).unwrap()
    };
    let value = sum.sum();
    let limit = 0.5 * (regular(1.0) + regular(-1.0));
    prop_assert_close!(value, limit, 1e-8 * limit.norm().max(1.0));
    Ok(())
}

/// With `Im k >= 0.8` the image sums converge absolutely, and a sum on a chain or a 2D
/// lattice is its direct shells (which skip an image at zero distance) until two
/// consecutive shells contribute less than `1e-16 (1 + |S|)`.
pub(super) fn check_direct_sum(sum: &Ewald) -> Result<(), TestCaseError> {
    if sum.k.im < 0.8 || sum.dim > 2 {
        return Ok(());
    }
    let (mut total, mut quiet) = (Complex::default(), 0);
    for shell in 0..80 {
        let term = sum.part(SumPart::Direct(shell)).unwrap();
        total += term;
        quiet = if term.norm() < 1e-16 * (1.0 + total.norm()) {
            quiet + 1
        } else {
            0
        };
        if quiet == 2 {
            break;
        }
    }
    prop_assert_eq!(quiet, 2, "direct shells did not converge");
    let value = sum.sum();
    prop_assert_close!(value, total, 1e-11 * (1.0 + total.norm()));
    Ok(())
}

/// Ewald sums are functions of the lattice, not of its description: they and all their
/// derivatives are independent of the split, periodic in the Bloch vector
/// (`q -> q + n b_cell`, with vector derivatives picking up the moving reciprocal row),
/// and invariant under a unimodular change `a -> U a` of the primitive basis (with
/// vector derivatives mapped by `U^T`).
pub(super) fn check_ewald_invariance(
    sum: &Ewald,
    cell: usize,
    cells: i32,
    shear: [i32; 3],
    sign: bool,
) -> Result<(), TestCaseError> {
    let expected = components(&sum.derivatives());
    for eta in [sum.eta + 0.4, Complex::default()] {
        check_split(sum, eta, 1e-10)?;
    }
    // S(q, a) = S(q + n b(a), a) for every a: the vector derivative at q is the one at
    // q + n b plus the Bloch derivative there times the derivative of n b, which is
    // `-n b_q b_pj / 2pi` in component (p, q).
    let b = sum.reciprocal();
    let n = f64::from(cells);
    let shifted = Ewald {
        kpar: std::array::from_fn(|j| sum.kpar[j] + n * b[cell][j]),
        ..sum.clone()
    }
    .derivatives();
    let mut chained = shifted;
    for (p, row) in chained.vectors.iter_mut().enumerate().take(sum.dim) {
        let slope: Complex = (0..sum.dim).map(|j| shifted.kpar[j] * b[p][j]).sum();
        for (entry, b) in row.iter_mut().zip(b[cell]).take(sum.dim) {
            *entry -= n * b * slope / TAU;
        }
    }
    prop_assert_jets_close(&components(&chained), &expected, 1e-10, "Bloch cell")?;
    // A unimodular U: a signed lower unitriangular shear, reflected by `sign`.
    let u = [
        [if sign { -1.0 } else { 1.0 }, 0.0, 0.0],
        [f64::from(shear[0]), 1.0, 0.0],
        [f64::from(shear[1]), f64::from(shear[2]), 1.0],
    ];
    let rows = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i < sum.dim && j < sum.dim {
                (0..sum.dim).map(|k| u[i][k] * sum.rows[k][j]).sum()
            } else {
                0.0
            }
        })
    });
    let sheared = Ewald {
        rows,
        ..sum.clone()
    }
    .derivatives();
    let mapped = Derivatives {
        vectors: std::array::from_fn(|p| {
            std::array::from_fn(|q| (0..sum.dim).map(|i| u[i][p] * sheared.vectors[i][q]).sum())
        }),
        ..sheared
    };
    prop_assert_jets_close(&components(&mapped), &expected, 1e-10, "basis")
}

/// Point-group and lattice covariance of Ewald sums: `S(r + a_i) = exp(-i q.a_i) S(r)`,
/// joint scaling `S(k/s, q/s, s a, s r) = S`, and for orthogonal maps `O` of shift,
/// lattice and Bloch vector: rotation about z by `alpha` multiplies by
/// `exp(i m alpha)`, the mirror `y -> -y` maps `S_{l,m}` to `(-1)^m S_{l,-m}`, the
/// mirror `z -> -z` multiplies spherical sums by `(-1)^(l+m)`, and inversion by
/// `(-1)^l` (spherical) or `(-1)^m` (cylindrical). Maps that move a lattice out of
/// its frame (rotating the x axis of a 1D cylindrical lattice) are skipped.
pub(super) fn check_ewald_symmetries(
    sum: &Ewald,
    cell: usize,
    angle: f64,
    scale: f64,
) -> Result<(), TestCaseError> {
    let value = sum.sum();
    let tolerance = 1e-10 * (1.0 + value.norm());
    let (axes, a) = (sum.axes(), sum.rows[cell]);
    let mut r = sum.r;
    for j in 0..sum.dim {
        r[axes[j]] += a[j];
    }
    let phase: f64 = (0..sum.dim).map(|j| sum.kpar[j] * a[j]).sum();
    let translated = Ewald { r, ..sum.clone() }.sum();
    prop_assert_close!(translated, (-Complex::i() * phase).exp() * value, tolerance);

    let scaled = Ewald {
        rows: sum.rows.map(|row| row.map(|x| x * scale)),
        kpar: sum.kpar.map(|q| q / scale),
        r: sum.r.map(|x| x * scale),
        k: sum.k / scale,
        ..sum.clone()
    };
    prop_assert_close!(scaled.sum(), value, tolerance, "scale {scale}");

    let (l, m, spherical) = match sum.wave {
        lattice::Family::Spherical { l, m } => (l, m, true),
        lattice::Family::Cylindrical { m } => (m, m, false),
    };
    let (sin, cos) = angle.sin_cos();
    let mirrored = if spherical { sw(l, -m) } else { cw(-m) };
    let z_mirror = if spherical { parity(l + m) } else { 1.0 };
    let rotation = [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]];
    let turn = (Complex::i() * f64::from(m) * angle).exp();
    for (o, wave, factor, name) in [
        (rotation, sum.wave, turn, "rotation"),
        (
            diagonal([1.0, -1.0, 1.0]),
            mirrored,
            parity(m).into(),
            "mirror y",
        ),
        (
            diagonal([1.0, 1.0, -1.0]),
            sum.wave,
            z_mirror.into(),
            "mirror z",
        ),
        (diagonal([-1.0; 3]), sum.wave, parity(l).into(), "inversion"),
    ] {
        if let Some(image) = sum.transformed(o) {
            prop_assert_close!(image.of(wave), factor * value, tolerance, "{name}");
        }
    }
    Ok(())
}

/// Exact derivative identities of Ewald sums, including at the origin, where the
/// self term supplies the derivatives:
/// - ladders: spherical sums obey `d_z S_lm = -k (a S_{l-1,m} - b S_{l+1,m})`,
///   `(d_x + i d_y) S_lm = -k (c S_{l+1,m+1} + c' S_{l-1,m+1})` and
///   `(d_x - i d_y) S_lm = k (d S_{l+1,m-1} + d' S_{l-1,m-1})` in the shift; cylindrical
///   sums obey `(d_x +- i d_y) S_m = +-k S_{m+-1}` and do not depend on z;
/// - rotation about z generates `i m S` from the shift, and from the Bloch vector
///   and lattice rows unless the lattice is a z axis;
/// - Euler: joint scaling of lengths and inverse lengths leaves the sum unchanged.
pub(super) fn check_ewald_derivative_identities(sum: &Ewald) -> Result<(), TestCaseError> {
    let d = sum.derivatives();
    let [dx, dy, dz] = d.shift;
    let (k, i) = (sum.k, Complex::i());
    let scale = 1.0 + d.value.norm() + dx.norm() + dy.norm() + dz.norm();
    let (m, raising, lowering) = match sum.wave {
        lattice::Family::Spherical { l, m } => {
            let c = |numerator: i32, denominator: i32| {
                (f64::from(numerator) / f64::from(denominator))
                    .max(0.0)
                    .sqrt()
            };
            let wave = |l, m| sum.of(sw(l, m));
            let (below, above) = ((2 * l - 1) * (2 * l + 1), (2 * l + 1) * (2 * l + 3));
            let axial = -k
                * (c((l - m) * (l + m), below) * wave(l - 1, m)
                    - c((l + m + 1) * (l - m + 1), above) * wave(l + 1, m));
            prop_assert_close!(dz, axial, 1e-10 * (scale + axial.norm()), "d_z");
            let raising = -k
                * (c((l + m + 1) * (l + m + 2), above) * wave(l + 1, m + 1)
                    + c((l - m) * (l - m - 1), below) * wave(l - 1, m + 1));
            let lowering = k
                * (c((l - m + 1) * (l - m + 2), above) * wave(l + 1, m - 1)
                    + c((l + m) * (l + m - 1), below) * wave(l - 1, m - 1));
            (m, raising, lowering)
        }
        lattice::Family::Cylindrical { m } => {
            prop_assert_eq!(dz, Complex::default());
            let wave = |m| sum.of(cw(m));
            (m, k * wave(m + 1), -k * wave(m - 1))
        }
    };
    let tolerance = |expected: Complex| 1e-10 * (scale + expected.norm());
    prop_assert_close!(dx + i * dy, raising, tolerance(raising), "raising");
    prop_assert_close!(dx - i * dy, lowering, tolerance(lowering), "lowering");

    let turn = |v: &[f64; 3], g: &[Complex; 3]| v[0] * g[1] - v[1] * g[0];
    let mut generator = turn(&sum.r, &d.shift);
    let mut magnitude = scale;
    if sum.dim >= 2 {
        generator += turn(&sum.kpar, &d.kpar);
        for (row, g) in sum.rows.iter().zip(&d.vectors).take(sum.dim) {
            generator += turn(row, g);
        }
        magnitude += d
            .kpar
            .iter()
            .chain(d.vectors.iter().flatten())
            .map(|g| g.norm())
            .sum::<f64>();
    }
    if sum.dim >= 2 || matches!(sum.wave, lattice::Family::Spherical { .. }) {
        let expected = i * f64::from(m) * d.value;
        prop_assert_close!(generator, expected, 1e-10 * magnitude, "rotation");
    }

    let (lengths, spectral) = euler(sum, &d);
    let tolerance = 1e-10 * (1.0 + d.value.norm() + spectral.norm());
    prop_assert_close!(lengths, spectral, tolerance, "Euler");
    Ok(())
}

/// The forward evaluation of every part (complete, real-space, reciprocal-space and
/// three direct shells), with its special paths for shifts on a lattice's axis or
/// plane, equals the value of the jet evaluation used for derivatives; and the real
/// and reciprocal parts (with the self correction) at another split add up to the
/// complete sum.
pub(super) fn check_ewald_forward_paths(sum: &Ewald) -> Result<(), TestCaseError> {
    let full = sum.sum();
    let resplit = sum.at(sum.eta + 0.4);
    let parts = resplit.part(SumPart::Real).unwrap() + resplit.part(SumPart::Reciprocal).unwrap();
    prop_assert_close!(parts, full, 1e-10 * (1.0 + full.norm()), "resplit parts");
    for (part, tolerance) in [
        (SumPart::Full, 1e-13),
        (SumPart::Real, 1e-13),
        (SumPart::Reciprocal, 1e-13),
        (SumPart::Direct(0), 1e-12),
        (SumPart::Direct(1), 1e-12),
        (SumPart::Direct(2), 1e-12),
    ] {
        let (forward, jets) = (sum.part(part).unwrap(), sum.part_derivatives(part));
        prop_assert_close!(
            forward,
            jets.value,
            tolerance * (1.0 + forward.norm()),
            "{part:?}"
        );
    }
    Ok(())
}

/// The forward evaluation of the complete sum and of its real-space and
/// reciprocal-space parts (with the self correction), with their special paths for
/// shifts on a lattice's axis or plane, equals the value of the jet evaluation, and the
/// parts add up to the complete sum at the same split.
pub(super) fn check_forward_parts(sum: &Ewald) -> Result<(), TestCaseError> {
    let full = sum.part(SumPart::Full).unwrap();
    let scale = 1.0 + full.norm();
    let mut parts = Complex::default();
    for part in [SumPart::Full, SumPart::Real, SumPart::Reciprocal] {
        let (forward, jets) = (sum.part(part).unwrap(), sum.part_derivatives(part));
        prop_assert_close!(forward, jets.value, 1e-13 * scale, "{part:?}");
        if !matches!(part, SumPart::Full) {
            parts += forward;
        }
    }
    prop_assert_close!(parts, full, 1e-12 * scale, "parts");
    Ok(())
}

/// Ewald sums obey the Euler identity of joint position, lattice, wavenumber and
/// Bloch scaling, and their complete derivative in every parameter matches Ridders'
/// extrapolation `D` of central differences ([`ridders`]) along a direction inside the
/// supported wavenumber domain, within `2e-7 (1 + |D|)` plus twice its error estimate.
/// Draws whose estimate exceeds half of that base, `1e-7 (1 + |D|)`, where the
/// differences cannot resolve the derivative to it, are rejected, so properties check
/// it last. Both bounds scale with `D`, not with the analytic derivative under test, so
/// that a wrongly small one fails rather than being rejected; a moved sum that comes
/// back non-finite fails too.
///
/// It checks the analytic derivative, not the values: values that are noisy or jump
/// within the steps raise the estimate, and such draws are mostly rejected rather than
/// failed. With relative noise of 1e-10 in the value-only sums it rejected 65% of
/// `ewald_at` draws and failed none (a fixed step of 1e-5 failed 94%); the properties
/// that compare value-only sums with jets or across splits failed from noise of 1e-11
/// on. The steps keep clear of thresholds and images, but move `k` and the lattice by up
/// to the longest step, [`FIRST_STEP`] along the direction, which turns `(k eta)^2` too:
/// splits at the edge of the supported domain lie outside what the check can evaluate,
/// and at the 89.9 degrees of a `SPLITS` row the moved sums do not converge. No property
/// checks derivatives at such splits.
pub(super) fn check_ewald_derivative(sum: &Ewald) -> Result<(), TestCaseError> {
    let d = sum.derivatives();
    let (lengths, spectral) = euler(sum, &d);
    prop_assert_close!(lengths, spectral, 1e-9 * (1.0 + d.value.norm()));
    let cylindrical = matches!(sum.wave, lattice::Family::Cylindrical { .. });
    // The differences reach `FIRST_STEP`. Keep each wavenumber component fixed near its
    // domain boundary: Im(k) = 0, and Re(k) = 0 for spherical sums.
    let dk = c(
        if cylindrical || sum.k.re > 0.1 * FIRST_STEP {
            0.1
        } else {
            0.0
        },
        if sum.k.im > 0.2 * FIRST_STEP {
            0.2
        } else {
            0.0
        },
    );
    // A sum at the origin leaves out the image there; any step in the shift brings it
    // back, so the shift stays fixed.
    let dr = if sum.r.iter().all(|&x| x == 0.0) {
        [0.0; 3]
    } else {
        [0.1, -0.07, if cylindrical { 0.0 } else { 0.05 }]
    };
    let dq = leading(sum.dim, |j| [0.13, -0.05, 0.08][j]);
    let directions = [[0.2, 0.05, -0.03], [-0.04, 0.1, 0.06], [0.03, -0.02, 0.15]];
    let da: [[f64; 3]; 3] = embed(directions, sum.dim);
    let direction = dk * d.k
        + dot_c(&dr, &d.shift)
        + dot_c(&dq, &d.kpar)
        + da.iter()
            .zip(&d.vectors)
            .map(|(a, g)| dot_c(a, g))
            .sum::<Complex>();
    let moved = |step: f64| Ewald {
        k: sum.k + step * dk,
        r: std::array::from_fn(|j| sum.r[j] + step * dr[j]),
        kpar: std::array::from_fn(|j| sum.kpar[j] + step * dq[j]),
        rows: std::array::from_fn(|i| std::array::from_fn(|j| sum.rows[i][j] + step * da[i][j])),
        ..sum.clone()
    };
    // Next to a diffraction threshold or an image a sum varies like `ln x` or `x^-p` in
    // the distance `x` to it, and the central differences expand in powers of `(h / x)^2`
    // (6.7e-6 off at h = 1e-5 on a chain at `|k_q| = 0.1`), a series that converges only
    // for `h < x`. The steps start a quarter of the way to the nearest such point, so that
    // even the longest lies well inside that radius, where the terms eventually fall by
    // `(1 / 4)^2` per order, and the shortest 83 times inside. The quarter is a margin,
    // not tuned; the draws below ran with it.
    let first = FIRST_STEP.min(0.25 * singular_distance(moved));
    let (numerical, error) = ridders(first, |step| {
        let value = moved(step).sum();
        // `ridders` would drop a NaN silently; the library returns `Error::NonFinite`.
        assert!(value.is_finite(), "non-finite sum {value} at step {step}");
        value
    });
    // Where the estimate exceeded 1e-8 of the scale, the error stayed within 2.5 times
    // it, and elsewhere within 5e-8 of the scale: with twice the estimate on top of the
    // base, the worst of 1.7e7 draws of `rotated_chain`, `far_chain` and `ewald_at`
    // outside the band of `skip_threshold_band` (half of them moved toward thresholds or
    // images) came to 0.32 of the tolerance. Draws whose estimate exceeds half the base
    // are rejected: 2 of 8e6 `rotated_chain` draws outside that band, 1 of 2e6 `ewald_at`
    // draws and none of 8e6 `far_chain` draws. On the scale of the analytic derivative
    // instead, it rejected zeroed jets rather than failing them in 3.6% of the
    // `rotated_chain` draws it checks.
    let scale = 1.0 + numerical.norm();
    prop_assume!(error <= 1e-7 * scale);
    prop_assert_close!(numerical, direction, 2.0f64.mul_add(error, 2e-7 * scale));
    Ok(())
}

/// The longest first step of [`check_ewald_derivative`], which moves `k` by 2.2e-3: the
/// sums round by up to 1.6e-11 of `|S|` (degree-8 chains at explicit splits), which its
/// shortest step, 4.8e-4, turns into 3.3e-8 of `|S|`.
const FIRST_STEP: f64 = 1e-2;

/// Ridders' extrapolation of the central differences of `f` at 0 (Numerical Recipes,
/// `dfridr`): the Neville tableau of `f'(0)` in powers of `h^2` over 10 steps from `h`,
/// each 1.4 times shorter. Returns the entry that differs least from the two it
/// extrapolates and, as its error estimate, the largest of that difference and its
/// distances from the last diagonal entry and from the answer of `dfridr`, the best
/// entry before the diagonal first moves by twice the difference.
///
/// Each guards against entries that agree by chance. Where the `h^2` and `h^4` terms of
/// the longest steps still cancel, `dfridr`, which stops there and returns the
/// difference alone, came back 7 times its estimate off for a sum 0.026 from an image
/// and failed 351 of 1e6 sums 1e-7 to 0.1 from an image. Where the rounding of the
/// shortest steps agrees, the whole tableau came back 120 times off for a degree-8 chain
/// next to a threshold. Once both agreed on an answer 13 times their estimate off; the
/// last diagonal entry did not. `f` must stay finite: the comparisons drop a NaN silently.
fn ridders(h: f64, f: impl Fn(f64) -> Complex) -> (Complex, f64) {
    let shrink = 1.4;
    let (mut best, mut error) = (Complex::default(), f64::INFINITY);
    let mut stopped = None;
    let mut previous: Vec<Complex> = Vec::new();
    let mut step = h;
    for _ in 0..10 {
        let mut column = vec![central(step, &f)];
        let mut factor = shrink * shrink;
        for (j, &left) in previous.iter().enumerate() {
            let next = (column[j] * factor - left) / (factor - 1.0);
            factor *= shrink * shrink;
            let change = (next - column[j]).norm().max((next - left).norm());
            if change <= error {
                (best, error) = (next, change);
            }
            column.push(next);
        }
        if let (Some(&last), Some(&diagonal)) = (column.last(), previous.last())
            && (last - diagonal).norm() >= 2.0 * error
        {
            stopped.get_or_insert(best);
        }
        previous = column;
        step /= shrink;
    }
    let last = *previous.last().expect("a column for every step");
    let spread = [last, stopped.unwrap_or(best)].map(|other| (best - other).norm());
    (best, error.max(spread[0]).max(spread[1]))
}

/// The distance along `moved`, in its parameter, to the nearest point where the sums are
/// singular: a diffraction threshold or empty-lattice pole, where `k^2 - |q + G|^2`
/// vanishes for a reciprocal vector `G` ([`order_gaps`]), or an image `R` of the cells
/// around the origin other than one at the shift, where `r - R` vanishes. Each moves
/// linearly at this scale, so its distance is its modulus over its rate of change.
fn singular_distance(moved: impl Fn(f64) -> Ewald) -> f64 {
    let delta = 1e-6;
    let [sum, up, down] = [0.0, delta, -delta].map(moved);
    let reach = order_reach(&sum);
    let distance = |x: &[f64], rate: &[f64]| {
        let size = x.iter().map(|x| x * x).sum::<f64>().sqrt();
        size / (rate.iter().map(|x| x * x).sum::<f64>().sqrt() / (2.0 * delta))
    };
    let mut nearest = f64::INFINITY;
    let gaps = [&sum, &up, &down].map(|sum| order_gaps(sum, reach));
    for (((gap, _), (ahead, _)), (behind, _)) in gaps[0].iter().zip(&gaps[1]).zip(&gaps[2]) {
        let rate = [ahead.re - behind.re, ahead.im - behind.im];
        nearest = nearest.min(distance(&[gap.re, gap.im], &rate));
    }
    let offsets = [&sum, &up, &down].map(image_offsets);
    for ((offset, ahead), behind) in offsets[0].iter().zip(&offsets[1]).zip(&offsets[2]) {
        if offset.iter().any(|&x| x != 0.0) {
            let rate: [f64; 3] = std::array::from_fn(|j| ahead[j] - behind[j]);
            nearest = nearest.min(distance(offset, &rate));
        }
    }
    nearest
}

/// The offsets `r - R` of the shift of `sum` from the images `R` of the cells around the
/// origin (1D spherical lattices lie along z).
fn image_offsets(sum: &Ewald) -> Vec<[f64; 3]> {
    let along_z = sum.dim == 1 && matches!(sum.wave, lattice::Family::Spherical { .. });
    cells(sum.dim, 1)
        .map(|n| {
            let point: [f64; 3] =
                std::array::from_fn(|j| (0..3).map(|i| n[i] * sum.rows[i][j]).sum());
            let point = if along_z { [0.0, 0.0, point[0]] } else { point };
            std::array::from_fn(|j| sum.r[j] - point[j])
        })
        .collect()
}

/// Each Ewald part at a fixed split, and each direct shell, obeys the Euler identity
/// of joint length and inverse-length scaling on its own.
pub(super) fn check_ewald_part_euler(sum: &Ewald) -> Result<(), TestCaseError> {
    for part in [SumPart::Real, SumPart::Reciprocal, SumPart::Direct(2)] {
        let d = sum.part_derivatives(part);
        let (lengths, spectral) = euler(sum, &d);
        prop_assert_close!(lengths, spectral, 2e-9 * (1.0 + d.value.norm()), "{part:?}");
    }
    Ok(())
}

/// Unaccelerated direct shells of a strongly damped wave converge to the Ewald sum.
pub(super) fn check_direct_shells(sum: &Ewald) -> Result<(), TestCaseError> {
    let sum = Ewald {
        k: c(sum.k.re, 1.6),
        eta: Complex::default(),
        ..sum.clone()
    };
    let total: Complex = (0..15)
        .map(|shell| sum.part(SumPart::Direct(shell)).unwrap())
        .sum();
    let ewald = sum.sum();
    prop_assert_close!(total, ewald, 2e-10 * (1.0 + ewald.norm()));
    Ok(())
}

/// The jets of `sum` and of the same sum at the split `eta` agree (see
/// [`prop_assert_jets_close`]).
pub(super) fn check_split(sum: &Ewald, eta: Complex, tolerance: f64) -> Result<(), TestCaseError> {
    let (actual, expected) = (sum.at(eta).derivatives(), sum.derivatives());
    prop_assert_jets_close(
        &components(&actual),
        &components(&expected),
        tolerance,
        "split",
    )
}

/// Asserts that two jets agree in value within `tolerance` of `max(|S|, 1)` and in each
/// derivative within `tolerance` of its scale (see [`component_scales`]).
pub(super) fn prop_assert_jets_on_their_scales(
    actual: &Derivatives,
    expected: &Derivatives,
    k: Complex,
    tolerance: f64,
) -> Result<(), TestCaseError> {
    let pairs = components(actual).into_iter().zip(components(expected));
    for (i, ((a, e), scale)) in pairs.zip(component_scales(expected, k)).enumerate() {
        prop_assert!(
            (a - e).norm() <= tolerance * scale,
            "component {i}: {a} against {e} ({actual:?} against {expected:?})"
        );
    }
    Ok(())
}

/// Asserts that two sums agree in value and every derivative, each relative to
/// `1 + |value| + |component|`.
pub(super) fn prop_assert_jets_close(
    actual: &[Complex],
    expected: &[Complex],
    tolerance: f64,
    context: &str,
) -> Result<(), TestCaseError> {
    let scale = 1.0 + expected[0].norm();
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        prop_assert_close!(
            *a,
            *e,
            tolerance * (scale + e.norm()),
            "{context}, component {i}"
        );
    }
    Ok(())
}

/// Whether the jet `late` is more than the scale of some component off the jet
/// `expected`: `|S' - S| > max(|S|, 1)` in value, or `|dS' - dS| + s |S' - S| >
/// |dS| + max(|S|, 1) s` in a derivative (see [`component_scales`]).
fn off_by_its_scale(late: &Derivatives, expected: &Derivatives, k: Complex) -> bool {
    let scale = expected.value.norm().max(1.0);
    let value = (late.value - expected.value).norm();
    let units = component_units(k);
    value > scale
        || components(late)
            .iter()
            .zip(components(expected))
            .zip(units)
            .skip(1)
            .any(|((&a, e), unit)| {
                unit.mul_add(value, (a - e).norm()) > scale.mul_add(unit, e.norm())
            })
}

/// Value and every derivative of a sum: `k`, shift, Bloch vector and lattice rows.
pub(super) fn components(d: &Derivatives) -> Vec<Complex> {
    [d.value, d.k]
        .into_iter()
        .chain(d.shift)
        .chain(d.kpar)
        .chain(d.vectors.into_iter().flatten())
        .collect()
}

/// The scale `s` of each of the [`components`] of a sum at the wavenumber `k`: `1 / |k|`
/// for `k` and the Bloch vector, `|k|` for the shift and the lattice rows (and one for the
/// value).
fn component_units(k: Complex) -> [f64; 16] {
    let (length, inverse) = (k.norm(), 1.0 / k.norm());
    std::array::from_fn(|i| match i {
        0 => 1.0,
        1 | 5..=7 => inverse,
        _ => length,
    })
}

/// The scale of each of the [`components`] of the sum `expected`: `max(|S|, 1)` for the
/// value and `|dS| + max(|S|, 1) s` for a derivative with the scale `s` of
/// [`component_units`], on which the Ewald sums count their losses.
fn component_scales(expected: &Derivatives, k: Complex) -> Vec<f64> {
    let scale = expected.value.norm().max(1.0);
    let units = component_units(k);
    let components = components(expected).into_iter().zip(units).enumerate();
    components
        .map(|(i, (e, unit))| {
            if i == 0 {
                scale
            } else {
                scale.mul_add(unit, e.norm())
            }
        })
        .collect()
}

/// A sum with the value `value` and no derivatives.
fn value_only(value: Complex) -> Derivatives {
    let zero = Complex::default();
    Derivatives {
        value,
        k: zero,
        eta: zero,
        shift: [zero; 3],
        kpar: [zero; 3],
        vectors: [[zero; 3]; 3],
    }
}

/// Whether `result` failed numerically ([`crate::Error::is_numerical`]) with one of `messages`.
pub(super) fn fails_with<T>(result: &crate::Result<T>, messages: &[&str]) -> bool {
    matches!(result, Err(error) if error.is_numerical() && messages.iter().any(|m| error.to_string().contains(m)))
}

/// `Σ_j v_j g_j` of a real vector and complex derivatives.
fn dot_c(v: &[f64; 3], g: &[Complex; 3]) -> Complex {
    v.iter().zip(g).map(|(v, g)| v * g).sum()
}

/// The Euler combination `r.d_r S + Σ_i a_i.d_(a_i) S` of length derivatives, and the
/// matching `k d_k S + q.d_q S` of inverse lengths; joint scaling makes them equal.
fn euler(sum: &Ewald, d: &Derivatives) -> (Complex, Complex) {
    let lengths = dot_c(&sum.r, &d.shift)
        + sum
            .rows
            .iter()
            .zip(&d.vectors)
            .map(|(a, g)| dot_c(a, g))
            .sum::<Complex>();
    (lengths, sum.k * d.k + dot_c(&sum.kpar, &d.kpar))
}

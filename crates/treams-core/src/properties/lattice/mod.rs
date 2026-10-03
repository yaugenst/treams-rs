//! Lattice sums and periodic expansions.

mod checks;
mod ewald;
mod periodic;
mod strategies;
mod tables;

use std::f64::consts::{PI, TAU};

use proptest::prelude::*;

use self::{
    checks::{
        check_chain, check_direct_jets, check_direct_shells, check_direct_sum, check_early_failure,
        check_ewald_derivative, check_ewald_derivative_identities, check_ewald_forward_paths,
        check_ewald_invariance, check_ewald_part_euler, check_ewald_symmetries,
        check_forward_parts, check_lattice_point, check_off_axis_chain, check_reduced_integrals,
        check_rejected_wavenumber, check_small_split, check_split, check_tiny_normal_shift,
        check_vanishing, components, fails_with, prop_assert_jets_close,
        prop_assert_jets_on_their_scales, spectral_chain_sum,
    },
    ewald::{Ewald, c, cw, cylinder_point, diagonal, explicit_split, pinned, sw},
    periodic::{
        check_cylindrical_periodic_axial_scale, check_lattice_table_adjoint,
        check_periodic_conversion_scale,
    },
    strategies::{
        ewald, ewald_at, far_chain, half_integer_sum, lattice_point, near_chain, near_point_sum,
        normal_shift, off_plane_sum, reflected_zero_sum, rotated_chain, shift, slope,
        small_split_sum, tiny_distance, turned_chain_sum, vanishing_sum,
    },
    tables::{RECORDED, SPLITS, ZEROS},
};
use crate::{
    Complex,
    lattice::{HALF_INTEGER_SERIES_T, SumPart, probes},
    sw::Mode,
    test_support::{
        DEFAULT_CASES, EXPENSIVE_CASES, complex_matrix, complex_vec, degree_order, log_uniform,
        prop_assert_close, table,
    },
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

    #[test]
    fn lattice_table_linear_adjoint(
        modes in prop::collection::vec(degree_order(1..4), 4),
        polarizations in prop::array::uniform4(0_u8..2),
        origins in prop::array::uniform2(prop::array::uniform3(-0.3_f64..0.3)),
        helicity in any::<bool>(),
        channels in 1_usize..=2,
        table in complex_vec(2 * 2 * 49, 1.0),
        g in complex_matrix(2, 2, 1.0),
    ) {
        let modes: Vec<_> = modes
            .iter()
            .zip(polarizations)
            .map(|(&(l, m), pol)| Mode { l, m, pol })
            .collect();
        let channels = if helicity { channels } else { 1 };
        check_lattice_table_adjoint(&modes, origins, helicity, channels, &table, &g)?;
    }

    /// The spectral series of 1D spherical sums against the Ewald sum at a split along
    /// `1 / k` (see `check_chain`), for Bloch vectors up to two zones out.
    #[test]
    fn chain_series_matches_the_ewald_sum(
        (l, m) in degree_order(0..25),
        (kr, ki) in (0.5_f64..6.0, prop_oneof![Just(0.0), 0.0_f64..0.8]),
        period in 0.8_f64..2.5,
        (kpar, zone) in (-0.5_f64..0.5, -2_i32..=2),
        place in (0.3_f64..2.5, -3.2_f64..3.2, -1.0_f64..1.0),
    ) {
        let k = Complex::new(kr, ki);
        let kpar = (kpar + f64::from(zone)) * (TAU / period);
        check_chain(&near_chain((l, m), k, period, kpar, place), Some(k.norm() / k))?;
    }

    /// As `chain_series_matches_the_ewald_sum` at real splits and splits rotated off
    /// `1 / k`, for `Im k` of either sign or nearly zero and Bloch vectors that include 0
    /// and `+-2 pi / a`. There the principal branch of a Kambe integral can differ from
    /// the root `Im k_q >= 0` that the series takes, and supported sums must take that
    /// root. Gain wavenumbers must be rejected.
    #[test]
    fn ewald_sums_take_the_series_root_at_every_split(
        (l, m) in degree_order(0..13),
        (kr, slope) in (0.5_f64..6.0, slope()),
        period in 0.8_f64..2.5,
        (kpar, zone) in (prop_oneof![Just(0.0), -0.5_f64..0.5], -1_i32..=1),
        rotation in prop_oneof![Just(None), (-0.3_f64..0.3).prop_map(Some)],
        place in (0.3_f64..2.5, -3.2_f64..3.2, -1.0_f64..1.0),
    ) {
        let k = Complex::new(kr, kr * slope);
        let kpar = (kpar + f64::from(zone)) * (TAU / period);
        let sum = near_chain((l, m), k, period, kpar, place);
        check_chain(&sum, Some(explicit_split(k, 1.0, rotation)))?;
    }

    /// 1D spherical sums at explicit splits (see `rotated_chain`) match the automatic
    /// split in value and every derivative, on both sides of `w = 2.5`, from which they
    /// try their spectral series first; their derivatives match central differences
    /// inside the supported domain, and each part obeys the Euler identity. Gain
    /// wavenumbers must be rejected.
    #[test]
    fn chain_sums_do_not_depend_on_the_split(sum in rotated_chain()) {
        if sum.k.im < 0.0 {
            check_rejected_wavenumber(&sum)?;
        } else {
            check_split(&sum.at(Complex::default()), sum.eta, 1e-10)?;
            check_ewald_derivative(&sum)?;
            check_ewald_part_euler(&sum)?;
        }
    }

    /// 2D spherical and 1D cylindrical sums (see `half_integer_sum`), whose reciprocal
    /// parts take gamma functions and Kambe integrals of half-integer degree on the sheet
    /// of the automatic split, match that split in value and every derivative, their
    /// forward and jet paths agree, and they equal their direct sums where those converge.
    #[test]
    fn plane_and_axis_sums_do_not_depend_on_the_split(sum in half_integer_sum()) {
        check_split(&sum.at(Complex::default()), sum.eta, 1e-10)?;
        check_forward_parts(&sum)?;
        check_direct_sum(&sum)?;
    }

    /// The two paths of the reduced integrals (see `check_reduced_integrals`) agree for
    /// propagating and evanescent orders of real and lossy `k` at real and rotated
    /// splits: the integer orders of 1D spherical sums across both switches between the
    /// paths, and the closed-form pair of the half-integer chain around
    /// [`HALF_INTEGER_SERIES_T`].
    #[test]
    fn reduced_integrals_agree_across_the_series_threshold(
        (twice_n, modulus) in prop_oneof![
            ((-12_i32..=0).prop_map(|n| 2 * n), log_uniform(-3.0..0.6)),
            (
                prop_oneof![Just(3), Just(1), Just(-1)],
                log_uniform(-0.48..0.48).prop_map(|scale| scale * HALF_INTEGER_SERIES_T),
            ),
        ],
        (kr, slope) in (0.5_f64..6.0, slope().prop_map(f64::abs)),
        beta in prop_oneof![0.0_f64..0.95, 1.05_f64..3.0],
        (size, rotation) in (0.6_f64..1.3, prop_oneof![Just(None), (-0.3_f64..0.3).prop_map(Some)]),
    ) {
        let k = Complex::new(kr, kr * slope);
        let eta = explicit_split(k, size, rotation);
        check_reduced_integrals(twice_n, k, (beta * kr).powi(2), eta, modulus)?;
    }

    /// 2D spherical and 1D cylindrical sums of degree and order up to 12 off their plane
    /// or axis (see `off_plane_sum`), across the band where their reciprocal parts pass
    /// from the series in `t` to the Kambe chain, equal their direct sums in value and
    /// every derivative to 5e-13 (at most 2.2e-13 measured).
    #[test]
    fn half_integer_sums_match_their_direct_sums_off_the_plane(sum in off_plane_sum()) {
        check_direct_jets(&sum, 5e-13)?;
    }

    /// 1D spherical sums at the automatic split, whichever of the Ewald sum and the
    /// spectral series each component comes from, match the plain Ewald sum where it keeps
    /// its accuracy in value and every derivative (see `check_chain`).
    #[test]
    fn chain_sums_keep_their_accuracy_at_the_automatic_split(
        (l, m) in degree_order(0..25),
        (kr, ki) in (0.5_f64..4.0, prop_oneof![Just(0.0), 0.0_f64..0.5]),
        period in 0.8_f64..2.5,
        kpar in -0.5_f64..0.5,
        place in (0.3_f64..3.0, -3.2_f64..3.2, -0.5_f64..0.5),
    ) {
        let k = Complex::new(kr, ki);
        check_chain(&near_chain((l, m), k, period, kpar * TAU / period, place), None)?;
    }

    /// Far off the axis, where 1D spherical sums take their spectral series, they keep the
    /// exact derivative identities, the lattice and point symmetries, their central
    /// differences inside the supported domain, agreeing forward and jet
    /// paths, and the Euler identity of each Ewald part.
    #[test]
    fn far_off_axis_chains_keep_the_ewald_identities(sum in far_chain()) {
        check_ewald_derivative_identities(&sum)?;
        check_ewald_symmetries(&sum, 0, 0.7, 1.3)?;
        check_ewald_derivative(&sum)?;
        check_ewald_forward_paths(&sum)?;
        check_ewald_part_euler(&sum)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(EXPENSIVE_CASES))]

    #[test]
    fn cylindrical_periodic_axial_scale_adjoint(kz in -0.3_f64..0.3, a in 1.5_f64..1.8) {
        check_cylindrical_periodic_axial_scale(kz, a)?;
    }

    #[test]
    fn periodic_conversion_scale_adjoint(
        k in 1.1_f64..2.0,
        kz in -0.7_f64..0.7,
        period in 1.2_f64..2.2,
        x in -0.3_f64..0.3,
        helicity in any::<bool>(),
    ) {
        check_periodic_conversion_scale(k, kz, period, x, helicity)?;
    }

    /// Sums at real splits of 0.11 to 0.3, below every automatic one, where the parts
    /// cancel by up to `e^41` (see `small_split_sum` and `check_small_split`).
    #[test]
    fn small_splits_keep_their_sums_or_fail(case in small_split_sum(0.11..0.3)) {
        check_small_split(&case, false)?;
    }

    /// Sums next to a lattice point below every automatic split, whose real-space terms
    /// add in phase (see `near_point_sum` and `check_small_split`).
    #[test]
    fn small_splits_keep_the_sums_next_to_a_lattice_point_or_fail(case in near_point_sum()) {
        check_small_split(&case, false)?;
    }

    /// Jets of sums that vanish by a reflection, whose derivatives normal to the plane or
    /// axis do not, below every automatic split (see `reflected_zero_sum` and
    /// `check_small_split`).
    #[test]
    fn small_split_jets_keep_the_derivatives_of_vanishing_sums_or_fail(
        case in reflected_zero_sum(),
    ) {
        check_small_split(&case, true)?;
    }

    /// 1D spherical jets below every automatic split turned 80 to 90 degrees off `1 / k`,
    /// whose far real-space terms peak up to hundreds of periods out (see
    /// `turned_chain_sum` and `check_small_split`).
    #[test]
    fn turned_small_split_jets_keep_their_derivatives_or_fail(case in turned_chain_sum()) {
        check_small_split(&case, true)?;
    }

    /// The early failure of sums below every automatic split (see `check_early_failure`) at
    /// real splits of 0.06 to 0.13, where most of the parts cancel beyond use.
    #[test]
    fn hopeless_small_splits_fail_as_their_complete_sums_would(
        case in small_split_sum(0.06..0.13),
    ) {
        check_early_failure(&case)?;
    }

    /// Sums that vanish by symmetry (see `vanishing_sum` and `check_vanishing`).
    #[test]
    fn sums_that_vanish_by_symmetry_are_zero_and_the_limit_of_their_neighbours(
        (sum, part) in vanishing_sum(),
        direction in prop::array::uniform3(-1.0_f64..1.0),
    ) {
        check_vanishing(&sum, part, direction)?;
    }

    #[test]
    fn ewald_complete_derivative(sum in ewald_at(shift())) {
        check_ewald_derivative(&sum)?;
    }

    #[test]
    fn ewald_parts_obey_the_euler_identity(sum in ewald()) {
        check_ewald_part_euler(&sum)?;
    }

    #[test]
    fn direct_shells_converge_to_ewald(sum in ewald()) {
        check_direct_shells(&sum)?;
    }

    #[test]
    fn ewald_derivative_identities(sum in ewald()) {
        check_ewald_derivative_identities(&sum)?;
    }

    #[test]
    fn ewald_forward_paths_match_the_jets(sum in ewald(), in_frame in any::<bool>()) {
        check_ewald_forward_paths(&if in_frame { sum.in_frame() } else { sum })?;
    }

    #[test]
    fn ewald_sums_are_lattice_covariant_and_point_symmetric(
        sum in ewald(),
        cell in 0_usize..3,
        angle in -3.2_f64..3.2,
        scale in log_uniform(-0.5..0.5),
    ) {
        check_ewald_symmetries(&sum, cell % sum.dim, angle, scale)?;
    }

    #[test]
    fn ewald_sums_depend_on_the_lattice_not_its_description(
        sum in ewald(),
        cell in 0_usize..3,
        cells in -12_i32..=12,
        shear in prop::array::uniform3(-8_i32..=8),
        sign in any::<bool>(),
    ) {
        check_ewald_invariance(&sum, cell % sum.dim, cells, shear, sign)?;
    }

    /// Sums at a lattice point at every split and for `k` in every quadrant and on both
    /// axes (see `lattice_point` and `check_lattice_point`); unsupported wavenumbers
    /// must be rejected.
    #[test]
    fn lattice_point_sums_continue_the_shifted_sums(
        sum in lattice_point(),
        direction in prop::array::uniform3(-1.0_f64..1.0),
    ) {
        check_lattice_point(&sum, direction)?;
    }

    /// Sums shifted off their lattice plane or axis by a tiny distance tend to the sums on
    /// it (see `normal_shift` and `check_tiny_normal_shift`).
    #[test]
    fn sums_tend_to_their_values_on_the_plane_and_axis(
        (sum, normal) in normal_shift(),
        distance in tiny_distance(),
    ) {
        check_tiny_normal_shift(&sum, normal, distance)?;
    }

    /// Degree-0 chain sums at the automatic split against their spectral series
    /// (`spectral_chain_sum`), from near the axis to `w = k rho eta` of about 20.
    #[test]
    fn off_axis_chains_match_the_spectral_series(
        k in 0.5_f64..8.0,
        period in 0.8_f64..2.0,
        kpar in -0.5_f64..0.5,
        distance in 0.05_f64..8.0,
        (azimuth, along) in (-3.2_f64..3.2, -0.5_f64..0.5),
    ) {
        let r = cylinder_point(distance * period, azimuth, along * period);
        check_off_axis_chain(k, period, kpar * TAU / period, r)?;
    }
}

/// Sums against the high-precision references of `references/lattice_sums.txt` (see
/// its header): the value of each part and, where asked, of the jet and its `k` and
/// Bloch derivatives, each within `tolerance * max(|S|, 1)` of its reference.
#[test]
fn sums_match_high_precision_references() {
    for (sum, fields, values) in pinned::<f64>(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/references/lattice_sums.txt"
    ))) {
        let [part, jet, tolerance] = &fields[..] else {
            panic!("reference fields {fields:?}");
        };
        let part = match part.as_str() {
            "full" => SumPart::Full,
            "real" => SumPart::Real,
            _ => SumPart::Reciprocal,
        };
        let tolerance: f64 = tolerance.parse().unwrap();
        let expected: Vec<Complex> = values
            .as_chunks::<2>()
            .0
            .iter()
            .map(|x| c(x[0], x[1]))
            .collect();
        let mut actual = vec![sum.part(part).unwrap()];
        if jet == "1" {
            let jet = sum.derivatives();
            actual.extend([jet.value, jet.k, jet.kpar[0]]);
        }
        // The jet's value has the reference of the value.
        let expected = std::iter::once(expected[0]).chain(expected);
        for (i, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).norm() <= tolerance * expected.norm().max(1.0),
                "{sum:?} {part:?}, component {i}: {actual} against {expected}"
            );
        }
    }
}

/// 1D spherical sums 2 to 5 periods off the axis (`references/lattice_chain.txt`),
/// where the Ewald sums with the automatic split cancel and the sums take their spectral
/// series: values within 2e-14 of `max(|S|, 1)` and derivatives within 5e-14 of
/// `|dS| + max(|S|, 1) s`, with the automatic split and the split 1.
#[test]
fn far_off_axis_chains_match_high_precision_references() {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/references/lattice_chain.txt"
    ))
    .lines()
    .filter(|line| !line.starts_with('#'))
    .collect::<Vec<_>>()
    .join("\n");
    let cases = table::<f64, f64>(&text);
    assert_eq!(cases.len(), 27);
    for (key, values) in cases {
        let [l, m, kr, ki, kpar, period, x, y, z] = key[..] else {
            panic!("reference key {key:?}");
        };
        let k = c(kr, ki);
        #[allow(clippy::cast_possible_truncation)] // Integral labels.
        let wave = sw(l as i32, m as i32);
        let expected: Vec<Complex> = values
            .as_chunks::<2>()
            .0
            .iter()
            .map(|x| c(x[0], x[1]))
            .collect();
        let scale = expected[0].norm().max(1.0);
        for eta in [Complex::default(), c(1.0, 0.0)] {
            let sum = Ewald::chain(wave, period, kpar, [x, y, z], k, eta);
            let d = sum.derivatives();
            assert_eq!(sum.sum(), d.value, "{sum:?}");
            let [dx, dy, dz] = d.shift;
            let actual = [d.value, d.k, d.kpar[0], dx, dy, dz, d.vectors[0][0]];
            for (i, (actual, expected)) in actual.into_iter().zip(&expected).enumerate() {
                let tolerance = match i {
                    0 => 2e-14 * scale,
                    1 | 2 => 5e-14 * (expected.norm() + scale / k.norm()),
                    _ => 5e-14 * (expected.norm() + scale * k.norm()),
                };
                assert!(
                    (actual - expected).norm() <= tolerance,
                    "{sum:?}, component {i}: {actual} against {expected}"
                );
            }
        }
    }
}

/// 1D spherical sums of degree up to 3 at `w = k rho eta` from 11 to 30, where the Ewald
/// sums cancel and the sums take their spectral series: the value and the jet agree
/// exactly, the Ewald parts at the split 1 add up to the complete sum, and degree 0
/// matches `spectral_chain_sum`, each to 1e-13 of `max(|S|, 1)`.
#[test]
fn far_off_axis_chains_keep_every_path() {
    let automatic = Complex::default();
    let mut cases = vec![
        (6.0, [6.0, 0.0, 0.3], automatic),
        (2.0, [8.0, 0.0, 0.3], automatic),
        (3.3, [3.4, 1.0, 0.0], c(1.0, 0.0)),
    ];
    cases.extend([4.0, 6.0, 8.0].map(|x| (12.0, [x, 0.0, 0.3], automatic)));
    for (k, r, eta) in cases {
        for (l, m) in (0..4).flat_map(|l| (-l..=l).map(move |m| (l, m))) {
            let sum = Ewald::chain(sw(l, m), 1.7, 0.3, r, c(k, 0.0), eta);
            let value = sum.sum();
            assert_eq!(value, sum.derivatives().value, "{sum:?}");
            let resplit = sum.at(c(1.0, 0.0));
            let parts =
                resplit.part(SumPart::Real).unwrap() + resplit.part(SumPart::Reciprocal).unwrap();
            let scale = value.norm().max(1.0);
            assert!(
                (parts - value).norm() <= 1e-13 * scale,
                "{sum:?}: parts {parts} against {value}"
            );
            if l == 0 {
                let expected = spectral_chain_sum(k, 1.7, 0.3, r);
                assert!(
                    (value - expected).norm() <= 1e-13 * scale,
                    "{sum:?}: {value} against {expected}"
                );
            }
        }
    }
}

/// The sums of [`SPLITS`] against the automatic split: `value` (or, for a sum that may
/// fail instead, `maybe`) within `tolerance * max(|S|, 1)`, `jet` within `tolerance` of
/// `1 + |S| + |component|` ([`prop_assert_jets_close`]) and `scales`, below every
/// automatic split (`Re(1 / (2 eta^2)) > 16 / pi`), within `tolerance` of the scales of
/// the components ([`prop_assert_jets_on_their_scales`]); `fails` the complete sum, the
/// jet or the real-space part with the message after the colon.
#[test]
fn explicit_splits_agree_with_the_automatic_split_or_fail() {
    for (sum, fields, message) in pinned::<String>(SPLITS) {
        let (fields, message) = (
            fields.iter().map(String::as_str).collect::<Vec<_>>(),
            message.join(" "),
        );
        let automatic = sum.at(Complex::default());
        let tolerance = || fields[1].parse::<f64>().unwrap();
        match fields[0] {
            "value" | "maybe" => match sum.try_sum() {
                Ok(value) => {
                    let expected = automatic.sum();
                    let tolerance = tolerance() * expected.norm().max(1.0);
                    assert!(
                        (value - expected).norm() < tolerance,
                        "{sum:?}: {value} against {expected}"
                    );
                }
                Err(error) => assert!(fields[0] == "maybe", "{sum:?}: {error}"),
            },
            "jet" => check_split(&automatic, sum.eta, tolerance()).unwrap(),
            "scales" => {
                // Below every automatic split.
                assert!((0.5 / (sum.eta * sum.eta)).re > 16.0 / PI, "{sum:?}");
                let (actual, expected) = (sum.derivatives(), automatic.derivatives());
                prop_assert_jets_on_their_scales(&actual, &expected, sum.k, tolerance()).unwrap();
            }
            _ => {
                let result = match fields[1] {
                    "jet" => sum.try_derivatives().map(|d| d.value),
                    "real" => sum.part(SumPart::Real),
                    _ => sum.try_sum(),
                };
                assert!(fails_with(&result, &[&message]), "{sum:?}: {result:?}");
            }
        }
    }
}

/// Sums that vanish by symmetry ([`ZEROS`]: odd degrees half a lattice vector from a
/// lattice point at a zero Bloch vector, or odd under a reflection in the lattice plane or
/// axis) are exactly zero at every split, as are their Ewald parts and the value and `k`
/// derivative of their automatic jets, which the symmetry keeps at every `k`. Their jets
/// match the automatic ones (`jet tolerance`, see [`prop_assert_jets_close`]) or fail
/// with the message after the colon ("non-finite Ewald summand" at splits turned 90
/// degrees or more off `1 / k`, `Re((k eta)^2) < 0`).
#[test]
fn sums_that_vanish_by_symmetry_are_exactly_zero() {
    for (sum, fields, message) in pinned::<String>(ZEROS) {
        let (fields, message) = (
            fields.iter().map(String::as_str).collect::<Vec<_>>(),
            message.join(" "),
        );
        let parts: &[SumPart] = if sum.eta == Complex::default() {
            &[SumPart::Full]
        } else {
            &[SumPart::Full, SumPart::Real, SumPart::Reciprocal]
        };
        for &part in parts {
            assert_eq!(
                sum.part(part).unwrap(),
                Complex::default(),
                "{sum:?} {part:?}"
            );
        }
        let automatic = sum.at(Complex::default()).derivatives();
        assert_eq!(
            [automatic.value, automatic.k],
            [Complex::default(); 2],
            "{sum:?}"
        );
        if let ["jet", tolerance] = fields[..] {
            let jet = components(&sum.derivatives());
            let tolerance = tolerance.parse().unwrap();
            prop_assert_jets_close(&jet, &components(&automatic), tolerance, "zero").unwrap();
        } else {
            if message.contains("non-finite") {
                // Turned 90 degrees or more off `1 / k`.
                assert!((sum.k * sum.eta).powi(2).re < 0.0, "{sum:?}");
            }
            let result = sum.try_derivatives();
            assert!(fails_with(&result, &[&message]), "{sum:?}: {result:?}");
        }
    }
}

/// The properties above at the pinned inputs of [`RECORDED`], shrunk failures and probes:
/// `normal x y z [distance]` ([`check_tiny_normal_shift`], at zero distance by default),
/// `direct tolerance` ([`check_direct_jets`], and the value-only sum within
/// `tolerance * max(|S|, 1)` of the direct sum plus the rounding of the images' unit
/// vectors), `point x y z` ([`check_lattice_point`]) and `derivative`
/// ([`check_ewald_derivative`]).
#[test]
fn recorded_cases_keep_their_properties() {
    for (sum, fields, _) in pinned::<String>(RECORDED) {
        let numbers: Vec<f64> = fields[1..].iter().map(|x| x.parse().unwrap()).collect();
        let vector = || [numbers[0], numbers[1], numbers[2]];
        match fields[0].as_str() {
            "normal" => check_tiny_normal_shift(&sum, vector(), *numbers.get(3).unwrap_or(&0.0)),
            "direct" => check_direct_jets(&sum, numbers[0]).and_then(|(direct, rounding)| {
                let tolerance = numbers[0] * direct.norm().max(1.0) + rounding;
                prop_assert_close!(sum.sum(), direct, tolerance, "value-only");
                Ok(())
            }),
            "derivative" => check_ewald_derivative(&sum),
            _ => check_lattice_point(&sum, vector()),
        }
        .unwrap_or_else(|error| panic!("{sum:?}: {error}"));
    }
}

/// Degree-0 sums at a lattice point on the sheets of `k eta` (see [`check_lattice_point`]):
/// spherical waves on 1D and 3D and cylindrical waves on 2D rectangular lattices, for `k`
/// in every quadrant, on
/// both imaginary half-axes and with a zero real or imaginary part of either sign, at the
/// automatic split, splits of modulus 0.7 rotated off `1 / k` by 0.2 either way and, for
/// `Re k != 0`, the real split 0.7 or, for imaginary `k`, the imaginary splits with a
/// zero real part of either sign (spherical ones where `Re(k eta) > 0`). Unsupported
/// wavenumbers must be rejected.
#[test]
fn lattice_point_sums_take_the_sheet_of_k_eta() {
    let wavenumbers = [
        c(1.2, 0.05),
        c(1.2, -0.05),
        c(-1.2, 0.3),
        c(-1.2, -0.05),
        c(-1.2, -0.0),
        c(0.0, 1.0),
        c(0.0, -1.0),
        c(-0.0, 1.0),
    ];
    let imaginary = [c(0.0, 0.7), c(0.0, -0.7), c(-0.0, 0.7), c(-0.0, -0.7)];
    for (wave, dim) in [(sw(0, 0), 1), (sw(0, 0), 3), (cw(0), 2)] {
        let spherical = dim != 2;
        let direction = if spherical {
            [0.6, 0.48, 0.64]
        } else {
            [0.6, 0.8, 0.0]
        };
        for &k in &wavenumbers {
            let mut splits = vec![Complex::default()];
            splits.extend([0.2, -0.2].map(|angle| explicit_split(k, 0.7, Some(angle))));
            if k.re == 0.0 {
                splits.extend(
                    imaginary
                        .into_iter()
                        .filter(|eta| !spherical || (k * eta).re > 0.0),
                );
            } else {
                splits.push(c(0.7, 0.0));
            }
            for eta in splits {
                let rows = diagonal([1.7, 1.5, 1.3]);
                let sum = Ewald::new(wave, dim, rows, [0.3, 0.1, 0.2], [0.0; 3], k, eta);
                check_lattice_point(&sum, direction)
                    .unwrap_or_else(|error| panic!("{sum:?}: {error}"));
            }
        }
    }
}

/// Complete sums and jets below every automatic split whose real-space terms predict a
/// loss beyond `EARLY_FAILURE_LOSS` fail with "split too small" (in a derivative, for
/// jets of sums that vanish by symmetry) before their far shells: after less than 1% of
/// the `127^3` terms of their grown real-space limit (`probes::real_terms`), and at
/// larger `|k|`, whose shells are few, after fewer terms than without the early failure
/// (`probes::late_failure`).
#[test]
fn hopeless_small_splits_fail_before_their_far_shells() {
    let terms = |sum: &Ewald, jet: bool, late: bool| {
        let counter = probes::real_terms();
        let evaluate = || {
            if jet {
                sum.try_derivatives().map(|d| d.value)
            } else {
                sum.try_sum()
            }
        };
        let result = if late {
            probes::late_failure(evaluate)
        } else {
            evaluate()
        };
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("split too small") && (!jet || error.contains("in a derivative")),
            "{sum:?} late={late}: {error}"
        );
        counter.count()
    };
    let full = 127_u64.pow(3);
    let rows = [[1.0, 0.0, 0.0], [0.2, 1.1, 0.0], [0.1, -0.1, 0.9]];
    let (wave, eta) = (sw(3, 1), c(0.1, 0.0));
    let sum = Ewald::new(
        wave,
        3,
        rows,
        [0.3, -0.4, 0.2],
        [0.3, 0.2, 0.1],
        c(0.7, 0.1),
        eta,
    );
    let early = terms(&sum, false, false);
    assert!(100 * early < full, "{early} of {full} terms");
    for k in [c(12.0, 0.5), c(8.0, 0.5)] {
        let sum = Ewald { k, ..sum.clone() };
        let (early, late) = (terms(&sum, false, false), terms(&sum, false, true));
        assert!(early < late, "k={k}: {early} against {late} terms");
    }
    for eta in [0.12, 0.08, 0.05] {
        let r = [0.5, 0.0, 0.0];
        let vanishing = Ewald {
            kpar: [0.0; 3],
            r,
            eta: c(eta, 0.0),
            ..sum.clone()
        };
        let count = terms(&vanishing, true, false);
        assert!(100 * count < full, "eta={eta}: {count} of {full} terms");
    }
}

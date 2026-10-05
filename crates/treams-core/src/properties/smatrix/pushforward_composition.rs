//! Directional derivatives and adjoint pairings of scattering-matrix compositions.
//!
//! treams-rs extension.

use super::*;
use crate::saved::SavedState;
use crate::test_support::patterned;

/// Array-only callbacks preserve all derivative state, and reject corrupt lengths
/// and oversized dimension headers before allocating the restored matrices.
fn restored<S: SavedState>(residual: &S, expected_size: usize) -> S {
    let bytes = residual.save_state().unwrap();
    assert_eq!(bytes.len(), expected_size);
    assert!(S::from_state(&bytes[..bytes.len() - 1]).is_err());
    let mut extended = bytes.clone();
    extended.push(0);
    assert!(S::from_state(&extended).is_err());
    let mut oversized = bytes.clone();
    oversized[..8].fill(u8::MAX);
    assert!(S::from_state(&oversized).is_err());
    let state = S::from_state(&bytes).unwrap();
    assert_eq!(state.save_state().unwrap(), bytes);
    state
}

#[test]
fn saved_state_sizes_reject_empty_and_overflowing_dimensions() {
    for n in [0, usize::MAX] {
        assert!(smatrix::AddResidual::state_size(n).is_err());
        assert!(smatrix::PeriodicResidual::state_size(n).is_err());
        assert!(smatrix::BandsResidual::state_size(n).is_err());
        assert!(smatrix::IlluminateResidual::state_size(n, 1).is_err());
        assert!(smatrix::IlluminateResidual::state_size(1, n).is_err());
        assert!(smatrix::FromArrayResidual::state_size(n, 1).is_err());
        assert!(smatrix::FromArrayResidual::state_size(1, n).is_err());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

    #[test]
    fn composition_and_illumination_pushforwards_match_values_and_pullbacks(
        (lower, upper, dlower, dupper, incoming, dincoming, g, fields_g)
            in (1_usize..=3, 1_usize..=3).prop_flat_map(|(n, p)| (
                scattering(n), scattering(n),
                prop::array::uniform4(complex_matrix(n, n, 0.3)),
                prop::array::uniform4(complex_matrix(n, n, 0.3)),
                prop::array::uniform2(complex_matrix(n, p, 0.5)),
                prop::array::uniform2(complex_matrix(n, p, 0.3)),
                prop::array::uniform4(complex_matrix(n, n, 0.5)),
                prop::array::uniform4(complex_matrix(n, p, 0.5)),
            )),
    ) {
        check_composition_pushforward(&lower, &upper, &dlower, &dupper, &g)?;
        check_illumination_pushforward(
            &lower, &upper, &dlower, &dupper, &incoming, &dincoming, &fields_g,
        )?;
    }

    #[test]
    fn array_pushforward_matches_values_and_pullback(
        response in complex_matrix(3, 3, 0.5),
        channels in prop::array::uniform4(complex_matrix(3, 2, 0.5)),
        dresponse in complex_matrix(3, 3, 0.3),
        dchannels in prop::array::uniform4(complex_matrix(3, 2, 0.3)),
        g in prop::array::uniform4(complex_matrix(2, 2, 0.5)),
    ) {
        check_array_pushforward(&response, &channels, &dresponse, &dchannels, &g)?;
    }

    #[test]
    fn periodic_pushforward_matches_values_and_pullback(
        (blocks, direction, g) in (1_usize..=3).prop_flat_map(|n| (
            scattering(n),
            prop::array::uniform4(complex_matrix(n, n, 0.3)),
            complex_matrix(2 * n, 2 * n, 0.5),
        )),
    ) {
        check_periodic_pushforward(&blocks, &direction, &g)?;
    }
}

fn shifted<const N: usize>(
    value: &[DMatrix<Complex>; N],
    direction: &[DMatrix<Complex>; N],
    step: f64,
) -> [DMatrix<Complex>; N] {
    std::array::from_fn(|b| &value[b] + &direction[b] * Complex::new(step, 0.0))
}

fn pairing<const N: usize>(a: &[DMatrix<Complex>; N], b: &[DMatrix<Complex>; N]) -> f64 {
    a.iter().zip(b).map(|(a, b)| re_dot(a, b)).sum()
}

/// Every output component agrees with a centered value difference. The O(h²)
/// truncation and O(epsilon/h) rounding terms are below this bound at h = 1e-5.
fn check_value_difference<const N: usize>(
    tangent: &[DMatrix<Complex>; N],
    value: impl Fn(f64) -> [DMatrix<Complex>; N],
) -> Result<(), TestCaseError> {
    let plus = value(1e-5);
    let minus = value(-1e-5);
    for ((actual, plus), minus) in tangent.iter().zip(plus).zip(minus) {
        let expected = (plus - minus) * Complex::new(0.5e5, 0.0);
        prop_assert_close!(actual, &expected, 2e-8 * (1.0 + actual.norm()));
    }
    Ok(())
}

/// The tangent solves the linearized internal field equations and is adjoint to
/// the existing pullback under the native complex pairing.
fn check_composition_pushforward(
    lower: &Blocks,
    upper: &Blocks,
    dlower: &Blocks,
    dupper: &Blocks,
    g: &Blocks,
) -> Result<(), TestCaseError> {
    let (_, residual) = smatrix::add(lower.clone(), upper.clone()).unwrap();
    let tangent = residual.pushforward(dlower, dupper).unwrap();
    let gradient = residual.pullback(g).unwrap();
    let loaded = restored(
        &residual,
        smatrix::AddResidual::state_size(lower[0].nrows()).unwrap(),
    );
    prop_assert_eq!(&loaded.pushforward(dlower, dupper).unwrap(), &tangent);
    let loaded_gradient = loaded.pullback(g).unwrap();
    prop_assert_eq!(&loaded_gradient.lower, &gradient.lower);
    prop_assert_eq!(&loaded_gradient.upper, &gradient.upper);
    prop_assert_eq!(&residual.pushforward(dlower, dupper).unwrap(), &tangent);
    check_value_difference(&tangent, |h| {
        smatrix::add(shifted(lower, dlower, h), shifted(upper, dupper, h))
            .unwrap()
            .0
    })?;
    let reverse = pairing(&gradient.lower, dlower) + pairing(&gradient.upper, dupper);
    prop_assert_close!(pairing(g, &tangent), reverse, 2e-12 * (1.0 + reverse.abs()));
    // A transparent lower layer leaves every upper-layer sensitivity unchanged.
    let n = upper[0].nrows();
    let zero = std::array::from_fn(|_| DMatrix::zeros(n, n));
    let (_, transparent) = smatrix::add(identity_blocks(n), upper.clone()).unwrap();
    let through = transparent.pushforward(&zero, dupper).unwrap();
    for (actual, expected) in through.iter().zip(dupper) {
        prop_assert_close!(actual, expected, 2e-14 * (1.0 + expected.norm()));
    }
    Ok(())
}

/// Arbitrary stack and incident tangents reproduce all four fields and pair with
/// their pullback, including unequal mode and illumination counts.
fn check_illumination_pushforward(
    lower: &Blocks,
    upper: &Blocks,
    dlower: &Blocks,
    dupper: &Blocks,
    incoming: &[DMatrix<Complex>; 2],
    dincoming: &[DMatrix<Complex>; 2],
    g: &[DMatrix<Complex>; 4],
) -> Result<(), TestCaseError> {
    let record = || {
        smatrix::illuminate(
            lower.each_ref().map(view),
            upper.each_ref().map(view),
            incoming.clone(),
        )
        .unwrap()
    };
    let residual = record().1;
    let tangent = residual.pushforward(dlower, dupper, dincoming).unwrap();
    let lower_rows = lower.each_ref().map(DMatrix::transpose);
    let upper_rows = upper.each_ref().map(DMatrix::transpose);
    let row_residual = smatrix::illuminate(
        lower_rows.each_ref().map(|a| view(a).transpose()),
        upper_rows.each_ref().map(|a| view(a).transpose()),
        incoming.clone(),
    )
    .unwrap()
    .1;
    let (n, columns) = row_residual.shape();
    let loaded = restored(
        &row_residual,
        smatrix::IlluminateResidual::state_size(n, columns).unwrap(),
    );
    let row_major = loaded.pushforward(dlower, dupper, dincoming).unwrap();
    let row_gradient = row_residual.pullback(g).unwrap();
    let loaded_gradient = loaded.pullback(g).unwrap();
    prop_assert_eq!(loaded_gradient.lower, row_gradient.lower);
    prop_assert_eq!(loaded_gradient.upper, row_gradient.upper);
    prop_assert_eq!(loaded_gradient.incoming, row_gradient.incoming);
    for (column, row) in tangent.iter().zip(row_major) {
        prop_assert_close!(column, &row, 2e-14 * (1.0 + column.norm()));
    }
    let gradient = residual.pullback(g).unwrap();
    prop_assert_eq!(
        &residual.pushforward(dlower, dupper, dincoming).unwrap(),
        &tangent
    );
    check_value_difference(&tangent, |h| {
        smatrix::illuminate_value(
            shifted(lower, dlower, h).each_ref().map(view),
            shifted(upper, dupper, h).each_ref().map(view),
            shifted(incoming, dincoming, h).each_ref().map(view),
        )
        .unwrap()
    })?;
    let reverse = pairing(&gradient.lower, dlower)
        + pairing(&gradient.upper, dupper)
        + pairing(&gradient.incoming, dincoming);
    prop_assert_close!(pairing(g, &tangent), reverse, 2e-12 * (1.0 + reverse.abs()));
    Ok(())
}

/// Emission uses a transpose rather than a Hermitian transpose; complex channel
/// tangents exercise that distinction in all four outgoing blocks.
fn check_array_pushforward(
    response: &DMatrix<Complex>,
    channels: &smatrix::Channels,
    dresponse: &DMatrix<Complex>,
    dchannels: &smatrix::Channels,
    g: &Blocks,
) -> Result<(), TestCaseError> {
    let record = || smatrix::from_array(response.clone(), channels.clone()).unwrap();
    let residual = record().1;
    let tangent = residual.pushforward(dresponse, dchannels).unwrap();
    let gradient = residual.pullback(g).unwrap();
    let loaded = restored(
        &residual,
        smatrix::FromArrayResidual::state_size(response.nrows(), channels[0].ncols()).unwrap(),
    );
    prop_assert_eq!(&loaded.pushforward(dresponse, dchannels).unwrap(), &tangent);
    let loaded_gradient = loaded.pullback(g).unwrap();
    prop_assert_eq!(&loaded_gradient.response, &gradient.response);
    prop_assert_eq!(&loaded_gradient.channels, &gradient.channels);
    prop_assert_eq!(
        &residual.pushforward(dresponse, dchannels).unwrap(),
        &tangent
    );
    check_value_difference(&tangent, |h| {
        smatrix::from_array(
            response + dresponse * Complex::new(h, 0.0),
            shifted(channels, dchannels, h),
        )
        .unwrap()
        .0
    })?;
    let reverse = re_dot(&gradient.response, dresponse) + pairing(&gradient.channels, dchannels);
    prop_assert_close!(pairing(g, &tangent), reverse, 2e-12 * (1.0 + reverse.abs()));
    Ok(())
}

/// Tangents of both the transfer solve's operator and right-hand side are active.
fn check_periodic_pushforward(
    blocks: &Blocks,
    direction: &Blocks,
    g: &DMatrix<Complex>,
) -> Result<(), TestCaseError> {
    let record = || smatrix::periodic(blocks.clone()).unwrap();
    let residual = record().1;
    let tangent = residual.pushforward(direction).unwrap();
    let gradient = residual.pullback(g).unwrap();
    let loaded = restored(
        &residual,
        smatrix::PeriodicResidual::state_size(blocks[0].nrows()).unwrap(),
    );
    prop_assert_eq!(&loaded.pushforward(direction).unwrap(), &tangent);
    prop_assert_eq!(&loaded.pullback(g).unwrap(), &gradient);
    prop_assert_eq!(&residual.pushforward(direction).unwrap(), &tangent);
    check_value_difference(std::array::from_ref(&tangent), |h| {
        [smatrix::periodic(shifted(blocks, direction, h)).unwrap().0]
    })?;
    let reverse = pairing(&gradient, direction);
    prop_assert_close!(re_dot(g, &tangent), reverse, 2e-12 * (1.0 + reverse.abs()));
    Ok(())
}

#[test]
fn band_pushforward_matches_wavenumbers_vectors_and_pullback() -> Result<(), TestCaseError> {
    // Distinct multipliers and unique phase pivots keep the eigenpair convention
    // smooth while all four blocks and the repeat distance vary together.
    let blocks = [
        Complex::new(0.9, 0.2),
        Complex::new(0.13, 0.05),
        Complex::new(-0.07, 0.03),
        Complex::new(1.1, -0.3),
    ]
    .map(|z| DMatrix::from_element(1, 1, z));
    let direction =
        std::array::from_fn(|b| patterned(1, 1, f64::from(u32::try_from(b).unwrap()) * 0.3));
    let (period, dperiod) = (1.7, 0.23);
    let record = || smatrix::bands(blocks.clone(), period).unwrap();
    let residual = record();
    let (dk, dv) = residual.pushforward(&direction, dperiod).unwrap();
    check_value_difference(std::array::from_ref(&dv), |h| {
        [
            smatrix::bands(shifted(&blocks, &direction, h), period + h * dperiod)
                .unwrap()
                .vectors()
                .clone(),
        ]
    })?;
    let plus = smatrix::bands(shifted(&blocks, &direction, 1e-5), period + 1e-5 * dperiod).unwrap();
    let minus =
        smatrix::bands(shifted(&blocks, &direction, -1e-5), period - 1e-5 * dperiod).unwrap();
    for ((actual, plus), minus) in dk.iter().zip(plus.wavenumbers()).zip(minus.wavenumbers()) {
        prop_assert_close!(*actual, (plus - minus) / 2e-5, 2e-8 * (1.0 + actual.norm()));
    }
    let gk = [Complex::new(0.1, -0.3), Complex::new(0.7, 0.2)];
    let gv = patterned(2, 2, 0.8);
    let gradient = residual.pullback(&gk, gv.clone()).unwrap();
    let loaded = restored(&residual, smatrix::BandsResidual::state_size(1).unwrap());
    let (loaded_dk, loaded_dv) = loaded.pushforward(&direction, dperiod).unwrap();
    prop_assert_eq!(&loaded_dk, &dk);
    prop_assert_eq!(&loaded_dv, &dv);
    let loaded_gradient = loaded.pullback(&gk, gv.clone()).unwrap();
    prop_assert_eq!(&loaded_gradient.blocks, &gradient.blocks);
    prop_assert_eq!(loaded_gradient.period.to_bits(), gradient.period.to_bits());
    let (dk_again, dv_again) = residual.pushforward(&direction, dperiod).unwrap();
    prop_assert_eq!(&dk_again, &dk);
    prop_assert_eq!(&dv_again, &dv);
    let forward = re_dot(gk, dk) + re_dot(&gv, &dv);
    let reverse = pairing(&gradient.blocks, &direction) + gradient.period * dperiod;
    prop_assert_close!(forward, reverse, 2e-12 * (1.0 + reverse.abs()));
    Ok(())
}

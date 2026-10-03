//! Cartesian cylindrical-wave translation and analytic derivatives.
//!
//! treams-rs extension: treams evaluates translations at polar arguments only.
#![allow(clippy::indexing_slicing)] // Fixed-size Cartesian axes and jet slots.

use super::{Mode, transverse, transverse_wavenumber};
use crate::{
    Complex, Error, Result,
    basis::ModeLabel,
    numerics::{Jet, finite},
    special::{Radial, SERIES_RADIUS, cylindrical_radial},
};

/// Translation coefficient and its continuous derivatives, with the axial wavenumbers of
/// both modes moving together.
#[derive(Clone, Copy, Debug, Default)]
pub struct CartesianTranslation {
    /// Complex coefficient.
    pub value: Complex,
    /// Derivatives of `value` with respect to the displacement components.
    pub position: [Complex; 3],
    /// Derivative of `value` with respect to the medium wavenumber `k`.
    pub k: Complex,
    /// Derivative of `value` with respect to the common axial wavenumber `kz`.
    pub kz: Complex,
}

/// Translation coefficient of the source mode `from` to the destination mode `to`, with
/// its derivatives.
///
/// `position` is the destination position minus the source position. Distinct axial
/// wavenumbers and polarizations decouple, and singular waves give zero at the origin.
///
/// treams-rs extension: the Cartesian single-pair jet behind the expansion matrices and
/// the tests of the polar coefficients; `treams.cw.translate` takes polar arguments.
#[allow(clippy::float_cmp)] // kz labels and the self origin are exact mode identities.
pub fn cartesian_translation(
    to: Mode,
    from: Mode,
    k: Complex,
    position: [f64; 3],
    radial: Radial,
) -> Result<CartesianTranslation> {
    to.validate()?;
    from.validate()?;
    if !finite(k) || k.norm_sqr() == 0.0 || position.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "require a finite nonzero wavenumber and finite displacement".into(),
        ));
    }
    if to.kz != from.kz
        || to.pol != from.pol
        || (radial == Radial::Singular && position == [0.0; 3])
    {
        return Ok(CartesianTranslation::default());
    }
    let order = from.m - to.m;
    let kz = from.kz;
    let [x, y, z] = position;
    let rho = x.hypot(y);
    let krho = transverse_wavenumber(k, kz);
    if krho.norm_sqr() == 0.0 {
        return Err(Error::InvalidInput(
            "cylindrical cutoff requires a limiting formulation".into(),
        ));
    }
    if radial == Radial::Regular && (krho * rho).norm() < SERIES_RADIUS {
        let kz = Jet::<5>::variable(kz, 4);
        let wave = regular_harmonic(
            order,
            transverse(Jet::variable(k, 3), kz),
            std::array::from_fn(|i| Jet::variable(position[i], i)),
            kz,
        );
        return Ok(CartesianTranslation {
            value: wave.value,
            position: std::array::from_fn(|i| wave.derivative[i]),
            k: wave.derivative[3],
            kz: wave.derivative[4],
        });
    }
    // Here rho > 0: regular waves on the axis took the series above, and outgoing
    // waves are singular there.
    let phi = y.atan2(x);
    let phase = (Complex::i() * (f64::from(order) * phi + kz * z)).exp();
    let radial_jet = cylindrical_radial(order, krho * rho, radial)?;
    let value = radial_jet.value * phase;
    let dr = radial_jet.first * krho * phase;
    let azimuthal = Complex::i() * f64::from(order) * (value / rho);
    Ok(CartesianTranslation {
        value,
        position: [
            dr * (x / rho) - azimuthal * (y / rho),
            dr * (y / rho) + azimuthal * (x / rho),
            Complex::i() * kz * value,
        ],
        k: radial_jet.first * rho * k / krho * phase,
        kz: -radial_jet.first * rho * kz / krho * phase + Complex::i() * z * value,
    })
}

/// Regular `J_m` harmonic as a Cartesian power series, for `|k_rho rho|` below
/// [`SERIES_RADIUS`].
/// Factoring the azimuthal polynomial keeps derivatives regular on the cylinder axis.
pub(crate) fn regular_harmonic<const N: usize>(
    order: i32,
    transverse: Jet<N>,
    r: [Jet<N>; 3],
    kz: Jet<N>,
) -> Jet<N> {
    let m = order.abs();
    let azimuth = r[0] + Complex::i() * if order < 0 { -r[1] } else { r[1] };
    let sign = if order < 0 && m % 2 == 1 { -1.0 } else { 1.0 };
    let step = -0.25 * transverse * transverse * (r[0] * r[0] + r[1] * r[1]);
    // The radial series depends on one scalar: differentiate it once, then
    // apply the Cartesian chain rule after summation instead of at every term.
    let step_norm = step.derivative.iter().map(|g| g.norm()).fold(0.0, f64::max);
    let mut term = Complex::new((-crate::special::log_factorial(m)).exp(), 0.0);
    let mut slope_term = Complex::default();
    let mut sum = term;
    let mut slope = Complex::default();
    for q in 1..32 {
        let denominator = f64::from(q * (m + q));
        if N > 0 {
            slope_term = (slope_term * step.value + term) / denominator;
            slope += slope_term;
        }
        term = term * step.value / denominator;
        sum += term;
        let term_norm = term.norm().max(slope_term.norm() * step_norm);
        let sum_norm = sum.norm().max(slope.norm() * step_norm);
        if term_norm <= f64::EPSILON * sum_norm {
            break;
        }
    }
    sign * (0.5 * transverse * azimuth).powi(m)
        * step.chain(sum, slope)
        * (Complex::i() * kz * r[2]).exp()
}

#[cfg(test)]
mod tests {
    //! The axis series and the Cartesian derivatives against Bessel values and order
    //! recurrences. Identities of translations are in `properties/waves.rs`.

    use super::{Complex, Jet, Mode, Radial, cartesian_translation, regular_harmonic};
    use crate::test_support::{DEFAULT_CASES, complex, prop_assert_close, radial};
    use proptest::{prelude::*, test_runner::TestCaseError};

    proptest! {
        // Above the default, so that each of the 25 orders appears about four times.
        #![proptest_config(ProptestConfig::with_cases(96))]
        #[test]
        fn regular_harmonic_value_and_cartesian_derivatives(
            order in -12_i32..13, x in -0.25_f64..0.25, y in -0.25_f64..0.25,
            z in -1.0_f64..1.0, kz in -0.5_f64..0.5, loss in -0.2_f64..0.2,
        ) {
            check_regular_harmonic(order, [x, y, z], kz, loss)?;
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

        #[test]
        fn translation_derivative_recurrences(
            order in -8_i32..=8,
            rho in prop_oneof![Just(0.0), 0.01_f64..0.35, 0.45_f64..2.0],
            phi in -3.1_f64..3.1,
            z in -1.0_f64..1.0,
            kz in -0.5_f64..0.5,
            k in complex(0.1).prop_map(|k| k + 1.2),
            radial in radial(),
        ) {
            check_derivative_recurrences(order, [rho, phi, z], kz, k, radial)?;
        }
    }

    /// The Cartesian power series of `J_m` and its position, `k_rho` and `kz`
    /// derivatives match Bessel values and their order recurrences, also on the axis.
    fn check_regular_harmonic(
        order: i32,
        [x, y, z]: [f64; 3],
        kz: f64,
        loss: f64,
    ) -> Result<(), TestCaseError> {
        let transverse = Complex::new(1.1, loss);
        for position in [[x, y, z], [1e-100, -1e-100, z], [0.0, 0.0, z]] {
            let wave = regular_harmonic(
                order,
                Jet::<5>::variable(transverse, 3),
                std::array::from_fn(|i| Jet::variable(position[i], i)),
                Jet::variable(kz, 4),
            );
            let rho = position[0].hypot(position[1]);
            let phase = |m: i32| {
                (Complex::i() * (f64::from(m) * position[1].atan2(position[0]) + kz * z)).exp()
            };
            let reference = |m: i32| {
                let sign = if m < 0 && m.abs() % 2 == 1 { -1.0 } else { 1.0 };
                sign * complex_bessel::besselj(f64::from(m.abs()), transverse * rho).unwrap()
                    * phase(m)
            };
            let [lower, value, upper] = [order - 1, order, order + 1].map(reference);
            let dx = 0.5 * transverse * (lower - upper);
            let dy = 0.5 * Complex::i() * transverse * (lower + upper);
            let expected = [
                dx,
                dy,
                Complex::i() * kz * value,
                (position[0] * dx + position[1] * dy) / transverse,
                Complex::i() * z * value,
            ];
            prop_assert_close!(wave.value, value, 2e-13 * (1.0 + value.norm()));
            for (actual, expected) in wave.derivative.into_iter().zip(expected) {
                prop_assert_close!(actual, expected, 2e-13 * (1.0 + expected.norm()));
            }
        }
        Ok(())
    }

    /// Exact derivative identities of `T_n = Z_n(k_rho rho) exp(i n phi + i kz z)` in
    /// terms of the neighbouring orders, on both sides of the regular series
    /// threshold and (for regular waves) on the axis:
    /// `∂x T_n = k_rho (T_{n-1} - T_{n+1}) / 2`, `∂y T_n = i k_rho (T_{n-1} + T_{n+1}) / 2`,
    /// `∂z T_n = i kz T_n`, `∂k T_n = k (x ∂x + y ∂y) T_n / k_rho²` and
    /// `∂kz T_n = i z T_n - kz (x ∂x + y ∂y) T_n / k_rho²`.
    fn check_derivative_recurrences(
        order: i32,
        [rho, phi, z]: [f64; 3],
        kz: f64,
        k: Complex,
        radial: Radial,
    ) -> Result<(), TestCaseError> {
        if rho == 0.0 && radial == Radial::Singular {
            return Ok(());
        }
        let position = [rho * phi.cos(), rho * phi.sin(), z];
        let at = |n: i32| {
            let mode = |m| Mode { kz, m, pol: 1 };
            cartesian_translation(mode(0), mode(n), k, position, radial).unwrap()
        };
        let [lower, jet, upper] = [order - 1, order, order + 1].map(at);
        let mut krho = (k * k - kz * kz).sqrt();
        if krho.im < 0.0 {
            krho = -krho;
        }
        let dx = 0.5 * krho * (lower.value - upper.value);
        let dy = 0.5 * Complex::i() * krho * (lower.value + upper.value);
        // (x ∂x + y ∂y) T_n / k_rho², the k_rho derivative divided by k_rho.
        let spectral = (position[0] * dx + position[1] * dy) / krho.powi(2);
        let expected = [
            dx,
            dy,
            Complex::i() * kz * jet.value,
            k * spectral,
            Complex::i() * z * jet.value - kz * spectral,
        ];
        let actual = [
            jet.position[0],
            jet.position[1],
            jet.position[2],
            jet.k,
            jet.kz,
        ];
        let scale = [lower, jet, upper]
            .iter()
            .map(|t| t.value.norm())
            .sum::<f64>();
        prop_assert_close!(
            actual,
            expected,
            1e-12 * (1.0 + scale * (1.0 + rho + z.abs()))
        );
        Ok(())
    }
}

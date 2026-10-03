//! Cylindrical translation coefficients in polar coordinates, of one order difference and
//! of one mode pair, with analytic pullbacks.
//!
//! Upstream: `treams.special.tl_vcw`, `tl_vcw_r` and `treams.cw.translate`.
#![allow(clippy::indexing_slicing)] // Fixed argument arrays.

use crate::{
    Complex, Error, MAX_DEGREE, Result,
    numerics::{Jet, broadcast, finite, parallel::Parallel},
    special::{MAX_ORDER, Radial, pol_index, radial_jet},
};

/// Cylindrical translation in `(k_rho*rho, phi, z, kz)` with a fixed order difference,
/// the kernel of [`tl_vcw`]. The two axial labels must move together; unequal labels are
/// uncoupled.
///
/// Upstream: `treams.special.tl_vcw` and `tl_vcw_r` at equal axial wavenumbers, with
/// `order = m - mu`.
pub fn polar_translation(order: i32, args: [Complex; 4], radial: Radial) -> Result<Complex> {
    Ok(cylindrical_jet::<0>(order, args, radial, None)?.value)
}

/// The phase `exp(i (order phi + kz z))` of a cylindrical translation.
fn cylindrical_phase<const N: usize>(order: i32, phi: Jet<N>, z: Jet<N>, kz: Jet<N>) -> Jet<N> {
    (Complex::i() * (f64::from(order) * phi + kz * z)).exp()
}

/// [`polar_translation`] with the derivatives with respect to its `N` leading arguments.
/// A forward (`N = 0`) uses `fixed_phase` when given, the phase shared by a radial sweep.
fn cylindrical_jet<const N: usize>(
    order: i32,
    args: [Complex; 4],
    radial: Radial,
    fixed_phase: Option<Complex>,
) -> Result<Jet<N>> {
    if order.unsigned_abs() > MAX_ORDER.unsigned_abs() || args.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput(
            // 256 is MAX_ORDER.
            "require |order difference| <= 256 and finite arguments".into(),
        ));
    }
    let [kr, phi, z, kz] = std::array::from_fn(|i| Jet::<N>::variable(args[i], i));
    let bessel = radial_jet(order, kr, radial, false)?;
    let phase = if N == 0
        && let Some(phase) = fixed_phase
    {
        Jet::constant(phase)
    } else {
        cylindrical_phase(order, phi, z, kz)
    };
    let value = bessel * phase;
    if !value.finite() {
        return Err(Error::NonFinite(
            "non-finite cylindrical translation or derivative".into(),
        ));
    }
    Ok(value)
}

/// The gradients of the radial argument, azimuth, axial position and common axial
/// wavenumber.
fn polar_translation_pullback(
    order: i32,
    args: [Complex; 4],
    radial: Radial,
    g: Complex,
) -> Result<[Complex; 4]> {
    if !finite(g) {
        return Err(Error::InvalidInput("cotangent must be finite".into()));
    }
    if g == Complex::default() {
        polar_translation(order, args, radial)?;
        return Ok([Complex::default(); 4]);
    }
    Ok(cylindrical_jet::<4>(order, args, radial, None)?
        .derivative
        .map(|d| d.conj() * g))
}

/// What [`polar_translation_array`] saves for its pullback: the orders, the radial kind
/// and the four arguments, each stored once when it is one value for all outputs.
///
/// The pullback keeps every output in index order and adds the cotangents of scalar
/// arguments in that order, so its result does not depend on the thread count.
#[derive(Debug)]
pub struct PolarTranslationResidual {
    orders: Vec<i32>,
    arguments: [Vec<Complex>; 4],
    radial: Radial,
    size: usize,
}

impl PolarTranslationResidual {
    fn element(&self, i: usize) -> (i32, [Complex; 4]) {
        (
            broadcast::element(&self.orders, i),
            self.arguments.each_ref().map(|a| broadcast::element(a, i)),
        )
    }

    /// The gradients of all four continuous arguments; an argument given as one value
    /// for all outputs gets the sum of its gradients.
    pub fn pullback(self, cotangent: &[Complex]) -> Result<[Vec<Complex>; 4]> {
        broadcast::pullback(
            cotangent,
            self.size,
            "cotangent must be finite and match output",
            self.arguments.each_ref().map(Vec::len),
            Parallel::Chunked(64),
            |i, &g| {
                let (order, args) = self.element(i);
                polar_translation_pullback(order, args, self.radial, g)
            },
        )
    }
}

/// Evaluate a broadcast array of cylindrical polar translation coefficients; `orders`
/// holds order differences, as in [`polar_translation`].
///
/// Upstream: `treams.special.tl_vcw` and `tl_vcw_r` at equal axial wavenumbers.
pub fn polar_translation_array(
    orders: Vec<i32>,
    arguments: [Vec<Complex>; 4],
    radial: Radial,
) -> Result<(Vec<Complex>, PolarTranslationResidual)> {
    // 256 is MAX_ORDER.
    let message = "require |order difference| <= 256 and equal lengths or scalar inputs";
    let [a, b, c, d] = arguments.each_ref().map(Vec::len);
    let size = broadcast::size(&[orders.len(), a, b, c, d], message)?;
    if orders
        .iter()
        .any(|m| m.unsigned_abs() > MAX_ORDER.unsigned_abs())
    {
        return Err(Error::InvalidInput(message.into()));
    }
    let residual = PolarTranslationResidual {
        orders,
        arguments,
        radial,
        size,
    };
    // A radial sweep shares its angular/axial phase. Keep all original inputs
    // in the residual so the pullback still differentiates all four arguments.
    let fixed_phase = if size > 1
        && residual.orders.len() == 1
        && residual.arguments[1..].iter().all(|a| a.len() == 1)
    {
        let (order, args) = residual.element(0);
        let [_, phi, z, kz] = args.map(Jet::<0>::constant);
        Some(cylindrical_phase(order, phi, z, kz).value)
    } else {
        None
    };
    let threshold = if radial == Radial::Regular { 512 } else { 64 };
    let value = broadcast::map(size, Parallel::Chunked(threshold), |i| {
        let (order, args) = residual.element(i);
        Ok(cylindrical_jet::<0>(order, args, radial, fixed_phase)?.value)
    })?;
    Ok((value, residual))
}

/// Cylindrical translation coefficient of the mode pair `(qz, m)` to `(kz, mu)`.
///
/// `args` is `[k_rho rho, phi, z]`, the displacement in cylindrical coordinates. Equal
/// axial wavenumbers give [`polar_translation`] of the order difference `m - mu`, and
/// unequal ones give zero for any displacement. The wavenumbers must be finite and the
/// orders satisfy `|mu|, |m| <= 128` ([`MAX_DEGREE`]); the orders are `i64` so that every
/// integer label gets this check before its conversion. At equal wavenumbers the
/// displacement must be finite too.
///
/// Upstream: `treams.special.tl_vcw` (singular) and `tl_vcw_r` (regular).
/// Differences: orders above 128, non-finite axial wavenumbers, non-finite `k_rho rho`,
/// `phi` or `z` at equal axial wavenumbers, singular waves at `k_rho rho = 0` and values
/// that overflow give an error, where treams returns a value, infinity or NaN.
#[inline]
#[allow(clippy::float_cmp)] // Axial wave labels use exact equality, as in basis expansions.
pub fn tl_vcw(
    kz: f64,
    mu: i64,
    qz: f64,
    m: i64,
    args: [Complex; 3],
    radial: Radial,
) -> Result<Complex> {
    let bound = u64::from(MAX_DEGREE.unsigned_abs());
    if !kz.is_finite() || !qz.is_finite() || mu.unsigned_abs() > bound || m.unsigned_abs() > bound {
        return Err(Error::InvalidInput(
            // 128 is MAX_DEGREE.
            "finite axial labels and |orders| <= 128 required".into(),
        ));
    }
    if kz != qz {
        return Ok(Complex::default());
    }
    let order = i32::try_from(m - mu)
        .map_err(|_| Error::InvalidInput("invalid cylindrical order".into()))?;
    polar_translation(order, [args[0], args[1], args[2], kz.into()], radial)
}

/// Cylindrical translation coefficient of one polarized mode pair.
///
/// The coefficient from `(qz, m, q)` to `(kz, mu, p)` at the displacement
/// `[k_rho rho, phi, z]` is [`tl_vcw`] for equal polarizations and zero across
/// polarizations; the polarizations must be 0 or 1. Singular waves give zero, as in
/// treams, where `|k_rho rho|` and `|z|` are both below `1e-16`; this self term includes
/// coinciding expansion centres.
///
/// Upstream: `treams.cw.translate`.
/// Differences: at the singular self term treams returns zero for any polarization
/// labels, while this function rejects labels other than 0 or 1 there too. Coefficients
/// that reach [`tl_vcw`] share its differences.
#[inline]
pub fn translate(
    kz: f64,
    mu: i64,
    p: i64,
    qz: f64,
    m: i64,
    q: i64,
    [kr, phi, z]: [Complex; 3],
    radial: Radial,
) -> Result<Complex> {
    let (p, q) = (pol_index(p)?, pol_index(q)?);
    if p != q || (radial == Radial::Singular && kr.norm() < 1e-16 && z.norm() < 1e-16) {
        Ok(Complex::default())
    } else {
        tl_vcw(kz, mu, qz, m, [kr, phi, z], radial)
    }
}

#[cfg(test)]
mod tests {
    //! The shared-phase radial sweep and polar coefficients against elementwise and
    //! Cartesian evaluations. Identities of translations are in `properties/waves.rs`.

    use super::{
        Complex, Radial, polar_translation, polar_translation_array, polar_translation_pullback,
    };
    use crate::{
        cw,
        test_support::{DEFAULT_CASES, five_point, prop_assert_close, radial_kind},
    };
    use proptest::{prelude::*, test_runner::TestCaseError};

    // The argument strategies below have seeds in proptest-regressions/cw/polar.txt;
    // keep their shapes when changing the checks.

    /// A shrunk failure of `cylindrical_cartesian_and_polar_adjoints`, written out so that
    /// it does not depend on the strategy that the seeds in
    /// `proptest-regressions/cw/polar.txt` replay: singular waves of order difference 9.
    #[test]
    fn recorded_cylindrical_polar_case() -> Result<(), TestCaseError> {
        check_cylindrical(-4, 5, -0.222_807_017_666_592_56, Radial::Singular)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]
        #[test]
        fn cylindrical_radial_sweep_preserves_values_and_all_pullbacks(
            order in -6_i32..=6, phi in -3.0_f64..3.0, axial in -0.5_f64..0.5,
            kz in -0.7_f64..0.7, regular in any::<bool>(), count in 2_u16..160,
        ) {
            check_radial_sweep(order, phi, axial, kz, radial_kind(regular), count)?;
        }

        #[test]
        fn cylindrical_cartesian_and_polar_adjoints(
            mu in -5_i32..6, m in -5_i32..6, phi in -3.0_f64..3.0, regular in any::<bool>(),
        ) {
            check_cylindrical(mu, m, phi, radial_kind(regular))?;
        }
    }

    /// A radial sweep with fixed angle, axial position and wavenumber (the shared-phase
    /// fast path) equals elementwise evaluations, and its pullback sums the scalar
    /// arguments' elementwise cotangents.
    fn check_radial_sweep(
        order: i32,
        phi: f64,
        axial: f64,
        kz: f64,
        radial: Radial,
        count: u16,
    ) -> Result<(), TestCaseError> {
        let arguments = [
            (0..count)
                .map(|i| Complex::new(0.5 + f64::from(i) / 100.0, 0.2))
                .collect(),
            vec![Complex::new(phi, 0.1)],
            vec![axial.into()],
            vec![kz.into()],
        ];
        let count = usize::from(count);
        let (values, residual) =
            polar_translation_array(vec![order], arguments.clone(), radial).unwrap();
        let cotangent = vec![Complex::new(0.2, -0.3); count];
        let mut expected = [
            vec![],
            vec![Complex::default()],
            vec![Complex::default()],
            vec![Complex::default()],
        ];
        prop_assert_eq!(values.len(), count);
        for (i, value) in values.iter().enumerate() {
            let args = [
                arguments[0][i],
                arguments[1][0],
                arguments[2][0],
                arguments[3][0],
            ];
            let scalar = polar_translation(order, args, radial).unwrap();
            prop_assert_close!(*value, scalar, 2e-14 * scalar.norm().max(1.0));
            let gradient = polar_translation_pullback(order, args, radial, cotangent[i]).unwrap();
            expected[0].push(gradient[0]);
            for axis in 1..4 {
                expected[axis][0] += gradient[axis];
            }
        }
        let gradients = residual.pullback(&cotangent).unwrap();
        for (actual, expected) in gradients.iter().zip(&expected) {
            prop_assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                prop_assert_close!(*actual, *expected, 2e-13 * expected.norm().max(1.0));
            }
        }
        Ok(())
    }

    /// Polar cylindrical coefficients equal Cartesian translations; the azimuth, axial
    /// position and axial wavenumber derivatives are exactly `i n T`, `i kz T` and
    /// `i z T` up to rounding relative to `T` (and to `n T` for the azimuth), and the
    /// radial one matches a complex central difference.
    fn check_cylindrical(mu: i32, m: i32, phi: f64, radial: Radial) -> Result<(), TestCaseError> {
        let args = [Complex::new(1.3, 0.1), phi.into(), 0.3.into(), 0.2.into()];
        let k = (args[0] * args[0] + args[3] * args[3]).sqrt();
        let mode = |m| cw::Mode { kz: 0.2, m, pol: 0 };
        let position = [phi.cos(), phi.sin(), 0.3];
        let expected = cw::cartesian_translation(mode(mu), mode(m), k, position, radial).unwrap();
        let value = polar_translation(m - mu, args, radial).unwrap();
        prop_assert_close!(value, expected.value, 1e-10 * (1.0 + value.norm()));
        let derivative = polar_translation_pullback(m - mu, args, radial, Complex::new(1.0, 0.0))
            .unwrap()
            .map(|g| g.conj());
        let tolerance = 1e-14 * value.norm();
        let i = Complex::i();
        let order = f64::from(m - mu);
        prop_assert_close!(
            derivative[1],
            i * order * value,
            (1.0 + order.abs()) * tolerance
        );
        prop_assert_close!(derivative[2], i * args[3] * value, tolerance);
        prop_assert_close!(derivative[3], i * args[2] * value, tolerance);
        let direction = Complex::new(0.2, 0.1);
        let numeric = five_point(1e-4, |t| {
            let mut shifted = args;
            shifted[0] += t * direction;
            polar_translation(m - mu, shifted, radial).unwrap()
        });
        // Compare the full complex derivative: a real projection may nearly cancel
        // while the high-order outgoing value is large.
        let analytic = derivative[0] * direction;
        prop_assert_close!(analytic, numeric, 1e-8 * (1.0 + analytic.norm()));
        Ok(())
    }
}

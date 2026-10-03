//! Spherical translation coefficients of mode pairs in polar coordinates `(kr, theta, phi)`,
//! broadcast over arrays, with analytic pullbacks.
//!
//! Upstream: `treams.sw.translate` and `treams.special.tl_vsw_A`, `tl_vsw_B`, `tl_vsw_rA`
//! and `tl_vsw_rB`.
#![allow(clippy::indexing_slicing)] // Fixed argument arrays, degree rows and deduplicated plans.

use crate::{
    Complex, Error, Result,
    basis::ModeLabel,
    numerics::{Jet, broadcast, finite, parallel::Parallel},
    special::{self, Bessel, Radial, SERIES_RADIUS, polar_trig, radial_jet},
    sw::{self, Mode},
};

/// Fixed spherical mode-pair couplings shared across displacements and pullbacks.
///
/// Upstream: `treams.sw.translate` of one mode pair, which combines
/// `treams.special.tl_vsw_A` and `tl_vsw_B` (`tl_vsw_rA` and `tl_vsw_rB` for regular
/// waves).
#[derive(Debug)]
pub struct PolarTranslation {
    /// Degree and weight of each term.
    terms: Vec<(i32, Complex)>,
    /// Order difference `source.m - destination.m`, shared by all terms.
    order: i32,
    /// Smallest and largest term degree.
    degrees: [i32; 2],
    radial: Radial,
}

impl PolarTranslation {
    /// Prepare one mode pair in helicity or parity polarization.
    pub fn new(destination: Mode, source: Mode, helicity: bool, radial: Radial) -> Result<Self> {
        destination.validate()?;
        source.validate()?;
        let order = source.m - destination.m;
        let terms: Vec<_> = sw::terms(destination, source, helicity)
            .into_iter()
            .map(|(degree, term_order, weight)| {
                debug_assert_eq!(term_order, order);
                (degree, weight)
            })
            .collect();
        let degrees = terms
            .iter()
            .fold([i32::MAX, 0], |[low, high], &(degree, _)| {
                [low.min(degree), high.max(degree)]
            });
        Ok(Self {
            terms,
            order,
            degrees,
            radial,
        })
    }

    /// Spherical radial jets `z_p(kr)` of the term degrees, indexed by `p` minus the
    /// smallest degree, or `None` when each term evaluates its own.
    ///
    /// Outgoing functions come from one AMOS sequence over the degree range. Regular
    /// derivatives outside the series region evaluate each single value once and
    /// share `j_p` between the value of degree `p` and the derivative of `p - 1`,
    /// so they equal those of [`crate::special::spherical_radial`] bitwise.
    fn radial_row<const N: usize>(&self, z: Jet<N>) -> Result<Option<Vec<Jet<N>>>> {
        let [first, last] = self.degrees;
        let last = last + i32::from(N > 0);
        // Term degrees lie in `first..=last`.
        let offset = |p: i32| usize::try_from(p - first).unwrap_or_default();
        let values = match self.radial {
            Radial::Singular => {
                if z.value.norm_sqr() == 0.0 {
                    return Err(Error::InvalidInput(
                        "outgoing spherical waves are singular at zero".into(),
                    ));
                }
                let degree = |p: i32| u32::try_from(p).unwrap_or_default();
                special::spherical_hankels(degree(first), degree(last), z.value)?
            }
            Radial::Regular if N > 0 && z.value.norm() >= SERIES_RADIUS => {
                let mut values = vec![Complex::default(); offset(last) + 1];
                for &(degree, _) in &self.terms {
                    for p in [degree, degree + 1] {
                        let value = &mut values[offset(p)];
                        if *value == Complex::default() {
                            *value = special::bessel(f64::from(p), z.value, Bessel::J, true, 0)?;
                        }
                    }
                }
                values
            }
            Radial::Regular => return Ok(None),
        };
        if N == 0 {
            return Ok(Some(values.into_iter().map(Jet::constant).collect()));
        }
        // The first derivative `z_p' = p z_p / z - z_{p+1}` of `special::spherical_radial`.
        Ok(Some(
            (first..)
                .zip(values.windows(2))
                .map(|(p, pair)| z.chain(pair[0], f64::from(p) * pair[0] / z.value - pair[1]))
                .collect(),
        ))
    }

    /// The coefficient `sum_p weight_p z_p(kr) P_p^order(cos theta) e^(i order phi)` at
    /// `args = [kr, theta, phi]`, with the derivatives with respect to the `N` leading
    /// arguments.
    fn evaluate<const N: usize>(&self, args: [Complex; 3]) -> Result<Jet<N>> {
        if args.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput(
                "translation arguments must be finite".into(),
            ));
        }
        if self.terms.is_empty() {
            return Ok(Jet::default());
        }
        let [kr, theta, phi] = std::array::from_fn(|i| Jet::<N>::variable(args[i], i));
        let [cosine, sine] = polar_trig(theta);
        let order = self.order;
        let phase = sine.powi(order.abs()) * (Complex::i() * f64::from(order) * phi).exp();
        let [low, high] = self.degrees;
        let mut legendre =
            Vec::with_capacity(usize::try_from(high - order.abs() + 1).unwrap_or_default());
        special::legendre_factors(high, order, cosine, |p| legendre.push(p));
        let row = self.radial_row(kr)?;
        let mut result = Jet::default();
        for &(degree, weight) in &self.terms {
            let radial = match &row {
                Some(row) => row[usize::try_from(degree - low).unwrap_or_default()],
                None => radial_jet(degree, kr, self.radial, true)?,
            };
            let factor = legendre[usize::try_from(degree - order.abs()).unwrap_or_default()];
            result += weight * radial * (phase * factor);
        }
        if !result.finite() {
            return Err(Error::NonFinite(
                "non-finite translation or derivative".into(),
            ));
        }
        Ok(result)
    }

    /// Whether the polarized coefficient of `radial` waves at the radial argument `kr` is
    /// the self term, which is zero: singular waves with `|kr| < 1e-16`, which includes
    /// coinciding expansion centres. The `tl_vsw_*` coefficients have no such rule.
    ///
    /// Upstream: `treams.sw.translate`, which returns zero there.
    /// Differences: treams returns this zero for any mode labels, such as `l = 0` or
    /// polarization 2; the `sw.translate` ufunc checks the labels first and rejects those.
    #[inline]
    #[must_use]
    pub fn is_self_term(radial: Radial, kr: Complex) -> bool {
        radial == Radial::Singular && kr.norm() < 1e-16
    }

    /// Evaluate at `(kr, theta, phi)`; all three arguments may be complex.
    pub fn value(&self, args: [Complex; 3]) -> Result<Complex> {
        Ok(self.evaluate::<0>(args)?.value)
    }

    /// The gradients of the three arguments from a complex scalar cotangent.
    pub fn pullback(&self, args: [Complex; 3], cotangent: Complex) -> Result<[Complex; 3]> {
        if !finite(cotangent) {
            return Err(Error::InvalidInput(
                "translation cotangent must be finite".into(),
            ));
        }
        if cotangent == Complex::default() {
            self.value(args)?;
            return Ok([Complex::default(); 3]);
        }
        Ok(self
            .evaluate::<3>(args)?
            .derivative
            .map(|d| d.conj() * cotangent))
    }
}

/// Spherical translation arrays evaluate in parallel from this many elements.
const PARALLEL: Parallel = Parallel::AtLeast(1024);
/// Coupling plans, which cost several evaluations each, build in parallel from this
/// many distinct mode pairs.
const PLANS: Parallel = Parallel::AtLeast(64);

/// What [`polar_translation_array`] saves for its pullback: the arguments and one
/// coupling plan per distinct mode pair.
///
/// The pullback keeps every output in index order and adds the cotangents of scalar
/// arguments in that order, so its result does not depend on the thread count.
#[derive(Debug)]
pub struct PolarTranslationResidual {
    plans: Vec<PolarTranslation>,
    /// The plan of each output, or of all outputs.
    indices: Vec<usize>,
    arguments: [Vec<Complex>; 3],
    size: usize,
}

impl PolarTranslationResidual {
    fn element(&self, i: usize) -> (&PolarTranslation, [Complex; 3]) {
        (
            &self.plans[broadcast::element(&self.indices, i)],
            self.arguments.each_ref().map(|a| broadcast::element(a, i)),
        )
    }

    /// The argument gradients from one cotangent per translation; an argument given as
    /// one value for all outputs gets the sum of its gradients.
    pub fn pullback(self, cotangent: &[Complex]) -> Result<[Vec<Complex>; 3]> {
        broadcast::pullback(
            cotangent,
            self.size,
            "translation cotangent must be finite and match output",
            self.arguments.each_ref().map(Vec::len),
            PARALLEL,
            |i, &g| {
                let (plan, args) = self.element(i);
                plan.pullback(args, g)
            },
        )
    }
}

/// Broadcast polar translations, preserving scalar arguments and fixed mode pairs.
///
/// Upstream: `treams.sw.translate`, `treams.special.tl_vsw_A` and `tl_vsw_B`.
pub fn polar_translation_array(
    modes: Vec<[Mode; 2]>,
    arguments: [Vec<Complex>; 3],
    helicity: bool,
    radial: Radial,
) -> Result<(Vec<Complex>, PolarTranslationResidual)> {
    let [a, b, c] = arguments.each_ref().map(Vec::len);
    let size = broadcast::size(
        &[modes.len(), a, b, c],
        "translation arrays must have equal lengths or scalar inputs",
    )?;
    let mut lookup = std::collections::HashMap::new();
    let mut distinct = Vec::new();
    let indices = modes
        .into_iter()
        .map(|pair| {
            *lookup.entry(pair).or_insert_with(|| {
                distinct.push(pair);
                distinct.len() - 1
            })
        })
        .collect();
    let plans = broadcast::map(distinct.len(), PLANS, |i| {
        let [to, from] = distinct[i];
        PolarTranslation::new(to, from, helicity, radial)
    })?;
    let residual = PolarTranslationResidual {
        plans,
        indices,
        arguments,
        size,
    };
    let values = broadcast::map(size, PARALLEL, |i| {
        let (plan, args) = residual.element(i);
        plan.value(args)
    })?;
    Ok((values, residual))
}

#[cfg(test)]
mod tests {
    //! Polar coefficients and their pullbacks against Cartesian translations. Identities
    //! of translations are in `properties/waves.rs`.

    use super::{Complex, Mode, PolarTranslation, Radial, sw};
    use crate::{
        special::{self, on_sphere},
        test_support::{DEFAULT_CASES, degree_order, log_uniform, prop_assert_close, radial},
    };
    use proptest::{prelude::*, test_runner::TestCaseError};
    use std::f64::consts::PI;

    /// Shrunk failures of the properties below, written out so that they do not depend
    /// on the strategy shapes that the seeds in `proptest-regressions/sw/polar.txt` replay.
    #[test]
    fn recorded_polar_translation_cases() -> Result<(), TestCaseError> {
        // `cartesian_translation_and_all_polar_adjoints`: singular waves at
        // `kr = 1.3 + 0.2i`, in the helicity and in the parity basis.
        let kr = Complex::new(1.3, 0.2);
        // Release-readiness regressions: the former real-projected finite difference
        // cancelled even though the full complex derivatives remained well conditioned.
        check_spherical(
            Mode { l: 4, m: 3, pol: 1 },
            Mode { l: 4, m: 3, pol: 1 },
            kr,
            [2.788_667_920_988_279_6, 0.0],
            true,
            Radial::Singular,
        )?;
        check_spherical(
            Mode { l: 4, m: 2, pol: 1 },
            Mode { l: 4, m: 2, pol: 1 },
            kr,
            [0.1, 0.0],
            false,
            Radial::Singular,
        )?;
        check_spherical(
            Mode { l: 5, m: 3, pol: 1 },
            Mode {
                l: 3,
                m: -2,
                pol: 1,
            },
            kr,
            [2.255_087_340_294_624_5, -0.095_390_013_154_433_45],
            true,
            Radial::Singular,
        )?;
        check_spherical(
            Mode {
                l: 3,
                m: -2,
                pol: 1,
            },
            Mode { l: 5, m: 3, pol: 1 },
            kr,
            [0.813_134_622_536_094_8, 0.0],
            false,
            Radial::Singular,
        )?;
        // `spherical_translation_across_argument_decades_and_poles`: a regular wave on
        // the south pole, where the order difference 2 makes the coefficient vanish.
        check_spherical(
            Mode { l: 2, m: 2, pol: 0 },
            Mode { l: 1, m: 0, pol: 0 },
            Complex::new(1.0, 0.0),
            [PI, 0.0],
            false,
            Radial::Regular,
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]
        #[test]
        fn cartesian_translation_and_all_polar_adjoints(
            (l, m) in degree_order(1..6), (lambda, mu) in degree_order(1..6), pol in 0_u8..2,
            theta in 0.1_f64..3.0, phi in -3.0_f64..3.0, helicity in any::<bool>(),
            radial in radial(),
        ) {
            let to = Mode { l: lambda, m: mu, pol: 1 };
            let from = Mode { l, m, pol };
            check_spherical(to, from, Complex::new(1.3, 0.2), [theta, phi], helicity, radial)?;
        }
    }

    proptest! {
        // Twice the default: the draw spans 3.5 decades of |kr|, both poles and degrees
        // up to 8.
        #![proptest_config(ProptestConfig::with_cases(2 * DEFAULT_CASES))]
        #[test]
        fn spherical_translation_across_argument_decades_and_poles(
            (l, m) in degree_order(1..9), (lambda, mu) in degree_order(1..9),
            pol in 0_u8..2, to_pol in 0_u8..2, magnitude in log_uniform(-2.0..1.5),
            angle in -0.3_f64..0.3, theta in prop_oneof![Just(0.0), Just(PI), 0.0..PI],
            phi in -3.0_f64..3.0, helicity in any::<bool>(), radial in radial(),
        ) {
            let to = Mode { l: lambda, m: mu, pol: to_pol };
            let from = Mode { l, m, pol };
            let kr = Complex::from_polar(magnitude, angle);
            check_spherical(to, from, kr, [theta, phi], helicity, radial)?;
        }
    }

    /// Polar spherical coefficients are Cartesian translations of the unit displacement
    /// `d(theta, phi)`, and their derivatives follow by the chain rule:
    /// `d/d(kr) T = d/dk T`, `d/dtheta T = grad T . d_theta d` and
    /// `d/dphi T = grad T . d_phi d`, which is also exactly `i (m_from - m_to) T`.
    /// On the poles the polar derivative is the one-sided limit from inside `(0, pi)`.
    ///
    /// The terms of a coefficient can exceed their sum by orders of magnitude, for
    /// example when `kr` has a large imaginary part, so rounding is bounded relative
    /// to the summed term magnitudes of [`term_scale`]; the azimuthal identity also
    /// scales with the order difference.
    fn check_spherical(
        to: Mode,
        from: Mode,
        kr: Complex,
        [theta, phi]: [f64; 2],
        helicity: bool,
        radial: Radial,
    ) -> Result<(), TestCaseError> {
        let args = [kr, theta.into(), phi.into()];
        let plan = PolarTranslation::new(to, from, helicity, radial).unwrap();
        // Like `polar_trig`, read exact multiples of the PI constant as the poles.
        let (sine, cosine) = if theta % PI == 0.0 {
            (0.0, theta.cos())
        } else {
            theta.sin_cos()
        };
        let position = [sine * phi.cos(), sine * phi.sin(), cosine];
        let reference =
            sw::cartesian_translation(to, from, kr, position, helicity, radial).unwrap();
        let tangents = [
            [cosine * phi.cos(), cosine * phi.sin(), -sine],
            [-sine * phi.sin(), sine * phi.cos(), 0.0],
        ];
        let [d_theta, d_phi] = tangents.map(|t| {
            (0..3)
                .map(|axis| reference.position[axis] * t[axis])
                .sum::<Complex>()
        });
        let scale = term_scale(&plan, kr, position);
        let tolerance = 1e-13 * scale + f64::MIN_POSITIVE;
        let value = plan.value(args).unwrap();
        prop_assert_close!(value, reference.value, tolerance);
        let g = Complex::new(0.3, -0.2);
        let derivative = plan.pullback(args, g).unwrap().map(|d| d.conj() / g.conj());
        prop_assert_close!(derivative[0], reference.k, tolerance, "kr");
        prop_assert_close!(derivative[1], d_theta, tolerance, "theta");
        prop_assert_close!(derivative[2], d_phi, tolerance, "phi");
        let order = f64::from(from.m - to.m);
        prop_assert_close!(
            derivative[2],
            Complex::i() * order * value,
            1e-14 * (1.0 + order.abs()) * scale + f64::MIN_POSITIVE,
            "azimuthal"
        );
        Ok(())
    }

    /// The rounding scale `Σ_p |w_p| (|z_p| + (1 + |kr|) |z_p'|) (|Y_p| + Σ_a |∂_a Y_p|)`
    /// of the terms `w_p z_p(kr) Y_p(d)` of `plan` at the unit displacement `d`, from the
    /// independent `special::spherical_radial` and `on_sphere` evaluations. It bounds
    /// each term's value, `kr` derivative and Cartesian gradient; the gradient also
    /// bounds the rounding of `Y_p` near its roots, where `|Y_p|` alone underestimates it.
    fn term_scale(plan: &PolarTranslation, kr: Complex, position: [f64; 3]) -> f64 {
        plan.terms
            .iter()
            .map(|&(degree, weight)| {
                let z = special::spherical_radial(degree.unsigned_abs(), kr, plan.radial).unwrap();
                let (harmonic, gradient) = on_sphere(degree, plan.order, position);
                let harmonic = harmonic.norm() + gradient.iter().map(|g| g.norm()).sum::<f64>();
                weight.norm() * (z.value.norm() + (1.0 + kr.norm()) * z.first.norm()) * harmonic
            })
            .sum()
    }
}

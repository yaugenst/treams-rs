//! Bessel functions and radial jets with analytic argument derivatives.
//!
//! Cylindrical and spherical Bessel functions of every kind ([`bessel`]), regular and
//! singular radial jets ([`spherical_radial`], [`cylindrical_radial`]), and their
//! broadcasting residual. Away from the small-argument power series they call the AMOS
//! library (the `complex_bessel` crate), as scipy does.
//!
//! Upstream: `treams.special.jv`, `yv`, `hankel1`, `hankel2`, `spherical_jn` and
//! `spherical_yn` (scipy's functions), and `jv_d`, `yv_d`, `hankel1_d`, `hankel2_d`,
//! `spherical_hankel1`, `spherical_hankel2` and their `_d` derivatives (`_bessel.pyx`).

use std::f64::consts::PI;

use super::{MAX_ORDER, SERIES_RADIUS, log_factorial};
use crate::{
    Complex, Error, Result,
    numerics::{
        Jet,
        broadcast::{self, element},
        finite,
        parallel::Parallel,
        ratio,
    },
};

/// Elementwise Bessel evaluations and pullbacks run in parallel from this many elements.
const PARALLEL: Parallel = Parallel::AtLeast(64);

/// Cylindrical Bessel solution, also used at half order for spherical functions.
///
/// Upstream: `treams.special.jv` (`J`), `yv` (`Y`), `hankel1` (`H1`) and `hankel2`
/// (`H2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bessel {
    /// Regular first-kind solution.
    J,
    /// Second-kind solution.
    Y,
    /// Outgoing Hankel solution.
    H1,
    /// Incoming Hankel solution.
    H2,
}

impl From<Radial> for Bessel {
    /// Regular waves are J, singular (outgoing) waves H1.
    fn from(radial: Radial) -> Self {
        match radial {
            Radial::Regular => Self::J,
            Radial::Singular => Self::H1,
        }
    }
}

fn amos_error(error: complex_bessel::Error) -> Error {
    Error::SpecialFunction(error.to_string())
}

/// `sqrt(pi / 2z)`, which turns cylindrical functions of order `l + 1/2` into
/// spherical functions of order `l`.
///
/// On the negative real axis AMOS takes the limit from above for either sign of a
/// zero imaginary part (it tests `Im z < 0`), where `pi / 2z` approaches the axis from
/// below. Real-by-complex division gives `pi / 2z` a `+0` imaginary part there, so the
/// prefactor takes that side explicitly; the spherical functions are single-valued,
/// and only the two factors' agreement matters.
fn spherical_prefactor(z: Complex) -> Complex {
    let mut w = PI / (2.0 * z);
    if z.im == 0.0 && z.re < 0.0 {
        w.im = -0.0;
    }
    crate::numerics::complex_sqrt(w)
}

/// One AMOS evaluation of a cylindrical function, or of a spherical one at order + 1/2.
#[inline]
fn bessel_raw(order: f64, z: Complex, kind: Bessel, spherical: bool) -> Result<Complex> {
    let order = if spherical { order + 0.5 } else { order };
    let value = match kind {
        Bessel::J => complex_bessel::besselj(order, z),
        Bessel::Y => complex_bessel::bessely(order, z),
        Bessel::H1 => complex_bessel::hankel1(order, z),
        Bessel::H2 => complex_bessel::hankel2(order, z),
    }
    .map_err(amos_error)?;
    Ok(if spherical {
        spherical_prefactor(z) * value
    } else {
        value
    })
}

/// Cylindrical functions of orders `start, start + 1, ...` from one AMOS recurrence.
fn sequence(kind: Bessel, start: f64, z: Complex, length: usize) -> Result<Vec<Complex>> {
    let unscaled = complex_bessel::Scaling::Unscaled;
    match kind {
        Bessel::J => complex_bessel::besselj_seq(start, z, length, unscaled),
        Bessel::Y => complex_bessel::bessely_seq(start, z, length, unscaled),
        Bessel::H1 => complex_bessel::hankel1_seq(start, z, length, unscaled),
        Bessel::H2 => complex_bessel::hankel2_seq(start, z, length, unscaled),
    }
    .map(|sequence| sequence.values)
    .map_err(amos_error)
}

/// The two values of a two-term sequence.
fn pair(values: Vec<Complex>) -> Result<[Complex; 2]> {
    <[Complex; 2]>::try_from(values)
        .map_err(|_| Error::SpecialFunction("invalid Bessel sequence length".into()))
}

/// Outgoing cylindrical functions `[H_(m-1)(z), H_m(z)]` from one AMOS sequence, with
/// `H_-1 = -H_1`: the derivative `u^m H_(m-1)(u)` of `u^m H_m(u)` without the
/// cancelling product rule.
pub(crate) fn hankel_below(order: u32, z: Complex) -> Result<[Complex; 2]> {
    if order == 0 {
        let [value, upper] = pair(sequence(Bessel::H1, 0.0, z, 2)?)?;
        return Ok([-upper, value]);
    }
    pair(sequence(Bessel::H1, f64::from(order - 1), z, 2)?)
}

/// Spherical functions of orders `order` and `order + 1`. Hankel functions share one
/// AMOS sequence, which agrees with single evaluations to 2e-14 relative. J and Y
/// sequences differ from single evaluations by up to 1e-12, so these stay single.
fn spherical_pair(kind: Bessel, order: f64, z: Complex) -> Result<[Complex; 2]> {
    let values = if matches!(kind, Bessel::H1 | Bessel::H2) {
        pair(sequence(kind, order + 0.5, z, 2)?)?
    } else {
        let single = |order: f64| bessel_raw(order + 0.5, z, kind, false);
        [single(order)?, single(order + 1.0)?]
    };
    let prefactor = spherical_prefactor(z);
    Ok(values.map(|value| prefactor * value))
}

/// `jet`, or an overflow error where one of its parts is not finite. Near zero the
/// prefactor `sqrt(pi / 2z)` and the derivatives' powers of `1/z` overflow where AMOS
/// does not, as for `h_0` below `|z| = 1e-103`.
fn checked(jet: RadialJet) -> Result<RadialJet> {
    if finite(jet.value) && finite(jet.first) && finite(jet.second) {
        Ok(jet)
    } else {
        Err(Error::SpecialFunction(
            "radial function or derivative overflow".into(),
        ))
    }
}

/// Derivatives of a spherical function from the next order, `f' = (l/z) f - f_(l+1)`,
/// and the spherical Bessel equation.
fn jet_from_next_order(order: f64, z: Complex, [value, next]: [Complex; 2]) -> RadialJet {
    let first = order * ratio(value, z) - next;
    let second = -2.0 * ratio(first, z) + order * (order + 1.0) * ratio(ratio(value, z), z) - value;
    RadialJet {
        value,
        first,
        second,
    }
}

/// Bessel value or one of its first two complex-argument derivatives.
///
/// `kind` selects J, Y, H1 or H2. `spherical` selects the spherical function of order
/// `order` (`j`, `y`, `h1` or `h2`): the cylindrical function of order `order + 1/2`
/// times `sqrt(pi / (2z))`. `derivative` is 0, 1 or 2, always with respect to `z`.
/// Spherical regular functions use analytic origin limits.
///
/// Upstream: `treams.special.jv`, `yv`, `hankel1`, `hankel2` and with `spherical`
/// `spherical_jn`, `spherical_yn`, `spherical_hankel1`, `spherical_hankel2`; their `_d`
/// variants for `derivative = 1`. treams has no second derivative.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Integral order in 0..=MAX_ORDER checked before conversion.
#[inline]
pub fn bessel(
    order: f64,
    z: Complex,
    kind: Bessel,
    spherical: bool,
    derivative: u8,
) -> Result<Complex> {
    if !order.is_finite() || !finite(z) || derivative > 2 {
        return Err(Error::InvalidInput(
            "finite order/argument and derivative 0, 1 or 2 required".into(),
        ));
    }
    // AMOS-based kernels lose accuracy for subnormal orders (including NaNs in
    // SciPy). Their zero-order limit is identical at double precision: even at
    // extreme finite z, order * log(z) is far below a representable correction.
    let order = if order.is_subnormal() { 0.0 } else { order };
    let jet = |jet: RadialJet| match derivative {
        0 => jet.value,
        1 => jet.first,
        _ => jet.second,
    };
    let evaluate = |v| bessel_raw(v, z, kind, spherical);
    let value = if spherical
        && kind == Bessel::J
        && z.norm() < SERIES_RADIUS
        && (0.0..=f64::from(MAX_ORDER)).contains(&order)
        && order.fract() == 0.0
    {
        jet(spherical_radial(order as u32, z, Radial::Regular)?)
    } else if derivative == 0 {
        evaluate(order)?
    } else if spherical {
        jet(jet_from_next_order(
            order,
            z,
            spherical_pair(kind, order, z)?,
        ))
    } else if matches!(kind, Bessel::H1 | Bessel::H2) && (order != 0.0 || derivative == 2) {
        let length = usize::from(2 * derivative + 1);
        match sequence(kind, order - f64::from(derivative), z, length)?.as_slice() {
            [left, _, right] => 0.5 * (left - right),
            [left, _, middle, _, right] => 0.25 * (left - 2.0 * middle + right),
            _ => {
                return Err(Error::SpecialFunction(
                    "invalid Hankel sequence length".into(),
                ));
            }
        }
    } else if derivative == 1 {
        if order == 0.0 {
            -evaluate(1.0)?
        } else {
            0.5 * (evaluate(order - 1.0)? - evaluate(order + 1.0)?)
        }
    } else {
        0.25 * (evaluate(order - 2.0)? - 2.0 * evaluate(order)? + evaluate(order + 2.0)?)
    };
    if !finite(value) {
        return Err(Error::SpecialFunction("non-finite Bessel result".into()));
    }
    Ok(value)
}

/// What [`bessel_array`] saves for its pullback: the orders, the arguments and the
/// function. Only the arguments get gradients.
#[derive(Debug)]
pub struct BesselResidual {
    orders: Vec<f64>,
    arguments: Vec<Complex>,
    kind: Bessel,
    spherical: bool,
    derivative: u8,
    size: usize,
}

/// Evaluate borrowed elementwise Bessel arguments without recording a pullback; the
/// forward pass of [`bessel_array`].
pub(crate) fn bessel_values(
    orders: &[f64],
    arguments: &[Complex],
    kind: Bessel,
    spherical: bool,
    derivative: u8,
) -> Result<Vec<Complex>> {
    let message = "Bessel arrays must have equal lengths or scalar inputs, and derivative 0 or 1";
    if derivative > 1 {
        return Err(Error::InvalidInput(message.into()));
    }
    let size = broadcast::size(&[orders.len(), arguments.len()], message)?;
    broadcast::map(size, PARALLEL, |i| {
        bessel(
            element(orders, i),
            element(arguments, i),
            kind,
            spherical,
            derivative,
        )
    })
}

/// Evaluate elementwise Bessel functions, broadcasting either scalar input.
///
/// `kind` and `spherical` select the function as in [`bessel`]. `derivative` is 0 or 1,
/// one below the cap of [`bessel`], because the pullback evaluates derivative + 1.
///
/// Upstream: the functions of [`bessel`] over arrays.
pub fn bessel_array(
    orders: Vec<f64>,
    arguments: Vec<Complex>,
    kind: Bessel,
    spherical: bool,
    derivative: u8,
) -> Result<(Vec<Complex>, BesselResidual)> {
    let value = bessel_values(&orders, &arguments, kind, spherical, derivative)?;
    let size = value.len();
    Ok((
        value,
        BesselResidual {
            orders,
            arguments,
            kind,
            spherical,
            derivative,
            size,
        },
    ))
}

impl BesselResidual {
    fn evaluate(&self, i: usize, derivative: u8) -> Result<Complex> {
        bessel(
            element(&self.orders, i),
            element(&self.arguments, i),
            self.kind,
            self.spherical,
            derivative,
        )
    }

    /// The gradient with respect to the arguments, given the gradient `cotangent` with
    /// respect to the values: `cotangent` times the conjugate of the next derivative. An
    /// argument given as one value for all elements receives the sum of its element
    /// gradients; the orders have no gradient.
    pub fn pullback(self, cotangent: &[Complex]) -> Result<Vec<Complex>> {
        let [gradient] = broadcast::pullback(
            cotangent,
            self.size,
            "Bessel cotangent must be finite and match output",
            [self.arguments.len()],
            PARALLEL,
            |i, &g| {
                if g == Complex::default() {
                    Ok([g])
                } else {
                    Ok([g * self.evaluate(i, self.derivative + 1)?.conj()])
                }
            },
        )?;
        Ok(gradient)
    }
}

/// Radial kind of multipole waves: regular or singular.
///
/// Upstream: the treams `modetype` argument, `'regular'` or `'singular'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Radial {
    /// Bessel function of the first kind, regular at the origin (treams modetype
    /// "regular").
    Regular,
    /// Outgoing Hankel function of the first kind (treams modetype "singular"); also used
    /// for cylindrical waves.
    #[doc(alias = "outgoing")]
    Singular,
}

/// A radial function and its first two argument derivatives: a second-order jet in one
/// complex variable, unlike the crate's first-order jets `numerics::Jet` in several.
#[derive(Clone, Copy, Debug)]
pub struct RadialJet {
    /// Function value.
    pub value: Complex,
    /// First derivative with respect to the complex argument.
    pub first: Complex,
    /// Second derivative with respect to the complex argument.
    pub second: Complex,
}

/// Terms of the small-argument power series of [`spherical_radial`] and
/// [`cylindrical_radial`]. Below `|z| = SERIES_RADIUS` (0.5) each term is at most
/// `1/(16 q^2)` of its predecessor, so the last one is below 1e-40 of the first.
const SERIES_TERMS: u32 = 16;

/// The even power series `sum_q t_q` with `t_0 = leading z^power` and
/// `t_q = factor(q) z^2 t_(q-1)`, differentiated term by term: every term reuses its
/// predecessor's powers.
fn even_series(power: u32, leading: f64, z: Complex, factor: impl Fn(f64) -> f64) -> RadialJet {
    let mut term = RadialJet {
        value: leading * z.powu(power),
        first: if power > 0 {
            leading * f64::from(power) * z.powu(power - 1)
        } else {
            Complex::default()
        },
        second: if power > 1 {
            leading * f64::from(power * (power - 1)) * z.powu(power - 2)
        } else {
            Complex::default()
        },
    };
    let mut sum = term;
    let square = z * z;
    for q in 1..SERIES_TERMS {
        let factor = factor(f64::from(q));
        term = RadialJet {
            value: factor * square * term.value,
            first: factor * (square * term.first + 2.0 * z * term.value),
            second: factor * (square * term.second + 4.0 * z * term.first + 2.0 * term.value),
        };
        sum.value += term.value;
        sum.first += term.first;
        sum.second += term.second;
    }
    sum
}

/// Evaluate a spherical radial function and its first two derivatives: `j_l` for
/// regular and `h1_l` for singular waves.
///
/// The regular solution uses its convergent power series near zero; away from
/// zero, complex Bessel evaluation and the spherical differential equation apply.
/// Singular (outgoing) waves fail at zero; near it, where the value or a derivative
/// overflows, they fail with an overflow error.
///
/// Upstream: `treams.special.spherical_jn` and `spherical_hankel1` with their `_d`
/// derivatives.
pub fn spherical_radial(l: u32, z: Complex, radial: Radial) -> Result<RadialJet> {
    if l > MAX_ORDER.unsigned_abs() || !finite(z) {
        return Err(Error::InvalidInput(
            // 256 is MAX_ORDER.
            "require finite argument and 0 <= l <= 256".into(),
        ));
    }
    if radial == Radial::Regular && z.norm() < SERIES_RADIUS {
        // j_l(z) = sum_q (-z^2/2)^q z^l / (q! (2l + 2q + 1)!!).
        let double_factorial: f64 = (0..=l).map(|i| f64::from(2 * i + 1)).product();
        let degree = f64::from(l);
        return Ok(even_series(l, double_factorial.recip(), z, |q| {
            -1.0 / (2.0 * q * (2.0 * (degree + q) + 1.0))
        }));
    }
    if z == Complex::default() {
        return Err(Error::InvalidInput(
            "outgoing spherical waves are singular at zero".into(),
        ));
    }
    let order = f64::from(l);
    checked(jet_from_next_order(
        order,
        z,
        spherical_pair(radial.into(), order, z)?,
    ))
}

/// [`spherical_radial`] jets of every degree `0..=order` at one argument. Away from the
/// regular power series, adjacent degrees share their AMOS evaluations: regular jets
/// reuse single J evaluations and equal [`spherical_radial`] bit for bit, and singular
/// jets come from one Hankel sequence instead of `order + 1` two-term sequences. Against
/// 30-digit values at `|Im z| <= 5` the long sequence stays within 2e-13 relative up to
/// order 80 (two-term sequences: 8e-14); below `Im z = 0` it loses up to a digit to
/// them, still within 1e-13 of each jet's size up to order 40.
pub(crate) fn spherical_radial_sequence(
    order: u32,
    z: Complex,
    radial: Radial,
) -> Result<Vec<RadialJet>> {
    if order > MAX_ORDER.unsigned_abs() || !finite(z) {
        return Err(Error::InvalidInput(
            // 256 is MAX_ORDER.
            "require finite argument and 0 <= l <= 256".into(),
        ));
    }
    if radial == Radial::Regular && z.norm() < SERIES_RADIUS {
        return (0..=order)
            .map(|l| spherical_radial(l, z, radial))
            .collect();
    }
    if z == Complex::default() {
        return Err(Error::InvalidInput(
            "outgoing spherical waves are singular at zero".into(),
        ));
    }
    let values = match radial {
        Radial::Regular => {
            let prefactor = spherical_prefactor(z);
            (0..=order + 1)
                .map(|l| Ok(prefactor * bessel_raw(f64::from(l) + 0.5, z, Bessel::J, false)?))
                .collect::<Result<Vec<_>>>()?
        }
        Radial::Singular => spherical_hankels(0, order + 1, z)?,
    };
    (0..=order)
        .zip(values.iter().zip(values.iter().skip(1)))
        .map(|(l, (&value, &next))| checked(jet_from_next_order(f64::from(l), z, [value, next])))
        .collect()
}

/// Spherical Hankel functions `h_p(z)` of the first kind for `p = first..=last` at a
/// nonzero argument, from one AMOS sequence.
pub(crate) fn spherical_hankels(first: u32, last: u32, z: Complex) -> Result<Vec<Complex>> {
    let count = usize::try_from(last.saturating_sub(first) + 1)
        .map_err(|_| Error::InvalidInput("invalid radial order".into()))?;
    let prefactor = spherical_prefactor(z);
    Ok(sequence(Bessel::H1, f64::from(first) + 0.5, z, count)?
        .into_iter()
        .map(|h| prefactor * h)
        .collect())
}

/// Integer-order cylindrical J (regular) or outgoing H1 (singular), with its first two
/// derivatives.
///
/// Upstream: `treams.special.jv` and `hankel1` with their `_d` derivatives.
pub fn cylindrical_radial(order: i32, z: Complex, radial: Radial) -> Result<RadialJet> {
    if order.unsigned_abs() > MAX_ORDER.unsigned_abs() || !finite(z) {
        return Err(Error::InvalidInput(
            // 256 is MAX_ORDER.
            "require finite argument and |m| <= 256".into(),
        ));
    }
    let m = order.unsigned_abs();
    let sign = if order < 0 && m % 2 == 1 { -1.0 } else { 1.0 };
    if z == Complex::default() {
        if radial == Radial::Singular {
            return Err(Error::InvalidInput(
                "outgoing cylindrical waves are singular at zero".into(),
            ));
        }
        return Ok(RadialJet {
            value: Complex::new(if m == 0 { 1.0 } else { 0.0 }, 0.0),
            first: Complex::new(if m == 1 { 0.5 * sign } else { 0.0 }, 0.0),
            second: Complex::new(
                if m == 0 {
                    -0.5
                } else if m == 2 {
                    0.25
                } else {
                    0.0
                },
                0.0,
            ),
        });
    }
    let degree = f64::from(m);
    if radial == Radial::Regular && z.norm() < SERIES_RADIUS {
        // J_m(z) = sum_q (-z^2/4)^q (z/2)^m / (q! (m + q)!).
        let leading = sign * (-log_factorial(order.abs())).exp() / 2.0_f64.powf(degree);
        return Ok(even_series(m, leading, z, |q| {
            -1.0 / (4.0 * q * (degree + q))
        }));
    }
    let [value, upper] = pair(sequence(radial.into(), degree, z, 2)?)?;
    let value = sign * value;
    let first = degree * value / z - sign * upper;
    let second = (degree.powi(2) / z.powu(2) - 1.0) * value - first / z;
    Ok(RadialJet {
        value,
        first,
        second,
    })
}

/// A spherical or cylindrical radial function of integer `order` at a jet argument:
/// without derivatives a single value-only Bessel evaluation, otherwise the value
/// and first derivative from [`spherical_radial`] or [`cylindrical_radial`].
pub(crate) fn radial_jet<const N: usize>(
    order: i32,
    z: Jet<N>,
    radial: Radial,
    spherical: bool,
) -> Result<Jet<N>> {
    if N == 0 {
        let value = bessel(f64::from(order), z.value, radial.into(), spherical, 0)?;
        return Ok(Jet::constant(value));
    }
    let jet = if spherical {
        spherical_radial(order.unsigned_abs(), z.value, radial)?
    } else {
        cylindrical_radial(order, z.value, radial)?
    };
    Ok(z.chain(jet.value, jet.first))
}

#[cfg(test)]
mod tests {
    //! Radial jets and sequences against single Bessel evaluations; the Bessel identities
    //! of the public functions are in `properties/special.rs`.

    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::{
        Bessel, Radial, SERIES_RADIUS, bessel, cylindrical_radial, spherical_radial,
        spherical_radial_sequence,
    };
    use crate::{
        Complex, Error,
        numerics::ratio,
        test_support::{ALGEBRA_CASES, log_polar, prop_assert_close, radial},
    };

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn spherical_radial_sequences_match_single_degrees(
            order in 0_u32..=40,
            // Just below and above the series switch.
            z in prop_oneof![
                log_polar(-3.0..(0.996 * SERIES_RADIUS).log10(), -3.0..3.0),
                log_polar(SERIES_RADIUS.log10()..2.0, -1.2..1.2),
            ],
            radial in radial(),
        ) {
            check_spherical_radial_sequence(order, z, radial)?;
        }

        #[test]
        fn cylindrical_jets_match_bessel(
            order in -20_i32..=20,
            z in log_polar(-5.0..1.3, -3.0..3.0),
            radial in radial(),
        ) {
            check_cylindrical_jet(order, z, radial)?;
        }
    }

    /// Each jet of a spherical sequence equals the single-degree jet: bit for bit for
    /// regular waves, whose J evaluations are shared, and to 1e-13 of the jet's size for
    /// singular waves (in the right half-plane of translation arguments), whose Hankel
    /// functions come from one AMOS sequence. The jet's size covers values that are
    /// small next to their neighbours, where both evaluations lose relative accuracy.
    fn check_spherical_radial_sequence(
        order: u32,
        z: Complex,
        radial: Radial,
    ) -> Result<(), TestCaseError> {
        let sequence = spherical_radial_sequence(order, z, radial).unwrap();
        prop_assert_eq!(sequence.len(), usize::try_from(order + 1).unwrap());
        for (l, jet) in (0..).zip(sequence) {
            let single = spherical_radial(l, z, radial).unwrap();
            let pairs = [
                (jet.value, single.value),
                (jet.first, single.first),
                (jet.second, single.second),
            ];
            let size = single.value.norm() + single.first.norm() + single.second.norm();
            for (d, (actual, expected)) in pairs.into_iter().enumerate() {
                if radial == Radial::Regular {
                    prop_assert_eq!(actual, expected, "degree {} derivative {}", l, d);
                } else {
                    let tolerance = 1e-13 * size;
                    prop_assert_close!(
                        actual,
                        expected,
                        tolerance,
                        "degree {} derivative {}",
                        l,
                        d
                    );
                }
            }
        }
        Ok(())
    }

    /// The cylindrical jet is a power series below `|z| = SERIES_RADIUS` (0.5) and a
    /// two-term AMOS sequence above it. Both match single-order [`bessel`] values and
    /// their order-difference or Hankel-sequence derivatives.
    fn check_cylindrical_jet(order: i32, z: Complex, radial: Radial) -> Result<(), TestCaseError> {
        let jet = cylindrical_radial(order, z, radial).unwrap();
        let kind = Bessel::from(radial);
        // Two-term J sequences differ from single J evaluations by up to 4e-13.
        let tolerance = if radial == Radial::Regular && z.norm() < SERIES_RADIUS {
            2e-13
        } else {
            2e-12
        };
        for (d, actual) in [0, 1, 2]
            .into_iter()
            .zip([jet.value, jet.first, jet.second])
        {
            let expected = bessel(f64::from(order), z, kind, false, d).unwrap();
            prop_assert_close!(
                actual,
                expected,
                tolerance * expected.norm(),
                "derivative {}",
                d
            );
        }
        Ok(())
    }

    #[test]
    fn cylindrical_tiny_nonzero_arguments_keep_representable_terms() {
        for z in [Complex::new(1e-200, 0.0), Complex::new(1e-200, -2e-200)] {
            let j0 = cylindrical_radial(0, z, Radial::Regular).unwrap();
            let j1 = cylindrical_radial(1, z, Radial::Regular).unwrap();
            let j2 = cylindrical_radial(2, z, Radial::Regular).unwrap();
            assert!((ratio(j1.value, 0.5 * z) - 1.0).norm() < 1e-14);
            assert!((ratio(j0.first, -0.5 * z) - 1.0).norm() < 1e-14);
            assert!((ratio(j2.first, 0.25 * z) - 1.0).norm() < 1e-14);
            assert!((ratio(j1.second, -0.375 * z) - 1.0).norm() < 1e-14);
        }
    }

    #[test]
    fn regular_jets_at_the_origin_and_singular_outgoing_waves() {
        // j_l^(k)(0) = k! [z^k] j_l: 1, 0, -1/3 for l = 0; 0, 1/3, 0 for l = 1; 0, 0, 2/15 for l = 2.
        let exact = [
            [1.0, 0.0, -1.0 / 3.0],
            [0.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, 2.0 / 15.0],
        ];
        for (l, expected) in (0..).zip(exact) {
            let jet = spherical_radial(l, Complex::default(), Radial::Regular).unwrap();
            for (actual, expected) in [jet.value, jet.first, jet.second].into_iter().zip(expected) {
                assert!(
                    (actual - expected).norm() < 1e-16,
                    "l = {l}: {actual} != {expected}"
                );
            }
        }
        // J_m^(k)(0) = k! [z^k] J_m: 1, 0, -1/2 for m = 0; 0, 1/2, 0 for m = 1; 0, 0, 1/4 for m = 2.
        let exact = [[1.0, 0.0, -0.5], [0.0, 0.5, 0.0], [0.0, 0.0, 0.25]];
        for (m, expected) in (0..).zip(exact) {
            for (order, sign) in [(m, 1.0), (-m, if m % 2 == 0 { 1.0 } else { -1.0 })] {
                let jet = cylindrical_radial(order, Complex::default(), Radial::Regular).unwrap();
                for (actual, expected) in
                    [jet.value, jet.first, jet.second].into_iter().zip(expected)
                {
                    assert_eq!(actual, Complex::new(sign * expected, 0.0), "m = {order}");
                }
            }
        }
        assert!(spherical_radial(1, Complex::default(), Radial::Singular).is_err());
        assert!(cylindrical_radial(1, Complex::default(), Radial::Singular).is_err());
        // Only zero itself is singular. Near it the jets overflow instead, in AMOS (h_2 ~
        // 3i/z^3) or in the derivatives' powers of 1/z: h_0 = -i e^(iz)/z has h_0'' ~
        // 2i/z^3, beyond 1e308 below |z| = 1e-103.
        let overflows = |result: crate::Result<()>, case: &str| {
            let Err(Error::SpecialFunction(message)) = result else {
                panic!("{case}: expected an overflow, got {result:?}");
            };
            assert!(message.contains("overflow"), "{case}: {message}");
        };
        for (radius, phase) in [(1e-200, -0.8), (1e-200, 2.9), (1e-120, 0.3), (1e-120, -2.2)] {
            let tiny = Complex::from_polar(radius, phase);
            for l in 0..=2 {
                let case = format!("h_{l}({tiny:e})");
                overflows(spherical_radial(l, tiny, Radial::Singular).map(drop), &case);
                overflows(
                    spherical_radial_sequence(l, tiny, Radial::Singular).map(drop),
                    &case,
                );
            }
        }
    }
}

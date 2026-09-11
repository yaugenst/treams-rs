//! Complex spherical Bessel functions and their analytic derivatives.

use std::f64::consts::PI;

use crate::{Complex, Error, Result, finite};
use rayon::prelude::*;

/// Cylindrical Bessel solution, also used at half order for spherical functions.
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

#[inline]
fn bessel_raw(order: f64, z: Complex, kind: Bessel, spherical: bool) -> Result<Complex> {
    let order = if spherical { order + 0.5 } else { order };
    let value = match kind {
        Bessel::J => complex_bessel::besselj(order, z),
        Bessel::Y => complex_bessel::bessely(order, z),
        Bessel::H1 => complex_bessel::hankel1(order, z),
        Bessel::H2 => complex_bessel::hankel2(order, z),
    }
    .map_err(|e| Error::SpecialFunction(e.to_string()))?;
    Ok(if spherical {
        (PI / (2.0 * z)).sqrt() * value
    } else {
        value
    })
}

/// Bessel value or one of its first two complex-argument derivatives.
///
/// Order is held fixed. Spherical regular functions use analytic origin limits.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Integral order in 0..=256 checked before conversion.
#[inline]
pub fn bessel(
    order: f64,
    z: Complex,
    kind: Bessel,
    spherical_kind: bool,
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
    if spherical_kind
        && kind == Bessel::J
        && z.norm() < 0.5
        && (0.0..=256.0).contains(&order)
        && order.fract() == 0.0
    {
        let jet = spherical(order as u32, z, Radial::Regular)?;
        return Ok(match derivative {
            0 => jet.value,
            1 => jet.first,
            _ => jet.second,
        });
    }
    let evaluate = |v| bessel_raw(v, z, kind, spherical_kind);
    let value = if derivative == 0 {
        evaluate(order)?
    } else if !spherical_kind
        && matches!(kind, Bessel::H1 | Bessel::H2)
        && (order != 0.0 || derivative == 2)
    {
        let sequence = match kind {
            Bessel::H1 => complex_bessel::hankel1_seq(
                order - f64::from(derivative),
                z,
                usize::from(2 * derivative + 1),
                complex_bessel::Scaling::Unscaled,
            ),
            _ => complex_bessel::hankel2_seq(
                order - f64::from(derivative),
                z,
                usize::from(2 * derivative + 1),
                complex_bessel::Scaling::Unscaled,
            ),
        }
        .map_err(|e| Error::SpecialFunction(e.to_string()))?;
        match sequence.values.as_slice() {
            [left, _, right] => 0.5 * (left - right),
            [left, _, middle, _, right] => 0.25 * (left - 2.0 * middle + right),
            _ => {
                return Err(Error::SpecialFunction(
                    "invalid Hankel sequence length".into(),
                ));
            }
        }
    } else if spherical_kind {
        let f = evaluate(order)?;
        let first = order * crate::ratio(f, z) - evaluate(order + 1.0)?;
        if derivative == 1 {
            first
        } else {
            -2.0 * crate::ratio(first, z)
                + order * (order + 1.0) * crate::ratio(crate::ratio(f, z), z)
                - f
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
        return Err(Error::SpecialFunction("nonfinite Bessel result".into()));
    }
    Ok(value)
}

/// Elementwise Bessel arguments retained for a complex-argument pullback.
#[derive(Debug)]
pub struct BesselResidual {
    orders: Vec<f64>,
    arguments: Vec<Complex>,
    kind: Bessel,
    spherical: bool,
    derivative: u8,
    size: usize,
}

#[allow(clippy::indexing_slicing)] // Scalar-or-element indexing after equal-length validation.
fn element<T: Copy>(values: &[T], i: usize) -> T {
    values[if values.len() == 1 { 0 } else { i }]
}

/// Evaluate borrowed elementwise Bessel arguments without recording a pullback.
pub fn bessel_values(
    orders: &[f64],
    arguments: &[Complex],
    kind: Bessel,
    spherical: bool,
    derivative: u8,
) -> Result<Vec<Complex>> {
    let size = if orders.is_empty() || arguments.is_empty() {
        0
    } else {
        orders.len().max(arguments.len())
    };
    if (orders.len() != size && orders.len() != 1)
        || (arguments.len() != size && arguments.len() != 1)
        || derivative > 1
    {
        return Err(Error::InvalidInput(
            "Bessel arrays must have equal lengths or scalar inputs, and derivative 0 or 1".into(),
        ));
    }
    let evaluate = |i| {
        bessel(
            element(orders, i),
            element(arguments, i),
            kind,
            spherical,
            derivative,
        )
    };
    if size >= 64 {
        (0..size).into_par_iter().map(evaluate).collect()
    } else {
        (0..size).map(evaluate).collect()
    }
}

/// Evaluate elementwise Bessel functions, broadcasting either scalar input.
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

    /// Contract the holomorphic argument derivative; scalar inputs receive a sum.
    pub fn pullback(self, cotangent: &[Complex]) -> Result<Vec<Complex>> {
        if cotangent.len() != self.size || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "Bessel cotangent must be finite and match output".into(),
            ));
        }
        let evaluate = |(i, &g): (usize, &Complex)| {
            if g == Complex::default() {
                Ok(g)
            } else {
                Ok(g * self.evaluate(i, self.derivative + 1)?.conj())
            }
        };
        let result: Vec<_> = if self.size >= 64 {
            cotangent
                .par_iter()
                .enumerate()
                .map(evaluate)
                .collect::<Result<_>>()?
        } else {
            cotangent
                .iter()
                .enumerate()
                .map(evaluate)
                .collect::<Result<_>>()?
        };
        Ok(if self.arguments.len() == 1 {
            vec![result.into_iter().sum()]
        } else {
            result
        })
    }
}

/// Regular or outgoing radial solution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Radial {
    /// Spherical Bessel function of the first kind.
    Regular,
    /// Outgoing spherical Hankel function of the first kind.
    Outgoing,
}

/// A radial function and its first two argument derivatives.
#[derive(Clone, Copy, Debug)]
pub struct RadialJet {
    /// Function value.
    pub value: Complex,
    /// First derivative with respect to the complex argument.
    pub first: Complex,
    /// Second derivative with respect to the complex argument.
    pub second: Complex,
}

fn raw(l: u32, z: Complex, kind: Radial) -> Result<Complex> {
    let order = f64::from(l) + 0.5;
    let value = match kind {
        Radial::Regular => complex_bessel::besselj(order, z),
        Radial::Outgoing => complex_bessel::hankel1(order, z),
    }
    .map_err(|error| Error::SpecialFunction(error.to_string()))?;
    Ok((PI / (2.0 * z)).sqrt() * value)
}

/// Evaluate a spherical radial function and its analytic derivatives.
///
/// The regular solution uses its convergent power series near zero; away from
/// zero, complex Bessel evaluation and the spherical differential equation apply.
pub fn spherical(l: u32, z: Complex, kind: Radial) -> Result<RadialJet> {
    if l > 256 || !finite(z) {
        return Err(Error::InvalidInput(
            "require finite argument and 0 <= l <= 256".into(),
        ));
    }
    if kind == Radial::Regular && z.norm() < 0.5 {
        let mut denominator = 1.0;
        for i in 0..=l {
            denominator *= f64::from(2 * i + 1);
        }
        let mut jet = RadialJet {
            value: Complex::default(),
            first: Complex::default(),
            second: Complex::default(),
        };
        let mut coefficient = denominator.recip();
        for k in 0..32 {
            let exponent = l + 2 * k;
            jet.value += coefficient * z.powu(exponent);
            if exponent >= 1 {
                jet.first += coefficient * f64::from(exponent) * z.powu(exponent - 1);
            }
            if exponent >= 2 {
                jet.second +=
                    coefficient * f64::from(exponent * (exponent - 1)) * z.powu(exponent - 2);
            }
            coefficient /= -2.0 * f64::from(k + 1) * f64::from(2 * l + 2 * k + 3);
        }
        return Ok(jet);
    }
    if z.norm_sqr() == 0.0 {
        return Err(Error::InvalidInput(
            "outgoing spherical waves are singular at zero".into(),
        ));
    }
    let value = raw(l, z, kind)?;
    let first = f64::from(l) * value / z - raw(l + 1, z, kind)?;
    let second = -2.0 * first / z + (f64::from(l * (l + 1)) / (z * z) - 1.0) * value;
    Ok(RadialJet {
        value,
        first,
        second,
    })
}

/// Associated Legendre polynomial with the treams complex-argument convention.
#[must_use]
pub fn legendre(l: i32, m: i32, z: Complex) -> Complex {
    if l < 0 || m.abs() > l {
        return Complex::default();
    }
    legendre_factor(l, m, crate::jet::Jet::<0>::constant(z)).value
        * (1.0 - z * z).sqrt().powi(m.abs())
}

// Associated Legendre polynomial after factoring out sin(theta)^|m|. Keeping this
// factor explicit avoids 0/0 and cancellation in angular functions near the poles.
pub(crate) fn legendre_factor<const N: usize>(
    l: i32,
    m: i32,
    z: crate::jet::Jet<N>,
) -> crate::jet::Jet<N> {
    use crate::jet::Jet;
    if l < 0 || m.abs() > l {
        return Jet::default();
    }
    let mut p = Jet::constant(1.0);
    if m >= 0 {
        for k in 1..=m {
            p *= -f64::from(2 * k - 1);
        }
    } else {
        for k in 1..=-m {
            p *= 1.0 / f64::from(2 * k);
        }
    }
    if l == m.abs() {
        return p;
    }
    let mut prev = p;
    p *= (f64::from(2 * m.abs() + 1) / f64::from(m.abs() - m + 1)) * z;
    for k in (m.abs() + 2)..=l {
        let next =
            (f64::from(2 * k - 1) * z * p - f64::from(k + m - 1) * prev) * (1.0 / f64::from(k - m));
        prev = p;
        p = next;
    }
    p
}

/// Local analytic angular functions with the sine factor kept explicit at poles.
pub(crate) fn angular_jets<const N: usize>(
    l: i32,
    m: i32,
    cosine: crate::jet::Jet<N>,
    sine: crate::jet::Jet<N>,
) -> [crate::jet::Jet<N>; 2] {
    use crate::jet::Jet;
    let legendre = |order: i32| sine.powi(order.abs()) * legendre_factor(l, order, cosine);
    let pi = if m == 0 {
        Jet::default()
    } else {
        f64::from(m) * sine.powi(m.abs() - 1) * legendre_factor(l, m, cosine)
    };
    let tau = 0.5 * (legendre(m + 1) - f64::from((l + m) * (l - m + 1)) * legendre(m - 1));
    [pi, tau]
}

/// Angular pi function, including its polar-axis limits.
#[must_use]
pub fn pi_fun(l: i32, m: i32, z: Complex) -> Complex {
    let st = (1.0 - z * z).sqrt();
    if st.norm_sqr() < 1e-40 {
        return match m {
            1 => -z.powi(l + 1) * f64::from(l * (l + 1)) * 0.5,
            -1 => -z.powi(l + 1) * 0.5,
            _ => Complex::default(),
        };
    }
    f64::from(m) * legendre(l, m, z) / st
}

/// Angular tau function using adjacent-order associated Legendre polynomials.
#[must_use]
pub fn tau_fun(l: i32, m: i32, z: Complex) -> Complex {
    if l == m {
        return -f64::from(l) * legendre(l, m - 1, z);
    }
    if l == -m {
        return 0.5 * legendre(l, m + 1, z);
    }
    0.5 * (legendre(l, m + 1, z) - f64::from((l + m) * (l - m + 1)) * legendre(l, m - 1, z))
}

/// Integer-order cylindrical J or outgoing H, with first two derivatives.
pub fn cylindrical(order: i32, z: Complex, radial: Radial) -> Result<RadialJet> {
    if order.unsigned_abs() > 256 || !finite(z) {
        return Err(Error::InvalidInput(
            "require finite argument and |m| <= 256".into(),
        ));
    }
    let m = order.unsigned_abs();
    let sign = if order < 0 && m % 2 == 1 { -1.0 } else { 1.0 };
    if z.norm_sqr() == 0.0 {
        if radial == Radial::Outgoing {
            return Err(Error::SpecialFunction(
                "outgoing cylindrical wave is singular at zero".into(),
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
    if radial == Radial::Regular && z.norm() < 0.5 {
        let mut result = RadialJet {
            value: Complex::default(),
            first: Complex::default(),
            second: Complex::default(),
        };
        let mut coefficient =
            sign * (-libm::lgamma(f64::from(m) + 1.0)).exp() / 2.0_f64.powf(f64::from(m));
        for q in 0_u32..32 {
            let power = m + 2 * q;
            result.value += coefficient * z.powu(power);
            if power > 0 {
                result.first += coefficient * f64::from(power) * z.powu(power - 1);
            }
            if power > 1 {
                result.second += coefficient * f64::from(power * (power - 1)) * z.powu(power - 2);
            }
            coefficient /= -4.0 * f64::from(q + 1) * f64::from(m + q + 1);
        }
        return Ok(result);
    }
    let sequence = match radial {
        Radial::Regular => {
            complex_bessel::besselj_seq(f64::from(m), z, 2, complex_bessel::Scaling::Unscaled)
        }
        Radial::Outgoing => {
            complex_bessel::hankel1_seq(f64::from(m), z, 2, complex_bessel::Scaling::Unscaled)
        }
    }
    .map_err(|e| Error::SpecialFunction(e.to_string()))?;
    let [value, upper] = sequence.values.as_slice() else {
        return Err(Error::SpecialFunction(
            "invalid cylindrical Bessel sequence length".into(),
        ));
    };
    let value = sign * value;
    let first = f64::from(m) * value / z - sign * upper;
    let second = (f64::from(m).powi(2) / z.powu(2) - 1.0) * value - first / z;
    Ok(RadialJet {
        value,
        first,
        second,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_origin_and_wronskian() -> Result<()> {
        let origin = spherical(1, Complex::default(), Radial::Regular)?;
        assert!(origin.value.norm() < 1e-15);
        assert!((origin.first.re - 1.0 / 3.0).abs() < 1e-15);
        for z in [Complex::new(0.2, 0.1), Complex::new(3.0, -0.2)] {
            for l in 0..8 {
                let j = spherical(l, z, Radial::Regular)?;
                let h = spherical(l, z, Radial::Outgoing)?;
                assert!(
                    (j.value * h.first - j.first * h.value - Complex::i() / z.powu(2)).norm()
                        < 1e-10
                );
            }
        }
        Ok(())
    }
}

/// Associated Legendre polynomial or one of the two vector-wave angular functions.
#[derive(Clone, Copy, Debug)]
pub enum Angular {
    /// Associated Legendre P, including its Condon-Shortley phase.
    Legendre,
    /// m P / sqrt(1-z^2), with analytic polar limits.
    Pi,
    /// Polar-angle derivative of P(cos(theta)).
    Tau,
}

fn sine_power<const N: usize>(z: crate::jet::Jet<N>, power: i32) -> crate::jet::Jet<N> {
    use crate::jet::Jet;
    let w = 1.0 - z * z;
    if power % 2 == 0 {
        w.powi(power / 2)
    } else if power > 1 && w.value == Complex::default() {
        // Odd powers >=3 have zero value and first derivative at the poles.
        Jet::default()
    } else {
        w.powi(power / 2) * w.sqrt()
    }
}

fn angular_legendre<const N: usize>(
    l: i32,
    m: i32,
    z: crate::jet::Jet<N>,
    power: i32,
) -> crate::jet::Jet<N> {
    if m.abs() > l {
        return crate::jet::Jet::default();
    }
    legendre_factor(l, m, z) * sine_power(z, power)
}

#[allow(clippy::cast_possible_truncation)] // Integer labels are checked and bounded before conversion.
fn angular_jet<const N: usize>(
    l: f64,
    m: f64,
    z: Complex,
    kind: Angular,
) -> Result<crate::jet::Jet<N>> {
    use crate::jet::Jet;
    if !finite(z)
        || !l.is_finite()
        || !m.is_finite()
        || l.fract() != 0.0
        || m.fract() != 0.0
        || !(0.0..=128.0).contains(&l)
    {
        return Err(Error::InvalidInput("angular functions require integer 0 <= degree <= 128, integer order and finite argument".into()));
    }
    if m.abs() > l {
        return Ok(Jet::default());
    }
    let (l, m) = (l as i32, m as i32);
    let z = Jet::<N>::variable(z, 0);
    let value = match kind {
        Angular::Legendre => angular_legendre(l, m, z, m.abs()),
        Angular::Pi if m == 0 => Jet::default(),
        Angular::Pi => f64::from(m) * angular_legendre(l, m, z, m.abs() - 1),
        Angular::Tau if m == 0 => angular_legendre(l, 1, z, 1),
        Angular::Tau => {
            // Adjacent orders share sin(theta)^(|m|-1); evaluate its root once.
            let mut upper = legendre_factor(l, m + 1, z);
            let mut lower = f64::from((l + m) * (l - m + 1)) * legendre_factor(l, m - 1, z);
            if m > 0 {
                upper *= 1.0 - z * z;
            } else {
                lower *= 1.0 - z * z;
            }
            0.5 * (upper - lower) * sine_power(z, m.abs() - 1)
        }
    };
    if !value.finite() {
        return Err(Error::SpecialFunction(
            if N == 0 {
                "nonfinite angular result"
            } else {
                "angular argument derivative is undefined or nonfinite"
            }
            .into(),
        ));
    }
    Ok(value)
}

/// Integer-degree angular function at a complex cosine argument.
pub fn angular_value(l: f64, m: f64, z: Complex, kind: Angular) -> Result<Complex> {
    Ok(angular_jet::<0>(l, m, z, kind)?.value)
}

/// Borrowed angular values with scalar-or-equal-length broadcasting.
pub fn angular_values(
    degrees: &[f64],
    orders: &[f64],
    arguments: &[Complex],
    kind: Angular,
) -> Result<Vec<Complex>> {
    let sizes = [degrees.len(), orders.len(), arguments.len()];
    let size = if sizes.contains(&0) {
        0
    } else {
        sizes.into_iter().max().unwrap_or_default()
    };
    if sizes.iter().any(|&n| n != 1 && n != size) {
        return Err(Error::InvalidInput(
            "angular arrays must have equal lengths or scalar inputs".into(),
        ));
    }
    let evaluate = |i| {
        angular_value(
            element(degrees, i),
            element(orders, i),
            element(arguments, i),
            kind,
        )
    };
    if size >= 1024 {
        (0..size).into_par_iter().map(evaluate).collect()
    } else {
        (0..size).map(evaluate).collect()
    }
}

/// Owned angular arguments; reverse recomputes local analytic derivatives.
#[derive(Debug)]
pub struct AngularResidual {
    degrees: Vec<f64>,
    orders: Vec<f64>,
    arguments: Vec<Complex>,
    kind: Angular,
    size: usize,
}

/// Broadcast angular values and a complex-argument residual; labels stay fixed.
pub fn angular_array(
    degrees: Vec<f64>,
    orders: Vec<f64>,
    arguments: Vec<Complex>,
    kind: Angular,
) -> Result<(Vec<Complex>, AngularResidual)> {
    let values = angular_values(&degrees, &orders, &arguments, kind)?;
    let size = values.len();
    Ok((
        values,
        AngularResidual {
            degrees,
            orders,
            arguments,
            kind,
            size,
        },
    ))
}

impl AngularResidual {
    /// Contract the argument derivative. A zero cotangent skips singular derivatives.
    pub fn pullback(self, cotangent: &[Complex]) -> Result<Vec<Complex>> {
        if cotangent.len() != self.size || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "angular cotangent must be finite and match output".into(),
            ));
        }
        let evaluate = |(i, &g): (usize, &Complex)| {
            if g == Complex::default() {
                return Ok(g);
            }
            let jet = angular_jet::<1>(
                element(&self.degrees, i),
                element(&self.orders, i),
                element(&self.arguments, i),
                self.kind,
            )?;
            Ok(g * jet.derivative.first().copied().unwrap_or_default().conj())
        };
        let values: Vec<_> = if self.size >= 1024 {
            cotangent
                .par_iter()
                .enumerate()
                .map(evaluate)
                .collect::<Result<_>>()?
        } else {
            cotangent
                .iter()
                .enumerate()
                .map(evaluate)
                .collect::<Result<_>>()?
        };
        Ok(if self.arguments.len() == 1 {
            vec![values.into_iter().sum()]
        } else {
            values
        })
    }
}

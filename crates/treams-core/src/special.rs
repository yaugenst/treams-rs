//! Complex spherical Bessel functions and their analytic derivatives.

use std::f64::consts::PI;

use crate::{Complex, Error, Result, finite};

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
    p *= f64::from(2 * m.abs() + 1) * z / f64::from(m.abs() - m + 1);
    for k in (m.abs() + 2)..=l {
        let next = (f64::from(2 * k - 1) * z * p - f64::from(k + m - 1) * prev) / f64::from(k - m);
        prev = p;
        p = next;
    }
    p
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
    let evaluate = |degree: u32| {
        match radial {
            Radial::Regular => complex_bessel::besselj(f64::from(degree), z),
            Radial::Outgoing => complex_bessel::hankel1(f64::from(degree), z),
        }
        .map_err(|e| Error::SpecialFunction(e.to_string()))
    };
    let value = sign * evaluate(m)?;
    let first = f64::from(m) * value / z - sign * evaluate(m + 1)?;
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

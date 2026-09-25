//! Spherical-wave basis ordering and translation kernels.
// Array indices denote three Cartesian axes and validated two-channel modes.
#![allow(clippy::indexing_slicing)]

use crate::angular::{angular, wigner3j};
use crate::special::{Radial, spherical};
use crate::{Complex, Error, Result};

/// One spherical-wave mode, ordered by degree, order, then positive helicity first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mode {
    /// Multipole degree.
    pub l: i32,
    /// Azimuthal order.
    pub m: i32,
    /// Polarization index: 1 or 0, as in treams.
    pub pol: u8,
}

impl Mode {
    /// Validate a mode at the public numerical boundary.
    pub fn validate(self) -> Result<()> {
        if !(1..=128).contains(&self.l)
            || self.m.unsigned_abs() > self.l.unsigned_abs()
            || self.pol > 1
        {
            return Err(Error::InvalidInput("invalid spherical mode".into()));
        }
        Ok(())
    }
}

/// Construct the standard treams spherical basis.
pub fn modes(lmax: u32) -> Result<Vec<Mode>> {
    if !(1..=128).contains(&lmax) {
        return Err(Error::InvalidInput("require 1 <= lmax <= 128".into()));
    }
    let lmax = i32::try_from(lmax).map_err(|_| Error::InvalidInput("invalid lmax".into()))?;
    Ok((1..=lmax)
        .flat_map(|l| (-l..=l).flat_map(move |m| [Mode { l, m, pol: 1 }, Mode { l, m, pol: 0 }]))
        .collect())
}

/// Translation coefficient with Cartesian and wavenumber derivatives.
#[derive(Clone, Copy, Debug, Default)]
pub struct Translation {
    /// Coefficient value.
    pub value: Complex,
    /// Derivatives with respect to the displacement components.
    pub position: [Complex; 3],
    /// Derivative with respect to the wave number.
    pub k: Complex,
}

fn helper(l: i32, m: i32, lambda: i32, mu: i32, p: i32, q: i32) -> Complex {
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
        * (0.5
            * (libm::lgamma(f64::from(p - m - mu + 1)) - libm::lgamma(f64::from(p + m + mu + 1))))
        .exp()
        * wigner3j(l, lambda, p, m, mu, -m - mu)
        * wigner3j(l, lambda, q, 0, 0, 0)
}

/// Translate a source mode to a destination basis origin.
/// `displacement = destination_origin - source_origin`.
pub fn translate(
    to: Mode,
    from: Mode,
    k: Complex,
    displacement: [f64; 3],
    helicity: bool,
    radial: Radial,
) -> Result<Translation> {
    to.validate()?;
    from.validate()?;
    if !crate::finite(k) || displacement.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "wavenumber and displacement must be finite".into(),
        ));
    }
    let mut result = Translation::default();
    let r = displacement.iter().map(|v| v * v).sum::<f64>().sqrt();
    if r == 0.0 && radial == Radial::Outgoing {
        return Ok(result);
    }
    for (p, m, weight) in terms(to, from, helicity) {
        let radial_value = spherical(
            u32::try_from(p).map_err(|_| Error::InvalidInput("negative radial order".into()))?,
            k * r,
            radial,
        )?;
        let term = harmonic(p, m, k, displacement, radial_value);
        result.value += weight * term.value;
        result.k += weight * term.k;
        for (g, derivative) in result.position.iter_mut().zip(term.position) {
            *g += weight * derivative;
        }
    }
    Ok(result)
}

pub(crate) fn harmonic(
    p: i32,
    m: i32,
    k: Complex,
    position: [f64; 3],
    radial: crate::special::RadialJet,
) -> Translation {
    let r = position.iter().map(|v| v * v).sum::<f64>().sqrt();
    if r == 0.0 {
        let mut result = Translation::default();
        if p == 0 {
            result.value = radial.value;
        }
        // j_1(kr) P_1^m(z/r) exp(im phi) is linear at the origin.
        if p == 1 {
            let gradient = match m {
                -1 => [
                    Complex::new(0.5, 0.0),
                    Complex::new(0.0, -0.5),
                    Complex::default(),
                ],
                0 => [
                    Complex::default(),
                    Complex::default(),
                    Complex::new(1.0, 0.0),
                ],
                1 => [
                    Complex::new(-1.0, 0.0),
                    Complex::new(0.0, -1.0),
                    Complex::default(),
                ],
                _ => [Complex::default(); 3],
            };
            result.position = gradient.map(|g| k * radial.first * g);
        }
        return result;
    }
    let (value, gradient) = angular(p, m, position);
    Translation {
        value: radial.value * value,
        k: radial.first * r * value,
        position: std::array::from_fn(|axis| {
            radial.first * k * position[axis] / r * value + radial.value * gradient[axis]
        }),
    }
}

/// Degrees `p` that `helper` admits for source `(l, m)` and destination `(lambda, mu)`,
/// descending in steps of two. `formal/Formal/SelectionRules.lean` proves this range.
pub(crate) fn degrees(
    l: i32,
    m: i32,
    lambda: i32,
    mu: i32,
    cross: bool,
) -> impl Iterator<Item = i32> {
    let start = l + lambda - i32::from(cross);
    let end = (lambda - l)
        .abs()
        .saturating_add(i32::from(cross))
        .max((m - mu).abs());
    (end..=start).rev().step_by(2)
}

pub(crate) fn terms(to: Mode, from: Mode, helicity: bool) -> Vec<(i32, i32, Complex)> {
    let mut terms = Vec::new();
    if helicity && to.pol != from.pol {
        return terms;
    }
    let (l, m, lambda, mu) = (from.l, from.m, to.l, to.m);
    let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
    let pref = 0.5
        * sign
        * (f64::from((2 * l + 1) * (2 * lambda + 1))
            / (f64::from(l * (l + 1)) * f64::from(lambda * (lambda + 1))))
        .sqrt();
    for cross in [false, true] {
        if !helicity && cross == (to.pol == from.pol) {
            continue;
        }
        let polarization = if cross && helicity {
            2.0 * f64::from(from.pol) - 1.0
        } else {
            1.0
        };
        for p in degrees(l, m, lambda, mu, cross) {
            let factor = if cross {
                (f64::from(l + lambda + 1 + p)
                    * f64::from(l + lambda + 1 - p)
                    * f64::from(p + lambda - l)
                    * f64::from(p - lambda + l))
                .sqrt()
            } else {
                f64::from(l * (l + 1) + lambda * (lambda + 1) - p * (p + 1))
            };
            let weight =
                pref * polarization * factor * helper(l, m, lambda, -mu, p, p - i32::from(cross));
            if weight.norm_sqr() > 0.0 {
                terms.push((p, m - mu, weight));
            }
        }
    }
    terms
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    #[test]
    fn degrees_match_lean_model() {
        // `just formal` keeps this file equal to `Treams.SelectionRules.termDegrees`.
        let golden = include_str!("../../../formal/golden/degrees.txt");
        for line in golden.lines() {
            let (case, degrees) = line.split_once(':').unwrap();
            let case: Vec<i32> = case
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            let expected: Vec<i32> = degrees
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            let actual: Vec<_> =
                super::degrees(case[0], case[1], case[2], case[3], case[4] == 1).collect();
            assert_eq!(actual, expected, "{line}");
        }
        assert_eq!(golden.lines().count(), 2450);
    }
}

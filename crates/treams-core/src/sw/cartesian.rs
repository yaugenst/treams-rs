//! Spherical translation coefficients of single mode pairs at Cartesian displacements,
//! with their displacement and wavenumber derivatives.
//!
//! treams-rs extension: treams evaluates translations at polar arguments only.
#![allow(clippy::indexing_slicing)] // Fixed indices of the three Cartesian axes.

use super::{Mode, coupling::terms};
use crate::special::{Radial, direction, on_sphere, spherical_radial};
use crate::{Complex, Error, Result, basis::ModeLabel};

/// Translation coefficient with its Cartesian and wavenumber derivatives.
#[derive(Clone, Copy, Debug, Default)]
pub struct CartesianTranslation {
    /// Coefficient value.
    pub value: Complex,
    /// Derivatives of `value` with respect to the displacement components.
    pub position: [Complex; 3],
    /// Derivative of `value` with respect to the wavenumber `k`.
    pub k: Complex,
}

/// Translation coefficient of the source mode `from` to the destination mode `to`, with
/// its derivatives.
///
/// `displacement` is the destination position minus the source position. Singular waves
/// give zero at the origin and at subnormal displacements.
///
/// treams-rs extension: the Cartesian single-pair jet behind the expansion matrices and
/// the tests of the polar coefficients; `treams.sw.translate` takes polar arguments.
pub fn cartesian_translation(
    to: Mode,
    from: Mode,
    k: Complex,
    displacement: [f64; 3],
    helicity: bool,
    radial: Radial,
) -> Result<CartesianTranslation> {
    to.validate()?;
    from.validate()?;
    if !crate::numerics::finite(k) || displacement.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "wavenumber and displacement must be finite".into(),
        ));
    }
    let mut result = CartesianTranslation::default();
    let geometry = direction(displacement);
    if geometry.is_none() && radial == Radial::Singular {
        return Ok(result);
    }
    let r = geometry.map_or(0.0, |(r, _)| r);
    for (p, m, weight) in terms(to, from, helicity) {
        let radial_value = spherical_radial(
            u32::try_from(p).map_err(|_| Error::InvalidInput("negative radial order".into()))?,
            k * r,
            radial,
        )?;
        let term = harmonic(p, m, k, geometry, radial_value);
        result.value += weight * term.value;
        result.k += weight * term.k;
        for (g, derivative) in result.position.iter_mut().zip(term.position) {
            *g += weight * derivative;
        }
    }
    Ok(result)
}

/// `z_p(k r) P_p^m(cos theta) exp(i m phi)` and its displacement and wavenumber
/// derivatives, from the radial jet at `k r` and the displacement's
/// [`direction`] (`None` at the origin and at subnormal displacements).
pub(crate) fn harmonic(
    p: i32,
    m: i32,
    k: Complex,
    geometry: Option<(f64, [f64; 3])>,
    radial: crate::special::RadialJet,
) -> CartesianTranslation {
    let Some((r, unit)) = geometry else {
        let mut result = CartesianTranslation::default();
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
    };
    let (value, tangent) = on_sphere(p, m, unit);
    combine(k, (r, unit), radial, value, tangent)
}

/// `z_p(k r) Y` and its displacement and wavenumber derivatives away from the origin,
/// from the unit-sphere harmonic `Y` at the direction `unit` and its tangential
/// gradient (see [`crate::special::tangent`]).
pub(crate) fn combine(
    k: Complex,
    (r, unit): (f64, [f64; 3]),
    radial: crate::special::RadialJet,
    value: Complex,
    tangent: [Complex; 3],
) -> CartesianTranslation {
    let scaled = radial.value / r;
    CartesianTranslation {
        value: radial.value * value,
        k: radial.first * r * value,
        position: std::array::from_fn(|axis| {
            radial.first * k * unit[axis] * value + scaled * tangent[axis]
        }),
    }
}

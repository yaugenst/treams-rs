//! Plane-wave illumination in spherical and cylindrical bases.
#![allow(clippy::indexing_slicing)] // Validated three-component vectors and polarizations.

use crate::{
    Complex, Error, Result, finite,
    special::{pi_fun, tau_fun},
    waves::Mode,
};

// Scale before squaring or dividing: nearly axial directions may have transverse
// components small enough that their squares underflow while their azimuth matters.
fn algebraic_norm(values: &[Complex]) -> Complex {
    let scale = values.iter().map(|v| v.norm()).fold(0.0, f64::max);
    if scale == 0.0 {
        return Complex::default();
    }
    values
        .iter()
        .map(|v| (v / scale).powu(2))
        .sum::<Complex>()
        .sqrt()
        * scale
}
fn ratio(numerator: Complex, denominator: Complex) -> Complex {
    let scale = denominator.re.abs().max(denominator.im.abs());
    (numerator / scale) / (denominator / scale)
}

fn wavenumbers(vector: [Complex; 3]) -> Result<(Complex, Complex, [Complex; 2])> {
    if vector.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput("wavevector must be finite".into()));
    }
    let scale = vector[..2].iter().map(|v| v.norm()).fold(0.0, f64::max);
    let (transverse, xy) = if scale == 0.0 {
        (Complex::default(), [Complex::default(); 2])
    } else {
        let scaled = [vector[0] / scale, vector[1] / scale];
        let norm = algebraic_norm(&scaled);
        if norm == Complex::default() {
            return Err(Error::InvalidInput(
                "undefined polarization for a null transverse vector".into(),
            ));
        }
        (norm * scale, scaled.map(|v| ratio(v, norm)))
    };
    let k = algebraic_norm(&vector);
    if k == Complex::default() || !finite(k) {
        return Err(Error::InvalidInput(
            "wavevector must have nonzero algebraic norm".into(),
        ));
    }
    Ok((k, transverse, xy))
}

/// Plane-wave electric vector at the origin in treams normalization.
pub fn polarization(vector: [Complex; 3], pol: u8, helicity: bool) -> Result<[Complex; 3]> {
    if pol > 1 {
        return Err(Error::InvalidInput("polarization must be 0 or 1".into()));
    }
    let (k, transverse, xy) = wavenumbers(vector)?;
    let z = vector[2];
    let (m, n) = if xy == [Complex::default(); 2] {
        let sign = if z.im == 0.0 {
            if z.re >= 0.0 { 1.0 } else { -1.0 }
        } else if z.im >= 0.0 {
            1.0
        } else {
            -1.0
        };
        (
            [Complex::default(), -Complex::i(), Complex::default()],
            [
                Complex::new(-sign, 0.0),
                Complex::default(),
                Complex::default(),
            ],
        )
    } else {
        (
            [
                Complex::i() * xy[1],
                -Complex::i() * xy[0],
                Complex::default(),
            ],
            [
                -xy[0] * ratio(z, k),
                -xy[1] * ratio(z, k),
                ratio(transverse, k),
            ],
        )
    };
    Ok(if helicity {
        std::array::from_fn(|a| {
            (n[a] + (2.0 * f64::from(pol) - 1.0) * m[a]) * std::f64::consts::FRAC_1_SQRT_2
        })
    } else if pol == 0 {
        m
    } else {
        n
    })
}

/// Spherical expansion coefficient for a unit-amplitude plane wave.
pub fn to_spherical(mode: Mode, vector: [Complex; 3], pol: u8, helicity: bool) -> Result<Complex> {
    mode.validate()?;
    if pol > 1 {
        return Err(Error::InvalidInput("polarization must be 0 or 1".into()));
    }
    let (k, _, xy) = wavenumbers(vector)?;
    if helicity && mode.pol != pol {
        return Ok(Complex::default());
    }
    let (l, m) = (mode.l, mode.m);
    let azimuth = if xy == [Complex::default(); 2] {
        Complex::new(1.0, 0.0)
    } else {
        (xy[0] - Complex::i() * xy[1]).powi(m)
    };
    let normalization = 2.0
        * (std::f64::consts::PI * f64::from(2 * l + 1) / f64::from(l * (l + 1))).sqrt()
        * (0.5 * (libm::lgamma(f64::from(l - m + 1)) - libm::lgamma(f64::from(l + m + 1)))).exp();
    let z = if xy == [Complex::default(); 2] {
        Complex::new(if (vector[2] / k).re >= 0.0 { 1.0 } else { -1.0 }, 0.0)
    } else {
        vector[2] / k
    };
    let angular = if helicity {
        tau_fun(l, m, z) + (2.0 * f64::from(pol) - 1.0) * pi_fun(l, m, z)
    } else if mode.pol == pol {
        tau_fun(l, m, z)
    } else {
        pi_fun(l, m, z)
    };
    Ok(normalization * Complex::i().powi(l) * azimuth * angular)
}

/// Regular spherical multipole amplitudes of one plane wave.
pub fn spherical(
    basis: &crate::basis::Basis,
    vector: [Complex; 3],
    pol: u8,
    helicity: bool,
) -> Result<Vec<Complex>> {
    basis.validate()?;
    basis
        .modes
        .iter()
        .map(|&(origin, mode)| {
            let phase = (Complex::i()
                * vector
                    .iter()
                    .zip(basis.positions[origin])
                    .map(|(k, r)| k * r)
                    .sum::<Complex>())
            .exp();
            Ok(phase * to_spherical(mode, vector, pol, helicity)?)
        })
        .collect()
}

/// Regular cylindrical multipole amplitudes of one plane wave.
#[allow(clippy::float_cmp)] // kz is an exact basis mode label, not a tolerance match.
pub fn cylindrical(
    basis: &crate::cylwaves::Basis,
    vector: [Complex; 3],
    pol: u8,
) -> Result<Vec<Complex>> {
    basis.validate()?;
    let (_, _, xy) = wavenumbers(vector)?;
    if pol > 1 || vector[2].im != 0.0 {
        return Err(Error::InvalidInput(
            "cylindrical expansion requires real axial wavenumber and polarization 0 or 1".into(),
        ));
    }
    basis
        .modes
        .iter()
        .map(|&(origin, mode)| {
            if mode.pol != pol || mode.kz != vector[2].re {
                return Ok(Complex::default());
            }
            let angular = if xy == [Complex::default(); 2] {
                Complex::i().powi(mode.m)
            } else {
                (Complex::i() * xy[0] + xy[1]).powi(mode.m)
            };
            let phase = (Complex::i()
                * vector
                    .iter()
                    .zip(basis.positions[origin])
                    .map(|(k, r)| k * r)
                    .sum::<Complex>())
            .exp();
            Ok(phase * angular)
        })
        .collect()
}

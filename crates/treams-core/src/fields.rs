//! Cartesian vector spherical waves without polar-coordinate singularities.
#![allow(clippy::indexing_slicing)] // Fixed Cartesian vectors and Hessians.

use crate::{
    Complex, Error, Result,
    angular::solid,
    finite,
    special::{Radial, spherical},
    waves::Mode,
};

/// Electric vector spherical wave and its analytic Cartesian derivatives.
#[derive(Clone, Copy, Debug)]
pub struct VectorWave {
    /// Cartesian electric field.
    pub value: [Complex; 3],
    /// `position[component][axis]` is the field's Cartesian Jacobian.
    pub position: [[Complex; 3]; 3],
    /// Complex wave number derivative.
    pub k: [Complex; 3],
}

fn cross(a: [Complex; 3], b: [Complex; 3]) -> [Complex; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// z_l(kr)/r^l has a finite regular limit. Evaluating its series directly
// avoids divisions by zero and cancellation in the near-origin field gradients.
fn scaled_radial(l: u32, k: Complex, r: f64, radial: Radial) -> Result<Complex> {
    let x = k * r;
    if radial == Radial::Regular && x.norm() < 0.5 {
        let mut denominator = 1.0;
        for i in 0..=l {
            denominator *= f64::from(2 * i + 1);
        }
        let mut coefficient = k.powu(l) / denominator;
        let mut sum = Complex::default();
        for q in 0..32 {
            sum += coefficient * x.powu(2 * q);
            coefficient /= -2.0 * f64::from(q + 1) * f64::from(2 * l + 2 * q + 3);
        }
        return Ok(sum);
    }
    Ok(spherical(l, x, radial)?.value / r.powf(f64::from(l)))
}

/// Evaluate a vector spherical wave in treams normalization.
/// Parity polarization 0 is M, 1 is N; helicity is `(N + (2pol-1) M)/sqrt(2)`.
pub fn spherical_wave(
    mode: Mode,
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    radial: Radial,
) -> Result<VectorWave> {
    mode.validate()?;
    if !finite(k) || k.norm_sqr() == 0.0 || position.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "require a finite nonzero wavenumber and finite position".into(),
        ));
    }
    let r2 = position.iter().map(|v| v * v).sum::<f64>();
    let l = u32::try_from(mode.l).map_err(|_| Error::InvalidInput("invalid degree".into()))?;
    let degree = f64::from(l);
    let c = scaled_radial(l, k, r2.sqrt(), radial)?;
    let e = scaled_radial(l + 1, k, r2.sqrt(), radial)?;
    let f = scaled_radial(l + 2, k, r2.sqrt(), radial)?;
    let ck = degree / k * c - r2 * e;
    let ek = (degree + 1.0) / k * e - r2 * f;
    let d = (degree + 1.0) / k * c - r2 * e;
    let dk = (degree + 1.0) / k * ck - (degree + 1.0) / k.powu(2) * c - r2 * ek;
    let solid = solid::<true>(mode.l, mode.m, position);
    let vector = position.map(|v| Complex::new(v, 0.0));
    let rotation = cross(vector, solid.gradient);
    let normalization = Complex::i()
        * ((2.0 * degree + 1.0) / (4.0 * std::f64::consts::PI * degree * (degree + 1.0))).sqrt()
        * (0.5
            * (libm::lgamma(f64::from(mode.l - mode.m + 1))
                - libm::lgamma(f64::from(mode.l + mode.m + 1))))
        .exp();
    let (weight_n, weight_m) = if helicity {
        (
            std::f64::consts::FRAC_1_SQRT_2,
            (2.0 * f64::from(mode.pol) - 1.0) * std::f64::consts::FRAC_1_SQRT_2,
        )
    } else if mode.pol == 1 {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let combine = |n, m| normalization * (weight_n * n - weight_m * m);
    let value = std::array::from_fn(|i| {
        combine(
            d * solid.gradient[i] + degree * e * solid.value * position[i],
            c * rotation[i],
        )
    });
    let wave_k = std::array::from_fn(|i| {
        combine(
            dk * solid.gradient[i] + degree * ek * solid.value * position[i],
            ck * rotation[i],
        )
    });
    let mut jacobian = [[Complex::default(); 3]; 3];
    for axis in 0..3 {
        let ca = -k * e * position[axis];
        let ea = -k * f * position[axis];
        let da = (degree + 1.0) / k * ca - 2.0 * position[axis] * e - r2 * ea;
        let mut unit = [Complex::default(); 3];
        unit[axis] = Complex::new(1.0, 0.0);
        let first = cross(unit, solid.gradient);
        let second = cross(vector, std::array::from_fn(|i| solid.hessian[i][axis]));
        for i in 0..3 {
            let dn = da * solid.gradient[i]
                + d * solid.hessian[i][axis]
                + degree
                    * (ea * solid.value * position[i]
                        + e * solid.gradient[axis] * position[i]
                        + if i == axis {
                            e * solid.value
                        } else {
                            Complex::default()
                        });
            let dm = ca * rotation[i] + c * (first[i] + second[i]);
            jacobian[i][axis] = combine(dn, dm);
        }
    }
    Ok(VectorWave {
        value,
        position: jacobian,
        k: wave_k,
    })
}

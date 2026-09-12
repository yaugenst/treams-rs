//! A square array of spheres, illuminated from vacuum below the xy plane.

use std::f64::consts::TAU;

use nalgebra::DMatrix;
use treams_core::{
    Complex, Error, Result, basis, channels, coeffs::Material, illumination, lattice::Lattice,
    tmatrix,
};
use wasm_bindgen::prelude::*;

/// Power diffraction by an infinite square array of identical passive spheres.
///
/// Lengths share arbitrary units; `angle_deg` is incidence from +z in the xz
/// plane. The embedding medium is vacuum, permeability is one, and `helicity`
/// is 0 or 1. One requested illumination is solved using the shared Ewald
/// coupling and multipole-to-plane channels, without constructing a full S matrix.
///
/// Returns `[R, T, A, N]`, then N records
/// `[order_x, order_y, kx/k0, ky/k0, abs(kz)/k0, R_order, T_order]`.
/// All propagating diffraction orders are included, sorted by x then y order.
/// Evanescent lattice coupling is retained by Ewald summation. Absorptance is
/// the unmodified balance `1 - R - T`; tiny negative roundoff is not clipped.
///
/// The interactive contract bounds lmax to 1–4, incidence to ±60 degrees and
/// period/wavelength to at most 2. Exact grazing diffraction thresholds are
/// excluded; callers should show a gap for an unsupported spectral sample.
#[wasm_bindgen]
pub fn metasurface(
    lmax: u32,
    radius: f64,
    period: f64,
    wavelength: f64,
    angle_deg: f64,
    epsilon_re: f64,
    epsilon_im: f64,
    helicity: u8,
) -> std::result::Result<Vec<f64>, JsError> {
    Ok(solve(
        lmax, radius, period, wavelength, angle_deg, epsilon_re, epsilon_im, helicity,
    )?)
}

fn solve(
    lmax: u32,
    radius: f64,
    period: f64,
    wavelength: f64,
    angle_deg: f64,
    epsilon_re: f64,
    epsilon_im: f64,
    helicity: u8,
) -> Result<Vec<f64>> {
    if !(1..=4).contains(&lmax)
        || helicity > 1
        || [
            radius, period, wavelength, angle_deg, epsilon_re, epsilon_im,
        ]
        .iter()
        .any(|x| !x.is_finite())
        || radius <= 0.0
        || wavelength <= 0.0
        || 2.0 * radius >= period
        || period / wavelength > 2.0
        || angle_deg.abs() > 60.0
        || epsilon_re <= 0.0
        || epsilon_im < 0.0
    {
        return Err(Error::InvalidInput("expected disjoint passive spheres, positive lengths/permittivity, lmax 1–4, helicity 0/1, angle within ±60°, and period/wavelength ≤2".into()));
    }
    let k0 = TAU / wavelength;
    let angle = angle_deg.to_radians();
    let bloch = [k0 * angle.sin(), 0.0];
    let reciprocal = TAU / period;
    let mut orders = Vec::new();
    let mut q = Vec::new();
    let mut pols = Vec::new();
    // period/wavelength <= 2 and |sin(angle)| < 1 guarantee |order| <= 4.
    for mx in -4..=4 {
        for my in -4..=4 {
            let transverse = [
                bloch[0] + f64::from(mx) * reciprocal,
                f64::from(my) * reciprocal,
            ];
            let axial_squared = 1.0 - transverse.iter().map(|v| (v / k0).powi(2)).sum::<f64>();
            if axial_squared.abs() < 1e-10 {
                return Err(Error::InvalidInput("exact grazing diffraction threshold: choose a nearby wavelength or incidence angle".into()));
            }
            if axial_squared > 0.0 {
                orders.push((mx, my, transverse, axial_squared.sqrt()));
                q.extend([transverse; 2]);
                pols.extend([0, 1]);
            }
        }
    }
    let spherical = super::basis(lmax, vec![[0.0; 3]])?;
    let d = spherical.modes.len();
    let coupling = basis::periodic(
        spherical.clone(),
        spherical.clone(),
        [Complex::from(k0); 2],
        true,
        Lattice::new(&[vec![period, 0.0], vec![0.0, period]], &bloch)?,
        Complex::default(),
    )?
    .value;
    let sphere = tmatrix::sphere(
        lmax,
        k0,
        &[radius],
        &[
            Material {
                epsilon: Complex::new(epsilon_re, epsilon_im),
                ..Material::default()
            },
            Material::default(),
        ],
    )?
    .value;
    let channels = channels::spherical(
        spherical,
        [Complex::from(k0); 2],
        q,
        pols,
        period * period,
        true,
    )?
    .value;
    let incident_order = orders
        .iter()
        .position(|&(mx, my, _, _)| mx == 0 && my == 0)
        .ok_or_else(|| {
            Error::InvalidInput("incident diffraction order is not propagating".into())
        })?;
    let incident_column = 2 * incident_order + usize::from(helicity);
    let incident = DMatrix::from_fn(d, 1, |row, _| channels[(row, incident_column)]);
    let scattered = illumination::Factor::new(sphere, coupling)?.solve(&incident)?;
    let mut reflected = 0.0;
    let mut transmitted = 0.0;
    let mut result = vec![0.0; 4];
    for (index, &(mx, my, transverse, axial)) in orders.iter().enumerate() {
        let mut powers = [0.0; 2];
        for (side, power) in powers.iter_mut().enumerate() {
            for pol in 0..2 {
                let column = 2 * index + pol;
                let mut amplitude: Complex = (0..d)
                    .map(|row| channels[((2 + side) * d + row, column)] * scattered[(row, 0)])
                    .sum();
                if side == 0 && column == incident_column {
                    amplitude += 1.0;
                }
                // Orthogonal helicities in vacuum; normalize normal energy flux.
                *power += amplitude.norm_sqr() * axial / angle.cos();
            }
        }
        transmitted += powers[0];
        reflected += powers[1];
        result.extend([
            f64::from(mx),
            f64::from(my),
            transverse[0] / k0,
            transverse[1] / k0,
            axial,
            powers[1],
            powers[0],
        ]);
        result[3] += 1.0;
    }
    result[..3].copy_from_slice(&[reflected, transmitted, 1.0 - reflected - transmitted]);
    if result.iter().any(|x| !x.is_finite()) {
        return Err(Error::SpecialFunction(
            "nonfinite metasurface diffraction power".into(),
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::solve;

    #[test]
    fn lossless_power_helicity_and_scale_invariance() -> treams_core::Result<()> {
        for wavelength in [0.73, 1.03, 1.25, 1.57] {
            let reference = solve(3, 0.23, 0.8, wavelength, 20.0, 12.25, 0.0, 0)?;
            assert!((reference[0] + reference[1] - 1.0).abs() < 2e-10);
            let opposite = solve(3, 0.23, 0.8, wavelength, 20.0, 12.25, 0.0, 1)?;
            for index in 0..3 {
                assert!((reference[index] - opposite[index]).abs() < 2e-10);
            }
            let scaled = solve(3, 0.69, 2.4, wavelength * 3.0, 20.0, 12.25, 0.0, 0)?;
            for (actual, expected) in scaled.iter().zip(&reference) {
                assert!((actual - expected).abs() < 2e-10);
            }
        }
        Ok(())
    }

    #[test]
    fn transparent_cell_and_excluded_grazing_order() -> treams_core::Result<()> {
        let transparent = solve(3, 0.23, 0.8, 0.73, 20.0, 1.0, 0.0, 0)?;
        assert!(transparent[0].abs() < 1e-20);
        assert!((transparent[1] - 1.0).abs() < 1e-13);
        assert!(solve(3, 0.23, 0.8, 0.8, 0.0, 12.25, 0.0, 0).is_err());
        Ok(())
    }
}

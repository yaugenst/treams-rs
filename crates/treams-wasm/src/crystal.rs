//! Normal-incidence one-dimensional dielectric crystal demonstrations.

use nalgebra::DMatrix;
use treams_core::{Complex, Error, Result, plane, smatrix};
use wasm_bindgen::prelude::*;

fn validate(n_a: f64, n_b: f64, fill_a: f64, periods: u32, frequencies: &[f64]) -> Result<()> {
    if ![n_a, n_b].iter().all(|n| (1.0..=4.0).contains(n))
        || !fill_a.is_finite()
        || fill_a <= 0.0
        || fill_a >= 1.0
        || !(1..=64).contains(&periods)
        || frequencies.is_empty()
        || frequencies.len() > 1024
        || frequencies
            .iter()
            .any(|f| !f.is_finite() || *f <= 0.0 || *f > 2.0)
    {
        return Err(Error::InvalidInput(
            "crystal requires indices in [1,4], filling in (0,1), 1..64 periods and 1..1024 frequencies in (0,2]".into(),
        ));
    }
    Ok(())
}

fn interface(k0: f64, below: f64, above: f64) -> Result<smatrix::Blocks> {
    Ok(smatrix::interface(
        [below, above].map(|n| [Complex::from(k0 * n); 2]),
        [below, above].map(|n| Complex::from(1.0 / n)),
        [0.0; 2],
        2,
    )?
    .value)
}

fn propagation(k0: f64, index: f64, distance: f64) -> Result<smatrix::Blocks> {
    Ok(smatrix::propagation(
        vec![
            [
                Complex::default(),
                Complex::default(),
                Complex::from(k0 * index)
            ];
            2
        ],
        [0.0, 0.0, distance],
    )?
    .value)
}

fn transparent() -> smatrix::Blocks {
    [
        DMatrix::identity(2, 2),
        DMatrix::zeros(2, 2),
        DMatrix::zeros(2, 2),
        DMatrix::identity(2, 2),
    ]
}

fn incident() -> DMatrix<Complex> {
    // Equal helicities with this phase give unit Ex and zero Ey.
    DMatrix::from_element(2, 1, Complex::from(-std::f64::consts::FRAC_1_SQRT_2))
}

fn cell(k0: f64, n_a: f64, n_b: f64, fill_a: f64) -> Result<smatrix::Blocks> {
    let mut value = interface(k0, 1.0, n_a)?;
    for next in [
        propagation(k0, n_a, fill_a)?,
        interface(k0, n_a, n_b)?,
        propagation(k0, n_b, 1.0 - fill_a)?,
        interface(k0, n_b, 1.0)?,
    ] {
        value = smatrix::add(value, next)?.value;
    }
    Ok(value)
}

fn repeat(mut cell: smatrix::Blocks, mut periods: u32) -> Result<smatrix::Blocks> {
    let mut value = transparent();
    while periods > 0 {
        if periods & 1 == 1 {
            value = smatrix::add(value, cell.clone())?.value;
        }
        periods >>= 1;
        if periods > 0 {
            cell = smatrix::add(cell.clone(), cell)?.value;
        }
    }
    Ok(value)
}

fn spectrum(
    n_a: f64,
    n_b: f64,
    fill_a: f64,
    periods: u32,
    frequencies: &[f64],
) -> Result<Vec<f64>> {
    validate(n_a, n_b, fill_a, periods, frequencies)?;
    let mut result = Vec::with_capacity(5 * frequencies.len());
    let incoming = incident();
    for &frequency in frequencies {
        let unit = cell(std::f64::consts::TAU * frequency, n_a, n_b, fill_a)?;
        let bands = smatrix::bands(unit.clone(), 1.0)?;
        let k = bands
            .wavenumbers
            .iter()
            .max_by(|a, b| a.im.total_cmp(&b.im))
            .ok_or(Error::Singular)?;
        let stack = repeat(unit, periods)?;
        // Both external ports are vacuum, so squared unit-field amplitudes
        // are already normalized power fluxes.
        let transmission = (&stack[0] * &incoming).norm_squared();
        let reflection = (&stack[2] * &incoming).norm_squared();
        result.extend([
            transmission,
            reflection,
            k.re.abs() / std::f64::consts::PI,
            k.im.abs(),
            k.cos().re,
        ]);
    }
    Ok(result)
}

/// Exact normal-incidence response of a lossless alternating dielectric stack.
///
/// The unit period is `a=1`; A occupies `fill_a`, followed by B. Both external
/// media are vacuum. Indices lie in `[1,4]`, filling in `(0,1)`, and periods in
/// 1..64. Frequencies are `a/lambda` in (0,2], with at most 1024 samples.
/// Each output row is `[T, R, abs(Re(K*a))/pi, abs(Im(K*a)), Re(cos(K*a))]`.
/// T/R describe the finite stack; K describes the infinite periodic crystal.
/// Im(K*a) is amplitude attenuation per period, not an intensity decay rate.
#[wasm_bindgen]
pub fn crystal_spectrum(
    n_a: f64,
    n_b: f64,
    fill_a: f64,
    periods: u32,
    frequencies: &[f64],
) -> std::result::Result<Vec<f64>, JsError> {
    Ok(spectrum(n_a, n_b, fill_a, periods, frequencies)?)
}

fn field(
    n_a: f64,
    n_b: f64,
    fill_a: f64,
    periods: u32,
    frequency: f64,
    positions: &[f64],
) -> Result<Vec<f64>> {
    validate(n_a, n_b, fill_a, periods, &[frequency])?;
    if positions.is_empty() || positions.len() > 4096 || positions.iter().any(|z| !z.is_finite()) {
        return Err(Error::InvalidInput(
            "crystal field requires 1..4096 finite positions z/a".into(),
        ));
    }
    let k0 = std::f64::consts::TAU * frequency;
    let layer_count = 2 * periods as usize;
    let indices: Vec<_> = (0..layer_count)
        .map(|i| if i % 2 == 0 { n_a } else { n_b })
        .collect();
    let widths: Vec<_> = (0..layer_count)
        .map(|i| if i % 2 == 0 { fill_a } else { 1.0 - fill_a })
        .collect();
    let starts: Vec<_> = (0..periods)
        .flat_map(|i| [f64::from(i), f64::from(i) + fill_a])
        .collect();
    let mut prefix = Vec::with_capacity(layer_count);
    let mut current = interface(k0, 1.0, n_a)?;
    for (i, (&index, &width)) in indices.iter().zip(&widths).enumerate() {
        prefix.push(current.clone());
        current = smatrix::add(current, propagation(k0, index, width)?)?.value;
        let next_index = indices.get(i + 1).copied().unwrap_or(1.0);
        current = smatrix::add(current, interface(k0, index, next_index)?)?.value;
    }
    let incoming = incident();
    let transmitted = &current[0] * &incoming;
    let reflected = &current[2] * &incoming;
    let zero = DMatrix::zeros(2, 1);
    let mut amplitudes = Vec::with_capacity(layer_count);
    let mut suffix = transparent();
    for i in (0..layer_count).rev() {
        let next_index = indices.get(i + 1).copied().unwrap_or(1.0);
        suffix = smatrix::add(interface(k0, indices[i], next_index)?, suffix)?.value;
        suffix = smatrix::add(propagation(k0, indices[i], widths[i])?, suffix)?.value;
        let [_, _, up, down] = smatrix::illuminate(
            prefix[i].clone(),
            suffix.clone(),
            [incoming.clone(), zero.clone()],
        )?
        .0;
        amplitudes.push([up, down]);
    }
    amplitudes.reverse();
    let mut result = Vec::with_capacity(2 * positions.len());
    for &position in positions {
        let (index, distance, up, down) = if position < 0.0 {
            (1.0, position, &incoming, &reflected)
        } else if position >= f64::from(periods) {
            (1.0, position - f64::from(periods), &transmitted, &zero)
        } else {
            let i = starts.partition_point(|&z| z <= position) - 1;
            (
                indices[i],
                position - starts[i],
                &amplitudes[i][0],
                &amplitudes[i][1],
            )
        };
        let mut value = Complex::default();
        for (sign, coefficients) in [(1.0, up), (-1.0, down)] {
            let wave = [
                Complex::default(),
                Complex::default(),
                Complex::from(sign * k0 * index),
            ];
            for helicity in 0..2 {
                let polarization = plane::polarization(wave, helicity, true)?;
                let electric = plane::field_value(
                    polarization,
                    wave,
                    [
                        Complex::default(),
                        Complex::default(),
                        Complex::from(distance),
                    ],
                )?;
                value += coefficients[(usize::from(helicity), 0)] * electric[0];
            }
        }
        result.extend([value.re, value.im]);
    }
    Ok(result)
}

/// Complex Ex field of the finite stack under unit x-polarized illumination.
///
/// Parameters follow `crystal_spectrum`; one frequency is used. The stack spans
/// z/a=0..periods, with vacuum outside. Positions need not be ordered; up to 4096
/// finite samples are accepted. Return interleaved `[Ex.re, Ex.im]` per position.
/// Time animation uses `Re(Ex * exp(-i*phase))`; the field includes all coherent
/// internal reflections and the incident field on the illuminated side.
#[wasm_bindgen]
pub fn crystal_field(
    n_a: f64,
    n_b: f64,
    fill_a: f64,
    periods: u32,
    frequency: f64,
    positions: &[f64],
) -> std::result::Result<Vec<f64>, JsError> {
    Ok(field(n_a, n_b, fill_a, periods, frequency, positions)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn lossless_spectra_and_closed_gap() {
        let frequencies: Vec<_> = (1..=39).map(|i| f64::from(i) / 60.0).collect();
        for (a, b) in [(1.0, 1.0), (1.45, 2.6), (4.0, 1.0)] {
            for fill in [0.1, 0.5, 0.9] {
                for periods in [1, 8, 24] {
                    let result = spectrum(a, b, fill, periods, &frequencies).unwrap();
                    for row in result.chunks_exact(5) {
                        assert!((row[0] + row[1] - 1.0).abs() < 2e-11);
                        if a == b {
                            assert!(row[1] < 1e-24);
                            assert!(row[3] < 1e-12);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn layer_field_is_continuous_and_vacuum_is_exact() {
        for (a, b) in [(1.0, 1.0), (1.45, 2.6)] {
            let fill = b / (a + b);
            let mut positions = vec![-0.5, 8.5];
            for cell in 0..8 {
                for boundary in [f64::from(cell), f64::from(cell) + fill] {
                    positions.extend([boundary - 1e-9, boundary + 1e-9]);
                }
            }
            let result = field(a, b, fill, 8, 0.27, &positions).unwrap();
            let complex: Vec<_> = result
                .chunks_exact(2)
                .map(|v| Complex::new(v[0], v[1]))
                .collect();
            for pair in complex[2..].chunks_exact(2) {
                assert!((pair[0] - pair[1]).norm() < 2e-8);
            }
            if a == b {
                for (&z, value) in positions.iter().zip(complex) {
                    assert!(
                        (value - (Complex::i() * std::f64::consts::TAU * 0.27 * z).exp()).norm()
                            < 1e-12
                    );
                }
            }
        }
    }
}

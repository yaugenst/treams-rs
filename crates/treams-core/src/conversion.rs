//! Regular and periodic spherical/cylindrical conversion, including displaced origins.
#![allow(clippy::indexing_slicing)] // Validated bases, mode polarizations and matrix dimensions.

use crate::{
    Complex, Error, Result,
    basis::{Basis, TranslationGradient},
    cylwaves, finite,
    jet::Jet,
    special::{Radial, angular_jets},
    waves::{Mode, Translation},
};
use nalgebra::DMatrix;
use rayon::prelude::*;

fn coefficient<const N: usize>(
    mode: Mode,
    kz: Jet<N>,
    wave: Jet<N>,
    pol: u8,
    helicity: bool,
) -> Result<Jet<N>> {
    let cosine = kz / wave;
    let sine = (1.0 - cosine * cosine).sqrt();
    if sine.value == Complex::default() {
        return Err(Error::InvalidInput(
            "cylindrical conversion cutoff requires a limiting formulation".into(),
        ));
    }
    let [pi, tau] = angular_jets(mode.l, mode.m, cosine, sine);
    let angular = if helicity {
        tau + (2.0 * f64::from(pol) - 1.0) * pi
    } else if mode.pol == pol {
        tau
    } else {
        pi
    };
    let (l, m) = (mode.l, mode.m);
    let normalization = (4.0 * std::f64::consts::PI * f64::from(2 * l + 1)
        / f64::from(l * (l + 1)))
    .sqrt()
        * (0.5 * (libm::lgamma(f64::from(l - m + 1)) - libm::lgamma(f64::from(l + m + 1)))).exp();
    Ok(Complex::i().powi(l - m) * normalization * angular)
}

#[allow(clippy::float_cmp)] // Exact axial origins determine the angular selection rule.
fn entry<const N: usize>(
    to: Mode,
    from: cylwaves::Mode,
    k: Complex,
    r: [f64; 3],
    helicity: bool,
) -> Result<Translation> {
    if helicity && to.pol != from.pol {
        return Ok(Translation::default());
    }
    // On axis only equal azimuthal orders survive; first spatial derivatives
    // additionally couple adjacent orders. Avoid evaluating all zero entries.
    if r[0] == 0.0 && r[1] == 0.0 && (to.m - from.m).unsigned_abs() > u32::from(N != 0) {
        return Ok(Translation::default());
    }
    let coefficient = coefficient(
        to,
        Jet::constant(from.kz),
        Jet::<N>::variable(k, 0),
        from.pol,
        helicity,
    )?;
    let translation = cylwaves::translate(
        cylwaves::Mode { m: to.m, ..from },
        from,
        k,
        r,
        Radial::Regular,
    )?;
    Ok(Translation {
        value: coefficient.value * translation.value,
        position: translation.position.map(|g| coefficient.value * g),
        k: coefficient.derivative.first().copied().unwrap_or_default() * translation.value
            + coefficient.value * translation.k,
    })
}

/// Retained conversion inputs; derivatives are recomputed and contracted in reverse.
#[derive(Debug)]
pub struct ConversionResidual {
    destination: Basis,
    source: cylwaves::Basis,
    ks: [Complex; 2],
    helicity: bool,
    /// Cylindrical input amplitudes mapped to regular spherical amplitudes.
    pub value: DMatrix<Complex>,
}

/// Expand regular cylindrical waves around arbitrary spherical expansion origins.
pub fn cylindrical_to_spherical(
    destination: Basis,
    source: cylwaves::Basis,
    ks: [Complex; 2],
    helicity: bool,
) -> Result<ConversionResidual> {
    destination.validate()?;
    source.validate()?;
    if ks.iter().any(|&k| !finite(k) || k == Complex::default()) || (!helicity && ks[0] != ks[1]) {
        return Err(Error::InvalidInput(
            "require nonzero finite medium wavenumbers; parity requires an achiral medium".into(),
        ));
    }
    let mut value = DMatrix::zeros(destination.modes.len(), source.modes.len());
    value
        .as_mut_slice()
        .par_chunks_mut(destination.modes.len())
        .enumerate()
        .try_for_each(|(j, column)| -> Result<()> {
            let (q, from) = source.modes[j];
            for (i, &(p, to)) in destination.modes.iter().enumerate() {
                let r =
                    std::array::from_fn(|a| destination.positions[p][a] - source.positions[q][a]);
                column[i] = entry::<0>(to, from, ks[usize::from(from.pol)], r, helicity)?.value;
            }
            Ok(())
        })?;
    Ok(ConversionResidual {
        destination,
        source,
        ks,
        helicity,
        value,
    })
}
impl ConversionResidual {
    /// Contract origin and complex-wavenumber derivatives; axial labels remain fixed.
    pub fn pullback(self, g: &DMatrix<Complex>) -> Result<TranslationGradient> {
        if g.shape() != self.value.shape() || g.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput("invalid conversion cotangent".into()));
        }
        let zero = || TranslationGradient {
            destination: vec![[0.0; 3]; self.destination.positions.len()],
            source: vec![[0.0; 3]; self.source.positions.len()],
            ks: [Complex::default(); 2],
        };
        self.source
            .modes
            .par_iter()
            .enumerate()
            .try_fold(zero, |mut result, (j, &(q, from))| -> Result<_> {
                for (i, &(p, to)) in self.destination.modes.iter().enumerate() {
                    let cot = g[(i, j)];
                    if cot == Complex::default() {
                        continue;
                    }
                    let r = std::array::from_fn(|a| {
                        self.destination.positions[p][a] - self.source.positions[q][a]
                    });
                    let wave =
                        entry::<1>(to, from, self.ks[usize::from(from.pol)], r, self.helicity)?;
                    result.ks[usize::from(from.pol)] += cot * wave.k.conj();
                    for axis in 0..3 {
                        let gradient = (cot.conj() * wave.position[axis]).re;
                        result.destination[p][axis] += gradient;
                        result.source[q][axis] -= gradient;
                    }
                }
                Ok(result)
            })
            .try_reduce(zero, |mut a, b| {
                for (a, b) in a
                    .destination
                    .iter_mut()
                    .flatten()
                    .chain(a.source.iter_mut().flatten())
                    .zip(
                        b.destination
                            .iter()
                            .flatten()
                            .chain(b.source.iter().flatten()),
                    )
                {
                    *a += b;
                }
                for (a, b) in a.ks.iter_mut().zip(b.ks) {
                    *a += b;
                }
                Ok(a)
            })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PeriodicEntry {
    wave: Translation,
    kz: Complex,
}
#[allow(clippy::float_cmp)] // Exact axial offsets select vanishing angular orders.
fn periodic_entry<const N: usize>(
    to: cylwaves::Mode,
    from: Mode,
    k: Complex,
    r: [f64; 3],
    period: f64,
    helicity: bool,
) -> Result<PeriodicEntry> {
    if (helicity && to.pol != from.pol)
        || (r[0] == 0.0 && r[1] == 0.0 && (to.m - from.m).unsigned_abs() > u32::from(N != 0))
    {
        return Ok(PeriodicEntry::default());
    }
    let wave = Jet::<N>::variable(k, 0);
    let kz = Jet::variable(to.kz, 1);
    // The outgoing periodic coefficient differs from the regular reciprocal
    // conversion by (-1)^(l-m)/(4*period*k).
    let sign = if (from.l - from.m) % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    let beta = coefficient(from, kz, wave, to.pol, helicity)? * (0.25 * sign / period) / wave;
    let translated = cylwaves::translate(
        to,
        cylwaves::Mode { m: from.m, ..to },
        k,
        r,
        Radial::Regular,
    )?;
    Ok(PeriodicEntry {
        wave: Translation {
            value: beta.value * translated.value,
            position: translated.position.map(|g| beta.value * g),
            k: beta.derivative.first().copied().unwrap_or_default() * translated.value
                + beta.value * translated.k,
        },
        kz: beta.derivative.get(1).copied().unwrap_or_default() * translated.value
            + beta.value * translated.kz,
    })
}

/// Retained geometry for a spherical z-periodic array radiating into cylindrical modes.
#[derive(Debug)]
pub struct PeriodicConversionResidual {
    destination: cylwaves::Basis,
    source: Basis,
    ks: [Complex; 2],
    period: f64,
    helicity: bool,
}
/// Origin, medium, per-output-mode axial wavenumber, and period cotangents.
#[derive(Debug)]
pub struct PeriodicConversionGradient {
    /// Both origin arrays and medium wavenumbers.
    pub expansion: TranslationGradient,
    /// Real axial-wavenumber cotangents, one per cylindrical output mode.
    pub kz: Vec<f64>,
    /// Real period cotangent, holding axial wavenumbers independent.
    pub period: f64,
}

/// Outgoing cylindrical expansion of a z-periodic spherical source array.
/// Axial Fourier wavenumbers are independent inputs; compose kz=kpar+2*pi*n/period for diffraction orders.
pub fn periodic_spherical_to_cylindrical(
    destination: cylwaves::Basis,
    source: Basis,
    ks: [Complex; 2],
    period: f64,
    helicity: bool,
) -> Result<(DMatrix<Complex>, PeriodicConversionResidual)> {
    destination.validate()?;
    source.validate()?;
    if !period.is_finite()
        || period <= 0.0
        || ks.iter().any(|&k| !finite(k) || k == Complex::default())
        || (!helicity && ks[0] != ks[1])
    {
        return Err(Error::InvalidInput("require a positive finite period and nonzero finite medium wavenumbers; parity requires an achiral medium".into()));
    }
    let mut value = DMatrix::zeros(destination.modes.len(), source.modes.len());
    value
        .as_mut_slice()
        .par_chunks_mut(destination.modes.len())
        .enumerate()
        .try_for_each(|(j, column)| -> Result<()> {
            let (q, from) = source.modes[j];
            for (out, &(p, to)) in column.iter_mut().zip(&destination.modes) {
                let r =
                    std::array::from_fn(|a| destination.positions[p][a] - source.positions[q][a]);
                *out =
                    periodic_entry::<0>(to, from, ks[usize::from(from.pol)], r, period, helicity)?
                        .wave
                        .value;
            }
            Ok(())
        })?;
    Ok((
        value,
        PeriodicConversionResidual {
            destination,
            source,
            ks,
            period,
            helicity,
        },
    ))
}
impl PeriodicConversionResidual {
    /// Cylindrical output and spherical input mode counts.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.destination.modes.len(), self.source.modes.len())
    }
    /// Recompute and contract local derivatives; no output Jacobian is retained.
    pub fn pullback(self, g: &DMatrix<Complex>) -> Result<PeriodicConversionGradient> {
        if g.shape() != self.shape() || g.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput(
                "invalid periodic conversion cotangent".into(),
            ));
        }
        let zero = || PeriodicConversionGradient {
            expansion: TranslationGradient {
                destination: vec![[0.0; 3]; self.destination.positions.len()],
                source: vec![[0.0; 3]; self.source.positions.len()],
                ks: [Complex::default(); 2],
            },
            kz: vec![0.0; self.destination.modes.len()],
            period: 0.0,
        };
        self.source
            .modes
            .par_iter()
            .enumerate()
            .try_fold(zero, |mut result, (j, &(q, from))| -> Result<_> {
                for (i, &(p, to)) in self.destination.modes.iter().enumerate() {
                    let cot = g[(i, j)];
                    if cot == Complex::default() {
                        continue;
                    }
                    let r = std::array::from_fn(|a| {
                        self.destination.positions[p][a] - self.source.positions[q][a]
                    });
                    let entry = periodic_entry::<2>(
                        to,
                        from,
                        self.ks[usize::from(from.pol)],
                        r,
                        self.period,
                        self.helicity,
                    )?;
                    result.expansion.ks[usize::from(from.pol)] += cot * entry.wave.k.conj();
                    result.kz[i] += (cot.conj() * entry.kz).re;
                    result.period -= (cot.conj() * entry.wave.value).re / self.period;
                    for axis in 0..3 {
                        let gradient = (cot.conj() * entry.wave.position[axis]).re;
                        result.expansion.destination[p][axis] += gradient;
                        result.expansion.source[q][axis] -= gradient;
                    }
                }
                Ok(result)
            })
            .try_reduce(zero, |mut a, b| {
                for (a, b) in a
                    .expansion
                    .destination
                    .iter_mut()
                    .flatten()
                    .chain(a.expansion.source.iter_mut().flatten())
                    .chain(&mut a.kz)
                    .zip(
                        b.expansion
                            .destination
                            .iter()
                            .flatten()
                            .chain(b.expansion.source.iter().flatten())
                            .chain(&b.kz),
                    )
                {
                    *a += b;
                }
                for (a, b) in a.expansion.ks.iter_mut().zip(b.expansion.ks) {
                    *a += b;
                }
                a.period += b.period;
                Ok(a)
            })
    }
}

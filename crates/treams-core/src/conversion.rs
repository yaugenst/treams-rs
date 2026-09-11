//! Regular cylindrical-to-spherical conversion, including displaced origins.
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
    let wave = Jet::<N>::variable(k, 0);
    let cosine = from.kz / wave;
    let sine = (1.0 - cosine * cosine).sqrt();
    if sine.value == Complex::default() {
        return Err(Error::InvalidInput(
            "cylindrical conversion cutoff requires a limiting formulation".into(),
        ));
    }
    // On axis only equal azimuthal orders survive; first spatial derivatives
    // additionally couple adjacent orders. Avoid evaluating all zero entries.
    if r[0] == 0.0 && r[1] == 0.0 && (to.m - from.m).unsigned_abs() > u32::from(N != 0) {
        return Ok(Translation::default());
    }
    let [pi, tau] = angular_jets(to.l, to.m, cosine, sine);
    let angular = if helicity {
        tau + (2.0 * f64::from(from.pol) - 1.0) * pi
    } else if to.pol == from.pol {
        tau
    } else {
        pi
    };
    let (l, m) = (to.l, to.m);
    let normalization = (4.0 * std::f64::consts::PI * f64::from(2 * l + 1)
        / f64::from(l * (l + 1)))
    .sqrt()
        * (0.5 * (libm::lgamma(f64::from(l - m + 1)) - libm::lgamma(f64::from(l + m + 1)))).exp();
    let coefficient = Complex::i().powi(l - m) * normalization * angular;
    let translation =
        cylwaves::translate(cylwaves::Mode { m, ..from }, from, k, r, Radial::Regular)?;
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

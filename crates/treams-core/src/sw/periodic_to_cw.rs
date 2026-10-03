//! Outgoing cylindrical expansion of z-periodic spherical arrays, including displaced
//! positions.
//!
//! Upstream: `treams.sw.periodic_to_cw` and the `treams.ExpandLattice` operator from a
//! spherical to a cylindrical basis.
#![allow(clippy::indexing_slicing)] // Validated bases, mode polarizations and matrix dimensions.

use super::{Basis, Mode};
use crate::{
    Complex, Error, Result,
    basis::{ExpansionGradient, ModeLabel, add_pair_gradient, validate_wavenumbers},
    cw::{self, coefficient},
    numerics::{Jet, finite, parallel::try_fold_ordered},
    special::Radial,
};
use nalgebra::DMatrix;
use rayon::prelude::*;

/// One outgoing periodic coefficient with its displacement, medium and axial
/// wavenumber derivatives.
#[allow(clippy::float_cmp)] // Exact axial offsets select vanishing angular orders.
fn periodic_entry<const N: usize>(
    to: cw::Mode,
    from: Mode,
    k: Complex,
    r: [f64; 3],
    period: f64,
    helicity: bool,
) -> Result<cw::CartesianTranslation> {
    if (helicity && to.pol != from.pol)
        || (r[0] == 0.0 && r[1] == 0.0 && (to.m - from.m).unsigned_abs() > u32::from(N != 0))
    {
        return Ok(cw::CartesianTranslation::default());
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
    // Forward at coincident positions needs no cylindrical translation or its jets.
    if N == 0 && r == [0.0; 3] {
        return Ok(cw::CartesianTranslation {
            value: beta.value,
            ..cw::CartesianTranslation::default()
        });
    }
    let translated =
        cw::cartesian_translation(to, cw::Mode { m: from.m, ..to }, k, r, Radial::Regular)?;
    Ok(cw::CartesianTranslation {
        value: beta.value * translated.value,
        position: translated.position.map(|g| beta.value * g),
        k: beta.derivative.first().copied().unwrap_or_default() * translated.value
            + beta.value * translated.k,
        kz: beta.derivative.get(1).copied().unwrap_or_default() * translated.value
            + beta.value * translated.kz,
    })
}

/// Direct outgoing spherical-to-cylindrical coefficient of a z-periodic array.
///
/// A negative `period` gives the coefficient of its absolute value, while
/// [`periodic_to_cw_matrix`] rejects `period <= 0`.
///
/// Upstream: `treams.sw.periodic_to_cw`.
pub fn periodic_to_cw(
    to: cw::Mode,
    from: Mode,
    k: Complex,
    period: f64,
    helicity: bool,
) -> Result<Complex> {
    to.validate()?;
    from.validate()?;
    if !finite(k) || k == Complex::default() || !period.is_finite() || period == 0.0 {
        return Err(Error::InvalidInput(
            "finite nonzero wavenumber and period required".into(),
        ));
    }
    Ok(periodic_entry::<0>(to, from, k, [0.0; 3], period.abs(), helicity)?.value)
}

/// What [`periodic_to_cw_matrix`] saves for its pullback: the bases, the wavenumbers,
/// the period and the polarization convention.
///
/// The pullback adds the per-column gradients in chunks of consecutive columns fixed by
/// the column count, and the chunk sums in chunk order
/// (`numerics::parallel::try_fold_ordered`), so the thread count does not change them.
#[derive(Debug)]
pub struct PeriodicToCwResidual {
    destination: cw::Basis,
    source: Basis,
    ks: [Complex; 2],
    period: f64,
    helicity: bool,
}

/// Position, medium, per-output-mode axial wavenumber, and period cotangents.
#[derive(Debug)]
pub struct PeriodicToCwGradient {
    /// Cotangents of both position arrays and the medium wavenumbers.
    pub expansion: ExpansionGradient,
    /// Real axial-wavenumber cotangents, one per cylindrical output mode.
    pub kz: Vec<f64>,
    /// Real period cotangent, holding axial wavenumbers independent.
    pub period: f64,
}

/// Outgoing cylindrical expansion of a z-periodic spherical source array.
/// Axial Fourier wavenumbers are independent inputs; compose kz=kpar+2*pi*n/period for diffraction orders.
///
/// Upstream: `treams.operators.expandlattice` from a spherical to a cylindrical basis.
/// Differences: treams pairs each axis only with the sphere chain of the same position index
/// and ignores the positions; this matrix couples every axis with every chain through the
/// translation between them.
pub fn periodic_to_cw_matrix(
    destination: cw::Basis,
    source: Basis,
    ks: [Complex; 2],
    period: f64,
    helicity: bool,
) -> Result<(DMatrix<Complex>, PeriodicToCwResidual)> {
    destination.validate()?;
    source.validate()?;
    if !period.is_finite() || period <= 0.0 {
        return Err(Error::InvalidInput(
            "period must be positive and finite".into(),
        ));
    }
    validate_wavenumbers(ks, helicity, false)?;
    let mut value = DMatrix::zeros(destination.modes.len(), source.modes.len());
    crate::threads::install(|| {
        value
            .as_mut_slice()
            .par_chunks_mut(destination.modes.len())
            .enumerate()
            .try_for_each(|(j, column)| -> Result<()> {
                let (q, from) = source.modes[j];
                for (out, &(p, to)) in column.iter_mut().zip(&destination.modes) {
                    let r = std::array::from_fn(|a| {
                        destination.positions[p][a] - source.positions[q][a]
                    });
                    *out = periodic_entry::<0>(
                        to,
                        from,
                        ks[usize::from(from.pol)],
                        r,
                        period,
                        helicity,
                    )?
                    .value;
                }
                Ok(())
            })
    })?;
    Ok((
        value,
        PeriodicToCwResidual {
            destination,
            source,
            ks,
            period,
            helicity,
        },
    ))
}

impl PeriodicToCwResidual {
    /// Cylindrical output and spherical input mode counts.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.destination.modes.len(), self.source.modes.len())
    }

    /// Recompute the local derivatives of every entry instead of keeping a Jacobian.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<PeriodicToCwGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput(
                "invalid periodic conversion cotangent".into(),
            ));
        }
        try_fold_ordered(
            self.source.modes.iter().collect(),
            true,
            || PeriodicToCwGradient {
                expansion: ExpansionGradient::zeros(
                    self.destination.positions.len(),
                    self.source.positions.len(),
                ),
                kz: vec![0.0; self.destination.modes.len()],
                period: 0.0,
            },
            |mut result, j, &(q, from)| -> Result<_> {
                for (i, &(p, to)) in self.destination.modes.iter().enumerate() {
                    let cot = cotangent[(i, j)];
                    if cot == Complex::default() {
                        continue;
                    }
                    let r = std::array::from_fn(|a| {
                        self.destination.positions[p][a] - self.source.positions[q][a]
                    });
                    let pol = usize::from(from.pol);
                    let entry =
                        periodic_entry::<2>(to, from, self.ks[pol], r, self.period, self.helicity)?;
                    add_pair_gradient(
                        &mut result.expansion,
                        [p, q, pol],
                        cot,
                        entry.position,
                        entry.k,
                    );
                    result.kz[i] += (cot.conj() * entry.kz).re;
                    result.period -= (cot.conj() * entry.value).re / self.period;
                }
                Ok(result)
            },
            |mut total, partial| {
                total.expansion.accumulate(&partial.expansion);
                for (a, b) in total.kz.iter_mut().zip(&partial.kz) {
                    *a += b;
                }
                total.period += partial.period;
                total
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::periodic_to_cw_matrix;
    use crate::{
        Complex, cw, sw,
        test_support::{assert_same_bits_on_pools, bits, patterned},
    };

    /// The per-column gradients add in chunks fixed by the column count: with more
    /// columns than chunks, every gradient repeats bit for bit on every pool size.
    #[test]
    fn pullback_does_not_depend_on_the_thread_count() {
        let modes = sw::modes(3).unwrap();
        let source = sw::Basis {
            modes: (0..3)
                .flat_map(|p| modes.iter().map(move |&mode| (p, mode)))
                .collect(),
            positions: vec![[0.1, -0.2, 0.3], [-0.3, 0.2, -0.1], [0.2, 0.25, 0.0]],
        };
        let destination = cw::Basis {
            modes: (0..2)
                .flat_map(|p| {
                    [0.2, -0.35].into_iter().flat_map(move |kz| {
                        (-3..=3).flat_map(move |m| [1, 0].map(|pol| (p, cw::Mode { kz, m, pol })))
                    })
                })
                .collect(),
            positions: vec![[1.1, 0.4, 0.0], [-0.9, -0.7, 0.3]],
        };
        let ks = [Complex::new(1.2, 0.05); 2];
        let g = patterned(destination.modes.len(), source.modes.len(), 0.2);
        assert_same_bits_on_pools(|| {
            let (_, residual) =
                periodic_to_cw_matrix(destination.clone(), source.clone(), ks, 1.3, true).unwrap();
            let g = residual.pullback(&g).unwrap();
            let expansion = &g.expansion;
            bits(&[
                &expansion.destination,
                &expansion.source,
                &expansion.ks,
                &g.kz,
                &g.period,
            ])
        });
    }
}

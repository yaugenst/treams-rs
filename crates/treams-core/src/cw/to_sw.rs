//! Regular cylindrical-to-spherical conversion, including displaced positions.
//!
//! Upstream: `treams.cw.to_sw` and the `treams.Expand` operator from a cylindrical to a
//! spherical basis.
#![allow(clippy::indexing_slicing)] // Validated bases, mode polarizations and matrix dimensions.

use crate::{
    Complex, Error, Result,
    basis::{ExpansionGradient, ModeLabel, add_pair_gradient, validate_wavenumbers},
    cw,
    numerics::{Jet, finite},
    special::{Radial, polarized_angular},
    sw::{Basis, CartesianTranslation, Mode},
};
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::collections::HashMap;

/// The coefficient of the spherical mode `mode` in a regular cylindrical wave of
/// polarization `pol` and axial wavenumber `kz` at one position, as a jet in `kz` and the
/// wavenumber `wave`: `i^(l - m) sqrt(4 pi (2l + 1) / (l (l + 1))) sqrt((l - m)! / (l + m)!)`
/// times the polarized angular function at `cos theta = kz / k`.
///
/// Upstream: the coefficient of `treams.cw.to_sw` for `m` equal on both sides.
pub(crate) fn coefficient<const N: usize>(
    mode: Mode,
    kz: Jet<N>,
    wave: Jet<N>,
    pol: u8,
    helicity: bool,
) -> Result<Jet<N>> {
    let cosine = kz / wave;
    let sine = (1.0 - cosine * cosine).sqrt();
    if N != 0 && sine.value == Complex::default() {
        return Err(Error::InvalidInput(
            "cylindrical conversion cutoff requires a limiting formulation".into(),
        ));
    }
    let angular = polarized_angular(mode.l, mode.m, [cosine, sine], [mode.pol, pol], helicity);
    let (l, m) = (mode.l, mode.m);
    let normalization =
        (4.0 * std::f64::consts::PI * f64::from(2 * l + 1) / f64::from(l * (l + 1))).sqrt()
            * crate::special::factorial_ratio_sqrt(l, m);
    Ok(Complex::i().powi(l - m) * normalization * angular)
}

/// The entries of one source column. Angular coefficients depend only on the
/// destination mode label and translations only on the destination position and order,
/// so each is evaluated once per column and shared by the modes of every destination
/// position, degree and polarization.
struct Column<'a, const N: usize> {
    from: cw::Mode,
    k: Complex,
    helicity: bool,
    destination: &'a [[f64; 3]],
    source: [f64; 3],
    coefficients: HashMap<Mode, Jet<N>>,
    translations: HashMap<(usize, i32), cw::CartesianTranslation>,
}

impl<'a, const N: usize> Column<'a, N> {
    fn new(
        destination: &'a [[f64; 3]],
        source: [f64; 3],
        from: cw::Mode,
        k: Complex,
        helicity: bool,
    ) -> Self {
        Self {
            from,
            k,
            helicity,
            destination,
            source,
            coefficients: HashMap::new(),
            translations: HashMap::new(),
        }
    }

    /// The entry of destination mode `to` at position `p`, with its displacement and
    /// wavenumber derivatives.
    #[allow(clippy::float_cmp)] // Exact on-axis displacements determine the angular selection rule.
    fn entry(&mut self, p: usize, to: Mode) -> Result<CartesianTranslation> {
        let from = self.from;
        if self.helicity && to.pol != from.pol {
            return Ok(CartesianTranslation::default());
        }
        let r: [f64; 3] = std::array::from_fn(|a| self.destination[p][a] - self.source[a]);
        // On axis only equal azimuthal orders survive; first spatial derivatives
        // additionally couple adjacent orders. Avoid evaluating all zero entries.
        if r[0] == 0.0 && r[1] == 0.0 && (to.m - from.m).unsigned_abs() > u32::from(N != 0) {
            return Ok(CartesianTranslation::default());
        }
        let coefficient = if let Some(&coefficient) = self.coefficients.get(&to) {
            coefficient
        } else {
            let wave = Jet::<N>::variable(self.k, 0);
            let coefficient =
                coefficient(to, Jet::constant(from.kz), wave, from.pol, self.helicity)?;
            self.coefficients.insert(to, coefficient);
            coefficient
        };
        let translation = if let Some(&translation) = self.translations.get(&(p, to.m)) {
            translation
        } else {
            let to = cw::Mode { m: to.m, ..from };
            let translation = cw::cartesian_translation(to, from, self.k, r, Radial::Regular)?;
            self.translations.insert((p, to.m), translation);
            translation
        };
        Ok(CartesianTranslation {
            value: coefficient.value * translation.value,
            position: translation.position.map(|g| coefficient.value * g),
            k: coefficient.derivative.first().copied().unwrap_or_default() * translation.value
                + coefficient.value * translation.k,
        })
    }
}

/// Cylindrical-to-spherical coefficient of a sphere and an axis at one position, with
/// fixed mode labels.
///
/// Upstream: `treams.cw.to_sw`.
pub fn to_sw(to: Mode, from: cw::Mode, k: Complex, helicity: bool) -> Result<Complex> {
    to.validate()?;
    from.validate()?;
    if !finite(k) || k == Complex::default() {
        return Err(Error::InvalidInput(
            "finite nonzero wavenumber required".into(),
        ));
    }
    if to.m != from.m || (helicity && to.pol != from.pol) {
        return Ok(Complex::default());
    }
    Ok(coefficient(
        to,
        Jet::<0>::constant(from.kz),
        Jet::constant(k),
        from.pol,
        helicity,
    )?
    .value)
}

/// What [`to_sw_matrix`] saves for its pullback: the bases, the wavenumbers and the
/// polarization convention. The pullback recomputes the derivatives of every entry.
///
/// The pullback adds the per-column gradients with Rayon's `try_fold` and `try_reduce`,
/// so their last bits can change with the thread count and from run to run.
#[derive(Debug)]
pub struct ToSwResidual {
    destination: Basis,
    source: cw::Basis,
    ks: [Complex; 2],
    helicity: bool,
}

/// Expand regular cylindrical waves around arbitrary spherical expansion positions: the
/// matrix maps cylindrical input amplitudes to regular spherical amplitudes.
///
/// Upstream: `treams.operators.expand` from a cylindrical to a spherical basis. Differences:
/// treams pairs each sphere only with the axis of the same position index and ignores the
/// positions; this matrix couples every sphere with every axis through the translation
/// between them.
pub fn to_sw_matrix(
    destination: Basis,
    source: cw::Basis,
    ks: [Complex; 2],
    helicity: bool,
) -> Result<(DMatrix<Complex>, ToSwResidual)> {
    destination.validate()?;
    source.validate()?;
    validate_wavenumbers(ks, helicity, false)?;
    let mut value = DMatrix::zeros(destination.modes.len(), source.modes.len());
    value
        .as_mut_slice()
        .par_chunks_mut(destination.modes.len())
        .enumerate()
        .try_for_each(|(j, column)| -> Result<()> {
            let (q, from) = source.modes[j];
            let k = ks[usize::from(from.pol)];
            let mut entries = Column::<0>::new(
                &destination.positions,
                source.positions[q],
                from,
                k,
                helicity,
            );
            for (value, &(p, to)) in column.iter_mut().zip(&destination.modes) {
                *value = entries.entry(p, to)?.value;
            }
            Ok(())
        })?;
    Ok((
        value,
        ToSwResidual {
            destination,
            source,
            ks,
            helicity,
        },
    ))
}

impl ToSwResidual {
    /// Spherical output and cylindrical input mode counts.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.destination.modes.len(), self.source.modes.len())
    }

    /// Position and complex wavenumber gradients. Axial mode labels stay fixed.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<ExpansionGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput("invalid conversion cotangent".into()));
        }
        let zero = || {
            ExpansionGradient::zeros(
                self.destination.positions.len(),
                self.source.positions.len(),
            )
        };
        self.source
            .modes
            .par_iter()
            .enumerate()
            .try_fold(zero, |mut result, (j, &(q, from))| -> Result<_> {
                let pol = usize::from(from.pol);
                let (destination, source) = (&self.destination.positions, self.source.positions[q]);
                let mut entries =
                    Column::<1>::new(destination, source, from, self.ks[pol], self.helicity);
                for (i, &(p, to)) in self.destination.modes.iter().enumerate() {
                    let cot = cotangent[(i, j)];
                    if cot == Complex::default() {
                        continue;
                    }
                    let wave = entries.entry(p, to)?;
                    add_pair_gradient(&mut result, [p, q, pol], cot, wave.position, wave.k);
                }
                Ok(result)
            })
            .try_reduce(zero, |mut a, b| {
                a.accumulate(&b);
                Ok(a)
            })
    }
}

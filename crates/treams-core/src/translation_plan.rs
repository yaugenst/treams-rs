//! Reuse angular couplings and special functions across a complete mode block.
#![allow(clippy::indexing_slicing)] // Internally constructed dense indices.

use crate::{
    Complex, Error, Result,
    special::{Radial, spherical},
    waves::{Mode, Translation, harmonic, terms},
};

#[derive(Clone, Debug)]
struct Term {
    index: usize,
    weight: Complex,
}

/// Geometry-independent spherical mode couplings, shared across particle pairs.
#[derive(Clone, Debug)]
pub(crate) struct TranslationPlan {
    entries: Vec<Vec<Term>>,
    order: i32,
}

impl TranslationPlan {
    pub(crate) fn new(modes: &[Mode]) -> Result<Self> {
        Self::between(modes, modes, true)
    }

    pub(crate) fn between(destination: &[Mode], source: &[Mode], helicity: bool) -> Result<Self> {
        for mode in destination.iter().chain(source) {
            mode.validate()?;
        }
        let order = destination.iter().map(|m| m.l).max().unwrap_or(0)
            + source.iter().map(|m| m.l).max().unwrap_or(0);
        let mut entries = Vec::with_capacity(destination.len() * source.len());
        for &from in source {
            for &to in destination {
                entries.push(
                    terms(to, from, helicity)
                        .into_iter()
                        .map(|(p, m, weight)| {
                            Ok(Term {
                                index: usize::try_from(p * p + p + m).map_err(|_| {
                                    Error::InvalidInput("invalid harmonic index".into())
                                })?,
                                weight,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                );
            }
        }
        Ok(Self { entries, order })
    }

    fn table(&self, k: Complex, position: [f64; 3], kind: Radial) -> Result<Vec<Translation>> {
        let r = position.iter().map(|x| x * x).sum::<f64>().sqrt();
        if r == 0.0 && kind == Radial::Outgoing {
            return Ok(vec![
                Translation::default();
                usize::try_from((self.order + 1).pow(2)).map_err(
                    |_| Error::InvalidInput("invalid order".into())
                )?
            ]);
        }
        let mut table = Vec::new();
        for p in 0..=self.order {
            let radial = spherical(
                u32::try_from(p).map_err(|_| Error::InvalidInput("invalid radial order".into()))?,
                k * r,
                kind,
            )?;
            for m in -p..=p {
                table.push(harmonic(p, m, k, position, radial));
            }
        }
        Ok(table)
    }

    pub(crate) fn evaluate(&self, k: Complex, position: [f64; 3]) -> Result<Vec<Complex>> {
        self.evaluate_with(k, position, Radial::Outgoing)
    }

    pub(crate) fn evaluate_with(
        &self,
        k: Complex,
        position: [f64; 3],
        radial: Radial,
    ) -> Result<Vec<Complex>> {
        let table = self.table(k, position, radial)?;
        Ok(self
            .entries
            .iter()
            .map(|terms| terms.iter().map(|t| t.weight * table[t.index].value).sum())
            .collect())
    }

    pub(crate) fn pullback(
        &self,
        k: Complex,
        position: [f64; 3],
        cotangent: &[Complex],
    ) -> Result<([f64; 3], f64)> {
        let (position, k) = self.pullback_with(k, position, Radial::Outgoing, cotangent)?;
        Ok((position, k.re))
    }

    pub(crate) fn pullback_with(
        &self,
        k: Complex,
        position: [f64; 3],
        radial: Radial,
        cotangent: &[Complex],
    ) -> Result<([f64; 3], Complex)> {
        let table = self.table(k, position, radial)?;
        let mut accumulated = vec![Complex::default(); table.len()];
        for (terms, g) in self.entries.iter().zip(cotangent) {
            for term in terms {
                accumulated[term.index] += g.conj() * term.weight;
            }
        }
        let mut gradient = [0.0; 3];
        let mut gk = Complex::default();
        for (g, item) in accumulated.iter().zip(table) {
            gk += (g * item.k).conj();
            for (value, derivative) in gradient.iter_mut().zip(item.position) {
                *value += (g * derivative).re;
            }
        }
        Ok((gradient, gk))
    }
}

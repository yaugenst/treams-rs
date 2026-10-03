//! Lattice sums over arrays of inputs, with their gradients.
//!
//! treams-rs extension: the `lsum*` ufuncs of treams broadcast their inputs but have no
//! gradients.

use super::{SumGradient, SumPart, cell::BlochLattice, derivatives_part, sum_part, wave::Family};
use crate::{
    Complex, Error, Result,
    numerics::{broadcast, finite, parallel::Parallel},
};

/// Batched sums split adaptively across the rayon pool from this many outputs.
const PARALLEL: Parallel = Parallel::AtLeast(8);

/// [`sum_part`] over arrays, with the residual for the pullback.
///
/// Each input holds one value for all outputs or one per output. The residual keeps
/// the inputs only; the pullback recomputes the local derivatives.
///
/// treams-rs extension: the `lsum*` ufuncs of `treams.lattice` broadcast their inputs
/// but have no gradients.
pub fn sum_array(
    waves: Vec<Family>,
    wavenumbers: Vec<Complex>,
    lattices: Vec<BlochLattice>,
    shifts: Vec<[f64; 3]>,
    etas: Vec<Complex>,
    parts: Vec<SumPart>,
) -> Result<(Vec<Complex>, SumResidual)> {
    let size = broadcast::size(
        &[
            waves.len(),
            wavenumbers.len(),
            lattices.len(),
            shifts.len(),
            etas.len(),
            parts.len(),
        ],
        "arrays must have equal lengths or scalar inputs",
    )?;
    if parts
        .iter()
        .any(|p| matches!(p, SumPart::Real | SumPart::Reciprocal))
        && etas.contains(&Complex::default())
    {
        return Err(Error::InvalidInput(
            "component adjoints require an explicit nonzero Ewald split".into(),
        ));
    }
    let residual = SumResidual {
        waves,
        wavenumbers,
        lattices,
        shifts,
        etas,
        parts,
        size,
    };
    let values = broadcast::map(size, PARALLEL, |i| {
        let (wave, k, lattice, r, eta, part) = residual.element(i);
        sum_part(wave, k, lattice, r, eta, part)
    })?;
    Ok((values, residual))
}

/// What [`sum_array`] saves for the pullback of a batch of lattice sums: the inputs,
/// each one value for all outputs or one per output.
///
/// The pullback computes the gradient of every output on its own, so its result does
/// not depend on the thread count.
#[derive(Debug)]
pub struct SumResidual {
    waves: Vec<Family>,
    wavenumbers: Vec<Complex>,
    lattices: Vec<BlochLattice>,
    shifts: Vec<[f64; 3]>,
    etas: Vec<Complex>,
    parts: Vec<SumPart>,
    size: usize,
}

impl SumResidual {
    /// The inputs of output `i`.
    fn element(&self, i: usize) -> (Family, Complex, &BlochLattice, [f64; 3], Complex, SumPart) {
        (
            broadcast::element(&self.waves, i),
            broadcast::element(&self.wavenumbers, i),
            &self.lattices[if self.lattices.len() == 1 { 0 } else { i }],
            broadcast::element(&self.shifts, i),
            broadcast::element(&self.etas, i),
            broadcast::element(&self.parts, i),
        )
    }

    /// One gradient per output, from `cotangent`, the gradient of a real loss with
    /// respect to the sums.
    pub fn pullback(self, cotangent: &[Complex]) -> Result<Vec<SumGradient>> {
        if cotangent.len() != self.size || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "cotangent must be finite and match lattice output".into(),
            ));
        }
        broadcast::map(self.size, PARALLEL, |i| {
            let g = cotangent[i];
            if g == Complex::default() {
                return Ok(SumGradient::default());
            }
            let (wave, k, lattice, r, eta, part) = self.element(i);
            Ok(derivatives_part(wave, k, lattice, r, eta, part)?.pullback(g))
        })
    }
}

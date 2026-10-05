//! Lattice sums over arrays of inputs, with their gradients.
//!
//! treams-rs extension: the `lsum*` ufuncs of treams broadcast their inputs but have no
//! gradients.

use super::{
    SumGradient, SumPart, SumTangent, cell::BlochLattice, derivatives_part, sum_part, wave::Family,
};
use crate::{
    Complex, Error, Result,
    numerics::{broadcast, finite, parallel::Parallel},
};

#[path = "saved_batch.rs"]
mod saved;

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
    /// Contract each sum's analytic local derivatives with one input direction.
    /// `tangents` contains one shared direction or one direction per output.
    pub fn pushforward(&self, tangents: &[SumTangent]) -> Result<Vec<Complex>> {
        if (tangents.len() != 1 && tangents.len() != self.size)
            || tangents.iter().any(|t| !t.is_finite())
        {
            return Err(Error::InvalidInput(
                "tangents must be finite and match lattice output".into(),
            ));
        }
        broadcast::map(self.size, PARALLEL, |i| {
            let tangent = broadcast::element(tangents, i);
            let (wave, k, lattice, r, eta, part) = self.element(i);
            Ok(derivatives_part(wave, k, lattice, r, eta, part)?.pushforward(&tangent))
        })
    }

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
    pub fn pullback(&self, cotangent: &[Complex]) -> Result<Vec<SumGradient>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_reuses_inputs_across_forward_and_reverse_directions() {
        use crate::saved::SavedState;
        let (_, residual) = sum_array(
            vec![Family::Spherical { l: 2, m: -1 }],
            vec![Complex::new(2.1, 0.2)],
            vec![BlochLattice::new(&[vec![1.6]], &[0.1]).unwrap()],
            vec![[0.19, 0.11, 0.07], [0.21, -0.09, 0.05]],
            vec![Complex::new(0.9, 0.03)],
            vec![SumPart::Full],
        )
        .unwrap();
        let direction = SumTangent {
            k: Complex::new(0.1, -0.2),
            ..SumTangent::default()
        };
        let first = residual.pushforward(&[direction]).unwrap();
        let bytes = residual.save_state().unwrap();
        assert_eq!(bytes.len(), SumResidual::state_size(2).unwrap());
        let restored = SumResidual::from_state(&bytes).unwrap();
        assert_eq!(first, restored.pushforward(&[direction]).unwrap());
        assert!(SumResidual::from_state(&bytes[..bytes.len() - 1]).is_err());
        let cotangent = [Complex::new(0.3, 0.4), Complex::new(-0.2, 0.1)];
        let gradient = residual.pullback(&cotangent).unwrap();
        let opposite = residual
            .pushforward(&[SumTangent {
                k: -direction.k,
                ..SumTangent::default()
            }])
            .unwrap();
        for ((forward, reverse), (g, negated)) in first
            .iter()
            .zip(&gradient)
            .zip(cotangent.iter().zip(opposite))
        {
            let pairing = (g.conj() * forward).re;
            let transposed = (reverse.k.conj() * direction.k).re;
            assert!((pairing - transposed).abs() < 1e-12 * (1.0 + pairing.abs()));
            assert_eq!(*forward, -negated);
        }
        assert_eq!(first, residual.pushforward(&[direction]).unwrap());
    }
}

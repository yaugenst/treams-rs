//! Polarization coefficients of plane waves under cyclic permutations of the Cartesian
//! axes.
//!
//! Upstream: `treams.pw.permute_xyz` and its kernels `_cxyz_to_yzx_*` (one turn) and
//! `_cxyz_to_zxy_*` (two turns, `inverse=True`).
#![allow(clippy::indexing_slicing)] // Validated vectors, polarizations and cotangent shapes.

use nalgebra::DMatrix;

use super::polarization::{Direction, polarization_jet, transverse_values, wavenumbers};
use crate::{
    Complex, Error, Result,
    numerics::{
        Jet, finite,
        parallel::{PARALLEL_ITEMS, try_fill_chunks},
    },
    special::{check_pol, helicity_sign},
};

/// What [`permutation`] saves for its pullback: the wavevectors, the polarizations, the
/// number of turns and the polarization convention.
///
/// The pullback writes each mode's gradient on its own, so its result does not depend
/// on the thread count.
#[derive(Debug)]
pub struct PermutationResidual {
    pub(super) vectors: Vec<[Complex; 3]>,
    pub(super) polarizations: Vec<u8>,
    pub(super) turns: usize,
    pub(super) helicity: bool,
}

/// The parity coefficients `[same, cross]` of the plane wave `vector` under `turns`
/// cyclic turns of the axes, as jets in the wavevector components.
///
/// After one turn the permuted wavevector is `(kz, kx, ky)`, after two `(ky, kz, kx)`.
/// `same` maps a parity polarization to itself, `cross` to the other one (upstream
/// `polout == polin` and `polout != polin`). Zero turns give `[1, 0]`.
///
/// Three branches compute the pair:
///
/// 1. The closed form of upstream, `[-y z, -i x k] / (sqrt(x^2 + y^2) sqrt(z^2 + x^2))`
///    for one turn and its twin for two, when the largest component lies within 1e±70
///    and the squared modulus of the denominator within 1e±140. The first bound keeps
///    the squares of the components finite; the second keeps the direction away from
///    an axis and the derivative jets, which divide by the denominator squared, finite.
/// 2. The [`Direction`] frames of the original and the permuted wavevector, which scale
///    extreme magnitudes, when both have a nonzero transverse part.
/// 3. On the axis of either frame, the products of the polarization vectors in the
///    axis gauge of [`polarization_jet`].
fn permutation_pair<const N: usize>(vector: [Complex; 3], turns: usize) -> Result<[Jet<N>; 2]> {
    if turns == 0 {
        wavenumbers(vector)?;
        return Ok([Jet::constant(1.0), Jet::default()]);
    }
    // Both coefficients share one denominator, so the closed form needs no separate
    // normalization of the four transverse components.
    let scale = vector
        .iter()
        .map(|v| v.re.abs().max(v.im.abs()))
        .fold(0.0, f64::max);
    if (1e-70..=1e70).contains(&scale) && vector.iter().all(|&v| finite(v)) {
        let [x, y, z] = std::array::from_fn(|i| Jet::<N>::variable(vector[i], i));
        let transverse = (x * x + y * y).sqrt();
        let destination = if turns == 1 {
            (z * z + x * x).sqrt()
        } else {
            (y * y + z * z).sqrt()
        };
        let denominator = transverse * destination;
        let k = (x * x + y * y + z * z).sqrt();
        if (1e-140..=1e140).contains(&denominator.value.norm_sqr()) && k.value != Complex::default()
        {
            let inverse = Jet::constant(1.0) / denominator;
            return Ok(if turns == 1 {
                [-y * z * inverse, -Complex::i() * x * k * inverse]
            } else {
                [-x * z * inverse, Complex::i() * y * k * inverse]
            });
        }
    }
    let rotated = std::array::from_fn(|axis| vector[(axis + 3 - turns) % 3]);
    let source_direction = Direction::<N>::new(vector)?;
    let (transverse, xy) = transverse_values([rotated[0], rotated[1]])?;
    // Cyclic rotation preserves the full norm; only the transverse frame changes.
    let destination =
        Direction::<N>::from_parts(rotated, source_direction.k.value, transverse, xy)?;
    if source_direction.transverse.value != Complex::default()
        && destination.transverse.value != Complex::default()
    {
        let unpermute = |mut jet: Jet<N>| {
            jet.derivative = std::array::from_fn(|j| jet.derivative[(j + turns) % 3]);
            jet
        };
        let pair = if turns == 1 {
            (
                -source_direction.xy[1] * unpermute(destination.xy[0]),
                -Complex::i()
                    * source_direction.xy[0]
                    * (source_direction.k / unpermute(destination.transverse)),
            )
        } else {
            (
                -source_direction.xy[0] * unpermute(destination.xy[1]),
                Complex::i()
                    * source_direction.xy[1]
                    * (source_direction.k / unpermute(destination.transverse)),
            )
        };
        return Ok(pair.into());
    }
    // At an axis, evaluate the defined Cartesian polarization gauges directly.
    let source = polarization_jet::<N>(vector, 0, false)?;
    let mut result = [Jet::default(); 2];
    for (p, output) in result.iter_mut().enumerate() {
        // The algebraic dual of the parity pair is (-M, N).
        let dual_pol = u8::from(p == 1);
        let sign = if p == 0 { -1.0 } else { 1.0 };
        let dual = polarization_jet::<N>(rotated, dual_pol, false)?;
        for (axis, mut component) in dual.into_iter().enumerate() {
            component.derivative = std::array::from_fn(|j| component.derivative[(j + turns) % 3]);
            *output += sign * component * source[(axis + 3 - turns) % 3];
        }
    }
    Ok(result)
}

fn permutation_polarization<const N: usize>(
    pair: [Jet<N>; 2],
    pol: u8,
    helicity: bool,
) -> [Jet<N>; 2] {
    let [same, cross] = pair;
    std::array::from_fn(|p| {
        if helicity {
            if p == usize::from(pol) {
                same + helicity_sign(pol) * cross
            } else {
                Jet::default()
            }
        } else if p == usize::from(pol) {
            same
        } else {
            cross
        }
    })
}

/// Polarization coefficients of plane modes under `turns` cyclic permutations of the
/// axes, shape (2, modes).
///
/// Each column contains both output polarizations for one input mode; matching
/// direction labels and construction of a dense basis operator are separate.
///
/// Upstream: `treams.pw.permute_xyz` on arrays of plane modes.
pub fn permutation(
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    turns: usize,
    helicity: bool,
) -> Result<(DMatrix<Complex>, PermutationResidual)> {
    if vectors.is_empty()
        || polarizations.len() != vectors.len()
        || polarizations.iter().any(|&p| p > 1)
    {
        return Err(Error::InvalidInput(
            "require matching nonempty vectors and polarizations".into(),
        ));
    }
    let turns = turns % 3;
    let mut value = DMatrix::zeros(2, vectors.len());
    let parallel = vectors.len() >= PARALLEL_ITEMS;
    try_fill_chunks(value.as_mut_slice(), 4, parallel, |group, output| {
        let first = 2 * group;
        let pair = permutation_pair::<0>(vectors[first], turns)?;
        for (offset, column) in output.chunks_mut(2).enumerate() {
            let i = first + offset;
            let pair = if offset == 0 || vectors[i] == vectors[first] {
                pair
            } else {
                permutation_pair::<0>(vectors[i], turns)?
            };
            for (out, coefficient) in
                column
                    .iter_mut()
                    .zip(permutation_polarization(pair, polarizations[i], helicity))
            {
                *out = coefficient.value;
            }
        }
        Ok(())
    })?;
    if value.iter().any(|&z| !finite(z)) {
        return Err(Error::NonFinite("plane permutation overflow".into()));
    }
    Ok((
        value,
        PermutationResidual {
            vectors,
            polarizations,
            turns,
            helicity,
        },
    ))
}

impl PermutationResidual {
    /// Output polarizations and input plane modes: the shape `(2, modes)` of the
    /// coefficients.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (2, self.vectors.len())
    }

    /// Validate a wavevector tangent for the recorded operation.
    pub fn validate_tangents(&self, vectors: &[[Complex; 3]]) -> Result<()> {
        if vectors.len() != self.vectors.len() || vectors.iter().flatten().any(|&k| !finite(k)) {
            return Err(Error::InvalidInput(
                "invalid plane-permutation tangent".into(),
            ));
        }
        Ok(())
    }

    /// Contract the polarization coefficient jets with one wavevector direction.
    pub fn pushforward(&self, vectors: &[[Complex; 3]]) -> Result<DMatrix<Complex>> {
        self.validate_tangents(vectors)?;
        let mut tangent = DMatrix::zeros(2, self.vectors.len());
        if self.turns == 0 {
            return Ok(tangent);
        }
        // Adjacent modes share their polarization jet when their directions agree,
        // while each mode still carries its independent wavevector tangent.
        try_fill_chunks(
            tangent.as_mut_slice(),
            4,
            self.vectors.len() >= PARALLEL_ITEMS,
            |group, output| -> Result<()> {
                let first = 2 * group;
                let mut previous = None;
                for (offset, column) in output.chunks_mut(2).enumerate() {
                    let i = first + offset;
                    if vectors[i].iter().all(|&v| v == Complex::default()) {
                        continue;
                    }
                    let pair = if let Some((vector, pair)) = previous
                        && vector == self.vectors[i]
                    {
                        pair
                    } else {
                        let pair = permutation_pair::<3>(self.vectors[i], self.turns)?;
                        previous = Some((self.vectors[i], pair));
                        pair
                    };
                    for (out, coefficient) in column.iter_mut().zip(permutation_polarization(
                        pair,
                        self.polarizations[i],
                        self.helicity,
                    )) {
                        *out = coefficient
                            .derivative
                            .iter()
                            .zip(vectors[i])
                            .map(|(&d, v)| d * v)
                            .sum();
                    }
                }
                Ok(())
            },
        )?;
        Ok(tangent)
    }

    /// Wavevector gradients for a `cotangent` of the coefficients' shape. Direction
    /// derivatives at an axial polarization gauge are undefined.
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<Vec<[Complex; 3]>> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput(
                "invalid plane-permutation cotangent".into(),
            ));
        }
        let modes = self.vectors.len();
        if self.turns == 0 {
            return Ok(vec![[Complex::default(); 3]; modes]);
        }
        // As in the forward pass, adjacent modes usually share one direction.
        let mut gradient = vec![[Complex::default(); 3]; modes];
        let parallel = modes >= PARALLEL_ITEMS;
        try_fill_chunks(&mut gradient, 2, parallel, |group, output| {
            let first = 2 * group;
            let pair = permutation_pair::<3>(self.vectors[first], self.turns)?;
            for (offset, gradient) in output.iter_mut().enumerate() {
                let i = first + offset;
                let pair = if offset == 0 || self.vectors[i] == self.vectors[first] {
                    pair
                } else {
                    permutation_pair::<3>(self.vectors[i], self.turns)?
                };
                let coefficients =
                    permutation_polarization(pair, self.polarizations[i], self.helicity);
                *gradient = std::array::from_fn(|axis| {
                    (0..2)
                        .map(|p| cotangent[(p, i)] * coefficients[p].derivative[axis].conj())
                        .sum()
                });
            }
            Ok(())
        })?;
        Ok(gradient)
    }
}

/// One cyclic-coordinate polarization coefficient without allocating a matrix: the
/// coefficient of polarization `destination` in the permuted plane wave for an input of
/// polarization `source`.
///
/// Upstream: `treams.pw.permute_xyz`; `inverse=False` is `turns = 1` and `inverse=True`
/// is `turns = 2`.
pub fn permute_xyz(
    vector: [Complex; 3],
    destination: u8,
    source: u8,
    turns: usize,
    helicity: bool,
) -> Result<Complex> {
    check_pol(destination)?;
    check_pol(source)?;
    let value =
        permutation_polarization(permutation_pair::<0>(vector, turns % 3)?, source, helicity)
            [usize::from(destination)]
        .value;
    if !finite(value) {
        return Err(Error::NonFinite("non-finite plane permutation".into()));
    }
    Ok(value)
}

//! The lattice and Bloch vector of a sum, and the automatic Ewald split.
//!
//! Upstream: `_check_eta` of `lattice/_esum.pyx` for the automatic split. The reduced
//! bases are a treams-rs extension.

use std::f64::consts::PI;

use super::geometry::{reciprocal, reduce_basis, volume};
use crate::{Complex, Error, Result};

/// Row lattice vectors, a real Bloch vector `kpar` and their reduced bases in one, two or
/// three dimensions.
///
/// They are bundled because their derivatives share one jet; treams passes the lattice
/// vectors `a` and the Bloch vector `kpar` separately.
///
/// The lattice vectors and the Bloch vector are given in the lattice frame: Cartesian
/// components along the lattice's axes, which are z for one-dimensional spherical sums,
/// x for one-dimensional cylindrical sums, the xy plane in two dimensions and xyz in
/// three. They are not fractional coordinates. Lattice-vector and Bloch derivatives use
/// the same frame.
///
/// The constructor also reduces the direct and the reciprocal rows (`reduce_basis`),
/// over which the Ewald parts sum. A reduction is stored as `None` where the given rows
/// are already reduced, and then the parts sum over the given rows.
#[derive(Clone, Debug)]
pub struct BlochLattice {
    pub(super) dim: usize,
    /// Row vectors in lattice coordinates, zero outside the leading `dim` block.
    pub(super) direct: [[f64; 3]; 3],
    /// Reciprocal rows with `a_i · b_j = 2π δ_ij` in the leading block.
    pub(super) reciprocal: [[f64; 3]; 3],
    /// Bloch vector in the lattice frame, zero beyond the leading `dim` entries.
    pub(super) kpar: [f64; 3],
    pub(super) measure: f64,
    /// Reduced basis of the direct rows for the real-space sum; `None` where the given
    /// rows are already reduced (or need coefficients beyond `2^20`), so the sum uses
    /// them as they are.
    pub(super) direct_reduction: Option<Reduction>,
    /// Independently reduced basis of the reciprocal rows for the reciprocal sum; `None`
    /// as for `direct_reduction`.
    pub(super) reciprocal_reduction: Option<Reduction>,
}

/// A reduced basis `U rows` of lattice `rows` from [`geometry::reduce_basis`]: the
/// integer rows of `U` and of its inverse transpose `W`, stored exactly as `f64`.
/// Ewald shells of a skewed basis are far from Euclidean shells and can stop before
/// short vectors with large indices; shells of the reduced basis cannot.
///
/// [`geometry::reduce_basis`]: super::geometry::reduce_basis
#[derive(Clone, Copy, Debug)]
pub(super) struct Reduction {
    pub(super) rows: [[f64; 3]; 3],
    pub(super) dual: [[f64; 3]; 3],
}

impl Reduction {
    #[allow(clippy::cast_precision_loss)] // Entries are bounded by 2^20.
    fn new(rows: [[f64; 3]; 3], dim: usize) -> Option<Self> {
        let [rows, dual] = reduce_basis(rows, dim)?.map(|m| m.map(|row| row.map(|x| x as f64)));
        Some(Self { rows, dual })
    }

    /// Reduced-basis coordinates of the lattice point nearest to the point with
    /// coordinates `fractional` in the given basis, rounding in reduced coordinates.
    /// `None` stands for a given basis that is already reduced.
    pub(super) fn round(reduction: Option<&Self>, fractional: [f64; 3], dim: usize) -> [f64; 3] {
        match reduction {
            None => fractional.map(f64::round),
            Some(reduction) => std::array::from_fn(|i| {
                (0..dim)
                    .map(|k| reduction.dual[i][k] * fractional[k])
                    .sum::<f64>()
                    .round()
            }),
        }
    }

    /// Given-basis coordinates `U^T c` of a point with reduced coordinates `c`. `None`
    /// stands for a given basis that is already reduced, where both coincide.
    pub(super) fn given(reduction: Option<&Self>, reduced: [f64; 3], dim: usize) -> [f64; 3] {
        reduction.map_or(reduced, |reduction| {
            std::array::from_fn(|k| (0..dim).map(|i| reduction.rows[i][k] * reduced[i]).sum())
        })
    }
}

impl BlochLattice {
    /// Number of independent lattice vectors.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Validate and precompute the reciprocal lattice. Rows are primitive vectors.
    pub fn new(vectors: &[Vec<f64>], kpar: &[f64]) -> Result<Self> {
        let dim = vectors.len();
        if !(1..=3).contains(&dim)
            || kpar.len() != dim
            || vectors.iter().any(|row| row.len() != dim)
        {
            return Err(Error::InvalidInput(
                "lattice must be a finite square 1D, 2D or 3D matrix, with matching Bloch vector"
                    .into(),
            ));
        }
        Self::from_array(
            std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    if i < dim && j < dim {
                        vectors[i][j]
                    } else {
                        0.0
                    }
                })
            }),
            std::array::from_fn(|i| if i < dim { kpar[i] } else { 0.0 }),
            dim,
        )
    }

    /// Construct from fixed storage without allocating intermediate row vectors.
    ///
    /// Only the leading `dim` block of `vectors` and the leading `dim` entries of `kpar`
    /// count; the rest is stored as zero, so callers need not pad.
    pub fn from_array(vectors: [[f64; 3]; 3], kpar: [f64; 3], dim: usize) -> Result<Self> {
        if !(1..=3).contains(&dim) || kpar[..dim].iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidInput(
                "invalid lattice dimension or Bloch vector".into(),
            ));
        }
        let kpar = std::array::from_fn(|i| if i < dim { kpar[i] } else { 0.0 });
        let direct = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i < dim && j < dim {
                    vectors[i][j]
                } else {
                    0.0
                }
            })
        });
        let reciprocal = reciprocal(direct, dim)?;
        let measure = volume(direct, dim)?.abs();
        if !measure.is_finite() {
            return Err(Error::InvalidInput(
                "lattice is numerically singular".into(),
            ));
        }
        Ok(Self {
            dim,
            direct,
            reciprocal,
            kpar,
            measure,
            direct_reduction: Reduction::new(direct, dim),
            reciprocal_reduction: Reduction::new(reciprocal, dim),
        })
    }
}

/// The Ewald split: `eta`, or for zero an automatic choice from `kL` with the cell
/// length `L = measure^(1/dim)`. The exact sum is independent of the split, so it is
/// held fixed during differentiation instead of differentiating the heuristic.
pub(crate) fn resolve_split(k: Complex, lattice: &BlochLattice, eta: Complex) -> Complex {
    if eta != Complex::default() {
        return eta;
    }
    let length = match lattice.dim {
        1 => lattice.measure,
        2 => lattice.measure.sqrt(),
        _ => lattice.measure.cbrt(),
    };
    let e = 1.0 / (k * length);
    (2.0 * PI).sqrt() * e * (0.125 / e.norm()).max(1.0)
}

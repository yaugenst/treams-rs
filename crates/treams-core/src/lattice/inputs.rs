//! The inputs of one sum as forward-mode jets, and their jet slots.
//!
//! treams-rs extension: treams has no derivatives of its lattice sums.

use std::f64::consts::PI;

use super::{
    Derivatives, SumPart,
    cell::{BlochLattice, Reduction},
    wave::Family,
};
use crate::{Complex, numerics::Jet};

/// Jet slot of the wavenumber.
pub(super) const K: usize = 0;
/// First of the three Cartesian shift slots.
const POSITION: usize = 1;
/// First of the `dim` Bloch slots, which the `dim * dim` lattice-vector slots follow.
pub(super) const KPAR: usize = 4;

/// Jet slot of component `j` of lattice vector `i`.
const fn vector_slot(dim: usize, i: usize, j: usize) -> usize {
    KPAR + dim + dim * i + j
}

/// The lattice vector `(i, j)` of a jet slot at or after the vector slots.
const fn vector_component(dim: usize, slot: usize) -> (usize, usize) {
    ((slot - KPAR - dim) / dim, (slot - KPAR - dim) % dim)
}

pub(super) fn unpack<const N: usize>(result: Jet<N>, dim: usize) -> Derivatives {
    Derivatives {
        value: result.value,
        k: result.derivative[K],
        eta: Complex::default(),
        shift: std::array::from_fn(|i| result.derivative[POSITION + i]),
        kpar: std::array::from_fn(|i| {
            if i < dim {
                result.derivative[KPAR + i]
            } else {
                Complex::default()
            }
        }),
        vectors: std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i < dim && j < dim {
                    result.derivative[vector_slot(dim, i, j)]
                } else {
                    Complex::default()
                }
            })
        }),
    }
}

/// What one evaluation of `evaluate` computes.
#[derive(Clone, Copy, Debug)]
pub(super) enum Evaluation {
    /// A part of the sum, differentiated in every jet slot.
    Part(SumPart),
    /// The split derivative of the real-space part in slot 0, all other inputs fixed.
    EtaDerivative,
}

impl Evaluation {
    /// Whether the real-space part contributes.
    pub(super) fn real(self) -> bool {
        matches!(
            self,
            Self::Part(SumPart::Full | SumPart::Real) | Self::EtaDerivative
        )
    }

    /// Whether the reciprocal part and the self correction contribute.
    pub(super) fn reciprocal(self) -> bool {
        matches!(self, Self::Part(SumPart::Full | SumPart::Reciprocal))
    }
}

/// The inputs of one sum as jets in the slots above (or constants for the split
/// derivative), in lattice coordinates except for the Cartesian shift.
pub(super) struct Inputs<const N: usize> {
    pub(super) dim: usize,
    /// Cartesian axis of each lattice coordinate.
    pub(super) axes: [usize; 3],
    pub(super) k: Jet<N>,
    pub(super) r: [Jet<N>; 3],
    pub(super) kpar: [Jet<N>; 3],
    pub(super) direct: [[Jet<N>; 3]; 3],
    /// Reciprocal rows; `a_i · b_j = 2π δ_ij` gives their vector derivatives.
    pub(super) reciprocal: [[Jet<N>; 3]; 3],
    /// Cell measure `|det a|`.
    pub(super) measure: Jet<N>,
}

impl<const N: usize> Inputs<N> {
    /// Without `with_reciprocal` the reciprocal rows and the measure, which only the Ewald
    /// parts read, stay zero: the direct shells skip their jets. Always inlined, so the
    /// caller builds the inputs in its own frame instead of copying a returned value.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(super) fn new(
        wave: Family,
        k: Complex,
        lattice: &BlochLattice,
        r: [f64; 3],
        variable: bool,
        with_reciprocal: bool,
    ) -> Self {
        let dim = lattice.dim;
        let jet = |value: Complex, slot| {
            if variable {
                Jet::<N>::variable(value, slot)
            } else {
                Jet::constant(value)
            }
        };
        // A lattice function whose derivative in vector component (p, q) is `slope(p, q)`.
        let lattice_jet = |value: f64, slope: &dyn Fn(usize, usize) -> f64| Jet::<N> {
            value: Complex::new(value, 0.0),
            derivative: std::array::from_fn(|slot| {
                if slot < KPAR + dim {
                    return Complex::default();
                }
                let (p, q) = vector_component(dim, slot);
                Complex::new(if p < dim && q < dim { slope(p, q) } else { 0.0 }, 0.0)
            }),
        };
        let direct = &lattice.direct;
        let reciprocal = &lattice.reciprocal;
        // Filled in place, so no `from_fn` temporaries are copied into the result.
        let mut inputs = Self {
            dim,
            axes: wave.axes(dim),
            k: jet(k, K),
            r: [Jet::default(); 3],
            kpar: [Jet::default(); 3],
            direct: [[Jet::default(); 3]; 3],
            reciprocal: [[Jet::default(); 3]; 3],
            measure: Jet::default(),
        };
        for (i, row) in inputs.direct.iter_mut().enumerate() {
            inputs.r[i] = jet(r[i].into(), POSITION + i);
            if i < dim {
                inputs.kpar[i] = jet(lattice.kpar[i].into(), KPAR + i);
            }
            for (j, entry) in row.iter_mut().enumerate() {
                *entry = if i < dim && j < dim {
                    jet(direct[i][j].into(), vector_slot(dim, i, j))
                } else {
                    Jet::constant(direct[i][j])
                };
            }
        }
        if with_reciprocal {
            for (i, row) in inputs.reciprocal.iter_mut().enumerate() {
                for (j, entry) in row.iter_mut().enumerate() {
                    *entry = lattice_jet(reciprocal[i][j], &|p, q| {
                        -reciprocal[i][q] * reciprocal[p][j] / (2.0 * PI)
                    });
                }
            }
            inputs.measure = lattice_jet(lattice.measure, &|p, q| {
                lattice.measure * reciprocal[p][q] / (2.0 * PI)
            });
        }
        inputs
    }

    /// Integer combination `Σ_i n_i rows_i` of lattice rows.
    #[allow(clippy::cast_precision_loss)] // Shell indices are bounded by i32::MAX, exactly representable.
    pub(super) fn point(&self, rows: &[[Jet<N>; 3]; 3], n: [i64; 3]) -> [Jet<N>; 3] {
        self.combination(rows, n.map(|n| n as f64))
    }

    /// Combination `Σ_i c_i rows_i` of lattice rows with integer-valued `c`.
    pub(super) fn combination(&self, rows: &[[Jet<N>; 3]; 3], c: [f64; 3]) -> [Jet<N>; 3] {
        std::array::from_fn(|j| (0..self.dim).map(|i| c[i] * rows[i][j]).sum())
    }

    /// The reduced rows `U rows`, or `rows` when they are reduced already.
    pub(super) fn reduce(
        &self,
        rows: &[[Jet<N>; 3]; 3],
        reduction: Option<&Reduction>,
    ) -> [[Jet<N>; 3]; 3] {
        reduction.map_or(*rows, |reduction| {
            std::array::from_fn(|i| self.combination(rows, reduction.rows[i]))
        })
    }

    /// Cartesian displacement `-r - R` of the image at lattice point `R`.
    pub(super) fn image(&self, point: &[Jet<N>; 3]) -> [Jet<N>; 3] {
        let mut shift = self.r.map(|v| -v);
        for j in 0..self.dim {
            shift[self.axes[j]] -= point[j];
        }
        shift
    }

    /// Bloch phase `q · R` of lattice point `R`.
    pub(super) fn phase(&self, point: &[Jet<N>; 3]) -> Jet<N> {
        (0..self.dim).map(|j| self.kpar[j] * point[j]).sum()
    }
}

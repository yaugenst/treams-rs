//! Plane-wave translation phases and fields, with analytic pullbacks.
//!
//! Upstream: `treams.pw.translate` and `treams.special.vpw_M`, `vpw_N` and `vpw_A`.
#![allow(clippy::indexing_slicing)] // Validated three-component vectors and polarizations.

use nalgebra::DMatrix;
use rayon::prelude::*;

use super::polarization::{polarization, polarization_jet};
use crate::{
    Complex, Error, Result,
    numerics::{
        Jet, finite,
        parallel::{PARALLEL_ENTRIES, try_fill_chunks, try_map},
    },
};

/// `exp(i k·r)` for a complex wavevector `k` and a real point `r`, from the real and
/// imaginary parts of `k·r` separately.
#[inline]
pub(crate) fn phase(vector: [Complex; 3], point: [f64; 3]) -> Complex {
    let angle = vector[0].re * point[0] + vector[1].re * point[1] + vector[2].re * point[2];
    let attenuation = vector[0].im * point[0] + vector[1].im * point[1] + vector[2].im * point[2];
    let (sine, cosine) = angle.sin_cos();
    let scale = (-attenuation).exp();
    Complex::new(scale * cosine, scale * sine)
}

/// What [`phases`] saves for its pullback: the points and the wavevectors, without the
/// phases of every point and mode.
#[derive(Debug)]
pub struct PhasesResidual {
    points: Vec<[f64; 3]>,
    vectors: Vec<[Complex; 3]>,
}
/// Cotangents of real displacements and complex full wavevectors.
#[derive(Debug)]
pub struct PhasesGradient {
    /// Real displacement cotangents.
    pub points: Vec<[f64; 3]>,
    /// Complex wavevector cotangents in the real Hermitian pairing.
    pub vectors: Vec<[Complex; 3]>,
}

/// Direct scalar plane translation phase.
///
/// Upstream: `treams.pw.translate`.
#[inline]
pub fn translate(vector: [Complex; 3], point: [f64; 3]) -> Result<Complex> {
    if vector.iter().any(|&v| !finite(v)) || point.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "plane translation requires finite arguments".into(),
        ));
    }
    let value = phase(vector, point);
    if !finite(value) {
        return Err(Error::NonFinite("non-finite plane translation".into()));
    }
    Ok(value)
}

/// Plane-wave translation phases with shape (displacements, wavevectors).
///
/// Upstream: `treams.pw.translate` on arrays of points and wavevectors.
pub fn phases(
    points: Vec<[f64; 3]>,
    vectors: Vec<[Complex; 3]>,
) -> Result<(DMatrix<Complex>, PhasesResidual)> {
    if vectors.is_empty()
        || vectors.iter().flatten().any(|&k| !finite(k))
        || points.iter().flatten().any(|r| !r.is_finite())
    {
        return Err(Error::InvalidInput(
            "require finite displacements and nonempty finite wavevectors".into(),
        ));
    }
    let mut value = DMatrix::zeros(points.len(), vectors.len());
    let parallel = value.len() >= PARALLEL_ENTRIES;
    try_fill_chunks(
        value.as_mut_slice(),
        points.len().max(1),
        parallel,
        |j, column| -> Result<()> {
            for (out, &point) in column.iter_mut().zip(&points) {
                *out = phase(vectors[j], point);
            }
            Ok(())
        },
    )?;
    if value.iter().any(|&v| !finite(v)) {
        return Err(Error::NonFinite("plane translation phase overflow".into()));
    }
    Ok((value, PhasesResidual { points, vectors }))
}
impl PhasesResidual {
    /// Number of displacements and wavevectors.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.points.len(), self.vectors.len())
    }
    /// Recompute each phase once and add its terms to the point and wavevector gradients.
    ///
    /// The longer axis is split into at most 32 blocks, fixed by the shape, that
    /// run in parallel; each block sums its own rows (or columns) in index order and
    /// a partial gradient of the shorter axis, and the partials are added in block
    /// order. The result is deterministic and the scratch is at most 32 gradients
    /// of the shorter axis.
    pub fn pullback<S: nalgebra::Storage<Complex, nalgebra::Dyn, nalgebra::Dyn> + Sync>(
        self,
        cotangent: &nalgebra::Matrix<Complex, nalgebra::Dyn, nalgebra::Dyn, S>,
    ) -> Result<PhasesGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput("invalid plane-phase cotangent".into()));
        }
        let (rows, columns) = self.shape();
        // Add the phase terms of points `i0..` and wavevectors `j0..` to gradients
        // indexed from those offsets; each entry sums over the other axis in order.
        let contract =
            |i0: usize, points: &mut [[f64; 3]], j0: usize, vectors: &mut [[Complex; 3]]| {
                for (j, (vector_gradient, &vector)) in
                    vectors.iter_mut().zip(&self.vectors[j0..]).enumerate()
                {
                    for (i, (point_gradient, &point)) in
                        points.iter_mut().zip(&self.points[i0..]).enumerate()
                    {
                        let factor = cotangent[(i0 + i, j0 + j)].conj()
                            * Complex::i()
                            * phase(vector, point);
                        for axis in 0..3 {
                            point_gradient[axis] += (factor * vector[axis]).re;
                            vector_gradient[axis] += factor.conj() * point[axis];
                        }
                    }
                }
            };
        let long = rows.max(columns);
        let block = long.div_ceil(32).max(16);
        let blocks = long.div_ceil(block);
        let task = |b: usize| {
            let (start, stop) = (b * block, ((b + 1) * block).min(long));
            if rows >= columns {
                let mut points = vec![[0.0; 3]; stop - start];
                let mut vectors = vec![[Complex::default(); 3]; columns];
                contract(start, &mut points, 0, &mut vectors);
                (points, vectors)
            } else {
                let mut points = vec![[0.0; 3]; rows];
                let mut vectors = vec![[Complex::default(); 3]; stop - start];
                contract(0, &mut points, start, &mut vectors);
                (points, vectors)
            }
        };
        let partials = try_map(
            blocks,
            cotangent.len() >= PARALLEL_ENTRIES,
            |b| -> Result<_> { Ok(task(b)) },
        )?;
        let mut gradient = PhasesGradient {
            points: Vec::with_capacity(rows),
            vectors: Vec::with_capacity(columns),
        };
        if rows >= columns {
            gradient.vectors = vec![[Complex::default(); 3]; columns];
            for (points, vectors) in partials {
                gradient.points.extend(points);
                for (target, partial) in gradient
                    .vectors
                    .iter_mut()
                    .flatten()
                    .zip(vectors.iter().flatten())
                {
                    *target += partial;
                }
            }
        } else {
            gradient.points = vec![[0.0; 3]; rows];
            for (points, vectors) in partials {
                gradient.vectors.extend(vectors);
                for (target, partial) in gradient
                    .points
                    .iter_mut()
                    .flatten()
                    .zip(points.iter().flatten())
                {
                    *target += partial;
                }
            }
        }
        Ok(gradient)
    }
}
/// What [`field`] saves for its pullback: its inputs, for weighted fields or for the
/// matrix that samples every mode.
///
/// Below 4096 points times modes (`PARALLEL_ENTRIES`) the pullback adds the modes in
/// order on the calling thread. From there on, Rayon's `try_fold` and `try_reduce` add
/// the per-mode point gradients, so their last bits can change with the thread count and
/// from run to run. Each amplitude and wavevector gradient belongs to one mode and does
/// not.
#[derive(Debug)]
pub struct FieldResidual {
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    points: Vec<[f64; 3]>,
    coefficients: Option<Vec<Complex>>,
    helicity: bool,
    fixed_vectors: bool,
}
/// Cotangents in the real Hermitian pairing.
#[derive(Debug)]
pub struct FieldGradient {
    /// Amplitude cotangents; empty for a field operator.
    pub coefficients: Vec<Complex>,
    /// Real sample-point cotangents.
    pub points: Vec<[f64; 3]>,
    /// Complex full-wavevector cotangents; zero when wavevectors are held fixed.
    pub vectors: Vec<[Complex; 3]>,
}
/// Evaluate weighted plane fields, or the full operator when coefficients are absent.
///
/// The rows hold the three Cartesian components of each point in turn. With
/// `fixed_vectors`, the pullback holds the wavevectors fixed and returns zero wavevector
/// gradients.
///
/// Upstream: sums of `treams.special.vpw_M`, `vpw_N` or `vpw_A` over the plane modes.
pub fn field(
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    points: Vec<[f64; 3]>,
    coefficients: Option<Vec<Complex>>,
    helicity: bool,
    fixed_vectors: bool,
) -> Result<(DMatrix<Complex>, FieldResidual)> {
    if vectors.is_empty()
        || vectors.len() != polarizations.len()
        || points.iter().flatten().any(|r| !r.is_finite())
        || coefficients
            .as_ref()
            .is_some_and(|c| c.len() != vectors.len() || c.iter().any(|&v| !finite(v)))
    {
        return Err(Error::InvalidInput(
            "require nonempty plane modes, finite points, matching finite amplitudes and polarizations"
                .into(),
        ));
    }
    let electric: Vec<_> = vectors
        .iter()
        .zip(&polarizations)
        .map(|(&k, &p)| polarization(k, p, helicity))
        .collect::<Result<_>>()?;
    let mut value = DMatrix::zeros(
        3 * points.len(),
        if coefficients.is_some() {
            1
        } else {
            vectors.len()
        },
    );
    let parallel = points.len() * vectors.len() >= PARALLEL_ENTRIES;
    if let Some(c) = &coefficients {
        try_fill_chunks(value.as_mut_slice(), 3, parallel, |p, out| -> Result<()> {
            for ((&k, e), &c) in vectors.iter().zip(&electric).zip(c) {
                let weight = c * phase(k, points[p]);
                for (out, &e) in out.iter_mut().zip(e) {
                    *out += weight * e;
                }
            }
            Ok(())
        })?;
    } else if !points.is_empty() {
        let size = 3 * points.len();
        try_fill_chunks(
            value.as_mut_slice(),
            size,
            parallel,
            |j, column| -> Result<()> {
                for (out, &point) in column.chunks_exact_mut(3).zip(&points) {
                    let phase = phase(vectors[j], point);
                    for (out, &e) in out.iter_mut().zip(&electric[j]) {
                        *out = phase * e;
                    }
                }
                Ok(())
            },
        )?;
    }
    Ok((
        value,
        FieldResidual {
            vectors,
            polarizations,
            points,
            coefficients,
            helicity,
            fixed_vectors,
        },
    ))
}
impl FieldResidual {
    /// Flattened output dimensions (3 * samples, modes or one weighted column).
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (
            3 * self.points.len(),
            if self.coefficients.is_some() {
                1
            } else {
                self.vectors.len()
            },
        )
    }
    /// Gradients for a `cotangent` of the field's shape. The pullback recomputes the
    /// polarization and phase derivatives instead of storing a dense field Jacobian.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<FieldGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput("invalid plane-field cotangent".into()));
        }
        let fixed_vectors = self.fixed_vectors;
        let zero = || FieldGradient {
            coefficients: vec![Complex::default(); self.coefficients.as_ref().map_or(0, Vec::len)],
            points: vec![[0.0; 3]; self.points.len()],
            vectors: vec![[Complex::default(); 3]; self.vectors.len()],
        };
        if self.points.is_empty() {
            return Ok(zero());
        }
        let contract = |mut result: FieldGradient, (j, &vector)| -> Result<_> {
            let electric = if fixed_vectors {
                polarization(vector, self.polarizations[j], self.helicity)?.map(Jet::<3>::constant)
            } else {
                polarization_jet::<3>(vector, self.polarizations[j], self.helicity)?
            };
            let coefficient = self
                .coefficients
                .as_ref()
                .map_or(Complex::new(1.0, 0.0), |c| c[j]);
            let fixed = fixed_vectors;
            // Each sample's three Cartesian cotangents, contiguous in the column.
            let column = cotangent.column(if self.coefficients.is_some() { 0 } else { j });
            let (samples, _) = column.as_slice().as_chunks::<3>();
            // Accumulate through references into the result, which the loop can
            // keep in registers, in the same order as indexing it would.
            let FieldGradient {
                coefficients,
                points,
                vectors,
            } = &mut result;
            // Amplitude cotangents are empty for an operator.
            let mut coefficient_gradient = coefficients.get_mut(j);
            let vector_gradient = &mut vectors[j];
            let mut polarization_cotangent = [Complex::default(); 3];
            for ((&point, rows), point_gradient) in self.points.iter().zip(samples).zip(points) {
                let phase = phase(vector, point);
                let weighted_phase = coefficient * phase;
                let paired: Complex = (0..3).map(|a| rows[a].conj() * electric[a].value).sum();
                if let Some(gradient) = coefficient_gradient.as_deref_mut() {
                    *gradient += (phase * paired).conj();
                }
                let paired_field = weighted_phase * paired * Complex::i();
                for axis in 0..3 {
                    point_gradient[axis] += (paired_field * vector[axis]).re;
                    if !fixed {
                        vector_gradient[axis] += (paired_field * point[axis]).conj();
                        polarization_cotangent[axis] += rows[axis] * weighted_phase.conj();
                    }
                }
            }
            // The polarization is the same at every sample: pull it back once per mode.
            if !fixed {
                for (axis, gradient) in vector_gradient.iter_mut().enumerate() {
                    *gradient += (0..3)
                        .map(|a| polarization_cotangent[a] * electric[a].derivative[axis].conj())
                        .sum::<Complex>();
                }
            }
            Ok(result)
        };
        if self.points.len() * self.vectors.len() < PARALLEL_ENTRIES {
            return self.vectors.iter().enumerate().try_fold(zero(), contract);
        }
        self.vectors
            .par_iter()
            .enumerate()
            .try_fold(zero, contract)
            .try_reduce(zero, |mut a, b| {
                for (a, b) in a
                    .coefficients
                    .iter_mut()
                    .chain(a.vectors.iter_mut().flatten())
                    .zip(b.coefficients.iter().chain(b.vectors.iter().flatten()))
                {
                    *a += b;
                }
                for (a, b) in a.points.iter_mut().flatten().zip(b.points.iter().flatten()) {
                    *a += b;
                }
                Ok(a)
            })
    }
}

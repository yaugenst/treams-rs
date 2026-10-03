//! Plane-wave translation phases and fields, with analytic pullbacks.
//!
//! Upstream: `treams.pw.translate` and `treams.special.vpw_M`, `vpw_N` and `vpw_A`.
#![allow(clippy::indexing_slicing)] // Validated three-component vectors and polarizations.

use nalgebra::DMatrix;

use super::polarization::{polarization, polarization_jet};
use crate::{
    Complex, Error, Result,
    numerics::{
        Jet, finite,
        parallel::{PARALLEL_ENTRIES, try_fill_chunks, try_fold_ordered, try_map},
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
/// The pullback splits the longer of the point and mode axes into at most 32 blocks,
/// fixed by the shape, as [`PhasesResidual::pullback`] does. Each block writes the
/// gradients of its own points or modes and a partial sum of the gradients of the
/// other axis, and the partial sums add in block order; a mode's polarization pulls
/// back once, from its total over all points. The blocks run in parallel from 4096
/// points times modes (`PARALLEL_ENTRIES`), and in the same order on the calling thread
/// below, so the thread count does not change the result.
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
        let (samples, modes) = (self.points.len(), self.vectors.len());
        let mut gradient = FieldGradient {
            coefficients: vec![Complex::default(); self.coefficients.as_ref().map_or(0, Vec::len)],
            points: vec![[0.0; 3]; samples],
            vectors: vec![[Complex::default(); 3]; modes],
        };
        if samples == 0 {
            return Ok(gradient);
        }
        // The blocks of `PhasesResidual::pullback`; one block adds in the order of a
        // plain loop over modes and points.
        let parallel = samples * modes >= PARALLEL_ENTRIES;
        let block = samples.max(modes).div_ceil(32).max(16);
        if modes >= samples {
            // Blocks of modes write the amplitude and wavevector gradients of their
            // modes and add into partial point gradients. An operator has no amplitude
            // gradients, so its blocks get none.
            let mut coefficient_blocks = gradient.coefficients.chunks_mut(block);
            let blocks: Vec<_> = gradient
                .vectors
                .chunks_mut(block)
                .map(|vectors| (vectors, coefficient_blocks.next()))
                .collect();
            gradient.points = try_fold_ordered(
                blocks,
                parallel,
                || vec![[0.0; 3]; samples],
                |mut points, b, (vectors, mut coefficients)| -> Result<_> {
                    for (offset, vector) in vectors.iter_mut().enumerate() {
                        let j = b * block + offset;
                        let mode = self.mode(j)?;
                        let mut sums = ModeSums::default();
                        let rows = self.rows(cotangent, j);
                        self.contract(&mode, &self.points, rows, &mut points, &mut sums);
                        let coefficient = coefficients.as_deref_mut().map(|c| &mut c[offset]);
                        self.finish(&mode, &sums, coefficient, vector);
                    }
                    Ok(points)
                },
                |mut total, partial| {
                    for (a, b) in total.iter_mut().flatten().zip(partial.iter().flatten()) {
                        *a += b;
                    }
                    total
                },
            )?;
        } else {
            // Blocks of points write the gradients of their points and add into partial
            // sums of every mode, which finish once all blocks are added.
            let terms = (0..modes)
                .map(|j| self.mode(j))
                .collect::<Result<Vec<_>>>()?;
            let blocks: Vec<_> = gradient.points.chunks_mut(block).collect();
            let sums = try_fold_ordered(
                blocks,
                parallel,
                || vec![ModeSums::default(); modes],
                |mut sums, b, point_gradients| -> Result<_> {
                    let range = b * block..b * block + point_gradients.len();
                    let points = &self.points[range.clone()];
                    for (j, (mode, sums)) in terms.iter().zip(&mut sums).enumerate() {
                        let rows = &self.rows(cotangent, j)[range.clone()];
                        self.contract(mode, points, rows, point_gradients, sums);
                    }
                    Ok(sums)
                },
                |mut total, partial| {
                    for (total, partial) in total.iter_mut().zip(&partial) {
                        total.add(partial);
                    }
                    total
                },
            )?;
            let mut coefficients = gradient.coefficients.iter_mut();
            for ((mode, sums), vector) in terms.iter().zip(&sums).zip(&mut gradient.vectors) {
                self.finish(mode, sums, coefficients.next(), vector);
            }
        }
        Ok(gradient)
    }

    /// The wavevector, polarization jet and amplitude of plane mode `j`.
    fn mode(&self, j: usize) -> Result<ModeTerms> {
        let vector = self.vectors[j];
        let electric = if self.fixed_vectors {
            polarization(vector, self.polarizations[j], self.helicity)?.map(Jet::<3>::constant)
        } else {
            polarization_jet::<3>(vector, self.polarizations[j], self.helicity)?
        };
        let coefficient = self
            .coefficients
            .as_ref()
            .map_or(Complex::new(1.0, 0.0), |c| c[j]);
        Ok(ModeTerms {
            vector,
            electric,
            coefficient,
        })
    }

    /// The three Cartesian cotangents of each sample, contiguous in the cotangent
    /// column of mode `j`, or in the one column of a weighted field.
    fn rows<'c>(&self, cotangent: &'c DMatrix<Complex>, j: usize) -> &'c [[Complex; 3]] {
        let length = 3 * self.points.len();
        let column = if self.coefficients.is_some() { 0 } else { j };
        cotangent.as_slice()[column * length..][..length]
            .as_chunks::<3>()
            .0
    }

    /// Add the terms of `mode` at `points`, whose cotangents are `rows`, to the point
    /// gradients `point_gradients` and to the mode's `sums`, in point order.
    fn contract(
        &self,
        mode: &ModeTerms,
        points: &[[f64; 3]],
        rows: &[[Complex; 3]],
        point_gradients: &mut [[f64; 3]],
        sums: &mut ModeSums,
    ) {
        let (weighted, fixed) = (self.coefficients.is_some(), self.fixed_vectors);
        let ModeTerms {
            vector,
            electric,
            coefficient,
        } = *mode;
        // Accumulate through references into the sums, which the loop can keep in
        // registers, in the same order as indexing them would.
        let ModeSums {
            coefficient: coefficient_gradient,
            vector: vector_gradient,
            polarization: polarization_cotangent,
        } = sums;
        for ((&point, rows), point_gradient) in points.iter().zip(rows).zip(point_gradients) {
            let phase = phase(vector, point);
            let weighted_phase = coefficient * phase;
            let paired: Complex = (0..3).map(|a| rows[a].conj() * electric[a].value).sum();
            if weighted {
                *coefficient_gradient += (phase * paired).conj();
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
    }

    /// Write the amplitude gradient `coefficient` (none for an operator) and the
    /// wavevector gradient `vector` of `mode` from its sums over all points.
    fn finish(
        &self,
        mode: &ModeTerms,
        sums: &ModeSums,
        coefficient: Option<&mut Complex>,
        vector: &mut [Complex; 3],
    ) {
        if let Some(coefficient) = coefficient {
            *coefficient = sums.coefficient;
        }
        *vector = sums.vector;
        // The polarization is the same at every point: pull it back once per mode.
        if !self.fixed_vectors {
            for (axis, gradient) in vector.iter_mut().enumerate() {
                *gradient += (0..3)
                    .map(|a| sums.polarization[a] * mode.electric[a].derivative[axis].conj())
                    .sum::<Complex>();
            }
        }
    }
}

/// What the field pullback needs of one plane mode at every point.
#[derive(Clone, Copy)]
struct ModeTerms {
    /// The complex wavevector.
    vector: [Complex; 3],
    /// The polarization, with its derivatives in the wavevector unless it is fixed.
    electric: [Jet<3>; 3],
    /// The amplitude; one for an operator.
    coefficient: Complex,
}

/// Sums over points of one plane mode's amplitude, wavevector and polarization
/// cotangents; the last two stay zero when the wavevectors are fixed.
#[derive(Clone, Copy, Default)]
struct ModeSums {
    coefficient: Complex,
    vector: [Complex; 3],
    polarization: [Complex; 3],
}

impl ModeSums {
    /// Add the sums over other points.
    fn add(&mut self, other: &Self) {
        self.coefficient += other.coefficient;
        for (a, b) in self
            .vector
            .iter_mut()
            .chain(&mut self.polarization)
            .zip(other.vector.iter().chain(&other.polarization))
        {
            *a += b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::field;
    use crate::{
        Complex,
        numerics::parallel::PARALLEL_ENTRIES,
        test_support::{assert_same_bits_on_pools, bits, patterned},
    };

    /// Field pullbacks add their partial sums in blocks fixed by the shape: blocks of
    /// modes and blocks of points, for weighted fields and operators, with and without
    /// wavevector derivatives, give gradients that repeat bit for bit on every pool
    /// size once the blocks run in parallel.
    #[test]
    fn pullbacks_do_not_depend_on_the_thread_count() {
        for (samples, modes) in [(30_u32, 200_u32), (300, 20)] {
            let vectors: Vec<[Complex; 3]> = (0..modes)
                .map(|j| {
                    let t = f64::from(j);
                    let (theta, phi) = (1.55 + 1.3 * (0.61 * t).sin(), 2.3 * t);
                    [
                        Complex::new(1.2 * theta.sin() * phi.cos(), 0.0),
                        Complex::new(1.2 * theta.sin() * phi.sin(), 0.0),
                        Complex::new(1.2 * theta.cos(), 0.01),
                    ]
                })
                .collect();
            let polarizations: Vec<u8> = (0..modes).map(|j| u8::from(j % 2 == 0)).collect();
            let points: Vec<[f64; 3]> = (0..samples)
                .map(|i| {
                    let t = f64::from(i);
                    [(0.37 * t).sin(), (0.53 * t).cos(), 0.8 * (0.71 * t).sin()]
                })
                .collect();
            assert!(points.len() * vectors.len() >= PARALLEL_ENTRIES);
            for weighted in [true, false] {
                let coefficients =
                    weighted.then(|| patterned(vectors.len(), 1, 0.3).as_slice().to_vec());
                let columns = if weighted { 1 } else { vectors.len() };
                let g = patterned(3 * points.len(), columns, 0.8);
                for fixed_vectors in [false, true] {
                    assert_same_bits_on_pools(|| {
                        let (_, residual) = field(
                            vectors.clone(),
                            polarizations.clone(),
                            points.clone(),
                            coefficients.clone(),
                            true,
                            fixed_vectors,
                        )
                        .unwrap();
                        let g = residual.pullback(&g).unwrap();
                        bits(&[&g.coefficients, &g.points, &g.vectors])
                    });
                }
            }
        }
    }
}

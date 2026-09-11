//! Plane-wave scattering composition and its factorization-reusing adjoint.
#![allow(clippy::indexing_slicing)] // Four blocks, validated equal matrix dimensions.

use faer::MatRef;
use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::{
    Complex, Error, Result, finite,
    interaction::{
        product, product_adjoint_left, product_adjoint_left_view, product_adjoint_right,
        product_adjoint_right_into, product_views, view, view_mut,
    },
    linalg::{self, Lu, SolveResidual},
    ratio,
};

/// Blocks ordered as transmission up, reflection up, reflection down, transmission down.
pub type Blocks = [DMatrix<Complex>; 4];

/// Owned square block that preserves its input memory order.
#[derive(Debug)]
pub struct StoredBlock {
    values: DMatrix<Complex>,
    row_major: bool,
}
impl From<DMatrix<Complex>> for StoredBlock {
    fn from(values: DMatrix<Complex>) -> Self {
        Self {
            values,
            row_major: false,
        }
    }
}
impl StoredBlock {
    /// Take ownership of row-major values without transposing their storage.
    pub fn from_rows(dimension: usize, values: Vec<Complex>) -> Result<Self> {
        if dimension.checked_mul(dimension) != Some(values.len()) {
            return Err(Error::InvalidInput(
                "block values must match its square dimension".into(),
            ));
        }
        Ok(Self {
            values: DMatrix::from_vec(dimension, dimension, values),
            row_major: true,
        })
    }
    fn copy_view(matrix: MatRef<'_, Complex>) -> Self {
        let row_major = matrix.col_stride() == 1;
        let matrix = if row_major {
            matrix.transpose()
        } else {
            matrix
        };
        let mut values = Vec::with_capacity(matrix.nrows() * matrix.ncols());
        if let Some(contiguous) = matrix.try_as_col_major() {
            for j in 0..matrix.ncols() {
                values.extend_from_slice(contiguous.col(j).as_slice());
            }
        } else {
            for j in 0..matrix.ncols() {
                for i in 0..matrix.nrows() {
                    values.push(matrix[(i, j)]);
                }
            }
        }
        Self {
            values: DMatrix::from_vec(matrix.nrows(), matrix.ncols(), values),
            row_major,
        }
    }
    fn view(&self) -> MatRef<'_, Complex> {
        let data = view(&self.values);
        if self.row_major {
            data.transpose()
        } else {
            data
        }
    }
    fn into_buffer(self) -> DMatrix<Complex> {
        self.values
    }
}

/// Geometry for compact, transversely averaged chirality forms.
#[derive(Debug)]
pub struct ChiralityResidual {
    ks: Vec<Complex>,
    normal: Vec<Complex>,
    interval: [f64; 2],
}

/// Complex full/normal wavenumber and real interval-endpoint cotangents.
#[derive(Debug)]
pub struct ChiralityGradient {
    /// Full wavenumbers.
    pub ks: Vec<Complex>,
    /// Normal wavenumbers.
    pub normal: Vec<Complex>,
    /// Start and end of the averaging interval.
    pub interval: [f64; 2],
}

fn mean_exp<const N: usize>(slope: Jet<N>, interval: [Jet<N>; 2]) -> Jet<N> {
    let width = slope * (interval[1] - interval[0]);
    if width.value.norm() < 0.1 {
        let square = (width * 0.5).powi(2);
        let mut term = Jet::constant(1.0);
        let mut sum = term;
        for j in 1..=6 {
            term *= square / f64::from(2 * j * (2 * j + 1));
            sum += term;
        }
        (slope * (interval[0] + interval[1]) * 0.5).exp() * sum
    } else {
        ((slope * interval[1]).exp() - (slope * interval[0]).exp()) / width
    }
}

fn chirality_mode<const N: usize>(k: Complex, normal: Complex, z: [f64; 2]) -> [Jet<N>; 3] {
    let scale = k.norm();
    let kr = Jet::variable(k.re, 0) / scale;
    let ki = Jet::variable(k.im, 1) / scale;
    let nr = Jet::variable(normal.re, 2);
    let ni = Jet::variable(normal.im, 3);
    let z = [Jet::variable(z[0], 4), Jet::variable(z[1], 5)];
    let denominator = kr.powi(2) + ki.powi(2);
    let same = 2.0 * (kr.powi(2) + (ni / scale).powi(2)) / denominator;
    let cross = 2.0 * (kr.powi(2) - (nr / scale).powi(2)) / denominator;
    [
        same * mean_exp(-2.0 * ni, z),
        same * mean_exp(2.0 * ni, z),
        2.0 * cross * mean_exp(2.0 * Complex::i() * nr, z),
    ]
}

/// Up, down and coherent cross coefficients, shape (3, modes), before polarization.
///
/// The density is 2 Re(E* . i Z H). The cross form contracts as Re(d* X u).
/// Real transverse wavevectors in an xy basis are assumed; full/normal k vary.
pub fn chirality_density(
    ks: Vec<Complex>,
    normal: Vec<Complex>,
    interval: [f64; 2],
) -> Result<(DMatrix<Complex>, ChiralityResidual)> {
    if ks.is_empty()
        || normal.len() != ks.len()
        || ks.iter().any(|&k| !finite(k) || k == Complex::default())
        || normal.iter().any(|&k| !finite(k))
        || interval.iter().any(|z| !z.is_finite())
    {
        return Err(Error::InvalidInput(
            "chirality requires matching finite wavenumbers, nonzero full k and finite interval"
                .into(),
        ));
    }
    let mut value = DMatrix::zeros(3, ks.len());
    for (j, (&k, &n)) in ks.iter().zip(&normal).enumerate() {
        for (i, coefficient) in chirality_mode::<0>(k, n, interval).iter().enumerate() {
            value[(i, j)] = coefficient.value;
        }
    }
    if value.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput("chirality density overflow".into()));
    }
    Ok((
        value,
        ChiralityResidual {
            ks,
            normal,
            interval,
        },
    ))
}

impl ChiralityResidual {
    /// Compact output dimensions.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (3, self.ks.len())
    }

    /// Recompute six local derivatives per mode; no dense Jacobian is retained.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<ChiralityGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput("invalid chirality cotangent".into()));
        }
        let mut gradient = ChiralityGradient {
            ks: Vec::with_capacity(self.ks.len()),
            normal: Vec::with_capacity(self.ks.len()),
            interval: [0.0; 2],
        };
        for (j, (&k, &n)) in self.ks.iter().zip(&self.normal).enumerate() {
            let coefficients = chirality_mode::<6>(k, n, self.interval);
            let g: [f64; 6] = std::array::from_fn(|p| {
                coefficients
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (cotangent[(i, j)].conj() * c.derivative[p]).re)
                    .sum()
            });
            gradient.ks.push(Complex::new(g[0], g[1]));
            gradient.normal.push(Complex::new(g[2], g[3]));
            gradient.interval[0] += g[4];
            gradient.interval[1] += g[5];
        }
        Ok(gradient)
    }
}

/// Real transverse geometry for chirality forms along any Cartesian normal.
#[derive(Debug)]
pub struct OrientedChiralityResidual {
    transverse: Vec<[f64; 2]>,
    normal: Vec<Complex>,
    polarizations: Vec<u8>,
    axis: usize,
    interval: [f64; 2],
}

/// Real transverse, complex normal and real interval-endpoint cotangents.
#[derive(Debug)]
pub struct OrientedChiralityGradient {
    /// The two transverse components in cyclic Cartesian order.
    pub transverse: Vec<[f64; 2]>,
    /// Normal wavenumbers.
    pub normal: Vec<Complex>,
    /// Start and end of the averaging interval.
    pub interval: [f64; 2],
}

fn oriented_chirality_mode<const N: usize>(
    transverse: [f64; 2],
    normal: Complex,
    pol: u8,
    axis: usize,
    interval: [f64; 2],
) -> Result<[Jet<N>; 3]> {
    let nr = Jet::variable(normal.re, 2);
    let ni = Jet::variable(normal.im, 3);
    let z = [Jet::variable(interval[0], 4), Jet::variable(interval[1], 5)];
    let sign = 2.0 * f64::from(pol) - 1.0;
    // The observable has a smooth limit even where the polarization gauge does not.
    if transverse.iter().all(|&q| q == 0.0) {
        if normal == Complex::default() {
            return Err(Error::InvalidInput("wavevector must be nonzero".into()));
        }
        return Ok([
            2.0 * sign * mean_exp(-2.0 * ni, z),
            2.0 * sign * mean_exp(2.0 * ni, z),
            Jet::default(),
        ]);
    }
    let mut vector = [Jet::default(); 3];
    vector[axis] = nr + Complex::i() * ni;
    vector[(axis + 1) % 3] = Jet::variable(transverse[0], 0);
    vector[(axis + 2) % 3] = Jet::variable(transverse[1], 1);
    let up = crate::plane::polarization_from_inputs(vector, pol)?;
    vector[axis] = -vector[axis];
    let down = crate::plane::polarization_from_inputs(vector, pol)?;
    // All six local parameters are real, so conjugation acts on their derivatives.
    let inner = |a: [Jet<N>; 3], b: [Jet<N>; 3]| -> Jet<N> {
        a.into_iter()
            .zip(b)
            .map(|(a, b)| {
                Jet {
                    value: a.value.conj(),
                    derivative: a.derivative.map(|d| d.conj()),
                } * b
            })
            .sum()
    };
    Ok([
        2.0 * sign * inner(up, up) * mean_exp(-2.0 * ni, z),
        2.0 * sign * inner(down, down) * mean_exp(2.0 * ni, z),
        4.0 * sign * inner(down, up) * mean_exp(2.0 * Complex::i() * nr, z),
    ])
}

/// Signed helicity up/down/cross forms, shape (3, modes), for any Cartesian normal.
///
/// The transverse components are real and follow the cyclic order after `axis`.
/// The cross form contracts as Re(down* X up). Geometry alone is retained.
pub fn oriented_chirality(
    transverse: Vec<[f64; 2]>,
    normal: Vec<Complex>,
    polarizations: Vec<u8>,
    axis: usize,
    interval: [f64; 2],
) -> Result<(DMatrix<Complex>, OrientedChiralityResidual)> {
    if transverse.is_empty()
        || transverse.len() != normal.len()
        || transverse.len() != polarizations.len()
        || axis > 2
        || transverse.iter().flatten().any(|q| !q.is_finite())
        || normal.iter().any(|&k| !finite(k))
        || polarizations.iter().any(|&p| p > 1)
        || interval.iter().any(|z| !z.is_finite())
    {
        return Err(Error::InvalidInput(
            "chirality requires matching finite geometry, polarization 0/1 and axis 0/1/2".into(),
        ));
    }
    let mut value = DMatrix::zeros(3, normal.len());
    let evaluate = |(j, column): (usize, &mut [Complex])| -> Result<()> {
        let coefficients = oriented_chirality_mode::<0>(
            transverse[j],
            normal[j],
            polarizations[j],
            axis,
            interval,
        )?;
        for (out, coefficient) in column.iter_mut().zip(coefficients) {
            *out = coefficient.value;
            if !finite(*out) {
                return Err(Error::InvalidInput("chirality density overflow".into()));
            }
        }
        Ok(())
    };
    if normal.len() >= 1024 {
        value
            .as_mut_slice()
            .par_chunks_mut(3)
            .enumerate()
            .try_for_each(evaluate)?;
    } else {
        value
            .as_mut_slice()
            .chunks_mut(3)
            .enumerate()
            .try_for_each(evaluate)?;
    }
    Ok((
        value,
        OrientedChiralityResidual {
            transverse,
            normal,
            polarizations,
            axis,
            interval,
        },
    ))
}

impl OrientedChiralityResidual {
    /// Compact output dimensions.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (3, self.normal.len())
    }

    /// Recompute and contract six local real derivatives per mode.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<OrientedChiralityGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput("invalid chirality cotangent".into()));
        }
        let evaluate = |j: usize| -> Result<[f64; 6]> {
            let coefficients = oriented_chirality_mode::<6>(
                self.transverse[j],
                self.normal[j],
                self.polarizations[j],
                self.axis,
                self.interval,
            )?;
            let g: [f64; 6] = std::array::from_fn(|p| {
                coefficients
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (cotangent[(i, j)].conj() * c.derivative[p]).re)
                    .sum()
            });
            if g.iter().any(|v| !v.is_finite()) {
                return Err(Error::InvalidInput("chirality derivative overflow".into()));
            }
            Ok(g)
        };
        let local: Vec<_> = if self.normal.len() >= 1024 {
            (0..self.normal.len())
                .into_par_iter()
                .map(evaluate)
                .collect::<Result<_>>()?
        } else {
            (0..self.normal.len())
                .map(evaluate)
                .collect::<Result<_>>()?
        };
        let mut gradient = OrientedChiralityGradient {
            transverse: Vec::with_capacity(local.len()),
            normal: Vec::with_capacity(local.len()),
            interval: [0.0; 2],
        };
        for g in local {
            gradient.transverse.push([g[0], g[1]]);
            gradient.normal.push(Complex::new(g[2], g[3]));
            gradient.interval[0] += g[4];
            gradient.interval[1] += g[5];
        }
        Ok(gradient)
    }
}

fn internal_operator(lower: MatRef<'_, Complex>, upper: MatRef<'_, Complex>) -> DMatrix<Complex> {
    let mut value = -product_views(lower, upper);
    for i in 0..value.nrows() {
        value[(i, i)] += 1.0;
    }
    value
}

/// Internal-field solve for specified incident amplitudes, retaining one LU.
#[derive(Debug)]
pub struct IlluminationResidual {
    lower: [StoredBlock; 4],
    upper: [StoredBlock; 4],
    incoming: [DMatrix<Complex>; 2],
    solve: SolveResidual,
    down: DMatrix<Complex>,
}

/// Illuminate a pair of stacks. Returns outgoing up/down and internal up/down.
/// Each input and output matrix has one column per independent illumination.
pub fn illuminate(
    lower: Blocks,
    upper: Blocks,
    incoming: [DMatrix<Complex>; 2],
) -> Result<([DMatrix<Complex>; 4], IlluminationResidual)> {
    illuminate_stored(
        lower.map(StoredBlock::from),
        upper.map(StoredBlock::from),
        incoming,
    )
}

/// Illuminate owned blocks in either memory order, preserving their buffers for reverse.
pub fn illuminate_stored(
    lower: [StoredBlock; 4],
    upper: [StoredBlock; 4],
    incoming: [DMatrix<Complex>; 2],
) -> Result<([DMatrix<Complex>; 4], IlluminationResidual)> {
    let computed = illumination_fields(
        lower.each_ref().map(StoredBlock::view),
        upper.each_ref().map(StoredBlock::view),
        incoming.each_ref().map(view),
    )?;
    Ok(illumination_result(lower, upper, incoming, computed))
}

/// Snapshot borrowed matrices in their physical order, then retain them for reverse.
pub fn illuminate_borrowed(
    lower: [MatRef<'_, Complex>; 4],
    upper: [MatRef<'_, Complex>; 4],
    incoming: [DMatrix<Complex>; 2],
) -> Result<([DMatrix<Complex>; 4], IlluminationResidual)> {
    let (lower, upper) = if lower[0].nrows() >= 512 {
        rayon::join(
            || lower.map(StoredBlock::copy_view),
            || upper.map(StoredBlock::copy_view),
        )
    } else {
        (
            lower.map(StoredBlock::copy_view),
            upper.map(StoredBlock::copy_view),
        )
    };
    illuminate_stored(lower, upper, incoming)
}

fn illumination_result(
    lower: [StoredBlock; 4],
    upper: [StoredBlock; 4],
    incoming: [DMatrix<Complex>; 2],
    computed: InternalFields,
) -> ([DMatrix<Complex>; 4], IlluminationResidual) {
    let (top, bottom, down, solve) = computed;
    let value = [top, bottom, solve.value.clone(), down.clone()];
    (
        value,
        IlluminationResidual {
            lower,
            upper,
            incoming,
            solve,
            down,
        },
    )
}

/// Illuminate borrowed scattering blocks without retaining an adjoint context.
/// Inputs can be row- or column-major; only the operator and thin fields are owned.
pub fn illuminate_forward(
    lower: [MatRef<'_, Complex>; 4],
    upper: [MatRef<'_, Complex>; 4],
    incoming: [MatRef<'_, Complex>; 2],
) -> Result<[DMatrix<Complex>; 4]> {
    let (top, bottom, down, solve) = illumination_fields(lower, upper, incoming)?;
    Ok([top, bottom, solve.value, down])
}

type InternalFields = (
    DMatrix<Complex>,
    DMatrix<Complex>,
    DMatrix<Complex>,
    SolveResidual,
);

fn illumination_fields(
    lower: [MatRef<'_, Complex>; 4],
    upper: [MatRef<'_, Complex>; 4],
    incoming: [MatRef<'_, Complex>; 2],
) -> Result<InternalFields> {
    let n = lower[0].nrows();
    let p = incoming[0].ncols();
    let nonfinite = |a: &MatRef<'_, Complex>| {
        let a = if a.row_stride() == 1 {
            *a
        } else {
            a.transpose()
        };
        if let Some(contiguous) = a.try_as_col_major() {
            // A full contiguous scan vectorizes; early exit per scalar does not.
            (0..a.ncols()).any(|j| {
                contiguous
                    .col(j)
                    .as_slice()
                    .iter()
                    .fold(false, |bad, &z| bad | !finite(z))
            })
        } else {
            (0..a.ncols()).any(|j| (0..a.nrows()).any(|i| !finite(a[(i, j)])))
        }
    };
    if n == 0
        || p == 0
        || lower.iter().chain(&upper).any(|a| a.shape() != (n, n))
        || incoming.iter().any(|a| a.shape() != (n, p))
        || if n >= 256 {
            lower
                .par_iter()
                .chain(upper.par_iter())
                .chain(incoming.par_iter())
                .any(nonfinite)
        } else {
            lower.iter().chain(&upper).chain(&incoming).any(nonfinite)
        }
    {
        return Err(Error::InvalidInput(
            "require matching finite S matrices and mode-by-illumination inputs".into(),
        ));
    }
    let direct = product_views(upper[3], incoming[1]);
    let rhs = product_views(lower[0], incoming[0]) + product_views(lower[1], view(&direct));
    let operator = internal_operator(lower[1], upper[2]);
    let solve = linalg::solve_owned(operator, rhs)?;
    let down = product_views(upper[2], view(&solve.value)) + direct;
    let top = product_views(upper[0], view(&solve.value)) + product_views(upper[1], incoming[1]);
    let bottom = product_views(lower[2], incoming[0]) + product_views(lower[3], view(&down));
    if top
        .iter()
        .chain(bottom.iter())
        .chain(down.iter())
        .any(|&z| !finite(z))
    {
        return Err(Error::Singular);
    }
    Ok((top, bottom, down, solve))
}

impl IlluminationResidual {
    /// Mode and independent-illumination counts.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.solve.value.shape()
    }

    /// Return lower/upper S matrices and incoming up/down amplitude cotangents.
    pub fn pullback(
        self,
        g: &[DMatrix<Complex>; 4],
    ) -> Result<(Blocks, Blocks, [DMatrix<Complex>; 2])> {
        if g.iter()
            .any(|a| a.shape() != self.shape() || a.iter().any(|&z| !finite(z)))
        {
            return Err(Error::InvalidInput(
                "invalid field coefficient cotangent".into(),
            ));
        }
        let down = &g[3] + product_adjoint_left_view(self.lower[3].view(), view(&g[1]));
        let up = &g[2]
            + product_adjoint_left_view(self.upper[0].view(), view(&g[0]))
            + product_adjoint_left_view(self.upper[2].view(), view(&down));
        let rhs = self.solve.adjoint_rhs(up)?;
        let direct = down + product_adjoint_left_view(self.lower[1].view(), view(&rhs));
        let incoming = [
            product_adjoint_left_view(self.lower[2].view(), view(&g[1]))
                + product_adjoint_left_view(self.lower[0].view(), view(&rhs)),
            product_adjoint_left_view(self.upper[1].view(), view(&g[0]))
                + product_adjoint_left_view(self.upper[3].view(), view(&direct)),
        ];
        // The primal blocks are dead after the amplitude pullback. Overwrite
        // them with rank-P gradients instead of allocating eight dense matrices.
        let mut lower = self.lower.map(StoredBlock::into_buffer);
        let mut upper = self.upper.map(StoredBlock::into_buffer);
        for ((output, left), right) in lower.iter_mut().zip([&rhs, &rhs, &g[1], &g[1]]).zip([
            &self.incoming[0],
            &self.down,
            &self.incoming[0],
            &self.down,
        ]) {
            product_adjoint_right_into(output, left, right);
        }
        for ((output, left), right) in upper.iter_mut().zip([&g[0], &g[0], &direct, &direct]).zip([
            &self.solve.value,
            &self.incoming[1],
            &self.solve.value,
            &self.incoming[1],
        ]) {
            product_adjoint_right_into(output, left, right);
        }
        Ok((lower, upper, incoming))
    }
}

/// Transfer-matrix solve for periodic repetition of an S matrix.
#[derive(Debug)]
pub struct PeriodicResidual {
    top: DMatrix<Complex>,
    reflection: DMatrix<Complex>,
    solve: SolveResidual,
}

/// Convert scattering blocks to the transfer matrix used for periodic bands.
pub fn periodic(blocks: Blocks) -> Result<(DMatrix<Complex>, PeriodicResidual)> {
    let n = dimension(&blocks)?;
    let [up, reflection_up, reflection_down, down] = blocks;
    let mut top = DMatrix::zeros(n, 2 * n);
    top.columns_mut(0, n).copy_from(&up);
    top.columns_mut(n, n).copy_from(&reflection_up);
    let mut rhs = -product(&reflection_down, &top);
    for i in 0..n {
        rhs[(i, n + i)] += 1.0;
    }
    let solve = linalg::solve_owned(down, rhs)?;
    let mut value = DMatrix::zeros(2 * n, 2 * n);
    value.rows_mut(0, n).copy_from(&top);
    value.rows_mut(n, n).copy_from(&solve.value);
    Ok((
        value,
        PeriodicResidual {
            top,
            reflection: reflection_down,
            solve,
        },
    ))
}

impl PeriodicResidual {
    /// Number of plane-wave modes in each propagation direction.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.top.nrows()
    }

    /// Pull back a transfer-matrix cotangent into all four scattering blocks.
    pub fn pullback(self, g: &DMatrix<Complex>) -> Result<Blocks> {
        let n = self.dimension();
        if g.shape() != (2 * n, 2 * n) || g.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput(
                "invalid periodic transfer cotangent".into(),
            ));
        }
        let (down, rhs) = self.solve.pullback(g.rows(n, n).into_owned())?;
        let reflection = -product_adjoint_right(&rhs, &self.top);
        let top = g.rows(0, n) - product_adjoint_left(&self.reflection, &rhs);
        Ok([
            top.columns(0, n).into_owned(),
            top.columns(n, n).into_owned(),
            reflection,
            down,
        ])
    }
}

/// Periodic Bloch modes and the native transfer/eigensystem reverse contexts.
#[derive(Debug)]
pub struct BandResidual {
    periodic: PeriodicResidual,
    eigen: linalg::EigenResidual,
    period: f64,
    /// Normal Bloch wavenumbers, using the principal complex logarithm.
    pub wavenumbers: Vec<Complex>,
}

/// Bloch modes of a periodically repeated S matrix with positive repeat distance.
pub fn bands(blocks: Blocks, period: f64) -> Result<BandResidual> {
    if !period.is_finite() || period <= 0.0 {
        return Err(Error::InvalidInput(
            "band period must be finite and positive".into(),
        ));
    }
    let (transfer, periodic) = periodic(blocks)?;
    let eigen = linalg::eig(&transfer)?;
    let wavenumbers: Vec<_> = eigen
        .values
        .iter()
        .map(|v| -Complex::i() * v.ln() / period)
        .collect();
    if wavenumbers.iter().any(|&z| !finite(z)) {
        return Err(Error::SpecialFunction(
            "zero or non-finite Bloch multiplier".into(),
        ));
    }
    Ok(BandResidual {
        periodic,
        eigen,
        period,
        wavenumbers,
    })
}

impl BandResidual {
    /// Unit right Bloch eigenvectors, with the largest component real positive.
    #[must_use]
    pub fn vectors(&self) -> &DMatrix<Complex> {
        &self.eigen.vectors
    }

    /// Return S-matrix and repeat-distance cotangents; logarithm branch is fixed.
    pub fn pullback(
        self,
        wavenumbers: &[Complex],
        vectors: DMatrix<Complex>,
    ) -> Result<(Blocks, f64)> {
        if wavenumbers.len() != self.wavenumbers.len() || wavenumbers.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput(
                "invalid Bloch wavenumber cotangent".into(),
            ));
        }
        let period = -wavenumbers
            .iter()
            .zip(&self.wavenumbers)
            .map(|(g, k)| (g.conj() * k).re)
            .sum::<f64>()
            / self.period;
        let values: Vec<_> = wavenumbers
            .iter()
            .zip(&self.eigen.values)
            .map(|(g, l)| g * ratio(-Complex::i() / self.period, *l).conj())
            .collect();
        let matrix = self.eigen.pullback(&values, vectors)?;
        Ok((self.periodic.pullback(&matrix)?, period))
    }
}

/// Radiation of an effective periodic multipole response into plane-wave ports.
#[derive(Clone, Debug)]
pub struct ArrayResidual {
    response: DMatrix<Complex>,
    channels: Blocks,
    scattered: [DMatrix<Complex>; 2],
    /// Four plane-wave scattering blocks, including direct transmission.
    pub value: Blocks,
}

/// Compose an effective response with incident/emitted, up/down channel arrays.
pub fn from_array(response: DMatrix<Complex>, channels: Blocks) -> Result<ArrayResidual> {
    let d = response.nrows();
    let c = channels[0].ncols();
    if d == 0
        || !response.is_square()
        || c == 0
        || response.iter().any(|&v| !finite(v))
        || channels
            .iter()
            .any(|a| a.shape() != (d, c) || a.iter().any(|&v| !finite(v)))
    {
        return Err(Error::InvalidInput(
            "require a finite square response and four matching multipole-by-plane channel arrays"
                .into(),
        ));
    }
    let scattered = std::array::from_fn(|side| product(&response, &channels[side]));
    let mut value =
        std::array::from_fn(|b| product(&channels[2 + b / 2].transpose(), &scattered[b % 2]));
    for block in [0, 3] {
        for i in 0..c {
            value[block][(i, i)] += 1.0;
        }
    }
    dimension(&value)
        .map_err(|_| Error::SpecialFunction("non-finite array scattering matrix".into()))?;
    Ok(ArrayResidual {
        response,
        channels,
        scattered,
        value,
    })
}
impl ArrayResidual {
    /// Effective-response and four-channel cotangents, consuming the residual.
    pub fn pullback(self, g: &Blocks) -> Result<(DMatrix<Complex>, Blocks)> {
        let c = dimension(g)?;
        if c != self.value[0].nrows() {
            return Err(Error::InvalidInput(
                "invalid array scattering cotangent shape".into(),
            ));
        }
        let mut response = DMatrix::zeros(self.response.nrows(), self.response.ncols());
        let mut channels: Blocks =
            std::array::from_fn(|_| DMatrix::zeros(self.response.nrows(), c));
        for side in 0..2 {
            let adjoint = product(&self.channels[2].conjugate(), &g[side])
                + product(&self.channels[3].conjugate(), &g[2 + side]);
            response += product(&adjoint, &self.channels[side].adjoint());
            channels[side] = product(&self.response.adjoint(), &adjoint);
            channels[2 + side] = (product(&g[2 * side], &self.scattered[0].adjoint())
                + product(&g[2 * side + 1], &self.scattered[1].adjoint()))
            .transpose();
        }
        Ok((response, channels))
    }
}

fn dimension(blocks: &Blocks) -> Result<usize> {
    let n = blocks[0].nrows();
    if n == 0
        || blocks
            .iter()
            .any(|m| m.shape() != (n, n) || m.iter().any(|&z| !finite(z)))
    {
        return Err(Error::InvalidInput(
            "S matrices require four finite, equally sized nonempty square blocks".into(),
        ));
    }
    Ok(n)
}

/// Retained internal fields and LU of one Redheffer composition.
#[derive(Clone, Debug)]
pub struct StackResidual {
    lower: Blocks,
    upper: Blocks,
    up: DMatrix<Complex>,
    down: DMatrix<Complex>,
    lu: Lu,
    /// Coupled scattering blocks.
    pub value: Blocks,
}

/// Place `upper` above `lower`, eliminating their internal incident fields.
pub fn add(lower: Blocks, upper: Blocks) -> Result<StackResidual> {
    let n = dimension(&lower)?;
    if dimension(&upper)? != n {
        return Err(Error::InvalidInput("S matrix dimensions must match".into()));
    }
    let lu = Lu::new(internal_operator(view(&lower[1]), view(&upper[2])))?;
    let mut up = DMatrix::zeros(n, 2 * n);
    up.columns_mut(0, n).copy_from(&lower[0]);
    up.columns_mut(n, n)
        .copy_from(&product(&lower[1], &upper[3]));
    lu.solve_in_place(view_mut(&mut up));
    let mut down = product(&upper[2], &up);
    down.columns_mut(n, n).add_assign(&upper[3]);
    let mut top = product(&upper[0], &up);
    top.columns_mut(n, n).add_assign(&upper[1]);
    let mut bottom = product(&lower[3], &down);
    bottom.columns_mut(0, n).add_assign(&lower[2]);
    let value = [
        top.columns(0, n).into_owned(),
        top.columns(n, n).into_owned(),
        bottom.columns(0, n).into_owned(),
        bottom.columns(n, n).into_owned(),
    ];
    dimension(&value).map_err(|_| Error::Singular)?;
    Ok(StackResidual {
        lower,
        upper,
        up,
        down,
        lu,
        value,
    })
}

use std::ops::AddAssign;

use crate::jet::Jet;

fn fresnel_values<const N: usize>(
    ks: [[Jet<N>; 2]; 2],
    kz: [[Jet<N>; 2]; 2],
    z: [Jet<N>; 2],
) -> [[[[Jet<N>; 2]; 2]; 2]; 2] {
    let ap = std::array::from_fn::<_, 2, _>(|i| ks[i][0] * kz[i][1] + ks[i][1] * kz[i][0]);
    let am = std::array::from_fn::<_, 2, _>(|i| ks[i][0] * kz[i][1] - ks[i][1] * kz[i][0]);
    let bp = ks[0][1] * kz[1][1] + ks[1][1] * kz[0][1];
    let cp = ks[0][0] * kz[1][0] + ks[1][0] * kz[0][0];
    let zd = z[0] - z[1];
    let zp = 4.0 * z[0] * z[1];
    let pref = 1.0 / (zd * zd * ap[0] * ap[1] + zp * bp * cp);
    let mut res = [[[[Jet::default(); 2]; 2]; 2]; 2];
    for i in 0..2 {
        let j = 1 - i;
        res[i][i][0][0] = (z[0] + z[1]) * ks[j][0] * kz[i][0] * bp * z[j] * 4.0 * pref;
        res[i][i][0][1] = -(z[i] - z[j])
            * ks[j][0]
            * kz[i][1]
            * (ks[j][1] * kz[i][0] - ks[i][0] * kz[j][1])
            * z[j]
            * 4.0
            * pref;
        res[i][i][1][0] = -(z[i] - z[j])
            * ks[j][1]
            * kz[i][0]
            * (ks[j][0] * kz[i][1] - ks[i][1] * kz[j][0])
            * z[j]
            * 4.0
            * pref;
        res[i][i][1][1] = (z[0] + z[1]) * ks[j][1] * kz[i][1] * cp * z[j] * 4.0 * pref;
        res[j][i][0][0] = (-zd * zd * am[i] * ap[j]
            - zp * bp * (ks[i][0] * kz[j][0] - ks[j][0] * kz[i][0]))
            * pref;
        res[j][i][0][1] = -2.0 * (z[j] * z[j] - z[i] * z[i]) * ks[i][0] * kz[i][1] * ap[j] * pref;
        res[j][i][1][0] = -2.0 * (z[j] * z[j] - z[i] * z[i]) * ks[i][1] * kz[i][0] * ap[j] * pref;
        res[j][i][1][1] = (zd * zd * am[i] * ap[j]
            - zp * cp * (ks[i][1] * kz[j][1] - ks[j][1] * kz[i][1]))
            * pref;
    }
    res
}

/// Chiral planar-interface residual, with no retained parameter Jacobian.
#[derive(Clone, Debug)]
pub struct FresnelResidual {
    ks: [[Complex; 2]; 2],
    kz: [[Complex; 2]; 2],
    z: [Complex; 2],
    /// Four helicity scattering blocks, with polarization order (0,1).
    pub value: Blocks,
}

/// Planar interface from full and axial wavenumbers and impedances (below, above).
pub fn fresnel(
    ks: [[Complex; 2]; 2],
    kz: [[Complex; 2]; 2],
    z: [Complex; 2],
) -> Result<FresnelResidual> {
    if ks
        .iter()
        .flatten()
        .chain(kz.iter().flatten())
        .chain(z.iter())
        .any(|&v| !finite(v))
    {
        return Err(Error::InvalidInput("Fresnel inputs must be finite".into()));
    }
    let values = fresnel_values::<0>(
        ks.map(|r| r.map(Jet::constant)),
        kz.map(|r| r.map(Jet::constant)),
        z.map(Jet::constant),
    );
    let value =
        std::array::from_fn(|b| DMatrix::from_fn(2, 2, |i, j| values[b / 2][b % 2][i][j].value));
    dimension(&value).map_err(|_| Error::Singular)?;
    Ok(FresnelResidual { ks, kz, z, value })
}

/// Cotangents of full wavenumbers, axial wavenumbers, and impedances.
pub type FresnelGradient = ([[Complex; 2]; 2], [[Complex; 2]; 2], [Complex; 2]);

impl FresnelResidual {
    /// Contract all ten complex input derivatives under the real Hermitian pairing.
    pub fn pullback(self, g: &Blocks) -> Result<FresnelGradient> {
        if dimension(g)? != 2 {
            return Err(Error::InvalidInput(
                "Fresnel cotangent blocks must be 2 by 2".into(),
            ));
        }
        let ks = std::array::from_fn(|i| {
            std::array::from_fn(|j| Jet::<10>::variable(self.ks[i][j], 2 * i + j))
        });
        let kz = std::array::from_fn(|i| {
            std::array::from_fn(|j| Jet::variable(self.kz[i][j], 4 + 2 * i + j))
        });
        let z = std::array::from_fn(|i| Jet::variable(self.z[i], 8 + i));
        let values = fresnel_values(ks, kz, z);
        let mut result = [Complex::default(); 10];
        for (b, block) in g.iter().enumerate() {
            for i in 0..2 {
                for j in 0..2 {
                    for (gradient, d) in
                        result.iter_mut().zip(values[b / 2][b % 2][i][j].derivative)
                    {
                        *gradient += block[(i, j)] * d.conj();
                    }
                }
            }
        }
        Ok((
            std::array::from_fn(|i| std::array::from_fn(|j| result[2 * i + j])),
            std::array::from_fn(|i| std::array::from_fn(|j| result[4 + 2 * i + j])),
            [result[8], result[9]],
        ))
    }
}

type InterfaceMatrix = nalgebra::SMatrix<Complex, 4, 4>;
type BoundaryJets<const N: usize> = [[Jet<N>; 4]; 4];

pub(crate) fn normal_component<const N: usize>(k: Jet<N>, q: [Jet<N>; 2]) -> Result<Jet<N>> {
    let mut normal = (k * k - q[0] * q[0] - q[1] * q[1]).sqrt();
    if normal.value == Complex::default() {
        return Err(Error::InvalidInput(
            "interface at exact diffraction threshold requires a limiting formulation".into(),
        ));
    }
    if normal.value.im < 0.0 || (normal.value.im == 0.0 && normal.value.re < 0.0) {
        normal = -normal;
    }
    Ok(normal)
}

fn port_boundary<const N: usize>(
    ks: [Jet<N>; 2],
    z: Jet<N>,
    q: [Jet<N>; 2],
    axis: usize,
) -> Result<[[[Jet<N>; 4]; 2]; 2]> {
    let mut waves = [[[Jet::default(); 4]; 2]; 2];
    let a = (axis + 1) % 3;
    let b = (axis + 2) % 3;
    for (pol, &wave_number) in ks.iter().enumerate() {
        let normal = normal_component(wave_number, q)?;
        for (side, polarizations) in waves.iter_mut().enumerate() {
            let mut vector = [Jet::default(); 3];
            vector[a] = q[0];
            vector[b] = q[1];
            vector[axis] = if side == 0 { normal } else { -normal };
            let e = crate::plane::polarization_from_inputs(vector, u8::from(pol != 0))?;
            let impedance = -Complex::i() * (if pol == 0 { -1.0 } else { 1.0 }) / z;
            polarizations[pol] = [e[a], e[b], impedance * e[a], impedance * e[b]];
        }
    }
    Ok(waves)
}

fn interface_boundary<const N: usize>(
    ks: [[Jet<N>; 2]; 2],
    z: [Jet<N>; 2],
    q: [Jet<N>; 2],
    axis: usize,
) -> Result<(BoundaryJets<N>, BoundaryJets<N>)> {
    let waves = [
        port_boundary(ks[0], z[0], q, axis)?,
        port_boundary(ks[1], z[1], q, axis)?,
    ];
    let lhs = std::array::from_fn(|row| {
        std::array::from_fn(|col| {
            if col < 2 {
                waves[1][0][col][row]
            } else {
                -waves[0][1][col - 2][row]
            }
        })
    });
    let rhs = std::array::from_fn(|row| {
        std::array::from_fn(|col| {
            if col < 2 {
                waves[0][0][col][row]
            } else {
                -waves[1][1][col - 2][row]
            }
        })
    });
    Ok((lhs, rhs))
}

/// Cartesian tangential-field matching for an interface with any coordinate normal.
#[derive(Debug)]
pub struct InterfaceResidual {
    ks: [[Complex; 2]; 2],
    z: [Complex; 2],
    q: [f64; 2],
    axis: usize,
    inverse: InterfaceMatrix,
    /// Four helicity scattering blocks, with polarization order (0,1).
    pub value: Blocks,
}

/// Interface from wavenumbers/impedances (below, above) and cyclic transverse components.
pub fn interface(
    ks: [[Complex; 2]; 2],
    z: [Complex; 2],
    q: [f64; 2],
    axis: usize,
) -> Result<InterfaceResidual> {
    if axis > 2
        || q.iter().any(|x| !x.is_finite())
        || ks
            .iter()
            .flatten()
            .chain(&z)
            .any(|&v| !finite(v) || v == Complex::default())
    {
        return Err(Error::InvalidInput("interface requires finite transverse components, nonzero finite wavenumbers/impedances, and a Cartesian normal".into()));
    }
    let (lhs, rhs) = interface_boundary::<0>(
        ks.map(|k| k.map(Jet::constant)),
        z.map(Jet::constant),
        q.map(Jet::constant),
        axis,
    )?;
    let matrix = InterfaceMatrix::from_fn(|i, j| lhs[i][j].value);
    let inverse = matrix.lu().try_inverse().ok_or(Error::Singular)?;
    let result = inverse * InterfaceMatrix::from_fn(|i, j| rhs[i][j].value);
    let value = std::array::from_fn(|block| {
        DMatrix::from_fn(2, 2, |i, j| {
            result[(2 * (block / 2) + i, 2 * (block % 2) + j)]
        })
    });
    dimension(&value).map_err(|_| Error::Singular)?;
    Ok(InterfaceResidual {
        ks,
        z,
        q,
        axis,
        inverse,
        value,
    })
}

/// Interface cotangents: medium wavenumbers, impedances and real transverse components.
pub type InterfaceGradient = ([[Complex; 2]; 2], [Complex; 2], [f64; 2]);
impl InterfaceResidual {
    /// Reuse the 4-by-4 inverse for the implicit solve adjoint; recompute local field derivatives.
    pub fn pullback(self, g: &Blocks, fixed_q: bool) -> Result<InterfaceGradient> {
        if dimension(g)? != 2 {
            return Err(Error::InvalidInput(
                "interface cotangent blocks must be 2 by 2".into(),
            ));
        }
        let cotangent = InterfaceMatrix::from_fn(|i, j| g[2 * (i / 2) + j / 2][(i % 2, j % 2)]);
        let value =
            InterfaceMatrix::from_fn(|i, j| self.value[2 * (i / 2) + j / 2][(i % 2, j % 2)]);
        let adjoint = self.inverse.adjoint() * cotangent;
        let operator = -adjoint * value.adjoint();
        let ks = std::array::from_fn(|i| {
            std::array::from_fn(|j| Jet::<8>::variable(self.ks[i][j], 2 * i + j))
        });
        let z = std::array::from_fn(|i| Jet::variable(self.z[i], 4 + i));
        // At normal incidence the xy-interface blocks depend on q only to second
        // order. Use their zero first derivative instead of an undefined
        // azimuth derivative in the intermediate polarization vectors.
        let fixed_q = fixed_q || (self.axis == 2 && self.q.iter().all(|&q| q == 0.0));
        let q = std::array::from_fn(|i| {
            if fixed_q {
                Jet::constant(self.q[i])
            } else {
                Jet::variable(self.q[i], 6 + i)
            }
        });
        let (lhs, rhs) = interface_boundary(ks, z, q, self.axis)?;
        let mut result = [Complex::default(); 8];
        for i in 0..4 {
            for j in 0..4 {
                for (a, g) in result.iter_mut().enumerate() {
                    *g += operator[(i, j)] * lhs[i][j].derivative[a].conj()
                        + adjoint[(i, j)] * rhs[i][j].derivative[a].conj();
                }
            }
        }
        Ok((
            std::array::from_fn(|i| std::array::from_fn(|j| result[2 * i + j])),
            [result[4], result[5]],
            [result[6].re, result[7].re],
        ))
    }
}

/// Translation of up/down reference planes, retaining only its physical inputs and output.
#[derive(Clone, Debug)]
pub struct PropagationResidual {
    vectors: Vec<[Complex; 3]>,
    distance: [f64; 3],
    /// Diagonal transmission blocks and zero reflection blocks.
    pub value: Blocks,
}

/// Propagate by a Cartesian displacement, given upgoing wavevectors for every mode.
pub fn propagation(vectors: Vec<[Complex; 3]>, distance: [f64; 3]) -> Result<PropagationResidual> {
    if vectors.is_empty()
        || vectors.iter().flatten().any(|&v| !finite(v))
        || distance.iter().any(|r| !r.is_finite())
    {
        return Err(Error::InvalidInput(
            "propagation requires finite nonempty wavevectors and displacement".into(),
        ));
    }
    let n = vectors.len();
    let mut value: Blocks = std::array::from_fn(|_| DMatrix::zeros(n, n));
    for (i, k) in vectors.iter().enumerate() {
        for (block, sign) in [(0, 1.0), (3, -1.0)] {
            value[block][(i, i)] = (Complex::i()
                * (sign * (k[0] * distance[0] + k[1] * distance[1]) + k[2] * distance[2]))
                .exp();
        }
    }
    dimension(&value)?;
    Ok(PropagationResidual {
        vectors,
        distance,
        value,
    })
}

impl PropagationResidual {
    /// Wavevector and displacement cotangents; consumes the native residual.
    pub fn pullback(self, g: &Blocks) -> Result<(Vec<[Complex; 3]>, [f64; 3])> {
        let n = dimension(g)?;
        if n != self.vectors.len() {
            return Err(Error::InvalidInput(
                "propagation cotangent shape mismatch".into(),
            ));
        }
        let mut vectors = vec![[Complex::default(); 3]; n];
        let mut distance = [0.0; 3];
        for (i, k) in self.vectors.iter().enumerate() {
            for (block, sign) in [(0, 1.0), (3, -1.0)] {
                let weight = g[block][(i, i)].conj() * Complex::i() * self.value[block][(i, i)];
                for (a, s) in [sign, sign, 1.0].into_iter().enumerate() {
                    vectors[i][a] += (weight * s * self.distance[a]).conj();
                    distance[a] += (weight * s * k[a]).re;
                }
            }
        }
        Ok((vectors, distance))
    }
}

impl StackResidual {
    /// Input cotangents under `dL = Re(sum(conj(g) * dx))`; consumes the residual.
    pub fn pullback(self, g: &Blocks) -> Result<(Blocks, Blocks)> {
        let n = dimension(g)?;
        if n != self.up.nrows() {
            return Err(Error::InvalidInput(
                "invalid S matrix cotangent shape".into(),
            ));
        }
        let mut top = DMatrix::zeros(n, 2 * n);
        top.columns_mut(0, n).copy_from(&g[0]);
        top.columns_mut(n, n).copy_from(&g[1]);
        let mut bottom = DMatrix::zeros(n, 2 * n);
        bottom.columns_mut(0, n).copy_from(&g[2]);
        bottom.columns_mut(n, n).copy_from(&g[3]);
        let down = product(&self.lower[3].adjoint(), &bottom);
        let mut adjoint =
            product(&self.upper[0].adjoint(), &top) + product(&self.upper[2].adjoint(), &down);
        self.lu.solve_adjoint_in_place(view_mut(&mut adjoint));
        let incident = down + product(&self.lower[1].adjoint(), &adjoint);
        let lower = [
            adjoint.columns(0, n).into_owned(),
            product(&adjoint, &self.down.adjoint()),
            g[2].clone(),
            product(&bottom, &self.down.adjoint()),
        ];
        let upper = [
            product(&top, &self.up.adjoint()),
            g[1].clone(),
            product(&incident, &self.up.adjoint()),
            incident.columns(n, n).into_owned(),
        ];
        Ok((lower, upper))
    }
}

// Compact tangential fields, indexed by port, propagation direction, polarization.
type PowerWaves<const N: usize> = [[[[Jet<N>; 4]; 2]; 2]; 2];
fn power_waves<const N: usize>(
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    q: [f64; 2],
    axis: usize,
    helicity: bool,
    fixed_q: bool,
) -> Result<PowerWaves<N>> {
    let ks: [[Jet<N>; 2]; 2] = std::array::from_fn(|port| {
        std::array::from_fn(|pol| Jet::variable(ks[port][pol], 2 * port + pol))
    });
    let zs: [Jet<N>; 2] = std::array::from_fn(|port| Jet::variable(zs[port], 4 + port));
    let q = std::array::from_fn(|j| {
        if fixed_q {
            Jet::constant(q[j])
        } else {
            Jet::variable(q[j], 6 + j)
        }
    });
    let mut waves = [
        port_boundary(ks[0], zs[0], q, axis)?,
        port_boundary(ks[1], zs[1], q, axis)?,
    ];
    if !helicity {
        for port in &mut waves {
            for side in port {
                let [minus, plus] = *side;
                *side = [
                    std::array::from_fn(|j| (plus[j] - minus[j]) * std::f64::consts::FRAC_1_SQRT_2),
                    std::array::from_fn(|j| (plus[j] + minus[j]) * std::f64::consts::FRAC_1_SQRT_2),
                ];
            }
        }
    }
    Ok(waves)
}

fn power_cross(e: [Complex; 4], h: [Complex; 4]) -> f64 {
    0.5 * (e[0] * h[3].conj() - e[1] * h[2].conj()).re
}
fn power_cross_pullback(
    e: [Complex; 4],
    h: [Complex; 4],
    weight: f64,
) -> ([Complex; 4], [Complex; 4]) {
    (
        [
            0.5 * weight * h[3],
            -0.5 * weight * h[2],
            Complex::default(),
            Complex::default(),
        ],
        [
            Complex::default(),
            Complex::default(),
            -0.5 * weight * e[1],
            0.5 * weight * e[0],
        ],
    )
}
fn power_aggregate(
    waves: &[PowerWaves<0>],
    modes: &[(usize, u8)],
    incident: &DMatrix<Complex>,
    outgoing: &[DMatrix<Complex>; 2],
    beam: usize,
    transmission: usize,
) -> Vec<[[Complex; 4]; 3]> {
    let reflection = 1 - transmission;
    let mut fields = vec![[[Complex::default(); 4]; 3]; waves.len()];
    for (mode, &(group, pol)) in modes.iter().enumerate() {
        for (which, (port, side, amplitude)) in [
            (reflection, transmission, incident[(mode, beam)]),
            (transmission, transmission, outgoing[0][(mode, beam)]),
            (reflection, reflection, outgoing[1][(mode, beam)]),
        ]
        .into_iter()
        .enumerate()
        {
            for (j, field) in fields[group][which].iter_mut().enumerate() {
                *field += waves[group][port][side][usize::from(pol)][j].value * amplitude;
            }
        }
    }
    fields
}

/// Owned transmittance/reflectance inputs; only the illuminated S-matrix column is retained.
#[derive(Debug)]
pub struct TransmissionResidual {
    matrices: [StoredBlock; 2],
    incident: DMatrix<Complex>,
    outgoing: [DMatrix<Complex>; 2],
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    q: Vec<[f64; 2]>,
    modes: Vec<(usize, u8)>,
    axis: usize,
    helicity: bool,
    transmission: usize,
    flux: Vec<f64>,
    /// Rows are transmittance/reflectance; columns are independent illuminations.
    pub value: DMatrix<f64>,
}

/// S-matrix, illumination, port wavenumber/impedance and transverse-wavevector cotangents.
#[derive(Debug)]
pub struct TransmissionGradient {
    /// Four scattering blocks.
    pub matrices: Blocks,
    /// One column per independent illumination.
    pub incident: DMatrix<Complex>,
    /// Port 0/1, helicity 0/1 wavenumbers.
    pub ks: [[Complex; 2]; 2],
    /// Port impedances.
    pub zs: [Complex; 2],
    /// One transverse vector per distinct diffraction direction.
    pub q: Vec<[f64; 2]>,
}

#[allow(clippy::float_cmp, clippy::type_complexity)] // Exact diffraction groups and compact forward intermediates.
fn transmission_evaluate(
    matrices: [MatRef<'_, Complex>; 2],
    incident: &DMatrix<Complex>,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    q: &[[f64; 2]],
    modes: &[(usize, u8)],
    axis: usize,
    helicity: bool,
    transmission: usize,
) -> Result<(DMatrix<f64>, Vec<f64>, [DMatrix<Complex>; 2])> {
    let n = modes.len();
    if n == 0
        || incident.nrows() != n
        || incident.ncols() == 0
        || matrices.iter().any(|m| m.nrows() != n || m.ncols() != n)
        || incident.iter().any(|&z| !finite(z))
        || matrices
            .iter()
            .any(|m| (0..n).any(|i| (0..n).any(|j| !finite(m[(i, j)]))))
        || axis > 2
        || transmission > 1
        || modes
            .iter()
            .any(|&(group, pol)| group >= q.len() || pol > 1)
        || q.iter().flatten().any(|v| !v.is_finite())
        || ks
            .iter()
            .flatten()
            .chain(&zs)
            .any(|&v| !finite(v) || v == Complex::default())
    {
        return Err(Error::InvalidInput("require matching finite square S blocks, illumination columns, port media and valid plane modes".into()));
    }
    if !helicity && ks.iter().any(|k| k[0] != k[1]) {
        return Err(Error::InvalidInput(
            "parity power requires achiral port media".into(),
        ));
    }
    let mut unique = std::collections::HashSet::new();
    if q.iter()
        .any(|q| !unique.insert(q.map(|v| if v == 0.0 { 0 } else { v.to_bits() })))
    {
        return Err(Error::InvalidInput(
            "transverse groups must be distinct; place coherent modes in the same group".into(),
        ));
    }
    let waves = q
        .iter()
        .map(|&q| power_waves::<0>(ks, zs, q, axis, helicity, true))
        .collect::<Result<Vec<_>>>()?;
    let outgoing = matrices.map(|m| product_views(m, view(incident)));
    let sign = if transmission == 0 { 1.0 } else { -1.0 };
    let mut flux = vec![0.0; incident.ncols()];
    let mut value = DMatrix::zeros(2, incident.ncols());
    for beam in 0..incident.ncols() {
        let mut transmitted = 0.0;
        let mut reflected = 0.0;
        for [input, trans, reflect] in
            power_aggregate(&waves, modes, incident, &outgoing, beam, transmission)
        {
            flux[beam] += sign
                * (power_cross(input, input)
                    + power_cross(input, reflect)
                    + power_cross(reflect, input));
            transmitted += sign * power_cross(trans, trans);
            reflected -= sign * power_cross(reflect, reflect);
        }
        if !flux[beam].is_finite() || flux[beam] <= 0.0 {
            return Err(Error::InvalidInput(
                "transmittance requires positive finite incident power flux".into(),
            ));
        }
        value[(0, beam)] = transmitted / flux[beam];
        value[(1, beam)] = reflected / flux[beam];
    }
    if value.iter().any(|v| !v.is_finite()) {
        return Err(Error::SpecialFunction(
            "nonfinite transmission or reflection".into(),
        ));
    }
    Ok((value, flux, outgoing))
}

/// Power transmission/reflection without retaining a reverse tape. Matrix views are
/// the transmission and reflection blocks for the chosen illumination direction.
pub fn transmittance_value(
    matrices: [MatRef<'_, Complex>; 2],
    incident: &DMatrix<Complex>,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    q: &[[f64; 2]],
    modes: &[(usize, u8)],
    axis: usize,
    helicity: bool,
    transmission: usize,
) -> Result<DMatrix<f64>> {
    Ok(transmission_evaluate(
        matrices,
        incident,
        ks,
        zs,
        q,
        modes,
        axis,
        helicity,
        transmission,
    )?
    .0)
}

/// Record analytic power pullbacks, including lossy-medium incident/reflected interference.
pub fn transmittance(
    matrices: [MatRef<'_, Complex>; 2],
    incident: DMatrix<Complex>,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    q: Vec<[f64; 2]>,
    modes: Vec<(usize, u8)>,
    axis: usize,
    helicity: bool,
    transmission: usize,
) -> Result<TransmissionResidual> {
    let (value, flux, outgoing) = transmission_evaluate(
        matrices,
        &incident,
        ks,
        zs,
        &q,
        &modes,
        axis,
        helicity,
        transmission,
    )?;
    Ok(TransmissionResidual {
        matrices: matrices.map(StoredBlock::copy_view),
        incident,
        outgoing,
        ks,
        zs,
        q,
        modes,
        axis,
        helicity,
        transmission,
        flux,
        value,
    })
}

impl TransmissionResidual {
    /// Contract output power cotangents. Fixed q supports exactly normal incidence;
    /// otherwise derivatives preserve the distinct diffraction-group topology.
    pub fn pullback(self, g: &DMatrix<f64>, fixed_q: bool) -> Result<TransmissionGradient> {
        if g.shape() != self.value.shape() || g.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidInput(
                "power cotangent must be finite and match output".into(),
            ));
        }
        let n = self.modes.len();
        let t = self.transmission;
        let r = 1 - t;
        let sign = if t == 0 { 1.0 } else { -1.0 };
        let waves = self
            .q
            .iter()
            .map(|&q| power_waves::<0>(self.ks, self.zs, q, self.axis, self.helicity, true))
            .collect::<Result<Vec<_>>>()?;
        let mut gwaves = vec![[[[[Complex::default(); 4]; 2]; 2]; 2]; self.q.len()];
        let mut incident = DMatrix::zeros(n, self.incident.ncols());
        let mut outgoing = [incident.clone(), incident.clone()];
        for beam in 0..self.incident.ncols() {
            let wt = g[(0, beam)] / self.flux[beam];
            let wr = g[(1, beam)] / self.flux[beam];
            let wi = -(g[(0, beam)] * self.value[(0, beam)] + g[(1, beam)] * self.value[(1, beam)])
                / self.flux[beam];
            let fields =
                power_aggregate(&waves, &self.modes, &self.incident, &self.outgoing, beam, t);
            let mut gradient = vec![[[Complex::default(); 4]; 3]; self.q.len()];
            for (group, f) in fields.iter().enumerate() {
                for (e, h, weight) in [
                    (0, 0, sign * wi),
                    (0, 2, sign * wi),
                    (2, 0, sign * wi),
                    (1, 1, sign * wt),
                    (2, 2, -sign * wr),
                ] {
                    let (ge, gh) = power_cross_pullback(f[e], f[h], weight);
                    for j in 0..4 {
                        gradient[group][e][j] += ge[j];
                        gradient[group][h][j] += gh[j];
                    }
                }
            }
            for (mode, &(group, pol)) in self.modes.iter().enumerate() {
                let amplitudes = [
                    self.incident[(mode, beam)],
                    self.outgoing[0][(mode, beam)],
                    self.outgoing[1][(mode, beam)],
                ];
                for (which, (port, side)) in [(r, t), (t, t), (r, r)].into_iter().enumerate() {
                    let mut amplitude_gradient = Complex::default();
                    for j in 0..4 {
                        let g = gradient[group][which][j];
                        amplitude_gradient +=
                            waves[group][port][side][usize::from(pol)][j].value.conj() * g;
                        gwaves[group][port][side][usize::from(pol)][j] +=
                            amplitudes[which].conj() * g;
                    }
                    if which == 0 {
                        incident[(mode, beam)] += amplitude_gradient;
                    } else {
                        outgoing[which - 1][(mode, beam)] += amplitude_gradient;
                    }
                }
            }
        }
        let mut matrices = std::array::from_fn(|_| DMatrix::zeros(n, n));
        for (which, side) in [t, r].into_iter().enumerate() {
            matrices[2 * side + t] = product_adjoint_right(&outgoing[which], &self.incident);
            incident +=
                product_adjoint_left_view(self.matrices[which].view(), view(&outgoing[which]));
        }
        let mut result = TransmissionGradient {
            matrices,
            incident,
            ks: [[Complex::default(); 2]; 2],
            zs: [Complex::default(); 2],
            q: vec![[0.0; 2]; self.q.len()],
        };
        for (group, &q) in self.q.iter().enumerate() {
            if gwaves[group]
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .all(|&v| v == Complex::default())
            {
                continue;
            }
            let jets = power_waves::<8>(self.ks, self.zs, q, self.axis, self.helicity, fixed_q)?;
            let mut parameters = [Complex::default(); 8];
            for (v, g) in jets
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .zip(gwaves[group].iter().flatten().flatten().flatten())
            {
                for (target, d) in parameters.iter_mut().zip(v.derivative) {
                    *target += g * d.conj();
                }
            }
            for port in 0..2 {
                for pol in 0..2 {
                    result.ks[port][pol] += parameters[2 * port + pol];
                }
                result.zs[port] += parameters[4 + port];
            }
            result.q[group] = [parameters[6].re, parameters[7].re];
        }
        Ok(result)
    }
}

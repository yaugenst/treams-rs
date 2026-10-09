//! Dense linear solves and general complex eigensystems with analytic derivatives, and
//! restarted GMRES. treams-rs extension.
//!
//! Two libraries share the linear algebra, by one rule:
//!
//! - nalgebra's `DMatrix` stores every dense matrix.
//! - faer factors and multiplies dense complex matrices of any size, through this
//!   module: LU solves ([`solve`], `Lu`), singular values ([`svdvals`]), eigensystems
//!   ([`eig`]) and products (`product` and its variants). `view` and `view_mut` lend a
//!   `DMatrix` to faer without a copy.
//! - Real symmetric eigenproblems use nalgebra's `SymmetricEigen`: `special::wigner`
//!   diagonalizes the tridiagonal generator of the small-d matrix, of size 2l + 1.
//! - Fixed-size matrices of at most 4 x 4 stay in nalgebra: the 3 x 3 inverse of
//!   `lattice::reciprocal`, the closed-form 2 x 2 inverse of `coeffs::mie`, the 4 x 4
//!   and 2 x 2 inverses of `coeffs::mie_cyl` and the 4 x 4 inverse of
//!   `smatrix::interface`.
#![allow(clippy::indexing_slicing)] // Validated matrix dimensions and eigenvector pivots.

mod gmres;
mod saved;

use std::sync::OnceLock;

pub use gmres::{Convergence, GmresOptions};
pub(crate) use gmres::{gmres, gmres_batch, norm};

use faer::{
    Accum, Conj, Mat, MatMut, MatRef, Spec,
    diag::Diag,
    dyn_stack::{MemBuffer, MemStack, StackReq},
    linalg::{
        evd,
        lu::partial_pivoting::{factor, solve as lu_solve},
        matmul::matmul,
        svd,
    },
    perm::Perm,
    traits::Conjugate,
};
use nalgebra::DMatrix;

use crate::{
    Complex, Error, Result,
    numerics::{self, finite, ratio},
};

/// Column-major faer view of a nalgebra matrix.
pub(crate) fn view(matrix: &DMatrix<Complex>) -> MatRef<'_, Complex> {
    MatRef::from_column_major_slice(matrix.as_slice(), matrix.nrows(), matrix.ncols())
}

/// Mutable column-major faer view of a nalgebra matrix.
pub(crate) fn view_mut(matrix: &mut DMatrix<Complex>) -> MatMut<'_, Complex> {
    let (rows, cols) = matrix.shape();
    MatMut::from_column_major_slice_mut(matrix.as_mut_slice(), rows, cols)
}

/// `out = left right` on the treams-rs pool ([`crate::threads::product`]). Pass
/// `.adjoint()` views for conjugate-transposed factors; faer conjugates them inside
/// the product kernel.
pub(crate) fn product_into<L, R>(
    out: MatMut<'_, Complex>,
    left: MatRef<'_, L>,
    right: MatRef<'_, R>,
) where
    L: Conjugate<Canonical = Complex>,
    R: Conjugate<Canonical = Complex>,
{
    let (m, n, k) = (out.nrows(), out.ncols(), left.ncols());
    crate::threads::product(m, n, k, |par| {
        matmul(
            out,
            Accum::Replace,
            left,
            right,
            Complex::new(1.0, 0.0),
            par,
        );
    });
}

/// `left right` in a new matrix; see [`product_into`].
pub(crate) fn product_views<L, R>(left: MatRef<'_, L>, right: MatRef<'_, R>) -> DMatrix<Complex>
where
    L: Conjugate<Canonical = Complex>,
    R: Conjugate<Canonical = Complex>,
{
    let mut result = DMatrix::zeros(left.nrows(), right.ncols());
    product_into(view_mut(&mut result), left, right);
    result
}

/// `left right`.
pub(crate) fn product(left: &DMatrix<Complex>, right: &DMatrix<Complex>) -> DMatrix<Complex> {
    product_views(view(left), view(right))
}

/// `left rightᴴ`.
pub(crate) fn product_adjoint_right(
    left: &DMatrix<Complex>,
    right: &DMatrix<Complex>,
) -> DMatrix<Complex> {
    product_views(view(left), view(right).adjoint())
}

/// `out = left rightᴴ`, reusing the allocation of `out`.
pub(crate) fn product_adjoint_right_into(
    out: &mut DMatrix<Complex>,
    left: &DMatrix<Complex>,
    right: &DMatrix<Complex>,
) {
    product_into(view_mut(out), view(left), view(right).adjoint());
}

/// Positive diagonal scales for the equivalent operator `R A C`.
///
/// A normal solve scales its right-hand side by `row` and its solution by
/// `column`. A conjugate-transpose solve uses the opposite order. These scales
/// are numerical coordinates, not additional inputs to the implicit derivative.
#[derive(Clone, Debug)]
pub(crate) struct Equilibration {
    /// Left (equation) scale.
    pub(crate) row: Vec<f64>,
    /// Right (unknown) scale.
    pub(crate) column: Vec<f64>,
}

/// Equilibrate severely unbalanced finite square operators in place.
///
/// Small-argument, high-order multipole operators can have unit diagonal and
/// off-diagonal entries exceeding `1e20`. Their unscaled LU loses accuracy even
/// when the equivalent equilibrated system is well conditioned. The scan that
/// rejects non-finite entries also measures the spread of entry sizes. An operator
/// whose nonzero entries, or whose diagonal entries, all lie within a factor of 10 of
/// the largest entry stays unchanged; others get unit row and column maxima, as in
/// LAPACK xGEEQU, then [`balance`] evens out their sums.
pub(crate) fn equilibrate(operator: &mut DMatrix<Complex>) -> Result<Option<Equilibration>> {
    let n = operator.nrows();
    if n == 0 || !operator.is_square() {
        return Err(Error::InvalidInput(
            "require a finite nonempty square operator".into(),
        ));
    }
    // LAPACK xLAQGE uses 0.1 for row/column scale ratios. Waiting until
    // sqrt(epsilon) loses observable digits in moderately unbalanced multipole
    // systems even though a change of coordinates makes them well conditioned.
    let relative_floor = 0.1;
    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0_f64;
    for &z in operator.iter() {
        if !finite(z) {
            return Err(Error::InvalidInput(
                "require a finite nonempty square operator".into(),
            ));
        }
        let magnitude = z.re.abs().max(z.im.abs());
        if magnitude > 0.0 {
            minimum = minimum.min(magnitude);
        }
        maximum = maximum.max(magnitude);
    }
    if maximum > 0.0
        && (minimum / maximum >= relative_floor
            || (0..n).all(|i| {
                let z = operator[(i, i)];
                z.re.abs().max(z.im.abs()) / maximum >= relative_floor
            }))
    {
        return Ok(None);
    }
    // Only potentially unbalanced operators allocate scratch or need another
    // pass. Isolated tiny entries do not by themselves require equilibration.
    let mut row = vec![0.0_f64; n];
    let mut column_min = f64::INFINITY;
    for values in operator.as_slice().chunks_exact(n) {
        let mut column_max = 0.0_f64;
        for (row_max, &z) in row.iter_mut().zip(values) {
            let magnitude = z.re.abs().max(z.im.abs());
            *row_max = row_max.max(magnitude);
            column_max = column_max.max(magnitude);
        }
        column_min = column_min.min(column_max);
    }
    let row_min = row.iter().copied().fold(f64::INFINITY, f64::min);
    if row_min == 0.0 || column_min == 0.0 {
        return Err(Error::Singular);
    }
    if row_min / maximum >= relative_floor && column_min / maximum >= relative_floor {
        return Ok(None);
    }
    if column_min < row_min {
        // Scale the more disparate axis first. Otherwise normalizing large rows
        // can underflow an entire small column before it gets its own scale.
        // Transposition swaps the two measured spreads, so this recurses once.
        operator.transpose_mut();
        let result = equilibrate(operator);
        operator.transpose_mut();
        return result.map(|scales| {
            scales.map(|scales| Equilibration {
                row: scales.column,
                column: scales.row,
            })
        });
    }
    let inverse = |value: f64| {
        value
            .clamp(f64::MIN_POSITIVE, f64::MIN_POSITIVE.recip())
            .recip()
    };
    for value in &mut row {
        *value = inverse(*value);
    }
    let mut column = Vec::with_capacity(n);
    for values in operator.as_mut_slice().chunks_exact_mut(n) {
        let mut column_max = 0.0_f64;
        for (value, &scale) in values.iter_mut().zip(&row) {
            *value *= scale;
            column_max = column_max.max(value.re.abs().max(value.im.abs()));
        }
        let scale = inverse(column_max);
        column.push(scale);
        for value in values {
            *value *= scale;
        }
    }
    balance(operator, &mut row, &mut column)?;
    Ok(Some(Equilibration { row, column }))
}

/// The most sweeps of [`balance`]; dense operators take at most three.
const BALANCE_SWEEPS: usize = 64;

/// Even out the row and column sums of a max-equilibrated operator in place with
/// power-of-two factors, folded into `row` and `column`.
///
/// Row and column maxima do not determine the scales: in `A = D_r H D_c` the largest
/// entry of a row depends on `D_c`, so an off-diagonal entry of `H` can end up as
/// large as its diagonal, partial pivoting may choose it, and a change of units costs
/// digits. Unit magnitude sums determine the scaled operator whenever it has total
/// support (Sinkhorn and Knopp), whatever the units.
///
/// Each sweep moves every column factor, then every row factor, to a power of two that
/// minimizes the convex potential `Σ |a_ij| x_i y_j − Σ ln x_i − Σ ln y_j` along it. The
/// sweeps stop after one that moves no factor by more than two, or after
/// [`BALANCE_SWEEPS`]. A sum that is not normal, or factors beyond `2^±256`, keep the
/// max scales.
fn balance(operator: &mut DMatrix<Complex>, row: &mut [f64], column: &mut [f64]) -> Result<()> {
    let n = operator.nrows();
    let magnitude = |z: &Complex| z.re.abs().max(z.im.abs());
    // Scale `factor` by the power of two that puts its `sum` in `[ln 2, 2 ln 2)`, which
    // minimizes `sum 2^τ − τ ln 2`; `Some(true)` reports a step beyond a factor of two.
    let advance = |factor: &mut f64, sum: f64| {
        let ratio = sum / std::f64::consts::LN_2;
        // The exponent bits alone: the largest power of two at most `ratio`.
        let step = f64::from_bits(ratio.to_bits() & 0x7ff0_0000_0000_0000).recip();
        *factor *= step;
        ratio.is_normal().then_some(!(0.5..=2.0).contains(&step))
    };
    let mut left = numerics::filled(n, 1.0_f64)?;
    let mut right = numerics::filled(n, 1.0_f64)?;
    // Row sums without their left factor.
    let mut partial = numerics::filled(n, 0.0_f64)?;
    for _ in 0..BALANCE_SWEEPS {
        let mut coarse = false;
        partial.fill(0.0);
        for (values, factor) in operator.as_slice().chunks_exact(n).zip(&mut right) {
            let sum: f64 = values
                .iter()
                .zip(&left)
                .map(|(z, &x)| x * magnitude(z))
                .sum();
            let Some(large) = advance(factor, sum * *factor) else {
                return Ok(());
            };
            coarse |= large;
            for (total, z) in partial.iter_mut().zip(values) {
                *total += magnitude(z) * *factor;
            }
        }
        for (factor, &total) in left.iter_mut().zip(&partial) {
            let Some(large) = advance(factor, total * *factor) else {
                return Ok(());
            };
            coarse |= large;
        }
        if !coarse {
            break;
        }
    }
    let bound = 2.0_f64.powi(256);
    let fits = |scales: &[f64], factors: &[f64]| {
        scales.iter().zip(factors).all(|(&scale, &factor)| {
            (bound.recip()..=bound).contains(&factor) && (scale * factor).is_normal()
        })
    };
    if fits(row, &left) && fits(column, &right) {
        for (values, &y) in operator.as_mut_slice().chunks_exact_mut(n).zip(&right) {
            // Both factors lie within 2^±256, so their product scales exactly.
            for (value, &x) in values.iter_mut().zip(&left) {
                *value *= x * y;
            }
        }
        for (scales, factors) in [(row, &left), (column, &right)] {
            for (scale, &factor) in scales.iter_mut().zip(factors) {
                *scale *= factor;
            }
        }
    }
    Ok(())
}

/// `matrix[(i, j)] scales[i] 2^shifts[j]`, rounded once unless subnormal. A shifted
/// column splits each scale into a mantissa and a power of two, and applies the power
/// first when it scales up, so no intermediate product overflows or flushes a
/// subnormal entry.
fn scale_rows(mut matrix: MatMut<'_, Complex>, scales: &[f64], shifts: &[i32]) {
    for (j, &shift) in shifts.iter().enumerate() {
        for (value, &scale) in matrix.as_mut().col_mut(j).iter_mut().zip(scales) {
            if shift == 0 {
                *value *= scale;
            } else {
                let (mantissa, exponent) = libm::frexp(scale);
                let power = exponent + shift;
                let part = |x: f64| {
                    if power > 0 {
                        libm::scalbn(x, power - 1) * (2.0 * mantissa)
                    } else {
                        libm::scalbn(x * mantissa, power)
                    }
                };
                *value = Complex::new(part(value.re), part(value.im));
            }
        }
    }
}

/// The faer workers of an LU factorization or solve with `rows` rows and `columns`
/// right-hand sides, on a pool of `budget` threads.
///
/// Solves with fewer than 32 right-hand sides run on one worker. Recursive LU and
/// triangular solves split the work into panels, one task each. A pool of up to four
/// threads runs whole: the panels stay wide enough for four workers. Larger pools get one worker per 512 rows up to four, one per 2048 rows
/// beyond that, and at most one per 16 right-hand sides, because narrower panels cost
/// more in task overhead than they gain. The constants are tuned up to 8192 rows;
/// `benches/lu_scheduling.rs` times a factorization and its solves for any worker
/// count, to check them on other hardware.
fn lu_threads(rows: usize, columns: usize, budget: usize) -> usize {
    // Narrow solves: faer's parallel triangular solves and products split the inner
    // dimension by worker count, which changes the last bits of the solution, and
    // their O(rows²) work per column gains little from workers.
    if columns < 32 {
        return 1;
    }
    if budget <= 4 {
        return budget;
    }
    (rows / 512)
        .min(4)
        .max(rows / 2048)
        .min(columns / 16)
        .clamp(1, budget)
}

/// The faer workers of an LU call on the treams-rs budget.
fn lu_workers(rows: usize, columns: usize) -> usize {
    // Faer takes no parallel branch here (LU and row swaps below 64 rows, products
    // below its M·N·K threshold, at most 64 right-hand sides in the triangular
    // solves), so the result is the same without the hand-off to the pool.
    if rows < 64 && columns <= 64 {
        return 1;
    }
    lu_threads(rows, columns, crate::threads::current_num_threads())
}

/// A faer workspace of `req`, or [`Error::OutOfMemory`] when the system refuses it or
/// the requirement overflowed.
fn scratch(req: StackReq) -> Result<MemBuffer> {
    MemBuffer::try_new(req).map_err(|_| {
        // dyn-stack marks an overflowed requirement with alignment 0 and size 0; it
        // asked for more than usize::MAX bytes.
        if req.align_bytes() == 0 {
            Error::out_of_memory(usize::MAX, 2)
        } else {
            Error::out_of_memory(req.size_bytes(), 1)
        }
    })
}

/// Packed pivoted LU: overwrite the operator and share its triangular storage.
#[derive(Clone, Debug)]
pub(crate) struct Lu {
    factors: DMatrix<Complex>,
    permutation: Perm<usize>,
    equilibration: Option<Equilibration>,
}

impl Lu {
    pub(crate) fn new(mut factors: DMatrix<Complex>) -> Result<Self> {
        let n = factors.nrows();
        let equilibration = equilibrate(&mut factors)?;
        let mut forward = numerics::filled(n, 0_usize)?;
        let mut inverse = numerics::filled(n, 0_usize)?;
        crate::threads::dense(lu_workers(n, n), |par| -> Result<()> {
            factor::lu_in_place(
                view_mut(&mut factors),
                &mut forward,
                &mut inverse,
                par,
                MemStack::new(&mut scratch(
                    factor::lu_in_place_scratch::<usize, Complex>(n, n, par, Spec::default()),
                )?),
                Spec::default(),
            );
            Ok(())
        })?;
        if factors.diagonal().iter().any(|&z| z == Complex::default()) {
            return Err(Error::Singular);
        }
        Ok(Self {
            factors,
            equilibration,
            permutation: Perm::new_checked(
                forward.into_boxed_slice(),
                inverse.into_boxed_slice(),
                n,
            ),
        })
    }

    /// Solve `A X = B` in place, with `A` the operator given to [`Lu::new`].
    pub(crate) fn solve_in_place(&self, rhs: MatMut<'_, Complex>) -> Result<()> {
        self.solve_with(rhs, false)
    }

    /// Solve `Aᴴ X = B` in place.
    pub(crate) fn solve_adjoint_in_place(&self, rhs: MatMut<'_, Complex>) -> Result<()> {
        self.solve_with(rhs, true)
    }

    // The factors hold `R A C`. A forward solve scales by the row scales, solves
    // and scales by the column scales; an adjoint solve uses the reverse order.
    fn solve_with(&self, mut rhs: MatMut<'_, Complex>, adjoint: bool) -> Result<()> {
        let (n, columns) = (self.factors.nrows(), rhs.ncols());
        let scales = self.equilibration.as_ref().map(|scales| {
            if adjoint {
                (&scales.column, &scales.row)
            } else {
                (&scales.row, &scales.column)
            }
        });
        // Shift a right-hand side by a power of two when a scaled entry could exceed 2^960,
        // which leaves 2^64 for growth in the triangular solves, or fall below 2^-1020: the
        // scales then neither overflow a system whose solution is near the largest float
        // nor flush its small entries. The shift nearest zero keeps the entries normal and
        // below 2^960; for too wide a range, nonzero and finite; failing that, there is
        // none. Others keep their bits.
        let mut shifts = Vec::new();
        if let Some((before, _)) = scales {
            shifts = numerics::filled(columns, 0_i32)?;
            for (j, shift) in shifts.iter_mut().enumerate() {
                let entries = || {
                    (rhs.as_ref().col(j).iter().zip(before))
                        .map(|(z, &scale)| (z.re.abs().max(z.im.abs()), scale))
                        .filter(|&(magnitude, _)| magnitude > 0.0)
                };
                // The extreme scaled entries, zero or infinite where they leave the range.
                let (smallest, largest) = entries()
                    .map(|(magnitude, scale)| magnitude * scale)
                    .fold((f64::INFINITY, 0.0_f64), |(low, high), x| {
                        (low.min(x), high.max(x))
                    });
                if largest > 2.0_f64.powi(960) || smallest < 2.0_f64.powi(-1020) {
                    // Scaled entries lie in [2^(e - 2), 2^e) for the exponent sums e.
                    let (low, high) = entries()
                        .map(|(magnitude, scale)| libm::frexp(magnitude).1 + libm::frexp(scale).1)
                        .fold((i32::MAX, i32::MIN), |(low, high), e| {
                            (low.min(e), high.max(e))
                        });
                    let (floor, ceiling) = (-1020 - low, 960 - high);
                    let (lowest, highest) = (-1072 - low, 1023 - high);
                    *shift = if floor <= ceiling {
                        0.clamp(floor, ceiling)
                    } else if lowest <= highest {
                        0.clamp(lowest, highest)
                    } else {
                        0
                    };
                }
            }
            scale_rows(rhs.as_mut(), before, &shifts);
        }
        let (factors, permutation) = (view(&self.factors), self.permutation.as_ref());
        crate::threads::dense(lu_workers(n, columns), |par| -> Result<()> {
            if adjoint {
                lu_solve::solve_transpose_in_place_with_conj(
                    factors,
                    factors,
                    permutation,
                    Conj::Yes,
                    rhs.as_mut(),
                    par,
                    MemStack::new(&mut scratch(lu_solve::solve_transpose_in_place_scratch::<
                        usize,
                        Complex,
                    >(n, columns, par))?),
                );
            } else {
                lu_solve::solve_in_place(
                    factors,
                    factors,
                    permutation,
                    rhs.as_mut(),
                    par,
                    MemStack::new(&mut scratch(lu_solve::solve_in_place_scratch::<
                        usize,
                        Complex,
                    >(n, columns, par))?),
                );
            }
            Ok(())
        })?;
        if let Some((_, after)) = scales {
            for shift in &mut shifts {
                *shift = -*shift;
            }
            scale_rows(rhs, after, &shifts);
        }
        Ok(())
    }
}

/// What [`solve`] saves for its derivatives: the LU factors and the solution. Both
/// derivative directions solve with the same factors.
#[derive(Clone, Debug)]
pub struct SolveResidual {
    lu: Lu,
    value: DMatrix<Complex>,
}

/// Solve A X = B with pivoted LU; B is a matrix of right-hand sides.
pub fn solve(operator: &DMatrix<Complex>, rhs: DMatrix<Complex>) -> Result<SolveResidual> {
    solve_owned(operator.clone(), rhs)
}

/// Solve while reusing the owned operator buffer for its LU factors.
pub fn solve_owned(operator: DMatrix<Complex>, mut rhs: DMatrix<Complex>) -> Result<SolveResidual> {
    let n = operator.nrows();
    if n == 0
        || !operator.is_square()
        || rhs.nrows() != n
        || rhs.ncols() == 0
        || rhs.iter().any(|&z| !finite(z))
    {
        return Err(Error::InvalidInput(
            "require a finite nonempty square operator and matching right-hand sides".into(),
        ));
    }
    let lu = Lu::new(operator)?;
    lu.solve_in_place(view_mut(&mut rhs))?;
    if rhs.iter().any(|&z| !finite(z)) {
        return Err(Error::Singular);
    }
    Ok(SolveResidual { lu, value: rhs })
}

/// Eigen- and singular-value decompositions run on the calling thread. faer's
/// parallel Hessenberg and bidiagonal reductions split inner products by worker
/// count, so their last bits, and the order of eigenvalues, would follow the
/// budget. On four workers they gained at most 1.3× (eigenvalues) and 1.7×
/// (singular values) at n = 768, and nothing below n = 384.
const DECOMPOSITION: faer::Par = faer::Par::Seq;

/// Thin singular vectors and values. faer's `Svd` reads faer's global parallelism,
/// so the decomposition is called with [`DECOMPOSITION`] instead.
#[derive(Debug)]
struct ThinSvd {
    u: Mat<Complex>,
    s: Diag<Complex>,
    v: Mat<Complex>,
}

/// Reserve decomposition vectors before initializing their entries.
fn faer_zeros(rows: usize, columns: usize) -> Result<Mat<Complex>> {
    let mut matrix = Mat::new();
    matrix
        .try_reserve(rows, columns)
        .map_err(|_| Error::out_of_memory(rows.saturating_mul(columns), size_of::<Complex>()))?;
    matrix.resize_with(rows, columns, |_, _| Complex::default());
    Ok(matrix)
}

fn thin_svd(a: MatRef<'_, Complex>) -> Result<ThinSvd> {
    let (m, n) = a.shape();
    let size = m.min(n);
    let mut factors = ThinSvd {
        u: faer_zeros(m, size)?,
        s: Diag::zeros(size),
        v: faer_zeros(n, size)?,
    };
    let thin = svd::ComputeSvdVectors::Thin;
    let par = DECOMPOSITION;
    svd::svd(
        a,
        factors.s.as_mut(),
        Some(factors.u.as_mut()),
        Some(factors.v.as_mut()),
        par,
        MemStack::new(&mut scratch(svd::svd_scratch::<Complex>(
            m,
            n,
            thin,
            thin,
            par,
            Spec::default(),
        ))?),
        Spec::default(),
    )
    .map_err(|err| Error::NotConverged(format!("singular-value decomposition: {err:?}")))?;
    Ok(factors)
}

/// Eigenvalues and right eigenvectors, called with an explicit `Par` as [`thin_svd`].
fn eigen(a: MatRef<'_, Complex>) -> Result<(Diag<Complex>, Mat<Complex>)> {
    let n = a.nrows();
    let mut values = Diag::zeros(n);
    let mut vectors = faer_zeros(n, n)?;
    let par = DECOMPOSITION;
    evd::evd_cplx(
        a,
        values.as_mut(),
        None,
        Some(vectors.as_mut()),
        par,
        MemStack::new(&mut scratch(evd::evd_scratch::<Complex>(
            n,
            evd::ComputeEigenvectors::No,
            evd::ComputeEigenvectors::Yes,
            par,
            Spec::default(),
        ))?),
        Spec::default(),
    )
    .map_err(|err| Error::NotConverged(format!("eigendecomposition: {err:?}")))?;
    Ok((values, vectors))
}

/// What [`svdvals`] saves for its derivatives: the thin singular vectors and the singular
/// values.
#[derive(Debug)]
pub struct SvdvalsResidual {
    u: Mat<Complex>,
    v: Mat<Complex>,
    values: Vec<f64>,
}

/// Singular values of a finite nonempty rectangular complex matrix.
pub fn svdvals(operator: &DMatrix<Complex>) -> Result<SvdvalsResidual> {
    if operator.is_empty() || operator.iter().any(|&z| !finite(z)) {
        return Err(Error::InvalidInput(
            "SVD requires a finite nonempty matrix".into(),
        ));
    }
    let scale = operator.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let scale = if scale == 0.0 { 1.0 } else { scale };
    let decomposition = thin_svd(view(&operator.map(|z| z / scale)))?;
    let values: Vec<_> = (0..operator.nrows().min(operator.ncols()))
        .map(|i| decomposition.s.as_ref()[i].re * scale)
        .collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFinite("non-finite singular values".into()));
    }
    Ok(SvdvalsResidual {
        u: decomposition.u,
        v: decomposition.v,
        values,
    })
}

impl SvdvalsResidual {
    /// The singular values, in descending order, which the pullback reads.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The shape of the input operator.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.u.nrows(), self.v.nrows())
    }

    /// Singular-value tangents `Re(uᴴ dA v)` for simple positive singular values.
    ///
    /// The ordered individual singular values have no linear derivative at repeated
    /// or zero values. Unlike the pullback of a smooth spectral sum, a pushforward
    /// returns each individual derivative and therefore rejects these cases.
    /// Values within `64 ε` of the largest value's scale are numerically unresolved
    /// and are treated as zero, including roundoff from rank-deficient matrices.
    pub fn pushforward(&self, operator: &DMatrix<Complex>) -> Result<Vec<f64>> {
        if operator.shape() != self.shape() || operator.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput(
                "invalid singular-value operator tangent".into(),
            ));
        }
        let tolerance = 64.0 * f64::EPSILON * self.values[0];
        if self.values.iter().any(|&value| value <= tolerance) {
            return Err(Error::InvalidInput(
                "singular values at zero or below numerical resolution have no pushforward".into(),
            ));
        }
        if self
            .values
            .windows(2)
            .any(|pair| pair[0] - pair[1] <= tolerance)
        {
            return Err(Error::InvalidInput(
                "individual repeated singular values have no pushforward".into(),
            ));
        }
        let moved = product_views(view(operator), self.v.as_ref());
        let u = self.u.as_ref();
        Ok((0..self.values.len())
            .map(|j| {
                (0..u.nrows())
                    .map(|i| (u[(i, j)].conj() * moved[(i, j)]).re)
                    .sum()
            })
            .collect())
    }

    /// Equal weights are supported at repeated positive values; numerically zero
    /// values require numerically zero weights because the singular value itself
    /// is not smooth there. Both tests use relative tolerance `64 ε`.
    pub fn pullback(&self, cotangent: &[f64]) -> Result<DMatrix<Complex>> {
        if cotangent.len() != self.values.len() || cotangent.iter().any(|x| !x.is_finite()) {
            return Err(Error::Derivative(
                crate::DerivativeError::InvalidSingularValueCotangent,
            ));
        }
        let tolerance = 64.0 * f64::EPSILON * self.values[0];
        let weight_tolerance =
            64.0 * f64::EPSILON * cotangent.iter().map(|x| x.abs()).fold(0.0, f64::max);
        for i in 0..cotangent.len() {
            if self.values[i] <= tolerance && cotangent[i].abs() > weight_tolerance {
                return Err(Error::Derivative(
                    crate::DerivativeError::UnresolvedSingularValue,
                ));
            }
            if i > 0
                && self.values[i - 1] > tolerance
                && (self.values[i - 1] - self.values[i]).abs() <= tolerance
                && (cotangent[i - 1] - cotangent[i]).abs()
                    > 64.0 * f64::EPSILON * cotangent[i - 1].abs().max(cotangent[i].abs())
            {
                return Err(Error::Derivative(
                    crate::DerivativeError::UnequalSingularValueWeights,
                ));
            }
        }
        let u = self.u.as_ref();
        let weighted =
            DMatrix::from_fn(u.nrows(), cotangent.len(), |i, j| u[(i, j)] * cotangent[j]);
        Ok(product_views(view(&weighted), self.v.as_ref().adjoint()))
    }
}

/// Cotangents of the inputs of [`solve`] and [`solve_owned`], in the order of their
/// arguments.
#[derive(Clone, Debug)]
pub struct SolveGradient {
    /// Cotangent of the operator `A`.
    pub operator: DMatrix<Complex>,
    /// Cotangent of the right-hand sides `B`.
    pub rhs: DMatrix<Complex>,
}

impl SolveResidual {
    /// The solution `X` of `A X = B`, one column per right-hand side, which the pullback
    /// reads.
    #[must_use]
    pub const fn value(&self) -> &DMatrix<Complex> {
        &self.value
    }

    /// The shape of the solution: unknowns and right-hand sides.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.value.shape()
    }

    /// Solve `A dX = dB - dA X`, reusing the forward LU and the tangent RHS buffer.
    pub fn pushforward(
        &self,
        operator: &DMatrix<Complex>,
        mut rhs: DMatrix<Complex>,
    ) -> Result<DMatrix<Complex>> {
        let n = self.value.nrows();
        if operator.shape() != (n, n)
            || rhs.shape() != self.shape()
            || operator.iter().chain(rhs.iter()).any(|&z| !finite(z))
        {
            return Err(Error::InvalidInput("invalid linear-solve tangent".into()));
        }
        rhs -= product(operator, &self.value);
        self.lu.solve_in_place(view_mut(&mut rhs))?;
        if rhs.iter().any(|&z| !finite(z)) {
            return Err(Error::Singular);
        }
        Ok(rhs)
    }

    pub(crate) fn adjoint_rhs(&self, mut cotangent: DMatrix<Complex>) -> Result<DMatrix<Complex>> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid linear-solve cotangent".into()));
        }
        self.lu.solve_adjoint_in_place(view_mut(&mut cotangent))?;
        Ok(cotangent)
    }

    /// Return operator and right-hand-side cotangents, reusing the forward LU.
    pub fn pullback(&self, cotangent: DMatrix<Complex>) -> Result<SolveGradient> {
        let rhs = self.adjoint_rhs(cotangent)?;
        Ok(SolveGradient {
            operator: -product_adjoint_right(&rhs, &self.value),
            rhs,
        })
    }
}

/// What [`eig`] saves for its derivatives: the right eigensystem, with unit vectors whose
/// largest component is real and positive.
#[derive(Debug)]
pub struct EigResidual {
    values: Vec<Complex>,
    vectors: DMatrix<Complex>,
    pivots: Vec<usize>,
    scale: f64,
    /// Factor the eigenvectors only when a derivative needs their inverse;
    /// every later direction and cotangent shares the same factors.
    vectors_lu: OnceLock<Result<Lu>>,
}

/// General complex eigendecomposition. Repeated eigenvalues are allowed in the primal.
pub fn eig(operator: &DMatrix<Complex>) -> Result<EigResidual> {
    let n = operator.nrows();
    if n == 0 || !operator.is_square() || operator.iter().any(|&z| !finite(z)) {
        return Err(Error::InvalidInput(
            "eigensystem requires a finite nonempty square matrix".into(),
        ));
    }
    let scale = operator.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let scale = if scale == 0.0 { 1.0 } else { scale };
    let (eigenvalues, eigenvectors) = eigen(view(&operator.map(|z| z / scale)))?;
    let values: Vec<_> = (0..n).map(|i| eigenvalues.as_ref()[i] * scale).collect();
    let mut vectors = DMatrix::from_fn(n, n, |i, j| eigenvectors[(i, j)]);
    let mut pivots = Vec::with_capacity(n);
    for mut column in vectors.column_iter_mut() {
        let pivot = (0..n)
            .max_by(|&i, &j| column[i].norm_sqr().total_cmp(&column[j].norm_sqr()))
            .ok_or(Error::Singular)?;
        let phase = column[pivot] / column[pivot].norm();
        let normalization = phase * column.norm();
        column /= normalization;
        pivots.push(pivot);
    }
    if values.iter().chain(vectors.iter()).any(|&z| !finite(z)) {
        return Err(Error::NonFinite("non-finite eigensystem".into()));
    }
    Ok(EigResidual {
        values,
        vectors,
        pivots,
        scale,
        vectors_lu: OnceLock::new(),
    })
}

/// Eigenvalues ordered by real part, then imaginary part. The same eigensystem
/// residual supplies derivatives without imposing an eigenvector phase constraint.
pub fn eigvals(operator: &DMatrix<Complex>) -> Result<EigResidual> {
    let mut residual = eig(operator)?;
    let n = residual.values.len();
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&i, &j| {
        residual.values[i]
            .re
            .total_cmp(&residual.values[j].re)
            .then_with(|| residual.values[i].im.total_cmp(&residual.values[j].im))
    });
    residual.values = order.iter().map(|&j| residual.values[j]).collect();
    residual.vectors = DMatrix::from_fn(n, n, |i, j| residual.vectors[(i, order[j])]);
    residual.pivots = order.iter().map(|&j| residual.pivots[j]).collect();
    Ok(residual)
}

impl EigResidual {
    fn vectors_lu(&self) -> Result<&Lu> {
        self.vectors_lu
            .get_or_init(|| Lu::new(self.vectors.clone()))
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The eigenvalues in the solver's order, which the pullback reads.
    #[must_use]
    pub fn values(&self) -> &[Complex] {
        &self.values
    }

    /// The right eigenvectors, one column per eigenvalue, which the pullback reads.
    #[must_use]
    pub const fn vectors(&self) -> &DMatrix<Complex> {
        &self.vectors
    }

    /// Whether the phase pivot is tied with another largest component.
    fn tied_pivot(&self, column: usize) -> bool {
        let pivot = self.pivots[column];
        let size = self.vectors[(pivot, column)].norm();
        (0..self.values.len()).any(|i| {
            i != pivot
                && (size - self.vectors[(i, column)].norm()).abs() <= 64.0 * f64::EPSILON * size
        })
    }

    /// Express the perturbation in the eigenbasis; its diagonal is the
    /// eigenvalue derivative and its off-diagonal entries determine eigenvectors.
    fn eigenbasis_tangent(&self, operator: &DMatrix<Complex>) -> Result<DMatrix<Complex>> {
        let n = self.values.len();
        if operator.shape() != (n, n) || operator.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid eigensystem tangent".into()));
        }
        for j in 0..n {
            for i in 0..j {
                if (self.values[j] - self.values[i]).norm() <= 64.0 * f64::EPSILON * self.scale {
                    return Err(Error::InvalidInput(
                        "individual eigenmodes have no pushforward at repeated eigenvalues".into(),
                    ));
                }
            }
        }
        let mut moved = product(operator, &self.vectors);
        self.vectors_lu()?.solve_in_place(view_mut(&mut moved))?;
        if moved.iter().any(|&z| !finite(z)) {
            return Err(Error::Singular);
        }
        Ok(moved)
    }

    /// An ordered spectrum switches modes when real parts cross. Only equal
    /// loss weights across a tied group make that change irrelevant.
    fn check_value_order(&self, weights: Option<&[Complex]>) -> Result<()> {
        for j in 0..self.values.len() {
            for i in 0..j {
                let tied = (self.values[j].re - self.values[i].re).abs()
                    <= 64.0 * f64::EPSILON * self.scale;
                let equal_weights = weights.is_some_and(|g| {
                    (g[j] - g[i]).norm() <= 64.0 * f64::EPSILON * g[j].norm().max(g[i].norm())
                });
                if tied && !equal_weights {
                    return Err(Error::InvalidInput(
                        "ordered eigenvalue derivatives require distinct real parts or equal loss weights at ordering ties".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Eigenvalue tangents with no eigenvector phase constraint. Distinct real
    /// parts keep [`eigvals`]' ordering fixed; tied phase pivots are supported.
    pub fn pushforward_values(&self, operator: &DMatrix<Complex>) -> Result<Vec<Complex>> {
        let moved = self.eigenbasis_tangent(operator)?;
        self.check_value_order(None)?;
        Ok(moved.diagonal().iter().copied().collect())
    }

    /// Pullback of [`eigvals`], allowing equal weights at ordering ties.
    pub fn pullback_values(&self, values: &[Complex]) -> Result<DMatrix<Complex>> {
        let n = self.values.len();
        let gradient = self.pullback(values, DMatrix::zeros(n, n))?;
        self.check_value_order(Some(values))?;
        Ok(gradient)
    }

    /// Eigenpair tangents with the same unit norm and real-positive pivot phase as
    /// the primal. Repeated eigenvalues and tied phase pivots have no pushforward of
    /// the complete eigenpair output. Use [`Self::pushforward_values`] when only
    /// eigenvalues are needed.
    pub fn pushforward(
        &self,
        operator: &DMatrix<Complex>,
    ) -> Result<(Vec<Complex>, DMatrix<Complex>)> {
        let n = self.values.len();
        let mut moved = self.eigenbasis_tangent(operator)?;
        if (0..n).any(|j| self.tied_pivot(j)) {
            return Err(Error::InvalidInput(
                "eigenvector pushforward requires a unique largest component; use eigvals for eigenvalues alone".into(),
            ));
        }
        let values: Vec<_> = moved.diagonal().iter().copied().collect();
        // E = V⁻¹ dA V; divide off-diagonal entries by λ_j - λ_i to get
        // the unnormalized eigenvector changes. Its diagonal already gave dλ.
        for j in 0..n {
            for i in 0..n {
                moved[(i, j)] = if i == j {
                    Complex::default()
                } else {
                    ratio(moved[(i, j)], self.values[j] - self.values[i])
                };
            }
        }
        let mut vectors = product(&self.vectors, &moved);
        for j in 0..n {
            let pivot = self.pivots[j];
            let normalization = self.vectors.column(j).dotc(&vectors.column(j)).re;
            let phase = vectors[(pivot, j)].im / self.vectors[(pivot, j)].re;
            let correction = Complex::new(normalization, phase);
            for i in 0..n {
                vectors[(i, j)] -= self.vectors[(i, j)] * correction;
            }
        }
        if values.iter().chain(vectors.iter()).any(|&z| !finite(z)) {
            return Err(Error::Singular);
        }
        Ok((values, vectors))
    }

    /// The operator gradient from the eigenvalue and eigenvector cotangents, with the
    /// phase of each vector fixed at its pivot.
    ///
    /// Repeated eigenvalues support equal eigenvalue weights and zero vector
    /// cotangents within each repeated group. Individual modes there have no pullback.
    pub fn pullback(
        &self,
        values: &[Complex],
        mut vectors: DMatrix<Complex>,
    ) -> Result<DMatrix<Complex>> {
        let n = self.values.len();
        if values.len() != n
            || vectors.shape() != (n, n)
            || values.iter().chain(vectors.iter()).any(|&z| !finite(z))
        {
            return Err(Error::InvalidInput("invalid eigensystem cotangent".into()));
        }
        let vector_active: Vec<_> = vectors
            .column_iter()
            .map(|v| v.iter().any(|&z| z != Complex::default()))
            .collect();
        if !vector_active.iter().any(|&active| active) && values.iter().all(|&g| g == values[0]) {
            // Sum of all eigenvalues is trace(A), including defective matrices.
            return Ok(DMatrix::identity(n, n) * values[0]);
        }
        // Project out normalization and pivot-phase changes before pairing the
        // cotangent with the off-diagonal eigenvector derivative.
        for j in 0..n {
            let inner = vectors.column(j).dotc(&self.vectors.column(j));
            let pivot = self.pivots[j];
            if inner.im.abs() > 64.0 * f64::EPSILON * vectors.column(j).norm() && self.tied_pivot(j)
            {
                return Err(Error::InvalidInput(
                    "phase-dependent eigenvector pullback requires a unique largest component"
                        .into(),
                ));
            }
            for i in 0..n {
                vectors[(i, j)] -= inner.re * self.vectors[(i, j)];
            }
            vectors[(pivot, j)] += Complex::i() * inner.im / self.vectors[(pivot, j)].re;
        }
        let mut g = product_views(view(&self.vectors).adjoint(), view(&vectors));
        for j in 0..n {
            for i in 0..n {
                if i == j {
                    g[(i, j)] = values[i];
                    continue;
                }
                let gap = self.values[j] - self.values[i];
                if gap.norm() <= 64.0 * f64::EPSILON * self.scale {
                    let weight_scale = values[i].norm().max(values[j].norm());
                    if vector_active[i]
                        || vector_active[j]
                        || (values[i] - values[j]).norm() > 64.0 * f64::EPSILON * weight_scale
                    {
                        return Err(Error::InvalidInput(
                            "individual eigenmodes have no pullback at repeated eigenvalues".into(),
                        ));
                    }
                    g[(i, j)] = Complex::default();
                } else {
                    g[(i, j)] = ratio(g[(i, j)], gap.conj());
                }
            }
        }
        let mut result = product_adjoint_right(&g, &self.vectors);
        self.vectors_lu()?
            .solve_adjoint_in_place(view_mut(&mut result))?;
        if result.iter().any(|&z| !finite(z)) {
            return Err(Error::Singular);
        }
        Ok(result)
    }
}

/// LU scheduling and workspaces, equilibration and the solve pullback.
#[cfg(test)]
mod tests {
    use faer::dyn_stack::StackReq;
    use nalgebra::DMatrix;
    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::{SolveGradient, eig, equilibrate, faer_zeros, lu_threads, scratch, solve_owned};
    use crate::{
        Complex, Error,
        test_support::{ALGEBRA_CASES, complex_matrix, prop_assert_close},
    };

    #[test]
    fn eigen_derivatives_share_one_lazy_factorization() {
        let operator = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
            Complex::new(2.0, 0.1),
            Complex::new(3.0, -0.2),
        ]));
        let residual = eig(&operator).unwrap();
        assert!(residual.vectors_lu.get().is_none());
        // A trace cotangent needs no inverse eigenvectors, including at defective
        // eigenvalues. It should not allocate a derivative factorization.
        let weights = [Complex::from(1.0); 2];
        assert_eq!(
            residual.pullback(&weights, DMatrix::zeros(2, 2)).unwrap(),
            DMatrix::identity(2, 2)
        );
        assert!(residual.vectors_lu.get().is_none());
        let direction = DMatrix::from_element(2, 2, Complex::new(0.2, 0.1));
        let tangent = residual.pushforward(&direction).unwrap();
        let factor = residual.vectors_lu().unwrap();
        let gradient = residual
            .pullback(&weights, DMatrix::from_element(2, 2, Complex::from(0.3)))
            .unwrap();
        assert_eq!(residual.pushforward(&direction).unwrap(), tangent);
        assert_eq!(
            residual
                .pullback(&weights, DMatrix::from_element(2, 2, Complex::from(0.3)))
                .unwrap(),
            gradient
        );
        assert!(std::ptr::eq(factor, residual.vectors_lu().unwrap()));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn lu_scheduling_respects_worker_and_rhs_budgets(
            rows in 1_usize..100_000,
            columns in 1_usize..100_000,
            budget in 1_usize..128,
        ) {
            check_lu_scheduling(rows, columns, budget)?;
        }
    }

    /// The worker count stays within the pool, is one below 32 right-hand sides, and
    /// otherwise uses a pool of up to four threads whole.
    fn check_lu_scheduling(
        rows: usize,
        columns: usize,
        budget: usize,
    ) -> Result<(), TestCaseError> {
        let threads = lu_threads(rows, columns, budget);
        prop_assert!((1..=budget).contains(&threads));
        if columns < 32 {
            prop_assert_eq!(threads, 1);
        } else if budget <= 4 {
            prop_assert_eq!(threads, budget);
        }
        Ok(())
    }

    /// Power-of-two equation (row) and unknown (column) units of a linear system.
    #[derive(Clone, Debug)]
    enum Units {
        /// Independent row exponents.
        Rows(Vec<i32>),
        /// Independent column exponents.
        Columns(Vec<i32>),
        /// Independent row and column exponents.
        Both(Vec<i32>, Vec<i32>),
        /// Rows alternating `2^∓e` and columns `2^±e/2`, or with the two spreads
        /// exchanged, which makes the columns the more disparate axis.
        Alternating { exponent: i32, transposed: bool },
    }

    impl Units {
        /// Row and column scales `(d_r, d_c)` of `A = d_r H d_c`.
        fn scales(&self, n: usize) -> (Vec<f64>, Vec<f64>) {
            let power = |e: i32| 2.0_f64.powi(e);
            let unit = vec![1.0; n];
            match self {
                Self::Rows(e) => (e.iter().map(|&e| power(e)).collect(), unit),
                Self::Columns(e) => (unit, e.iter().map(|&e| power(e)).collect()),
                Self::Both(r, c) => (
                    r.iter().map(|&e| power(e)).collect(),
                    c.iter().map(|&e| power(e)).collect(),
                ),
                &Self::Alternating {
                    exponent,
                    transposed,
                } => {
                    let sign = |i: usize| if i.is_multiple_of(2) { -1 } else { 1 };
                    let (wide, narrow) = (exponent, exponent / 2);
                    let (row, column) = if transposed {
                        (narrow, wide)
                    } else {
                        (wide, narrow)
                    };
                    (
                        (0..n).map(|i| power(sign(i) * row)).collect(),
                        (0..n).map(|j| power(-sign(j) * column)).collect(),
                    )
                }
            }
        }
    }

    fn system() -> impl Strategy<Value = (DMatrix<Complex>, Units, DMatrix<Complex>)> {
        (1_usize..=40, 1_usize..=3).prop_flat_map(|(n, columns)| {
            let exponents = || prop::collection::vec(-400_i32..=400, n);
            (
                complex_matrix(n, n, 0.5),
                Just((0..n).collect::<Vec<_>>()).prop_shuffle(),
                prop_oneof![
                    exponents().prop_map(Units::Rows),
                    exponents().prop_map(Units::Columns),
                    (exponents(), exponents()).prop_map(|(r, c)| Units::Both(r, c)),
                    (0_i32..=400, any::<bool>()).prop_map(|(exponent, transposed)| {
                        Units::Alternating {
                            exponent,
                            transposed,
                        }
                    }),
                ],
                complex_matrix(n, columns, 1.0),
            )
                .prop_map(move |(random, rows, units, y)| {
                    // `random + n I` with permuted rows, so pivoting must reorder them.
                    let size = Complex::from(f64::from(u32::try_from(n).unwrap()));
                    let mut h = random + DMatrix::identity(n, n) * size;
                    h = DMatrix::from_fn(n, n, |i, j| h[(rows[i], j)]);
                    (h, units, y)
                })
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn solve_and_adjoint_are_invariant_to_equation_and_unknown_units(
            (h, units, y) in system(),
        ) {
            check_units(&h, &units, &y)?;
        }
    }

    /// Alternating units put the row maxima of `2^20 I` plus entries of order one off
    /// its diagonal. With unit maxima alone, pivoting chose them and the solve kept only
    /// ten digits.
    #[test]
    fn units_keep_the_pivots_of_a_dominant_diagonal() -> Result<(), TestCaseError> {
        let t = |k: usize| f64::from(u32::try_from(k).unwrap());
        let h = DMatrix::from_fn(6, 6, |i, j| {
            Complex::new(0.5 - 0.125 * t(i), 0.25 + 0.125 * t(j))
        }) + DMatrix::identity(6, 6) * Complex::from(2.0_f64.powi(20));
        let y = DMatrix::from_fn(6, 1, |i, _| Complex::new(1.0, t(i)));
        for transposed in [false, true] {
            let units = Units::Alternating {
                exponent: 64,
                transposed,
            };
            check_units(&h, &units, &y)?;
        }
        Ok(())
    }

    /// For `A = d_r H d_c` with a well-conditioned `H`, the solve of `A X = d_r H Y`
    /// gives `d_c X = Y` and the adjoint solve of `Aᴴ Z = d_c Hᴴ Y` gives `d_r Z = Y`,
    /// whatever the units: equilibration removes them in both the direct and the
    /// transposed branch. The returned scales reproduce the equilibrated operator,
    /// and the operator cotangent is `-Z Xᴴ`, so scaling `A` and `B` jointly leaves
    /// the loss unchanged.
    fn check_units(
        h: &DMatrix<Complex>,
        units: &Units,
        y: &DMatrix<Complex>,
    ) -> Result<(), TestCaseError> {
        let n = h.nrows();
        let (row, column) = units.scales(n);
        let operator = DMatrix::from_fn(n, n, |i, j| h[(i, j)] * row[i] * column[j]);
        let hy = h * y;
        let rhs = DMatrix::from_fn(n, y.ncols(), |i, j| hy[(i, j)] * row[i]);
        let tolerance = 1e-13 * y.norm();
        let residual = solve_owned(operator.clone(), rhs.clone()).unwrap();
        let unknowns = DMatrix::from_fn(n, y.ncols(), |i, j| residual.value[(i, j)] * column[i]);
        prop_assert_close!(&unknowns, y, tolerance);
        let hz = h.adjoint() * y;
        let cotangent = DMatrix::from_fn(n, y.ncols(), |i, j| hz[(i, j)] * column[i]);
        let adjoint = residual.adjoint_rhs(cotangent.clone()).unwrap();
        let equations = DMatrix::from_fn(n, y.ncols(), |i, j| adjoint[(i, j)] * row[i]);
        prop_assert_close!(&equations, y, tolerance);

        let SolveGradient {
            operator: ga,
            rhs: gb,
        } = residual.pullback(cotangent).unwrap();
        prop_assert_eq!(&gb, &adjoint);
        for i in 0..n {
            for j in 0..n {
                let terms = (0..y.ncols()).map(|p| (adjoint[(i, p)], residual.value[(j, p)]));
                let expected: Complex = terms.clone().map(|(z, x)| -z * x.conj()).sum();
                let scale: f64 = terms.map(|(z, x)| z.norm() * x.norm()).sum();
                prop_assert_close!(ga[(i, j)], expected, 1e-14 * scale, "entry ({}, {})", i, j);
            }
        }
        // Scaling A and B jointly leaves X unchanged: Re⟨ga, A⟩ + Re⟨gb, B⟩ = 0.
        let pairs = ga.iter().zip(&operator).chain(gb.iter().zip(&rhs));
        let magnitude: f64 = pairs.clone().map(|(g, x)| g.norm() * x.norm()).sum();
        let joint: f64 = pairs.map(|(g, x)| (g.conj() * x).re).sum();
        prop_assert_close!(joint, 0.0, 1e-13 * magnitude);

        // Equilibration in the direct or transposed branch.
        let Units::Alternating {
            exponent,
            transposed,
        } = *units
        else {
            return Ok(());
        };
        // Past 2^32 the spreads, not the random entries of H, decide the branch.
        if n < 2 || exponent < 32 {
            return Ok(());
        }
        let magnitude = |z: Complex| z.re.abs().max(z.im.abs());
        let row_min = operator
            .row_iter()
            .map(|r| r.iter().copied().map(magnitude).fold(0.0, f64::max))
            .fold(f64::INFINITY, f64::min);
        let column_min = operator
            .column_iter()
            .map(|c| c.iter().copied().map(magnitude).fold(0.0, f64::max))
            .fold(f64::INFINITY, f64::min);
        prop_assert_eq!(column_min < row_min, transposed);
        let mut equilibrated = operator.clone();
        let scales = equilibrate(&mut equilibrated).unwrap().unwrap();
        for i in 0..n {
            for j in 0..n {
                let expected = operator[(i, j)] * scales.row[i] * scales.column[j];
                let actual = equilibrated[(i, j)];
                let tolerance = 4.0 * f64::EPSILON * expected.norm() + f64::MIN_POSITIVE;
                prop_assert_close!(actual, expected, tolerance, "entry ({}, {})", i, j);
            }
        }
        let mut transpose = operator.transpose();
        let swapped = equilibrate(&mut transpose).unwrap().unwrap();
        prop_assert_eq!(&scales.row, &swapped.column);
        prop_assert_eq!(&scales.column, &swapped.row);
        Ok(())
    }

    #[test]
    fn scaled_solves_keep_solutions_near_the_float_limits() {
        let p = |e: i32| 2.0_f64.powi(e);
        let (h, t, u) = (p(1023), p(-100), p(-1022));
        let systems = [
            // Balancing doubles row 1; the second block needs its small equations.
            (
                4,
                vec![
                    1.0, 1.0, 0.0, 0.0, 0.0, t, 0.0, 0.0, 0.0, 0.0, 1.0, u, 0.0, 0.0, 1.0, -u,
                ],
                vec![0.0, h, 0.0, 1.0],
            ),
            // The max scales double row 0.
            (2, vec![0.5, 0.5, 0.0, t], vec![1.5 * h, 1.5 * h]),
            // The right-hand side spans 2^2000 until the row scales apply.
            (2, vec![1.0, 0.0, 0.0, p(-1000)], vec![p(1000), 1.0]),
            // A subnormal right-hand side entry whose row scale makes it normal.
            (2, vec![t, 0.0, 0.0, 1.0], vec![p(-974), h]),
        ];
        for (n, a, x) in systems {
            let a = DMatrix::from_row_slice(n, n, &a).map(Complex::from);
            let x = DMatrix::from_vec(n, 1, x).map(Complex::from);
            let b = &a * &x;
            assert_eq!(solve_owned(a.clone(), b.clone()).unwrap().value, x);
            let transposed = solve_owned(a.adjoint(), DMatrix::zeros(n, 1)).unwrap();
            assert_eq!(transposed.adjoint_rhs(b).unwrap(), x);
        }
    }

    #[test]
    fn refused_and_overflowing_workspaces_return_out_of_memory() {
        let more = format!("cannot allocate more than {} bytes", usize::MAX);
        let half = StackReq::new::<Complex>(1 << 59);
        for overflow in [StackReq::new::<Complex>(usize::MAX), half.and(half)] {
            assert!(
                matches!(scratch(overflow), Err(Error::OutOfMemory(message)) if message == more),
                "{overflow:?}"
            );
        }
        // 2^52 bytes, beyond the user address space of every supported system.
        assert!(matches!(
            scratch(StackReq::new::<Complex>(1 << 48)),
            Err(Error::OutOfMemory(message)) if message == "cannot allocate 4503599627370496 bytes"
        ));
        assert!(scratch(StackReq::new::<Complex>(4)).is_ok());
        assert!(matches!(
            faer_zeros(1 << 24, 1 << 24),
            Err(Error::OutOfMemory(_))
        ));
        assert!(matches!(
            faer_zeros(usize::MAX, 2),
            Err(Error::OutOfMemory(_))
        ));
        let matrix = faer_zeros(3, 2).unwrap();
        assert_eq!(matrix.shape(), (3, 2));
        assert_eq!(matrix[(2, 1)], Complex::default());
    }

    #[test]
    fn ordinary_operator_is_not_modified() {
        let mut operator = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex::new(2.0, 1.0),
                Complex::new(0.5, -0.25),
                Complex::new(-0.25, 0.5),
                Complex::new(3.0, -1.0),
            ],
        );
        let original = operator.clone();
        assert!(equilibrate(&mut operator).unwrap().is_none());
        assert_eq!(operator, original);
        operator[(0, 1)] = Complex::from(2.0_f64.powi(-100));
        assert!(equilibrate(&mut operator).unwrap().is_none());
        operator *= Complex::from(2.0_f64.powi(-600));
        assert!(equilibrate(&mut operator).unwrap().is_none());
    }

    #[test]
    fn extreme_column_units_do_not_underflow_during_row_scaling() {
        let high = 2.0_f64.powi(600);
        let low = high.recip();
        let operator = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex::from(high),
                Complex::from(low),
                Complex::from(high),
                Complex::from(2.0 * low),
            ],
        );
        let rhs = DMatrix::from_column_slice(2, 1, &[Complex::from(1.0), Complex::from(2.0)]);
        let residual = solve_owned(operator, rhs).unwrap();
        assert_eq!(residual.value[(0, 0)], Complex::default());
        assert_eq!(residual.value[(1, 0)], Complex::from(high));
        // A^H [1, -1] = [0, -low], with both exact adjoint coordinates finite.
        let cotangent =
            DMatrix::from_column_slice(2, 1, &[Complex::default(), Complex::from(-low)]);
        let adjoint = residual.adjoint_rhs(cotangent).unwrap();
        assert_eq!(adjoint[(0, 0)], Complex::from(1.0));
        assert_eq!(adjoint[(1, 0)], Complex::from(-1.0));
    }
}

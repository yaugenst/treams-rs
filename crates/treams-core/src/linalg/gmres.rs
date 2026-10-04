//! Restarted GMRES (generalized minimal residual method) for complex linear systems given only
//! through matrix-vector products. treams-rs extension.
//!
//! Each cycle builds an orthonormal Krylov basis of at most `restart` vectors by
//! Arnoldi iteration, with two passes of modified Gram-Schmidt per vector, and takes
//! the least-squares update through Givens rotations. A cycle ends early when the
//! rotated residual estimate meets the tolerance, the new Krylov vector vanishes, or
//! the iteration limit is reached. Before it accepts an answer or
//! restarts, the solver recomputes the true residual `b - A x` with one more operator
//! application, and only that residual decides convergence.
#![allow(clippy::indexing_slicing)] // Krylov basis and Hessenberg indices below `restart + 1`.

use crate::{Complex, Error, Result};

/// Accuracy and memory budget of restarted GMRES.
///
/// The solve converges when the true residual norm is at most
/// `max(atol, rtol * |b|)`. The defaults are `rtol = 1e-10`, `atol = 0`,
/// `restart = 30` and `max_iterations = 300`; the Python constants
/// `iterative.DEFAULT_RTOL`, `DEFAULT_ATOL`, `DEFAULT_RESTART` and
/// `DEFAULT_MAX_ITERATIONS` repeat them.
#[derive(Clone, Copy, Debug)]
pub struct GmresOptions {
    /// Relative residual tolerance with respect to the right-hand side norm.
    pub rtol: f64,
    /// Absolute residual tolerance, for right-hand sides near zero.
    pub atol: f64,
    /// Maximum number of Krylov vectors retained before a restart.
    pub restart: usize,
    /// Maximum total operator applications in the Arnoldi iterations.
    pub max_iterations: usize,
}

impl Default for GmresOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-10,
            atol: 0.0,
            restart: 30,
            max_iterations: 300,
        }
    }
}

impl GmresOptions {
    /// Reject non-finite or negative tolerances, two zero tolerances, and a zero
    /// restart length or iteration limit.
    pub(crate) fn validate(self) -> Result<()> {
        if !self.rtol.is_finite()
            || !self.atol.is_finite()
            || self.rtol < 0.0
            || self.atol < 0.0
            || self.rtol + self.atol == 0.0
            || self.restart == 0
            || self.max_iterations == 0
        {
            return Err(Error::InvalidInput(
                "GMRES needs finite nonnegative tolerances, a positive tolerance, restart and iteration limit".into(),
            ));
        }
        Ok(())
    }
}

/// Proof of convergence for one right-hand side: the residual `b - A x` recomputed
/// from the returned answer `x`, not the estimate of the Arnoldi iteration.
///
/// `residual_norm <= max(atol, rtol * rhs_norm)` holds for every returned value.
#[derive(Clone, Copy, Debug)]
pub struct Convergence {
    /// Arnoldi iterations performed, including all restarts.
    pub iterations: usize,
    /// Euclidean norm of the recomputed residual `b - A x`.
    pub residual_norm: f64,
    /// Euclidean norm of the right-hand side.
    pub rhs_norm: f64,
}

/// Euclidean norm without overflow or underflow of the squared terms.
pub(crate) fn norm(vector: &[Complex]) -> f64 {
    vector.iter().fold(0.0_f64, |sum, z| sum.hypot(z.norm()))
}

/// Solve `A x = rhs` for the operator that `apply` multiplies with, starting from
/// `x = 0`. Fails with [`Error::NotConverged`] when the residual is not finite, when
/// the iterations reach `options.max_iterations`, or when the Arnoldi iteration breaks
/// down before convergence; errors of `apply` pass through.
pub(crate) fn gmres(
    rhs: &[Complex],
    options: GmresOptions,
    apply: impl Fn(&[Complex]) -> Result<Vec<Complex>>,
) -> Result<(Vec<Complex>, Convergence)> {
    let (answer, reports) = gmres_batch(rhs, 1, options, |x, _| apply(x))?;
    Ok((answer, reports[0]))
}

/// Independent GMRES solves of `columns` column-major right-hand sides. Only the
/// operator applications are shared: each column keeps its own Krylov basis,
/// rotations, restart decisions and true-residual convergence certificate.
///
/// The caller bounds `columns` to set the Krylov memory budget. `apply` receives
/// only columns that still need that application, in column-major order.
pub(crate) fn gmres_batch(
    rhs: &[Complex],
    columns: usize,
    options: GmresOptions,
    apply: impl Fn(&[Complex], usize) -> Result<Vec<Complex>>,
) -> Result<(Vec<Complex>, Vec<Convergence>)> {
    let dimension = rhs.len() / columns;
    let restart = options.restart.min(dimension).min(options.max_iterations);
    let mut states: Vec<_> = (0..columns)
        .map(|column| Krylov::new(&rhs[column * dimension..][..dimension], options))
        .collect();
    loop {
        let mut cycle = Vec::with_capacity(columns);
        for (column, state) in states.iter_mut().enumerate() {
            if !state.converged(options)? {
                state.start(restart);
                cycle.push(column);
            }
        }
        if cycle.is_empty() {
            break;
        }
        let mut extending = cycle.clone();
        for j in 0..restart {
            let mut product = apply_columns(
                &extending,
                |column| states[column].basis[j].as_slice(),
                &apply,
            )?;
            let mut next = Vec::with_capacity(extending.len());
            for (&column, w) in extending.iter().zip(product.chunks_exact_mut(dimension)) {
                if states[column].step(w, j, restart, options)? {
                    next.push(column);
                }
            }
            extending = next;
            if extending.is_empty() {
                break;
            }
        }
        for &column in &cycle {
            states[column].update(restart);
        }
        // Never certify convergence using the Hessenberg estimate alone.
        let product = apply_columns(&cycle, |column| states[column].answer.as_slice(), &apply)?;
        for (&column, product) in cycle.iter().zip(product.chunks_exact(dimension)) {
            states[column].residual = rhs[column * dimension..][..dimension]
                .iter()
                .zip(product)
                .map(|(b, a)| b - a)
                .collect();
        }
    }
    let reports = states.iter().map(|state| state.report).collect();
    let answer = if let [state] = states.as_mut_slice() {
        std::mem::take(&mut state.answer)
    } else {
        states.into_iter().flat_map(|state| state.answer).collect()
    };
    Ok((answer, reports))
}

/// Pack active columns only when there is more than one: scalar solves and the
/// tail of a batch lend their existing Krylov or answer buffer directly.
fn apply_columns<'a>(
    columns: &[usize],
    vector: impl Fn(usize) -> &'a [Complex],
    apply: &impl Fn(&[Complex], usize) -> Result<Vec<Complex>>,
) -> Result<Vec<Complex>> {
    if let [column] = columns {
        apply(vector(*column), 1)
    } else {
        let input: Vec<_> = columns
            .iter()
            .flat_map(|&column| vector(column).iter().copied())
            .collect();
        apply(&input, columns.len())
    }
}

/// One column's independent Arnoldi iteration. Batching changes when its operator
/// products run, not its arithmetic or convergence decisions.
struct Krylov {
    answer: Vec<Complex>,
    residual: Vec<Complex>,
    tolerance: f64,
    report: Convergence,
    basis: Vec<Vec<Complex>>,
    h: Vec<Complex>,
    rotations: Vec<(f64, Complex)>,
    g: Vec<Complex>,
    used: usize,
}

impl Krylov {
    fn new(rhs: &[Complex], options: GmresOptions) -> Self {
        let rhs_norm = norm(rhs);
        Self {
            answer: vec![Complex::default(); rhs.len()],
            residual: rhs.to_vec(),
            tolerance: options.atol.max(options.rtol * rhs_norm),
            report: Convergence {
                iterations: 0,
                residual_norm: rhs_norm,
                rhs_norm,
            },
            basis: Vec::new(),
            h: Vec::new(),
            rotations: Vec::new(),
            g: Vec::new(),
            used: 0,
        }
    }

    fn converged(&mut self, options: GmresOptions) -> Result<bool> {
        self.report.residual_norm = norm(&self.residual);
        let Convergence {
            residual_norm,
            iterations,
            ..
        } = self.report;
        if residual_norm.is_finite() && residual_norm <= self.tolerance {
            return Ok(true);
        }
        if !residual_norm.is_finite() || iterations >= options.max_iterations {
            return Err(Error::NotConverged(format!(
                "GMRES did not converge after {iterations} iterations: residual {residual_norm:e}, tolerance {:e}",
                self.tolerance
            )));
        }
        Ok(false)
    }

    fn start(&mut self, restart: usize) {
        // Initially converged columns need no Krylov workspace, even when the
        // requested restart budget is large. Reuse it after the first cycle.
        self.basis.clear();
        self.basis.reserve(restart + 1);
        self.basis.push(
            self.residual
                .iter()
                .map(|z| z / self.report.residual_norm)
                .collect(),
        );
        self.h.clear();
        self.h.resize((restart + 1) * restart, Complex::default());
        self.rotations.clear();
        self.rotations.reserve(restart);
        self.g.clear();
        self.g.resize(restart + 1, Complex::default());
        self.g[0] = Complex::new(self.report.residual_norm, 0.0);
        self.used = 0;
    }

    /// Advance one Arnoldi step and return whether this cycle needs another.
    fn step(
        &mut self,
        w: &mut [Complex],
        j: usize,
        restart: usize,
        options: GmresOptions,
    ) -> Result<bool> {
        // Two modified Gram-Schmidt passes keep orthogonality near resonance.
        for _ in 0..2 {
            for i in 0..=j {
                let dot: Complex = self.basis[i]
                    .iter()
                    .zip(&*w)
                    .map(|(v, w)| v.conj() * w)
                    .sum();
                self.h[j * (restart + 1) + i] += dot;
                for (w, v) in w.iter_mut().zip(&self.basis[i]) {
                    *w -= dot * v;
                }
            }
        }
        let next = norm(w);
        self.h[j * (restart + 1) + j + 1] = Complex::new(next, 0.0);
        for (i, &(c, s)) in self.rotations.iter().enumerate() {
            let offset = j * (restart + 1) + i;
            let a = self.h[offset];
            let b = self.h[offset + 1];
            self.h[offset] = c * a + s * b;
            self.h[offset + 1] = -s.conj() * a + c * b;
        }
        let diagonal = j * (restart + 1) + j;
        let a = self.h[diagonal];
        let b = self.h[diagonal + 1];
        let length = a.norm().hypot(b.norm());
        if !length.is_finite() || length == 0.0 {
            return Err(Error::NotConverged(
                "GMRES Arnoldi breakdown before convergence".into(),
            ));
        }
        let phase = if a.norm() == 0.0 {
            Complex::new(1.0, 0.0)
        } else {
            a / a.norm()
        };
        let c = a.norm() / length;
        let s = phase * b.conj() / length;
        self.h[diagonal] = phase * length;
        self.h[diagonal + 1] = Complex::default();
        self.rotations.push((c, s));
        self.g[j + 1] = -s.conj() * self.g[j];
        self.g[j] *= c;
        self.used = j + 1;
        self.report.iterations += 1;
        if self.g[j + 1].norm() <= self.tolerance
            || next == 0.0
            || self.report.iterations == options.max_iterations
        {
            return Ok(false);
        }
        self.basis.push(w.iter().map(|z| z / next).collect());
        Ok(true)
    }

    fn update(&mut self, restart: usize) {
        for i in (0..self.used).rev() {
            let tail: Complex = ((i + 1)..self.used)
                .map(|j| self.h[j * (restart + 1) + i] * self.g[j])
                .sum();
            self.g[i] = (self.g[i] - tail) / self.h[i * (restart + 1) + i];
            for (x, v) in self.answer.iter_mut().zip(&self.basis[i]) {
                *x += self.g[i] * v;
            }
        }
    }
}

/// Convergence, restarts and failures of GMRES against direct solves.
#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use nalgebra::{DMatrix, DVector};
    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::{GmresOptions, Krylov, gmres, gmres_batch, norm};
    use crate::{
        Complex,
        test_support::{DEFAULT_CASES, complex_matrix, prop_assert_close},
    };

    fn options() -> GmresOptions {
        GmresOptions {
            rtol: 2e-12,
            restart: 8,
            max_iterations: 100,
            ..GmresOptions::default()
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

        #[test]
        fn restarted_gmres_matches_a_direct_solve(
            (m, rhs, restart) in (1_usize..=24).prop_flat_map(|n| {
                (complex_matrix(n, n, 1.0), complex_matrix(n, 1, 1.0), 1..=n + 2)
            }),
        ) {
            check_gmres(&m, rhs.as_slice(), restart)?;
        }
    }

    /// Restarted GMRES on the nonnormal `A = I + M / (2 ‖M‖)`, whose field of values
    /// stays at least 1/2 from zero, converges for every restart length to the direct
    /// solution, reports its true residual within the tolerance, and without restarts
    /// terminates within `n` iterations.
    fn check_gmres(
        m: &DMatrix<Complex>,
        rhs: &[Complex],
        restart: usize,
    ) -> Result<(), TestCaseError> {
        let n = m.nrows();
        let shrink = if m.norm() > 0.0 { 0.5 / m.norm() } else { 0.0 };
        let a = DMatrix::identity(n, n) + m * Complex::from(shrink);
        let options = GmresOptions {
            rtol: 1e-10,
            atol: 0.0,
            restart,
            max_iterations: 4000,
        };
        let apply = |x: &[Complex]| Ok((&a * DVector::from_column_slice(x)).as_slice().to_vec());
        let (answer, report) = gmres(rhs, options, apply).unwrap();
        let b = DVector::from_column_slice(rhs);
        let direct = a.clone().lu().solve(&b).unwrap();
        prop_assert_close!(
            answer.as_slice(),
            direct.as_slice(),
            1e-8 * (1.0 + direct.norm())
        );
        prop_assert_close!(
            report.rhs_norm,
            b.norm(),
            1e-14 * b.norm() + f64::MIN_POSITIVE
        );
        let residual = (&b - &a * DVector::from_column_slice(&answer)).norm();
        prop_assert_close!(
            report.residual_norm,
            residual,
            1e-14 * b.norm() + f64::MIN_POSITIVE
        );
        prop_assert!(report.residual_norm <= options.rtol * report.rhs_norm);
        if restart >= n {
            prop_assert!(report.iterations <= n, "{} iterations", report.iterations);
        }
        Ok(())
    }

    #[test]
    fn gmres_zero_happy_breakdown_restarts_and_failure() {
        let identity = |x: &[Complex]| Ok(x.to_vec());
        let (zero, report) = gmres(&[Complex::default(); 3], options(), identity).unwrap();
        assert_eq!(zero, vec![Complex::default(); 3]);
        assert_eq!(report.iterations, 0);
        let rhs = vec![Complex::new(0.2, -0.3); 3];
        let (answer, report) = gmres(&rhs, options(), identity).unwrap();
        assert!((norm(&answer) - norm(&rhs)).abs() < 1e-14);
        assert_eq!(report.iterations, 1);
        let diagonal = |x: &[Complex]| {
            Ok(x.iter()
                .enumerate()
                .map(|(i, x)| {
                    Complex::new(1.0 + f64::from(u32::try_from(i).unwrap()) * 0.3, 0.1) * x
                })
                .collect())
        };
        let limited = GmresOptions {
            restart: 1,
            max_iterations: 1,
            ..options()
        };
        assert!(gmres(&rhs, limited, diagonal).is_err());
        let restarted = GmresOptions {
            restart: 1,
            ..options()
        };
        let (_, report) = gmres(&rhs, restarted, diagonal).unwrap();
        assert!(report.iterations > 1);
        assert!(report.residual_norm <= restarted.rtol * report.rhs_norm);
        assert!(gmres(&rhs, options(), |_| Ok(vec![Complex::default(); 3])).is_err());
    }

    /// Sharing operator applications preserves each column's arithmetic, even when
    /// zero columns skip the solve and eigenvectors finish before the other columns.
    #[test]
    fn batched_columns_keep_independent_convergence() {
        let diagonal = [
            Complex::new(1.0, 0.1),
            Complex::new(1.3, 0.1),
            Complex::new(1.6, 0.1),
        ];
        let apply = |input: &[Complex]| {
            Ok(input
                .iter()
                .enumerate()
                .map(|(i, x)| diagonal[i % 3] * x)
                .collect::<Vec<_>>())
        };
        for (rtol, atol) in [(1e-10, 0.0), (0.0, 1e-10)] {
            let options = GmresOptions {
                rtol,
                atol,
                restart: 2,
                max_iterations: 100,
            };
            let mut rhs = vec![Complex::default(); 18];
            rhs[0] = Complex::new(1.0, 2.0);
            for (column, scale) in [0.7, 1e-120, if rtol > 0.0 { 1e120 } else { 1e-3 }]
                .into_iter()
                .enumerate()
            {
                for (i, z) in [
                    Complex::new(0.2, -0.3),
                    Complex::new(0.5, 0.7),
                    Complex::new(0.6, -0.1),
                ]
                .into_iter()
                .enumerate()
                {
                    rhs[(column + 2) * 3 + i] = z * scale;
                }
            }
            rhs[16] = Complex::new(1.0, -0.3);
            let widths = RefCell::new(Vec::new());
            let (answer, reports) = gmres_batch(&rhs, 6, options, |input, columns| {
                assert_eq!(input.len(), columns * 3);
                widths.borrow_mut().push(columns);
                apply(input)
            })
            .unwrap();
            for (column, right) in rhs.as_chunks::<3>().0.iter().enumerate() {
                let (independent, report) = gmres(right, options, apply).unwrap();
                assert_eq!(&answer[column * 3..][..3], independent.as_slice());
                assert_eq!(reports[column].iterations, report.iterations);
                assert_eq!(
                    reports[column].rhs_norm.to_bits(),
                    report.rhs_norm.to_bits()
                );
                assert_eq!(
                    reports[column].residual_norm.to_bits(),
                    report.residual_norm.to_bits()
                );
                assert!(report.residual_norm <= atol.max(rtol * report.rhs_norm));
            }
            assert_eq!(reports[1].iterations, 0);
            assert_eq!(reports[0].iterations, 1);
            assert!(reports[2].iterations > 2);
            let widths = widths.into_inner();
            assert!(widths[0] > 1);
            assert!(widths.last().unwrap() < &widths[0]);
            let independent_calls: usize = reports
                .iter()
                .map(|report| report.iterations + report.iterations.div_ceil(2))
                .sum();
            assert!(widths.len() < independent_calls);
        }
    }

    /// Different eigenvector columns each need one iteration. A shared Krylov
    /// polynomial would incorrectly exhaust this independently sufficient budget.
    #[test]
    fn batched_eigenvectors_obey_each_columns_iteration_limit() {
        let rhs = [1.0, 0.0, 0.0, 1.0].map(Complex::from);
        let options = GmresOptions {
            restart: 1,
            max_iterations: 1,
            ..options()
        };
        let (answer, reports) = gmres_batch(&rhs, 2, options, |input, _| {
            Ok(input
                .iter()
                .enumerate()
                .map(|(i, x)| x * if i % 2 == 0 { 2.0 } else { 3.0 })
                .collect())
        })
        .unwrap();
        assert_eq!(answer, [0.5, 0.0, 0.0, 1.0 / 3.0].map(Complex::from));
        assert!(reports.iter().all(|report| report.iterations == 1));
        let (answer, reports) = gmres_batch(&[Complex::default(); 8], 4, options, |_, _| {
            panic!("zero right-hand sides must not apply the operator")
        })
        .unwrap();
        assert_eq!(answer, vec![Complex::default(); 8]);
        assert!(reports.iter().all(|report| report.iterations == 0));
    }

    #[test]
    fn initially_converged_columns_allocate_no_krylov_workspace() {
        let options = GmresOptions {
            atol: 1e-12,
            restart: usize::MAX,
            max_iterations: usize::MAX,
            ..options()
        };
        for scale in [0.0, 1e-14] {
            let mut state = Krylov::new(&[Complex::from(scale); 3], options);
            assert!(state.converged(options).unwrap());
            assert_eq!(state.basis.capacity(), 0);
            assert_eq!(state.h.capacity(), 0);
            assert_eq!(state.rotations.capacity(), 0);
            assert_eq!(state.g.capacity(), 0);
        }
    }
}

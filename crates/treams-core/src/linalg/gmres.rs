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
    let dimension = rhs.len();
    let restart = options.restart.min(dimension).min(options.max_iterations);
    let rhs_norm = norm(rhs);
    let tolerance = options.atol.max(options.rtol * rhs_norm);
    let mut answer = vec![Complex::default(); dimension];
    let mut residual = rhs.to_vec();
    let mut iterations = 0;
    loop {
        let residual_norm = norm(&residual);
        if residual_norm.is_finite() && residual_norm <= tolerance {
            return Ok((
                answer,
                Convergence {
                    iterations,
                    residual_norm,
                    rhs_norm,
                },
            ));
        }
        if !residual_norm.is_finite() || iterations >= options.max_iterations {
            return Err(Error::NotConverged(format!(
                "GMRES did not converge after {iterations} iterations: residual {residual_norm:e}, tolerance {tolerance:e}"
            )));
        }
        let mut basis = Vec::with_capacity(restart + 1);
        basis.push(
            residual
                .iter()
                .map(|z| z / residual_norm)
                .collect::<Vec<_>>(),
        );
        let mut h = vec![Complex::default(); (restart + 1) * restart];
        let mut rotations = Vec::<(f64, Complex)>::with_capacity(restart);
        let mut g = vec![Complex::default(); restart + 1];
        g[0] = Complex::new(residual_norm, 0.0);
        let mut used = 0;
        for j in 0..restart {
            let mut w = apply(&basis[j])?;
            // Two modified Gram-Schmidt passes keep orthogonality near resonance.
            for _ in 0..2 {
                for i in 0..=j {
                    let dot: Complex = basis[i].iter().zip(&w).map(|(v, w)| v.conj() * w).sum();
                    h[j * (restart + 1) + i] += dot;
                    for (w, v) in w.iter_mut().zip(&basis[i]) {
                        *w -= dot * v;
                    }
                }
            }
            let next = norm(&w);
            h[j * (restart + 1) + j + 1] = Complex::new(next, 0.0);
            for (i, &(c, s)) in rotations.iter().enumerate() {
                let offset = j * (restart + 1) + i;
                let a = h[offset];
                let b = h[offset + 1];
                h[offset] = c * a + s * b;
                h[offset + 1] = -s.conj() * a + c * b;
            }
            let diagonal = j * (restart + 1) + j;
            let a = h[diagonal];
            let b = h[diagonal + 1];
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
            h[diagonal] = phase * length;
            h[diagonal + 1] = Complex::default();
            rotations.push((c, s));
            g[j + 1] = -s.conj() * g[j];
            g[j] *= c;
            used = j + 1;
            iterations += 1;
            if g[j + 1].norm() <= tolerance || next == 0.0 || iterations == options.max_iterations {
                break;
            }
            basis.push(w.into_iter().map(|z| z / next).collect());
        }
        for i in (0..used).rev() {
            let tail: Complex = ((i + 1)..used)
                .map(|j| h[j * (restart + 1) + i] * g[j])
                .sum();
            g[i] = (g[i] - tail) / h[i * (restart + 1) + i];
            for (x, v) in answer.iter_mut().zip(&basis[i]) {
                *x += g[i] * v;
            }
        }
        // Never certify convergence using the Hessenberg estimate alone.
        residual = rhs
            .iter()
            .zip(apply(&answer)?)
            .map(|(b, a)| b - a)
            .collect();
    }
}

/// Convergence, restarts and failures of GMRES against direct solves.
#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};
    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::{GmresOptions, gmres, norm};
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
}

//! faer LU scheduling micro-benchmark.
//!
//! Times a faer partial-pivoting LU factorization, a solve and an adjoint
//! (conjugate-transpose) solve of a diagonally dominant complex128 matrix, all on
//! one existing Rayon pool, for the given rows, right-hand sides and requested
//! faer workers. It calibrates the worker count that `linalg::lu_threads` picks:
//! compare 1, 2, 4, 8 and 16 workers on the same CPU affinity.
//!
//! ```text
//! RAYON_NUM_THREADS=16 cargo bench -p treams-core --bench lu_scheduling -- 1024 64 2 5
//! ```
//!
//! Arguments are matrix rows, right-hand-side columns, requested faer workers and
//! measured repetitions. A first, unreported repetition warms up and checks the
//! normal and adjoint equation residuals; every measured repetition prints one
//! JSON line with the factor, solve and adjoint seconds.
#![allow(clippy::cast_precision_loss)] // Small integer indices seed the matrix entries.

use faer::{
    Conj, MatMut, MatRef, Par, Spec,
    dyn_stack::{MemBuffer, MemStack},
    linalg::lu::partial_pivoting::{factor, solve},
    perm::PermRef,
};
use nalgebra::DMatrix;
use num_complex::Complex64 as C;
use std::{hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `cargo bench` appends `--bench`, which is not a positional argument.
    let args: Vec<_> = std::env::args().filter(|arg| arg != "--bench").collect();
    let n: usize = args.get(1).map_or(Ok(1024), |s| s.parse())?;
    let k: usize = args.get(2).map_or(Ok(64), |s| s.parse())?;
    let threads: usize = args.get(3).map_or(Ok(2), |s| s.parse())?;
    let reps: usize = args.get(4).map_or(Ok(5), |s| s.parse())?;
    assert!(n > 0 && k > 0 && threads > 0 && reps > 0);
    let par = if threads == 1 {
        Par::Seq
    } else {
        Par::rayon(threads.min(rayon::current_num_threads()))
    };
    let a = DMatrix::from_fn(n, n, |i, j| {
        let x = (i * 17 + j * 31) as f64;
        C::new(
            x.sin() + if i == j { (n + 1) as f64 } else { 0. },
            (x * 1.37).cos(),
        )
    });
    let b = DMatrix::from_fn(n, k, |i, j| {
        let x = (i * 19 + j * 13 + 2) as f64;
        C::new(x.sin(), x.cos())
    });
    let mut p = vec![0usize; n];
    let mut pi = vec![0usize; n];
    for rep in 0..=reps {
        let mut lu = a.clone();
        let mut rhs = b.clone();
        let now = Instant::now();
        factor::lu_in_place(
            MatMut::from_column_major_slice_mut(lu.as_mut_slice(), n, n),
            &mut p,
            &mut pi,
            par,
            MemStack::new(&mut MemBuffer::new(
                factor::lu_in_place_scratch::<usize, C>(n, n, par, Spec::default()),
            )),
            Spec::default(),
        );
        let factor_seconds = now.elapsed().as_secs_f64();
        let perm = PermRef::new_checked(&p, &pi, n);
        let factors = MatRef::from_column_major_slice(lu.as_slice(), n, n);
        let now = Instant::now();
        solve::solve_in_place(
            factors,
            factors,
            perm,
            MatMut::from_column_major_slice_mut(rhs.as_mut_slice(), n, k),
            par,
            MemStack::new(&mut MemBuffer::new(
                solve::solve_in_place_scratch::<usize, C>(n, k, par),
            )),
        );
        let solve_seconds = now.elapsed().as_secs_f64();
        let solution = rhs.clone();
        let now = Instant::now();
        solve::solve_transpose_in_place_with_conj(
            factors,
            factors,
            perm,
            Conj::Yes,
            MatMut::from_column_major_slice_mut(rhs.as_mut_slice(), n, k),
            par,
            MemStack::new(&mut MemBuffer::new(
                solve::solve_transpose_in_place_scratch::<usize, C>(n, k, par),
            )),
        );
        let adjoint_seconds = now.elapsed().as_secs_f64();
        black_box(&rhs);
        if rep == 0 {
            let check = |adjoint: bool, value: &DMatrix<C>, target: &DMatrix<C>| {
                let mut product = DMatrix::zeros(n, k);
                let a_view = MatRef::from_column_major_slice(a.as_slice(), n, n);
                faer::linalg::matmul::matmul_with_conj(
                    MatMut::from_column_major_slice_mut(product.as_mut_slice(), n, k),
                    faer::Accum::Replace,
                    if adjoint { a_view.transpose() } else { a_view },
                    if adjoint { Conj::Yes } else { Conj::No },
                    MatRef::from_column_major_slice(value.as_slice(), n, k),
                    Conj::No,
                    C::new(1.0, 0.0),
                    par,
                );
                assert!((&product - target).norm() / target.norm() < 2e-12);
            };
            check(false, &solution, &b);
            check(true, &rhs, &solution);
        }
        if rep > 0 {
            println!(
                "{{\"n\":{n},\"rhs\":{k},\"threads\":{threads},\"pool\":{},\"factor\":{factor_seconds},\"solve\":{solve_seconds},\"adjoint\":{adjoint_seconds}}}",
                rayon::current_num_threads()
            );
        }
    }
    Ok(())
}

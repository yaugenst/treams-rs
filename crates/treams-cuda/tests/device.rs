//! Run explicitly with a GPU: cargo test -p treams-cuda --features cuda -- --ignored.
#![cfg(feature = "cuda")]
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::indexing_slicing
)]

use nalgebra::DMatrix;
use proptest::prelude::*;
use treams_core::{Complex, linalg};
use treams_cuda::gpu::{Error, Gpu};

fn matrix(rows: usize, cols: usize, seed: f64) -> DMatrix<Complex> {
    DMatrix::from_fn(rows, cols, |i, j| {
        let x = (i + 7 * j) as f64 + seed;
        Complex::new(x.sin(), (1.3 * x).cos()) / (rows as f64).sqrt()
    })
}

fn close(left: &DMatrix<Complex>, right: &DMatrix<Complex>, tolerance: f64) {
    assert_eq!(left.shape(), right.shape());
    assert!((left - right).norm() <= tolerance * right.norm().max(1.0));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    #[ignore = "requires a CUDA GPU and CUDA13 shared libraries"]
    fn solve_product_and_adjoint(n in 1usize..33, columns in 1usize..9, seed in 0.0f64..30.0) {
        let gpu = Gpu::new(0).unwrap();
        let mut a = matrix(n, n, seed);
        for i in 0..n { a[(i,i)] += Complex::new(n as f64 + 1.0, 0.0); }
        let b = matrix(n, columns, seed + 1.0);
        let g = matrix(n, columns, seed + 2.0);
        let cpu = linalg::solve(&a, b.clone()).unwrap();
        let lu = gpu.factor(gpu.upload(&a).unwrap()).unwrap();
        let x = lu.solve(gpu.upload(&b).unwrap(), false).unwrap();
        close(&x.download().unwrap(), &cpu.value, 2e-12);
        close(&gpu.matmul(&gpu.upload(&a).unwrap(), &x).unwrap().download().unwrap(), &b, 2e-12);
        let (ga,gb) = lu.pullback(&x, gpu.upload(&g).unwrap()).unwrap();
        let (ca,cb) = cpu.pullback(g).unwrap();
        close(&ga.download().unwrap(), &ca, 4e-12);
        close(&gb.download().unwrap(), &cb, 4e-12);
    }
}

#[test]
#[ignore = "requires a CUDA GPU and CUDA13 shared libraries"]
fn pivoting_singular_and_ownership() {
    let gpu = Gpu::new(0).unwrap();
    let a = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 1.0),
            Complex::new(1.0, -1.0),
            Complex::new(0.3, 0.8),
        ],
    );
    let b = matrix(2, 3, 2.0);
    let lu = gpu.factor(gpu.upload(&a).unwrap()).unwrap();
    let x = lu
        .solve(gpu.upload(&b).unwrap(), true)
        .unwrap()
        .download()
        .unwrap();
    close(&(&a.adjoint() * x), &b, 2e-12);
    let zeros = gpu.upload(&DMatrix::zeros(3, 3)).unwrap();
    assert!(matches!(gpu.factor(zeros), Err(Error::Singular(_))));
    let another = Gpu::new(0).unwrap();
    assert!(
        gpu.matmul(&gpu.upload(&a).unwrap(), &another.upload(&a).unwrap())
            .is_err()
    );
}

#[test]
#[ignore = "requires a CUDA GPU and CUDA13 shared libraries"]
fn shared_equilibration_and_implicit_pullback() {
    let gpu = Gpu::new(0).unwrap();
    let n = 8;
    let row: Vec<_> = (0..n)
        .map(|i| 10.0_f64.powi(i32::try_from(i).unwrap() * 8 - 28))
        .collect();
    let column: Vec<_> = (0..n)
        .map(|i| 10.0_f64.powi(i32::try_from(i).unwrap() * 6 - 21))
        .collect();
    let a = DMatrix::from_fn(n, n, |i, j| {
        let base = Complex::new(if i == j { 10.0 } else { 0.2 }, (i as f64 - j as f64).sin());
        base * row[i] * column[j]
    });
    let b = DMatrix::from_fn(n, 2, |i, j| Complex::new((i + j + 1) as f64, 0.3) * row[i]);
    let g = DMatrix::from_fn(n, 2, |i, j| {
        Complex::new((i + j + 1) as f64, 0.1) * column[i]
    });
    let cpu = linalg::solve(&a, b.clone()).unwrap();
    let factor = gpu.factor_host(a).unwrap();
    let actual = factor.solve(gpu.upload(&b).unwrap(), false).unwrap();
    let x = actual.download().unwrap();
    for j in 0..2 {
        for i in 0..n {
            assert!(((x[(i, j)] - cpu.value[(i, j)]) * column[i]).norm() < 2e-12);
        }
    }
    let (ga, gb) = factor.pullback(&actual, gpu.upload(&g).unwrap()).unwrap();
    let (ca, cb) = cpu.pullback(g).unwrap();
    close(&ga.download().unwrap(), &ca, 4e-12);
    close(&gb.download().unwrap(), &cb, 4e-12);
}

#[cfg(not(target_os = "linux"))]
#[test]
fn unsupported_host_returns_error_without_loading_cuda() {
    assert!(matches!(Gpu::new(0), Err(Error::Unavailable(_))));
}

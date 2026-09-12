//! Equal-precision dense-kernel benchmark; times transfers and reusable residency separately.
#![cfg(feature = "cuda")]
#![allow(
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::print_stdout
)]

use std::time::Instant;

use faer::linalg::solvers::Solve;
use nalgebra::DMatrix;
use serde_json::json;
use treams_core::{Complex, linalg};
use treams_cuda::gpu::Gpu;

fn cpu_product(a: &DMatrix<Complex>, b: &DMatrix<Complex>) -> DMatrix<Complex> {
    let mut output = DMatrix::zeros(a.nrows(), b.ncols());
    faer::linalg::matmul::matmul(
        faer::MatMut::from_column_major_slice_mut(output.as_mut_slice(), a.nrows(), b.ncols()),
        faer::Accum::Replace,
        faer::MatRef::from_column_major_slice(a.as_slice(), a.nrows(), a.ncols()),
        faer::MatRef::from_column_major_slice(b.as_slice(), b.nrows(), b.ncols()),
        Complex::new(1.0, 0.0),
        faer::get_global_parallelism(),
    );
    output
}

fn timed<T>(mut operation: impl FnMut() -> T) -> (f64, T) {
    let start = Instant::now();
    let value = operation();
    (start.elapsed().as_secs_f64(), value)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args.get(1).map_or(Ok(1024), |s| s.parse())?;
    let columns: usize = args.get(2).map_or(Ok(64), |s| s.parse())?;
    let repetitions: usize = args.get(3).map_or(Ok(5), |s| s.parse())?;
    let a = DMatrix::from_fn(n, n, |i, j| {
        let x = (i * 17 + j * 31) as f64;
        let diagonal = if i == j { (n + 1) as f64 } else { 0.0 };
        Complex::new(x.sin() + diagonal, (x * 1.37).cos())
    });
    let b = DMatrix::from_fn(n, columns, |i, j| {
        let x = (i * 19 + j * 13 + 2) as f64;
        Complex::new(x.sin(), x.cos())
    });
    let (setup, gpu) = timed(|| Gpu::new(0));
    let gpu = gpu?;
    // Warm both implementations and verify outside the timing samples.
    let cpu = linalg::solve(&a, b.clone())?;
    let cpu_factor = faer::MatRef::from_column_major_slice(a.as_slice(), n, n).partial_piv_lu();
    let factor = gpu.factor_host(a.clone())?;
    let resident_b = gpu.upload(&b)?;
    let actual = factor.solve(resident_b.try_clone()?, false)?.download()?;
    let relative_error = (&actual - &cpu.value).norm() / cpu.value.norm();
    let backward_error =
        (cpu_product(&a, &actual) - &b).norm() / (a.norm() * actual.norm() + b.norm());
    assert!(relative_error < 2e-12 && backward_error < 2e-13);
    let resident_a = gpu.upload(&a)?;
    let product_reference = cpu_product(&a, &b);
    let product = gpu.matmul(&resident_a, &resident_b)?.download()?;
    let product_error = (&product - &product_reference).norm() / product_reference.norm();
    assert!(product_error < 2e-12);
    let mut samples = Vec::new();
    for _ in 0..repetitions {
        let (cpu_seconds, result) = timed(|| linalg::solve(&a, b.clone()));
        std::hint::black_box(result?);
        let (cpu_solve_reuse_seconds, result) = timed(|| {
            cpu_factor.solve(faer::MatRef::from_column_major_slice(
                b.as_slice(),
                n,
                columns,
            ))
        });
        std::hint::black_box(result);
        let (gpu_seconds, result) = timed(|| {
            gpu.factor_host(a.clone())?
                .solve(gpu.upload(&b)?, false)?
                .download()
        });
        std::hint::black_box(result?);
        let (solve_resident_seconds, result) = timed(|| {
            let result = factor.solve(resident_b.try_clone()?, false)?;
            gpu.synchronize()?;
            Ok::<_, treams_cuda::gpu::Error>(result)
        });
        std::hint::black_box(result?);
        let (cpu_product_seconds, product) = timed(|| cpu_product(&a, &b));
        std::hint::black_box(product);
        let (gpu_product_seconds, result) = timed(|| {
            let result = gpu.matmul(&resident_a, &resident_b)?;
            gpu.synchronize()?;
            Ok::<_, treams_cuda::gpu::Error>(result)
        });
        std::hint::black_box(result?);
        let (gpu_product_host_seconds, result) =
            timed(|| gpu.matmul(&gpu.upload(&a)?, &gpu.upload(&b)?)?.download());
        std::hint::black_box(result?);
        samples.push(json!({"cpu_solve_seconds":cpu_seconds,"cpu_solve_reuse_seconds":cpu_solve_reuse_seconds,"gpu_solve_host_seconds":gpu_seconds,
            "gpu_solve_resident_seconds":solve_resident_seconds,"cpu_product_seconds":cpu_product_seconds,
            "gpu_product_resident_seconds":gpu_product_seconds,"gpu_product_host_seconds":gpu_product_host_seconds}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "device":gpu.name()?,"precision":"complex128","matrix_dimension":n,"right_hand_sides":columns,
            "setup_seconds":setup,"relative_error":relative_error,"backward_error":backward_error,
            "product_relative_error":product_error,"factor_device_bytes":factor.bytes(),"samples":samples,
            "cpu_threads":std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_|"default".into()),
            "scope":"dense kernels, synthetic well-conditioned operator; host timings include every transfer and synchronization; resident timings include allocations and synchronization but exclude initial upload/factorization"
        }))?
    );
    Ok(())
}

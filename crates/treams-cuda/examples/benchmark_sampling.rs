//! Repeated physical sampling with the same cached complex128 operator on CPU and GPU.
#![cfg(feature = "cuda")]
#![recursion_limit = "256"]
#![allow(
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::print_stdout
)]

use std::{hint::black_box, time::Instant};

use nalgebra::DMatrix;
use rayon::prelude::*;
use serde_json::json;
use treams_core::{Complex, plane};
use treams_cuda::gpu::Gpu;

fn cpu_product(
    operator: faer::MatRef<'_, Complex>,
    coefficients: &DMatrix<Complex>,
) -> DMatrix<Complex> {
    let mut field = DMatrix::zeros(operator.nrows(), 1);
    faer::linalg::matmul::matmul(
        faer::MatMut::from_column_major_slice_mut(field.as_mut_slice(), operator.nrows(), 1),
        faer::Accum::Replace,
        operator,
        faer::MatRef::from_column_major_slice(coefficients.as_slice(), coefficients.nrows(), 1),
        Complex::new(1.0, 0.0),
        faer::get_global_parallelism(),
    );
    field
}

fn cpu_adjoint(
    operator: faer::MatRef<'_, Complex>,
    cotangent: &DMatrix<Complex>,
) -> DMatrix<Complex> {
    let mut gradient = DMatrix::zeros(operator.ncols(), 1);
    faer::linalg::matmul::matmul_with_conj(
        faer::MatMut::from_column_major_slice_mut(gradient.as_mut_slice(), operator.ncols(), 1),
        faer::Accum::Replace,
        operator.transpose(),
        faer::Conj::Yes,
        faer::MatRef::from_column_major_slice(cotangent.as_slice(), cotangent.nrows(), 1),
        faer::Conj::No,
        Complex::new(1.0, 0.0),
        faer::get_global_parallelism(),
    );
    gradient
}

fn cpu_prepared_fused(
    geometry: &[([f64; 3], [Complex; 3])],
    points: &[[f64; 3]],
    coefficients: &DMatrix<Complex>,
) -> DMatrix<Complex> {
    // Coefficients change on every application, so this O(modes) preparation is timed.
    let weighted: Vec<_> = geometry
        .iter()
        .zip(coefficients.iter())
        .map(|(&(k, electric), &c)| (k, electric.map(|e| e * c)))
        .collect();
    let mut field = DMatrix::zeros(3 * points.len(), 1);
    points
        .par_iter()
        .zip(field.as_mut_slice().par_chunks_mut(3))
        .for_each(|(point, out)| {
            for &(k, electric) in &weighted {
                // This benchmark generates real wavevectors: exp(i k.r) is exactly sin/cos.
                let phase = k[0] * point[0] + k[1] * point[1] + k[2] * point[2];
                let (sin, cos) = phase.sin_cos();
                let phase = Complex::new(cos, sin);
                for (out, e) in out.iter_mut().zip(electric) {
                    *out += e * phase;
                }
            }
        });
    field
}

fn peak_resident_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}

fn timed<T>(operation: impl FnOnce() -> T) -> (f64, T) {
    let start = Instant::now();
    let result = operation();
    (start.elapsed().as_secs_f64(), result)
}

struct SampleSeconds {
    cpu_fused: f64,
    cpu_prepared_fused: f64,
    cpu_column_major: f64,
    cpu_row_major: f64,
    coefficient_upload: f64,
    gpu_resident: f64,
    field_download: f64,
    gpu_roundtrip: f64,
    cpu_adjoint_column_major: f64,
    cpu_adjoint_row_major: f64,
    field_cotangent_upload: f64,
    gpu_adjoint_resident: f64,
    coefficient_gradient_download: f64,
    gpu_adjoint_roundtrip: f64,
}

impl SampleSeconds {
    fn json(&self, iteration: usize) -> serde_json::Value {
        json!({
            "iteration": iteration,
            "cpu_fused_seconds": self.cpu_fused,
            "cpu_prepared_fused_seconds": self.cpu_prepared_fused,
            "cpu_column_major_seconds": self.cpu_column_major,
            "cpu_row_major_seconds": self.cpu_row_major,
            "coefficient_upload_seconds": self.coefficient_upload,
            "gpu_resident_seconds": self.gpu_resident,
            "field_download_seconds": self.field_download,
            "gpu_roundtrip_seconds": self.gpu_roundtrip,
            "cpu_adjoint_column_major_seconds": self.cpu_adjoint_column_major,
            "cpu_adjoint_row_major_seconds": self.cpu_adjoint_row_major,
            "field_cotangent_upload_seconds": self.field_cotangent_upload,
            "gpu_adjoint_resident_seconds": self.gpu_adjoint_resident,
            "coefficient_gradient_download_seconds": self.coefficient_gradient_download,
            "gpu_adjoint_roundtrip_seconds": self.gpu_adjoint_roundtrip,
        })
    }
}

fn median(samples: impl Iterator<Item = f64>) -> f64 {
    let mut samples: Vec<_> = samples.collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn validate(
    actual: &DMatrix<Complex>,
    expected: &DMatrix<Complex>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut maximum = 0.0f64;
    for (a, b) in actual.iter().zip(expected.iter()) {
        let error = (*a - b).norm();
        if !error.is_finite() || error > 2e-12 * (1.0 + b.norm()) {
            return Err(format!("sampling result fails reference: {a} vs {b}").into());
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}

fn validate_pairing(
    cotangent: &DMatrix<Complex>,
    delta_field: &DMatrix<Complex>,
    gradient: &DMatrix<Complex>,
    delta_coefficients: &DMatrix<Complex>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let forward = cotangent.dotc(delta_field);
    let reverse = gradient.dotc(delta_coefficients);
    let scale = (cotangent.norm() * delta_field.norm()).max(1.0);
    let error = (forward - reverse).norm() / scale;
    if !error.is_finite() || error > 2e-12 {
        return Err(
            format!("sampling adjoint fails Hermitian pairing: {forward} vs {reverse}").into(),
        );
    }
    Ok(error)
}

fn break_even(setup: f64, baseline: f64, application: f64) -> Option<f64> {
    (baseline > application).then(|| (setup.max(0.0) / (baseline - application)).ceil())
}

#[test]
fn prepared_fused_and_adjoint_match_native_core() -> Result<(), Box<dyn std::error::Error>> {
    let vectors: Vec<_> = [[1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.3, 0.4, -0.5]]
        .map(|k| k.map(|v| Complex::new(v, 0.0)))
        .into();
    let labels = vec![0, 1, 0];
    let points = vec![[0.0; 3], [1.0, -2.0, 3.0], [-17.0, 23.0, 41.0]];
    let geometry = vectors
        .iter()
        .zip(&labels)
        .map(|(&k, &label)| {
            plane::polarization(k, label, true).map(|electric| (k.map(|v| v.re), electric))
        })
        .collect::<treams_core::Result<Vec<_>>>()?;
    let (operator, _) = plane::field(vectors.clone(), labels.clone(), points.clone(), None, true)?;
    let row_major = operator.transpose();
    let column_operator = faer::MatRef::from_column_major_slice(
        operator.as_slice(),
        operator.nrows(),
        operator.ncols(),
    );
    let row_operator = faer::MatRef::from_row_major_slice(
        row_major.as_slice(),
        operator.nrows(),
        operator.ncols(),
    );
    for iteration in 0..3 {
        let coefficients = DMatrix::from_fn(vectors.len(), 1, |i, _| {
            Complex::new((i + iteration) as f64 * 0.37, i as f64 * -0.23)
        });
        let (reference, _) = plane::field(
            vectors.clone(),
            labels.clone(),
            points.clone(),
            Some(coefficients.as_slice().to_vec()),
            true,
        )?;
        validate(
            &cpu_prepared_fused(&geometry, &points, &coefficients),
            &reference,
        )?;
        let cotangent = DMatrix::from_fn(operator.nrows(), 1, |i, _| {
            Complex::new(i as f64 * 0.19, (i + iteration) as f64 * -0.13)
        });
        let expected = operator.adjoint() * &cotangent;
        let delta_field = cpu_product(column_operator, &coefficients);
        for layout in [column_operator, row_operator] {
            let gradient = cpu_adjoint(layout, &cotangent);
            validate(&gradient, &expected)?;
            validate_pairing(&cotangent, &delta_field, &gradient, &coefficients)?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let points_count: usize = args.get(1).map_or(Ok(16_384), |s| s.parse())?;
    let modes: usize = args.get(2).map_or(Ok(512), |s| s.parse())?;
    let repetitions: usize = args.get(3).map_or(Ok(7), |s| s.parse())?;
    if points_count == 0 || modes == 0 || repetitions == 0 {
        return Err("points, modes, and repetitions must be positive".into());
    }
    let rows = points_count.checked_mul(3).ok_or("too many points")?;
    let matrix_bytes = rows
        .checked_mul(modes)
        .and_then(|n| n.checked_mul(size_of::<Complex>()))
        .ok_or("operator size overflow")?;
    i32::try_from(rows)?;
    i32::try_from(modes)?;

    // Propagating waves on one unit wave-number shell, incident from the upper hemisphere.
    let vectors: Vec<_> = (0..modes)
        .map(|i| {
            let azimuth = i as f64 * 2.399_963_229_728_653;
            let z = 0.15 + 0.8 * (i as f64 + 0.5) / modes as f64;
            let radial = (1.0 - z * z).sqrt();
            [
                Complex::new(radial * azimuth.cos(), 0.0),
                Complex::new(radial * azimuth.sin(), 0.0),
                Complex::new(z, 0.0),
            ]
        })
        .collect();
    let labels: Vec<_> = (0..modes).map(|i| u8::from(i % 2 == 1)).collect();
    let points: Vec<_> = (0..points_count)
        .map(|i| {
            // An irregular 3D cloud avoids giving a dense-method comparison an
            // artificial advantage over separable Cartesian-grid algorithms.
            let t = i as f64 + 0.5;
            [
                6.4 * (t * 0.754_877_666).fract() - 3.2,
                12.8 * (t * 0.569_840_296).fract() - 6.4,
                2.0 * (t * 0.438_579_021).fract() - 1.0,
            ]
        })
        .collect();
    let coefficients = |iteration: usize| {
        DMatrix::from_fn(modes, 1, |i, _| {
            Complex::new(
                (i as f64 * 0.3 + iteration as f64 * 0.37).sin(),
                (i as f64 * 0.7 - iteration as f64 * 0.11).cos(),
            ) / modes as f64
        })
    };
    let field_cotangent = |iteration: usize| {
        DMatrix::from_fn(rows, 1, |i, _| {
            Complex::new(
                (i as f64 * 0.17 + iteration as f64 * 0.19).sin(),
                (i as f64 * 0.31 - iteration as f64 * 0.29).cos(),
            ) / (rows as f64).sqrt()
        })
    };
    let fused_field = |coefficients: &DMatrix<Complex>| {
        plane::field(
            vectors.clone(),
            labels.clone(),
            points.clone(),
            Some(coefficients.as_slice().to_vec()),
            true,
        )
        .map(|(field, _)| field)
    };
    let (cpu_prepare_geometry_seconds, prepared_geometry) = timed(|| {
        vectors
            .iter()
            .zip(&labels)
            .map(|(&k, &label)| {
                plane::polarization(k, label, true).map(|electric| (k.map(|v| v.re), electric))
            })
            .collect::<treams_core::Result<Vec<_>>>()
    });
    let prepared_geometry = prepared_geometry?;
    let (assembly_seconds, operator) = timed(|| {
        plane::field(vectors.clone(), labels.clone(), points.clone(), None, true)
            .map(|(operator, _)| operator)
    });
    let operator = operator?;
    assert_eq!(operator.shape(), (rows, modes));

    // Both CPU layouts reuse exactly the same values. Packing is paid once, like GPU upload.
    let (cpu_row_major_pack_seconds, row_major) =
        timed(|| DMatrix::from_fn(modes, rows, |column, row| operator[(row, column)]));
    let column_operator = faer::MatRef::from_column_major_slice(operator.as_slice(), rows, modes);
    let row_operator = faer::MatRef::from_row_major_slice(row_major.as_slice(), rows, modes);

    let (context_seconds, gpu) = timed(|| Gpu::new(0));
    let gpu = gpu?;
    let (operator_upload_seconds, resident_operator) = timed(|| {
        let value = gpu.upload(&operator)?;
        gpu.synchronize()?;
        Ok::<_, treams_cuda::gpu::Error>(value)
    });
    let resident_operator = resident_operator?;
    let initial_coefficients = coefficients(0);
    let resident_coefficients = gpu.upload(&initial_coefficients)?;
    gpu.synchronize()?;
    let (cold_application_seconds, initial_field) = timed(|| {
        let value = gpu.matmul(&resident_operator, &resident_coefficients)?;
        gpu.synchronize()?;
        Ok::<_, treams_cuda::gpu::Error>(value)
    });
    let initial_field = initial_field?.download()?;
    let initial_reference = fused_field(&initial_coefficients)?;
    let mut maximum_error = validate(&initial_field, &initial_reference)?;
    maximum_error = maximum_error.max(validate(
        &cpu_product(column_operator, &initial_coefficients),
        &initial_reference,
    )?);

    maximum_error = maximum_error.max(validate(
        &cpu_product(row_operator, &initial_coefficients),
        &initial_reference,
    )?);
    maximum_error = maximum_error.max(validate(
        &cpu_prepared_fused(&prepared_geometry, &points, &initial_coefficients),
        &initial_reference,
    )?);

    let initial_cotangent = field_cotangent(0);
    let initial_gradient = cpu_adjoint(column_operator, &initial_cotangent);
    let mut maximum_adjoint_error = validate(
        &cpu_adjoint(row_operator, &initial_cotangent),
        &initial_gradient,
    )?;
    let resident_cotangent = gpu.upload(&initial_cotangent)?;
    let initial_gpu_gradient =
        gpu.matmul_with_adjoint(&resident_operator, &resident_cotangent, true, false)?;
    assert_eq!(initial_gpu_gradient.shape(), (modes, 1));
    let adjoint_logical_device_buffer_bytes =
        resident_operator.bytes() + resident_cotangent.bytes() + initial_gpu_gradient.bytes();
    let initial_gpu_gradient = initial_gpu_gradient.download()?;
    maximum_adjoint_error =
        maximum_adjoint_error.max(validate(&initial_gpu_gradient, &initial_gradient)?);
    let mut maximum_adjoint_pairing_error = validate_pairing(
        &initial_cotangent,
        &cpu_product(column_operator, &initial_coefficients),
        &initial_gpu_gradient,
        &initial_coefficients,
    )?;

    let mut samples = Vec::with_capacity(repetitions);
    for iteration in 1..=repetitions {
        // Coefficient generation is shared application work and excluded from both backends.
        let c = coefficients(iteration);
        let (fused_seconds, reference) = timed(|| fused_field(&c));
        let reference = reference?;
        let (cpu_prepared_fused_seconds, cpu_field) =
            timed(|| cpu_prepared_fused(&prepared_geometry, &points, &c));
        maximum_error = maximum_error.max(validate(&cpu_field, &reference)?);
        black_box(cpu_field);
        let (cpu_column_major_seconds, cpu_field) = timed(|| cpu_product(column_operator, &c));
        maximum_error = maximum_error.max(validate(&cpu_field, &reference)?);
        black_box(cpu_field);
        let (cpu_row_major_seconds, cpu_field) = timed(|| cpu_product(row_operator, &c));
        maximum_error = maximum_error.max(validate(&cpu_field, &reference)?);
        black_box(cpu_field);

        let (coefficient_upload_seconds, resident_c) = timed(|| {
            let value = gpu.upload(&c)?;
            gpu.synchronize()?;
            Ok::<_, treams_cuda::gpu::Error>(value)
        });
        let resident_c = resident_c?;
        let (gpu_resident_seconds, field) = timed(|| {
            let value = gpu.matmul(&resident_operator, &resident_c)?;
            gpu.synchronize()?;
            Ok::<_, treams_cuda::gpu::Error>(value)
        });
        let field = field?;
        let (field_download_seconds, downloaded) = timed(|| field.download());
        maximum_error = maximum_error.max(validate(&downloaded?, &reference)?);

        let (gpu_roundtrip_seconds, downloaded) =
            timed(|| gpu.matmul(&resident_operator, &gpu.upload(&c)?)?.download());
        maximum_error = maximum_error.max(validate(&downloaded?, &reference)?);

        // Cotangent generation and invariant checks are outside application timings.
        let g = field_cotangent(iteration);
        let delta_field = cpu_product(column_operator, &c);
        let (cpu_adjoint_column_major_seconds, gradient_reference) =
            timed(|| cpu_adjoint(column_operator, &g));
        maximum_adjoint_pairing_error = maximum_adjoint_pairing_error.max(validate_pairing(
            &g,
            &delta_field,
            &gradient_reference,
            &c,
        )?);
        let (cpu_adjoint_row_major_seconds, gradient) = timed(|| cpu_adjoint(row_operator, &g));
        maximum_adjoint_error =
            maximum_adjoint_error.max(validate(&gradient, &gradient_reference)?);
        black_box(gradient);
        let (field_cotangent_upload_seconds, resident_g) = timed(|| {
            let value = gpu.upload(&g)?;
            gpu.synchronize()?;
            Ok::<_, treams_cuda::gpu::Error>(value)
        });
        let resident_g = resident_g?;
        let (gpu_adjoint_resident_seconds, gradient) = timed(|| {
            let value = gpu.matmul_with_adjoint(&resident_operator, &resident_g, true, false)?;
            gpu.synchronize()?;
            Ok::<_, treams_cuda::gpu::Error>(value)
        });
        let gradient = gradient?;
        let (coefficient_gradient_download_seconds, downloaded) = timed(|| gradient.download());
        let downloaded = downloaded?;
        maximum_adjoint_error =
            maximum_adjoint_error.max(validate(&downloaded, &gradient_reference)?);
        maximum_adjoint_pairing_error =
            maximum_adjoint_pairing_error.max(validate_pairing(&g, &delta_field, &downloaded, &c)?);
        let (gpu_adjoint_roundtrip_seconds, downloaded) = timed(|| {
            gpu.matmul_with_adjoint(&resident_operator, &gpu.upload(&g)?, true, false)?
                .download()
        });
        let downloaded = downloaded?;
        maximum_adjoint_error =
            maximum_adjoint_error.max(validate(&downloaded, &gradient_reference)?);
        maximum_adjoint_pairing_error =
            maximum_adjoint_pairing_error.max(validate_pairing(&g, &delta_field, &downloaded, &c)?);
        samples.push(SampleSeconds {
            cpu_fused: fused_seconds,
            cpu_prepared_fused: cpu_prepared_fused_seconds,
            cpu_column_major: cpu_column_major_seconds,
            cpu_row_major: cpu_row_major_seconds,
            coefficient_upload: coefficient_upload_seconds,
            gpu_resident: gpu_resident_seconds,
            field_download: field_download_seconds,
            gpu_roundtrip: gpu_roundtrip_seconds,
            cpu_adjoint_column_major: cpu_adjoint_column_major_seconds,
            cpu_adjoint_row_major: cpu_adjoint_row_major_seconds,
            field_cotangent_upload: field_cotangent_upload_seconds,
            gpu_adjoint_resident: gpu_adjoint_resident_seconds,
            coefficient_gradient_download: coefficient_gradient_download_seconds,
            gpu_adjoint_roundtrip: gpu_adjoint_roundtrip_seconds,
        });
    }
    let time = |select: fn(&SampleSeconds) -> f64| median(samples.iter().map(select));
    let cpu_column_major = time(|sample| sample.cpu_column_major);
    let cpu_row_major = time(|sample| sample.cpu_row_major);
    let (cpu_cached, cpu_layout, cpu_pack) = if cpu_row_major < cpu_column_major {
        (cpu_row_major, "row-major", cpu_row_major_pack_seconds)
    } else {
        (cpu_column_major, "column-major", 0.0)
    };
    let cpu_fused = time(|sample| sample.cpu_fused);
    let cpu_prepared_fused = time(|sample| sample.cpu_prepared_fused);
    let (cpu_strongest, cpu_baseline, cpu_setup) = if cpu_prepared_fused < cpu_cached.min(cpu_fused)
    {
        (
            cpu_prepared_fused,
            "prepared-fused",
            cpu_prepare_geometry_seconds,
        )
    } else if cpu_fused < cpu_cached {
        (cpu_fused, "native-core-fused", 0.0)
    } else {
        (cpu_cached, "cached-operator", assembly_seconds + cpu_pack)
    };
    let gpu_resident = time(|sample| sample.gpu_resident);
    let gpu_roundtrip = time(|sample| sample.gpu_roundtrip);
    let gpu_cold_setup = context_seconds
        + operator_upload_seconds
        + (cold_application_seconds - gpu_resident).max(0.0);
    let cpu_adjoint_column_major = time(|sample| sample.cpu_adjoint_column_major);
    let cpu_adjoint_row_major = time(|sample| sample.cpu_adjoint_row_major);
    let (cpu_adjoint_cached, cpu_adjoint_layout) =
        if cpu_adjoint_row_major < cpu_adjoint_column_major {
            (cpu_adjoint_row_major, "row-major")
        } else {
            (cpu_adjoint_column_major, "column-major")
        };
    let gpu_adjoint_resident = time(|sample| sample.gpu_adjoint_resident);
    let gpu_adjoint_roundtrip = time(|sample| sample.gpu_adjoint_roundtrip);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "device": gpu.name()?,
            "precision": "complex128",
            "point_geometry": "deterministic irregular three-dimensional cloud",
            "points": points_count,
            "modes": modes,
            "operator_shape": [rows, modes],
            "operator_bytes": matrix_bytes,
            "coefficient_bytes": modes * size_of::<Complex>(),
            "field_bytes": rows * size_of::<Complex>(),
            "cpu_threads": std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into()),
            "actual_cpu_threads": faer::get_global_parallelism().degree(),
            "process_peak_resident_kib": peak_resident_kib(),
            "assembly_seconds": assembly_seconds,
            "cpu_row_major_pack_seconds": cpu_row_major_pack_seconds,
            "cpu_prepare_geometry_seconds": cpu_prepare_geometry_seconds,
            "selected_cpu_layout": cpu_layout,
            "selected_cpu_baseline": cpu_baseline,
            "median_cpu_column_major_seconds": cpu_column_major,
            "median_cpu_row_major_seconds": cpu_row_major,
            "context_seconds": context_seconds,
            "operator_upload_seconds": operator_upload_seconds,
            "cold_application_seconds": cold_application_seconds,
            "max_abs_error": maximum_error,
            "median_cpu_fused_seconds": cpu_fused,
            "median_cpu_prepared_fused_seconds": cpu_prepared_fused,
            "median_cpu_cached_seconds": cpu_cached,
            "median_cpu_strongest_seconds": cpu_strongest,
            "strongest_cpu_setup_seconds": cpu_setup,
            "median_gpu_resident_seconds": gpu_resident,
            "median_gpu_roundtrip_seconds": gpu_roundtrip,
            "resident_speedup_vs_cpu_cached": cpu_cached / gpu_resident,
            "roundtrip_speedup_vs_cpu_cached": cpu_cached / gpu_roundtrip,
            "resident_speedup_vs_cpu_strongest": cpu_strongest / gpu_resident,
            "roundtrip_speedup_vs_cpu_strongest": cpu_strongest / gpu_roundtrip,
            "cpu_nominal_operator_read_gb_per_second": matrix_bytes as f64 / cpu_cached / 1e9,
            "gpu_nominal_operator_read_gb_per_second": matrix_bytes as f64 / gpu_resident / 1e9,
            "gpu_upload_break_even_applications_vs_cpu_cached": break_even(operator_upload_seconds - cpu_pack, cpu_cached, gpu_roundtrip),
            "gpu_cold_break_even_applications_vs_cpu_cached": break_even(gpu_cold_setup - cpu_pack, cpu_cached, gpu_roundtrip),
            "gpu_upload_break_even_applications_vs_cpu_strongest": break_even(assembly_seconds + operator_upload_seconds - cpu_setup, cpu_strongest, gpu_roundtrip),
            "gpu_cold_break_even_applications_vs_cpu_strongest": break_even(assembly_seconds + gpu_cold_setup - cpu_setup, cpu_strongest, gpu_roundtrip),
            "cpu_cache_break_even_applications_vs_fused": break_even(assembly_seconds + cpu_pack, cpu_fused, cpu_cached),
            "gpu_cache_break_even_applications_vs_fused": break_even(assembly_seconds + gpu_cold_setup, cpu_fused, gpu_roundtrip),
            "selected_cpu_adjoint_layout": cpu_adjoint_layout,
            "median_cpu_adjoint_column_major_seconds": cpu_adjoint_column_major,
            "median_cpu_adjoint_row_major_seconds": cpu_adjoint_row_major,
            "median_cpu_adjoint_cached_seconds": cpu_adjoint_cached,
            "median_field_cotangent_upload_seconds": time(|sample| sample.field_cotangent_upload),
            "median_gpu_adjoint_resident_seconds": gpu_adjoint_resident,
            "median_coefficient_gradient_download_seconds": time(|sample| sample.coefficient_gradient_download),
            "median_gpu_adjoint_roundtrip_seconds": gpu_adjoint_roundtrip,
            "adjoint_resident_speedup_vs_cpu_cached": cpu_adjoint_cached / gpu_adjoint_resident,
            "adjoint_roundtrip_speedup_vs_cpu_cached": cpu_adjoint_cached / gpu_adjoint_roundtrip,
            "max_adjoint_abs_error": maximum_adjoint_error,
            "max_adjoint_normalized_pairing_error": maximum_adjoint_pairing_error,
            "adjoint_shared_operator_bytes": resident_operator.bytes(),
            "adjoint_extra_operator_bytes": 0,
            "adjoint_logical_device_buffer_bytes": adjoint_logical_device_buffer_bytes,
            "adjoint_scope": "Fixed-operator coefficient pullback F^H g reuses the same complex128 device operator as forward sampling. Each application uses a different host field cotangent. CPU compares both cached layouts using conjugate-transpose views; a fused CPU adjoint is not measured. Resident GPU time includes coefficient-gradient allocation and synchronization. Cotangent upload and coefficient-gradient download are separate; round trip includes both. Logical device buffer bytes count one shared F, one field cotangent, and one coefficient gradient; no transposed F is allocated or uploaded. This is a buffer accounting proof, not measured peak VRAM, and excludes library workspaces, allocator state, and other benchmark buffers. GPU gradients are checked against CPU gradients and the complex Hermitian pairing with a CPU forward coefficient perturbation. Operator/geometry derivatives and full-workflow speedups are outside this measurement.",
            "samples": samples.iter().enumerate().map(|(i, sample)| sample.json(i + 1)).collect::<Vec<_>>(),
            "scope": "Identical fixed physical sampling operator F is cached on both CPU and GPU. Each measured application uses a different host coefficient vector. Resident GPU time includes output allocation and synchronization; round trip adds coefficient upload and full field download. CPU output allocation is included. Cached CPU metrics use the faster column-major or prepacked row-major median at the selected thread count. The strongest CPU baseline is the minimum of cached, native-core fused, and prepared fused medians. Prepared fused caches shared-core polarization and real wavevectors, then includes coefficient weighting and a parallel point sum with sin_cos in each application; this specialization is exact for this benchmark's real wavevectors. Operator assembly, CPU row packing, CPU geometry preparation, and initial upload are excluded from application timings and reported separately. Nominal bandwidth counts one operator read; it is not a hardware-counter measurement. VmHWM includes both host operator layouts and transfer staging, not just a production cached operator. Break-even counts subtract the selected CPU baseline's setup cost; row packing is credited only when cached row-major is selected. Strongest-baseline break-even includes operator assembly unless shared with the selected cached CPU baseline. Cold setup includes first-call excess above warm resident time. Counts are steady-median estimates and are null when no positive advantage exists."
        }))?
    );
    Ok(())
}

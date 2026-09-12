//! Run on an idle GPU host; timings include point uploads and result downloads.
#![allow(clippy::cast_precision_loss, clippy::indexing_slicing)]

#[cfg(target_os = "linux")]
mod qualification {
    use std::hint::black_box;
    use std::time::Instant;
    use treams_core::{Complex, plane};
    use treams_cuda_tile::PlaneWaves;

    fn median(mut samples: Vec<f64>) -> f64 {
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }

    pub(super) fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args().skip(1);
        let modes: usize = args.next().unwrap_or_else(|| "64".into()).parse()?;
        let samples: usize = args.next().unwrap_or_else(|| "16384".into()).parse()?;
        let vectors: Vec<_> = (0..modes)
            .map(|i| {
                let a = i as f64 * 0.371;
                [
                    Complex::new(a.cos(), 0.01),
                    Complex::new(a.sin(), -0.003),
                    Complex::new(0.7, 0.005),
                ]
            })
            .collect();
        let polarizations: Vec<_> = (0..modes).map(|i| u8::from(i % 2 == 1)).collect();
        let coefficients: Vec<_> = (0..modes)
            .map(|i| Complex::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()) / modes as f64)
            .collect();
        let points: Vec<_> = (0..samples)
            .map(|i| {
                let a = i as f64 * 0.013;
                [2.0 * a.cos(), 1.5 * a.sin(), (a * 0.2).sin()]
            })
            .collect();
        let cpu = || {
            plane::field(
                vectors.clone(),
                polarizations.clone(),
                points.clone(),
                Some(coefficients.clone()),
                true,
            )
            .map(|(value, _)| value)
        };
        let cold = Instant::now();
        let gpu = PlaneWaves::new(0, &vectors, &polarizations, &coefficients, true)?;
        let actual = gpu.evaluate(&points)?;
        let cold_seconds = cold.elapsed().as_secs_f64();
        let expected = cpu()?;
        let max_abs_error = actual
            .iter()
            .flatten()
            .zip(expected.as_slice())
            .map(|(a, e)| (*a - e).norm())
            .fold(0.0, f64::max);
        let reference_scale = expected.iter().map(|z| z.norm()).fold(0.0, f64::max);
        assert!(
            max_abs_error <= 2e-12 * (1.0 + reference_scale),
            "GPU/core mismatch: {max_abs_error}"
        );
        let mut cpu_times = Vec::new();
        let mut gpu_times = Vec::new();
        for repeat in 0..7 {
            if repeat % 2 == 0 {
                let start = Instant::now();
                black_box(cpu()?);
                cpu_times.push(start.elapsed().as_secs_f64());
            }
            let start = Instant::now();
            black_box(gpu.evaluate(&points)?);
            gpu_times.push(start.elapsed().as_secs_f64());
            if repeat % 2 != 0 {
                let start = Instant::now();
                black_box(cpu()?);
                cpu_times.push(start.elapsed().as_secs_f64());
            }
        }
        let cpu_median = median(cpu_times);
        let gpu_median = median(gpu_times);
        let device_array_bytes = 128 * modes + 96 * samples;
        println!(
            "{{\"workload\":\"weighted-plane-fields\",\"precision\":\"complex128\",\"modes\":{modes},\"points\":{samples},\"cpu_seconds\":{cpu_median},\"gpu_seconds\":{gpu_median},\"speedup\":{},\"cold_gpu_seconds\":{cold_seconds},\"max_abs_error\":{max_abs_error},\"device_array_bytes\":{device_array_bytes},\"timing\":\"median of 7; warm JIT; point transfer and result transfer included; mode expansion retained on device\"}}",
            cpu_median / gpu_median
        );
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    {
        qualification::run()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err("CUDA Tile qualification requires Linux".into())
    }
}

//! Controlled f64 kernel probes; reports transfers separately from retained-buffer launches.
#![allow(clippy::cast_precision_loss, clippy::wildcard_imports)]

#[cfg(all(feature = "cuda-tile", target_os = "linux"))]
mod probe {
    use cutile::prelude::*;
    use rayon::prelude::*;
    use std::hint::black_box;
    use std::time::Instant;
    use treams_core::{Complex, plane};

    #[cutile::module]
    mod kernels {
        use cutile::core::*;
        #[cutile::entry()]
        fn baseline<const B: i32>(
            real: &mut Tensor<f64, { [B, 4] }>,
            imag: &mut Tensor<f64, { [B, 4] }>,
            points: &Tensor<f64, { [-1, -1] }>,
            vectors: &Tensor<f64, { [-1, -1] }>,
            pol_real: &Tensor<f64, { [-1, -1] }>,
            pol_imag: &Tensor<f64, { [-1, -1] }>,
        ) {
            let block = get_tile_block_id().0;
            let r = points.partition(shape![B, 1]);
            let x = r.load([block, 0]);
            let y = r.load([block, 1]);
            let z = r.load([block, 2]);
            let k = vectors.partition(shape![1, 1]);
            let pr = pol_real.partition(shape![1, 4]);
            let pi = pol_imag.partition(shape![1, 4]);
            let mut result_real = 0.0f64.broadcast(shape![B, 4]);
            let mut result_imag = 0.0f64.broadcast(shape![B, 4]);
            for m in 0..num_tiles(&k, 0) {
                let phase = k.load([m, 0]).broadcast(shape![B, 1]) * x
                    + k.load([m, 2]).broadcast(shape![B, 1]) * y
                    + k.load([m, 4]).broadcast(shape![B, 1]) * z;
                let decay = k.load([m, 1]).broadcast(shape![B, 1]) * x
                    + k.load([m, 3]).broadcast(shape![B, 1]) * y
                    + k.load([m, 5]).broadcast(shape![B, 1]) * z;
                let magnitude = exp(negf(decay));
                let phase_real = (magnitude * cos(phase)).broadcast(shape![B, 4]);
                let phase_imag = (magnitude * sin(phase)).broadcast(shape![B, 4]);
                let a = pr.load([m, 0]).broadcast(shape![B, 4]);
                let b = pi.load([m, 0]).broadcast(shape![B, 4]);
                result_real = result_real + a * phase_real - b * phase_imag;
                result_imag = result_imag + a * phase_imag + b * phase_real;
            }
            real.store(result_real);
            imag.store(result_imag);
        }
        #[cutile::entry()]
        fn component_scalar<const B: i32>(
            real: &mut Tensor<f64, { [B, 4] }>,
            imag: &mut Tensor<f64, { [B, 4] }>,
            points: &Tensor<f64, { [-1, -1] }>,
            vectors: &Tensor<f64, { [-1, -1] }>,
            pol_real: &Tensor<f64, { [-1, -1] }>,
            pol_imag: &Tensor<f64, { [-1, -1] }>,
        ) {
            let block = get_tile_block_id().0;
            let r = points.partition(shape![B, 1]);
            let x = r.load([block, 0]).broadcast(shape![B, 1]);
            let y = r.load([block, 1]).broadcast(shape![B, 1]);
            let z = r.load([block, 2]).broadcast(shape![B, 1]);
            let k = vectors.partition(shape![1, 1]);
            let pr = pol_real.partition(shape![1, 1]);
            let pi = pol_imag.partition(shape![1, 1]);
            let mut xr = 0.0f64.broadcast(shape![B, 1]);
            let mut xi = 0.0f64.broadcast(shape![B, 1]);
            let mut yr = 0.0f64.broadcast(shape![B, 1]);
            let mut yi = 0.0f64.broadcast(shape![B, 1]);
            let mut zr = 0.0f64.broadcast(shape![B, 1]);
            let mut zi = 0.0f64.broadcast(shape![B, 1]);
            for m in 0..num_tiles(&k, 0) {
                let phase = k.load([m, 0]).transpose().broadcast(shape![B, 1]) * x
                    + k.load([m, 2]).transpose().broadcast(shape![B, 1]) * y
                    + k.load([m, 4]).transpose().broadcast(shape![B, 1]) * z;
                let decay = k.load([m, 1]).transpose().broadcast(shape![B, 1]) * x
                    + k.load([m, 3]).transpose().broadcast(shape![B, 1]) * y
                    + k.load([m, 5]).transpose().broadcast(shape![B, 1]) * z;
                let magnitude = exp(negf(decay));
                let phase_real = magnitude * cos(phase);
                let phase_imag = magnitude * sin(phase);
                let a = pr.load([m, 0]).transpose().broadcast(shape![B, 1]);
                let b = pi.load([m, 0]).transpose().broadcast(shape![B, 1]);
                xr = xr + a * phase_real - b * phase_imag;
                xi = xi + a * phase_imag + b * phase_real;
                let a = pr.load([m, 1]).transpose().broadcast(shape![B, 1]);
                let b = pi.load([m, 1]).transpose().broadcast(shape![B, 1]);
                yr = yr + a * phase_real - b * phase_imag;
                yi = yi + a * phase_imag + b * phase_real;
                let a = pr.load([m, 2]).transpose().broadcast(shape![B, 1]);
                let b = pi.load([m, 2]).transpose().broadcast(shape![B, 1]);
                zr = zr + a * phase_real - b * phase_imag;
                zi = zi + a * phase_imag + b * phase_real;
            }
            let zero = 0.0f64.broadcast(shape![B, 1]);
            let real_xy: Tile<f64, { [B, 2] }> = cat(xr, yr, 1);
            let real_z0: Tile<f64, { [B, 2] }> = cat(zr, zero, 1);
            let imag_xy: Tile<f64, { [B, 2] }> = cat(xi, yi, 1);
            let imag_z0: Tile<f64, { [B, 2] }> = cat(zi, zero, 1);
            let real_out: Tile<f64, { [B, 4] }> = cat(real_xy, real_z0, 1);
            let imag_out: Tile<f64, { [B, 4] }> = cat(imag_xy, imag_z0, 1);
            real.store(real_out);
            imag.store(imag_out);
        }

        #[cutile::entry()]
        fn component_fma<const B: i32>(
            real: &mut Tensor<f64, { [B, 4] }>,
            imag: &mut Tensor<f64, { [B, 4] }>,
            points: &Tensor<f64, { [-1, -1] }>,
            vectors: &Tensor<f64, { [-1, -1] }>,
            pol_real: &Tensor<f64, { [-1, -1] }>,
            pol_imag: &Tensor<f64, { [-1, -1] }>,
            real_k: bool,
        ) {
            let block = get_tile_block_id().0;
            let r = points.partition(shape![B, 1]);
            let x = r.load([block, 0]).broadcast(shape![B, 1]);
            let y = r.load([block, 1]).broadcast(shape![B, 1]);
            let z = r.load([block, 2]).broadcast(shape![B, 1]);
            let k = vectors.partition(shape![1, 1]);
            let pr = pol_real.partition(shape![1, 1]);
            let pi = pol_imag.partition(shape![1, 1]);
            let mut xr = 0.0f64.broadcast(shape![B, 1]);
            let mut xi = 0.0f64.broadcast(shape![B, 1]);
            let mut yr = 0.0f64.broadcast(shape![B, 1]);
            let mut yi = 0.0f64.broadcast(shape![B, 1]);
            let mut zr = 0.0f64.broadcast(shape![B, 1]);
            let mut zi = 0.0f64.broadcast(shape![B, 1]);
            for m in 0..num_tiles(&k, 0) {
                let phase = k.load([m, 0]).transpose().broadcast(shape![B, 1]) * x
                    + k.load([m, 2]).transpose().broadcast(shape![B, 1]) * y
                    + k.load([m, 4]).transpose().broadcast(shape![B, 1]) * z;
                let magnitude: Tile<f64, { [B, 1] }> = if real_k {
                    1.0f64.broadcast(shape![B, 1])
                } else {
                    let decay = k.load([m, 1]).transpose().broadcast(shape![B, 1]) * x
                        + k.load([m, 3]).transpose().broadcast(shape![B, 1]) * y
                        + k.load([m, 5]).transpose().broadcast(shape![B, 1]) * z;
                    exp(negf(decay))
                };
                let phase_real = magnitude * cos(phase);
                let phase_imag = magnitude * sin(phase);
                let a = pr.load([m, 0]).transpose().broadcast(shape![B, 1]);
                let b = pi.load([m, 0]).transpose().broadcast(shape![B, 1]);
                xr = fma(a, phase_real, xr, rounding::NearestEven, ftz::Disabled);
                xr = fma(
                    negf(b),
                    phase_imag,
                    xr,
                    rounding::NearestEven,
                    ftz::Disabled,
                );
                xi = fma(a, phase_imag, xi, rounding::NearestEven, ftz::Disabled);
                xi = fma(b, phase_real, xi, rounding::NearestEven, ftz::Disabled);
                let a = pr.load([m, 1]).transpose().broadcast(shape![B, 1]);
                let b = pi.load([m, 1]).transpose().broadcast(shape![B, 1]);
                yr = fma(a, phase_real, yr, rounding::NearestEven, ftz::Disabled);
                yr = fma(
                    negf(b),
                    phase_imag,
                    yr,
                    rounding::NearestEven,
                    ftz::Disabled,
                );
                yi = fma(a, phase_imag, yi, rounding::NearestEven, ftz::Disabled);
                yi = fma(b, phase_real, yi, rounding::NearestEven, ftz::Disabled);
                let a = pr.load([m, 2]).transpose().broadcast(shape![B, 1]);
                let b = pi.load([m, 2]).transpose().broadcast(shape![B, 1]);
                zr = fma(a, phase_real, zr, rounding::NearestEven, ftz::Disabled);
                zr = fma(
                    negf(b),
                    phase_imag,
                    zr,
                    rounding::NearestEven,
                    ftz::Disabled,
                );
                zi = fma(a, phase_imag, zi, rounding::NearestEven, ftz::Disabled);
                zi = fma(b, phase_real, zi, rounding::NearestEven, ftz::Disabled);
            }
            let zero = 0.0f64.broadcast(shape![B, 1]);
            let real_xy: Tile<f64, { [B, 2] }> = cat(xr, yr, 1);
            let real_z0: Tile<f64, { [B, 2] }> = cat(zr, zero, 1);
            let imag_xy: Tile<f64, { [B, 2] }> = cat(xi, yi, 1);
            let imag_z0: Tile<f64, { [B, 2] }> = cat(zi, zero, 1);
            let real_out: Tile<f64, { [B, 4] }> = cat(real_xy, real_z0, 1);
            let imag_out: Tile<f64, { [B, 4] }> = cat(imag_xy, imag_z0, 1);
            real.store(real_out);
            imag.store(imag_out);
        }

        #[cutile::entry()]
        fn mode_tiled<const B: i32>(
            real: &mut Tensor<f64, { [B, 4] }>,
            imag: &mut Tensor<f64, { [B, 4] }>,
            points: &Tensor<f64, { [-1, -1] }>,
            vectors: &Tensor<f64, { [-1, -1] }>,
            pol_real: &Tensor<f64, { [-1, -1] }>,
            pol_imag: &Tensor<f64, { [-1, -1] }>,
        ) {
            let block = get_tile_block_id().0;
            let r = points.partition(shape![B, 1]);
            let x = r.load([block, 0]).broadcast(shape![B, 16]);
            let y = r.load([block, 1]).broadcast(shape![B, 16]);
            let z = r.load([block, 2]).broadcast(shape![B, 16]);
            let k = vectors.partition(shape![16, 1]);
            let pr = pol_real.partition(shape![16, 1]);
            let pi = pol_imag.partition(shape![16, 1]);
            let mut xr = 0.0f64.broadcast(shape![B, 1]);
            let mut xi = 0.0f64.broadcast(shape![B, 1]);
            let mut yr = 0.0f64.broadcast(shape![B, 1]);
            let mut yi = 0.0f64.broadcast(shape![B, 1]);
            let mut zr = 0.0f64.broadcast(shape![B, 1]);
            let mut zi = 0.0f64.broadcast(shape![B, 1]);
            for m in 0..num_tiles(&k, 0) {
                let phase = k.load([m, 0]).transpose().broadcast(shape![B, 16]) * x
                    + k.load([m, 2]).transpose().broadcast(shape![B, 16]) * y
                    + k.load([m, 4]).transpose().broadcast(shape![B, 16]) * z;
                let decay = k.load([m, 1]).transpose().broadcast(shape![B, 16]) * x
                    + k.load([m, 3]).transpose().broadcast(shape![B, 16]) * y
                    + k.load([m, 5]).transpose().broadcast(shape![B, 16]) * z;
                let magnitude = exp(negf(decay));
                let phase_real = magnitude * cos(phase);
                let phase_imag = magnitude * sin(phase);
                let a = pr.load([m, 0]).transpose().broadcast(shape![B, 16]);
                let b = pi.load([m, 0]).transpose().broadcast(shape![B, 16]);
                let partial: Tile<f64, { [B] }> = reduce_sum(a * phase_real - b * phase_imag, 1);
                xr = xr + partial.reshape(shape![B, 1]);
                let partial: Tile<f64, { [B] }> = reduce_sum(a * phase_imag + b * phase_real, 1);
                xi = xi + partial.reshape(shape![B, 1]);
                let a = pr.load([m, 1]).transpose().broadcast(shape![B, 16]);
                let b = pi.load([m, 1]).transpose().broadcast(shape![B, 16]);
                let partial: Tile<f64, { [B] }> = reduce_sum(a * phase_real - b * phase_imag, 1);
                yr = yr + partial.reshape(shape![B, 1]);
                let partial: Tile<f64, { [B] }> = reduce_sum(a * phase_imag + b * phase_real, 1);
                yi = yi + partial.reshape(shape![B, 1]);
                let a = pr.load([m, 2]).transpose().broadcast(shape![B, 16]);
                let b = pi.load([m, 2]).transpose().broadcast(shape![B, 16]);
                let partial: Tile<f64, { [B] }> = reduce_sum(a * phase_real - b * phase_imag, 1);
                zr = zr + partial.reshape(shape![B, 1]);
                let partial: Tile<f64, { [B] }> = reduce_sum(a * phase_imag + b * phase_real, 1);
                zi = zi + partial.reshape(shape![B, 1]);
            }
            let zero = 0.0f64.broadcast(shape![B, 1]);
            let real_xy: Tile<f64, { [B, 2] }> = cat(xr, yr, 1);
            let real_z0: Tile<f64, { [B, 2] }> = cat(zr, zero, 1);
            let imag_xy: Tile<f64, { [B, 2] }> = cat(xi, yi, 1);
            let imag_z0: Tile<f64, { [B, 2] }> = cat(zi, zero, 1);
            let real_out: Tile<f64, { [B, 4] }> = cat(real_xy, real_z0, 1);
            let imag_out: Tile<f64, { [B, 4] }> = cat(imag_xy, imag_z0, 1);
            real.store(real_out);
            imag.store(imag_out);
        }
    }

    fn median(mut samples: [f64; 7]) -> f64 {
        samples.sort_by(f64::total_cmp);
        samples[3]
    }

    pub(super) fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args().skip(1);
        let modes: usize = args.next().unwrap_or_else(|| "256".into()).parse()?;
        let count: usize = args.next().unwrap_or_else(|| "131072".into()).parse()?;
        let selected: Option<u8> = args.next().map(|a| a.parse()).transpose()?;
        let lossless = args.next().as_deref() == Some("lossless");
        let force_general = args.next().as_deref() == Some("general");
        if modes == 0 || count == 0 || selected.is_some_and(|v| v > 3) {
            return Err("require positive modes and points; variant is 0, 1, 2, or 3".into());
        }
        i32::try_from(modes.checked_mul(8).ok_or("mode array too large")?)?;
        i32::try_from(count.checked_mul(4).ok_or("point array too large")?)?;
        let im = if lossless { 0.0 } else { 1.0 };
        let vectors: Vec<_> = (0..modes)
            .map(|i| {
                let a = i as f64 * 0.371;
                [
                    Complex::new(a.cos(), 0.01 * im),
                    Complex::new(a.sin(), -0.003 * im),
                    Complex::new(0.7, 0.005 * im),
                ]
            })
            .collect();
        // A host-validated, uniform branch; no lossy or evanescent mode uses the shortcut.
        let real_k_specialized = !force_general && vectors.iter().flatten().all(|k| k.im == 0.0);
        let labels: Vec<_> = (0..modes).map(|i| u8::from(i % 2 == 1)).collect();
        let coefficients: Vec<_> = (0..modes)
            .map(|i| Complex::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()) / modes as f64)
            .collect();
        let points: Vec<_> = (0..count)
            .map(|i| {
                let a = i as f64 * 0.013;
                [2.0 * a.cos(), 1.5 * a.sin(), (a * 0.2).sin()]
            })
            .collect();
        let electric = vectors
            .iter()
            .zip(&labels)
            .zip(&coefficients)
            .map(|((&k, &p), &c)| plane::polarization(k, p, true).map(|v| v.map(|z| z * c)))
            .collect::<treams_core::Result<Vec<_>>>()?;
        let wave_data = Arc::new(
            vectors
                .iter()
                .flat_map(|k| {
                    [
                        k[0].re, k[0].im, k[1].re, k[1].im, k[2].re, k[2].im, 0.0, 0.0,
                    ]
                })
                .collect::<Vec<_>>(),
        );
        let point_data = Arc::new(
            points
                .iter()
                .flat_map(|p| [p[0], p[1], p[2], 0.0])
                .collect::<Vec<_>>(),
        );
        let real_data = Arc::new(
            electric
                .iter()
                .flat_map(|p| [p[0].re, p[1].re, p[2].re, 0.0])
                .collect::<Vec<_>>(),
        );
        let imag_data = Arc::new(
            electric
                .iter()
                .flat_map(|p| [p[0].im, p[1].im, p[2].im, 0.0])
                .collect::<Vec<_>>(),
        );
        let cpu_original = || {
            plane::field(
                vectors.clone(),
                labels.clone(),
                points.clone(),
                Some(coefficients.clone()),
                true,
            )
            .map(|(value, _)| value)
        };
        let cpu_preweighted = || {
            points
                .par_iter()
                .map(|&r| {
                    let mut result = [Complex::default(); 3];
                    for (&k, p) in vectors.iter().zip(&electric) {
                        let angle = k[0].re * r[0] + k[1].re * r[1] + k[2].re * r[2];
                        let (sine, cosine) = angle.sin_cos();
                        let scale = if real_k_specialized {
                            1.0
                        } else {
                            let attenuation = k[0].im * r[0] + k[1].im * r[1] + k[2].im * r[2];
                            (-attenuation).exp()
                        };
                        let phase = Complex::new(scale * cosine, scale * sine);
                        for (value, &coefficient) in result.iter_mut().zip(p) {
                            *value += coefficient * phase;
                        }
                    }
                    result
                })
                .collect::<Vec<_>>()
        };
        let expected = cpu_original()?;
        let preweighted = cpu_preweighted();
        for (a, e) in preweighted.iter().flatten().zip(expected.as_slice()) {
            if (*a - e).norm() > 2e-12 * (1.0 + e.norm()) {
                return Err("preweighted CPU oracle mismatch".into());
            }
        }
        let mut cpu_original_times = [0.0; 7];
        let mut cpu_preweighted_times = [0.0; 7];
        for (original, preweighted) in cpu_original_times
            .iter_mut()
            .zip(&mut cpu_preweighted_times)
        {
            let start = Instant::now();
            black_box(cpu_original()?);
            *original = start.elapsed().as_secs_f64();
            let start = Instant::now();
            black_box(cpu_preweighted());
            *preweighted = start.elapsed().as_secs_f64();
        }
        let cpu_original_seconds = median(cpu_original_times);
        let cpu_preweighted_seconds = median(cpu_preweighted_times);
        let start = Instant::now();
        let device = Device::new(0)?;
        let stream = device.new_stream()?;
        let context_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let vectors = api::copy_host_vec_to_device(&wave_data)
            .sync_on(&stream)?
            .reshape(&[modes, 8])?;
        let pr = api::copy_host_vec_to_device(&real_data)
            .sync_on(&stream)?
            .reshape(&[modes, 4])?;
        let pi = api::copy_host_vec_to_device(&imag_data)
            .sync_on(&stream)?
            .reshape(&[modes, 4])?;
        let mode_upload_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let positions = api::copy_host_vec_to_device(&point_data)
            .sync_on(&stream)?
            .reshape(&[count, 4])?;
        let point_upload_seconds = start.elapsed().as_secs_f64();
        // Exclude the fill kernel's first JIT compilation from steady output allocation.
        drop(api::zeros::<f64>(&[count, 4]).sync_on(&stream)?);
        for variant in 0..4u8 {
            if selected.is_some_and(|v| v != variant) {
                continue;
            }
            let start = Instant::now();
            let mut real = api::zeros::<f64>(&[count, 4]).sync_on(&stream)?;
            let mut imag = api::zeros::<f64>(&[count, 4]).sync_on(&stream)?;
            let output_allocation_seconds = start.elapsed().as_secs_f64();
            let (cold_kernel_seconds, warm_kernel_seconds) = {
                let mut launch = || -> Result<(), Error> {
                    match variant {
                        0 => {
                            kernels::baseline(
                                (&mut real).partition([128, 4]),
                                (&mut imag).partition([128, 4]),
                                &positions,
                                &vectors,
                                &pr,
                                &pi,
                            )
                            .sync_on(&stream)?;
                        }
                        1 => {
                            kernels::component_scalar(
                                (&mut real).partition([128, 4]),
                                (&mut imag).partition([128, 4]),
                                &positions,
                                &vectors,
                                &pr,
                                &pi,
                            )
                            .sync_on(&stream)?;
                        }
                        2 => {
                            kernels::mode_tiled(
                                (&mut real).partition([32, 4]),
                                (&mut imag).partition([32, 4]),
                                &positions,
                                &vectors,
                                &pr,
                                &pi,
                            )
                            .sync_on(&stream)?;
                        }
                        _ => {
                            kernels::component_fma(
                                (&mut real).partition([128, 4]),
                                (&mut imag).partition([128, 4]),
                                &positions,
                                &vectors,
                                &pr,
                                &pi,
                                real_k_specialized,
                            )
                            .sync_on(&stream)?;
                        }
                    }
                    Ok(())
                };
                let start = Instant::now();
                launch()?;
                let cold = start.elapsed().as_secs_f64();
                let mut times = [0.0; 7];
                for time in &mut times {
                    let start = Instant::now();
                    launch()?;
                    *time = start.elapsed().as_secs_f64();
                }
                (cold, median(times))
            };
            let start = Instant::now();
            let real = Arc::new(real).to_host_vec().sync_on(&stream)?;
            let imag = Arc::new(imag).to_host_vec().sync_on(&stream)?;
            let download_seconds = start.elapsed().as_secs_f64();
            let actual = real
                .chunks_exact(4)
                .zip(imag.chunks_exact(4))
                .flat_map(|(r, i)| r.iter().zip(i).take(3).map(|(&r, &i)| Complex::new(r, i)));
            let mut max_error = 0.0f64;
            for (a, e) in actual.zip(expected.as_slice()) {
                let error = (a - e).norm();
                if !error.is_finite() || error > 2e-12 * (1.0 + e.norm()) {
                    return Err(format!("variant {variant} fails CPU oracle: {a} vs {e}").into());
                }
                max_error = max_error.max(error);
            }
            println!(
                "{{\"variant\":{variant},\"modes\":{modes},\"points\":{count},\"lossless\":{lossless},\"real_k_specialized\":{real_k_specialized},\"context_seconds\":{context_seconds},\"mode_upload_seconds\":{mode_upload_seconds},\"point_upload_seconds\":{point_upload_seconds},\"cpu_original_seconds\":{cpu_original_seconds},\"cpu_preweighted_seconds\":{cpu_preweighted_seconds},\"output_allocation_seconds\":{output_allocation_seconds},\"cold_kernel_seconds\":{cold_kernel_seconds},\"warm_kernel_seconds\":{warm_kernel_seconds},\"download_seconds\":{download_seconds},\"max_abs_error\":{max_error}}}"
            );
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(all(feature = "cuda-tile", target_os = "linux"))]
    {
        probe::run()
    }
    #[cfg(not(all(feature = "cuda-tile", target_os = "linux")))]
    {
        Err("probe_plane requires Linux and feature cuda-tile".into())
    }
}

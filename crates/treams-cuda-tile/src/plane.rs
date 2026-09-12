// The DSL macro rewrites tensor types and discards module-level lint attributes.
#![allow(clippy::wildcard_imports)]

use cutile::cuda_core::Stream;
use cutile::error::TensorError;
use cutile::prelude::*;
use treams_core::{Complex, plane};

#[cutile::module]
mod kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn weighted_plane<const B: i32>(
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
}

/// A weighted plane-wave expansion retained on one CUDA device.
///
/// Evaluates `sum_m coefficient[m] polarization[m] exp(i k[m] · r)` in f64.
/// Polarization normalization is shared with `treams_core::plane::polarization`.
/// Only requested points are evaluated; no point-by-mode operator is allocated.
/// The first evaluation JIT-compiles the Rust kernel and can be much slower.
pub struct PlaneWaves {
    stream: Arc<Stream>,
    vectors: Tensor<f64>,
    pol_real: Tensor<f64>,
    pol_imag: Tensor<f64>,
    modes: usize,
}

impl std::fmt::Debug for PlaneWaves {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlaneWaves")
            .field("modes", &self.modes)
            .finish_non_exhaustive()
    }
}

fn invalid(message: &str) -> Error {
    Error::Tensor(TensorError(message.into()))
}

impl PlaneWaves {
    /// Normalize and upload a finite, nonempty plane-wave expansion.
    ///
    /// Complex wavevectors support propagating, evanescent, and lossy waves.
    /// Polarization labels and helicity/parity conventions are those of the CPU core.
    pub fn new(
        device: usize,
        vectors: &[[Complex; 3]],
        polarizations: &[u8],
        coefficients: &[Complex],
        helicity: bool,
    ) -> Result<Self, Error> {
        let modes = vectors.len();
        if modes == 0 || modes != polarizations.len() || modes != coefficients.len() {
            return Err(invalid(
                "require nonempty, matching modes, labels, and coefficients",
            ));
        }
        if coefficients
            .iter()
            .any(|c| !c.re.is_finite() || !c.im.is_finite())
        {
            return Err(invalid("plane coefficients must be finite"));
        }
        modes
            .checked_mul(8)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| invalid("too many plane modes"))?;
        let electric = vectors
            .iter()
            .zip(polarizations)
            .zip(coefficients)
            .map(|((&k, &p), &c)| plane::polarization(k, p, helicity).map(|e| e.map(|v| v * c)))
            .collect::<treams_core::Result<Vec<_>>>()
            .map_err(|e| invalid(&e.to_string()))?;
        if electric
            .iter()
            .flatten()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(invalid("nonfinite weighted polarization"));
        }
        let wave_data: Vec<_> = vectors
            .iter()
            .flat_map(|k| {
                [
                    k[0].re, k[0].im, k[1].re, k[1].im, k[2].re, k[2].im, 0.0, 0.0,
                ]
            })
            .collect();
        let real_data: Vec<_> = electric
            .iter()
            .flat_map(|p| [p[0].re, p[1].re, p[2].re, 0.0])
            .collect();
        let imag_data: Vec<_> = electric
            .iter()
            .flat_map(|p| [p[0].im, p[1].im, p[2].im, 0.0])
            .collect();
        let device = Device::new(device)?;
        let stream = device.new_stream()?;
        let vectors = api::copy_host_vec_to_device(&Arc::new(wave_data))
            .sync_on(&stream)?
            .reshape(&[modes, 8])?;
        let pol_real = api::copy_host_vec_to_device(&Arc::new(real_data))
            .sync_on(&stream)?
            .reshape(&[modes, 4])?;
        let pol_imag = api::copy_host_vec_to_device(&Arc::new(imag_data))
            .sync_on(&stream)?
            .reshape(&[modes, 4])?;
        Ok(Self {
            stream,
            vectors,
            pol_real,
            pol_imag,
            modes,
        })
    }

    /// Evaluate Cartesian electric fields at finite real coordinates.
    ///
    /// Transfers only points and the resulting three complex components. Host
    /// and device storage are O(points + modes), including power-of-two padding.
    pub fn evaluate(&self, points: &[[f64; 3]]) -> Result<Vec<[Complex; 3]>, Error> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        if points.iter().flatten().any(|x| !x.is_finite()) {
            return Err(invalid("plane-field coordinates must be finite"));
        }
        // cuTile transfer tensors use a signed 32-bit flat element count.
        points
            .len()
            .checked_mul(4)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| invalid("too many field points for cuTile"))?;
        let data: Vec<_> = points
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 0.0])
            .collect();
        let positions = api::copy_host_vec_to_device(&Arc::new(data))
            .sync_on(&self.stream)?
            .reshape(&[points.len(), 4])?;
        let mut real = api::zeros::<f64>(&[points.len(), 4]).sync_on(&self.stream)?;
        let mut imag = api::zeros::<f64>(&[points.len(), 4]).sync_on(&self.stream)?;
        kernel::weighted_plane(
            (&mut real).partition([128, 4]),
            (&mut imag).partition([128, 4]),
            &positions,
            &self.vectors,
            &self.pol_real,
            &self.pol_imag,
        )
        .sync_on(&self.stream)?;
        let real = Arc::new(real).to_host_vec().sync_on(&self.stream)?;
        let imag = Arc::new(imag).to_host_vec().sync_on(&self.stream)?;
        let result: Vec<_> = real
            .chunks_exact(4)
            .zip(imag.chunks_exact(4))
            .map(|(re, im)| {
                let mut field = [Complex::default(); 3];
                for (value, (&r, &i)) in field.iter_mut().zip(re.iter().zip(im)) {
                    *value = Complex::new(r, i);
                }
                field
            })
            .collect();
        if result
            .iter()
            .flatten()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(invalid("nonfinite plane-wave field"));
        }
        Ok(result)
    }
}

//! T-matrices of multilayer chiral cylinders from their Mie coefficients.
//!
//! Upstream: `treams.TMatrixC.cylinder`.
#![allow(clippy::indexing_slicing)] // Block, solve and axial-wavenumber indices built here.

use crate::{
    Complex, Error, MAX_DEGREE, Result,
    coeffs::{LayerGradient, Material, MieCylResidual, mie_cyl},
    numerics::{finite, label_bits},
};

/// What [`cylinder`] saves for its pullback: one boundary solve per `(kz, m)` pair,
/// shared with the mirror pair `(-kz, -m)` when both are requested.
#[derive(Debug)]
pub struct CylinderResidual {
    blocks: Vec<Block>,
    solves: Vec<MieCylResidual>,
    kz_count: usize,
    boundaries: usize,
}

/// One `(kz, m)` block: its axial wavenumber index and the boundary solve it
/// shares with its mirror image `(-kz, -m)` under a half turn about a transverse axis,
/// which leaves the cylinder and hence the coefficients unchanged.
#[derive(Clone, Copy, Debug)]
struct Block {
    kz_index: usize,
    solve: usize,
    mirrored: bool,
}

/// Gradients of the inputs of [`cylinder`], in the order of its arguments.
#[derive(Debug)]
pub struct CylinderGradient {
    /// Gradient of each axial wavenumber in `kzs`.
    pub kzs: Vec<f64>,
    /// Gradient of the vacuum wavenumber.
    pub k0: f64,
    /// Gradients of the radii and the materials.
    pub layers: LayerGradient,
}

/// The T-matrix of concentric multilayer chiral cylinders, ordered by `kzs`, then
/// ascending order `m` from `-mmax` to `mmax`, then positive and negative helicity.
///
/// The boundary solves of [`mie_cyl`] run in parallel. A block `(kz, m)` whose mirror
/// `(-kz, -m)` is requested as well reuses the mirror's solve when `kz < 0`, or
/// `kz = 0` and `m < 0`.
///
/// Upstream: `treams.TMatrixC.cylinder`. Differences: the shared mirror solves, which
/// agree with separate solves to rounding.
pub fn cylinder(
    kzs: &[f64],
    mmax: u32,
    k0: f64,
    radii: &[f64],
    materials: &[Material],
) -> Result<(nalgebra::DMatrix<Complex>, CylinderResidual)> {
    use rayon::prelude::*;
    if kzs.is_empty() || kzs.iter().any(|k| !k.is_finite()) || mmax > MAX_DEGREE.unsigned_abs() {
        return Err(Error::InvalidInput(
            // 128 is MAX_DEGREE.
            "require finite nonempty kzs and mmax <= 128".into(),
        ));
    }
    let mut seen = std::collections::HashSet::new();
    if kzs.iter().any(|&kz| !seen.insert(label_bits(kz))) {
        return Err(Error::InvalidInput(
            "axial wavenumbers must be unique".into(),
        ));
    }
    let bound = i32::try_from(mmax).map_err(|_| Error::InvalidInput("invalid mmax".into()))?;
    let mirrored =
        |kz: f64, m: i32| (kz < 0.0 || (kz == 0.0 && m < 0)) && seen.contains(&label_bits(-kz));
    let labels: Vec<_> = kzs
        .iter()
        .enumerate()
        .flat_map(|(index, &kz)| (-bound..=bound).map(move |m| (index, kz, m)))
        .collect();
    let mut solved = Vec::new();
    let mut solve = std::collections::HashMap::new();
    for &(_, kz, m) in labels.iter().filter(|&&(_, kz, m)| !mirrored(kz, m)) {
        solve.insert((label_bits(kz), m), solved.len());
        solved.push((kz, m));
    }
    let solves = solved
        .par_iter()
        .map(|&(kz, m)| mie_cyl(kz, m, k0, radii, materials))
        .collect::<Result<Vec<_>>>()?;
    let blocks = labels
        .iter()
        .map(|&(index, kz, m)| {
            let mirrored = mirrored(kz, m);
            let (kz, m) = if mirrored { (-kz, -m) } else { (kz, m) };
            Ok(Block {
                kz_index: index,
                solve: *solve
                    .get(&(label_bits(kz), m))
                    .ok_or_else(|| Error::InvalidInput("missing cylinder block".into()))?,
                mirrored,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut value = nalgebra::DMatrix::zeros(2 * blocks.len(), 2 * blocks.len());
    for (index, block) in blocks.iter().enumerate() {
        value
            .fixed_view_mut::<2, 2>(2 * index, 2 * index)
            .copy_from(&crate::coeffs::to_mode_order(solves[block.solve].value()));
    }
    Ok((
        value,
        CylinderResidual {
            blocks,
            solves,
            kz_count: kzs.len(),
            boundaries: radii.len(),
        },
    ))
}

impl CylinderResidual {
    /// The shape of the T-matrix: two helicities for each `(kz, m)` block.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (2 * self.blocks.len(), 2 * self.blocks.len())
    }

    /// Gradients of `kzs`, `k0`, the radii and the materials from `cotangent`, the
    /// gradient of a real loss with respect to the T-matrix.
    ///
    /// The blocks run in parallel and their gradients add in block order, so the Rayon
    /// pool does not change them.
    pub fn pullback(self, cotangent: &nalgebra::DMatrix<Complex>) -> Result<CylinderGradient> {
        use rayon::prelude::*;
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "invalid cylinder T-matrix cotangent".into(),
            ));
        }
        let solves = &self.solves;
        let gradients = self
            .blocks
            .par_iter()
            .enumerate()
            .map(|(index, block)| {
                let g = cotangent
                    .fixed_view::<2, 2>(2 * index, 2 * index)
                    .into_owned();
                let gradient = solves[block.solve].gradient(&crate::coeffs::to_mode_order(&g))?;
                // The mirror block depends on its own kz with the opposite sign.
                let kz = if block.mirrored {
                    -gradient.kz
                } else {
                    gradient.kz
                };
                Ok((kz, gradient))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut result = CylinderGradient {
            kzs: vec![0.0; self.kz_count],
            k0: 0.0,
            layers: LayerGradient::zeros(self.boundaries),
        };
        for (block, (kz, gradient)) in self.blocks.iter().zip(gradients) {
            result.kzs[block.kz_index] += kz;
            result.k0 += gradient.k0;
            result.layers.accumulate(&gradient.layers);
        }
        Ok(result)
    }
}

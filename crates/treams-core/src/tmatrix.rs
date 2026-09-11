//! Spherical T-matrices and complete finite-cluster forward/pullback execution.
// Modes, Cartesian axes, and block offsets are constructed and validated here.
#![allow(clippy::indexing_slicing)]

use nalgebra::DMatrix;

use rayon::prelude::*;

use crate::coeffs::{Material, Matrix2, MieGradient, MieResidual, mie_forward};
use crate::interaction::{self, InteractionResidual};
use crate::translation_plan::TranslationPlan;
use crate::waves::{self, Mode};
use crate::{Complex, Error, Result, finite};

/// Normalized global helicity T-matrix observables.
#[derive(Clone, Copy, Debug)]
pub enum Metric {
    /// Absorption circular dichroism in a real embedding medium.
    CircularDichroism,
    /// Fraction of scattering that changes helicity.
    DualityBreaking,
    /// Electromagnetic chirality from the four helicity-block spectra.
    Chirality,
}

/// Scalar metric and its first-order gradient; no singular vectors remain live.
#[derive(Debug)]
pub struct MetricResidual {
    /// Metric value; chirality may be zero even when its derivative is undefined.
    pub value: f64,
    gradient: Result<DMatrix<Complex>>,
    ks: [f64; 2],
    dimension: usize,
}

impl MetricResidual {
    /// Matrix and real embedding-wavenumber cotangents.
    pub fn pullback(self, weight: f64) -> Result<(DMatrix<Complex>, [f64; 2])> {
        if !weight.is_finite() {
            return Err(Error::InvalidInput(
                "metric cotangent must be finite".into(),
            ));
        }
        if weight == 0.0 {
            return Ok((DMatrix::zeros(self.dimension, self.dimension), [0.0; 2]));
        }
        Ok((
            self.gradient?.map(|z| z * weight),
            self.ks.map(|k| k * weight),
        ))
    }
}

/// Evaluate one metric of a global helicity T matrix. Wavenumbers affect CD only.
pub fn metric(
    matrix: &DMatrix<Complex>,
    pol: &[u8],
    ks: [f64; 2],
    kind: Metric,
) -> Result<MetricResidual> {
    let dimension = matrix.nrows();
    if dimension == 0
        || !matrix.is_square()
        || pol.len() != dimension
        || pol.iter().any(|&p| p > 1)
        || matrix.iter().any(|&z| !finite(z))
        || ks.iter().any(|k| !k.is_finite() || *k <= 0.0)
    {
        return Err(Error::InvalidInput("require a finite square helicity matrix, matching polarizations and positive real wavenumbers".into()));
    }
    if matches!(kind, Metric::CircularDichroism) {
        let weights: Vec<_> = pol.iter().map(|&p| ks[usize::from(p)].powi(-2)).collect();
        let mut absorption = [0.0; 2];
        for j in 0..dimension {
            absorption[usize::from(pol[j])] -= matrix[(j, j)].re * weights[j]
                + (0..dimension)
                    .map(|i| matrix[(i, j)].norm_sqr() * weights[i])
                    .sum::<f64>();
        }
        let total = absorption.iter().sum::<f64>();
        if total == 0.0 || !total.is_finite() {
            return Err(Error::InvalidInput(
                "circular dichroism requires nonzero total absorption".into(),
            ));
        }
        let value = (absorption[1] - absorption[0]) / total;
        let g = [
            -2.0 * absorption[1] / total / total,
            2.0 * absorption[0] / total / total,
        ];
        let mut gradient = DMatrix::from_fn(dimension, dimension, |i, j| {
            -2.0 * g[usize::from(pol[j])] * weights[i] * matrix[(i, j)]
        });
        let mut gks = [0.0; 2];
        for i in 0..dimension {
            let p = usize::from(pol[i]);
            gradient[(i, i)] -= g[p] * weights[i];
            gks[p] += 2.0 * weights[i] / ks[p]
                * (g[p] * matrix[(i, i)].re
                    + (0..dimension)
                        .map(|j| g[usize::from(pol[j])] * matrix[(i, j)].norm_sqr())
                        .sum::<f64>());
        }
        return Ok(MetricResidual {
            value,
            gradient: Ok(gradient),
            ks: gks,
            dimension,
        });
    }
    // Normalized metrics are invariant to a common matrix scale. Work at unit
    // scale so norm squares do not overflow or underflow for weak scattering.
    let scale = matrix.iter().map(|z| z.norm()).fold(0.0, f64::max);
    if scale == 0.0 {
        return Err(Error::InvalidInput(
            "normalized metric requires nonzero scattering".into(),
        ));
    }
    let matrix = matrix.map(|z| z / scale);
    let norm = matrix.norm_squared();
    let (value, gradient) = if matches!(kind, Metric::DualityBreaking) {
        let flipped = (0..dimension)
            .flat_map(|j| (0..dimension).map(move |i| (i, j)))
            .filter(|&(i, j)| pol[i] != pol[j])
            .map(|(i, j)| matrix[(i, j)].norm_sqr())
            .sum::<f64>();
        let value = flipped / norm;
        (
            value,
            Ok(DMatrix::from_fn(dimension, dimension, |i, j| {
                matrix[(i, j)] * (2.0 * (f64::from(pol[i] != pol[j]) - value) / norm)
            })),
        )
    } else {
        let indices: [Vec<_>; 2] = std::array::from_fn(|p| {
            pol.iter()
                .enumerate()
                .filter_map(|(i, &q)| (usize::from(q) == p).then_some(i))
                .collect()
        });
        let count = indices[0].len();
        if count == 0 || count != indices[1].len() {
            return Err(Error::InvalidInput(
                "chirality requires equally sized nonempty helicity groups".into(),
            ));
        }
        let spectra = (0..4)
            .map(|b| {
                crate::linalg::svdvals(&DMatrix::from_fn(count, count, |i, j| {
                    matrix[(indices[b / 2][i], indices[b % 2][j])]
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        let differences: [Vec<_>; 2] = std::array::from_fn(|p| {
            spectra[3 - p]
                .values
                .iter()
                .zip(&spectra[p].values)
                .map(|(a, b)| a - b)
                .collect()
        });
        let contrast = differences
            .iter()
            .flatten()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let value = contrast / norm.sqrt();
        let gradient = (|| {
            if contrast == 0.0 {
                return Err(Error::InvalidInput(
                    "chirality is not differentiable at zero contrast".into(),
                ));
            }
            let mut result = matrix.map(|z| -value * z / norm);
            for (b, residual) in spectra.into_iter().enumerate() {
                let p = usize::from(b == 1 || b == 2);
                let sign = if b < 2 { -1.0 } else { 1.0 };
                let weights: Vec<_> = differences[p]
                    .iter()
                    .map(|x| sign * x / contrast / norm.sqrt())
                    .collect();
                let block = residual.pullback(&weights)?;
                for j in 0..count {
                    for i in 0..count {
                        result[(indices[b / 2][i], indices[b % 2][j])] += block[(i, j)];
                    }
                }
            }
            Ok(result)
        })();
        (value, gradient)
    };
    Ok(MetricResidual {
        value,
        gradient: gradient.map(|g| g.map(|z| z / scale)),
        ks: [0.0; 2],
        dimension,
    })
}
/// Native retained spherical T-matrix computation in helicity basis.
#[derive(Clone, Debug)]
pub struct SphereResidual {
    k0: f64,
    radii: Vec<f64>,
    orders: Vec<MieResidual>,
    modes: Vec<Mode>,
    /// Spherical T-matrix in treams basis order.
    pub value: DMatrix<Complex>,
}

/// Sphere input cotangents.
#[derive(Clone, Debug)]
pub struct SphereGradient {
    /// Vacuum wavenumber derivative.
    pub k0: f64,
    /// Boundary radius derivatives.
    pub radii: Vec<f64>,
    /// Layer material derivatives; size derivatives are omitted from this payload.
    pub materials: MieGradient,
}

/// Construct a multilayer/chiral spherical T-matrix.
pub fn sphere(lmax: u32, k0: f64, radii: &[f64], materials: &[Material]) -> Result<SphereResidual> {
    if !k0.is_finite() || k0 <= 0.0 {
        return Err(Error::InvalidInput("k0 must be finite and positive".into()));
    }
    let modes = waves::modes(lmax)?;
    let sizes: Vec<_> = radii.iter().map(|r| k0 * r).collect();
    let orders = (1..=lmax)
        .map(|l| mie_forward(l, &sizes, materials))
        .collect::<Result<Vec<_>>>()?;
    let mut value = DMatrix::zeros(modes.len(), modes.len());
    for (i, mode) in modes.iter().enumerate() {
        let order =
            usize::try_from(mode.l - 1).map_err(|_| Error::InvalidInput("invalid mode".into()))?;
        let block = i / 2 * 2;
        for pol in 0..2 {
            value[(i, block + 1 - pol)] = orders[order].value[(usize::from(mode.pol), pol)];
        }
    }
    Ok(SphereResidual {
        k0,
        radii: radii.to_vec(),
        orders,
        modes,
        value,
    })
}

impl SphereResidual {
    /// Contract a T-matrix cotangent into radii, wavenumber, and material cotangents.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<SphereGradient> {
        if cotangent.shape() != self.value.shape() || cotangent.iter().any(|z| !finite(*z)) {
            return Err(Error::InvalidInput(
                "invalid spherical T-matrix cotangent".into(),
            ));
        }
        let mut blocks = vec![Matrix2::zeros(); self.orders.len()];
        for (i, mode) in self.modes.iter().enumerate() {
            let order = usize::try_from(mode.l - 1)
                .map_err(|_| Error::InvalidInput("invalid mode".into()))?;
            let block = i / 2 * 2;
            for pol in 0..2 {
                blocks[order][(usize::from(mode.pol), pol)] += cotangent[(i, block + 1 - pol)];
            }
        }
        let mut sum = MieGradient {
            sizes: vec![0.0; self.radii.len()],
            epsilon: vec![Complex::default(); self.radii.len() + 1],
            mu: vec![Complex::default(); self.radii.len() + 1],
            kappa: vec![Complex::default(); self.radii.len() + 1],
        };
        for (order, g) in self.orders.into_iter().zip(blocks) {
            let gradient = order.pullback(&g)?;
            for (dst, src) in sum.sizes.iter_mut().zip(gradient.sizes) {
                *dst += src;
            }
            for (dst, src) in sum.epsilon.iter_mut().zip(gradient.epsilon) {
                *dst += src;
            }
            for (dst, src) in sum.mu.iter_mut().zip(gradient.mu) {
                *dst += src;
            }
            for (dst, src) in sum.kappa.iter_mut().zip(gradient.kappa) {
                *dst += src;
            }
        }
        let k0 = sum.sizes.iter().zip(&self.radii).map(|(g, r)| g * r).sum();
        let radii = sum.sizes.iter().map(|g| self.k0 * g).collect();
        sum.sizes.clear();
        Ok(SphereGradient {
            k0,
            radii,
            materials: sum,
        })
    }
}

/// Retained solve for non-overlapping homogeneous spheres in vacuum.
#[derive(Clone, Debug)]
pub struct ClusterResidual {
    k0: f64,
    positions: Vec<[f64; 3]>,
    modes: Vec<Mode>,
    plan: TranslationPlan,
    spheres: Vec<SphereResidual>,
    interaction: InteractionResidual,
}

/// Input cotangents for a finite cluster.
#[derive(Clone, Debug)]
pub struct ClusterGradient {
    /// Sphere radius derivatives.
    pub radii: Vec<f64>,
    /// Position derivatives, including the dependence of all pair couplings.
    pub positions: Vec<[f64; 3]>,
    /// Sphere permittivity cotangents.
    pub epsilon: Vec<Complex>,
    /// Vacuum wavenumber derivative.
    pub k0: f64,
}

/// Solve a finite cluster of homogeneous, nonmagnetic spheres in vacuum.
pub fn cluster(
    lmax: u32,
    k0: f64,
    radii: &[f64],
    epsilon: &[Complex],
    positions: &[[f64; 3]],
) -> Result<ClusterResidual> {
    if radii.is_empty()
        || radii.len() != epsilon.len()
        || radii.len() != positions.len()
        || positions.iter().flatten().any(|x| !x.is_finite())
    {
        return Err(Error::InvalidInput(
            "radii, epsilon and finite positions must describe the same nonempty cluster".into(),
        ));
    }
    for i in 0..radii.len() {
        for j in 0..i {
            let d2 = (0..3)
                .map(|a| (positions[i][a] - positions[j][a]).powi(2))
                .sum::<f64>();
            if d2 <= (radii[i] + radii[j]).powi(2) {
                return Err(Error::InvalidInput(
                    "spherical particles must not touch or overlap".into(),
                ));
            }
        }
    }
    let modes = waves::modes(lmax)?;
    let dimension = modes.len();
    let spheres = radii
        .iter()
        .zip(epsilon)
        .map(|(&r, &eps)| {
            sphere(
                lmax,
                k0,
                &[r],
                &[
                    Material {
                        epsilon: eps,
                        ..Material::default()
                    },
                    Material::default(),
                ],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let size = dimension
        .checked_mul(spheres.len())
        .ok_or_else(|| Error::InvalidInput("cluster is too large".into()))?;
    let mut coupling = DMatrix::zeros(size, size);
    let plan = TranslationPlan::new(&modes)?;
    // Each worker owns complete source-particle columns; no locks or dense pair copies.
    coupling
        .as_mut_slice()
        .par_chunks_mut(size * dimension)
        .enumerate()
        .try_for_each(|(j, columns)| -> Result<()> {
            for i in 0..spheres.len() {
                if i == j {
                    continue;
                }
                let displacement =
                    std::array::from_fn(|axis| positions[i][axis] - positions[j][axis]);
                let values = plan.evaluate(Complex::new(k0, 0.0), displacement)?;
                for col in 0..dimension {
                    let start = col * size + i * dimension;
                    columns[start..start + dimension]
                        .copy_from_slice(&values[col * dimension..(col + 1) * dimension]);
                }
            }
            Ok(())
        })?;
    let blocks = spheres.iter().map(|sphere| sphere.value.clone()).collect();
    Ok(ClusterResidual {
        k0,
        positions: positions.to_vec(),
        modes,
        spheres,
        plan,
        interaction: interaction::forward_blocks(blocks, coupling)?,
    })
}

impl ClusterResidual {
    /// Interacting T-matrix in the local multipole bases.
    #[must_use]
    pub fn value(&self) -> &DMatrix<Complex> {
        &self.interaction.value
    }

    /// Complete native cluster pullback, including analytic position derivatives.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<ClusterGradient> {
        let (local, coupling) = self.interaction.pullback(cotangent)?;
        let dimension = self.modes.len();
        let n = self.spheres.len();
        let mut result = ClusterGradient {
            radii: vec![0.0; n],
            positions: vec![[0.0; 3]; n],
            epsilon: vec![Complex::default(); n],
            k0: 0.0,
        };
        for (i, sphere) in self.spheres.into_iter().enumerate() {
            let gradient = sphere.pullback(
                &local
                    .view((i * dimension, i * dimension), (dimension, dimension))
                    .into_owned(),
            )?;
            result.radii[i] = gradient.radii[0];
            result.epsilon[i] = gradient.materials.epsilon[0];
            result.k0 += gradient.k0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let displacement =
                    std::array::from_fn(|axis| self.positions[i][axis] - self.positions[j][axis]);
                let block = coupling
                    .view((i * dimension, j * dimension), (dimension, dimension))
                    .into_owned();
                let (gradient, gk) = self.plan.pullback(
                    Complex::new(self.k0, 0.0),
                    displacement,
                    block.as_slice(),
                )?;
                result.k0 += gk;
                for (axis, g) in gradient.into_iter().enumerate() {
                    result.positions[i][axis] += g;
                    result.positions[j][axis] -= g;
                }
            }
        }
        Ok(result)
    }
}

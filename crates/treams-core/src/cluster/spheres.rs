//! Dense clusters of homogeneous nonmagnetic spheres in vacuum, with gradients of the
//! radii, permittivities and positions.
//!
//! Upstream: `treams.TMatrix.cluster` of `TMatrix.sphere` T-matrices, then
//! `interaction.solve()`. The gradients and the reusable factor of
//! [`sphere_cluster_factor`] are treams-rs extensions.
#![allow(clippy::indexing_slicing)] // Validated spheres, mode blocks and Cartesian axes.

use nalgebra::DMatrix;

use rayon::prelude::*;

use super::{InteractionGradient, InteractionResidual, interaction_blocks};
use crate::coeffs::Material;
use crate::special::Radial;
use crate::sw::{self, TranslationPlan};
use crate::tmatrix::{SphereResidual, sphere};
use crate::{
    Complex, Error, Result,
    numerics::{self, finite},
};

/// What [`sphere_cluster`] saves for its pullback: the Mie residual of each sphere,
/// the translation plan between two spheres, the positions and the interaction solve.
#[derive(Clone, Debug)]
pub struct SphereClusterResidual {
    k0: f64,
    positions: Vec<[f64; 3]>,
    plan: TranslationPlan,
    spheres: Vec<SphereResidual>,
    interaction: InteractionResidual,
}

/// Gradients of the inputs of [`sphere_cluster`], in the order of its arguments.
#[derive(Clone, Debug)]
pub struct SphereClusterGradient {
    /// Gradient of the vacuum wavenumber.
    pub k0: f64,
    /// Gradients of the sphere radii.
    pub radii: Vec<f64>,
    /// Gradients of the sphere permittivities.
    pub epsilon: Vec<Complex>,
    /// Gradients of the sphere positions, through every pair coupling.
    pub positions: Vec<[f64; 3]>,
}

/// Solve a finite cluster of homogeneous, nonmagnetic spheres in vacuum.
///
/// Returns the interacting T-matrix in the local bases of the spheres, each in
/// [`sw::modes`] order up to `lmax`. [`particle_cluster`](super::particle_cluster)
/// couples any particle T-matrices, without the radius and permittivity gradients.
///
/// Upstream: `treams.TMatrix.cluster` of `TMatrix.sphere` T-matrices, followed by
/// `.interaction.solve()`.
pub fn sphere_cluster(
    lmax: u32,
    k0: f64,
    radii: &[f64],
    epsilon: &[Complex],
    positions: &[[f64; 3]],
) -> Result<SphereClusterResidual> {
    let ClusterParts {
        blocks,
        spheres,
        plan,
        coupling,
    } = cluster_parts(lmax, k0, radii, epsilon, positions)?;
    Ok(SphereClusterResidual {
        k0,
        positions: positions.to_vec(),
        spheres,
        plan,
        interaction: interaction_blocks(blocks, coupling)?,
    })
}

/// Assemble and factor a sphere cluster for repeated requested illuminations.
///
/// The returned factor differentiates the local T blocks, the coupling and the incident
/// fields. Use [`IterativeSphereCluster`](super::IterativeSphereCluster) for the
/// gradients of the radii, permittivities and positions.
///
/// treams-rs extension.
pub fn sphere_cluster_factor(
    lmax: u32,
    k0: f64,
    radii: &[f64],
    epsilon: &[Complex],
    positions: &[[f64; 3]],
) -> Result<super::InteractionFactor> {
    let parts = cluster_parts(lmax, k0, radii, epsilon, positions)?;
    super::InteractionFactor::from_blocks(parts.blocks, parts.coupling)
}

/// Check a cluster of homogeneous spheres in vacuum: a positive finite vacuum
/// wavenumber and radii, one finite permittivity and position per sphere, and no two
/// spheres touching or overlapping.
pub(crate) fn validate_spheres(
    k0: f64,
    radii: &[f64],
    epsilon: &[Complex],
    positions: &[[f64; 3]],
) -> Result<()> {
    if !k0.is_finite()
        || k0 <= 0.0
        || radii.is_empty()
        || radii.len() != epsilon.len()
        || radii.len() != positions.len()
        || radii.iter().any(|r| !r.is_finite() || *r <= 0.0)
        || epsilon.iter().any(|&z| !finite(z))
        || positions.iter().flatten().any(|x| !x.is_finite())
    {
        return Err(Error::InvalidInput(
            "require positive finite k0 and radii, and matching finite permittivities and positions".into(),
        ));
    }
    for i in 0..radii.len() {
        for j in 0..i {
            let distance = positions[i]
                .iter()
                .zip(positions[j])
                .fold(0.0_f64, |norm, (a, b)| norm.hypot(a - b));
            if distance <= radii[i] + radii[j] || !distance.is_finite() {
                return Err(Error::InvalidInput(
                    "spherical particles must have finite separation and must not touch or overlap"
                        .into(),
                ));
            }
        }
    }
    Ok(())
}

/// The media of a homogeneous nonmagnetic sphere in vacuum.
pub(crate) fn vacuum_sphere(epsilon: Complex) -> [Material; 2] {
    [
        Material {
            epsilon,
            ..Material::default()
        },
        Material::default(),
    ]
}

struct ClusterParts {
    /// The T-matrix of each sphere.
    blocks: Vec<DMatrix<Complex>>,
    spheres: Vec<SphereResidual>,
    plan: TranslationPlan,
    coupling: DMatrix<Complex>,
}

fn cluster_parts(
    lmax: u32,
    k0: f64,
    radii: &[f64],
    epsilon: &[Complex],
    positions: &[[f64; 3]],
) -> Result<ClusterParts> {
    validate_spheres(k0, radii, epsilon, positions)?;
    let modes = sw::modes(lmax)?;
    let modes_per_particle = modes.len();
    let (blocks, spheres): (Vec<_>, Vec<_>) = crate::threads::install(|| {
        radii
            .par_iter()
            .zip(epsilon)
            .map(|(&radius, &epsilon)| sphere(lmax, k0, &[radius], &vacuum_sphere(epsilon)))
            .collect::<Result<Vec<_>>>()
    })?
    .into_iter()
    .unzip();
    let dimension = modes_per_particle
        .checked_mul(spheres.len())
        .ok_or_else(|| Error::InvalidInput("cluster is too large".into()))?;
    let mut coupling = numerics::zeros(dimension, dimension)?;
    let plan = TranslationPlan::between(&modes, &modes, true)?;
    // Each worker owns complete source-particle columns; no locks or dense pair copies.
    crate::threads::install(|| {
        coupling
            .as_mut_slice()
            .par_chunks_mut(dimension * modes_per_particle)
            .enumerate()
            .try_for_each(|(j, columns)| -> Result<()> {
                for i in 0..spheres.len() {
                    if i == j {
                        continue;
                    }
                    let displacement =
                        std::array::from_fn(|axis| positions[i][axis] - positions[j][axis]);
                    let values =
                        plan.evaluate(Complex::new(k0, 0.0), displacement, Radial::Singular)?;
                    for col in 0..modes_per_particle {
                        let start = col * dimension + i * modes_per_particle;
                        columns[start..start + modes_per_particle].copy_from_slice(
                            &values[col * modes_per_particle..(col + 1) * modes_per_particle],
                        );
                    }
                }
                Ok(())
            })
    })?;
    Ok(ClusterParts {
        blocks,
        spheres,
        plan,
        coupling,
    })
}

impl SphereClusterResidual {
    /// Interacting T-matrix in the local multipole bases.
    #[must_use]
    pub const fn value(&self) -> &DMatrix<Complex> {
        self.interaction.value()
    }

    /// The shape of the interacting T-matrix.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.interaction.shape()
    }

    /// Gradients of `k0`, the radii, the permittivities and the positions from
    /// `cotangent`, the gradient of a real loss with respect to the interacting T-matrix.
    ///
    /// The spheres run in parallel; their gradients add in sphere order, so the Rayon
    /// pool does not change them. The LU solve and the matrix products of
    /// [`InteractionResidual::pullback_blocks`] run in faer with the worker count of
    /// [`linalg`](crate::linalg).
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<SphereClusterGradient> {
        let InteractionGradient { local, coupling } =
            self.interaction.pullback_blocks(cotangent)?;
        let n = self.spheres.len();
        let modes_per_particle = coupling.nrows() / n;
        let (k0, plan, positions) = (self.k0, &self.plan, &self.positions);
        // Each sphere's Mie pullback and the couplings into it run in parallel; the
        // gradients are summed in sphere order.
        let parts = crate::threads::install(|| {
            self.spheres
                .into_par_iter()
                .zip(local)
                .enumerate()
                .map(|(i, (sphere, local))| {
                    let gradient = sphere.pullback(&local)?;
                    let mut block = DMatrix::zeros(modes_per_particle, modes_per_particle);
                    let pairs = (0..n)
                        .filter(|&j| j != i)
                        .map(|j| {
                            block.copy_from(&coupling.view(
                                (i * modes_per_particle, j * modes_per_particle),
                                (modes_per_particle, modes_per_particle),
                            ));
                            let displacement =
                                std::array::from_fn(|axis| positions[i][axis] - positions[j][axis]);
                            plan.pullback(
                                Complex::new(k0, 0.0),
                                displacement,
                                Radial::Singular,
                                block.as_slice(),
                            )
                            .map(|(position, k)| (position, k.re))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok((gradient, pairs))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let mut result = SphereClusterGradient {
            k0: 0.0,
            radii: vec![0.0; n],
            epsilon: vec![Complex::default(); n],
            positions: vec![[0.0; 3]; n],
        };
        for (i, (sphere, pairs)) in parts.into_iter().enumerate() {
            result.radii[i] = sphere.layers.radii[0];
            result.epsilon[i] = sphere.layers.epsilon[0];
            result.k0 += sphere.k0;
            for (j, (gradient, gk)) in (0..n).filter(|&j| j != i).zip(pairs) {
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

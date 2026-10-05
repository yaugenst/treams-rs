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
    saved::{Reader, SavedState, Writer},
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
    let plan = TranslationPlan::between(&modes, &modes, true)?;
    let coupling = coupling_matrix(spheres.len(), modes_per_particle, |i, j| {
        let displacement = std::array::from_fn(|axis| positions[i][axis] - positions[j][axis]);
        plan.evaluate(Complex::new(k0, 0.0), displacement, Radial::Singular)
    })?;
    Ok(ClusterParts {
        blocks,
        spheres,
        plan,
        coupling,
    })
}

/// Assemble complete particle columns in parallel, for either coupling values or
/// their directional derivative. Self-couplings are exactly zero in both cases.
fn coupling_matrix(
    particles: usize,
    modes_per_particle: usize,
    pair: impl Fn(usize, usize) -> Result<Vec<Complex>> + Sync,
) -> Result<DMatrix<Complex>> {
    let dimension = modes_per_particle
        .checked_mul(particles)
        .ok_or_else(|| Error::InvalidInput("cluster is too large".into()))?;
    let mut coupling = numerics::zeros(dimension, dimension)?;
    // Each worker owns complete source-particle columns; no locks or dense pair copies.
    crate::threads::install(|| {
        coupling
            .as_mut_slice()
            .par_chunks_mut(dimension * modes_per_particle)
            .enumerate()
            .try_for_each(|(j, columns)| -> Result<()> {
                for i in 0..particles {
                    if i == j {
                        continue;
                    }
                    let values = pair(i, j)?;
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
    Ok(coupling)
}

impl SphereClusterResidual {
    /// Byte count for a cluster's saved numerical state, from its static layout.
    pub fn state_size(lmax: usize, particles: usize) -> Result<usize> {
        if particles == 0 {
            return Err(invalid_state());
        }
        let sphere_size = SphereResidual::state_size(lmax, 1)?;
        let modes = lmax
            .checked_add(2)
            .and_then(|value| value.checked_mul(lmax))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(invalid_state)?;
        let particle_size = sphere_size.checked_add(24).ok_or_else(invalid_state)?;
        let interaction_size = InteractionResidual::state_size_uniform(modes, particles)?;
        particles
            .checked_mul(particle_size)
            .and_then(|size| size.checked_add(24))
            .and_then(|size| size.checked_add(interaction_size))
            .ok_or_else(invalid_state)
    }

    /// Number of spheres in the recorded cluster.
    #[must_use]
    pub fn particles(&self) -> usize {
        self.spheres.len()
    }

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

    /// Directional derivative for vacuum wavenumber, radii, permittivities and
    /// positions. Particle and translation tangents feed one solve with the saved LU.
    pub fn pushforward(
        &self,
        k0: f64,
        radii: &[f64],
        epsilon: &[Complex],
        positions: &[[f64; 3]],
    ) -> Result<DMatrix<Complex>> {
        let n = self.spheres.len();
        if !k0.is_finite()
            || radii.len() != n
            || epsilon.len() != n
            || positions.len() != n
            || radii.iter().any(|r| !r.is_finite())
            || epsilon.iter().any(|&z| !finite(z))
            || positions.iter().flatten().any(|x| !x.is_finite())
        {
            return Err(Error::InvalidInput(
                "sphere tangents must be finite and match the recorded cluster".into(),
            ));
        }
        let blocks = crate::threads::install(|| {
            self.spheres
                .par_iter()
                .zip(radii)
                .zip(epsilon)
                .map(|((sphere, &radius), &epsilon)| {
                    let zero = Material {
                        epsilon: Complex::default(),
                        mu: Complex::default(),
                        kappa: Complex::default(),
                    };
                    sphere.pushforward(k0, &[radius], &[Material { epsilon, ..zero }, zero])
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let modes_per_particle = self.shape().0 / n;
        let coupling = coupling_matrix(n, modes_per_particle, |i, j| {
            let displacement =
                std::array::from_fn(|axis| self.positions[i][axis] - self.positions[j][axis]);
            let tangent = std::array::from_fn(|axis| positions[i][axis] - positions[j][axis]);
            self.plan.pushforward(
                Complex::new(self.k0, 0.0),
                displacement,
                Radial::Singular,
                tangent,
                Complex::new(k0, 0.0),
            )
        })?;
        self.interaction.pushforward_blocks(&blocks, &coupling)
    }

    /// Gradients of `k0`, the radii, the permittivities and the positions from
    /// `cotangent`, the gradient of a real loss with respect to the interacting T-matrix.
    ///
    /// The spheres run in parallel; their gradients add in sphere order, so the Rayon
    /// pool does not change them. The LU solve and the matrix products of
    /// [`InteractionResidual::pullback_blocks`] run in faer with the worker count of
    /// [`linalg`](crate::linalg).
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<SphereClusterGradient> {
        let InteractionGradient { local, coupling } =
            self.interaction.pullback_blocks(cotangent)?;
        let n = self.spheres.len();
        let modes_per_particle = coupling.nrows() / n;
        let (k0, plan, positions) = (self.k0, &self.plan, &self.positions);
        // Each sphere's Mie pullback and the couplings into it run in parallel; the
        // gradients are summed in sphere order.
        let parts = crate::threads::install(|| {
            self.spheres
                .par_iter()
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

fn invalid_state() -> Error {
    Error::InvalidInput("invalid native sphere-cluster state".into())
}

impl SavedState for SphereClusterResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let lmax = self.spheres.first().ok_or_else(invalid_state)?.lmax();
        let particles = self.particles();
        let mut writer = Writer::new(Self::state_size(lmax, particles)?);
        writer.usize(lmax);
        writer.usize(particles);
        writer.f64(self.k0);
        for position in &self.positions {
            for &coordinate in position {
                writer.f64(coordinate);
            }
        }
        for sphere in &self.spheres {
            sphere.write_state(&mut writer);
        }
        self.interaction.write_state(&mut writer);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let lmax = reader.usize()?;
        let particles = reader.usize()?;
        // Prove the complete layout fits before allocating particle or matrix data.
        if bytes.len() != Self::state_size(lmax, particles)? {
            return Err(invalid_state());
        }
        let k0 = reader.f64()?;
        if !k0.is_finite() || k0 <= 0.0 {
            return Err(invalid_state());
        }
        let positions = (0..particles)
            .map(|_| {
                let position = [reader.f64()?, reader.f64()?, reader.f64()?];
                if position.iter().any(|value| !value.is_finite()) {
                    return Err(invalid_state());
                }
                Ok(position)
            })
            .collect::<Result<Vec<_>>>()?;
        let spheres = (0..particles)
            .map(|_| {
                let sphere = SphereResidual::read_state(&mut reader)?;
                if sphere.lmax() != lmax || sphere.boundaries() != 1 {
                    return Err(invalid_state());
                }
                Ok(sphere)
            })
            .collect::<Result<Vec<_>>>()?;
        let interaction = InteractionResidual::read_state(&mut reader)?;
        let modes_per_particle = 2 * lmax * (lmax + 2);
        if interaction.local_shapes().len() != particles
            || interaction
                .local_shapes()
                .any(|shape| shape != (modes_per_particle, modes_per_particle))
        {
            return Err(invalid_state());
        }
        reader.finish()?;
        // The plan depends only on mode labels; all numerical forward state,
        // including the interaction LU, comes directly from the saved bytes.
        let modes = sw::modes(u32::try_from(lmax).map_err(|_| invalid_state())?)?;
        let plan = TranslationPlan::between(&modes, &modes, true)?;
        Ok(Self {
            k0,
            positions,
            plan,
            spheres,
            interaction,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // Saved state must preserve the exact numerical results.

    use super::*;

    #[test]
    fn saved_state_preserves_value_and_both_derivatives() -> Result<()> {
        let cluster = sphere_cluster(
            1,
            0.9,
            &[0.25, 0.3],
            &[Complex::new(2.1, 0.04), Complex::new(1.7, 0.02)],
            &[[0.0, 0.0, 0.0], [1.3, 0.2, -0.1]],
        )?;
        let bytes = cluster.save_state()?;
        assert_eq!(bytes.len(), SphereClusterResidual::state_size(1, 2)?);
        let restored = SphereClusterResidual::from_state(&bytes)?;
        assert_eq!(cluster.value(), restored.value());
        assert_eq!(restored.save_state()?, bytes);

        let tangent = |residual: &SphereClusterResidual| {
            residual.pushforward(
                0.12,
                &[0.04, -0.02],
                &[Complex::new(0.13, 0.01), Complex::new(-0.04, 0.02)],
                &[[0.01, -0.03, 0.02], [-0.02, 0.04, 0.01]],
            )
        };
        assert_eq!(tangent(&cluster)?, tangent(&restored)?);
        let cotangent = cluster
            .value()
            .map(|value| value + Complex::new(0.03, 0.07));
        let original = cluster.pullback(&cotangent)?;
        let roundtrip = restored.pullback(&cotangent)?;
        assert_eq!(original.k0, roundtrip.k0);
        assert_eq!(original.radii, roundtrip.radii);
        assert_eq!(original.epsilon, roundtrip.epsilon);
        assert_eq!(original.positions, roundtrip.positions);

        assert!(SphereClusterResidual::from_state(&bytes[..bytes.len() - 1]).is_err());
        let mut extra = bytes;
        extra.push(0);
        assert!(SphereClusterResidual::from_state(&extra).is_err());
        Ok(())
    }
}

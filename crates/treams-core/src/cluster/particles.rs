//! Clusters of arbitrary spherical or cylindrical particle T-matrices.
//!
//! Upstream: `treams.TMatrix.cluster(tmats, positions).interaction.solve()` and its
//! `TMatrixC` twin. The gradients are a treams-rs extension.
#![allow(clippy::indexing_slicing)] // Validated particle blocks, positions and Cartesian axes.

use nalgebra::DMatrix;

use super::{InteractionGradient, InteractionResidual, interaction_blocks};
use crate::sw;
use crate::{Complex, Error, Result, basis::validate_wavenumbers};

#[derive(Debug)]
enum ParticleExpansion {
    Spherical(sw::ExpansionResidual),
    Cylindrical(crate::cw::ExpansionResidual),
}

/// What [`particle_cluster`] and [`cylindrical_particle_cluster`] save for their
/// pullback: the interaction solve, with one local block per particle, and the
/// expansion residual of the coupling.
#[derive(Debug)]
pub struct ParticleClusterResidual {
    expansion: ParticleExpansion,
    interaction: InteractionResidual,
}

/// Gradients of the inputs of [`particle_cluster`] and [`cylindrical_particle_cluster`],
/// in the order of their arguments: the local matrices, the positions of the basis and
/// the wavenumbers.
#[derive(Debug)]
pub struct ParticleClusterGradient {
    /// One matrix gradient per particle, in input order.
    pub local: Vec<DMatrix<Complex>>,
    /// Gradients of the particle positions.
    pub positions: Vec<[f64; 3]>,
    /// Gradients of the embedding wavenumbers, negative helicity first.
    pub ks: [Complex; 2],
}

/// Couple the spherical T-matrices of different particles in a common embedding
/// medium and solve for the interacting T-matrix.
///
/// Basis modes must be grouped in particle order, matching the local blocks.
/// The caller ensures the particles' enclosing spheres do not overlap.
///
/// Upstream: `treams.TMatrix.cluster(tmats, positions).interaction.solve()`.
pub fn particle_cluster(
    local: Vec<DMatrix<Complex>>,
    basis: sw::Basis,
    ks: [Complex; 2],
    helicity: bool,
) -> Result<ParticleClusterResidual> {
    basis.validate()?;
    validate_particle_blocks(
        &local,
        &basis.positions,
        basis.modes.iter().map(|&(p, _)| p),
    )?;
    let (coupling, expansion) = sw::expansion(
        basis.clone(),
        basis,
        ks,
        helicity,
        crate::special::Radial::Singular,
    )?;
    couple(local, coupling, ParticleExpansion::Spherical(expansion))
}

/// Solve the interaction of the local matrices through the singular expansion
/// `coupling`. The solve owns the only coupling matrix; the expansion pullback needs
/// just the geometry.
fn couple(
    local: Vec<DMatrix<Complex>>,
    coupling: DMatrix<Complex>,
    expansion: ParticleExpansion,
) -> Result<ParticleClusterResidual> {
    Ok(ParticleClusterResidual {
        expansion,
        interaction: interaction_blocks(local, coupling)?,
    })
}

fn validate_particle_blocks(
    local: &[DMatrix<Complex>],
    positions: &[[f64; 3]],
    mut pidxs: impl ExactSizeIterator<Item = usize>,
) -> Result<()> {
    if local.len() != positions.len()
        || local.iter().map(DMatrix::nrows).sum::<usize>() != pidxs.len()
    {
        return Err(Error::InvalidInput(
            "one local matrix and mode block required per position".into(),
        ));
    }
    for (i, block) in local.iter().enumerate() {
        if pidxs.by_ref().take(block.nrows()).any(|p| p != i)
            || positions[..i].contains(&positions[i])
        {
            return Err(Error::InvalidInput(
                "particle modes must be grouped at distinct positions".into(),
            ));
        }
    }
    Ok(())
}

/// Couple the cylindrical T-matrices of different particles with fixed axial mode
/// labels and solve for the interacting T-matrix.
///
/// Local modes are grouped in particle order. Enclosing cylinders must not overlap.
///
/// Upstream: `treams.TMatrixC.cluster(tmats, positions).interaction.solve()`.
#[allow(clippy::float_cmp)] // Exact common transverse positions are singular for cylinders.
pub fn cylindrical_particle_cluster(
    local: Vec<DMatrix<Complex>>,
    basis: crate::cw::Basis,
    ks: [Complex; 2],
    helicity: bool,
) -> Result<ParticleClusterResidual> {
    basis.validate()?;
    validate_particle_blocks(
        &local,
        &basis.positions,
        basis.modes.iter().map(|&(p, _)| p),
    )?;
    // cw::expansion takes no polarization convention, so the parity check happens here.
    validate_wavenumbers(ks, helicity, false)?;
    for (i, position) in basis.positions.iter().enumerate() {
        if basis.positions[..i].iter().any(|p| p[..2] == position[..2]) {
            return Err(Error::InvalidInput(
                "cylinders require distinct transverse positions".into(),
            ));
        }
    }
    let (coupling, expansion) =
        crate::cw::expansion(basis.clone(), basis, ks, crate::special::Radial::Singular)?;
    couple(local, coupling, ParticleExpansion::Cylindrical(expansion))
}

impl ParticleClusterResidual {
    /// Interacting response in the particles' local multipole coordinates.
    #[must_use]
    pub const fn value(&self) -> &DMatrix<Complex> {
        self.interaction.value()
    }

    /// The shape of the interacting response.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.interaction.shape()
    }

    /// Gradients of the local matrices, the positions and the wavenumbers from
    /// `cotangent`, the gradient of a real loss with respect to the interacting
    /// T-matrix.
    ///
    /// The position and wavenumber gradients come from the pullback of
    /// [`sw::ExpansionResidual`] or [`cw::ExpansionResidual`](crate::cw::ExpansionResidual)
    /// and add in their order; the LU solve and the matrix products run in faer with
    /// the worker count of [`linalg`](crate::linalg).
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<ParticleClusterGradient> {
        let InteractionGradient { local, coupling } =
            self.interaction.pullback_blocks(cotangent)?;
        let geometry = match self.expansion {
            ParticleExpansion::Spherical(expansion) => expansion.pullback(&coupling)?,
            ParticleExpansion::Cylindrical(expansion) => expansion.pullback(&coupling)?,
        };
        let positions = geometry
            .destination
            .into_iter()
            .zip(geometry.source)
            .map(|(a, b)| std::array::from_fn(|i| a[i] + b[i]))
            .collect();
        Ok(ParticleClusterGradient {
            local,
            positions,
            ks: geometry.ks,
        })
    }
}

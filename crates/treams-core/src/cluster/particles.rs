//! Clusters of arbitrary spherical or cylindrical particle T-matrices.
//!
//! Upstream: `treams.TMatrix.cluster(tmats, positions).interaction.solve()` and its
//! `TMatrixC` twin. The gradients are a treams-rs extension.
#![allow(clippy::indexing_slicing)] // Validated particle blocks, positions and Cartesian axes.

use nalgebra::DMatrix;

use super::{InteractionGradient, InteractionResidual, interaction_blocks};
use crate::sw;
use crate::{
    Complex, Error, Result,
    basis::validate_wavenumbers,
    saved::{Reader, SavedState, Writer},
};

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

fn invalid_state() -> Error {
    Error::InvalidInput("invalid native particle-cluster state".into())
}

/// The two sections have independent fixed layouts. Framing lets each component
/// validate its complete byte count before restoring numerical arrays.
fn framed_state_size(expansion: usize, interaction: usize) -> Result<usize> {
    17_usize
        .checked_add(expansion)
        .and_then(|size| size.checked_add(interaction))
        .ok_or_else(invalid_state)
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
    /// Byte count for a spherical cluster's state, from its local block sizes.
    pub fn spherical_state_size(local_sizes: &[usize]) -> Result<usize> {
        let (expansion, interaction) = Self::section_sizes(local_sizes, false)?;
        framed_state_size(expansion, interaction)
    }

    /// Byte count for a cylindrical cluster's state, from its local block sizes.
    pub fn cylindrical_state_size(local_sizes: &[usize]) -> Result<usize> {
        let (expansion, interaction) = Self::section_sizes(local_sizes, true)?;
        framed_state_size(expansion, interaction)
    }

    fn section_sizes(local_sizes: &[usize], cylindrical: bool) -> Result<(usize, usize)> {
        let interaction = InteractionResidual::state_size(local_sizes)?;
        let modes = local_sizes.iter().try_fold(0_usize, |sum, &size| {
            sum.checked_add(size).ok_or_else(invalid_state)
        })?;
        let positions = local_sizes.len();
        let expansion = if cylindrical {
            crate::cw::ExpansionResidual::state_size(modes, positions, modes, positions)?
        } else {
            sw::ExpansionResidual::state_size(modes, positions, modes, positions)?
        };
        Ok((expansion, interaction))
    }

    /// Shapes of the recorded particle T-matrices, in particle order.
    #[must_use]
    pub fn local_shapes(&self) -> impl ExactSizeIterator<Item = (usize, usize)> {
        self.interaction.local_shapes()
    }

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

    /// Directional derivative of the local matrices, particle positions and
    /// embedding wavenumbers. Both sides of each translation share the same
    /// position tangent; the interaction reuses its forward LU.
    pub fn pushforward(
        &self,
        local: &[DMatrix<Complex>],
        positions: &[[f64; 3]],
        ks: [Complex; 2],
    ) -> Result<DMatrix<Complex>> {
        let coupling = match &self.expansion {
            ParticleExpansion::Spherical(expansion) => {
                expansion.pushforward(positions, positions, ks)?
            }
            ParticleExpansion::Cylindrical(expansion) => {
                expansion.pushforward(positions, positions, ks)?
            }
        };
        self.interaction.pushforward_blocks(local, &coupling)
    }

    /// Gradients of the local matrices, the positions and the wavenumbers from
    /// `cotangent`, the gradient of a real loss with respect to the interacting
    /// T-matrix.
    ///
    /// The position and wavenumber gradients come from the pullback of
    /// [`sw::ExpansionResidual`] or [`cw::ExpansionResidual`](crate::cw::ExpansionResidual)
    /// and add in their order; the LU solve and the matrix products run in faer with
    /// the worker count of [`linalg`](crate::linalg).
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<ParticleClusterGradient> {
        let InteractionGradient { local, coupling } =
            self.interaction.pullback_blocks(cotangent)?;
        let geometry = match &self.expansion {
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

impl SavedState for ParticleClusterResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let cylindrical = matches!(&self.expansion, ParticleExpansion::Cylindrical(_));
        let local_sizes = self
            .local_shapes()
            .map(|(size, _)| size)
            .collect::<Vec<_>>();
        let (expansion, interaction) = Self::section_sizes(&local_sizes, cylindrical)?;
        let mut writer = Writer::new(framed_state_size(expansion, interaction)?);
        writer.byte(u8::from(cylindrical));
        writer.usize(expansion);
        writer.usize(interaction);
        match &self.expansion {
            ParticleExpansion::Spherical(expansion) => expansion.write_state(&mut writer),
            ParticleExpansion::Cylindrical(expansion) => expansion.write_state(&mut writer),
        }
        self.interaction.write_state(&mut writer);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let kind = reader.byte()?;
        if kind > 1 {
            return Err(invalid_state());
        }
        let expansion_size = reader.usize()?;
        let interaction_size = reader.usize()?;
        if bytes.len() != framed_state_size(expansion_size, interaction_size)? {
            return Err(invalid_state());
        }
        let expansion_bytes = reader.raw(expansion_size)?;
        let interaction_bytes = reader.raw(interaction_size)?;
        reader.finish()?;
        let (expansion, shape) = if kind == 0 {
            let residual = sw::ExpansionResidual::from_state(expansion_bytes)?;
            let shape = residual.shape();
            (ParticleExpansion::Spherical(residual), shape)
        } else {
            let residual = crate::cw::ExpansionResidual::from_state(expansion_bytes)?;
            let shape = residual.shape();
            (ParticleExpansion::Cylindrical(residual), shape)
        };
        let interaction = InteractionResidual::from_state(interaction_bytes)?;
        if shape != interaction.shape() {
            return Err(invalid_state());
        }
        Ok(Self {
            expansion,
            interaction,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::patterned;

    fn fixture(cylindrical: bool) -> ParticleClusterResidual {
        let positions = vec![[0.1, -0.2, 0.3], [1.7, 0.4, -0.5]];
        let ks = [Complex::new(1.1, 0.05), Complex::new(1.4, 0.08)];
        let local = vec![
            patterned(1, 1, 0.2) * Complex::from(0.02),
            patterned(2, 2, 0.3) * Complex::from(0.03),
        ];
        let labels = [(0, 0), (1, 0), (1, 1)];
        if cylindrical {
            let modes = labels
                .map(|(p, pol)| {
                    (
                        p,
                        crate::cw::Mode {
                            kz: 0.2,
                            m: i32::from(pol),
                            pol,
                        },
                    )
                })
                .to_vec();
            cylindrical_particle_cluster(local, crate::cw::Basis { modes, positions }, ks, true)
                .unwrap()
        } else {
            let modes = labels
                .map(|(p, pol)| {
                    (
                        p,
                        sw::Mode {
                            l: 1,
                            m: i32::from(pol),
                            pol,
                        },
                    )
                })
                .to_vec();
            particle_cluster(local, sw::Basis { modes, positions }, ks, true).unwrap()
        }
    }

    #[test]
    fn saved_state_preserves_particle_blocks_values_and_derivatives() {
        let local = [patterned(1, 1, 0.4), patterned(2, 2, 0.5)];
        let positions = [[0.1, 0.2, -0.3], [-0.2, 0.1, 0.4]];
        let ks = [Complex::new(0.2, -0.1), Complex::new(-0.15, 0.05)];
        let weight = patterned(3, 3, 0.6);
        for cylindrical in [false, true] {
            let original = fixture(cylindrical);
            let bytes = original.save_state().unwrap();
            let expected_size = if cylindrical {
                ParticleClusterResidual::cylindrical_state_size(&[1, 2])
            } else {
                ParticleClusterResidual::spherical_state_size(&[1, 2])
            }
            .unwrap();
            assert_eq!(bytes.len(), expected_size);
            let restored = ParticleClusterResidual::from_state(&bytes).unwrap();
            assert_eq!(
                restored.local_shapes().collect::<Vec<_>>(),
                [(1, 1), (2, 2)]
            );
            assert_eq!(restored.value(), original.value());
            let tangent = original.pushforward(&local, &positions, ks).unwrap();
            assert_eq!(
                restored.pushforward(&local, &positions, ks).unwrap(),
                tangent
            );
            let actual = restored.pullback(&weight).unwrap();
            let expected = original.pullback(&weight).unwrap();
            assert_eq!(actual.local, expected.local);
            assert_eq!(actual.positions, expected.positions);
            assert_eq!(actual.ks, expected.ks);
            assert_eq!(
                restored.pushforward(&local, &positions, ks).unwrap(),
                tangent
            );
            assert_eq!(restored.save_state().unwrap(), bytes);
        }
    }

    #[test]
    fn saved_state_rejects_inconsistent_section_sizes() {
        let bytes = fixture(false).save_state().unwrap();
        for length in [0, 1, 16, bytes.len() - 1] {
            assert!(ParticleClusterResidual::from_state(&bytes[..length]).is_err());
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(ParticleClusterResidual::from_state(&extra).is_err());
        let mut overflow = bytes;
        overflow[1..9].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(ParticleClusterResidual::from_state(&overflow).is_err());
    }
}

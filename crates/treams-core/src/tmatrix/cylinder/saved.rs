//! Fixed derivative state with one complete Mie state per output block.
//!
//! Mirrored blocks copy their shared solve when saved, so the byte count depends
//! only on input shapes. Restoring this state never solves a boundary problem.
//! treams-rs extension.

use super::{Block, CylinderResidual};
use crate::{
    MAX_DEGREE, Result,
    coeffs::MieCylResidual,
    saved::{Reader, SavedState, Writer, invalid},
};

impl CylinderResidual {
    /// Byte count of saved derivative state for these input shape parameters.
    pub fn state_size(kz_count: usize, mmax: usize, boundaries: usize) -> Result<usize> {
        if kz_count == 0 || mmax > MAX_DEGREE.unsigned_abs() as usize {
            return Err(invalid());
        }
        let blocks = kz_count.checked_mul(2 * mmax + 1).ok_or_else(invalid)?;
        MieCylResidual::state_size(boundaries)?
            .checked_add(9)
            .and_then(|per_block| per_block.checked_mul(blocks))
            .and_then(|states| states.checked_add(24))
            .ok_or_else(invalid)
    }
}

impl SavedState for CylinderResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mmax = (self.blocks.len() / self.kz_count - 1) / 2;
        let mut writer = Writer::new(Self::state_size(self.kz_count, mmax, self.boundaries)?);
        writer.usize(self.kz_count);
        writer.usize(self.boundaries);
        writer.usize(self.blocks.len());
        for block in &self.blocks {
            writer.usize(block.kz_index);
            writer.byte(u8::from(block.mirrored));
            self.solves[block.solve].write_state(&mut writer);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let kz_count = reader.usize()?;
        let boundaries = reader.usize()?;
        let block_count = reader.usize()?;
        if kz_count == 0 || block_count == 0 || !block_count.is_multiple_of(kz_count) {
            return Err(invalid());
        }
        let orders = block_count / kz_count;
        if orders.is_multiple_of(2)
            || Self::state_size(kz_count, (orders - 1) / 2, boundaries)? != bytes.len()
        {
            return Err(invalid());
        }
        let mut blocks = Vec::with_capacity(block_count);
        let mut solves = Vec::with_capacity(block_count);
        for solve in 0..block_count {
            let kz_index = reader.usize()?;
            if kz_index >= kz_count {
                return Err(invalid());
            }
            let mirrored = match reader.byte()? {
                0 => false,
                1 => true,
                _ => return Err(invalid()),
            };
            let residual = MieCylResidual::read_state(&mut reader)?;
            if residual.boundaries() != boundaries {
                return Err(invalid());
            }
            blocks.push(Block {
                kz_index,
                solve,
                mirrored,
            });
            solves.push(residual);
        }
        reader.finish()?;
        Ok(Self {
            blocks,
            solves,
            kz_count,
            boundaries,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // Serialization preserves numerical bits exactly.

    use super::*;
    use crate::{Complex, coeffs::Material, tmatrix::cylinder};
    use nalgebra::DMatrix;

    #[test]
    fn saved_cylinder_preserves_mirrors_and_independent_kz_directions() -> Result<()> {
        let materials = [
            Material {
                epsilon: Complex::new(2.2, 0.1),
                kappa: Complex::new(0.06, 0.0),
                ..Material::default()
            },
            Material::default(),
        ];
        let (_, original) = cylinder(&[-0.2, 0.0, 0.2], 1, 1.3, &[0.7], &materials)?;
        let bytes = original.save_state()?;
        assert_eq!(bytes.len(), CylinderResidual::state_size(3, 1, 1)?);
        let restored = CylinderResidual::from_state(&bytes)?;
        let direction = [Material {
            epsilon: Complex::new(0.2, 0.1),
            mu: Complex::new(-0.1, 0.0),
            kappa: Complex::new(0.03, 0.0),
        }; 2];
        let kzs = [0.1, -0.2, 0.3];
        let tangent = original.pushforward(&kzs, 0.2, &[-0.1], &direction)?;
        let (rows, cols) = original.shape();
        let cotangent = DMatrix::from_element(rows, cols, Complex::new(0.3, -0.2));
        let expected = original.pullback(&cotangent)?;
        for _ in 0..2 {
            assert_eq!(
                restored.pushforward(&kzs, 0.2, &[-0.1], &direction)?,
                tangent
            );
            let actual = restored.pullback(&cotangent)?;
            assert_eq!(actual.kzs, expected.kzs);
            assert_eq!(actual.k0, expected.k0);
            assert_eq!(actual.layers.radii, expected.layers.radii);
            assert_eq!(actual.layers.epsilon, expected.layers.epsilon);
            assert_eq!(actual.layers.mu, expected.layers.mu);
            assert_eq!(actual.layers.kappa, expected.layers.kappa);
        }
        assert_eq!(restored.save_state()?, bytes);
        Ok(())
    }

    #[test]
    fn saved_cylinder_checks_lengths_and_indices() -> Result<()> {
        let (_, residual) = cylinder(&[0.2], 0, 1.0, &[0.5], &[Material::default(); 2])?;
        let bytes = residual.save_state()?;
        assert!(CylinderResidual::from_state(&bytes[..bytes.len() - 1]).is_err());
        let mut extended = bytes.clone();
        extended.push(0);
        assert!(CylinderResidual::from_state(&extended).is_err());
        for offset in [0, 8, 16, 24, 33] {
            let mut corrupt = bytes.clone();
            corrupt[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(CylinderResidual::from_state(&corrupt).is_err());
        }
        let mut corrupt = bytes;
        corrupt[32] = 2;
        assert!(CylinderResidual::from_state(&corrupt).is_err());
        assert!(CylinderResidual::state_size(usize::MAX, 1, 1).is_err());
        assert!(CylinderResidual::state_size(1, 1, usize::MAX).is_err());
        Ok(())
    }
}

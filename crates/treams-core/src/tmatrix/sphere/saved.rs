//! Fixed derivative state: shape, physical radii and complete per-degree Mie states.
//! treams-rs extension.

use super::SphereResidual;
use crate::{
    MAX_DEGREE, Result,
    coeffs::MieResidual,
    saved::{Reader, SavedState, Writer, invalid},
};

impl SphereResidual {
    pub(crate) fn lmax(&self) -> usize {
        self.degrees.len()
    }

    /// Byte count of saved derivative state for this degree and layer count.
    pub fn state_size(lmax: usize, boundaries: usize) -> Result<usize> {
        if lmax == 0 || lmax > MAX_DEGREE.unsigned_abs() as usize {
            return Err(invalid());
        }
        let degrees = MieResidual::state_size(boundaries)?
            .checked_mul(lmax)
            .ok_or_else(invalid)?;
        boundaries
            .checked_mul(8)
            .and_then(|radii| radii.checked_add(24))
            .and_then(|header| header.checked_add(degrees))
            .ok_or_else(invalid)
    }

    pub(crate) fn write_state(&self, writer: &mut Writer) {
        writer.usize(self.degrees.len());
        writer.usize(self.radii.len());
        writer.f64(self.k0);
        for &radius in &self.radii {
            writer.f64(radius);
        }
        for degree in &self.degrees {
            degree.write_state(writer);
        }
    }

    pub(crate) fn read_state(reader: &mut Reader<'_>) -> Result<Self> {
        let available = reader.remaining_len();
        let lmax = reader.usize()?;
        let boundaries = reader.usize()?;
        if Self::state_size(lmax, boundaries)? > available {
            return Err(invalid());
        }
        let k0 = reader.f64()?;
        let radii = (0..boundaries)
            .map(|_| reader.f64())
            .collect::<Result<Vec<_>>>()?;
        let degrees = (0..lmax)
            .map(|_| {
                let degree = MieResidual::read_state(reader)?;
                if degree.boundaries() != boundaries {
                    return Err(invalid());
                }
                Ok(degree)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { k0, radii, degrees })
    }
}

impl SavedState for SphereResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(self.degrees.len(), self.radii.len())?);
        self.write_state(&mut writer);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let residual = Self::read_state(&mut reader)?;
        reader.finish()?;
        Ok(residual)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // Serialization preserves numerical bits exactly.

    use super::*;
    use crate::{Complex, coeffs::Material, tmatrix::sphere};
    use nalgebra::DMatrix;

    #[test]
    fn saved_sphere_reuses_identical_derivatives() -> Result<()> {
        let materials = [
            Material {
                epsilon: Complex::new(2.2, 0.1),
                kappa: Complex::new(0.06, 0.0),
                ..Material::default()
            },
            Material::default(),
        ];
        let (_, original) = sphere(2, 1.3, &[0.7], &materials)?;
        let bytes = original.save_state()?;
        assert_eq!(bytes.len(), SphereResidual::state_size(2, 1)?);
        let restored = SphereResidual::from_state(&bytes)?;
        let direction = [Material {
            epsilon: Complex::new(0.2, 0.1),
            mu: Complex::new(-0.1, 0.0),
            kappa: Complex::new(0.03, 0.0),
        }; 2];
        let tangent = original.pushforward(0.2, &[-0.1], &direction)?;
        let (rows, cols) = original.shape();
        let cotangent = DMatrix::from_element(rows, cols, Complex::new(0.3, -0.2));
        let expected = original.pullback(&cotangent)?;
        for _ in 0..2 {
            assert_eq!(restored.pushforward(0.2, &[-0.1], &direction)?, tangent);
            let actual = restored.pullback(&cotangent)?;
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
    fn saved_sphere_checks_lengths_before_allocation() -> Result<()> {
        let (_, residual) = sphere(1, 1.0, &[0.5], &[Material::default(); 2])?;
        let bytes = residual.save_state()?;
        assert!(SphereResidual::from_state(&bytes[..bytes.len() - 1]).is_err());
        let mut extended = bytes.clone();
        extended.push(0);
        assert!(SphereResidual::from_state(&extended).is_err());
        for offset in [0, 8, 32] {
            let mut corrupt = bytes.clone();
            corrupt[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(SphereResidual::from_state(&corrupt).is_err());
        }
        assert!(SphereResidual::state_size(1, usize::MAX).is_err());
        Ok(())
    }
}

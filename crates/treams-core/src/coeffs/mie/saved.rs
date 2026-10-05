//! Exact-size numerical state for the Mie derivative callbacks.
//!
//! treams-rs extension.

use super::{Interface, Material, MieResidual, Riccati};
use crate::{
    Complex, Error, Result,
    saved::{Reader, SavedState, Writer},
};

fn invalid() -> Error {
    Error::InvalidInput("invalid Mie derivative state".into())
}

impl MieResidual {
    /// Exact byte count of the saved Mie state for `boundaries` layers.
    /// The layout contains one count, the sizes, three material parameters, 62
    /// complex entries per interface, and two coefficient matrices.
    pub fn state_size(boundaries: usize) -> Result<usize> {
        if boundaries == 0 {
            return Err(invalid());
        }
        boundaries
            .checked_mul(1048)
            .and_then(|size| size.checked_add(184))
            .ok_or_else(invalid)
    }

    pub(crate) fn write_state(&self, writer: &mut Writer) {
        writer.usize(self.sizes.len());
        for &size in &self.sizes {
            writer.f64(size);
        }
        for material in &self.materials {
            writer.complex(material.epsilon);
            writer.complex(material.mu);
            writer.complex(material.kappa);
        }
        for interface in &self.interfaces {
            for &value in interface.x.iter().flatten().chain(&interface.z) {
                writer.complex(value);
            }
            for radial in interface.radials.iter().flatten() {
                for &(value, derivative) in radial.waves.iter().chain(&radial.slopes) {
                    writer.complex(value);
                    writer.complex(derivative);
                }
            }
            for &value in interface.matrix.iter().chain(interface.before.iter()) {
                writer.complex(value);
            }
        }
        for &value in self.inverse.iter().chain(self.value.iter()) {
            writer.complex(value);
        }
    }

    pub(crate) fn read_state(reader: &mut Reader<'_>) -> Result<Self> {
        let boundaries = reader.usize()?;
        let expected = Self::state_size(boundaries)?;
        if reader.remaining_len() < expected - 8 {
            return Err(invalid());
        }
        let sizes = (0..boundaries)
            .map(|_| reader.f64())
            .collect::<Result<Vec<_>>>()?;
        let materials = (0..=boundaries)
            .map(|_| {
                Ok(Material {
                    epsilon: reader.complex()?,
                    mu: reader.complex()?,
                    kappa: reader.complex()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut interfaces = Vec::with_capacity(boundaries);
        for _ in 0..boundaries {
            let mut x = [[Complex::default(); 2]; 2];
            let mut z = [Complex::default(); 2];
            for value in x.iter_mut().flatten().chain(&mut z) {
                *value = reader.complex()?;
            }
            let mut radials = [[Riccati {
                waves: [(Complex::default(), Complex::default()); 2],
                slopes: [(Complex::default(), Complex::default()); 2],
            }; 2]; 2];
            for radial in radials.iter_mut().flatten() {
                for (value, derivative) in radial.waves.iter_mut().chain(&mut radial.slopes) {
                    *value = reader.complex()?;
                    *derivative = reader.complex()?;
                }
            }
            interfaces.push(Interface {
                x,
                z,
                radials,
                matrix: read_matrix(reader)?,
                before: read_matrix(reader)?,
            });
        }
        Ok(Self {
            sizes,
            materials,
            interfaces,
            inverse: read_matrix(reader)?,
            value: read_matrix(reader)?,
        })
    }
}

fn read_matrix<const R: usize, const C: usize>(
    reader: &mut Reader<'_>,
) -> Result<nalgebra::SMatrix<Complex, R, C>> {
    let mut matrix = nalgebra::SMatrix::<Complex, R, C>::zeros();
    for value in matrix.iter_mut() {
        *value = reader.complex()?;
    }
    Ok(matrix)
}

impl SavedState for MieResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(self.sizes.len())?);
        self.write_state(&mut writer);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let result = Self::read_state(&mut reader)?;
        reader.finish()?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coeffs::{Matrix2, mie};

    #[test]
    fn state_roundtrip_preserves_value_and_both_derivatives() {
        let materials = [
            Material {
                epsilon: Complex::new(2.7, 0.2),
                mu: Complex::new(1.1, 0.03),
                kappa: Complex::new(0.12, 0.02),
            },
            Material {
                epsilon: Complex::new(1.8, 0.1),
                ..Material::default()
            },
            Material::default(),
        ];
        let residual = mie(3, &[0.4, 0.7], &materials).unwrap();
        let state = residual.save_state().unwrap();
        assert_eq!(state.len(), MieResidual::state_size(2).unwrap());
        let restored = MieResidual::from_state(&state).unwrap();
        assert_eq!(restored.value(), residual.value());
        let directions = [Material {
            epsilon: Complex::new(0.1, -0.02),
            mu: Complex::new(-0.03, 0.02),
            kappa: Complex::new(0.02, 0.01),
        }; 3];
        assert_eq!(
            restored.pushforward(&[0.1, -0.2], &directions).unwrap(),
            residual.pushforward(&[0.1, -0.2], &directions).unwrap()
        );
        let cotangent = Matrix2::repeat(Complex::new(0.2, -0.3));
        let actual = restored.pullback(&cotangent).unwrap();
        let expected = residual.pullback(&cotangent).unwrap();
        assert_eq!(actual.sizes, expected.sizes);
        assert_eq!(actual.epsilon, expected.epsilon);
        assert_eq!(actual.mu, expected.mu);
        assert_eq!(actual.kappa, expected.kappa);
        assert_eq!(restored.save_state().unwrap(), state);
    }

    #[test]
    fn state_rejects_inconsistent_dimensions_before_allocation() {
        let residual = mie(1, &[0.5], &[Material::default(); 2]).unwrap();
        let mut state = residual.save_state().unwrap();
        assert!(MieResidual::from_state(&state[..state.len() - 1]).is_err());
        state[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(MieResidual::from_state(&state).is_err());
        assert!(MieResidual::state_size(usize::MAX).is_err());
        assert!(MieResidual::state_size(0).is_err());
    }
}

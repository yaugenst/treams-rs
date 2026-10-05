//! Fixed numerical state for a cylindrical coefficient record, without recomputation.
//!
//! treams-rs extension.

use super::{Interface, Material, Matrix2, Matrix4, Matrix42, MieCylResidual, RadialJet, Side};
use crate::{
    Error, MAX_DEGREE, Result,
    saved::{Reader, SavedState, Writer},
};

fn invalid() -> Error {
    Error::InvalidInput("invalid cylindrical Mie derivative state".into())
}

impl MieCylResidual {
    /// Byte count of the saved numerical state for `boundaries` layer boundaries.
    pub fn state_size(boundaries: usize) -> Result<usize> {
        if boundaries == 0 {
            return Err(invalid());
        }
        // Two integer headers, n+2 real numbers, and 81*n+5 complex numbers.
        boundaries
            .checked_mul(1304)
            .and_then(|size| size.checked_add(112))
            .ok_or_else(invalid)
    }

    #[allow(clippy::cast_sign_loss)] // The recorded order is in -MAX_DEGREE..=MAX_DEGREE.
    pub(crate) fn write_state(&self, writer: &mut Writer) {
        writer.usize(self.radii.len());
        writer.usize((self.order + MAX_DEGREE) as usize);
        writer.f64(self.kz);
        writer.f64(self.k0);
        for &radius in &self.radii {
            writer.f64(radius);
        }
        for material in &self.materials {
            writer.complex(material.epsilon);
            writer.complex(material.mu);
            writer.complex(material.kappa);
        }
        for interface in &self.interfaces {
            interface.inside.write_state(writer);
            interface.outside.write_state(writer);
            for &value in interface
                .inverse_outer
                .iter()
                .chain(interface.transfer.iter())
                .chain(interface.before.iter())
            {
                writer.complex(value);
            }
        }
        for &value in self.inverse.iter().chain(self.value.iter()) {
            writer.complex(value);
        }
    }

    pub(crate) fn read_state(reader: &mut Reader<'_>) -> Result<Self> {
        let boundaries = reader.usize()?;
        let size = Self::state_size(boundaries)?;
        if reader.remaining_len() < size - 8 {
            return Err(invalid());
        }
        let encoded = i32::try_from(reader.usize()?).map_err(|_| invalid())?;
        if encoded > 2 * MAX_DEGREE {
            return Err(invalid());
        }
        let order = encoded - MAX_DEGREE;
        let kz = reader.f64()?;
        let k0 = reader.f64()?;
        let radii = (0..boundaries)
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
        for i in 0..boundaries {
            let inside = Side::read_state(reader, i == 0)?;
            let outside = Side::read_state(reader, false)?;
            let mut inverse_outer = Matrix4::zeros();
            let mut transfer = Matrix4::zeros();
            let mut before = Matrix42::zeros();
            for value in inverse_outer
                .iter_mut()
                .chain(transfer.iter_mut())
                .chain(before.iter_mut())
            {
                *value = reader.complex()?;
            }
            interfaces.push(Interface {
                inside,
                outside,
                inverse_outer,
                transfer,
                before,
            });
        }
        let mut inverse = Matrix2::zeros();
        let mut value = Matrix2::zeros();
        for entry in inverse.iter_mut().chain(value.iter_mut()) {
            *entry = reader.complex()?;
        }
        Ok(Self {
            order,
            kz,
            k0,
            radii,
            materials,
            interfaces,
            inverse,
            value,
        })
    }
}

impl Side {
    fn write_state(&self, writer: &mut Writer) {
        for value in self
            .ks
            .into_iter()
            .chain([self.impedance])
            .chain(self.kr)
            .chain(self.x)
        {
            writer.complex(value);
        }
        // The first inside side has only regular waves. Its absent singular
        // entries follow from the boundary topology and need no tags.
        for radial in self.radial.iter().flatten().flatten() {
            writer.complex(radial.value);
            writer.complex(radial.first);
            writer.complex(radial.second);
        }
    }

    fn read_state(reader: &mut Reader<'_>, regular_only: bool) -> Result<Self> {
        let ks = [reader.complex()?, reader.complex()?];
        let impedance = reader.complex()?;
        let kr = [reader.complex()?, reader.complex()?];
        let x = [reader.complex()?, reader.complex()?];
        let mut radial = [[None; 2]; 2];
        for polarization in &mut radial {
            for entry in polarization
                .iter_mut()
                .take(if regular_only { 1 } else { 2 })
            {
                *entry = Some(RadialJet {
                    value: reader.complex()?,
                    first: reader.complex()?,
                    second: reader.complex()?,
                });
            }
        }
        Ok(Self {
            ks,
            impedance,
            kr,
            x,
            radial,
        })
    }
}

impl SavedState for MieCylResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(self.boundaries())?);
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
    use crate::{Complex, coeffs::mie_cyl};

    fn record(order: i32, boundaries: usize) -> MieCylResidual {
        let radii: Vec<_> = (0..boundaries)
            .map(|i| 0.2 + 0.3 * f64::from(u32::try_from(i).unwrap()))
            .collect();
        let materials: Vec<_> = (0..=boundaries)
            .map(|i| Material {
                epsilon: Complex::new(2.0 + 0.4 * f64::from(u32::try_from(i).unwrap()), 0.1),
                mu: Complex::new(1.2, 0.03),
                kappa: Complex::new(0.02, 0.004),
            })
            .collect();
        mie_cyl(0.3, order, 1.2, &radii, &materials).unwrap()
    }

    #[test]
    fn saved_state_preserves_values_and_both_derivatives() {
        for boundaries in [1, 2, 3] {
            for order in [-3, 0, 2] {
                let original = record(order, boundaries);
                let bytes = original.save_state().unwrap();
                assert_eq!(bytes.len(), MieCylResidual::state_size(boundaries).unwrap());
                let restored = MieCylResidual::from_state(&bytes).unwrap();
                assert_eq!(restored.save_state().unwrap(), bytes);
                assert_eq!(restored.value(), original.value());
                let radii = vec![0.07; boundaries];
                let materials = vec![
                    Material {
                        epsilon: Complex::new(0.1, 0.03),
                        mu: Complex::new(-0.02, 0.01),
                        kappa: Complex::new(0.04, -0.02),
                    };
                    boundaries + 1
                ];
                assert_eq!(
                    restored
                        .pushforward(0.03, -0.04, &radii, &materials)
                        .unwrap(),
                    original
                        .pushforward(0.03, -0.04, &radii, &materials)
                        .unwrap()
                );
                let cotangent = Matrix2::from_fn(|i, j| {
                    Complex::new(
                        0.3 + f64::from(u32::try_from(i).unwrap()) * 0.1,
                        -0.2 + f64::from(u32::try_from(j).unwrap()) * 0.15,
                    )
                });
                let actual = restored.pullback(&cotangent).unwrap();
                let expected = original.pullback(&cotangent).unwrap();
                assert_eq!(actual.kz.to_bits(), expected.kz.to_bits());
                assert_eq!(actual.k0.to_bits(), expected.k0.to_bits());
                assert_eq!(actual.layers.radii, expected.layers.radii);
                assert_eq!(actual.layers.epsilon, expected.layers.epsilon);
                assert_eq!(actual.layers.mu, expected.layers.mu);
                assert_eq!(actual.layers.kappa, expected.layers.kappa);
            }
        }
    }

    #[test]
    fn saved_state_rejects_invalid_counts_orders_and_lengths() {
        let bytes = record(1, 2).save_state().unwrap();
        for length in [0, 7, 8, 15, bytes.len() - 1] {
            assert!(MieCylResidual::from_state(&bytes[..length]).is_err());
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(MieCylResidual::from_state(&trailing).is_err());
        for boundaries in [0_u64, 1, u64::MAX] {
            let mut invalid = bytes.clone();
            invalid[..8].copy_from_slice(&boundaries.to_le_bytes());
            assert!(MieCylResidual::from_state(&invalid).is_err());
        }
        for boundaries in [0_u64, 3, u64::MAX] {
            let mut invalid = bytes.clone();
            invalid[..8].copy_from_slice(&boundaries.to_le_bytes());
            assert!(MieCylResidual::read_state(&mut Reader::new(&invalid)).is_err());
        }
        for order in [257_u64, u64::MAX] {
            let mut invalid = bytes.clone();
            invalid[8..16].copy_from_slice(&order.to_le_bytes());
            assert!(MieCylResidual::from_state(&invalid).is_err());
        }
        assert!(MieCylResidual::state_size(usize::MAX).is_err());
    }
}

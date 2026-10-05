//! Fixed invocation-state layout of independent planar channels.
//!
//! treams-rs extension. Every nested interface and Redheffer residual stores its
//! computed matrices and factors; restoration never reruns the layer stack.

use super::{Channel, LayerStackResidual, Step};
use crate::{
    Result,
    saved::{Reader, SavedState, Writer},
    smatrix::{
        AddResidual, InterfaceResidual,
        saved::{invalid, product, read_complex, read_f64, read_matrix, total, write_matrix},
    },
};

impl LayerStackResidual {
    /// Fixed state byte count for the media and independent transverse channels.
    pub fn state_size(media: usize, channels: usize) -> Result<usize> {
        if media < 2 || channels == 0 {
            return Err(invalid());
        }
        let layers = media - 2;
        let interface = InterfaceResidual::state_size()?;
        let step = total(&[256, AddResidual::state_size(2)?, interface])?;
        let channel = total(&[interface, product(&[layers, step])?])?;
        total(&[
            17,
            product(&[media, 32])?,
            product(&[channels, 16])?,
            product(&[layers, 8])?,
            product(&[channels, channel])?,
        ])
    }
}

impl SavedState for LayerStackResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(self.ks.len(), self.channels.len())?);
        writer.usize(self.ks.len());
        writer.usize(self.channels.len());
        writer.byte(u8::from(self.fixed_q));
        for &value in self.ks.iter().flatten() {
            writer.complex(value);
        }
        for &value in self.q.iter().flatten().chain(&self.thickness) {
            writer.f64(value);
        }
        for channel in &self.channels {
            writer.raw(&channel.initial.save_state()?);
            for step in &channel.steps {
                for &value in step.normal.iter().chain(&step.phase) {
                    writer.complex(value);
                }
                for block in &step.below {
                    write_matrix(&mut writer, block);
                }
                writer.raw(&step.boundary.save_state()?);
                writer.raw(&step.interface.save_state()?);
            }
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let (media, count) = (reader.usize()?, reader.usize()?);
        if bytes.len() != Self::state_size(media, count)? {
            return Err(invalid());
        }
        let fixed_q = match reader.byte()? {
            0 => false,
            1 => true,
            _ => return Err(invalid()),
        };
        let ks = (0..media)
            .map(|_| Ok([read_complex(&mut reader)?, read_complex(&mut reader)?]))
            .collect::<Result<Vec<_>>>()?;
        let q = (0..count)
            .map(|_| Ok([read_f64(&mut reader)?, read_f64(&mut reader)?]))
            .collect::<Result<Vec<_>>>()?;
        let thickness = (0..media - 2)
            .map(|_| {
                let value = read_f64(&mut reader)?;
                if value < 0.0 {
                    return Err(invalid());
                }
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()?;
        let interface_size = InterfaceResidual::state_size()?;
        let boundary_size = AddResidual::state_size(2)?;
        let channels = (0..count)
            .map(|_| {
                let initial = InterfaceResidual::from_state(reader.raw(interface_size)?)?;
                let steps = (0..media - 2)
                    .map(|_| {
                        let normal = [read_complex(&mut reader)?, read_complex(&mut reader)?];
                        let phase = [read_complex(&mut reader)?, read_complex(&mut reader)?];
                        let below = [
                            read_matrix(&mut reader, 2, 2)?,
                            read_matrix(&mut reader, 2, 2)?,
                            read_matrix(&mut reader, 2, 2)?,
                        ];
                        let boundary = AddResidual::from_state(reader.raw(boundary_size)?)?;
                        if boundary.shape() != (2, 2) {
                            return Err(invalid());
                        }
                        let interface = InterfaceResidual::from_state(reader.raw(interface_size)?)?;
                        Ok(Step {
                            normal,
                            phase,
                            below,
                            boundary,
                            interface,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Channel { initial, steps })
            })
            .collect::<Result<Vec<_>>>()?;
        reader.finish()?;
        Ok(Self {
            ks,
            q,
            thickness,
            channels,
            fixed_q,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Complex, smatrix::layer_stack, test_support::patterned};

    #[test]
    fn layer_state_roundtrip_reuses_both_derivative_directions() {
        let c = Complex::new;
        let ks = vec![
            [c(1.0, 0.01); 2],
            [c(1.5, 0.1), c(1.6, 0.1)],
            [c(1.2, 0.02); 2],
        ];
        let zs = [c(1.0, 0.0), c(0.7, 0.0), c(0.9, 0.0)];
        let q = vec![[0.2, 0.1], [0.1, -0.2]];
        let (_, residual) = layer_stack(ks.clone(), &zs, q.clone(), &[0.4], 2, false).unwrap();
        let state = residual.save_state().unwrap();
        assert_eq!(state.len(), LayerStackResidual::state_size(3, 2).unwrap());
        let restored = LayerStackResidual::from_state(&state).unwrap();
        let tangent = residual.pushforward(&ks, &zs, &q, &[0.1]).unwrap();
        assert_eq!(tangent, restored.pushforward(&ks, &zs, &q, &[0.1]).unwrap());
        let cotangent = vec![std::array::from_fn(|_| patterned(2, 2, 0.3)); 2];
        let expected = residual.pullback(cotangent.clone()).unwrap();
        let actual = restored.pullback(cotangent).unwrap();
        assert_eq!(expected.ks, actual.ks);
        assert_eq!(expected.zs, actual.zs);
        assert_eq!(expected.q, actual.q);
        assert_eq!(expected.thickness, actual.thickness);
        assert_eq!(tangent, restored.pushforward(&ks, &zs, &q, &[0.1]).unwrap());
        assert!(LayerStackResidual::from_state(&state[..state.len() - 1]).is_err());
        let mut bad = state;
        bad[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(LayerStackResidual::from_state(&bad).is_err());
    }
}

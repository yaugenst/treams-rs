//! Numeric port fields and powers preserved across an array-only callback.
//! treams-rs extension.

use super::super::saved::{
    invalid, product, read_complex, read_f64, read_matrix, read_stored, total, write_matrix,
    write_stored,
};
use super::{PortWaves, TrForward, TrPorts, TrResidual};
use crate::{
    Complex, Result,
    numerics::Jet,
    saved::{Reader, SavedState, Writer},
};
use nalgebra::DMatrix;

impl TrResidual {
    /// Byte length for mode count, illumination columns and distinct port groups.
    pub fn state_size(n: usize, columns: usize, groups: usize) -> Result<usize> {
        if n == 0 || columns == 0 || groups == 0 {
            return Err(invalid());
        }
        total(&[
            140,
            product(&[32, n, n])?,
            product(&[48, n, columns])?,
            product(&[528, groups])?,
            product(&[9, n])?,
            product(&[24, columns])?,
        ])
    }
}

impl SavedState for TrResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let (n, columns, groups) = self.input_shape();
        let mut writer = Writer::new(Self::state_size(n, columns, groups)?);
        for dimension in [n, columns, groups, self.ports.axis, self.ports.direction] {
            writer.usize(dimension);
        }
        writer.byte(u8::from(self.ports.helicity));
        writer.byte(u8::from(self.fixed_q));
        for matrix in &self.matrices {
            write_stored(&mut writer, matrix);
        }
        write_matrix(&mut writer, &self.incident);
        for &value in self.ports.ks.iter().flatten().chain(&self.ports.zs) {
            writer.complex(value);
        }
        for &value in self.ports.q.iter().flatten() {
            writer.f64(value);
        }
        for &(group, pol) in &self.ports.modes {
            writer.usize(group);
            writer.byte(pol);
        }
        for &value in self.forward.value.iter().chain(&self.forward.flux) {
            writer.f64(value);
        }
        for matrix in &self.forward.outgoing {
            write_matrix(&mut writer, matrix);
        }
        for value in self
            .forward
            .waves
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .flatten()
        {
            writer.complex(value.value);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let n = reader.usize()?;
        let columns = reader.usize()?;
        let groups = reader.usize()?;
        let axis = reader.usize()?;
        let direction = reader.usize()?;
        let helicity = reader.byte()?;
        let fixed_q = reader.byte()?;
        if axis > 2
            || direction > 1
            || helicity > 1
            || fixed_q > 1
            || bytes.len() != Self::state_size(n, columns, groups)?
        {
            return Err(invalid());
        }
        let matrices = [read_stored(&mut reader, n)?, read_stored(&mut reader, n)?];
        let incident = read_matrix(&mut reader, n, columns)?;
        let mut ks = [[Complex::default(); 2]; 2];
        let mut zs = [Complex::default(); 2];
        for value in ks.iter_mut().flatten().chain(&mut zs) {
            *value = read_complex(&mut reader)?;
        }
        let q = (0..groups)
            .map(|_| Ok([read_f64(&mut reader)?, read_f64(&mut reader)?]))
            .collect::<Result<Vec<_>>>()?;
        let modes = (0..n)
            .map(|_| Ok((reader.usize()?, reader.byte()?)))
            .collect::<Result<Vec<_>>>()?;
        let ports = TrPorts {
            ks,
            zs,
            q,
            modes,
            axis,
            direction,
            helicity: helicity != 0,
        };
        ports.validate()?;
        let mut value = DMatrix::zeros(2, columns);
        for v in value.iter_mut() {
            *v = read_f64(&mut reader)?;
        }
        let flux = (0..columns)
            .map(|_| {
                let flux = read_f64(&mut reader)?;
                if flux > 0.0 { Ok(flux) } else { Err(invalid()) }
            })
            .collect::<Result<Vec<_>>>()?;
        let outgoing = [
            read_matrix(&mut reader, n, columns)?,
            read_matrix(&mut reader, n, columns)?,
        ];
        let mut waves: Vec<PortWaves<0>> = vec![Default::default(); groups];
        for value in waves.iter_mut().flatten().flatten().flatten().flatten() {
            *value = Jet::constant(read_complex(&mut reader)?);
        }
        reader.finish()?;
        Ok(Self {
            matrices,
            incident,
            ports,
            forward: TrForward {
                value,
                flux,
                outgoing,
                waves,
            },
            fixed_q: fixed_q != 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        linalg::view,
        smatrix::{self, Blocks},
    };

    #[test]
    fn tr_roundtrip_preserves_saved_fields_derivatives_and_matrix_layout() {
        let n = 2;
        let matrix = DMatrix::from_row_slice(
            n,
            n,
            &[
                Complex::new(0.8, 0.1),
                Complex::new(0.1, 0.03),
                Complex::new(-0.03, 0.07),
                Complex::new(0.7, 0.12),
            ],
        );
        let reflected = &matrix * Complex::from(0.1);
        let incident = DMatrix::from_element(n, 3, Complex::new(0.3, -0.1));
        let ports = TrPorts {
            ks: [[Complex::new(1.3, 0.03); 2], [Complex::new(1.2, 0.04); 2]],
            zs: [Complex::new(0.9, 0.01), Complex::new(1.1, -0.01)],
            q: vec![[0.2, -0.1]],
            modes: vec![(0, 0), (0, 1)],
            axis: 2,
            helicity: true,
            direction: 0,
        };
        // The transposed view exercises preservation of row-major saved storage.
        let original = smatrix::tr(
            [view(&matrix).transpose(), view(&reflected)],
            incident.clone(),
            ports,
            false,
        )
        .unwrap();
        let mut bytes = original.save_state().unwrap();
        assert_eq!(bytes.len(), TrResidual::state_size(n, 3, 1).unwrap());
        assert!(TrResidual::from_state(&bytes[..bytes.len() - 1]).is_err());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(TrResidual::from_state(&extra).is_err());
        let restored = TrResidual::from_state(&bytes).unwrap();
        assert!(restored.matrices[0].row_major);
        assert!(!restored.matrices[1].row_major);
        assert_eq!(original.value(), restored.value());
        let matrices: Blocks = std::array::from_fn(|_| &matrix * Complex::from(0.03));
        let ks = [[Complex::new(0.1, -0.02); 2]; 2];
        let zs = [Complex::new(0.03, -0.01); 2];
        let q = [[0.02, -0.01]];
        assert_eq!(
            original
                .pushforward(&matrices, &incident, ks, zs, &q)
                .unwrap(),
            restored
                .pushforward(&matrices, &incident, ks, zs, &q)
                .unwrap()
        );
        let cotangent = DMatrix::from_element(2, 3, 0.4);
        let (a, b) = (
            original.pullback(&cotangent).unwrap(),
            restored.pullback(&cotangent).unwrap(),
        );
        assert_eq!(a.matrices, b.matrices);
        assert_eq!(a.incident, b.incident);
        assert_eq!(a.ks, b.ks);
        assert_eq!(a.zs, b.zs);
        assert_eq!(a.q, b.q);
        bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(TrResidual::from_state(&bytes).is_err());
    }
}

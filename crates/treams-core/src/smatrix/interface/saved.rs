//! Fixed numeric layouts for interface and propagation derivative state.
//! treams-rs extension.

use super::super::saved::{invalid, product, read_complex, read_f64, total};
use super::{FresnelResidual, InterfaceMatrix, InterfaceResidual, PropagationResidual};
use crate::{
    Complex, Result,
    saved::{Reader, SavedState, Writer},
};

impl FresnelResidual {
    /// Byte length of one Fresnel derivative state.
    pub fn state_size() -> Result<usize> {
        Ok(160)
    }
}

impl SavedState for FresnelResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size()?);
        for &value in self
            .ks
            .iter()
            .flatten()
            .chain(self.kzs.iter().flatten())
            .chain(&self.zs)
        {
            writer.complex(value);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Self::state_size()? {
            return Err(invalid());
        }
        let mut reader = Reader::new(bytes);
        let mut ks = [[Complex::default(); 2]; 2];
        let mut kzs = ks;
        let mut zs = [Complex::default(); 2];
        for value in ks
            .iter_mut()
            .flatten()
            .chain(kzs.iter_mut().flatten())
            .chain(&mut zs)
        {
            *value = read_complex(&mut reader)?;
        }
        reader.finish()?;
        Ok(Self { ks, kzs, zs })
    }
}

impl InterfaceResidual {
    /// Byte length including fixed storage for the optional 4-by-4 inverse.
    pub fn state_size() -> Result<usize> {
        Ok(634)
    }
}

impl SavedState for InterfaceResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size()?);
        for &value in self.ks.iter().flatten().chain(&self.zs) {
            writer.complex(value);
        }
        self.q.iter().for_each(|&v| writer.f64(v));
        writer.usize(self.axis);
        writer.byte(u8::from(self.fixed_q));
        writer.byte(u8::from(self.inverse.is_some()));
        let inverse = self.inverse.unwrap_or_else(InterfaceMatrix::zeros);
        for &value in inverse.iter().chain(self.result.iter()) {
            writer.complex(value);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Self::state_size()? {
            return Err(invalid());
        }
        let mut reader = Reader::new(bytes);
        let mut ks = [[Complex::default(); 2]; 2];
        let mut zs = [Complex::default(); 2];
        for value in ks.iter_mut().flatten().chain(&mut zs) {
            *value = read_complex(&mut reader)?;
        }
        let q = [read_f64(&mut reader)?, read_f64(&mut reader)?];
        let axis = reader.usize()?;
        let fixed_q = reader.byte()?;
        let has_inverse = reader.byte()?;
        if axis > 2 || fixed_q > 1 || has_inverse > 1 {
            return Err(invalid());
        }
        let mut inverse = InterfaceMatrix::zeros();
        let mut result = InterfaceMatrix::zeros();
        for value in inverse.iter_mut().chain(result.iter_mut()) {
            *value = read_complex(&mut reader)?;
        }
        reader.finish()?;
        Ok(Self {
            ks,
            zs,
            q,
            axis,
            inverse: (has_inverse != 0).then_some(inverse),
            result,
            fixed_q: fixed_q != 0,
        })
    }
}

impl PropagationResidual {
    /// Byte length for `n` upgoing wavevectors and their two saved phases.
    pub fn state_size(n: usize) -> Result<usize> {
        if n == 0 {
            return Err(invalid());
        }
        total(&[32, product(&[80, n])?])
    }
}

impl SavedState for PropagationResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(self.vectors.len())?);
        writer.usize(self.vectors.len());
        for &value in self.vectors.iter().flatten() {
            writer.complex(value);
        }
        self.distance.iter().for_each(|&v| writer.f64(v));
        for &value in self.phases.iter().flatten() {
            writer.complex(value);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let n = reader.usize()?;
        if bytes.len() != Self::state_size(n)? {
            return Err(invalid());
        }
        let mut vectors = vec![[Complex::default(); 3]; n];
        for value in vectors.iter_mut().flatten() {
            *value = read_complex(&mut reader)?;
        }
        let distance = [
            read_f64(&mut reader)?,
            read_f64(&mut reader)?,
            read_f64(&mut reader)?,
        ];
        let mut phases = vec![[Complex::default(); 2]; n];
        for value in phases.iter_mut().flatten() {
            *value = read_complex(&mut reader)?;
        }
        reader.finish()?;
        Ok(Self {
            vectors,
            distance,
            phases,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smatrix::{self, Blocks};
    use nalgebra::DMatrix;

    fn cotangent() -> Blocks {
        std::array::from_fn(|_| DMatrix::from_element(2, 2, Complex::new(0.3, -0.1)))
    }

    fn invalid_lengths<T: SavedState>(bytes: &[u8]) {
        assert!(T::from_state(&bytes[..bytes.len() - 1]).is_err());
        let mut extra = bytes.to_vec();
        extra.push(0);
        assert!(T::from_state(&extra).is_err());
    }

    #[test]
    fn fresnel_roundtrip_preserves_both_derivatives() {
        let ks = [[Complex::new(1.2, 0.1); 2], [Complex::new(1.6, 0.2); 2]];
        let kz = ks.map(|row| row.map(|k| (k * k - 0.2).sqrt()));
        let zs = [Complex::from(0.9), Complex::from(1.1)];
        let (_, original) = smatrix::fresnel(ks, kz, zs).unwrap();
        let bytes = original.save_state().unwrap();
        assert_eq!(bytes.len(), FresnelResidual::state_size().unwrap());
        invalid_lengths::<FresnelResidual>(&bytes);
        let restored = FresnelResidual::from_state(&bytes).unwrap();
        assert_eq!(
            original.pushforward(ks, ks, zs).unwrap(),
            restored.pushforward(ks, ks, zs).unwrap()
        );
        let (a, b) = (
            original.pullback(&cotangent()).unwrap(),
            restored.pullback(&cotangent()).unwrap(),
        );
        assert_eq!(a.ks, b.ks);
        assert_eq!(a.kz, b.kz);
        assert_eq!(a.z, b.z);
    }

    #[test]
    fn interface_roundtrip_preserves_inverse_and_grazing_option() {
        let ks = [[Complex::new(1.2, 0.1); 2], [Complex::new(1.6, 0.2); 2]];
        let zs = [Complex::from(0.9), Complex::from(1.1)];
        let (_, original) = smatrix::interface(ks, zs, [0.2, -0.1], 1, false).unwrap();
        let bytes = original.save_state().unwrap();
        assert_eq!(bytes.len(), InterfaceResidual::state_size().unwrap());
        invalid_lengths::<InterfaceResidual>(&bytes);
        let restored = InterfaceResidual::from_state(&bytes).unwrap();
        assert_eq!(
            original.pushforward(ks, zs, [0.1, -0.2]).unwrap(),
            restored.pushforward(ks, zs, [0.1, -0.2]).unwrap()
        );
        let (a, b) = (
            original.pullback(&cotangent()).unwrap(),
            restored.pullback(&cotangent()).unwrap(),
        );
        assert_eq!(a.ks, b.ks);
        assert_eq!(a.z, b.z);
        assert_eq!(a.q.map(f64::to_bits), b.q.map(f64::to_bits));
        let (_, grazing) = smatrix::interface(
            [[Complex::from(1.2); 2]; 2],
            [zs[0]; 2],
            [1.2, 0.0],
            2,
            true,
        )
        .unwrap();
        let restored = InterfaceResidual::from_state(&grazing.save_state().unwrap()).unwrap();
        assert!(restored.inverse.is_none());
        assert!(restored.pullback(&cotangent()).is_err());
    }

    #[test]
    fn propagation_roundtrip_preserves_both_derivatives_and_rejects_bad_shape() {
        let vectors = vec![
            [
                Complex::from(0.2),
                Complex::from(-0.1),
                Complex::new(1.2, 0.1)
            ];
            2
        ];
        let (_, original) = smatrix::propagation(vectors.clone(), [0.1, -0.2, 0.3]).unwrap();
        let mut bytes = original.save_state().unwrap();
        assert_eq!(bytes.len(), PropagationResidual::state_size(2).unwrap());
        invalid_lengths::<PropagationResidual>(&bytes);
        let restored = PropagationResidual::from_state(&bytes).unwrap();
        assert_eq!(
            original.pushforward(&vectors, [0.2, 0.1, -0.1]).unwrap(),
            restored.pushforward(&vectors, [0.2, 0.1, -0.1]).unwrap()
        );
        let (a, b) = (
            original.pullback(&cotangent()).unwrap(),
            restored.pullback(&cotangent()).unwrap(),
        );
        assert_eq!(a.vectors, b.vectors);
        assert_eq!(a.distance.map(f64::to_bits), b.distance.map(f64::to_bits));
        bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(PropagationResidual::from_state(&bytes).is_err());
    }
}

//! Fixed numeric layouts of the per-mode chirality derivative state.
//! treams-rs extension.

use super::super::saved::{invalid, product, read_complex, read_f64, total};
use super::{ChiralityDensityResidual, OrientedChiralityResidual};
use crate::{
    Result,
    saved::{Reader, SavedState, Writer},
};

impl ChiralityDensityResidual {
    /// Byte length for `n` full and normal wavenumbers and an averaging interval.
    pub fn state_size(n: usize) -> Result<usize> {
        if n == 0 {
            return Err(invalid());
        }
        total(&[24, product(&[32, n])?])
    }
}

impl SavedState for ChiralityDensityResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(self.ks.len())?);
        writer.usize(self.ks.len());
        for &value in self.ks.iter().chain(&self.normal) {
            writer.complex(value);
        }
        self.interval.iter().for_each(|&v| writer.f64(v));
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let n = reader.usize()?;
        if bytes.len() != Self::state_size(n)? {
            return Err(invalid());
        }
        let ks = (0..n)
            .map(|_| read_complex(&mut reader))
            .collect::<Result<Vec<_>>>()?;
        let normal = (0..n)
            .map(|_| read_complex(&mut reader))
            .collect::<Result<Vec<_>>>()?;
        let interval = [read_f64(&mut reader)?, read_f64(&mut reader)?];
        reader.finish()?;
        Ok(Self {
            ks,
            normal,
            interval,
        })
    }
}

impl OrientedChiralityResidual {
    /// Byte length for `n` transverse/normal mode vectors and static polarizations.
    pub fn state_size(n: usize) -> Result<usize> {
        if n == 0 {
            return Err(invalid());
        }
        total(&[32, product(&[33, n])?])
    }
}

impl SavedState for OrientedChiralityResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(self.normal.len())?);
        writer.usize(self.normal.len());
        writer.usize(self.axis);
        for &value in self.transverse.iter().flatten() {
            writer.f64(value);
        }
        self.normal.iter().for_each(|&v| writer.complex(v));
        self.polarizations.iter().for_each(|&v| writer.byte(v));
        self.interval.iter().for_each(|&v| writer.f64(v));
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let n = reader.usize()?;
        let axis = reader.usize()?;
        if axis > 2 || bytes.len() != Self::state_size(n)? {
            return Err(invalid());
        }
        let transverse = (0..n)
            .map(|_| Ok([read_f64(&mut reader)?, read_f64(&mut reader)?]))
            .collect::<Result<Vec<_>>>()?;
        let normal = (0..n)
            .map(|_| read_complex(&mut reader))
            .collect::<Result<Vec<_>>>()?;
        let polarizations = (0..n)
            .map(|_| {
                let pol = reader.byte()?;
                if pol > 1 { Err(invalid()) } else { Ok(pol) }
            })
            .collect::<Result<Vec<_>>>()?;
        let interval = [read_f64(&mut reader)?, read_f64(&mut reader)?];
        reader.finish()?;
        Ok(Self {
            transverse,
            normal,
            polarizations,
            axis,
            interval,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Complex, smatrix};
    use nalgebra::DMatrix;

    fn invalid_lengths<T: SavedState>(bytes: &[u8]) {
        assert!(T::from_state(&bytes[..bytes.len() - 1]).is_err());
        let mut extra = bytes.to_vec();
        extra.push(0);
        assert!(T::from_state(&extra).is_err());
    }

    #[test]
    fn chirality_roundtrip_preserves_both_derivatives_and_rejects_bad_shape() {
        let ks = vec![Complex::new(1.2, 0.1), Complex::new(1.4, 0.2)];
        let normal = vec![Complex::new(1.1, 0.12), Complex::new(1.3, 0.23)];
        let (_, original) =
            smatrix::chirality_density(ks.clone(), normal.clone(), [0.1, 0.5]).unwrap();
        let mut bytes = original.save_state().unwrap();
        assert_eq!(
            bytes.len(),
            ChiralityDensityResidual::state_size(2).unwrap()
        );
        invalid_lengths::<ChiralityDensityResidual>(&bytes);
        let restored = ChiralityDensityResidual::from_state(&bytes).unwrap();
        assert_eq!(
            original.pushforward(&ks, &normal, [-0.1, 0.2]).unwrap(),
            restored.pushforward(&ks, &normal, [-0.1, 0.2]).unwrap()
        );
        let cotangent = DMatrix::from_element(3, 2, Complex::new(0.2, -0.3));
        let (a, b) = (
            original.pullback(&cotangent).unwrap(),
            restored.pullback(&cotangent).unwrap(),
        );
        assert_eq!(a.ks, b.ks);
        assert_eq!(a.normal, b.normal);
        assert_eq!(a.interval.map(f64::to_bits), b.interval.map(f64::to_bits));
        bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(ChiralityDensityResidual::from_state(&bytes).is_err());
    }

    #[test]
    fn oriented_roundtrip_preserves_both_derivatives_and_rejects_bad_shape() {
        let transverse = vec![[0.2, -0.1], [-0.3, 0.2]];
        let normal = vec![Complex::new(1.1, 0.12), Complex::new(1.3, 0.23)];
        let (_, original) = smatrix::oriented_chirality(
            transverse.clone(),
            normal.clone(),
            vec![0, 1],
            0,
            [0.1, 0.5],
        )
        .unwrap();
        let mut bytes = original.save_state().unwrap();
        assert_eq!(
            bytes.len(),
            OrientedChiralityResidual::state_size(2).unwrap()
        );
        invalid_lengths::<OrientedChiralityResidual>(&bytes);
        let restored = OrientedChiralityResidual::from_state(&bytes).unwrap();
        assert_eq!(
            original
                .pushforward(&transverse, &normal, [-0.1, 0.2])
                .unwrap(),
            restored
                .pushforward(&transverse, &normal, [-0.1, 0.2])
                .unwrap()
        );
        let cotangent = DMatrix::from_element(3, 2, Complex::new(0.2, -0.3));
        let (a, b) = (
            original.pullback(&cotangent).unwrap(),
            restored.pullback(&cotangent).unwrap(),
        );
        assert_eq!(a.transverse, b.transverse);
        assert_eq!(a.normal, b.normal);
        assert_eq!(a.interval.map(f64::to_bits), b.interval.map(f64::to_bits));
        bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(OrientedChiralityResidual::from_state(&bytes).is_err());
    }
}

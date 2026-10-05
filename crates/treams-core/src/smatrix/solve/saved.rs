//! The original internal solve strategy and its numerical state, with a fixed
//! factor slot that never builds a reflection product solely for serialization.
//!
//! treams-rs extension.

use super::{InternalFactor, InternalSolve};
use crate::{
    Result,
    linalg::Lu,
    saved::{Reader, SavedState, Writer},
    smatrix::saved::{invalid, product, read_matrix, total, write_matrix},
};

impl InternalSolve {
    pub(in crate::smatrix) fn state_size(n: usize, columns: usize) -> Result<usize> {
        if n == 0 || columns == 0 {
            return Err(invalid());
        }
        total(&[17, product(&[16, n, columns])?, Lu::state_size(n)?])
    }
}

fn write_zeros(writer: &mut Writer, count: usize) {
    const ZEROES: [u8; 256] = [0; 256];
    for _ in 0..count / ZEROES.len() {
        writer.raw(&ZEROES);
    }
    writer.raw(&ZEROES[..count % ZEROES.len()]);
}

impl SavedState for InternalSolve {
    fn save_state(&self) -> Result<Vec<u8>> {
        let (n, columns) = self.value.shape();
        let mut writer = Writer::new(Self::state_size(n, columns)?);
        writer.usize(n);
        writer.usize(columns);
        writer.byte(match &self.factor {
            InternalFactor::Reflections => 0,
            InternalFactor::Krylov(_) => 1,
            InternalFactor::Lu(_) => 2,
        });
        write_matrix(&mut writer, &self.value);
        match &self.factor {
            InternalFactor::Reflections => write_zeros(&mut writer, Lu::state_size(n)?),
            InternalFactor::Krylov(operator) => {
                write_matrix(&mut writer, operator);
                write_zeros(&mut writer, Lu::state_size(n)? - product(&[16, n, n])?);
            }
            InternalFactor::Lu(lu) => lu.write_state(&mut writer),
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let (n, columns, tag) = (reader.usize()?, reader.usize()?, reader.byte()?);
        if bytes.len() != Self::state_size(n, columns)? || tag > 2 {
            return Err(invalid());
        }
        let value = read_matrix(&mut reader, n, columns)?;
        let factor = match tag {
            0 => InternalFactor::Reflections,
            1 => InternalFactor::Krylov(read_matrix(&mut reader, n, n)?),
            _ => InternalFactor::Lu(Lu::read_state(&mut reader, n)?),
        };
        // Unused slots are canonical zeros; no matrix is reconstructed for them.
        if reader
            .raw(reader.remaining_len())?
            .iter()
            .any(|&byte| byte != 0)
        {
            return Err(invalid());
        }
        reader.finish()?;
        Ok(Self { value, factor })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Complex, linalg::view, test_support::patterned};
    use nalgebra::DMatrix;

    #[test]
    fn saved_state_preserves_lu_and_both_krylov_strategies() {
        let n = 24;
        let weak = DMatrix::identity(n, n) * Complex::new(0.1, 0.03);
        let cancellation = DMatrix::from_fn(n, n, |_, j| {
            Complex::new(if j % 2 == 0 { 0.1 } else { -0.1 }, 0.0)
        });
        let constant = DMatrix::from_element(n, n, Complex::new(0.1, 0.0));
        let rhs = patterned(n, 2, 0.4);
        let direction = patterned(n, 2, 0.9);
        for (lower, upper, threshold, tag) in [
            (&weak, &weak, n + 1, 2),
            (&weak, &weak, n, 0),
            (&cancellation, &constant, n, 1),
        ] {
            let original =
                InternalSolve::with_threshold(view(lower), view(upper), rhs.clone(), threshold)
                    .unwrap();
            let bytes = original.save_state().unwrap();
            assert_eq!(bytes.len(), InternalSolve::state_size(n, 2).unwrap());
            assert_eq!(bytes[16], tag);
            let loaded = InternalSolve::from_state(&bytes).unwrap();
            assert_eq!(loaded.save_state().unwrap(), bytes);
            assert_eq!(loaded.value, original.value);
            assert_eq!(
                loaded
                    .solve_forward(view(lower), view(upper), direction.clone())
                    .unwrap(),
                original
                    .solve_forward(view(lower), view(upper), direction.clone())
                    .unwrap(),
            );
            assert_eq!(
                loaded
                    .solve_adjoint(view(lower), view(upper), direction.clone())
                    .unwrap()
                    .adjoint,
                original
                    .solve_adjoint(view(lower), view(upper), direction.clone())
                    .unwrap()
                    .adjoint,
            );
            assert!(InternalSolve::from_state(&bytes[..bytes.len() - 1]).is_err());
            let mut extended = bytes.clone();
            extended.push(0);
            assert!(InternalSolve::from_state(&extended).is_err());
            let mut oversized = bytes;
            oversized[..8].fill(u8::MAX);
            assert!(InternalSolve::from_state(&oversized).is_err());
        }
    }
}

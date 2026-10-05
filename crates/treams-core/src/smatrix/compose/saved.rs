//! Fixed saved state for Redheffer composition.
//!
//! treams-rs extension.

use super::AddResidual;
use crate::{
    Result,
    saved::{Reader, SavedState, Writer},
    smatrix::{
        saved::{invalid, product, read_matrix, total, write_matrix},
        solve::InternalSolve,
    },
};

impl AddResidual {
    /// Byte count for saved composition state with `n` modes per direction.
    pub fn state_size(n: usize) -> Result<usize> {
        let columns = product(&[2, n])?;
        total(&[
            8,
            product(&[96, n, n])?,
            InternalSolve::state_size(n, columns)?,
        ])
    }
}

impl SavedState for AddResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let n = self.down.nrows();
        let mut writer = Writer::new(Self::state_size(n)?);
        writer.usize(n);
        for matrix in self.lower.iter().chain(&self.upper).chain([&self.down]) {
            write_matrix(&mut writer, matrix);
        }
        writer.raw(&self.solve.save_state()?);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let n = reader.usize()?;
        if bytes.len() != Self::state_size(n)? {
            return Err(invalid());
        }
        let columns = product(&[2, n])?;
        let lower = [
            read_matrix(&mut reader, n, n)?,
            read_matrix(&mut reader, n, n)?,
        ];
        let upper = [
            read_matrix(&mut reader, n, n)?,
            read_matrix(&mut reader, n, n)?,
        ];
        let down = read_matrix(&mut reader, n, columns)?;
        let solve = InternalSolve::from_state(reader.raw(InternalSolve::state_size(n, columns)?)?)?;
        if solve.value.shape() != (n, columns) {
            return Err(invalid());
        }
        reader.finish()?;
        Ok(Self {
            lower,
            upper,
            solve,
            down,
        })
    }
}

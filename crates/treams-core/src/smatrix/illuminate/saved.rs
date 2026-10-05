//! Fixed saved state for external and internal illumination fields.
//!
//! treams-rs extension.

use super::IlluminateResidual;
use crate::{
    Result,
    saved::{Reader, SavedState, Writer},
    smatrix::{
        saved::{invalid, product, read_matrix, read_stored, total, write_matrix, write_stored},
        solve::InternalSolve,
    },
};

impl IlluminateResidual {
    /// Byte count for saved illumination state with mode and illumination counts.
    pub fn state_size(n: usize, columns: usize) -> Result<usize> {
        total(&[
            24,
            product(&[128, n, n])?,
            product(&[48, n, columns])?,
            InternalSolve::state_size(n, columns)?,
        ])
    }
}

impl SavedState for IlluminateResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let (n, columns) = self.shape();
        let mut writer = Writer::new(Self::state_size(n, columns)?);
        writer.usize(n);
        writer.usize(columns);
        for block in self.lower.iter().chain(&self.upper) {
            write_stored(&mut writer, block);
        }
        for matrix in self.incoming.iter().chain([&self.down]) {
            write_matrix(&mut writer, matrix);
        }
        writer.raw(&self.solve.save_state()?);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let (n, columns) = (reader.usize()?, reader.usize()?);
        if bytes.len() != Self::state_size(n, columns)? {
            return Err(invalid());
        }
        let mut block = || read_stored(&mut reader, n);
        let lower = [block()?, block()?, block()?, block()?];
        let upper = [block()?, block()?, block()?, block()?];
        let mut matrix = || read_matrix(&mut reader, n, columns);
        let incoming = [matrix()?, matrix()?];
        let down = matrix()?;
        let solve = InternalSolve::from_state(reader.raw(InternalSolve::state_size(n, columns)?)?)?;
        if solve.value.shape() != (n, columns) {
            return Err(invalid());
        }
        reader.finish()?;
        Ok(Self {
            lower,
            upper,
            incoming,
            solve,
            down,
        })
    }
}

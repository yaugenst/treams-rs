//! Fixed saved state for periodic-array radiation.
//!
//! treams-rs extension.

use super::FromArrayResidual;
use crate::{
    Result,
    saved::{Reader, SavedState, Writer},
    smatrix::saved::{invalid, product, read_matrix, total, write_matrix},
};

impl FromArrayResidual {
    /// Byte count for array radiation state with multipole and plane-wave counts.
    pub fn state_size(multipoles: usize, modes: usize) -> Result<usize> {
        if multipoles == 0 || modes == 0 {
            return Err(invalid());
        }
        total(&[
            16,
            product(&[16, multipoles, multipoles])?,
            product(&[96, multipoles, modes])?,
        ])
    }
}

impl SavedState for FromArrayResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let (multipoles, modes) = self.input_shape();
        let mut writer = Writer::new(Self::state_size(multipoles, modes)?);
        writer.usize(multipoles);
        writer.usize(modes);
        write_matrix(&mut writer, &self.response);
        for matrix in self.channels.iter().chain(&self.scattered) {
            write_matrix(&mut writer, matrix);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let (multipoles, modes) = (reader.usize()?, reader.usize()?);
        if bytes.len() != Self::state_size(multipoles, modes)? {
            return Err(invalid());
        }
        let response = read_matrix(&mut reader, multipoles, multipoles)?;
        let mut read = || read_matrix(&mut reader, multipoles, modes);
        let channels = [read()?, read()?, read()?, read()?];
        let scattered = [read()?, read()?];
        reader.finish()?;
        Ok(Self {
            response,
            channels,
            scattered,
        })
    }
}

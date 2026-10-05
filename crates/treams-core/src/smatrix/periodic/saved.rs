//! Fixed saved transfer-matrix and Bloch eigensystem state.
//!
//! treams-rs extension.

use super::{BandsResidual, PeriodicResidual};
use crate::{
    Complex, Result,
    linalg::{EigResidual, SolveResidual},
    numerics,
    saved::{Reader, SavedState, Writer},
    smatrix::saved::{invalid, product, read_complex, read_f64, read_matrix, total, write_matrix},
};

impl PeriodicResidual {
    /// Byte count for saved transfer-matrix state with `n` modes per direction.
    pub fn state_size(n: usize) -> Result<usize> {
        let columns = product(&[2, n])?;
        total(&[
            8,
            product(&[48, n, n])?,
            SolveResidual::state_size(n, columns)?,
        ])
    }
}

impl SavedState for PeriodicResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let n = self.top.nrows();
        let mut writer = Writer::new(Self::state_size(n)?);
        writer.usize(n);
        write_matrix(&mut writer, &self.top);
        write_matrix(&mut writer, &self.reflection);
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
        let top = read_matrix(&mut reader, n, columns)?;
        let reflection = read_matrix(&mut reader, n, n)?;
        let solve = SolveResidual::from_state(reader.raw(SolveResidual::state_size(n, columns)?)?)?;
        if solve.shape() != (n, columns) {
            return Err(invalid());
        }
        reader.finish()?;
        Ok(Self {
            top,
            reflection,
            solve,
        })
    }
}

impl BandsResidual {
    /// Byte count for saved Bloch bands with `n` modes per propagation direction.
    pub fn state_size(n: usize) -> Result<usize> {
        let modes = product(&[2, n])?;
        total(&[
            16,
            product(&[16, modes])?,
            PeriodicResidual::state_size(n)?,
            EigResidual::state_size(modes)?,
        ])
    }
}

impl SavedState for BandsResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let n = self.periodic.top.nrows();
        let mut writer = Writer::new(Self::state_size(n)?);
        writer.usize(n);
        writer.f64(self.period);
        for &wavenumber in &self.wavenumbers {
            writer.complex(wavenumber);
        }
        writer.raw(&self.periodic.save_state()?);
        writer.raw(&self.eigen.save_state()?);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let n = reader.usize()?;
        if bytes.len() != Self::state_size(n)? {
            return Err(invalid());
        }
        let modes = product(&[2, n])?;
        let period = read_f64(&mut reader)?;
        if period <= 0.0 {
            return Err(invalid());
        }
        let mut wavenumbers = numerics::filled(modes, Complex::default())?;
        for wavenumber in &mut wavenumbers {
            *wavenumber = read_complex(&mut reader)?;
        }
        let periodic = PeriodicResidual::from_state(reader.raw(PeriodicResidual::state_size(n)?)?)?;
        let eigen = EigResidual::from_state(reader.raw(EigResidual::state_size(modes)?)?)?;
        if periodic.top.nrows() != n || eigen.values().len() != modes {
            return Err(invalid());
        }
        reader.finish()?;
        Ok(Self {
            periodic,
            eigen,
            period,
            wavenumbers,
        })
    }
}

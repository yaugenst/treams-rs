//! Canonical numerical lattice geometry, including optional reduced bases.
//! treams-rs extension.

use super::{BlochLattice, cell::Reduction};
use crate::{
    Error, Result,
    saved::{Reader, Writer, read_f64},
};

fn invalid() -> Error {
    Error::InvalidInput("invalid saved lattice geometry".into())
}

impl BlochLattice {
    /// Fixed geometry size; absent reductions occupy zero-filled slots.
    pub(crate) const STATE_SIZE: usize = 474;

    pub(crate) fn write_state(&self, writer: &mut Writer) {
        writer.usize(self.dim);
        for &value in self
            .direct
            .iter()
            .flatten()
            .chain(self.reciprocal.iter().flatten())
            .chain(&self.kpar)
            .chain(std::iter::once(&self.measure))
        {
            writer.f64(value);
        }
        for reduction in [self.direct_reduction, self.reciprocal_reduction] {
            writer.byte(u8::from(reduction.is_some()));
            let reduction = reduction.unwrap_or(Reduction {
                rows: [[0.0; 3]; 3],
                dual: [[0.0; 3]; 3],
            });
            for &value in reduction
                .rows
                .iter()
                .flatten()
                .chain(reduction.dual.iter().flatten())
            {
                writer.f64(value);
            }
        }
    }

    pub(crate) fn read_state(reader: &mut Reader<'_>) -> Result<Self> {
        let dim = reader.usize()?;
        if !(1..=3).contains(&dim) {
            return Err(invalid());
        }
        let mut direct = [[0.0; 3]; 3];
        let mut reciprocal = [[0.0; 3]; 3];
        let mut kpar = [0.0; 3];
        let mut measure = 0.0;
        for value in direct
            .iter_mut()
            .flatten()
            .chain(reciprocal.iter_mut().flatten())
            .chain(&mut kpar)
            .chain(std::iter::once(&mut measure))
        {
            *value = read_f64(reader)?;
        }
        if measure <= 0.0 {
            return Err(invalid());
        }
        let mut reductions = [None, None];
        for reduction in &mut reductions {
            let present = reader.byte()?;
            if present > 1 {
                return Err(invalid());
            }
            let mut rows = [[0.0; 3]; 3];
            let mut dual = [[0.0; 3]; 3];
            for value in rows.iter_mut().flatten().chain(dual.iter_mut().flatten()) {
                *value = read_f64(reader)?;
            }
            *reduction = (present == 1).then_some(Reduction { rows, dual });
        }
        let [direct_reduction, reciprocal_reduction] = reductions;
        Ok(Self {
            dim,
            direct,
            reciprocal,
            kpar,
            measure,
            direct_reduction,
            reciprocal_reduction,
        })
    }
}

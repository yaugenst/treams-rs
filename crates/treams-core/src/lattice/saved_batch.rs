//! Invocation-owned state for broadcast sums; restoring it performs no Ewald sum.
//! treams-rs extension.

use super::SumResidual;
use crate::{
    Error, Result,
    lattice::{BlochLattice, Family, SumPart},
    numerics::finite,
    saved::{Reader, SavedState, Writer},
};

fn invalid() -> Error {
    Error::InvalidInput("invalid saved lattice-sum state".into())
}

impl SumResidual {
    /// Byte count for `size` broadcast outputs, independent of the numerical inputs.
    pub fn state_size(size: usize) -> Result<usize> {
        size.checked_mul(74 + BlochLattice::STATE_SIZE)
            .and_then(|size| size.checked_add(8))
            .ok_or_else(invalid)
    }
}

impl SavedState for SumResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(self.size)?);
        writer.usize(self.size);
        for i in 0..self.size {
            let (wave, k, lattice, shift, eta, part) = self.element(i);
            let (spherical, l, m) = match wave {
                Family::Spherical { l, m } => (true, l, m),
                Family::Cylindrical { m } => (false, 0, m),
            };
            writer.byte(u8::from(spherical));
            writer.i32(l);
            writer.i32(m);
            writer.complex(k);
            lattice.write_state(&mut writer);
            for value in shift {
                writer.f64(value);
            }
            writer.complex(eta);
            let (kind, shell) = match part {
                SumPart::Full => (0, 0),
                SumPart::Real => (1, 0),
                SumPart::Reciprocal => (2, 0),
                SumPart::Direct(shell) => (3, shell),
            };
            writer.byte(kind);
            writer.usize(usize::try_from(shell).map_err(|_| invalid())?);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let size = reader.usize()?;
        if bytes.len() != Self::state_size(size)? {
            return Err(invalid());
        }
        let mut waves = Vec::with_capacity(size);
        let mut wavenumbers = Vec::with_capacity(size);
        let mut lattices = Vec::with_capacity(size);
        let mut shifts = Vec::with_capacity(size);
        let mut etas = Vec::with_capacity(size);
        let mut parts = Vec::with_capacity(size);
        for _ in 0..size {
            let spherical = reader.byte()?;
            let l = reader.i32()?;
            let m = reader.i32()?;
            let wave = match spherical {
                0 => Family::Cylindrical { m },
                1 => Family::Spherical { l, m },
                _ => return Err(invalid()),
            };
            let k = reader.complex()?;
            let lattice = BlochLattice::read_state(&mut reader)?;
            wave.validate(lattice.dimension())?;
            let shift = [reader.f64()?, reader.f64()?, reader.f64()?];
            let eta = reader.complex()?;
            let kind = reader.byte()?;
            let shell = i64::try_from(reader.usize()?).map_err(|_| invalid())?;
            let part = match kind {
                0 => SumPart::Full,
                1 => SumPart::Real,
                2 => SumPart::Reciprocal,
                3 if shell <= i64::from(i32::MAX) => SumPart::Direct(shell),
                _ => return Err(invalid()),
            };
            if !finite(k) || !finite(eta) || shift.iter().any(|x| !x.is_finite()) {
                return Err(invalid());
            }
            waves.push(wave);
            wavenumbers.push(k);
            lattices.push(lattice);
            shifts.push(shift);
            etas.push(eta);
            parts.push(part);
        }
        reader.finish()?;
        Ok(Self {
            waves,
            wavenumbers,
            lattices,
            shifts,
            etas,
            parts,
            size,
        })
    }
}

//! Saved periodic geometry and coupling labels; restoring performs no Ewald sum.
//! treams-rs extension.

use super::{Couplings, LatticeExpansionResidual};
use crate::{
    Result,
    basis::validate_wavenumbers,
    lattice::BlochLattice,
    numerics::finite,
    saved::{self, Reader, SavedState, Writer},
};

impl LatticeExpansionResidual {
    /// Fixed byte count from the destination and source basis shapes.
    pub fn state_size(
        destination_modes: usize,
        destination_positions: usize,
        source_modes: usize,
        source_positions: usize,
    ) -> Result<usize> {
        saved::cw_basis_size(destination_modes, destination_positions)?
            .checked_add(saved::cw_basis_size(source_modes, source_positions)?)
            .and_then(|size| size.checked_add(BlochLattice::STATE_SIZE + 48))
            .ok_or_else(saved::invalid)
    }
}

impl SavedState for LatticeExpansionResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(
            self.destination.modes.len(),
            self.destination.positions.len(),
            self.source.modes.len(),
            self.source.positions.len(),
        )?);
        saved::write_cw_basis(&mut writer, &self.destination);
        saved::write_cw_basis(&mut writer, &self.source);
        for k in self.ks {
            writer.complex(k);
        }
        writer.complex(self.eta);
        self.lattice.write_state(&mut writer);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut shape = Reader::new(bytes);
        let (dm, dp) = saved::read_cw_basis_dimensions(&mut shape)?;
        let (sm, sp) = saved::read_cw_basis_dimensions(&mut shape)?;
        if bytes.len() != Self::state_size(dm, dp, sm, sp)? {
            return Err(saved::invalid());
        }
        let mut reader = Reader::new(bytes);
        let destination = saved::read_cw_basis(&mut reader)?;
        let source = saved::read_cw_basis(&mut reader)?;
        let ks = [reader.complex()?, reader.complex()?];
        let eta = reader.complex()?;
        let lattice = BlochLattice::read_state(&mut reader)?;
        reader.finish()?;
        validate_wavenumbers(ks, true, false)?;
        if !finite(eta) || lattice.dimension() > 2 {
            return Err(saved::invalid());
        }
        let couplings = Couplings::new(&destination, &source, ks)?;
        Ok(Self {
            destination,
            source,
            ks,
            lattice,
            eta,
            couplings,
        })
    }
}

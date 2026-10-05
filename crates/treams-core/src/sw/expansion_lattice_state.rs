//! Saved periodic geometry and static angular-plan inputs, without Ewald reevaluation.
//! treams-rs extension.

use super::{
    LatticeExpansionFromTableResidual, LatticeExpansionResidual, blocks, validate_expansion,
};
use crate::{
    Result,
    lattice::BlochLattice,
    numerics::finite,
    saved::{self, Reader, SavedState, Writer},
};

fn basis_pair_size(dm: usize, dp: usize, sm: usize, sp: usize, extra: usize) -> Result<usize> {
    saved::sw_basis_size(dm, dp)?
        .checked_add(saved::sw_basis_size(sm, sp)?)
        .and_then(|size| size.checked_add(extra))
        .ok_or_else(saved::invalid)
}

impl LatticeExpansionResidual {
    /// Fixed byte count from the destination and source basis shapes.
    pub fn state_size(
        destination_modes: usize,
        destination_positions: usize,
        source_modes: usize,
        source_positions: usize,
    ) -> Result<usize> {
        basis_pair_size(
            destination_modes,
            destination_positions,
            source_modes,
            source_positions,
            BlochLattice::STATE_SIZE + 49,
        )
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
        saved::write_sw_basis(&mut writer, &self.destination);
        saved::write_sw_basis(&mut writer, &self.source);
        for k in self.ks {
            writer.complex(k);
        }
        writer.complex(self.eta);
        writer.byte(u8::from(self.helicity));
        self.lattice.write_state(&mut writer);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut shape = Reader::new(bytes);
        let (dm, dp) = saved::read_sw_basis_dimensions(&mut shape)?;
        let (sm, sp) = saved::read_sw_basis_dimensions(&mut shape)?;
        if bytes.len() != Self::state_size(dm, dp, sm, sp)? {
            return Err(saved::invalid());
        }
        let mut reader = Reader::new(bytes);
        let destination = saved::read_sw_basis(&mut reader)?;
        let source = saved::read_sw_basis(&mut reader)?;
        let ks = [reader.complex()?, reader.complex()?];
        let eta = reader.complex()?;
        let helicity = match reader.byte()? {
            0 => false,
            1 => true,
            _ => return Err(saved::invalid()),
        };
        let lattice = BlochLattice::read_state(&mut reader)?;
        reader.finish()?;
        validate_expansion(&destination, &source, ks, helicity)?;
        if !finite(eta) {
            return Err(saved::invalid());
        }
        let blocks = blocks(&destination, &source, helicity, false)?;
        Ok(Self {
            destination,
            source,
            ks,
            lattice,
            eta,
            helicity,
            blocks,
        })
    }
}

impl LatticeExpansionFromTableResidual {
    /// Fixed byte count from the destination and source basis shapes.
    pub fn state_size(
        destination_modes: usize,
        destination_positions: usize,
        source_modes: usize,
        source_positions: usize,
    ) -> Result<usize> {
        basis_pair_size(
            destination_modes,
            destination_positions,
            source_modes,
            source_positions,
            33,
        )
    }
}

impl SavedState for LatticeExpansionFromTableResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(
            self.destination.modes.len(),
            self.destination.positions.len(),
            self.source.modes.len(),
            self.source.positions.len(),
        )?);
        saved::write_sw_basis(&mut writer, &self.destination);
        saved::write_sw_basis(&mut writer, &self.source);
        writer.byte(u8::from(self.helicity));
        for size in self.table_shape {
            writer.usize(size);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut shape = Reader::new(bytes);
        let (dm, dp) = saved::read_sw_basis_dimensions(&mut shape)?;
        let (sm, sp) = saved::read_sw_basis_dimensions(&mut shape)?;
        if bytes.len() != Self::state_size(dm, dp, sm, sp)? {
            return Err(saved::invalid());
        }
        let mut reader = Reader::new(bytes);
        let destination = saved::read_sw_basis(&mut reader)?;
        let source = saved::read_sw_basis(&mut reader)?;
        let helicity = match reader.byte()? {
            0 => false,
            1 => true,
            _ => return Err(saved::invalid()),
        };
        let table_shape = [
            reader.usize()?,
            reader.usize()?,
            reader.usize()?,
            reader.usize()?,
        ];
        reader.finish()?;
        let [destinations, sources, channels, harmonics] = table_shape;
        let order = destination
            .modes
            .iter()
            .map(|(_, mode)| mode.l)
            .max()
            .unwrap_or(0)
            + source
                .modes
                .iter()
                .map(|(_, mode)| mode.l)
                .max()
                .unwrap_or(0);
        if destinations != dp
            || sources != sp
            || !(1..=2).contains(&channels)
            || (!helicity && channels != 1)
            || usize::try_from((order + 1).pow(2)) != Ok(harmonics)
        {
            return Err(saved::invalid());
        }
        let blocks = blocks(&destination, &source, helicity, true)?;
        Ok(Self {
            destination,
            source,
            helicity,
            blocks,
            table_shape,
            shape: (dm, sm),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Complex,
        sw::{Basis, Mode, lattice_expansion},
    };
    use nalgebra::DMatrix;

    #[test]
    fn extreme_saved_split_returns_an_error_without_panicking() {
        let destination = Basis {
            modes: vec![(0, Mode { l: 1, m: 0, pol: 0 })],
            positions: vec![[0.0; 3]],
        };
        let source = Basis {
            positions: vec![[0.1, 0.0, 0.0]],
            ..destination.clone()
        };
        let lattice = BlochLattice::new(&[vec![1.5, 0.0], vec![0.3, 1.4]], &[0.1, 0.2]).unwrap();
        let (_, mut residual) = lattice_expansion(
            destination,
            source,
            [Complex::new(1.2, 0.1); 2],
            true,
            lattice,
            Complex::new(0.9, 0.1),
        )
        .unwrap();
        for eta in [
            Complex::new(1e200, 0.0),
            Complex::new(1e-200, 0.0),
            Complex::new(0.0, 1e200),
            Complex::new(1e200, 1e200),
            Complex::new(f64::MAX, 0.0),
            Complex::new(0.0, f64::MAX),
        ] {
            residual.eta = eta;
            let restored =
                LatticeExpansionResidual::from_state(&residual.save_state().unwrap()).unwrap();
            assert!(
                restored
                    .pullback(&DMatrix::from_element(1, 1, Complex::new(1.0, 0.0)))
                    .is_err(),
                "eta={eta}"
            );
        }
    }
}
